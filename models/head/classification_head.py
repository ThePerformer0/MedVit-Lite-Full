"""
MedViT-Lite — Classification Head avec Estimation d'Incertitude
================================================================
Dernier module du modèle. Prend le token CLS produit par l'HTA
et produit :
  1. Les logits de classification (une valeur par pathologie)
  2. Une estimation d'incertitude via Monte Carlo Dropout

Rappel du scope :
  Entrée  : vecteur CLS [B, embed_dim]  (résumé de l'image)
  Sortie  : (logits [B, num_classes], uncertainty [B, num_classes])

Classification multi-label :
  Chaque pathologie est traitée indépendamment.
  Une même image peut avoir : Pneumonie ET Épanchement simultanément.
  → On utilise BCEWithLogitsLoss (pas CrossEntropy)
  → À l'inférence : sigmoid(logits) > 0.5 = pathologie présente

Estimation d'incertitude (Monte Carlo Dropout) :
  Idée : garder le Dropout ACTIF pendant l'inférence.
  → Faire N passes forward avec le même input
  → La variance des prédictions = mesure d'incertitude

  Interprétation :
    Variance faible  → le modèle est sûr de sa prédiction
    Variance élevée  → le modèle est incertain → recommander avis expert

  Référence : Gal & Ghahramani, "Dropout as a Bayesian Approximation"
              (ICML 2016) — l'une des méthodes les plus simples et
              les plus utilisées pour quantifier l'incertitude.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """
    Tête de classification multi-label avec incertitude Monte Carlo.

    Architecture :
      CLS token [B, D]
        → LayerNorm
        → Linear(D → hidden_dim)  + GELU + Dropout
        → Linear(hidden_dim → num_classes)
        → logits [B, num_classes]

    Pourquoi deux couches linéaires et pas une seule ?
    → Une couche cachée permet d'apprendre des combinaisons de features
      avant la prédiction finale. Cela améliore généralement les performances
      sur des tâches multi-label où les classes sont corrélées
      (ex: Œdème et Épanchement co-occurrent souvent).

    Args:
        embed_dim   : dimension du token CLS en entrée
        hidden_dim  : dimension de la couche cachée
        num_classes : nombre de pathologies à prédire (14 pour NIH)
        dropout     : dropout (aussi utilisé pour MC uncertainty)
        mc_samples  : nombre de passes MC à l'inférence pour l'incertitude
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 256,
        num_classes: int = 14,
        dropout: float = 0.3,
        mc_samples: int = 10,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.mc_samples = mc_samples
        self.dropout_rate = dropout

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),          # ← Ce Dropout reste actif en MC mode
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()
        logger.info(
            f"ClassificationHead : {embed_dim}→{hidden_dim}→{num_classes} classes, "
            f"dropout={dropout}, mc_samples={mc_samples}"
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Passe forward standard (entraînement).

        Args:
            cls_token : [B, embed_dim]
        Returns:
            logits    : [B, num_classes]
                        → appliquer sigmoid pour avoir des probabilités
        """
        return self.head(cls_token)

    def predict_with_uncertainty(
        self,
        cls_token: torch.Tensor,
        threshold: float = 0.5,
        uncertainty_threshold: float = 0.15,
    ) -> dict:
        """
        Prédiction avec estimation d'incertitude via Monte Carlo Dropout.

        Active le Dropout pendant l'inférence et fait N passes forward.
        La variance entre les passes = mesure de l'incertitude du modèle.

        Args:
            cls_token             : [B, embed_dim]
            threshold             : seuil de classification (sigmoid > threshold)
            uncertainty_threshold : au-delà, marquer comme "cas ambigu"

        Returns:
            dict avec :
              'probabilities'  : [B, num_classes]  — probabilités moyennes
              'uncertainty'    : [B, num_classes]  — variance entre les passes MC
              'predictions'    : [B, num_classes]  — prédictions binaires (0/1)
              'is_uncertain'   : [B]               — True si cas ambigu global
              'confidence'     : [B, num_classes]  — 1 - uncertainty (lisible)
        """
        # ── Activer le Dropout même en eval ──────────────────────────────────
        # On met tous les sous-modules Dropout en mode "train" pour qu'ils
        # restent actifs, même si le modèle global est en mode eval.
        self._enable_mc_dropout()

        all_probs = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits = self.head(cls_token)          # [B, num_classes]
                probs  = torch.sigmoid(logits)          # [B, num_classes]
                all_probs.append(probs)

        # Restaurer le mode eval normal
        self._disable_mc_dropout()

        # ── Statistiques sur les N passes ─────────────────────────────────────
        # all_probs : liste de N tenseurs [B, num_classes]
        probs_stack = torch.stack(all_probs, dim=0)  # [N, B, num_classes]

        mean_probs   = probs_stack.mean(dim=0)   # [B, num_classes]
        uncertainty  = probs_stack.var(dim=0)    # [B, num_classes]

        # ── Décisions finales ─────────────────────────────────────────────────
        predictions = (mean_probs > threshold).float()  # [B, num_classes]

        # Un cas est globalement ambigu si l'incertitude max sur toutes les
        # classes dépasse le seuil configuré
        is_uncertain = (uncertainty.max(dim=1).values > uncertainty_threshold)  # [B]

        return {
            'probabilities': mean_probs,          # [B, 14]
            'uncertainty':   uncertainty,          # [B, 14]
            'predictions':   predictions,          # [B, 14] binaire
            'is_uncertain':  is_uncertain,         # [B] booléen
            'confidence':    1.0 - uncertainty,    # [B, 14] lisible
        }

    def _enable_mc_dropout(self):
        """Active tous les modules Dropout (pour inférence MC)."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _disable_mc_dropout(self):
        """Remet tous les modules Dropout en mode eval."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    def format_prediction(
        self,
        result: dict,
        pathology_names: list,
        sample_idx: int = 0,
    ) -> str:
        """
        Formate la prédiction de manière lisible pour un clinicien.

        Args:
            result         : sortie de predict_with_uncertainty()
            pathology_names: liste des noms de pathologies
            sample_idx     : index dans le batch à afficher

        Returns:
            Chaîne formatée pour affichage
        """
        probs    = result['probabilities'][sample_idx]
        uncert   = result['uncertainty'][sample_idx]
        preds    = result['predictions'][sample_idx]
        ambig    = result['is_uncertain'][sample_idx].item()

        lines = ["─" * 50]
        lines.append("RÉSULTAT DIAGNOSTIC MedViT-Lite")
        lines.append("─" * 50)

        detected = []
        for i, name in enumerate(pathology_names):
            if preds[i].item() == 1:
                detected.append((name, probs[i].item(), uncert[i].item()))

        if detected:
            lines.append(f"\nPathologies détectées ({len(detected)}) :")
            for name, prob, unc in sorted(detected, key=lambda x: -x[1]):
                confidence_bar = "█" * int(prob * 20)
                lines.append(
                    f"  ✓ {name:<25} "
                    f"prob={prob:.1%}  "
                    f"incert={unc:.3f}  "
                    f"|{confidence_bar:<20}|"
                )
        else:
            lines.append("\n  ✓ Aucune pathologie détectée (Normal)")

        if ambig:
            lines.append("\n  ⚠️  CAS AMBIGU — Incertitude élevée")
            lines.append("      Recommandation : avis d'expert requis")

        lines.append("─" * 50)
        return "\n".join(lines)


if __name__ == "__main__":
    from data.datasets.chest_mnist import PATHOLOGY_NAMES

    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)

    head = ClassificationHead(
        embed_dim=384,
        hidden_dim=256,
        num_classes=14,
        dropout=0.3,
        mc_samples=10,
    )
    head.eval()

    # Simuler des tokens CLS (sortie de l'HTA)
    cls_tokens = torch.randn(4, 384)

    # Test inférence standard
    logits = head(cls_tokens)
    print(f"Test forward standard :")
    print(f"  Entrée : {cls_tokens.shape}")
    print(f"  Sortie : {logits.shape}  (logits, appliquer sigmoid pour proba)")

    # Test avec incertitude MC
    print(f"\nTest Monte Carlo ({head.mc_samples} passes) :")
    result = head.predict_with_uncertainty(cls_tokens, threshold=0.5)

    print(f"  probabilities : {result['probabilities'].shape}")
    print(f"  uncertainty   : {result['uncertainty'].shape}")
    print(f"  is_uncertain  : {result['is_uncertain']}")

    # Affichage clinique pour le premier sample
    print(f"\n{head.format_prediction(result, PATHOLOGY_NAMES, sample_idx=0)}")
