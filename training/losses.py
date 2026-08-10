"""
MedViT-Lite — Fonctions de perte (Loss Functions)
==================================================
Pour un problème de classification multi-label médicale, la perte
standard est Binary Cross-Entropy (BCE) appliquée indépendamment
sur chaque classe.

Problème du déséquilibre de classes :
--------------------------------------
Dans NIH Chest X-Ray / ChestMNIST, les classes sont très déséquilibrées :
  - Infiltration : ~18% des images
  - Hernia       :  ~0.2% des images

Sans correction, le modèle apprendrait à toujours prédire "pas de Hernia"
car c'est presque toujours vrai. Accuracy élevée, mais sensibilité nulle.

Solutions implémentées :
  1. Weighted BCE     : pénalise plus les erreurs sur les classes rares
  2. Focal Loss       : réduit la contribution des exemples "faciles"
                        et met l'accent sur les cas difficiles (mal classés)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    BCE avec pondération par classe pour gérer le déséquilibre.

    BCEWithLogitsLoss = Sigmoid + BCE en une seule opération
    (numériquement plus stable que d'appliquer sigmoid séparément)

    Formule :
      L = -w_pos * y * log(σ(x)) - (1 - y) * log(1 - σ(x))

    Args:
        pos_weight : [num_classes] tensor de poids pour les positifs
                     Si None, calculé à partir du dataset (recommandé)
        reduction  : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

        if pos_weight is not None:
            logger.info(
                f"WeightedBCE : pos_weight min={pos_weight.min():.2f}, "
                f"max={pos_weight.max():.2f}, mean={pos_weight.mean():.2f}"
            )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits  : [B, num_classes]  — sorties brutes du modèle (avant sigmoid)
            targets : [B, num_classes]  — labels binaires (0 ou 1)

        Returns:
            loss : scalaire
        """
        # Déplacer pos_weight sur le bon device si nécessaire
        if self.pos_weight is not None and self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)

        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )


class FocalLoss(nn.Module):
    """
    Focal Loss pour la classification multi-label médicale.

    Motivation :
      Dans un dataset médical déséquilibré, la plupart des exemples sont
      "faciles" (négatifs évidents). La BCE leur accorde autant d'importance
      qu'aux exemples difficiles (cas limites, pathologies rares).

      La Focal Loss réduit la contribution des exemples faciles via un
      facteur (1 - p)^gamma, ce qui force le modèle à se concentrer sur
      les cas difficiles.

    Formule :
      FL = -α * (1 - p)^γ * log(p)

      Avec γ=2 (standard) :
        - Exemple facile (p=0.95) : (1-0.95)² = 0.0025  → contribution ×400 moins
        - Exemple difficile (p=0.5) : (1-0.5)² = 0.25   → contribution normale

    Référence : Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)

    Args:
        alpha : facteur de pondération des positifs (0.25 dans le papier original)
        gamma : facteur de focalisation (2.0 recommandé)
        reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        logger.info(f"FocalLoss : alpha={alpha}, gamma={gamma}")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits  : [B, num_classes]
            targets : [B, num_classes]  — labels binaires

        Returns:
            loss : scalaire
        """
        # Probabilités après sigmoid
        probs = torch.sigmoid(logits)

        # BCE de base (sans réduction)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Facteur focal : (1 - p_t)^gamma
        # p_t = p si target=1, (1-p) si target=0
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Facteur alpha : alpha si target=1, (1-alpha) si target=0
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    Combinaison de BCE pondéré + Focal Loss.

    L_total = λ * L_weighted_bce + (1-λ) * L_focal

    Permet de bénéficier des deux : pondération de classe (BCE)
    et focus sur les cas difficiles (Focal).

    Args:
        pos_weight    : poids des positifs pour BCE [num_classes]
        focal_alpha   : alpha de la Focal Loss
        focal_gamma   : gamma de la Focal Loss
        bce_weight    : λ (contribution de la BCE, entre 0 et 1)
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        bce_weight: float = 0.5,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.focal_weight = 1.0 - bce_weight

        self.bce   = WeightedBCEWithLogitsLoss(pos_weight)
        self.focal = FocalLoss(focal_alpha, focal_gamma)

        logger.info(
            f"CombinedLoss : BCE×{bce_weight:.1f} + Focal×{1-bce_weight:.1f}"
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        l_bce   = self.bce(logits, targets)
        l_focal = self.focal(logits, targets)
        return self.bce_weight * l_bce + self.focal_weight * l_focal


def build_loss(config: dict, pos_weight: Optional[torch.Tensor] = None):
    """
    Factory : construit la fonction de perte à partir de la config YAML.

    Args:
        config     : section config["training"]["loss"]
        pos_weight : poids calculés depuis le dataset

    Returns:
        loss_fn : module de perte
    """
    loss_name = config.get("name", "bce_with_logits")

    if loss_name == "bce_with_logits":
        return WeightedBCEWithLogitsLoss(pos_weight)
    elif loss_name == "focal":
        return FocalLoss(
            alpha=config.get("focal_alpha", 0.25),
            gamma=config.get("focal_gamma", 2.0),
        )
    elif loss_name == "combined":
        return CombinedLoss(
            pos_weight=pos_weight,
            focal_alpha=config.get("focal_alpha", 0.25),
            focal_gamma=config.get("focal_gamma", 2.0),
            bce_weight=config.get("bce_weight", 0.5),
        )
    else:
        raise ValueError(f"Loss inconnue : '{loss_name}'")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)

    B, C = 8, 14  # batch=8, 14 classes

    logits  = torch.randn(B, C)
    targets = torch.randint(0, 2, (B, C)).float()

    # Simuler des poids de classe (classes rares ont poids élevé)
    pos_weight = torch.tensor([1.0, 5.0, 2.0, 3.0, 8.0, 4.0, 10.0,
                                6.0, 2.5, 3.5, 7.0, 5.5, 4.5, 15.0])

    print("Test des fonctions de perte :")
    for LossCls, kwargs in [
        (WeightedBCEWithLogitsLoss, {"pos_weight": pos_weight}),
        (FocalLoss, {"alpha": 0.25, "gamma": 2.0}),
        (CombinedLoss, {"pos_weight": pos_weight}),
    ]:
        loss_fn = LossCls(**kwargs)
        loss    = loss_fn(logits, targets)
        print(f"  {LossCls.__name__:<30} : {loss.item():.4f}")
