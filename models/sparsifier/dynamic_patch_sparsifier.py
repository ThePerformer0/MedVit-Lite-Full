"""
MedViT-Lite — Dynamic Patch Sparsifier (DPS)
==============================================
INNOVATION 1 : Réduction du coût computationnel par sélection
               intelligente des patches informatifs.

Principe :
----------
Un Transformer standard traite TOUS les patches avec la même attention.
Pour une image médicale 224×224 avec patches 16×16 → 196 patches.
Beaucoup de ces patches sont du fond non-informatif (zones noires, 
contours d'image).

Le DPS attribue un score d'importance à chaque patch et ne transmet
au Transformer que les top-K patches (défaut : K = 50% = 98 patches).

Avantage computationnel :
  Attention complexity = O(N²)
  Avec DPS à 50% : O((N/2)²) = O(N²/4) → 4× moins de calcul attention

Deux modes :
  - Hard (inférence) : sélection binaire des top-K patches
  - Soft (entraînement) : pondération différentiable via Gumbel-Softmax

Référence théorique :
  DynamicViT (Rao et al., NeurIPS 2021) - inspiré de ce travail
  mais adapté au contexte médical avec un scorer plus léger.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImportanceScorer(nn.Module):
    """
    Réseau léger qui prédit un score d'importance pour chaque patch.

    Architecture : MLP 2 couches avec bottleneck
    Entrée  : tokens de patches [B, N, D]
    Sortie  : scores [B, N, 1]  (entre 0 et 1 après sigmoid)

    Pourquoi un MLP léger et pas une couche d'attention ?
    → On veut que le scorer lui-même soit rapide. Si le scorer est
      aussi coûteux que le Transformer, on perd tout le bénéfice.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            # Pas de sigmoid ici — on applique après selon le mode
        )
        self._init_weights()

    def _init_weights(self):
        """Initialisation proche de zéro pour commencer avec peu de biais."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens : [B, N, D] — tokens de patches (embeddings)
        Returns:
            scores : [B, N, 1] — score d'importance brut (logit)
        """
        return self.scorer(tokens)  # [B, N, 1]


class DynamicPatchSparsifier(nn.Module):
    """
    Module principal de sparsification dynamique des patches.

    Workflow :
        1. Le scorer évalue chaque patch → score ∈ [0,1]
        2. On sélectionne les top-K patches (mode hard) ou on pondère
           tous les patches par leur score (mode soft pour l'entraînement)
        3. Seuls les patches sélectionnés passent au Transformer

    Args:
        embed_dim    : dimension des embeddings de patch
        keep_ratio   : fraction de patches à garder (0.5 = 50%)
        hidden_dim   : dimension cachée du scorer MLP
        temperature  : température pour Gumbel-Softmax (entraînement)
        hard         : si True, sélection binaire (inférence)
    """

    def __init__(
        self,
        embed_dim: int,
        keep_ratio: float = 0.5,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        hard: bool = False,
    ):
        super().__init__()

        assert 0.0 < keep_ratio <= 1.0, \
            f"keep_ratio doit être dans (0, 1], reçu: {keep_ratio}"

        self.keep_ratio = keep_ratio
        self.temperature = temperature
        self.hard = hard

        self.scorer = ImportanceScorer(embed_dim, hidden_dim)

        logger.info(
            f"DPS initialisé : keep_ratio={keep_ratio:.0%}, "
            f"mode={'hard' if hard else 'soft'}"
        )

    def forward(
        self,
        tokens: torch.Tensor,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tokens       : [B, N, D] — tous les patches en entrée
            return_scores: si True, retourne aussi les scores bruts

        Returns:
            sparse_tokens : [B, K, D] — patches sélectionnés (K = N × keep_ratio)
            topk_indices  : [B, K]    — indices des patches gardés
            scores        : [B, N, 1] — scores d'importance (si return_scores)

        Note sur les indices :
            On retourne les indices pour pouvoir reconstruire la position
            spatiale des patches sélectionnés dans l'image originale.
            C'est crucial pour GradCAM++ (on doit savoir quel patch
            correspond à quelle zone de l'image).
        """
        B, N, D = tokens.shape
        K = max(1, int(N * self.keep_ratio))  # nombre de patches à garder

        # ── 1. Calcul des scores d'importance ───────────────────────────────
        # scores shape : [B, N, 1]
        logits = self.scorer(tokens)
        scores = torch.sigmoid(logits)  # [B, N, 1]

        if self.training and not self.hard:
            # ── Mode soft (entraînement différentiable) ──────────────────────
            # Gumbel-Softmax permet de "simuler" une sélection binaire
            # tout en restant différentiable (gradients peuvent circuler)
            gumbel_scores = self._gumbel_softmax_topk(logits.squeeze(-1), K)
            # gumbel_scores : [B, N] — valeurs proches de 0 ou 1
            
            # Pondérer les tokens par leur score gumbel
            sparse_tokens = tokens * gumbel_scores.unsqueeze(-1)  # [B, N, D]
            
            # Pour les indices, on prend quand même les top-K (pour la visualisation)
            topk_indices = logits.squeeze(-1).topk(K, dim=1).indices  # [B, K]

        else:
            # ── Mode hard (inférence ou debug) ───────────────────────────────
            # Sélection stricte des K patches avec les scores les plus élevés
            topk_values, topk_indices = logits.squeeze(-1).topk(K, dim=1)
            # topk_indices : [B, K]

            # Extraire seulement les patches sélectionnés
            # expand_indices : [B, K, D]
            expand_indices = topk_indices.unsqueeze(-1).expand(-1, -1, D)
            sparse_tokens = tokens.gather(1, expand_indices)  # [B, K, D]

        if return_scores:
            return sparse_tokens, topk_indices, scores
        return sparse_tokens, topk_indices, None

    def _gumbel_softmax_topk(
        self, logits: torch.Tensor, K: int
    ) -> torch.Tensor:
        """
        Approximation différentiable de la sélection top-K via Gumbel noise.

        Intuition : on ajoute du bruit Gumbel pour "relaxer" la sélection
        discrète en une opération continue et différentiable.

        Args:
            logits : [B, N] — scores bruts (non normalisés)
            K      : nombre de patches à sélectionner

        Returns:
            mask : [B, N] — masque doux (valeurs proches de 0 ou 1)
        """
        # Ajouter du bruit Gumbel
        gumbel_noise = -torch.log(
            -torch.log(torch.rand_like(logits) + 1e-10) + 1e-10
        )
        perturbed = (logits + gumbel_noise) / self.temperature

        # Top-K softmax
        # On veut que les K meilleurs aient des valeurs proches de 1
        # et les autres proches de 0
        topk_vals, _ = perturbed.topk(K, dim=1)
        threshold = topk_vals[:, -1:].expand_as(perturbed)

        # Masque doux : 1 si score >= threshold, 0 sinon (avec relaxation)
        mask = torch.sigmoid((perturbed - threshold) / self.temperature)
        return mask

    def set_hard_mode(self, hard: bool):
        """Basculer entre mode soft (entraînement) et hard (inférence)."""
        self.hard = hard
        logger.debug(f"DPS mode → {'hard' if hard else 'soft'}")

    def get_sparsity_ratio(self) -> float:
        """Retourne le ratio de sparsité (fraction de patches ignorés)."""
        return 1.0 - self.keep_ratio

    def extra_repr(self) -> str:
        return (
            f"keep_ratio={self.keep_ratio:.0%}, "
            f"temperature={self.temperature}, "
            f"hard={self.hard}"
        )


if __name__ == "__main__":
    # ── Test unitaire du DPS ─────────────────────────────────────────────────
    logging.basicConfig(level=logging.INFO)

    torch.manual_seed(42)

    # Simuler des tokens de patches (batch=4, 196 patches, dim=384)
    B, N, D = 4, 196, 384
    fake_tokens = torch.randn(B, N, D)

    # Test mode soft (entraînement)
    dps = DynamicPatchSparsifier(embed_dim=D, keep_ratio=0.5)
    dps.train()
    sparse, indices, scores = dps(fake_tokens, return_scores=True)

    print(f"[Mode SOFT - entraînement]")
    print(f"  Entrée  : {fake_tokens.shape}   (B={B}, N={N}, D={D})")
    print(f"  Sortie  : {sparse.shape}   (B={B}, K={N//2}={N}×50%)")
    print(f"  Indices : {indices.shape}")
    print(f"  Scores  : {scores.shape} — mean={scores.mean():.3f}")
    print(f"  Réduction calcul attention : {(N//2)**2 / N**2 * 100:.0f}% du coût original\n")

    # Test mode hard (inférence)
    dps.set_hard_mode(True)
    dps.eval()
    with torch.no_grad():
        sparse_hard, indices_hard, _ = dps(fake_tokens)

    print(f"[Mode HARD - inférence]")
    print(f"  Patches gardés : {sparse_hard.shape[1]}/{N} ({sparse_hard.shape[1]/N:.0%})")
    print(f"  Indices uniques ? {indices_hard[0].unique().shape[0] == indices_hard.shape[1]}")
