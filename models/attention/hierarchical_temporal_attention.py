"""
MedViT-Lite — Hierarchical Temporal Attention (HTA)
====================================================
Cœur du Transformer : mécanisme d'attention à deux niveaux.

Niveau 1 — Attention Locale (spatiale, intra-frame) :
  "Quelles zones de CETTE image sont liées entre elles ?"
  → Standard Multi-Head Self-Attention sur les patches d'une frame

Niveau 2 — Attention Globale (temporelle, inter-frames) :
  "Comment les frames ÉVOLUENT-ELLES dans le temps ?"
  → Attention entre le token CLS de chaque frame de la séquence

Pourquoi deux niveaux séparés et pas un seul ?
-----------------------------------------------
Un seul bloc d'attention sur toutes les frames simultanément
aurait une complexité O((T×N)²) avec T=frames, N=patches.
Pour T=16 frames et N=196 patches → 9.8M paires à calculer.

En séparant :
  - Attention locale  : O(N²) × T = gérable
  - Attention globale : O(T²) × 1 = très petit (T << N)

Inspiré de : TimeSformer (Bertasius et al., NeurIPS 2021)
             "Is Space-Time Attention All You Need for Video Understanding?"
Mais adapté : travaille sur patches sparsifiés (sortie du DPS).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional
import logging
import math

logger = logging.getLogger(__name__)


class MultiHeadSelfAttention(nn.Module):
    """
    Attention multi-têtes standard.

    Rappel du mécanisme :
      Q, K, V = projections linéaires des tokens d'entrée
      Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V

      Intuition :
        Q = "ce que je cherche"
        K = "ce que je propose"
        V = "ce que je donne si on me sélectionne"

    Args:
        embed_dim  : dimension des tokens
        num_heads  : nombre de têtes d'attention (embed_dim doit être divisible)
        dropout    : dropout sur les poids d'attention
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 6,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) doit être divisible par num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/√d_k (facteur de normalisation)

        # Projection Q, K, V en une seule opération (efficace)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x    : [B, N, D]  — séquence de tokens
            mask : [B, N, N]  — masque d'attention optionnel (1=ignorer)

        Returns:
            out  : [B, N, D]
        """
        B, N, D = x.shape

        # ── Calcul Q, K, V ──────────────────────────────────────────────────
        qkv = self.qkv(x)  # [B, N, 3*D]
        # Séparation et reshape pour le multi-head
        qkv = rearrange(qkv, 'b n (three h d) -> three b h n d',
                        three=3, h=self.num_heads)
        q, k, v = qkv.unbind(0)  # chacun : [B, num_heads, N, head_dim]

        # ── Scores d'attention ───────────────────────────────────────────────
        # QKᵀ / √d_k → [B, num_heads, N, N]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Appliquer le masque (remplir par -inf pour ignorer)
        if mask is not None:
            attn = attn.masked_fill(mask == 1, float('-inf'))

        # Softmax sur la dernière dimension (softmax par ligne)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # ── Agrégation des valeurs V ─────────────────────────────────────────
        out = torch.matmul(attn, v)  # [B, num_heads, N, head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, N, D]
        out = self.proj(out)
        out = self.proj_dropout(out)

        return out


class TransformerBlock(nn.Module):
    """
    Bloc Transformer standard (Pre-LN variant).

    Structure :
      x → LayerNorm → Attention → + x   (connexion résiduelle)
        → LayerNorm → FFN       → + x   (connexion résiduelle)

    Pourquoi Pre-LN (LayerNorm avant, pas après) ?
    → Plus stable à l'entraînement, moins sensible au learning rate.
    → Utilisé dans GPT-2, ViT moderne.

    FFN (Feed-Forward Network) :
      Linear(D → 4D) → GELU → Dropout → Linear(4D → D)
      Le facteur ×4 est empirique mais très répandu.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.1,
    ):
        super().__init__()
        mlp_dim = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads,
                                            attn_dropout, mlp_dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(mlp_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, N, D]
        Returns:
            x : [B, N, D]
        """
        # Attention avec connexion résiduelle
        x = x + self.attn(self.norm1(x))
        # FFN avec connexion résiduelle
        x = x + self.ffn(self.norm2(x))
        return x


class LocalSpatialAttention(nn.Module):
    """
    Niveau 1 — Attention intra-frame.

    Traite chaque frame indépendamment.
    "Pour chaque frame, quelles zones sont liées spatalement ?"

    Si on a T frames avec N patches chacune :
      → T appels indépendants à TransformerBlock(N tokens)
      → Complexité : T × O(N²)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int = 3,          # nombre de blocs Transformer locaux
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, T, N, D]  — batch de séquences de frames
                  B=batch, T=frames, N=patches, D=embed_dim
                  (pour une seule image : T=1)
        Returns:
            x : [B, T, N, D]
        """
        B, T, N, D = x.shape

        # Traiter chaque frame indépendamment
        # → fusionner B et T en une seule dimension "batch élargi"
        x = rearrange(x, 'b t n d -> (b t) n d')  # [(B×T), N, D]

        for block in self.blocks:
            x = block(x)

        # Restituer la dimension temporelle
        x = rearrange(x, '(b t) n d -> b t n d', b=B, t=T)

        return x


class GlobalTemporalAttention(nn.Module):
    """
    Niveau 2 — Attention inter-frames (temporelle).

    "Comment les frames évoluent-elles dans le temps ?"

    Stratégie : on n'applique pas l'attention sur tous les patches de
    toutes les frames simultanément (trop coûteux). On utilise seulement
    le token CLS de chaque frame comme représentant de la frame.

    Le token CLS (position 0) après l'attention locale contient un
    résumé de tout ce qui est dans la frame → c'est le bon représentant.

    Pour T frames : attention sur T tokens CLS → O(T²) très faible.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int = 2,          # nombre de blocs Transformer globaux
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Encodage positionnel temporel (appris)
        # Le modèle doit savoir quelle frame est "avant" ou "après"
        self.temporal_pos_embedding = None  # initialisé dynamiquement

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def _get_temporal_pos_embedding(
        self, T: int, D: int, device: torch.device
    ) -> torch.Tensor:
        """
        Génère un encodage positionnel temporel de taille [1, T, D].
        Utilise l'encodage sinusoïdal (pas de paramètre à apprendre).
        """
        position = torch.arange(T, device=device).unsqueeze(1)   # [T, 1]
        div_term = torch.exp(
            torch.arange(0, D, 2, device=device) * -(math.log(10000.0) / D)
        )
        pe = torch.zeros(T, D, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:D // 2])
        return pe.unsqueeze(0)  # [1, T, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, T, N, D]  — sortie de l'attention locale

        Returns:
            x : [B, T, N, D]  — avec contexte temporel intégré
        """
        B, T, N, D = x.shape

        if T == 1:
            # Une seule frame → pas d'attention temporelle à faire
            return x

        # ── Extraire les tokens CLS (position 0 de chaque frame) ────────────
        cls_tokens = x[:, :, 0, :]  # [B, T, D]

        # ── Ajouter encodage positionnel temporel ────────────────────────────
        temporal_pe = self._get_temporal_pos_embedding(T, D, x.device)
        cls_tokens = cls_tokens + temporal_pe  # [B, T, D]

        # ── Attention temporelle sur les tokens CLS ───────────────────────────
        for block in self.blocks:
            cls_tokens = block(cls_tokens)
        cls_tokens = self.norm(cls_tokens)  # [B, T, D]

        # ── Réinjecter les CLS mis à jour dans les tokens de chaque frame ────
        # Le CLS enrichi temporellement remplace l'ancien CLS
        x = x.clone()
        x[:, :, 0, :] = cls_tokens

        return x


class HierarchicalTemporalAttention(nn.Module):
    """
    Module d'attention complet : Local + Global.

    Orchestre les deux niveaux d'attention en séquence :
      1. Attention locale  → comprendre chaque frame
      2. Attention globale → comprendre l'évolution temporelle

    Usage :
      - Image seule (T=1)   : seule l'attention locale est active
      - Séquence (T>1)      : les deux niveaux sont actifs

    Args:
        embed_dim      : dimension des tokens
        num_heads      : têtes d'attention
        local_depth    : blocs Transformer pour l'attention locale
        global_depth   : blocs Transformer pour l'attention temporelle
        mlp_ratio      : ratio MLP hidden dim / embed_dim
        dropout        : dropout général
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        local_depth: int = 4,
        global_depth: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.local_attn  = LocalSpatialAttention(
            embed_dim, num_heads, local_depth, mlp_ratio, dropout
        )
        self.global_attn = GlobalTemporalAttention(
            embed_dim, num_heads, global_depth, mlp_ratio, dropout
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(
            f"HTA : local_depth={local_depth}, global_depth={global_depth}, "
            f"embed_dim={embed_dim}, heads={num_heads} "
            f"({total_params:.2f}M params)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, T, N, D]  — séquence de frames avec patches
                Si image seule, T=1 (le modèle fonctionne pour les deux cas)

        Returns:
            cls_output : [B, D]  — représentation globale pour la classification
                         (token CLS de la dernière frame après attention globale)
        """
        # ── Niveau 1 : attention spatiale (intra-frame) ─────────────────────
        x = self.local_attn(x)   # [B, T, N, D]

        # ── Niveau 2 : attention temporelle (inter-frames) ──────────────────
        x = self.global_attn(x)  # [B, T, N, D]

        # ── Extraction du token CLS pour la classification ───────────────────
        # On prend le CLS de la DERNIÈRE frame (elle a le plus de contexte)
        cls_output = x[:, -1, 0, :]  # [B, D]
        cls_output = self.final_norm(cls_output)

        return cls_output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)

    # Test avec une séquence de 4 frames, 98 patches (après DPS 50%), dim=384
    B, T, N, D = 2, 4, 99, 384  # 98 patches + 1 CLS = 99
    fake_sequence = torch.randn(B, T, N, D)

    hta = HierarchicalTemporalAttention(
        embed_dim=D, num_heads=6,
        local_depth=4, global_depth=2
    )

    print(f"\nTest HierarchicalTemporalAttention :")
    print(f"  Entrée  : {fake_sequence.shape}  [B, T, N, D]")

    output = hta(fake_sequence)
    print(f"  Sortie  : {output.shape}  [B, D]  (token CLS pour classification)")

    n_params = sum(p.numel() for p in hta.parameters()) / 1e6
    print(f"  Params  : {n_params:.2f}M")

    # Test avec image seule (T=1)
    single_image_tokens = torch.randn(B, 1, N, D)
    output_single = hta(single_image_tokens)
    print(f"\n  Test image seule (T=1) : {single_image_tokens.shape} → {output_single.shape} ✅")
