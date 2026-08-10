"""
MedViT-Lite — CNN Patch Encoder (Backbone)
===========================================
Ce module convertit une image brute en tokens de patches,
exactement comme le fait ViT, mais avec un encodeur CNN léger
avant le découpage en patches.

Pourquoi un CNN avant le Transformer ?
---------------------------------------
ViT vanilla découpe l'image directement en patches 16×16 et les
projette linéairement. Cela fonctionne bien, mais nécessite beaucoup
de données pour apprendre les features locales from scratch.

Notre approche hybride :
  Image → CNN léger (features locales) → Découpage en patches → Tokens

Le CNN apprend des features locales (bords, textures) efficacement.
Le Transformer s'occupe ensuite des relations globales.
C'est l'approche de CvT (Convolutional Vision Transformer, Wu et al. 2021).

Architecture choisie : Convolutional Stem + Patch Projection
  - 3 couches conv (stride progressif) → feature map réduite
  - Puis projection linéaire vers embed_dim
  - Résultat : N patches de dimension embed_dim
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class ConvStem(nn.Module):
    """
    Encodeur CNN léger pour l'extraction de features locales.

    Réduit l'image 224×224×3 en feature map 56×56×stem_dim
    via 3 couches convolutives avec BatchNorm et GELU.

    Pourquoi stride=2 répété 2 fois (et pas stride=16 d'un coup) ?
    → Plusieurs petits strides apprennent de meilleures representations
      que un grand stride unique (voir ResNet vs AlexNet).
    """

    def __init__(self, in_channels: int = 3, stem_dim: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            # Conv 1 : 224×224×3 → 112×112×stem_dim/2
            nn.Conv2d(in_channels, stem_dim // 2, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim // 2),
            nn.GELU(),

            # Conv 2 : 112×112×stem_dim/2 → 56×56×stem_dim
            nn.Conv2d(stem_dim // 2, stem_dim, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),

            # Conv 3 : 56×56×stem_dim → 56×56×stem_dim (raffinage, stride=1)
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 3, 224, 224]
        Returns:
            features : [B, stem_dim, 56, 56]
        """
        return self.stem(x)


class PatchEmbedding(nn.Module):
    """
    Convertit la feature map CNN en séquence de tokens de patches.

    Opération :
      feature map [B, C, H, W] → patches [B, N, embed_dim]
      où N = (H/patch_size) × (W/patch_size)

    Pour notre config par défaut :
      Input  : [B, 64, 56, 56]  (sortie ConvStem)
      Patch  : 4×4 (sur la feature map, = 16×16 sur l'image originale)
      Output : [B, 196, 384]    (196 = 14×14 patches)
    """

    def __init__(
        self,
        stem_dim: int = 64,
        patch_size: int = 4,      # patch sur la feature map (≡ 16px sur l'image)
        embed_dim: int = 384,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Projection : patch 4×4×stem_dim → vecteur embed_dim
        self.proj = nn.Conv2d(
            stem_dim, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            features : [B, stem_dim, H, W]

        Returns:
            tokens   : [B, N, embed_dim]  (N = H/patch_size × W/patch_size)
            h_patches: nombre de patches en hauteur
            w_patches: nombre de patches en largeur
        """
        # [B, embed_dim, H/patch_size, W/patch_size]
        x = self.proj(features)
        h_patches, w_patches = x.shape[2], x.shape[3]

        # Aplatir les patches en séquence
        # [B, embed_dim, h, w] → [B, h*w, embed_dim]
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.norm(x)

        return x, h_patches, w_patches


class CLSTokenAndPositionalEncoding(nn.Module):
    """
    Ajoute un token [CLS] et un encodage positionnel appris.

    Token [CLS] :
      Comme dans BERT, on ajoute un token spécial en tête de séquence.
      Après le Transformer, ce token agrège l'information globale de l'image
      et est utilisé pour la classification.

    Encodage positionnel :
      Appris (pas sinusoïdal) → le modèle apprend quelle position est où.
      Taille : [1, N+1, embed_dim] (N patches + 1 CLS)
    """

    def __init__(self, embed_dim: int, num_patches: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens : [B, N, D]
        Returns:
            tokens : [B, N+1, D]  (avec CLS en position 0)
        """
        B = tokens.shape[0]

        # Répliquer le CLS token pour tout le batch
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]

        # Concaténer CLS + patches
        tokens = torch.cat([cls, tokens], dim=1)  # [B, N+1, D]

        # Ajouter l'encodage positionnel
        tokens = tokens + self.pos_embedding

        return tokens


class CNNPatchEncoder(nn.Module):
    """
    Module complet d'encodage : Image → Tokens de patches.

    Assemble ConvStem + PatchEmbedding + CLS + PositionalEncoding.

    Args:
        image_size  : taille de l'image carrée (224)
        in_channels : canaux d'entrée (3 pour RGB)
        patch_size  : taille de patch sur l'image originale (16)
        embed_dim   : dimension des tokens de sortie (384)
        stem_dim    : dimension intermédiaire du CNN stem (64)
        dropout     : dropout sur les embeddings
    """

    def __init__(
        self,
        image_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 384,
        stem_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert image_size % patch_size == 0, \
            f"image_size ({image_size}) doit être divisible par patch_size ({patch_size})"

        # Nombre de patches = (image_size / patch_size)²
        # ConvStem réduit ×4 → patch_size_on_features = patch_size / 4
        stem_reduction = 4
        feature_map_size = image_size // stem_reduction       # 56
        patch_size_on_features = patch_size // stem_reduction  # 4
        self.num_patches = (feature_map_size // patch_size_on_features) ** 2  # 196

        self.stem = ConvStem(in_channels, stem_dim)
        self.patch_embed = PatchEmbedding(stem_dim, patch_size_on_features, embed_dim)
        self.cls_pos = CLSTokenAndPositionalEncoding(embed_dim, self.num_patches)
        self.dropout = nn.Dropout(dropout)

        logger.info(
            f"CNNPatchEncoder : {image_size}×{image_size}px → "
            f"{self.num_patches} patches × {embed_dim}d"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, C, H, W]  image batch
        Returns:
            tokens : [B, N+1, embed_dim]  (N patches + 1 CLS)
        """
        # CNN features
        features = self.stem(x)

        # Patch tokens
        tokens, h, w = self.patch_embed(features)

        # CLS + positional encoding
        tokens = self.cls_pos(tokens)

        # Dropout
        tokens = self.dropout(tokens)

        return tokens


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)

    encoder = CNNPatchEncoder(
        image_size=224, patch_size=16, embed_dim=384, stem_dim=64
    )

    x = torch.randn(4, 3, 224, 224)
    tokens = encoder(x)

    n_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    print(f"\nTest CNNPatchEncoder :")
    print(f"  Entrée  : {x.shape}")
    print(f"  Sortie  : {tokens.shape}  ([B, N_patches+1, embed_dim])")
    print(f"  Patches : {encoder.num_patches} + 1 CLS = {encoder.num_patches+1} tokens")
    print(f"  Params  : {n_params:.2f}M")
