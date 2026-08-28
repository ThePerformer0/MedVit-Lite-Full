"""
MedViT-Lite — Explainability: Attention Rollout & DPS Visualizer
===============================================================
Permet de visualiser :
1. Les matrices d'attention cumulées à travers les couches Transformer (Attention Rollout)
2. Les patches conservés vs élagués par le Dynamic Patch Sparsifier (DPS)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def compute_attention_rollout(
    attention_matrices: List[torch.Tensor],
    discard_ratio: float = 0.1,
    head_fusion: str = "mean",
) -> np.ndarray:
    """
    Calcule l'Attention Rollout (Abnar & Zuidema, 2020) sur une liste de matrices d'attention.

    Args:
        attention_matrices: Liste de tenseurs [B, num_heads, N, N] pour chaque couche
        discard_ratio: Pourcentage des poids les plus faibles à éliminer pour débruiter
        head_fusion: "mean", "max" ou "min"

    Returns:
        rollout: [N, N] matrice d'attention cumulée
    """
    result = torch.eye(attention_matrices[0].shape[-1])

    with torch.no_grad():
        for attn in attention_matrices:
            # Fusion des têtes d'attention
            if head_fusion == "mean":
                attn_fused = attn.mean(dim=1).squeeze(0)
            elif head_fusion == "max":
                attn_fused = attn.max(dim=1)[0].squeeze(0)
            elif head_fusion == "min":
                attn_fused = attn.min(dim=1)[0].squeeze(0)
            else:
                raise ValueError(f"Fusion inconnue : {head_fusion}")

            # Éliminer les bruits résiduels
            flat = attn_fused.flatten()
            val, _ = torch.topk(flat, int(len(flat) * discard_ratio), largest=False)
            if len(val) > 0:
                threshold = val[-1]
                attn_fused = torch.where(attn_fused <= threshold, torch.tensor(0.0), attn_fused)

            # Normalisation avec identité (skip connection)
            I = torch.eye(attn_fused.shape[-1])
            a = 0.5 * attn_fused + 0.5 * I
            a = a / a.sum(dim=-1, keepdim=True)

            result = torch.matmul(a.cpu(), result)

    return result.numpy()


def visualize_dps_mask(
    image: np.ndarray,
    keep_indices: torch.Tensor,
    patch_size: int = 16,
    image_size: int = 224,
) -> np.ndarray:
    """
    Crée une visualisation visuelle des zones conservées par le Dynamic Patch Sparsifier.
    Les patches élagués sont assombris pour mettre en évidence les zones cliniques actives.

    Args:
        image: [H, W, 3] ou [H, W] image originale
        keep_indices: Indices 1D des patches conservés [K]
        patch_size: taille d'un patch (ex: 16)
        image_size: taille de l'image (ex: 224)

    Returns:
        visualized: Image avec masque DPS appliqué
    """
    grid_size = image_size // patch_size
    mask = np.zeros((grid_size, grid_size), dtype=np.float32)

    indices = keep_indices.detach().cpu().numpy().flatten()
    for idx in indices:
        r = idx // grid_size
        c = idx % grid_size
        if r < grid_size and c < grid_size:
            mask[r, c] = 1.0

    # Redimensionner le masque en pleine résolution
    import scipy.ndimage as ndimage
    zoom_factors = (image_size / grid_size, image_size / grid_size)
    full_mask = ndimage.zoom(mask, zoom_factors, order=0)

    if image.max() > 1.0:
        image = image / 255.0

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # Assombrir les patches ignorés (0.2x luminosité) et surligner les patches actifs
    dimmed = image * 0.25
    highlighted = image * full_mask[:, :, None] + dimmed * (1.0 - full_mask[:, :, None])
    return np.clip(highlighted, 0.0, 1.0)
