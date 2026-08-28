"""
MedViT-Lite — Explainability Module: GradCAM++
=============================================
Génère des cartes d'activation thermique (heatmaps) pour expliquer
visuellement les prédictions multi-labels des modèles CNN et Transformers médicaux.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple
from PIL import Image


class GradCAMPlusPlus:
    """
    Grad-CAM++ pour modèles de classification médicale multi-label.
    Compatible avec les CNN (ResNet) et les Vision Transformers (MedViT-Lite).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """
        Calcule la carte thermique Grad-CAM++ pour une classe cible donnée.

        Args:
            input_tensor: [1, C, H, W]
            target_class: indice de la pathologie cible (0 à 13)

        Returns:
            heatmap: [H, W] numpy array normalisé entre 0 et 1
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        score = logits[0, target_class]

        # Backward pass
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Les gradients ou activations n'ont pas été capturés.")

        # [B, C, H, W] ou [B, N, C]
        grads = self.gradients
        acts = self.activations

        if grads.ndim == 3:  # ViT tokens [B, N, D]
            B, N, D = grads.shape
            side = int(np.sqrt(N))
            grads = grads.permute(0, 2, 1).view(B, D, side, side)
            acts = acts.permute(0, 2, 1).view(B, D, side, side)

        # Grad-CAM++ calcul des poids alpha
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads

        sum_acts = torch.sum(acts, dim=(-2, -1), keepdim=True)
        eps = 1e-7
        aij = grads_power_2 / (2 * grads_power_2 + sum_acts * grads_power_3 + eps)
        aij = torch.where(grads != 0, aij, torch.zeros_like(aij))

        weights = torch.sum(aij * F.relu(grads), dim=(-2, -1), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalisation et redimensionnement à la taille de l'image d'entrée
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().detach().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        colormap: str = "jet",
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Superpose la heatmap colorisée sur l'image radiologique."""
        import matplotlib.cm as cm

        cmap = cm.get_cmap(colormap)
        colored_heatmap = cmap(heatmap)[:, :, :3]  # RGB [0, 1]

        if image.max() > 1.0:
            image = image / 255.0

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        overlay = (1 - alpha) * image + alpha * colored_heatmap
        overlay = np.clip(overlay, 0.0, 1.0)
        return overlay

    def remove_hooks(self):
        """Désenregistre les hooks pour libérer la mémoire."""
        for handle in self.handles:
            handle.remove()
