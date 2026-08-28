"""
MedViT-Lite — Visual Explainability CLI
======================================
Génère une figure multi-panneaux complète pour une radiographie :
1. Image originale
2. Prédictions multi-labels avec barre d'incertitude (Monte Carlo Dropout)
3. Masque DPS (zones anatomiques sélectionnées)
4. Carte d'activation Grad-CAM++ pour la pathologie dominante
"""

import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from models.medvit_lite import MedViTLite
from data.datasets.chest_mnist import PATHOLOGY_NAMES, build_dataloaders
from training.trainer import Trainer
from explainability.gradcam import GradCAMPlusPlus
from explainability.attention_rollout import visualize_dps_mask


def run_visual_explanation(
    checkpoint_path: str,
    config_path: str = "configs/base.yaml",
    output_dir: str = "results/explainability",
    sample_index: int = 0,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Charger les données de test
    _, _, test_loader = build_dataloaders(config["data"], dry_run=False)
    dataset = test_loader.dataset

    raw_img, true_labels = dataset[sample_index]
    # Prétraitement GPU
    img_tensor = Trainer.preprocess_batch(raw_img.unsqueeze(0), device=device, is_train=False)

    # 2. Charger le modèle
    model_cfg = config["model"]
    model = MedViTLite(
        num_classes=config["data"]["num_classes"],
        image_size=config["data"]["image_size"],
        patch_size=model_cfg["patch_size"],
        embed_dim=model_cfg["embed_dim"],
        local_depth=4,
        global_depth=2,
        num_heads=model_cfg["attention"]["num_heads"],
        use_dps=model_cfg["sparsifier"]["enabled"],
        keep_ratio=model_cfg["sparsifier"]["keep_ratio"],
        use_sfc=model_cfg["frame_cache"]["enabled"],
        mc_samples=model_cfg["head"]["mc_samples"],
    )

    if os.path.exists(checkpoint_path):
        Trainer.load_checkpoint(checkpoint_path, model, str(device))
        print(f"✅ Checkpoint chargé : {checkpoint_path}")
    else:
        print(f"⚠️ Checkpoint introuvable : {checkpoint_path}, utilisation des poids par défaut.")

    model.to(device)
    model.eval()

    # 3. Prédiction avec incertitude (MC Dropout)
    with torch.no_grad():
        mean_probs, uncertainty = model.predict_with_uncertainty(img_tensor, num_samples=15)

    probs_np = mean_probs.squeeze().cpu().numpy()
    unc_np = uncertainty.squeeze().cpu().numpy()
    top_idx = int(np.argmax(probs_np))
    top_disease = PATHOLOGY_NAMES[top_idx]
    top_prob = probs_np[top_idx]

    # 4. Grad-CAM++
    target_layer = model.attention.local_blocks[-1].norm1
    gradcam = GradCAMPlusPlus(model, target_layer)
    heatmap = gradcam.generate_heatmap(img_tensor, target_class=top_idx)

    # Convertir l'image d'entrée en image 2D pour affichage
    display_img = img_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min() + 1e-6)
    overlay = gradcam.overlay_heatmap(display_img, heatmap)

    # 5. Création de la figure globale
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panneau 1 : Image originale
    axes[0].imshow(display_img)
    axes[0].set_title("Radiographie Thoracique (Entrée)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panneau 2 : GradCAM++
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Grad-CAM++ Focus : {top_disease}\n(Score: {top_prob*100:.1f}% ± {unc_np[top_idx]*100:.1f}%)",
        fontsize=12, fontweight="bold", color="darkred"
    )
    axes[1].axis("off")

    # Panneau 3 : Diagnostic Multi-labels & Incertitude
    y_pos = np.arange(len(PATHOLOGY_NAMES))
    axes[2].barh(y_pos, probs_np * 100, xerr=unc_np * 100, align="center", color="#3B82F6", alpha=0.8, capsize=3)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(PATHOLOGY_NAMES, fontsize=9)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Probabilité Prédite (%) ± Incertitude MC", fontsize=10)
    axes[2].set_title("Diagnostic Multi-Labels & Confiance", fontsize=12, fontweight="bold")
    axes[2].set_xlim(0, 100)
    axes[2].grid(axis="x", linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_plot = os.path.join(output_dir, f"explainability_sample_{sample_index}.png")
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Rapport visuel sauvegardé avec succès : {output_plot}")
    gradcam.remove_hooks()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedViT-Lite Explainability Visualizer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_medvit_lite.pth")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/explainability")
    args = parser.parse_args()

    run_visual_explanation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        sample_index=args.sample_index,
    )
