"""
MedViT-Lite — Entraînement Baseline ResNet-50
==============================================
Ce script entraîne ResNet-50 sur ChestMNIST comme référence CNN.

Pourquoi ResNet-50 comme baseline ?
  - Modèle CNN de référence le plus utilisé en imagerie médicale
  - Bien documenté dans la littérature (AUC ~0.82-0.85 sur NIH Chest X-Ray)
  - Sert de plancher de performance : MedViT-Lite doit faire mieux
    ou de manière équivalente avec moins de calcul

Utilisation :
  python experiments/baseline_cnn.py --config configs/base.yaml
  python experiments/baseline_cnn.py --config configs/base.yaml --gpu 0
"""

import argparse
import logging
import sys
import os
import torch
import torch.nn as nn
import timm
import yaml

# Ajouter le répertoire racine au path Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.datasets.chest_mnist import build_dataloaders, PATHOLOGY_NAMES
from training.losses import build_loss
from training.metrics import MetricsTracker
from training.trainer import Trainer, build_optimizer, build_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Construit ResNet-50 adapté à la classification médicale multi-label.

    Modifications par rapport au ResNet-50 standard ImageNet :
      - Dernière couche FC adaptée pour num_classes (14 ici)
      - Pas de softmax (on utilise sigmoid pour multi-label)
      - Dropout ajouté avant la couche finale (régularisation)
    """
    # timm gère le téléchargement des poids pré-entraînés
    model = timm.create_model(
        "resnet50",
        pretrained=pretrained,
        num_classes=0,        # Enlève la tête ImageNet (1000 classes)
        global_pool="avg",    # Global Average Pooling
    )
    embed_dim = model.num_features  # 2048 pour ResNet-50

    # Nouvelle tête adaptée au multi-label médical
    classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(embed_dim, 256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )

    # Attacher la tête au modèle
    model.head = classifier

    # Modifier le forward pour utiliser notre tête
    original_forward = model.forward_features

    def custom_forward(x):
        features = original_forward(x)
        return model.head(features)

    model.forward = custom_forward

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"ResNet-50 : {n_params:.1f}M paramètres (pretrained={pretrained})")
    return model


def main(args):
    # ── Chargement de la configuration ────────────────────────────────────────
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override GPU si spécifié
    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() \
             else "cpu"
    logger.info(f"Device : {device}")

    torch.manual_seed(config["project"]["seed"])

    # ── Données ───────────────────────────────────────────────────────────────
    logger.info("Chargement des données...")
    train_loader, val_loader, test_loader = build_dataloaders(
        image_size  = config["data"]["image_size"],
        batch_size  = config["training"]["batch_size"],
        num_workers = config["data"]["num_workers"],
    )

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = build_resnet50(
        num_classes=config["data"]["num_classes"],
        pretrained=not args.from_scratch,
    )

    # ── Poids de classe (pour la perte pondérée) ──────────────────────────────
    pos_weight = train_loader.dataset.get_class_weights()
    logger.info(f"Poids de classe calculés (min={pos_weight.min():.2f}, max={pos_weight.max():.2f})")

    # ── Perte, Optimizer, Scheduler ───────────────────────────────────────────
    loss_fn   = build_loss(config["training"]["loss"], pos_weight)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, config["training"]["epochs"])

    # ── Métriques ─────────────────────────────────────────────────────────────
    metrics = MetricsTracker(
        threshold=config["evaluation"]["threshold"],
        pathology_names=PATHOLOGY_NAMES,
    )

    # ── Entraînement ──────────────────────────────────────────────────────────
    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        optimizer    = optimizer,
        scheduler    = scheduler,
        loss_fn      = loss_fn,
        metrics      = metrics,
        config       = config["training"],
        device       = device,
        save_dir     = config["logging"]["save_dir"],
        use_wandb    = config["logging"]["use_wandb"] and not args.no_wandb,
        run_name     = "baseline_resnet50",
    )

    trainer.train()

    # ── Test final ────────────────────────────────────────────────────────────
    logger.info("Évaluation sur le test set...")
    best_ckpt_path = os.path.join(
        config["logging"]["save_dir"], "best_baseline_resnet50.pth"
    )
    if os.path.exists(best_ckpt_path):
        Trainer.load_checkpoint(best_ckpt_path, model, device)

    model.eval()
    test_metrics_tracker = MetricsTracker(
        threshold=0.5, pathology_names=PATHOLOGY_NAMES
    )

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            test_metrics_tracker.update(logits, labels)

    test_metrics = test_metrics_tracker.compute()
    logger.info("Résultats TEST SET (ResNet-50 baseline) :")
    logger.info(test_metrics_tracker.summary(test_metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline ResNet-50")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index (-1 pour CPU)")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Entraîner sans poids pré-entraînés")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Désactiver W&B logging")
    args = parser.parse_args()

    main(args)
