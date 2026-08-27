"""
MedViT-Lite — Script d'entraînement principal
==============================================
Entraîne le modèle MedViT-Lite complet avec toutes ses innovations.

Utilisation :
  # Configuration complète (DPS + SFC + HTA)
  python experiments/medvit_lite_train.py --config configs/base.yaml

  # Ablation : désactiver le DPS
  python experiments/medvit_lite_train.py --config configs/base.yaml --no-dps

  # Ablation : désactiver le SFC
  python experiments/medvit_lite_train.py --config configs/base.yaml --no-sfc

  # Sans les deux innovations (= ViT de base)
  python experiments/medvit_lite_train.py --config configs/base.yaml --no-dps --no-sfc
"""

import argparse
import logging
import sys
import os
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.datasets.chest_mnist import build_dataloaders, PATHOLOGY_NAMES
from models.medvit_lite import MedViTLite
from training.losses import build_loss
from training.metrics import MetricsTracker
from training.trainer import Trainer, build_optimizer, build_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(args):
    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count()
    logger.info(f"Device : {device} ({n_gpus} GPU(s))")

    torch.manual_seed(config["project"]["seed"])
    if device == "cuda":
        torch.backends.cudnn.benchmark = True  # Optimise les convolutions

    # ── Nom de l'expérience (reflète la configuration) ────────────────────────
    run_name = "medvit_lite"
    if args.no_dps:
        run_name += "_noDPS"
    if args.no_sfc:
        run_name += "_noSFC"
    if args.no_dps and args.no_sfc:
        run_name = "medvit_lite_noInnovations"

    logger.info(f"Expérience : {run_name}")

    # ── Données ───────────────────────────────────────────────────────────────
    logger.info("Chargement des données...")
    # Avec multi-GPU, le batch_size est par GPU → ×num_gpus pour le total
    batch_size = config["training"]["batch_size"] * max(1, n_gpus)
    logger.info(f"Batch size effectif : {batch_size} ({config['training']['batch_size']} × {max(1,n_gpus)} GPUs)")

    train_loader, val_loader, test_loader = build_dataloaders(
        image_size  = config["data"]["image_size"],
        batch_size  = config["training"]["batch_size"],  # par GPU
        num_workers = config["data"]["num_workers"],
        dry_run     = args.dry_run,
    )

    # ── Modèle ────────────────────────────────────────────────────────────────
    model_cfg    = config["model"]
    sparsi_cfg   = model_cfg["sparsifier"]
    cache_cfg    = model_cfg["frame_cache"]
    attn_cfg     = model_cfg["attention"]
    head_cfg     = model_cfg["head"]

    model = MedViTLite(
        num_classes  = config["data"]["num_classes"],
        image_size   = config["data"]["image_size"],
        patch_size   = model_cfg["patch_size"],
        embed_dim    = model_cfg["embed_dim"],
        local_depth  = 4,
        global_depth = 2,
        num_heads    = attn_cfg["num_heads"],
        mlp_ratio    = 4.0,
        dropout      = attn_cfg["dropout"],
        head_dropout = head_cfg["dropout"],

        # Innovations (peuvent être désactivées pour ablation)
        use_dps      = sparsi_cfg["enabled"] and not args.no_dps,
        keep_ratio   = sparsi_cfg["keep_ratio"],
        use_sfc      = cache_cfg["enabled"] and not args.no_sfc,
        cache_size   = cache_cfg["cache_size"],
        sim_threshold= cache_cfg["similarity_threshold"],
        mc_samples   = head_cfg["mc_samples"],
    )

    # Afficher le nombre de paramètres par module
    params = model.count_parameters()
    logger.info("Paramètres du modèle :")
    for name, count in params.items():
        logger.info(f"  {name:<15} : {count:.2f}M")

    # ── Perte, Optimizer, Scheduler ───────────────────────────────────────────
    pos_weight = train_loader.dataset.get_class_weights()
    loss_fn    = build_loss(config["training"]["loss"], pos_weight)
    optimizer  = build_optimizer(model, config)
    scheduler  = build_scheduler(
        optimizer, config, config["training"]["epochs"]
    )

    # ── Métriques ─────────────────────────────────────────────────────────────
    metrics = MetricsTracker(
        threshold       = config["evaluation"]["threshold"],
        pathology_names = PATHOLOGY_NAMES,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
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
        run_name     = run_name,
        dry_run      = args.dry_run,
        resume       = args.resume,
    )

    trainer.train()

    # ── Évaluation finale sur le test set ─────────────────────────────────────
    logger.info("Évaluation finale sur le test set...")
    best_ckpt = os.path.join(config["logging"]["save_dir"], f"best_{run_name}.pth")

    if os.path.exists(best_ckpt):
        Trainer.load_checkpoint(best_ckpt, model, device)

    model.eval()
    model.reset_cache()

    test_tracker = MetricsTracker(threshold=0.5, pathology_names=PATHOLOGY_NAMES)
    image_size = config["data"].get("image_size", 224)
    with torch.no_grad():
        for images, labels in test_loader:
            images = Trainer.preprocess_batch(
                images, device=device, image_size=image_size, is_train=False
            )
            labels = labels.to(device)
            logits = model(images)
            test_tracker.update(logits, labels)

    test_metrics = test_tracker.compute()
    logger.info(f"=== Résultats TEST SET ({run_name}) ===")
    logger.info(test_tracker.summary(test_metrics))

    # Sauvegarder les résultats pour comparaison
    results_dir = config["logging"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{run_name}_test_results.yaml")
    clean_metrics = {k: float(v) for k, v in test_metrics.items()}
    with open(results_path, "w") as f:
        yaml.dump(clean_metrics, f, default_flow_style=False)
    logger.info(f"Résultats sauvegardés : {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedViT-Lite Training")
    parser.add_argument("--config",    type=str, default="configs/base.yaml")
    parser.add_argument("--no-dps",   action="store_true",
                        help="Ablation : désactiver Dynamic Patch Sparsifier")
    parser.add_argument("--no-sfc",   action="store_true",
                        help="Ablation : désactiver Selective Frame Cache")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Désactiver W&B logging")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Exécuter un test rapide sur un sous-ensemble de données (2 epochs)")
    parser.add_argument("--resume",   action="store_true",
                        help="Reprendre l'entraînement depuis le dernier checkpoint")
    args = parser.parse_args()
    main(args)
