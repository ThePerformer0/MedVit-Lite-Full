"""
MedViT-Lite — Trainer principal
=================================
Gère l'entraînement complet : boucle epoch, validation, checkpoints,
multi-GPU (DataParallel), mixed precision (AMP), early stopping, logging.

Design :
  - Compatible avec n'importe quel modèle PyTorch
  - Config-driven (lit depuis un dict / YAML)
  - Sauvegarde le meilleur checkpoint (sur val_auc)
  - W&B logging optionnel
"""

import os
import time
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast   # API unifiée (torch 2.x)
from typing import Optional, Dict, Any
import logging
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, MofNCompleteColumn

logger = logging.getLogger(__name__)
console = Console()


class Trainer:
    """
    Classe d'entraînement générique pour MedViT-Lite et ses baselines.

    Args:
        model        : modèle PyTorch (non-wrappé, le Trainer s'occupe du multi-GPU)
        train_loader : DataLoader d'entraînement
        val_loader   : DataLoader de validation
        optimizer    : optimizer PyTorch
        scheduler    : scheduler de learning rate
        loss_fn      : fonction de perte
        metrics      : MetricsTracker
        config       : dict de configuration (section "training" du YAML)
        device       : device PyTorch ('cuda' ou 'cpu')
        save_dir     : répertoire de sauvegarde des checkpoints
        use_wandb    : activer le logging W&B
        run_name     : nom de l'expérience (pour les logs)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: nn.Module,
        metrics,
        config: dict,
        device: str = "cuda",
        save_dir: str = "./checkpoints",
        use_wandb: bool = False,
        run_name: str = "medvit_lite",
    ):
        self.config = config
        self.device = torch.device(device)
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        self.run_name = run_name

        os.makedirs(save_dir, exist_ok=True)

        # ── Multi-GPU (DataParallel si plusieurs GPUs) ────────────────────────
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            logger.info(f"Multi-GPU : {self.num_gpus} GPUs détectés → DataParallel")
            model = nn.DataParallel(model)
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.loss_fn      = loss_fn.to(self.device)
        self.metrics      = metrics

        # ── Mixed Precision (AMP) ────────────────────────────────────────────
        self.use_amp = config.get("amp", True) and str(device).startswith("cuda")
        self.scaler  = GradScaler("cuda") if self.use_amp else None
        if self.use_amp:
            logger.info("Mixed Precision (AMP fp16) activé")

        # ── Hyperparamètres ───────────────────────────────────────────────────
        self.max_epochs  = config.get("epochs", 50)
        self.grad_accum  = config.get("gradient_accumulation", 1)
        self.log_every   = config.get("log_every_n_steps", 10)

        # ── Early Stopping ────────────────────────────────────────────────────
        es_config = config.get("early_stopping", {})
        self.es_patience = es_config.get("patience", 10)
        self.es_monitor  = es_config.get("monitor", "val_auc_mean")
        self.es_mode     = es_config.get("mode", "max")

        self.best_metric = -float("inf") if self.es_mode == "max" else float("inf")
        self.patience_counter = 0
        self.best_epoch = 0

        # ── W&B ──────────────────────────────────────────────────────────────
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.get("wandb_project", "medvit-lite"),
                    name=run_name,
                    config=config,
                )
                self.wandb = wandb
                logger.info(f"W&B initialisé : projet={config.get('wandb_project')}")
            except ImportError:
                logger.warning("wandb non installé — logging W&B désactivé")
                self.use_wandb = False

        logger.info(
            f"Trainer prêt : {self.max_epochs} epochs, "
            f"device={device}, amp={self.use_amp}, gpus={self.num_gpus}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    def train(self):
        """Boucle d'entraînement principale."""
        console.rule(f"[bold blue]Début de l'entraînement : {self.run_name}")

        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()

            # ── Epoch d'entraînement ──────────────────────────────────────────
            train_loss = self._train_epoch(epoch)

            # ── Validation ────────────────────────────────────────────────────
            val_metrics = self._validate_epoch(epoch)
            val_loss    = val_metrics.pop("val_loss", 0.0)

            # ── Scheduler ─────────────────────────────────────────────────────
            if self.scheduler is not None:
                self.scheduler.step()

            # ── Logging ───────────────────────────────────────────────────────
            epoch_time = time.time() - epoch_start
            self._log_epoch(epoch, train_loss, val_loss, val_metrics, epoch_time)

            # ── Checkpoint (sauvegarde si meilleur résultat) ──────────────────
            monitor_value = val_metrics.get(self.es_monitor, 0.0)
            is_best = self._is_better(monitor_value)

            if is_best:
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                self.best_metric = monitor_value
                self.best_epoch  = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Sauvegarde checkpoint toutes les 5 epochs
            if epoch % 5 == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)

            # ── Early Stopping ────────────────────────────────────────────────
            if self.patience_counter >= self.es_patience:
                console.print(
                    f"\n[yellow]⚠ Early stopping à l'epoch {epoch} "
                    f"(patience={self.es_patience} sans amélioration)\n"
                    f"  Meilleur résultat : epoch {self.best_epoch}, "
                    f"{self.es_monitor}={self.best_metric:.4f}[/yellow]"
                )
                break

        console.rule("[bold green]Entraînement terminé")
        console.print(
            f"Meilleur modèle : epoch {self.best_epoch}, "
            f"{self.es_monitor} = {self.best_metric:.4f}\n"
            f"Checkpoint : {self.save_dir}/best_{self.run_name}.pth"
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _train_epoch(self, epoch: int) -> float:
        """Une epoch d'entraînement."""
        self.model.train()
        self.metrics.reset()

        total_loss = 0.0
        n_batches  = len(self.train_loader)

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            console=console, transient=True,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Epoch {epoch}/{self.max_epochs} — Train",
                total=n_batches
            )

            for step, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # ── Forward avec AMP ──────────────────────────────────────────
                with autocast("cuda", enabled=self.use_amp):
                    logits = self.model(images)
                    loss   = self.loss_fn(logits, labels)
                    loss   = loss / self.grad_accum

                # ── Backward ──────────────────────────────────────────────────
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # ── Gradient accumulation ─────────────────────────────────────
                if (step + 1) % self.grad_accum == 0:
                    if self.use_amp:
                        # Clipping avant unscale
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                # ── Métriques ─────────────────────────────────────────────────
                total_loss += loss.item() * self.grad_accum
                self.metrics.update(logits.detach(), labels.detach())

                # ── W&B step log ──────────────────────────────────────────────
                if self.use_wandb and step % self.log_every == 0:
                    self.wandb.log({
                        "train/step_loss": loss.item() * self.grad_accum,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                    })

                progress.advance(task)

            # Nettoyage mémoire de fin d'epoch
            del images, labels, logits, loss
            gc.collect()

        avg_loss = total_loss / n_batches
        return avg_loss

    # ─────────────────────────────────────────────────────────────────────────
    def _validate_epoch(self, epoch: int) -> dict:
        """Une epoch de validation."""
        self.model.eval()
        self.metrics.reset()

        total_loss = 0.0
        n_batches  = len(self.val_loader)

        with torch.no_grad():
            with Progress(
                "[progress.description]{task.description}",
                BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                console=console, transient=True,
            ) as progress:
                task = progress.add_task(
                    f"[magenta]Epoch {epoch}/{self.max_epochs} — Val  ",
                    total=n_batches
                )

                for images, labels in self.val_loader:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    with autocast("cuda", enabled=self.use_amp):
                        logits = self.model(images)
                        loss   = self.loss_fn(logits, labels)

                    total_loss += loss.item()
                    self.metrics.update(logits, labels)
                    progress.advance(task)

                # Nettoyage mémoire de fin d'epoch
                del images, labels, logits, loss
                gc.collect()

        val_metrics = self.metrics.compute()
        val_metrics["val_loss"] = total_loss / n_batches

        # Préfixer les métriques avec "val_"
        val_metrics = {
            (k if k == "val_loss" else f"val_{k}"): v
            for k, v in val_metrics.items()
        }

        if self.use_wandb:
            self.wandb.log({f"val/{k}": v for k, v in val_metrics.items()})

        return val_metrics

    # ─────────────────────────────────────────────────────────────────────────
    def _log_epoch(
        self, epoch: int, train_loss: float, val_loss: float,
        val_metrics: dict, epoch_time: float
    ):
        auc  = val_metrics.get("val_auc_mean", 0)
        sens = val_metrics.get("val_sens@spec95_mean", 0)
        f1   = val_metrics.get("val_f1_mean", 0)
        lr   = self.optimizer.param_groups[0]["lr"]

        console.print(
            f"[bold]Epoch {epoch:3d}[/bold] | "
            f"loss train={train_loss:.4f} val={val_loss:.4f} | "
            f"AUC={auc:.4f}  Sens@95={sens:.4f}  F1={f1:.4f} | "
            f"lr={lr:.2e}  [{epoch_time:.1f}s]"
        )

        if self.use_wandb:
            self.wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/auc_mean": auc,
                "val/sens_at_spec95": sens,
                "val/f1_mean": f1,
                "train/lr": lr,
            })

    # ─────────────────────────────────────────────────────────────────────────
    def _is_better(self, current: float) -> bool:
        """Vérifie si la métrique courante est meilleure que le meilleur."""
        if self.es_mode == "max":
            return current > self.best_metric
        return current < self.best_metric

    def _save_checkpoint(
        self, epoch: int, metrics: dict, is_best: bool = False
    ):
        """Sauvegarde un checkpoint du modèle."""
        # Gérer le cas DataParallel (extraire le module réel)
        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, nn.DataParallel)
            else self.model.state_dict()
        )

        checkpoint = {
            "epoch":          epoch,
            "model_state":    model_state,
            "optimizer_state": self.optimizer.state_dict(),
            "metrics":        metrics,
            "best_metric":    self.best_metric,
            "run_name":       self.run_name,
        }

        filename = (
            f"best_{self.run_name}.pth"
            if is_best
            else f"{self.run_name}_epoch{epoch:03d}.pth"
        )
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)

        if is_best:
            logger.info(
                f"✅ Nouveau meilleur checkpoint : {path} "
                f"({self.es_monitor}={self.best_metric:.4f})"
            )

    @classmethod
    def load_checkpoint(
        cls, checkpoint_path: str, model: nn.Module, device: str = "cuda"
    ) -> Dict[str, Any]:
        """Charge un checkpoint et retourne les métadonnées."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        logger.info(
            f"Checkpoint chargé : {checkpoint_path} "
            f"(epoch={checkpoint['epoch']}, "
            f"metrics={checkpoint.get('metrics', {})})"
        )
        return checkpoint


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Construit l'optimizer à partir de la config."""
    opt_config = config["training"]["optimizer"]
    name = opt_config.get("name", "adamw")

    params = [
        {"params": [p for n, p in model.named_parameters()
                    if "bias" not in n and "norm" not in n],
         "weight_decay": opt_config.get("weight_decay", 0.05)},
        {"params": [p for n, p in model.named_parameters()
                    if "bias" in n or "norm" in n],
         "weight_decay": 0.0},  # Pas de weight decay sur les biais et norms
    ]

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=opt_config["lr"],
            betas=tuple(opt_config.get("betas", [0.9, 0.999])),
        )
    elif name == "sgd":
        return torch.optim.SGD(
            params,
            lr=opt_config["lr"],
            momentum=opt_config.get("momentum", 0.9),
        )
    raise ValueError(f"Optimizer inconnu : '{name}'")


def build_scheduler(optimizer, config: dict, num_epochs: int):
    """Construit le scheduler à partir de la config."""
    sched_config = config["training"]["scheduler"]
    name = sched_config.get("name", "cosine_warmup")

    if name == "cosine_warmup":
        warmup = sched_config.get("warmup_epochs", 5)

        def lr_lambda(epoch):
            if epoch < warmup:
                return epoch / max(1, warmup)  # Warmup linéaire
            progress = (epoch - warmup) / max(1, num_epochs - warmup)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs,
            eta_min=sched_config.get("min_lr", 1e-6),
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config.get("step_size", 10),
            gamma=sched_config.get("gamma", 0.5),
        )
    raise ValueError(f"Scheduler inconnu : '{name}'")
