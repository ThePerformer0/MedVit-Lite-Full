"""
MedViT-Lite — Métriques d'évaluation
======================================
Métriques standard pour la classification médicale multi-label.

Pourquoi ces métriques spécifiques ?
--------------------------------------
L'accuracy classique est trompeuse en médecine :
  Si 95% des images sont "Normal", un modèle qui dit toujours "Normal"
  a 95% d'accuracy → mais est complètement inutile cliniquement.

On utilise à la place :

1. AUC-ROC (Area Under the ROC Curve)
   → Mesure la capacité à classer les positifs avant les négatifs
   → Insensible au déséquilibre de classes
   → 1.0 = parfait, 0.5 = aléatoire

2. Sensitivity at Fixed Specificity (95%)
   → "Si on s'autorise 5% de faux positifs, combien de vrais positifs
      détecte-t-on ?"
   → C'est la question clinique réelle : les faux positifs (alarmes
      inutiles) ont un coût, les faux négatifs (pathologies manquées)
      ont un coût encore plus grand.

3. F1-Score (macro-averaged)
   → Moyenne harmonique précision/rappel
   → Utile pour l'équilibre global des classes

4. Average Precision (AP)
   → Aire sous la courbe Précision-Rappel
   → Plus informatif que l'AUC quand les classes sont très déséquilibrées

5. Expected Calibration Error (ECE)
   → Mesure si les probabilités sont bien calibrées
   → Un modèle qui dit "70% de confiance" doit avoir raison 70% du temps
   → Crucial pour la sécurité clinique
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    roc_curve,
)
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


def compute_auc(
    targets: np.ndarray,
    probs: np.ndarray,
    pathology_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calcule l'AUC-ROC pour chaque classe et la moyenne macro.

    Args:
        targets         : [N, num_classes]  — labels vrais (0/1)
        probs           : [N, num_classes]  — probabilités prédites [0,1]
        pathology_names : noms des classes (optionnel)

    Returns:
        dict : {'auc_Pneumonia': 0.85, ..., 'auc_mean': 0.84}
    """
    num_classes = targets.shape[1]
    results = {}
    aucs = []

    for c in range(num_classes):
        # Ignorer les classes sans positifs ou sans négatifs
        # (AUC indéfinie dans ce cas)
        if len(np.unique(targets[:, c])) < 2:
            logger.warning(
                f"Classe {c} ignorée pour AUC (une seule classe dans le batch)"
            )
            continue

        auc = roc_auc_score(targets[:, c], probs[:, c])
        aucs.append(auc)

        key = pathology_names[c] if pathology_names else f"class_{c}"
        results[f"auc_{key}"] = auc

    results["auc_mean"] = float(np.mean(aucs)) if aucs else 0.0
    return results


def compute_sensitivity_at_specificity(
    targets: np.ndarray,
    probs: np.ndarray,
    target_specificity: float = 0.95,
    pathology_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calcule la sensibilité à une spécificité fixée (95% par défaut).

    Procédure :
      1. Calculer la courbe ROC (FPR vs TPR pour chaque seuil)
      2. Trouver le seuil où FPR ≤ (1 - target_specificity)
      3. Lire la sensibilité (TPR) à ce seuil

    Note :
      FPR = 1 - Spécificité
      Si on veut Spécificité ≥ 95% → FPR ≤ 5%

    Args:
        targets            : [N, num_classes]
        probs              : [N, num_classes]
        target_specificity : spécificité cible (0.95 = 95%)
        pathology_names    : noms des classes

    Returns:
        dict : {'sens@spec95_Pneumonia': 0.72, ..., 'sens@spec95_mean': 0.68}
    """
    num_classes = targets.shape[1]
    target_fpr = 1.0 - target_specificity
    results = {}
    sensitivities = []

    for c in range(num_classes):
        if len(np.unique(targets[:, c])) < 2:
            continue

        fpr, tpr, thresholds = roc_curve(targets[:, c], probs[:, c])

        # Trouver l'index où FPR est le plus proche de target_fpr
        # (sans le dépasser, pour respecter la contrainte de spécificité)
        valid_indices = np.where(fpr <= target_fpr)[0]

        if len(valid_indices) == 0:
            sensitivity = 0.0
        else:
            # Prendre le FPR le plus élevé (mais ≤ target_fpr) → sensibilité max
            best_idx = valid_indices[np.argmax(tpr[valid_indices])]
            sensitivity = float(tpr[best_idx])

        sensitivities.append(sensitivity)
        key = pathology_names[c] if pathology_names else f"class_{c}"
        results[f"sens@spec{int(target_specificity*100)}_{key}"] = sensitivity

    spec_pct = int(target_specificity * 100)
    results[f"sens@spec{spec_pct}_mean"] = (
        float(np.mean(sensitivities)) if sensitivities else 0.0
    )
    return results


def compute_f1(
    targets: np.ndarray,
    preds: np.ndarray,
    pathology_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calcule le F1-score macro et par classe.

    Args:
        targets         : [N, num_classes]  — labels vrais (0/1)
        preds           : [N, num_classes]  — prédictions binaires (0/1)
        pathology_names : noms des classes

    Returns:
        dict : {'f1_mean': 0.71, 'f1_Pneumonia': 0.68, ...}
    """
    num_classes = targets.shape[1]
    results = {}

    # F1 macro (moyenne sur toutes les classes)
    results["f1_mean"] = float(
        f1_score(targets, preds, average="macro", zero_division=0)
    )

    # F1 par classe
    per_class_f1 = f1_score(targets, preds, average=None, zero_division=0)
    for c, f1 in enumerate(per_class_f1):
        key = pathology_names[c] if pathology_names else f"class_{c}"
        results[f"f1_{key}"] = float(f1)

    return results


def compute_average_precision(
    targets: np.ndarray,
    probs: np.ndarray,
    pathology_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calcule l'Average Precision (AP = aire sous courbe Précision-Rappel).

    Args:
        targets         : [N, num_classes]
        probs           : [N, num_classes]
        pathology_names : noms des classes

    Returns:
        dict : {'ap_mean': 0.62, 'ap_Pneumonia': 0.58, ...}
    """
    num_classes = targets.shape[1]
    results = {}
    aps = []

    for c in range(num_classes):
        if len(np.unique(targets[:, c])) < 2:
            continue

        ap = average_precision_score(targets[:, c], probs[:, c])
        aps.append(ap)

        key = pathology_names[c] if pathology_names else f"class_{c}"
        results[f"ap_{key}"] = float(ap)

    results["ap_mean"] = float(np.mean(aps)) if aps else 0.0
    return results


def compute_ece(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).

    Mesure si les probabilités sont bien calibrées :
    "Si le modèle dit 70% de confiance, est-ce qu'il a raison 70% du temps ?"

    Procédure :
      1. Grouper les prédictions en n_bins bins de confiance
      2. Pour chaque bin : calculer l'accuracy réelle vs confiance moyenne
      3. ECE = moyenne pondérée des écarts |accuracy - confiance|

    Un ECE proche de 0 indique une bonne calibration.

    Args:
        probs   : [N, num_classes]  — probabilités (post-sigmoid)
        targets : [N, num_classes]  — labels vrais
        n_bins  : nombre de bins de calibration

    Returns:
        ece : scalaire [0, 1]
    """
    # Aplatir toutes les classes ensemble
    probs_flat   = probs.flatten()
    targets_flat = targets.flatten()

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs_flat)

    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        mask = (probs_flat >= lower) & (probs_flat < upper)

        if mask.sum() == 0:
            continue

        bin_confidence = probs_flat[mask].mean()
        bin_accuracy   = targets_flat[mask].mean()
        bin_weight     = mask.sum() / total

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


class MetricsTracker:
    """
    Collecte les prédictions et labels sur tout un epoch,
    puis calcule toutes les métriques en fin d'epoch.

    Usage :
        tracker = MetricsTracker(threshold=0.5)

        for batch in dataloader:
            logits, labels = model(images), batch_labels
            tracker.update(logits, labels)

        metrics = tracker.compute()
        tracker.reset()
    """

    def __init__(
        self,
        threshold: float = 0.5,
        pathology_names: Optional[List[str]] = None,
    ):
        self.threshold = threshold
        self.pathology_names = pathology_names
        self.reset()

    def reset(self):
        """Vider les buffers (appeler en début d'epoch)."""
        self._all_probs   = []
        self._all_targets = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Ajouter les prédictions d'un batch.

        Args:
            logits  : [B, num_classes]  — logits bruts (avant sigmoid)
            targets : [B, num_classes]  — labels vrais
        """
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        self._all_probs.append(probs)
        self._all_targets.append(targets.detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """
        Calcule toutes les métriques sur les données accumulées.

        Returns:
            dict de toutes les métriques
        """
        if not self._all_probs:
            return {}

        probs   = np.concatenate(self._all_probs,   axis=0)  # [N, num_classes]
        targets = np.concatenate(self._all_targets, axis=0)  # [N, num_classes]
        preds   = (probs > self.threshold).astype(float)

        metrics = {}

        # AUC-ROC
        metrics.update(compute_auc(targets, probs, self.pathology_names))

        # Sensitivity @ Specificity 95%
        metrics.update(
            compute_sensitivity_at_specificity(
                targets, probs, 0.95, self.pathology_names
            )
        )

        # F1
        metrics.update(compute_f1(targets, preds, self.pathology_names))

        # Average Precision
        metrics.update(
            compute_average_precision(targets, probs, self.pathology_names)
        )

        # Calibration
        metrics["ece"] = compute_ece(probs, targets)

        # Convertir toutes les valeurs en float standard Python (évite les erreurs de sérialisation PyYAML)
        return {k: float(v) for k, v in metrics.items()}

    def summary(self, metrics: Dict[str, float]) -> str:
        """Résumé concis des métriques clés."""
        lines = [
            f"  AUC (mean)        : {metrics.get('auc_mean', 0):.4f}",
            f"  Sens@Spec95 (mean): {metrics.get('sens@spec95_mean', 0):.4f}",
            f"  F1 (macro)        : {metrics.get('f1_mean', 0):.4f}",
            f"  AP (mean)         : {metrics.get('ap_mean', 0):.4f}",
            f"  ECE               : {metrics.get('ece', 0):.4f}",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    from data.datasets.chest_mnist import PATHOLOGY_NAMES

    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    N, C = 500, 14  # 500 samples, 14 classes

    # Simuler des prédictions réalistes
    targets = (np.random.rand(N, C) > 0.8).astype(float)
    logits  = torch.randn(N, C)
    probs   = torch.sigmoid(logits).numpy()

    print("Test des métriques :")
    tracker = MetricsTracker(threshold=0.5, pathology_names=PATHOLOGY_NAMES)
    tracker.update(logits, torch.tensor(targets))
    metrics = tracker.compute()

    print(tracker.summary(metrics))
    print(f"\n  AUC par classe :")
    for name in PATHOLOGY_NAMES:
        auc = metrics.get(f"auc_{name}", 0)
        print(f"    {name:<25} : {auc:.3f}")
