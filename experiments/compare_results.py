"""
MedViT-Lite — Comparaison et visualisation des résultats
=========================================================
Agrège les résultats de toutes les expériences et génère :
  1. Le tableau de comparaison (console + CSV)
  2. Le graphique de comparaison des AUC
  3. Les courbes ROC par classe

Utilisation :
  python experiments/compare_results.py
  python experiments/compare_results.py --results-dir ./results
"""

import argparse
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Noms des expériences dans l'ordre d'affichage
EXPERIMENTS = {
    "baseline_resnet50":           "ResNet-50 (CNN baseline)",
    "medvit_lite_noInnovations":   "MedViT-Lite w/o DPS+SFC (ViT baseline)",
    "medvit_lite_noDPS":           "MedViT-Lite w/o DPS",
    "medvit_lite_noSFC":           "MedViT-Lite w/o SFC",
    "medvit_lite":                 "MedViT-Lite (FULL — ours)",
}

COLORS = ["#6B7280", "#93C5FD", "#60A5FA", "#3B82F6", "#1D4ED8"]

METRICS_TO_SHOW = [
    ("auc_mean",           "AUC-ROC (mean)"),
    ("sens@spec95_mean",   "Sensitivity @ Spec 95% (mean)"),
    ("f1_mean",            "F1-Score (macro)"),
    ("ap_mean",            "Average Precision (mean)"),
    ("ece",                "ECE (Calibration ↓)"),
]


def load_results(results_dir: str) -> dict:
    """Charge les fichiers YAML de résultats de chaque expérience."""
    results = {}
    for exp_id, exp_name in EXPERIMENTS.items():
        path = os.path.join(results_dir, f"{exp_id}_test_results.yaml")
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    results[exp_id] = yaml.safe_load(f)
                except Exception:
                    f.seek(0)
                    try:
                        results[exp_id] = yaml.load(f, Loader=yaml.FullLoader)
                    except Exception:
                        results[exp_id] = None
            print(f"  ✅ {exp_name}")
        else:
            print(f"  ❌ {exp_name} — résultats non trouvés ({path})")
            results[exp_id] = None
    return results


def print_comparison_table(results: dict):
    """Affiche le tableau de comparaison dans le terminal."""
    header = f"{'Modèle':<45}"
    for _, metric_name in METRICS_TO_SHOW:
        header += f" {metric_name[:12]:>14}"
    print("\n" + "=" * (45 + 14 * len(METRICS_TO_SHOW)))
    print(header)
    print("=" * (45 + 14 * len(METRICS_TO_SHOW)))

    for exp_id, exp_name in EXPERIMENTS.items():
        if results[exp_id] is None:
            na_cells = "".join([f" {'N/A':>14}" for _ in METRICS_TO_SHOW])
            print(f"{exp_name:<45}{na_cells}")
            continue

        row = f"{exp_name:<45}"
        for metric_key, _ in METRICS_TO_SHOW:
            val = results[exp_id].get(metric_key, None)
            if val is None:
                row += f" {'—':>14}"
            else:
                row += f" {val:>14.4f}"
        print(row)

    print("=" * (45 + 14 * len(METRICS_TO_SHOW)))


def save_csv(results: dict, output_path: str):
    """Sauvegarde le tableau en CSV."""
    import csv
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        # En-tête
        header = ["Model"] + [name for _, name in METRICS_TO_SHOW]
        writer.writerow(header)

        for exp_id, exp_name in EXPERIMENTS.items():
            if results[exp_id] is None:
                continue
            row = [exp_name]
            for metric_key, _ in METRICS_TO_SHOW:
                val = results[exp_id].get(metric_key, "")
                row.append(f"{val:.4f}" if isinstance(val, float) else "")
            writer.writerow(row)

    print(f"\nTableau CSV sauvegardé : {output_path}")


def plot_auc_comparison(results: dict, output_path: str):
    """Graphique en barres de comparaison des AUC."""
    models = []
    aucs   = []

    for exp_id, exp_name in EXPERIMENTS.items():
        if results[exp_id] is None:
            continue
        auc = results[exp_id].get("auc_mean")
        if auc is not None:
            models.append(exp_name.replace(" — ours", "\n★ (ours)"))
            aucs.append(auc)

    if not models:
        print("Pas de données AUC disponibles pour le graphique.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    colors  = COLORS[:len(models)]
    bars    = ax.bar(models, aucs, color=colors, width=0.6, edgecolor="white",
                     linewidth=1.5)

    # Annotation des valeurs
    for bar, auc in zip(bars, aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{auc:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    # Ligne de référence (meilleur modèle)
    if aucs:
        best = max(aucs)
        ax.axhline(best, color="red", linestyle="--", alpha=0.4, linewidth=1)

    ax.set_ylabel("AUC-ROC (mean over 14 classes)", fontsize=12)
    ax.set_title("Model Comparison — AUC-ROC on ChestMNIST Test Set",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(min(aucs) * 0.97 if aucs else 0.7, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graphique AUC sauvegardé : {output_path}")


def main(args):
    results_dir = args.results_dir
    print(f"\nChargement des résultats depuis : {results_dir}")

    results = load_results(results_dir)

    available = {k: v for k, v in results.items() if v is not None}
    if not available:
        print("\nAucun résultat disponible. Lance d'abord les expériences :")
        print("  bash scripts/run_experiments.sh")
        return

    print("\n" + "=" * 70)
    print("  TABLEAU DE COMPARAISON — MedViT-Lite vs Baselines")
    print("=" * 70)
    print_comparison_table(results)

    # CSV
    csv_path = os.path.join(results_dir, "comparison_table.csv")
    save_csv(results, csv_path)

    # Graphique AUC
    plot_path = os.path.join(results_dir, "auc_comparison.png")
    plot_auc_comparison(results, plot_path)

    print("\nAnalyse terminée.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="./results")
    args = parser.parse_args()
    main(args)
