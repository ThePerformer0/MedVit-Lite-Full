# 🩺 MedViT-Lite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A lightweight hierarchical Vision Transformer, designed and trained **from scratch**, for multi-pathology screening on chest X-rays in resource-constrained environments.

> 📄 Full technical report and detailed experimental protocol: [`paper/report.md`](paper/report.md)

---

## 🎯 What This Project Demonstrates

This is not a production-ready model or commercial research. It is a personal, exploratory comparative study between a Vision Transformer architecture designed and trained entirely from scratch, and a classic ImageNet-pretrained CNN baseline (ResNet-50).

The objective was to design, implement, and **honestly evaluate** three architectural ideas under real-world computational constraints: not to claim state-of-the-art results, but to explore, learn, understand what works and what doesn't, and lay down solid foundations for future improvements as more resources become available over time.

---

## 🏗️ Proposed Architecture & Key Modules

```text
Chest X-Ray (224×224)
        │
        ▼
CNN Patch Embedding (384d, 196 patches)
        │
        ▼
Dynamic Patch Sparsifier (DPS)        ← keeps the top 50% most informative patches
        │                                (Gumbel-Softmax scoring, 4x attention FLOP reduction)
        ▼
Selective Frame Cache (SFC)           ← reuses stable representations across frames
        │                                (designed for continuous video/streaming workflows)
        ▼
Hierarchical Temporal Attention (HTA) ← local windows (7×7) + global aggregation tokens
        │
        ▼
Classification Head + MC Dropout (10 passes)  ← epistemic uncertainty estimation
        │
        ▼
Multi-label predictions (14 pathologies) + predictive confidence bounds
```

---

## 📊 Results — Honestly Presented

Evaluated on the **ChestMNIST** benchmark (112,120 chest X-rays, 14 thoracic pathologies), compared against an ImageNet-pretrained ResNet-50:

| Model | Parameters | Macro AUC-ROC | Macro F1-Score | ECE (Calibration Error ↓) |
|---|:---:|:---:|:---:|:---:|
| **ResNet-50 (Pretrained)** | 24.0 M | **0.768** | 0.059 | 0.0124 |
| **MedViT-Lite (From Scratch)** | **11.36 M** *(−52.7%)* | 0.617 | 0.000* | **0.0078** *(−37%)* |

*\*Note on F1: Measuring F1 at a fixed hard threshold of 0.5 significantly underestimates real performance on an extreme multi-label imbalanced dataset (class prevalences as low as 0.18%); macro AUC-ROC remains the standard benchmark metric.*

### 🔍 What these numbers actually say:
- **Raw discrimination gap:** MedViT-Lite performs better than random chance (AUC 0.617 > 0.50), but **remains behind the pretrained CNN baseline in raw diagnostic power**. This is completely consistent with deep learning literature: Vision Transformers lack the hard local inductive bias of CNNs and typically require large-scale pretraining on millions of images to form robust spatial representations, whereas CNNs generalize much better from small data. I have not yet been able to run large-scale pretraining due to limited compute resources.
- **Model footprint & calibration:** On the other hand, MedViT-Lite achieves a **lower probabilistic calibration error (ECE)** with **52.7% fewer parameters**—a compact footprint that is encouraging for edge devices, even though it does not yet bridge the raw accuracy gap.
- **Zero F1 artifact:** The 0.000 F1 score is an artifact of using a fixed 0.5 decision threshold on rare disease classes. Tuning optimal decision thresholds per class (e.g., via Youden's J statistic on the computed ROC curves) is the most immediate post-processing improvement identified.

Detailed metrics and comparison charts: [`results/comparison_table.csv`](results/comparison_table.csv), [`results/auc_comparison.png`](results/auc_comparison.png).

---

## ⚠️ Known Limitations & Context

1. **Pretraining Asymmetry:** ResNet-50 benefits from ImageNet pretraining weights (1.28M natural images); MedViT-Lite was trained entirely *from scratch* on ChestMNIST—a structural comparison that strongly favors the baseline.
2. **Native Resolution:** ChestMNIST images are natively $28 \times 28$ pixels upscaled to $224 \times 224$, which removes fine texture details that are critical for tiny lesions (nodules $\le 5\text{mm}$).
3. **Fixed Decision Threshold:** F1 and sensitivity were evaluated at a default 0.5 threshold, which is sub-optimal under severe class imbalance.
4. **Hardware Constraints:** All training was performed under strict free-tier cloud GPU constraints (single Kaggle T4 sessions), which dictated the experimental scope.

---

## 🗺️ Future Roadmap & Next Steps

As compute resources, time, and tools become available, I plan to:
- Implement **self-supervised medical pretraining** (Masked Autoencoders / DINOv2) prior to fine-tuning, to bridge the transfer learning gap.
- Calibrate per-class decision thresholds (Youden's index / F1-optimal) for representative clinical operating points.
- Test on full-resolution datasets (**NIH ChestX-ray14**, **MIMIC-CXR**) at $1024 \times 1024$.
- Quantize and export (ONNX / TensorRT) for actual benchmark measurements on physical edge hardware (Jetson / Raspberry Pi).

---

## 🔬 Reproducing Experiments

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run baseline and MedViT-Lite
bash scripts/run.sh --session1

# 3. Generate comparison tables and plots
python experiments/compare_results.py
```

Interactive demonstration notebook (inference + Grad-CAM explainability): [`notebooks/01_demo_inference_and_explainability.ipynb`](notebooks/01_demo_inference_and_explainability.ipynb)

---

## 📂 Repository Structure

```text
MedVit-Lite-Full/
├── models/
│   ├── backbone/          # CNN patch embedding
│   ├── sparsifier/        # Dynamic Patch Sparsifier
│   ├── cache/             # Selective Frame Cache
│   ├── attention/         # Hierarchical Temporal Attention
│   ├── head/              # Classification head + MC Dropout
│   └── medvit_lite.py     # Full model assembly
├── training/              # Fast GPU Trainer, losses, metrics
├── explainability/        # Grad-CAM++, attention rollout
├── experiments/           # Training and evaluation scripts
├── results/               # Generated test metrics and comparison plot
├── paper/report.md        # Technical benchmark report
└── notebooks/             # Interactive demo notebook
```

---

## 📄 License

Distributed under the [MIT License](./LICENSE).

---

*Developed by [Feke Jimmy Wilson](https://github.com/ThePerformer0) — Master 2 Computer Engineering, ENSPY*