# MedViT-Lite 🏥

**A Hierarchical Adaptive Transformer for Streaming Medical Diagnosis on Edge Devices**

> *Work in progress*

---

## Motivation

In many resource-constrained regions, access to medical imaging specialists is severely limited. A clinician may have an ultrasound or X-ray machine but no radiologist nearby to interpret results.

**MedViT-Lite** explores whether a lightweight Vision Transformer can:
- Detect pulmonary pathologies from chest X-rays
- Run on a tablet or low-power edge device
- Explain its decisions visually to clinicians
- Know when it is uncertain and defer to a human expert

The goal is not clinical deployment but **rigorous scientific evaluation** of architectural innovations for efficient medical AI.

---

## Proposed Architecture

```
Input Image (224×224)
        │
        ▼
┌─────────────────────────────────┐
│   CNN Patch Encoder             │  Patch embedding backbone
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Dynamic Patch Sparsifier (DPS) │  ← Innovation 1
│  Score each patch → keep top K% │  Reduces attention cost
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Selective Frame Cache (SFC)    │  ← Innovation 2 (video/streaming)
│  Reuse features for similar     │  Reduces redundant computation
│  frames in medical video        │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Hierarchical Temporal Attention│  Local (intra-frame) + Global (inter-frame)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Classification Head            │  Multi-label prediction
│  + Uncertainty Estimator        │  Monte Carlo Dropout
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  GradCAM++ Explainability       │  Visual explanation for clinicians
└─────────────────────────────────┘
```

---

## Key Design Choices

| Component | Motivation |
|---|---|
| **Dynamic Patch Sparsification** | Most patches in a medical image are uninformative background. Only process what matters. |
| **Selective Frame Caching** | Consecutive frames in medical video are often redundant. Avoid recomputing identical features. |
| **Hierarchical Temporal Attention** | Capture both local (within a frame) and global (across frames) structure. |
| **Monte Carlo Uncertainty** | A model that cannot express uncertainty is unsafe in a clinical context. |
| **GradCAM++ Explainability** | Clinicians must understand *why* a prediction was made, not just *what* it is. |

---

## Dataset

**ChestMNIST** (prototyping) → **NIH Chest X-Ray** (full evaluation)

- 112,120 frontal chest X-rays from NIH Clinical Center
- 14 pathology labels (multi-label classification)
- Publicly available

---

## Experimental Plan

We will compare MedViT-Lite against the following baselines:

- **ResNet-50** — standard CNN reference
- **ViT-B/16** — vanilla Vision Transformer (no medical adaptation)
- **TimeSformer** — video Transformer reference (Bertasius et al., 2021)

We will also run an **ablation study** removing each innovation one at a time to measure its individual contribution.

*Results will be added here once experiments are complete.*

---

## Evaluation Metrics

- **AUC-ROC** — standard medical AI benchmark
- **Sensitivity @ Specificity = 95%** — clinically relevant threshold
- **GFLOPs** — computational cost (edge deployment feasibility)
- **Uncertainty Calibration (ECE)** — safety of confidence scores

---

## Repository Structure

```
Med-Vit-Lite/
├── configs/               # Hyperparameter configurations (YAML)
├── data/
│   ├── datasets/          # ChestMNIST and NIH Chest X-Ray loaders
│   └── transforms/        # Medical augmentation pipeline
├── models/
│   ├── backbone/          # CNN encoder (patch embedding)
│   ├── sparsifier/        # Dynamic Patch Sparsifier  [Innovation 1]
│   ├── cache/             # Selective Frame Cache      [Innovation 2]
│   ├── attention/         # Hierarchical Temporal Attention
│   ├── head/              # Classification + Uncertainty head
│   └── medvit_lite.py     # Full model assembly
├── training/              # Trainer, losses, metrics
├── explainability/        # GradCAM++, Attention Rollout
├── experiments/           # Training scripts
└── scripts/               # CloudLab setup, training launchers
```

---

## Setup

```bash
# Clone
git clone https://github.com/ThePerformer0/MedVit-Lite-Full.git
cd MedVit-Lite-Full

# Install dependencies
pip install -r requirements.txt

# CloudLab setup (c4130, 4× V100)
bash scripts/setup_cloudlab.sh
```

---

## Status

| Component | Status |
|---|---|
| Project structure | ✅ Done |
| Data pipeline (ChestMNIST) | ✅ Done |
| Dynamic Patch Sparsifier | ✅ Done |
| Selective Frame Cache | ✅ Done |
| Hierarchical Temporal Attention | 🔄 In progress |
| Classification Head + Uncertainty | 🔄 In progress |
| Full model assembly | ⏳ Pending |
| Baseline experiments | ⏳ Pending |
| MedViT-Lite training | ⏳ Pending |
| Ablation study | ⏳ Pending |
| GradCAM++ explainability | ⏳ Pending |
| Results & analysis | ⏳ Pending |

---

## Disclaimer

This is a research prototype. It has **not** been validated for clinical use and must **not** be used for actual medical diagnosis. Always consult a qualified medical professional.

---
