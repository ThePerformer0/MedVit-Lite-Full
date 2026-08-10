# MedViT-Lite 🏥

**A Hierarchical Adaptive Transformer for Streaming Medical Diagnosis on Edge Devices**

> *"Can a small, efficient model diagnose diseases as well as a large one — and explain why?"*

---

## Motivation

In sub-Saharan Africa and other resource-constrained regions, access to medical imaging specialists is severely limited. A general practitioner in a rural clinic may have an ultrasound machine but no radiologist nearby to interpret the results.

**MedViT-Lite** is a research prototype exploring whether a lightweight Vision Transformer can:
- Detect 14 pulmonary pathologies from chest X-rays
- Run on a tablet or low-power edge device
- Explain its decisions visually to clinicians
- Know when it is uncertain and should defer to a human expert

This is a final-year Master's research project. The goal is not clinical deployment but **rigorous scientific evaluation** of architectural innovations for efficient medical AI.

---

## Architecture Overview

```
Input Image (224×224)
        │
        ▼
┌─────────────────────────────────┐
│   CNN Patch Encoder             │  EfficientNet-lite backbone
│   → 196 patch tokens [B,196,D]  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Dynamic Patch Sparsifier (DPS) │  ← INNOVATION 1
│  Score each patch → keep top 50%│  98 tokens (vs 196)
│  4× less attention computation  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Selective Frame Cache (SFC)    │  ← INNOVATION 2 (video/streaming)
│  Reuse features for similar     │  ~70% cache hit rate
│  frames → 3× faster inference   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Hierarchical Temporal Attention│  Local (intra-frame) + Global (inter-frame)
│  Local  : what is in this frame │
│  Global : how does it evolve?   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Classification Head            │  Multi-label (14 pathologies)
│  + Uncertainty Estimator        │  Monte Carlo Dropout (N=10 passes)
└──────────────┬──────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
  Predictions    Uncertainty
  "Pneumonia     "Confidence: 94%
   Effusion"      ⚠️ Atelectasis: uncertain"
        │
        ▼
┌─────────────────────────────────┐
│  GradCAM++ Explainability       │  Heatmap: WHY this prediction?
└─────────────────────────────────┘
```

---

## Key Innovations

| Innovation | Problem Solved | Expected Gain |
|---|---|---|
| **Dynamic Patch Sparsification** | Transformers compute attention over ALL patches (even background) | 4× fewer attention operations at 50% keep-ratio |
| **Selective Frame Caching** | Video processing recomputes identical frames | ~70% cache hit rate on medical video |
| **Hierarchical Temporal Attention** | Single-scale attention misses multi-scale temporal patterns | Captures both frame-level and sequence-level information |
| **Monte Carlo Uncertainty** | Overconfident predictions are dangerous in medicine | Quantified uncertainty → reject ambiguous cases |

---

## Experimental Protocol

We compare MedViT-Lite against three baselines on the NIH Chest X-Ray dataset:

| Model | Type | Parameters | GFLOPs |
|---|---|---|---|
| ResNet-50 | CNN | 25M | 4.1 |
| ViT-B/16 | Transformer | 86M | 16.8 |
| TimeSformer | Video Transformer | 121M | 196 |
| **MedViT-Lite** | **Ours** | **~35M** | **~3.2*** |

*\* estimated with DPS at 50% keep-ratio*

### Ablation Study

To prove each component contributes, we run ablations:

| Config | DPS | SFC | HTA | AUC | GFLOPs |
|---|---|---|---|---|---|
| Full MedViT-Lite | ✅ | ✅ | ✅ | TBD | TBD |
| w/o DPS | ❌ | ✅ | ✅ | TBD | TBD |
| w/o SFC | ✅ | ❌ | ✅ | TBD | TBD |
| w/o HTA | ✅ | ✅ | ❌ | TBD | TBD |
| Baseline ViT | ❌ | ❌ | ❌ | TBD | TBD |

---

## Dataset

**ChestMNIST** (prototyping) → **NIH Chest X-Ray** (evaluation)

- 112,120 frontal chest X-rays
- 14 pathology labels (multi-label)
- Public domain (NIH Clinical Center)

```python
# Download automatically via MedMNIST
from medmnist import ChestMNIST
dataset = ChestMNIST(split='train', download=True)
```

---

## Setup & Reproduction

### Requirements
- Python 3.10+
- PyTorch 2.1+ with CUDA 11.8
- 4× NVIDIA V100 16GB (or equivalent)

### Installation

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/Med-Vit-Lite.git
cd Med-Vit-Lite

# Environment (CloudLab)
bash scripts/setup_cloudlab.sh

# Or local
pip install -r requirements.txt
```

### Training

```bash
# Step 1: Train baselines
python experiments/baseline_cnn.py --config configs/baseline_cnn.yaml

# Step 2: Train MedViT-Lite
python experiments/medvit_lite_train.py --config configs/base.yaml

# Step 3: Run ablations
bash scripts/run_ablations.sh
```

### Evaluation & Explainability

```bash
python experiments/evaluate.py --checkpoint checkpoints/best_medvit_lite.pth
```

---

## Repository Structure

```
Med-Vit-Lite/
├── configs/               # Hyperparameter configurations (YAML)
├── data/
│   ├── datasets/          # ChestMNIST, NIH Chest X-Ray loaders
│   └── transforms/        # Medical augmentation pipeline
├── models/
│   ├── backbone/          # CNN encoder (patch embedding)
│   ├── sparsifier/        # Dynamic Patch Sparsifier [INNOVATION 1]
│   ├── cache/             # Selective Frame Cache    [INNOVATION 2]
│   ├── attention/         # Hierarchical Temporal Attention
│   ├── head/              # Classification + Uncertainty head
│   └── medvit_lite.py     # Full model assembly
├── training/              # Trainer, losses, metrics
├── explainability/        # GradCAM++, Attention Rollout
├── experiments/           # Training scripts (baselines + MedViT-Lite)
├── scripts/               # CloudLab setup, training launchers
└── paper/                 # LaTeX draft
```

---

## Evaluation Metrics

| Metric | Description | Clinical Relevance |
|---|---|---|
| **AUC-ROC** | Area Under ROC Curve | Standard medical AI benchmark |
| **Sensitivity @ Specificity=95%** | True positive rate with controlled false positives | Clinically meaningful threshold |
| **F1 Score (macro)** | Harmonic mean of precision and recall | Balanced performance on imbalanced labels |
| **GFLOPs** | Computational cost | Edge deployment feasibility |
| **Uncertainty Calibration** | ECE (Expected Calibration Error) | Safety of confidence scores |

---

## Research Questions

1. **Does temporal memory improve diagnosis?** → Compare with/without SFC on video data
2. **Can smaller models outperform large ones?** → MedViT-Lite vs TimeSformer
3. **How explainable is the model to clinicians?** → GradCAM++ faithfulness evaluation

---

## Citation

If you use this work, please cite:

```bibtex
@misc{medvitlite2026,
  title   = {MedViT-Lite: A Hierarchical Adaptive Transformer for Streaming Medical Diagnosis},
  author  = {[Author Name]},
  year    = {2026},
  note    = {Master's Capstone Project, ENSPY}
}
```

---

## Disclaimer

This is a research prototype. It has **not** been validated for clinical use and should **not** be used for actual medical diagnosis. Always consult a qualified medical professional.

---

*Built as part of the ENSPY Final-Year Capstone Project on Generative and Agentic AI Architectures.*
