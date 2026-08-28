# MedViT-Lite 🏥✨

**A Hierarchical Adaptive Vision Transformer for Resource-Constrained Medical Image Diagnosis**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg?style=flat&logo=python)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark: ChestMNIST](https://img.shields.io/badge/Benchmark-ChestMNIST%20(112k)-green.svg)](https://medmnist.com/)
[![Made with Passion](https://img.shields.io/badge/Made%20with-Passion%20%26%20Curiosity-ff69b4.svg)](#-about-this-project--the-philosophy)

---

> 💡 **In a nutshell:** A passion-driven exploration to see if we can run a smart, well-calibrated, and ultra-lightweight medical AI model on resource-constrained edge devices (tablets, portable ultrasound or remote clinic hardware) without blowing up memory!

---

## 👋 About This Project & The Philosophy

Hey there! If you stumbled upon this repository, welcome! 🚀

This project was born out of a **genuine passion for Deep Learning and scientific curiosity**. I love understanding how modern vision architectures (especially Vision Transformers) operate under the hood, and finding creative ways to adapt them to real-world challenges—like medical diagnostic support in underserved regions where access to high-end hardware and specialized radiologists is limited.

### My Approach & Values:
* 🗣️ **Keeping it simple & fun:** Medical AI can often feel locked behind thick jargon. I like keeping explanations straightforward, intuitive, and fun so that anyone—whether you're a student, enthusiast, or veteran engineer—can follow along.
* 🔬 **Humility & Scientific Honesty:** I am not claiming to have built a "groundbreaking model that replaces doctors". This is an **exploratory research prototype**—a promising direction showing how we can prune redundant computations while improving clinical safety.
* 🛠️ **Built with resourcefulness:** Trained with care on free cloud GPUs (Kaggle T4)! If I had access to a high-end compute cluster (A100s / H100s), I would scale up large self-supervised pretraining on full-resolution scans in a heartbeat.
* 💬 **Always open to chat!** Whether you're curious, working on similar ideas, or want to share feedback: **the door is wide open!** Feel free to open a *Discussion*, an *Issue*, or reach out to connect!

---

## 🎯 In Plain English: What Problem Are We Tackling?

Picture a rural medical dispensary with a portable X-ray machine or ultrasound probe, but no radiologist within hundreds of miles. 
Standard heavy AI models (like deep CNNs or giant ViTs) require massive memory, run slowly on low-power chips, and suffer from a dangerous flaw: **they are often overconfident even when they are wrong**.

**MedViT-Lite's core recipe:**
1. **Prune the background noise (Dynamic Patch Sparsifier - DPS):** On a chest X-ray, up to 50% of the image is just empty dark background. Why waste energy computing attention on it? We keep only the informative anatomical patches.
2. **Don't compute the same thing twice (Selective Frame Cache - SFC):** In medical video or continuous imaging, consecutive frames share massive visual overlap. We cache and reuse stationary features.
3. **Know when to say "I don't know" (Monte Carlo Uncertainty):** If a scan is ambiguous or noisy, the model provides an uncertainty score ($\pm \sigma$) to refer the case to a human specialist rather than guessing wildly.

---

## 🏆 Experimental Results (On 112,120 Chest X-Rays)

Here are the real test results benchmarked on **ChestMNIST** (**22,433 independent test scans**, 14 thoracic pathologies):

| Model Architecture | Parameter Count | Macro AUC-ROC | Sensitivity @ 95% Spec | Expected Calibration Error (ECE ↓) |
|:---|:---:|:---:|:---:|:---:|
| **ResNet-50 (Standard CNN, ImageNet Pretrained)** | 24.0 Million | **0.7678** | **0.2732** | 0.0124 |
| **MedViT-Lite (Our Transformer, Trained From Scratch)** | **11.36 Million** *(−52.7%)* | **0.6174** | **0.1033** | **0.0078** *(−37.1% error!)* 🥇 |

![AUC Comparison](results/auc_comparison.png)

### 💡 What do these numbers actually tell us?

1. **Over 50% lighter (11.36M vs 24.0M parameters):** MedViT-Lite cuts the parameter footprint in half, significantly lowering VRAM and compute requirements.
2. **Superior Probabilistic Calibration (ECE) 🥇:** Achieving an **ECE of 0.0078** (vs 0.0124 for ResNet-50) means **37.1% less overconfident error**. Its predicted probabilities reflect true diagnostic risk much more reliably.
3. **Strong baseline on focal findings (Without Pretraining):** Even without any ImageNet pretraining, MedViT-Lite scores **0.7967 AUC on Edema**, **0.6954 on Cardiomegaly**, and **0.6895 on Lung Consolidation**.

---

## 🏗️ Architecture at a Glance

```
                         Input Chest X-Ray (224×224)
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │   CNN Patch Embedding (196 Patches × 384-dim)   │
             └────────────────────────┬────────────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │  ✂️ Dynamic Patch Sparsifier (DPS)               │  ◄── Innovation 1
             │  Prunes background noise, keeps top 50% tokens  │      (4x FLOP reduction)
             └────────────────────────┬────────────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │  💾 Selective Frame Cache (SFC)                 │  ◄── Innovation 2
             │  Caches stationary features for stream reuse    │      (Ideal for video/echo)
             └────────────────────────┬────────────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │  🧠 Hierarchical Temporal Attention (HTA)       │  ◄── Innovation 3
             │  Local window details + global contextual tokens│
             └────────────────────────┬────────────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │  🎯 Classification Head + MC Dropout (10x)      │  ◄── Clinical Safety
             │  Predicts 14 pathologies with uncertainty bounds│
             └────────────────────────┬────────────────────────┘
                                      │
                                      ▼
                        Diagnostic Saliency Map (Grad-CAM++)
```

---

## 🔍 Visual Explainability & Clinical Trust (Grad-CAM++)

For clinicians to trust an AI system, it must show *where* it is looking:
- **Grad-CAM++**: Highlights fine pathological regions (e.g., cardiomegaly enlargement, pleural effusion opacities).
- **Monte Carlo Dropout**: Generates an error bar ($\pm \sigma$) across all 14 multi-label findings.

```bash
# Generate a complete multi-panel visual explanation report
python explainability/visualize.py --checkpoint checkpoints/best_medvit_lite.pth --sample-index 0
```

---

## 📁 Repository Structure

```
Med-Vit-Lite/
├── README.md                                      # 🌟 Main project overview & philosophy
├── paper/
│   └── report.md                                  # 📄 Full academic & technical report
├── explainability/
│   ├── gradcam.py                                 # 🔍 Grad-CAM++ engine for CNN & ViT
│   ├── attention_rollout.py                       # 🎯 Transformer attention rollout tracker
│   └── visualize.py                               # 📊 CLI clinical visualizer
├── notebooks/
│   └── 01_demo_inference_and_explainability.ipynb # 📓 Step-by-step interactive demo notebook
├── results/
│   ├── comparison_table.csv                       # 📊 Benchmark summary table (CSV)
│   ├── auc_comparison.png                         # 📈 Official benchmark comparison plot
│   ├── baseline_resnet50_test_results.yaml        # 📑 Detailed per-class metrics (ResNet-50)
│   └── medvit_lite_test_results.yaml              # 📑 Detailed per-class metrics (MedViT-Lite)
├── models/                                        # 🧠 PyTorch modules (DPS, SFC, HTA, MedViT)
├── training/                                      # ⚙️ Fast GPU trainer, AMP, Early Stopping
└── configs/base.yaml                              # 🛠️ Master configuration file
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/ThePerformer0/MedVit-Lite-Full.git
cd MedVit-Lite-Full
pip install -r requirements.txt
```

### 2. Quick Dry-Run (30-second verification)

```bash
bash scripts/run.sh --session1 --dry-run
```

### 3. Full Training & Benchmark Execution

```bash
# Run full benchmark (ResNet-50 + MedViT-Lite)
bash scripts/run.sh --session1

# Generate summary table and plots
python experiments/compare_results.py --results-dir results
```

---

## ⚠️ Limitations & Future Roadmap

1. **Pretraining Gap:** ResNet-50 was pretrained on 1.28M natural images (ImageNet-1k), giving it strong visual priors. MedViT-Lite was trained from scratch. Pretraining the Transformer with medical self-supervised objectives (**DINOv2 / Masked Autoencoders**) is the natural next step.
2. **Native Resolution:** ChestMNIST uses $28 \times 28$ native resolution. Testing on full-resolution ($1024 \times 1024$) **NIH ChestX-ray14** or **MIMIC-CXR** will help capture tiny micronodules ($\le 5\text{mm}$).
3. **Edge Compilation:** Exporting the pipeline to **TensorRT-FP16** and **ONNX Runtime** for sub-10ms inference on embedded platforms like NVIDIA Jetson Orin.

---

## 🏷️ GitHub Repository Metadata

* **Repository Description:**
  > 🏥 A lightweight, well-calibrated Vision Transformer for multi-label chest pathology screening on edge devices, built with passion & curiosity. Featuring Dynamic Patch Sparsification (DPS) & Monte Carlo uncertainty.

* **Topics / Tags:**
  `deep-learning` • `vision-transformer` • `pytorch` • `medical-imaging` • `chestmnist` • `edge-ai` • `explainable-ai` • `grad-cam` • `uncertainty-estimation` • `sparse-attention` • `healthcare`

---

## 🤝 Let's Connect & Chat!

Have ideas, questions about the architecture, suggestions, or just want to talk about Deep Learning and computer vision?
* Feel free to open an **Issue** or a **Discussion** on this repository!
* I'm always happy to connect, exchange thoughts, and learn together.

---

## 📜 Citation

```bibtex
@article{medvitlite2026,
  title={MedViT-Lite: A Hierarchical Adaptive Vision Transformer for Resource-Constrained Medical Image Diagnosis},
  author={ThePerformer0 and Contributors},
  journal={GitHub Repository: ThePerformer0/MedVit-Lite-Full},
  year={2026}
}
```

---

## ⚖️ Disclaimer

*MedViT-Lite is an exploratory research prototype developed for scientific inquiry. It has **not** been certified for clinical diagnosis and must **not** be used as a substitute for professional medical judgment.*
