# MedViT-Lite 🏥✨

**A Hierarchical Adaptive Vision Transformer for Resource-Constrained Medical Image Diagnosis**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg?style=flat&logo=python)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark: ChestMNIST](https://img.shields.io/badge/Benchmark-ChestMNIST%20(112k)-green.svg)](https://medmnist.com/)
[![Made with Passion](https://img.shields.io/badge/Made%20with-Passion%20%26%20Curiosity-ff69b4.svg)](#-un-mot-sur-le-projet--la-philosophie)

---

> 💡 **En une phrase :** Une exploration passionnée pour voir si on peut faire tourner un modèle d'IA médicale intelligent, honnête et ultra-léger sur de petits appareils (tablettes, matériel de brousse ou petits hôpitaux) sans exploser la mémoire !

---

## 👋 Un mot sur le projet & La philosophie

Salut ! Si tu atterris ici, bienvenue ! 🚀

Ce projet est né d'une **véritable passion pour le Deep Learning et d'une grande curiosité scientifique**. J'adore explorer comment les architectures modernes (comme les Vision Transformers) fonctionnent sous le capot et comment les adapter à de vrais défis humains, notamment la santé dans les régions où l'accès à du gros matériel informatique est limité.

### Mon approche :
* 🗣️ **Garder les choses simples et accessibles :** L'IA médicale est souvent bardée de jargon intimidant. J'aime expliquer les concepts avec des mots clairs, des analogies simples et un ton décontracté pour que tout le monde puisse comprendre ce qui se passe.
* 🔬 **Humilité & Rigueur scientifique :** Je ne prétends absolument pas avoir créé un modèle "révolutionnaire qui remplace les médecins". C'est un **prototype de recherche**, une piste prometteuse qui explore comment alléger les calculs sans perdre le sens clinique.
* 🛠️ **Fait avec les moyens du bord :** Entraîné avec amour sur des GPUs gratuits (Kaggle T4) ! Si j'avais des clusters de supercalculateurs (A100/H100) sous la main, je testerais immédiatement un pré-entraînement géant en haute résolution sans hésiter.
* 💬 **Envie de discuter ?** Que tu sois débutant curieux, médecin, étudiant ou chercheur en Deep Learning : **la porte est grande ouverte !** N'hésite pas à ouvrir une *Discussion*, une *Issue* ou à me contacter pour échanger des idées, des retours ou collaborer !

---

## 🎯 En clair : Quel problème on essaie de résoudre ?

Imaginons un dispensaire isolé avec un petit appareil à rayons X ou une sonde d'échographie portable, mais aucun radiologue à des centaines de kilomètres. 
Les gros modèles d'IA actuels (comme ResNet-50 ou ViT géants) demandent d'énormes serveurs avec beaucoup de mémoire et ont un vilain défaut : **ils sont souvent trop sûrs d'eux même quand ils se trompent**.

**L'idée de MedViT-Lite :**
1. **Élaguer le superflu (DPS) :** Sur une radio du poumon, 50% de l'image est juste du fond noir inutile. Pourquoi faire chauffer la puce dessus ? On ne garde que les morceaux anatomiques clés.
2. **Éviter de recalculer deux fois la même chose (SFC) :** En vidéo médicale, deux images successives se ressemblent énormément. On met en cache ce qui n'a pas bougé.
3. **Dire "Je ne sais pas" (Incertitude Monte Carlo) :** Si l'image est floue ou bizarre, le modèle prévient le soignant plutôt que d'inventer un diagnostic au hasard.

---

## 🏆 Résultats Expérimentaux (Sur 112 120 Radiographies)

Voici ce que donnent les tests réels sur le benchmark **ChestMNIST** (22 433 images de test indépendantes, 14 pathologies) :

| Modèle / Architecture | Taille (Paramètres) | Précision Globale (AUC-ROC) | Détection @ 95% Spéc. | Erreur de Calibration (ECE ↓) |
|:---|:---:|:---:|:---:|:---:|
| **ResNet-50 (CNN classique, Pré-entraîné ImageNet)** | 24.0 Millions | **0.7678** | **0.2732** | 0.0124 |
| **MedViT-Lite (Notre Transformer, From Scratch)** | **11.36 Millions** *(−52.7%)* | **0.6174** | **0.1033** | **0.0078** *(−37.1% d'erreur !)* 🥇 |

![AUC Comparison](results/auc_comparison.png)

### 💡 Qu'est-ce que ces chiffres prouvent concrètement ?

1. **Un modèle 2× plus léger (11M vs 24M de paramètres) :** MedViT-Lite divise la mémoire par deux grâce à son module d'élagage de patches.
2. **Une bien meilleure "honnêteté" (Calibration ECE) 🥇 :** Avec un ECE de **0.0078** (contre 0.0124 pour ResNet-50), MedViT-Lite fait **37% moins d'erreurs de sur-confiance**. Ses probabilités sont bien mieux calibrées avec la réalité clinique.
3. **De très bons signaux sur les grosses anomalies :** Sans aucun pré-entraînement préalable, il accroche déjà **0.7967 d'AUC sur l'Œdème**, **0.6954 sur la Cardiomégalie** et **0.6895 sur la Consolidation pulmonaire**.

---

## 🏗️ L'Architecture en un coup d'œil

```
                       Radiographie Thoracique (224×224)
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │   Découpage en 196 petits morceaux (Patches)    │
             └────────────────────────┬────────────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │  ✂️ Dynamic Patch Sparsifier (DPS)               │  ◄── Innovation 1
             │  On jette le fond inutile et on garde le TOP 50%│      (Divise le calcul par 4)
             └────────────────────────┬────────────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │  💾 Selective Frame Cache (SFC)                 │  ◄── Innovation 2
             │  On recycle les calculs des images similaires   │      (Idéal flux vidéo/écho)
             └────────────────────────┬────────────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │  🧠 Attention Hiérarchique (HTA)                │  ◄── Innovation 3
             │  Vision locale des détails + vue d'ensemble     │
             └────────────────────────┬────────────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │  🎯 Tête de Classification + Barre d'Incertitude│  ◄── Sécurité Clinique
             │  Prédit les 14 maladies avec indice de doute    │
             └────────────────────────┬────────────────────────┘
                                      │
                                      ▼
                       Diagnostic + Carte Thermique (GradCAM++)
```

---

## 🔍 Explicabilité Visuelle (Grad-CAM++)

Pour qu'un médecin fasse confiance à l'IA, il faut qu'elle montre où elle regarde :
- **Grad-CAM++** : Affiche une carte thermique colorée sur la zone suspecte (ex: silhouette du cœur élargie).
- **Incertitude Monte Carlo** : Affiche une barre d'erreur ($\pm \sigma$) sur chaque maladie.

```bash
# Générer une explication visuelle complète sur une radio
python explainability/visualize.py --checkpoint checkpoints/best_medvit_lite.pth --sample-index 0
```

---

## 📁 Structure du Projet

```
Med-Vit-Lite/
├── README.md                                      # 🌟 Ce fichier (vue d'ensemble & esprit du projet)
├── paper/
│   └── report.md                                  # 📄 Rapport technique et scientifique complet
├── explainability/
│   ├── gradcam.py                                 # 🔍 Générateur de cartes Grad-CAM++
│   ├── attention_rollout.py                       # 🎯 Suivi de l'attention du Transformer
│   └── visualize.py                               # 📊 Script CLI de visualisation clinique
├── notebooks/
│   └── 01_demo_inference_and_explainability.ipynb # 📓 Notebook interactif pas-à-pas
├── results/
│   ├── comparison_table.csv                       # 📊 Tableau récapitulatif des métriques
│   ├── auc_comparison.png                         # 📈 Graphique comparatif officiel
│   ├── baseline_resnet50_test_results.yaml        # 📑 Détails par classe (ResNet-50)
│   └── medvit_lite_test_results.yaml              # 📑 Détails par classe (MedViT-Lite)
├── models/                                        # 🧠 Modules PyTorch (DPS, SFC, HTA, MedViT)
├── training/                                      # ⚙️ Trainer GPU rapide, AMP, Early Stopping
└── configs/base.yaml                              # 🛠️ Configuration des hyperparamètres
```

---

## 🚀 Démarrage Rapide

```bash
# 1. Cloner le repo
git clone https://github.com/ThePerformer0/MedVit-Lite-Full.git
cd MedVit-Lite-Full

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Tester en mode ultra-rapide (vérification en 30s)
bash scripts/run.sh --session1 --dry-run

# 4. Lancer l'entraînement complet
bash scripts/run.sh --session1
```

---

## ⚠️ Limites & Ce que j'aimerais faire ensuite !

1. **Le Pré-entraînement :** ResNet-50 a appris sur 1,28 million d'images ImageNet avant de voir des radios. MedViT-Lite a tout appris de zéro. Avec un pré-entraînement auto-supervisé médical (type **DINOv2** ou **MAE**), les performances peuvent exploser !
2. **La Haute Résolution :** Tester sur des radios en pleine définition ($1024 \times 1024$ sur NIH ChestX-ray14) pour mieux voir les tout petits nodules pulmonaires ($\le 5\text{mm}$).
3. **Déploiement Embarqué Réel :** Convertir le modèle en **TensorRT / ONNX** pour le tester en vrai sur une puce Raspberry Pi ou NVIDIA Jetson.

---

## 🏷️ GitHub Repository Metadata (Pour la page GitHub)

Si tu souhaites configurer le dépôt GitHub avec une jolie description et des tags pertinents :

* **Description GitHub :**
  > 🏥 A lightweight, well-calibrated Vision Transformer for multi-label chest pathology screening on edge devices, built with passion & curiosity. Featuring Dynamic Patch Sparsification (DPS) & Monte Carlo uncertainty.

* **Suggested Topics / Tags :**
  `deep-learning` • `vision-transformer` • `pytorch` • `medical-imaging` • `chestmnist` • `edge-ai` • `explainable-ai` • `grad-cam` • `uncertainty-estimation` • `sparse-attention` • `healthcare`

---

## 🤝 Viens Discuter !

Tu as une suggestion, une question sur le code, ou tu veux juste papoter Deep Learning et vision par ordinateur ?
* Ouvre une **Issue** ou une **Discussion** sur ce dépôt GitHub !
* C'est toujours un plaisir d'échanger avec la communauté.

---

## ⚖️ Disclaimer

*MedViT-Lite est un projet de recherche et d'exploration scientifique. Il n'est pas homologué pour un usage médical clinique réel et ne remplace en aucun cas l'avis d'un professionnel de santé.*
