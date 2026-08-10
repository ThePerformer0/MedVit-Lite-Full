#!/bin/bash
# ============================================================
# MedViT-Lite — Setup CloudLab
# ============================================================

set -e  # Arrêter immédiatement en cas d'erreur

echo "=============================================="
echo "  MedViT-Lite — Setup CloudLab"
echo "=============================================="

# ── 1. Mise à jour système ──────────────────────────────────
echo -e "\n[1/6] Mise à jour des paquets système..."
sudo apt-get update -qq
sudo apt-get install -y -qq git wget curl build-essential libopencv-dev python3-dev python3-pip htop tree

# ── 2. Installation Miniconda ────────────────────────────────
echo -e "\n[3/6] Installation Miniconda..."
if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
fi
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# ── 3. Création environnement conda ──────────────────────────
echo -e "\n[4/6] Création environnement 'medvit'..."
conda create -n medvit python=3.10 -y
conda activate medvit

# ── 4. Installation PyTorch ──────────────────────────────────
echo -e "\n[5/6] Installation PyTorch 2.1 (CUDA 11.8)..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# ── 5. Installation dépendances projet ───────────────────────
echo -e "\n[6/6] Installation dépendances MedViT-Lite..."
pip install -r requirements.txt

# ── 6. Pré-téléchargement ChestMNIST ────────────────────────
echo -e "\n[6/6] Pré-téléchargement ChestMNIST..."
python -c "
import os
from medmnist import ChestMNIST
os.makedirs('./data/raw', exist_ok=True)
for split in ['train','val','test']:
    ChestMNIST(split=split, download=True, root='./data/raw')
"

echo -e "\n=============================================="
echo "  Setup terminé avec succès !"
echo "=============================================="
