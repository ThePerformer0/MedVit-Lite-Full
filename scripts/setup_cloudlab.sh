#!/bin/bash
# ============================================================
# MedViT-Lite — Setup CloudLab LEGACY (2× Tesla K40m)
# ============================================================
# Utilisation de Python 3.9 et PyTorch 1.10.2 avec CUDA 11.3
# pour assurer la compatibilité avec l'architecture Kepler (sm_35)

set -e  # Arrêter immédiatement en cas d'erreur

echo "=============================================="
echo "  MedViT-Lite — Setup CloudLab LEGACY"
echo "  GPU : 2× NVIDIA Tesla K40m (Kepler, 2013)"
echo "=============================================="

# ── 1. Mise à jour système et Drivers NVIDIA ────────────────
echo -e "\n[1/6] Mise à jour des paquets et Drivers NVIDIA 470..."
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git wget curl build-essential \
    libopencv-dev python3-dev python3-pip \
    htop tree nvidia-driver-470-server

# ── 2. Installation Miniconda ────────────────────────────────
echo -e "\n[2/6] Installation Miniconda..."
if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
fi
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# Accepter les termes de Conda pour éviter l'erreur
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# ── 3. Création environnement conda (Python 3.9) ─────────────
echo -e "\n[3/6] Création environnement 'medvit' avec Python 3.9..."
conda create -n medvit python=3.9 -y
conda activate medvit

# ── 4. Installation PyTorch 1.10.2 (CUDA 11.3) ───────────────
echo -e "\n[4/6] Installation PyTorch 1.10.2 (Dernière version supportant K40m)..."
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# ── 5. Installation des dépendances du projet ────────────────
echo -e "\n[5/6] Installation des dépendances MedViT-Lite..."
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
echo "  IMPORTANT: Le système doit maintenant être redémarré"
echo "  pour que les drivers NVIDIA soient activés."
echo "=============================================="
