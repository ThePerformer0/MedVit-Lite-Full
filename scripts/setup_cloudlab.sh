#!/bin/bash
# ============================================================
# MedViT-Lite — Setup CloudLab (c4130 : 4× V100 16GB)
# ============================================================
# Usage : bash scripts/setup_cloudlab.sh
# Durée estimée : ~10 minutes (téléchargement CUDA + packages)

set -e  # Arrêter immédiatement en cas d'erreur

echo "=============================================="
echo "  MedViT-Lite — Setup CloudLab c4130"
echo "  GPU : 4× NVIDIA V100 16GB"
echo "=============================================="

# ── 1. Mise à jour système ───────────────────────────────────
echo -e "\n[1/6] Mise à jour des paquets système..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git wget curl build-essential \
    libopencv-dev python3-dev python3-pip \
    htop nvtop tree

# ── 2. Vérification GPU ──────────────────────────────────────
echo -e "\n[2/6] Vérification des GPUs..."
nvidia-smi
echo "GPU count : $(nvidia-smi --list-gpus | wc -l)"

# ── 3. Installation Miniconda ────────────────────────────────
echo -e "\n[3/6] Installation Miniconda..."
if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
fi
export PATH="$HOME/miniconda3/bin:$PATH"
conda init bash
source ~/.bashrc

# ── 4. Création environnement conda ─────────────────────────
echo -e "\n[4/6] Création environnement 'medvit'..."
conda create -n medvit python=3.10 -y
conda activate medvit

# ── 5. Installation PyTorch avec CUDA 11.8 ──────────────────
echo -e "\n[5/6] Installation PyTorch 2.1 + CUDA 11.8..."
# V100 supporte CUDA jusqu'à 11.8 optimal
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Vérification PyTorch + GPU
python -c "
import torch
print(f'PyTorch version : {torch.__version__}')
print(f'CUDA disponible : {torch.cuda.is_available()}')
print(f'Nombre de GPUs  : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)')
"

# ── 6. Installation des dépendances du projet ────────────────
echo -e "\n[6/6] Installation des dépendances MedViT-Lite..."
pip install -r requirements.txt

# ── Configuration W&B (Weights & Biases) ────────────────────
echo -e "\nConfiguration Weights & Biases pour le tracking..."
echo "→ Va sur https://wandb.ai, crée un compte gratuit, et copie ton API key"
echo "→ Lance ensuite : wandb login"

# ── Téléchargement préalable des données ────────────────────
echo -e "\nPré-téléchargement ChestMNIST..."
python -c "
import medmnist
from medmnist import ChestMNIST
print('Téléchargement ChestMNIST (train)...')
_ = ChestMNIST(split='train', download=True, root='./data/raw')
print('Téléchargement ChestMNIST (val)...')
_ = ChestMNIST(split='val',   download=True, root='./data/raw')
print('Téléchargement ChestMNIST (test)...')
_ = ChestMNIST(split='test',  download=True, root='./data/raw')
print('Done !')
"

echo -e "\n=============================================="
echo "  Setup terminé avec succès !"
echo "  Pour activer l'environnement : conda activate medvit"
echo "  Pour lancer l'entraînement   : bash scripts/train_baseline.sh"
echo "=============================================="
