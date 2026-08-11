#!/bin/bash

set -e

echo "=============================================="
echo "  MedViT-Lite — Setup Kaggle"
echo "  $(date)"
echo "=============================================="

# ── 1. Vérification GPU ────────────────────────────────────
echo -e "\n[1/4] GPUs disponibles :"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ── 2. Packages manquants (PyTorch déjà installé sur Kaggle) ─
echo -e "\n[2/4] Installation des dépendances manquantes..."
pip install -q einops timm medmnist rich wandb

# ── 3. Cloner le repo ────────────────────────────────────────
echo -e "\n[3/4] Clonage du repo MedViT-Lite..."
cd /kaggle/working

if [ ! -d "MedVit-Lite-Full" ]; then
    git clone https://github.com/ThePerformer0/MedVit-Lite-Full.git
else
    echo "  (repo déjà présent, mise à jour...)"
    cd MedVit-Lite-Full && git pull && cd ..
fi

cd MedVit-Lite-Full

# ── 4. Créer les dossiers ────────────────────────────────────
echo -e "\n[4/4] Création des répertoires..."
mkdir -p /kaggle/working/checkpoints
mkdir -p /kaggle/working/results
mkdir -p /kaggle/working/data/raw
mkdir -p logs

# Lien symbolique vers la config Kaggle
cp configs/kaggle.yaml configs/active.yaml

# ── 5. Pré-téléchargement ChestMNIST ─────────────────────────
echo -e "\nTéléchargement ChestMNIST (~200 Mo)..."
python -c "
import os
os.makedirs('/kaggle/working/data/raw', exist_ok=True)
try:
    from medmnist import ChestMNIST
    for split in ['train', 'val', 'test']:
        ChestMNIST(split=split, download=True, root='/kaggle/working/data/raw')
    print('  ChestMNIST téléchargé avec succès')
except Exception as e:
    print(f'  Erreur : {e}')
"

echo ""
echo "=============================================="
echo "  Setup terminé !"
echo "=============================================="
