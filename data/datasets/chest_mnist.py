"""
MedViT-Lite — ChestMNIST Dataset Module
========================================
ChestMNIST est une version 28×28 (ou 224×224 en high-res) du NIH Chest X-Ray.
Il contient 112,120 images avec 14 labels de pathologies pulmonaires.

Pourquoi ce dataset en premier :
- Téléchargement automatique via medmnist (1 ligne de code)
- Même distribution que NIH Chest X-Ray (référence académique)
- Idéal pour valider rapidement l'architecture avant les vrais runs

Labels (14 classes, multi-label) :
  0: Atelectasis     1: Cardiomegaly   2: Effusion
  3: Infiltration    4: Mass           5: Nodule
  6: Pneumonia       7: Pneumothorax   8: Consolidation
  9: Edema          10: Emphysema     11: Fibrosis
 12: Pleural Thickening              13: Hernia
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import medmnist
from medmnist import ChestMNIST as MedMNISTChest
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Noms lisibles des 14 pathologies
PATHOLOGY_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia"
]


class ChestMNISTDataset(Dataset):
    """
    Wrapper autour du dataset ChestMNIST pour MedViT-Lite.

    Caractéristiques :
    - Multi-label : une image peut avoir plusieurs pathologies simultanément
    - Déséquilibre de classes important (Hernia << Infiltration)
    - Images en niveaux de gris → converties en RGB pour les modèles pré-entraînés

    Args:
        split      : "train", "val" ou "test"
        image_size : taille de redimensionnement (224 recommandé pour ViT)
        root       : dossier de téléchargement des données
        download   : télécharger si absent
    """

    def __init__(
        self,
        split: str = "train",
        image_size: int = 224,
        root: str = "./data/raw",
        download: bool = True,
    ):
        assert split in ("train", "val", "test"), \
            f"split doit être 'train', 'val' ou 'test', reçu: {split}"

        self.split = split
        self.image_size = image_size

        # ── Transformations ──────────────────────────────────────────────────
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                # Conversion grayscale → RGB (répéter le canal 3 fois)
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                # Normalisation ImageNet (même si images médicales, c'est standard)
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            # Pas d'augmentation pour val/test
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        # ── Chargement du dataset ────────────────────────────────────────────
        logger.info(f"Chargement ChestMNIST [{split}]...")
        import os
        os.makedirs(root, exist_ok=True)   # garantit que le dossier existe
        self.dataset = MedMNISTChest(
            split=split,
            transform=self.transform,
            download=download,
            root=root,
            # size=image_size : non supporté (valeurs valides : 28, 64, 128)
            # Le resize à 224 est déjà géré par transforms.Resize ci-dessus
            as_rgb=True,
        )

        # Les labels sont de forme (N, 14) — multi-label binaire
        self.labels = self.dataset.labels  # shape: (N, 14)
        logger.info(
            f"  → {len(self.dataset)} images chargées "
            f"({self.labels.sum(axis=0).astype(int)} positifs par classe)"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[idx]

        # Convertir label en float tensor (pour BCEWithLogitsLoss)
        label = torch.tensor(label, dtype=torch.float32).squeeze()

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Calcule les poids de classe pour gérer le déséquilibre.
        
        Formule : w_c = N_total / (N_classes × N_positifs_c)
        Plus une classe est rare, plus son poids est élevé.
        """
        n_total = len(self.labels)
        n_positives = self.labels.sum(axis=0)  # shape: (14,)
        
        # Éviter la division par zéro pour les classes vides
        n_positives = np.maximum(n_positives, 1)
        
        weights = n_total / (len(PATHOLOGY_NAMES) * n_positives)
        return torch.tensor(weights, dtype=torch.float32)

    def get_prevalence(self) -> dict:
        """Retourne la prévalence (%) de chaque pathologie."""
        n_total = len(self.labels)
        prevalence = {}
        for i, name in enumerate(PATHOLOGY_NAMES):
            prevalence[name] = 100.0 * self.labels[:, i].sum() / n_total
        return prevalence


def build_dataloaders(
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 8,
    root: str = "./data/raw",
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construit les DataLoaders train/val/test pour ChestMNIST.

    Args:
        image_size  : taille des images (224 pour ViT)
        batch_size  : taille de batch PAR GPU
        num_workers : threads de chargement
        root        : dossier des données
        pin_memory  : accélère le transfert CPU→GPU

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = ChestMNISTDataset("train", image_size, root)
    val_dataset   = ChestMNISTDataset("val",   image_size, root)
    test_dataset  = ChestMNISTDataset("test",  image_size, root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # mélanger à chaque epoch
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,        # ignorer le dernier batch incomplet
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,   # ×2 car pas de gradient → plus de mémoire
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(
        f"DataLoaders créés : "
        f"train={len(train_loader)} batches, "
        f"val={len(val_loader)} batches, "
        f"test={len(test_loader)} batches"
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # ── Test rapide du pipeline ──────────────────────────────────────────────
    logging.basicConfig(level=logging.INFO)

    print("Test du pipeline ChestMNIST...")
    train_loader, val_loader, test_loader = build_dataloaders(
        image_size=224, batch_size=32, num_workers=0
    )

    # Vérifier un batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch de test :")
    print(f"  images : {images.shape}  (B, C, H, W)")
    print(f"  labels : {labels.shape}  (B, 14) — multi-label")
    print(f"  min/max pixels : {images.min():.2f} / {images.max():.2f}")
    print(f"  labels sample  : {labels[0].numpy().astype(int)}")

    # Prévalences
    train_ds = train_loader.dataset
    prev = train_ds.get_prevalence()
    print(f"\nPrévalences dans le train set :")
    for name, pct in sorted(prev.items(), key=lambda x: -x[1]):
        bar = "█" * int(pct / 2)
        print(f"  {name:<25} {pct:5.1f}%  {bar}")
