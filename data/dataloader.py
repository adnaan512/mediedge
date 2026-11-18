"""
MediEdge — Data loading utilities using MedMNIST.

Supported datasets:
  - dermamnist  : 7-class skin lesion classification
  - pathmnist   : 9-class colon histology classification
"""

import os
from typing import Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO


# ── Registry ──────────────────────────────────────────────────────────────────

SUPPORTED_DATASETS = {
    "dermamnist": "DermaMNIST",
    "pathmnist": "PathMNIST",
}


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(split: str, image_size: int = 28) -> transforms.Compose:
    """Return augmentation pipeline for train/val/test splits."""
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


# ── Dataset info ───────────────────────────────────────────────────────────────

def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Return MedMNIST metadata for a given dataset."""
    dataset_name = dataset_name.lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Choose from: {list(SUPPORTED_DATASETS.keys())}"
        )
    info = INFO[dataset_name]
    return {
        "name": dataset_name,
        "n_classes": len(info["label"]),
        "label_names": list(info["label"].values()),
        "task": info["task"],
        "n_channels": info["n_channels"],
    }


# ── Loaders ────────────────────────────────────────────────────────────────────

def get_loaders(
    dataset_name: str = "dermamnist",
    batch_size: int = 64,
    image_size: int = 28,
    data_dir: str = "./data/cache",
    num_workers: int = 2,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for a MedMNIST dataset.

    Returns:
        train_loader, val_loader, test_loader
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Choose from: {list(SUPPORTED_DATASETS.keys())}"
        )

    os.makedirs(data_dir, exist_ok=True)

    DatasetClass = getattr(medmnist, SUPPORTED_DATASETS[dataset_name])

    train_dataset = DatasetClass(
        split="train",
        transform=get_transforms("train", image_size),
        download=download,
        root=data_dir,
        as_rgb=True,
    )
    val_dataset = DatasetClass(
        split="val",
        transform=get_transforms("val", image_size),
        download=download,
        root=data_dir,
        as_rgb=True,
    )
    test_dataset = DatasetClass(
        split="test",
        transform=get_transforms("test", image_size),
        download=download,
        root=data_dir,
        as_rgb=True,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,  # CPU-only
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    info = get_dataset_info(dataset_name)
    print(f"[MediEdge] Dataset : {dataset_name.upper()}")
    print(f"           Task    : {info['task']}")
    print(f"           Classes : {info['n_classes']} → {info['label_names']}")
    print(f"           Splits  : train={len(train_dataset)}, "
          f"val={len(val_dataset)}, test={len(test_dataset)}")

    return train_loader, val_loader, test_loader
