"""
data_utils.py
-------------
CIFAR-10 dataset loading + IID / non-IID partitioning helpers.

IID   – randomly shuffle and split evenly among clients.
Non-IID – Dirichlet(alpha) label distribution.  Lower alpha = more skew.
          alpha=0.5 is a common "moderate heterogeneity" benchmark value.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple


# ------------------------------------------------------------------
# Standard CIFAR-10 transforms
# ------------------------------------------------------------------
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])


def get_datasets(data_dir: str = "./data"):
    import fcntl, os, time
    lock_path = os.path.join(data_dir, ".download.lock")
    os.makedirs(data_dir, exist_ok=True)
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=TRAIN_TRANSFORM)
            test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=TEST_TRANSFORM)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return train_ds, test_ds


# ------------------------------------------------------------------
# IID partition
# ------------------------------------------------------------------
def iid_partition(dataset, num_clients: int, seed: int = 42) -> List[List[int]]:
    """Randomly shuffle indices and split into num_clients equal chunks."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset)).tolist()
    chunk = len(indices) // num_clients
    return [indices[i * chunk:(i + 1) * chunk] for i in range(num_clients)]


# ------------------------------------------------------------------
# Non-IID Dirichlet partition
# ------------------------------------------------------------------
def dirichlet_partition(dataset, num_clients: int, alpha: float = 0.5,
                        seed: int = 42) -> List[List[int]]:
    """
    Dirichlet(alpha) label-distribution split.
    Each client gets a multinomial mix of classes; lower alpha = more skew.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    # Collect per-class indices
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c_idx in class_indices:
        rng.shuffle(c_idx)
        # Dirichlet proportions for this class
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Convert to integer counts
        counts = (proportions * len(c_idx)).astype(int)
        # Make sure we use all samples (fix rounding)
        counts[-1] = len(c_idx) - counts[:-1].sum()
        cursor = 0
        for client_id, count in enumerate(counts):
            client_indices[client_id].extend(c_idx[cursor:cursor + count].tolist())
            cursor += count

    return client_indices


# ------------------------------------------------------------------
# Build DataLoader for a single client
# ------------------------------------------------------------------
def make_client_loader(dataset, indices: List[int], batch_size: int = 64,
                       shuffle: bool = True) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True)


def make_test_loader(dataset, batch_size: int = 256) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=2, pin_memory=True)


# ------------------------------------------------------------------
# Quick stats helper
# ------------------------------------------------------------------
def partition_stats(dataset, partitions: List[List[int]]) -> dict:
    labels = np.array(dataset.targets)
    stats = {}
    for i, idx in enumerate(partitions):
        counts = np.bincount(labels[idx], minlength=10).tolist()
        stats[f"client_{i}"] = {"total": len(idx), "class_counts": counts}
    return stats
