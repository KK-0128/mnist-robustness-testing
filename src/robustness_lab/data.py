"""MNIST loading and preprocessing (kept minimal, as in the report)."""

from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import DataConfig


def build_loaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader). Pixels are normalized to [0, 1]."""
    transform = transforms.ToTensor()
    kwargs = {"num_workers": cfg.num_workers, "pin_memory": True} if cfg.num_workers > 0 else {}
    train_loader = DataLoader(
        datasets.MNIST(cfg.root, train=True, download=cfg.download, transform=transform),
        batch_size=cfg.train_batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = DataLoader(
        datasets.MNIST(cfg.root, train=False, download=cfg.download, transform=transform),
        batch_size=cfg.test_batch_size,
        shuffle=False,
        **kwargs,
    )
    return train_loader, test_loader
