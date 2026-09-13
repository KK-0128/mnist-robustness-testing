"""Standard / adversarial / gaussian-augmented training routines.

Faithful to the ``train`` function in the report appendix (``adv=False`` for
standard training; ``adv=True`` + an attack for adversarial training), plus the
Gaussian data-augmentation training used for the randomized-smoothing base
classifier (report section 3.3.7).
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from ..attacks.base import Attack
from ..config import TrainConfig


def set_seed(seed: int) -> None:
    """Seed every RNG so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> optim.Optimizer:
    return optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )


def train_standard(
    model: torch.nn.Module,
    train_loader,
    cfg: TrainConfig,
    device: str,
) -> torch.nn.Module:
    """Standard training (report's ``train(..., adv=False)``)."""
    set_seed(cfg.seed)
    model.train()
    optimizer = _make_optimizer(model, cfg)
    for _ in range(cfg.epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return model


def train_adversarial(
    model: torch.nn.Module,
    train_loader,
    attack: Attack,
    cfg: TrainConfig,
    device: str,
) -> torch.nn.Module:
    """Adversarial training: attack every batch, then train on adversarial samples.

    ``attack.generate`` temporarily switches the model to eval mode and restores
    it, so training mode is preserved between batches.
    """
    set_seed(cfg.seed)
    model.train()
    optimizer = _make_optimizer(model, cfg)
    for _ in range(cfg.epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            adv_data = attack.generate(model, data, target)
            optimizer.zero_grad()
            output = model(adv_data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return model


def train_gaussian(
    model: torch.nn.Module,
    train_loader,
    cfg: TrainConfig,
    sigma: float,
    device: str,
) -> torch.nn.Module:
    """Gaussian data-augmentation training: add N(0, sigma^2 I) noise per batch.

    Improves the base classifier's compatibility with randomized smoothing
    (report section 3.3.7, strategy 1).
    """
    set_seed(cfg.seed)
    model.train()
    optimizer = _make_optimizer(model, cfg)
    for _ in range(cfg.epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            noisy = data + sigma * torch.randn_like(data)
            optimizer.zero_grad()
            output = model(noisy)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return model
