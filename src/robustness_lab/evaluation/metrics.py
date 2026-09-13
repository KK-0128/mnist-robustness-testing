"""Pure evaluation functions (no side effects, no file I/O) — unit-testable.

The three metrics mirror the report's evaluation indicators:

- clean accuracy (standard accuracy);
- adversarial accuracy (empirical robustness under a specific attack);
- smoothed accuracy + certified L2 radius (randomized smoothing).
"""

from __future__ import annotations

import torch

from ..attacks.base import Attack
from ..defenses.randomized_smoothing import certify, smooth_predict


def clean_accuracy(model: torch.nn.Module, loader, device: str) -> float:
    """Classification accuracy on unmodified data."""
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += int((pred == target).sum().item())
            total += target.size(0)
    return correct / total if total else 0.0


def adversarial_accuracy(
    model: torch.nn.Module,
    loader,
    attack: Attack,
    device: str,
) -> float:
    """Accuracy under a given attack (per-sample generation, as in the report)."""
    correct = total = 0
    model.eval()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        adv = attack.generate(model, data, target)
        with torch.no_grad():
            pred = model(adv).argmax(dim=1)
        correct += int((pred == target).sum().item())
        total += target.size(0)
    return correct / total if total else 0.0


def smoothed_accuracy(
    model: torch.nn.Module,
    loader,
    n_samples: int,
    sigma: float,
    device: str,
    num_classes: int = 10,
) -> float:
    """Accuracy of the smoothed classifier (majority vote over noisy passes)."""
    correct = total = 0
    model.eval()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        for i in range(data.size(0)):
            pred, _ = smooth_predict(
                model, data[i : i + 1], n_samples, sigma, device, num_classes
            )
            correct += int(pred == int(target[i].item()))
            total += 1
    return correct / total if total else 0.0


def certified_radii(
    model: torch.nn.Module,
    loader,
    n_samples: int,
    sigma: float,
    alpha: float,
    device: str,
    num_classes: int = 10,
    max_samples: int | None = None,
) -> list[float]:
    """Certified L2 radius per sample; 0.0 means the sample is not certified."""
    radii: list[float] = []
    for data, target in loader:
        data = data.to(device)
        for i in range(data.size(0)):
            if max_samples is not None and len(radii) >= max_samples:
                return radii
            _, radius = certify(
                model, data[i : i + 1], n_samples, sigma, alpha, device, num_classes
            )
            radii.append(radius)
    return radii


def certified_fraction(radii: list[float], threshold: float) -> float:
    """Fraction of samples whose certified radius is >= ``threshold``."""
    if not radii:
        return 0.0
    return sum(1.0 for r in radii if r >= threshold) / len(radii)
