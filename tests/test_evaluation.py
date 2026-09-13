"""Unit tests for evaluation metrics (pure-function correctness)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from robustness_lab.attacks.fgsm import FGSM
from robustness_lab.evaluation.metrics import (
    adversarial_accuracy,
    certified_fraction,
    clean_accuracy,
)

pytestmark = pytest.mark.smoke


class ConstantModel(nn.Module):
    """Predicts class ``cls`` for every input — lets us compute expected acc by hand."""

    def __init__(self, cls: int = 0) -> None:
        super().__init__()
        self.cls = cls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.full((x.size(0), 10), -1e3, device=x.device)
        logits[:, self.cls] = 1e3
        return F.log_softmax(logits, dim=1)


def _fake_loader(batches):
    return [(x, y) for x, y in batches]


def test_clean_accuracy_known_labels():
    loader = _fake_loader(
        [
            (torch.zeros(4, 1, 8, 8), torch.zeros(4, dtype=torch.long)),
            (torch.ones(6, 1, 8, 8), torch.ones(6, dtype=torch.long)),
        ]
    )
    # ConstantModel(0): 4/4 correct on batch 1, 0/6 correct on batch 2 -> 0.4
    acc = clean_accuracy(ConstantModel(0), loader, "cpu")
    assert acc == pytest.approx(0.4)


def test_zero_eps_fgsm_equals_clean(trained_model, mnist_test_subset, device):
    """FGSM with eps=0 must reproduce clean accuracy exactly (invariant)."""
    acc_clean = clean_accuracy(trained_model, mnist_test_subset, device)
    acc_adv = adversarial_accuracy(
        trained_model, mnist_test_subset, FGSM(eps=0.0, device=device), device
    )
    assert acc_adv == pytest.approx(acc_clean)


def test_certified_fraction():
    radii = [0.1, 0.6, 1.2, 0.0, 0.5]
    assert certified_fraction(radii, threshold=0.5) == pytest.approx(3 / 5)
    assert certified_fraction([], threshold=0.5) == 0.0


def test_clean_accuracy_bounds(trained_model, mnist_test_subset, device):
    acc = clean_accuracy(trained_model, mnist_test_subset, device)
    assert 0.0 <= acc <= 1.0
