"""Reproducibility: seeded runs must produce identical results."""

from __future__ import annotations

import pytest

from robustness_lab.config import TrainConfig
from robustness_lab.defenses.adversarial_training import train_standard
from robustness_lab.evaluation.metrics import clean_accuracy
from robustness_lab.models import build_model

pytestmark = pytest.mark.smoke


def test_same_seed_same_clean_accuracy(device, mnist_train_subset, mnist_test_subset):
    accs = []
    for _ in range(2):
        # build_model(42) + train_standard(seed=42) must be fully deterministic
        model = build_model(42).to(device)
        train_standard(
            model, mnist_train_subset, TrainConfig(epochs=1, seed=42), device
        )
        accs.append(clean_accuracy(model, mnist_test_subset, device))

    assert accs[0] == pytest.approx(accs[1], abs=1e-5), (
        f"seeded runs diverged: {accs[0]:.6f} vs {accs[1]:.6f}"
    )


def test_different_seed_changes_result(device, mnist_train_subset, mnist_test_subset):
    """A sanity check that the seed actually matters (not a no-op)."""
    accs = []
    for seed in (42, 43):
        model = build_model(seed).to(device)
        train_standard(
            model, mnist_train_subset, TrainConfig(epochs=1, seed=seed), device
        )
        accs.append(clean_accuracy(model, mnist_test_subset, device))

    assert accs[0] != accs[1], f"different seeds produced identical results: {accs[0]}"
