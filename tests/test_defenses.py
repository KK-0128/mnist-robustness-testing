"""Smoke tests for defense training + randomized smoothing primitives."""

from __future__ import annotations

import pytest
import torch

from robustness_lab.attacks.pgd import PGD
from robustness_lab.config import TrainConfig
from robustness_lab.defenses.adversarial_training import train_adversarial, train_standard
from robustness_lab.defenses.randomized_smoothing import lower_confidence_bound, smooth_predict
from robustness_lab.evaluation.metrics import adversarial_accuracy
from robustness_lab.models import build_model

pytestmark = pytest.mark.smoke


def test_pgd_at_improves_pgd_robustness(device, mnist_train_strong, mnist_test_subset):
    """PGD adversarial training must beat standard training against PGD(0.1).

    Trained on 4000 samples for 5 epochs (measured margin ~0.11 on CPU).
    """
    eval_attack = PGD(eps=0.1, alpha=0.01, iters=10, device=device)

    standard = build_model(42).to(device)
    train_standard(standard, mnist_train_strong, TrainConfig(epochs=5, seed=42), device)
    acc_standard = adversarial_accuracy(standard, mnist_test_subset, eval_attack, device)

    defended = build_model(42).to(device)
    train_attack = PGD(eps=0.1, alpha=0.01, iters=10, device=device)
    train_adversarial(
        defended, mnist_train_strong, train_attack, TrainConfig(epochs=5, seed=42), device
    )
    acc_defended = adversarial_accuracy(defended, mnist_test_subset, eval_attack, device)

    assert acc_defended > acc_standard, (
        f"PGD-AT did not improve robustness: standard={acc_standard:.3f} pgd_at={acc_defended:.3f}"
    )


def test_smooth_predict_majority_vote(trained_model, device):
    """smooth_predict returns a valid class and counts that sum to n_samples."""
    x = torch.zeros(1, 1, 28, 28, device=device)

    pred, counts = smooth_predict(trained_model, x, n_samples=5, sigma=0.25, device=device)

    assert 0 <= pred < 10
    assert int(counts.sum().item()) == 5
    assert int(counts.max().item()) == counts[pred].item()


def test_lower_confidence_bound_matches_clopper_pearson():
    """The certified-radius math must match the textbook Clopper-Pearson formula."""
    from scipy.stats import beta

    counts = torch.zeros(10, dtype=torch.long)
    counts[0] = 29
    counts[1] = 1
    n_samples, alpha = 30, 0.001

    p_lower = lower_confidence_bound(counts, n_samples, alpha)
    expected = float(beta.ppf(alpha, 29, 30 - 29 + 1))

    assert p_lower == pytest.approx(expected)
    assert p_lower > 0.5  # 29/30 successes certifies a positive radius


def test_certify_radius_tiny_noise_consistent(trained_model, device):
    """With nearly zero noise the certified prediction matches the clean one."""
    from robustness_lab.defenses.randomized_smoothing import certify

    x = torch.zeros(1, 1, 28, 28, device=device)
    with torch.no_grad():
        clean_pred = int(trained_model(x).argmax(dim=1).item())

    pred, radius = certify(trained_model, x, n_samples=30, sigma=1e-6, alpha=0.001, device=device)

    assert pred == clean_pred
    assert radius > 0.0
