"""Behavioral tests for the three attacks.

Smoke-level guarantees:

- an attack must reduce accuracy (it is actually finding defects);
- FGSM/PGD must respect their L-infinity perturbation budget;
- outputs stay in the valid pixel range;
- ``generate`` must not mutate its input tensors.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.smoke


def _batch_accuracy(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    return float((pred == y).float().mean().item())


@pytest.mark.parametrize("attack_name", ["fgsm", "pgd", "cw"])
def test_attack_reduces_accuracy(attack_name, request, trained_model, mnist_eval_batch, device):
    attack = request.getfixturevalue(attack_name)
    x, y = mnist_eval_batch

    acc_clean = _batch_accuracy(trained_model, x, y)
    adv = attack.generate(trained_model, x, y)
    acc_adv = _batch_accuracy(trained_model, adv, y)

    assert acc_clean > 0.5, f"model too weak for a meaningful test: clean={acc_clean:.3f}"
    assert acc_adv < acc_clean, (
        f"{attack_name} did not reduce accuracy: clean={acc_clean:.3f} adv={acc_adv:.3f}"
    )


@pytest.mark.parametrize("attack_name", ["fgsm", "pgd"])
def test_perturbation_budget_respected(attack_name, request, trained_model, mnist_eval_batch, device):
    attack = request.getfixturevalue(attack_name)
    x, y = mnist_eval_batch

    adv = attack.generate(trained_model, x, y)
    delta = (adv - x).abs()

    assert delta.max().item() <= attack.eps + 1e-6, (
        f"{attack_name} violated the L-inf budget: {delta.max().item():.6f} > {attack.eps}"
    )
    assert adv.min().item() >= 0.0 and adv.max().item() <= 1.0
    assert adv.shape == x.shape


def test_generate_does_not_mutate_input(fgsm, pgd, trained_model, mnist_eval_batch, device):
    x, y = mnist_eval_batch
    x_before = x.clone()

    fgsm.generate(trained_model, x, y)
    pgd.generate(trained_model, x, y)

    assert torch.equal(x, x_before), "attack.generate mutated its input"


def test_attack_preserves_model_mode(pgd, trained_model, mnist_eval_batch, device):
    """generate() must restore the model's training mode after eval."""
    x, y = mnist_eval_batch

    trained_model.train()
    pgd.generate(trained_model, x, y)
    assert trained_model.training is True

    trained_model.eval()
    pgd.generate(trained_model, x, y)
    assert trained_model.training is False
