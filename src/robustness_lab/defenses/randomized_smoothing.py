"""Randomized smoothing: smoothed prediction + certified L2 radius.

Faithful to the report (Cohen et al., 2019 recipe):

- smoothed prediction: majority vote over N Gaussian-noised forward passes;
- certified radius: Clopper-Pearson lower confidence bound via
  ``scipy.stats.beta.ppf``, then ``R = sigma * Phi^{-1}(p_lower)``.
"""

from __future__ import annotations

import torch
from scipy.stats import beta, norm


def smooth_predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_samples: int,
    sigma: float,
    device: str,
    num_classes: int = 10,
) -> tuple[int, torch.Tensor]:
    """Return ``(predicted_class, vote_counts)`` for a single input ``x``.

    ``x`` must have shape ``(1, C, H, W)``. Noise is N(0, sigma^2 I), as in the
    report (sigma=0.25, N=30).
    """
    model.eval()
    x = x.to(device)
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(n_samples):
            noise = torch.randn_like(x) * sigma
            pred = model(x + noise).argmax(dim=1).item()
            counts[pred] += 1
    return int(counts.argmax().item()), counts


def lower_confidence_bound(
    counts: torch.Tensor,
    n_samples: int,
    alpha: float,
) -> float:
    """Clopper-Pearson lower bound for the top class at ``1 - alpha`` confidence."""
    top = int(counts.argmax().item())
    n_a = int(counts[top])
    # beta.ppf(alpha, k, n - k + 1) is the exact binomial lower bound
    return float(beta.ppf(alpha, n_a, n_samples - n_a + 1))


def certify(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_samples: int,
    sigma: float,
    alpha: float,
    device: str,
    num_classes: int = 10,
) -> tuple[int, float]:
    """Return ``(predicted_class, certified_L2_radius)``.

    A radius of 0.0 means the sample could not be certified (p_lower <= 0.5).
    """
    pred, counts = smooth_predict(model, x, n_samples, sigma, device, num_classes)
    p_lower = lower_confidence_bound(counts, n_samples, alpha)
    radius = sigma * float(norm.ppf(p_lower)) if p_lower > 0.5 else 0.0
    return pred, radius
