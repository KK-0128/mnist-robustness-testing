"""Benchmark runner: trains the defended model and collects all metrics."""

from __future__ import annotations

import logging

import torch

from ..attacks.cw import CW
from ..attacks.fgsm import FGSM
from ..attacks.pgd import PGD
from ..config import ExperimentConfig
from ..data import build_loaders
from ..defenses.adversarial_training import (
    train_adversarial,
    train_gaussian,
    train_standard,
)
from ..models import build_model
from .metrics import (
    adversarial_accuracy,
    certified_fraction,
    certified_radii,
    clean_accuracy,
    smoothed_accuracy,
)

log = logging.getLogger(__name__)

# Defenses for which randomized smoothing evaluation is meaningful.
_SMOOTHING_DEFENSES = frozenset({"none", "gaussian", "pgd_rs"})


def train_defended_model(
    cfg: ExperimentConfig,
    device: str,
) -> torch.nn.Module:
    """Train a model according to ``cfg.defense`` and return it."""
    train_loader, _ = build_loaders(cfg.data)
    model = build_model(cfg.train.seed).to(device)

    defense = cfg.defense
    if defense in ("none", ""):
        train_standard(model, train_loader, cfg.train, device)
    elif defense == "fgsm":
        attack = FGSM(
            cfg.fgsm.eps, cfg.fgsm.clip_min, cfg.fgsm.clip_max, device=device
        )
        train_adversarial(model, train_loader, attack, cfg.train, device)
    elif defense == "pgd":
        attack = PGD(
            cfg.pgd.eps,
            cfg.pgd.alpha,
            cfg.pgd.iters,
            cfg.pgd.clip_min,
            cfg.pgd.clip_max,
            device=device,
        )
        train_adversarial(model, train_loader, attack, cfg.train, device)
    elif defense == "cw":
        attack = CW(
            cfg.cw.c,
            cfg.cw.k,
            cfg.cw.lr,
            cfg.cw.max_iter,
            cfg.cw.clip_min,
            cfg.cw.clip_max,
            device=device,
        )
        train_adversarial(model, train_loader, attack, cfg.train, device)
    elif defense == "gaussian":
        train_gaussian(model, train_loader, cfg.train, cfg.rs.sigma, device)
    elif defense == "pgd_rs":
        # Combined method from the report: PGD-AT (stronger eps) + randomized
        # smoothing certification. Training uses the PGD attack config.
        attack = PGD(
            cfg.pgd.eps,
            cfg.pgd.alpha,
            cfg.pgd.iters,
            cfg.pgd.clip_min,
            cfg.pgd.clip_max,
            device=device,
        )
        train_adversarial(model, train_loader, attack, cfg.train, device)
    else:
        raise ValueError(f"Unknown defense: {defense!r}")
    return model


def run_benchmark(
    cfg: ExperimentConfig,
    device: str | None = None,
) -> dict:
    """Train the model from ``cfg`` and evaluate it across the metric matrix."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = build_loaders(cfg.data)

    log.info("Training model with defense=%r on %s", cfg.defense, device)
    model = train_defended_model(cfg, device)

    results: dict = {
        "defense": cfg.defense,
        "device": device,
        "clean_accuracy": clean_accuracy(model, test_loader, device),
    }

    # Epsilon sweep: FGSM and PGD (eps=0 is clean accuracy, reported above).
    fgsm_scores: dict[float, float] = {}
    pgd_scores: dict[float, float] = {}
    for eps in cfg.eval.epsilons:
        if eps == 0:
            continue
        fgsm_scores[eps] = adversarial_accuracy(
            model, test_loader, FGSM(eps, device=device), device
        )
        pgd_scores[eps] = adversarial_accuracy(
            model,
            test_loader,
            PGD(
                eps,
                cfg.eval.pgd_alpha,
                max(int(eps / cfg.eval.pgd_alpha), 1),
                device=device,
            ),
            device,
        )
    results["fgsm_accuracy"] = fgsm_scores
    results["pgd_accuracy"] = pgd_scores

    # CW (single configuration, as in the report).
    cw_attack = CW(
        cfg.cw.c, cfg.cw.k, cfg.cw.lr, cfg.cw.max_iter, device=device
    )
    results["cw_accuracy"] = adversarial_accuracy(
        model, test_loader, cw_attack, device
    )

    # Randomized smoothing evaluation (report sections 4.4-4.5).
    if cfg.defense in _SMOOTHING_DEFENSES:
        results["smoothed_accuracy"] = smoothed_accuracy(
            model, test_loader, cfg.rs.n_samples, cfg.rs.sigma, device
        )
        radii = certified_radii(
            model,
            test_loader,
            cfg.rs.n_samples,
            cfg.rs.sigma,
            cfg.rs.alpha,
            device,
            max_samples=cfg.rs.cert_max_samples,
        )
        results["certified_radii"] = radii
        results["certified_fraction_0_5"] = certified_fraction(radii, 0.5)
    else:
        results["smoothed_accuracy"] = None
        results["certified_radii"] = None
        results["certified_fraction_0_5"] = None

    return results
