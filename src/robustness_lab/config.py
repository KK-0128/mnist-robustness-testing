"""Typed configuration objects.

All experiment hyperparameters live here (or in YAML files that map onto these
dataclasses) so that every run is reproducible and self-documenting. Unknown
keys in YAML are ignored; missing keys fall back to dataclass defaults.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


def _from_dict(cls: type, data: dict) -> object:
    """Build a dataclass from a dict, keeping only declared fields."""
    valid = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in valid}
    # YAML lists -> tuples for hashable fields
    if "epsilons" in kwargs and isinstance(kwargs["epsilons"], list):
        kwargs["epsilons"] = tuple(kwargs["epsilons"])
    return cls(**kwargs)


@dataclass
class DataConfig:
    root: str = "./data"
    train_batch_size: int = 64
    test_batch_size: int = 1  # per-sample attack generation, as in the report
    num_workers: int = 1
    download: bool = True


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 5e-4
    seed: int = 42


@dataclass
class FGSMConfig:
    eps: float = 0.1
    clip_min: float = 0.0
    clip_max: float = 1.0


@dataclass
class PGDConfig:
    eps: float = 0.1
    alpha: float = 0.01
    iters: int = 10  # report: int(eps / alpha)
    clip_min: float = 0.0
    clip_max: float = 1.0


@dataclass
class CWConfig:
    # Defaults follow the corrected untargeted C&W objective (see attacks/cw.py).
    c: float = 10.0
    k: float = 0.0
    lr: float = 0.01
    max_iter: int = 100
    clip_min: float = 0.0
    clip_max: float = 1.0


@dataclass
class RSDConfig:
    """Randomized smoothing configuration (Cohen et al., 2019)."""

    sigma: float = 0.25
    n_samples: int = 30
    alpha: float = 0.001  # 1 - alpha = 0.999 confidence
    cert_max_samples: int = 100  # report certifies first 100 test samples


@dataclass
class EvalConfig:
    epsilons: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
    pgd_alpha: float = 0.01


@dataclass
class ExperimentConfig:
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    fgsm: FGSMConfig = dataclasses.field(default_factory=FGSMConfig)
    pgd: PGDConfig = dataclasses.field(default_factory=PGDConfig)
    cw: CWConfig = dataclasses.field(default_factory=CWConfig)
    rs: RSDConfig = dataclasses.field(default_factory=RSDConfig)
    eval: EvalConfig = dataclasses.field(default_factory=EvalConfig)
    defense: str = "none"  # none | fgsm | pgd | cw | gaussian | pgd_rs
    output_dir: str = "experiments/results"


_NESTED: dict[str, type] = {
    "data": DataConfig,
    "train": TrainConfig,
    "fgsm": FGSMConfig,
    "pgd": PGDConfig,
    "cw": CWConfig,
    "rs": RSDConfig,
    "eval": EvalConfig,
}

_SCALARS = ("defense", "output_dir")


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from YAML; missing keys fall back to defaults."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = ExperimentConfig()
    for key, cls in _NESTED.items():
        if key in raw and isinstance(raw[key], dict):
            setattr(cfg, key, _from_dict(cls, raw[key]))
    for key in _SCALARS:
        if key in raw:
            setattr(cfg, key, raw[key])
    return cfg
