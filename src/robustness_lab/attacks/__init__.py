"""Adversarial attack implementations (unified Attack interface)."""

from .base import Attack
from .cw import CW
from .fgsm import FGSM
from .pgd import PGD

__all__ = ["CW", "FGSM", "PGD", "Attack"]
