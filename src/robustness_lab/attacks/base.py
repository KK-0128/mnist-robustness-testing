"""Abstract interface every attack implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Attack(ABC):
    """Generates adversarial examples within a bounded perturbation budget.

    Implementations must guarantee:

    - ``||generate(model, x, y) - x||_p <= eps`` for the norm used by the attack;
    - output values stay within ``[clip_min, clip_max]``;
    - the input tensors are NOT mutated in place.

    ``generate`` temporarily switches the model to eval mode and restores the
    previous training mode afterwards, so it is safe to call during training.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    @abstractmethod
    def generate(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return adversarial examples for the given batch."""
        raise NotImplementedError
