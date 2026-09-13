"""Fast Gradient Sign Method (Goodfellow et al., 2015).

Faithful to ``fgsm_attack`` in the report appendix:

    x_adv = clamp(x + eps * sign(grad_x J), 0, 1)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Attack


class FGSM(Attack):
    def __init__(
        self,
        eps: float = 0.1,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(device=device)
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max

    def generate(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        was_training = model.training
        model.eval()
        images = images.to(self.device).clone().detach().requires_grad_(True)
        labels = labels.to(self.device)
        try:
            output = model(images)
            loss = F.nll_loss(output, labels)
            model.zero_grad()
            loss.backward()
            grad_sign = images.grad.data.sign()
            perturbed = images + self.eps * grad_sign
            return torch.clamp(perturbed, self.clip_min, self.clip_max).detach()
        finally:
            model.train(was_training)
