"""Projected Gradient Descent attack (Madry et al., 2018).

Faithful to ``pgd_attack`` in the report appendix: iterative sign-gradient
updates, each step projected back into the L-infinity ball of radius ``eps``
and clipped to valid pixel range.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Attack


class PGD(Attack):
    def __init__(
        self,
        eps: float = 0.1,
        alpha: float = 0.01,
        iters: int = 10,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(device=device)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters if iters is not None else max(int(eps / alpha), 1)
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
        images = images.to(self.device).clone().detach()
        labels = labels.to(self.device)
        ori_images = images.data.clone()
        adv = images
        try:
            for _ in range(self.iters):
                adv = adv.clone().requires_grad_(True)
                output = model(adv)
                loss = F.nll_loss(output, labels)
                model.zero_grad()
                loss.backward()
                adv = adv + self.alpha * adv.grad.sign()
                eta = torch.clamp(adv - ori_images, min=-self.eps, max=self.eps)
                adv = torch.clamp(ori_images + eta, self.clip_min, self.clip_max).detach()
            return adv
        finally:
            model.train(was_training)
