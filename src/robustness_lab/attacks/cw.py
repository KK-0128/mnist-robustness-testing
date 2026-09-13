"""Carlini & Wagner L2 attack (Carlini & Wagner, 2017).

Single-image optimizer-based attack using the atanh reparametrization
(x = tanh(w)/2 + 0.5 maps unconstrained w onto the valid pixel range) and
Adam, faithful to ``cw_attack`` in the report appendix.

.. note::
   The report appendix implements ``f_val = clamp(other - real + k, 0)``,
   which is the **targeted** form with the target set to the true label —
   minimizing it *protects* the correct class and cannot produce adversarial
   examples. The correct **untargeted** C&W objective is the reverse margin:

       f(x') = max( Z(x')_y - max_{i != y} Z(x')_i + kappa, 0 )

   where ``Z`` is the model output (log-probabilities here, which preserves
   the margin ordering). We implement the corrected objective; this is
   documented in the README.

For batches with more than one sample we loop over samples to stay faithful to
the report (test batch size = 1). Pass ``seed`` for reproducible attacks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import optim

from .base import Attack


def _atanh(t: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.log((1 + t) / (1 - t))


class CW(Attack):
    def __init__(
        self,
        c: float = 10.0,
        k: float = 0.0,
        lr: float = 0.01,
        max_iter: int = 100,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(device=device)
        self.c = c
        self.k = k
        self.lr = lr
        self.max_iter = max_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.seed = seed

    def _attack_one(
        self,
        model: torch.nn.Module,
        x_orig: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        x_orig = x_orig.detach().to(self.device)
        label = label.to(self.device)
        if self.seed is not None:
            torch.manual_seed(self.seed)
        x_tanh = _atanh((x_orig - 0.5) * 1.999999)
        w = (x_tanh + 1e-3 * torch.randn_like(x_tanh)).requires_grad_(True)
        optimizer = optim.Adam([w], lr=self.lr)

        for _ in range(self.max_iter):
            optimizer.zero_grad()
            x_adv = torch.tanh(w) / 2 + 0.5
            l2dist = F.mse_loss(x_adv, x_orig, reduction="sum")
            logits = model(x_adv)
            real = logits[0, label]
            other = torch.max(torch.cat([logits[0, :label], logits[0, label + 1 :]]))
            # Untargeted C&W: push the true class below the best other class.
            f_val = torch.clamp(real - other + self.k, min=0)
            loss = l2dist + self.c * f_val
            loss.backward()
            optimizer.step()

        x_adv = torch.tanh(w) / 2 + 0.5
        return torch.clamp(x_adv, self.clip_min, self.clip_max).detach()

    def generate(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        was_training = model.training
        model.eval()
        images = images.to(self.device).detach()
        labels = labels.to(self.device)
        try:
            adv = torch.empty_like(images)
            for i in range(images.size(0)):
                adv[i] = self._attack_one(model, images[i : i + 1], labels[i])
            return adv
        finally:
            model.train(was_training)
