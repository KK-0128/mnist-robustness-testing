"""CNN model used in the report (Net), plus a factory function."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    """Standard CNN from the report: conv -> pool -> conv -> dropout -> fc."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def build_model(seed: int | None = None) -> Net:
    """Create a fresh Net; optionally seed PyTorch first for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
    return Net()
