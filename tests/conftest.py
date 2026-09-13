"""Shared fixtures.

Everything here is deliberately small so that CI on CPU stays fast:
- MNIST is downloaded once per session into a tmp dir;
- 3000 train / 200 test samples are used;
- the shared model trains for 3 epochs (clean acc ~0.75, ~seconds on CPU).
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from robustness_lab.attacks.cw import CW
from robustness_lab.attacks.fgsm import FGSM
from robustness_lab.attacks.pgd import PGD
from robustness_lab.config import TrainConfig
from robustness_lab.defenses.adversarial_training import train_standard
from robustness_lab.models import build_model

TRAIN_SUBSET = 3000
TRAIN_SUBSET_STRONG = 4000  # used by the PGD-AT robustness test
TEST_SUBSET = 200


@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def mnist_root(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("mnist"))


def _subset_loader(root: str, train: bool, n: int, batch_size: int) -> DataLoader:
    torch.manual_seed(0)  # deterministic subset for stable tests
    ds = datasets.MNIST(root, train=train, download=True, transform=transforms.ToTensor())
    indices = torch.randperm(len(ds))[:n]
    return DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=False)


@pytest.fixture(scope="session")
def mnist_train_subset(mnist_root) -> DataLoader:
    return _subset_loader(mnist_root, train=True, n=TRAIN_SUBSET, batch_size=64)


@pytest.fixture(scope="session")
def mnist_train_strong(mnist_root) -> DataLoader:
    return _subset_loader(mnist_root, train=True, n=TRAIN_SUBSET_STRONG, batch_size=64)


@pytest.fixture(scope="session")
def mnist_test_subset(mnist_root) -> DataLoader:
    return _subset_loader(mnist_root, train=False, n=TEST_SUBSET, batch_size=1)


@pytest.fixture(scope="session")
def trained_model(device, mnist_train_subset) -> torch.nn.Module:
    """A standard model trained for 3 epochs on 3000 samples (clean ~0.75)."""
    model = build_model(42).to(device)
    train_standard(
        model, mnist_train_subset, TrainConfig(epochs=3, seed=42), device
    )
    model.eval()
    return model


@pytest.fixture(scope="session")
def mnist_eval_batch(mnist_test_subset, device):
    """First 20 test samples as one batch (for attack-effectiveness checks)."""
    xs, ys = [], []
    for data, target in mnist_test_subset:
        xs.append(data)
        ys.append(target)
        if len(xs) >= 20:
            break
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


# --- attack fixtures (CW seeded + capped iterations so smoke stays fast) ---


@pytest.fixture(scope="session")
def fgsm(device) -> FGSM:
    return FGSM(eps=0.1, device=device)


@pytest.fixture(scope="session")
def pgd(device) -> PGD:
    return PGD(eps=0.1, alpha=0.01, iters=10, device=device)


@pytest.fixture(scope="session")
def cw(device) -> CW:
    return CW(c=10.0, k=0.0, lr=0.01, max_iter=100, device=device, seed=7)
