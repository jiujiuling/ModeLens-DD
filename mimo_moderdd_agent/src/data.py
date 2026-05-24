from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class ToyDataset:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    num_classes: int


def make_toy_dataset(
    name: str = "moons",
    n_samples: int = 1200,
    test_size: float = 0.35,
    seed: int = 7,
    device: str = "cpu",
) -> ToyDataset:
    """Create a small but non-trivial classification dataset.

    The data are deliberately 2D so that distribution distillation behavior can be
    inspected visually. This makes the project easy to demo without downloading
    any external dataset.
    """
    rng = np.random.RandomState(seed)
    name = name.lower()
    if name == "moons":
        x, y = make_moons(n_samples=n_samples, noise=0.10, random_state=seed)
    elif name == "circles":
        x, y = make_circles(n_samples=n_samples, noise=0.06, factor=0.45, random_state=seed)
    elif name == "blobs3":
        centers = [(-2.0, -1.5), (1.8, -1.2), (0.2, 1.8)]
        x, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=[0.55, 0.65, 0.50], random_state=seed)
    elif name == "blobs4":
        centers = [(-2.0, -2.0), (2.0, -1.8), (-1.8, 2.0), (2.0, 2.0)]
        x, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.55, random_state=seed)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from moons/circles/blobs3/blobs4.")

    x = StandardScaler().fit_transform(x).astype(np.float32)
    # Add a mild anisotropic perturbation to make mode coverage meaningful.
    x = x @ np.array([[1.15, 0.25], [-0.10, 0.90]], dtype=np.float32)
    x += rng.normal(scale=0.01, size=x.shape).astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y.astype(np.int64), test_size=test_size, random_state=seed, stratify=y
    )
    return ToyDataset(
        x_train=torch.tensor(x_train, dtype=torch.float32, device=device),
        y_train=torch.tensor(y_train, dtype=torch.long, device=device),
        x_test=torch.tensor(x_test, dtype=torch.float32, device=device),
        y_test=torch.tensor(y_test, dtype=torch.long, device=device),
        num_classes=int(np.max(y) + 1),
    )


def init_synthetic_from_real(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    ipc: int,
    seed: int = 7,
    noise: float = 0.08,
) -> Tuple[torch.nn.Parameter, torch.Tensor]:
    """Initialize class-balanced synthetic points from real samples."""
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    xs, ys = [], []
    for c in range(num_classes):
        idx = torch.where(y == c)[0]
        if len(idx) < ipc:
            chosen = idx[torch.randint(len(idx), (ipc,), generator=gen, device=x.device)]
        else:
            perm = torch.randperm(len(idx), generator=gen, device=x.device)[:ipc]
            chosen = idx[perm]
        base = x[chosen].clone()
        base += noise * torch.randn(base.shape, generator=gen, device=x.device)
        xs.append(base)
        ys.append(torch.full((ipc,), c, dtype=torch.long, device=x.device))
    syn_x = torch.nn.Parameter(torch.cat(xs, dim=0))
    syn_y = torch.cat(ys, dim=0)
    return syn_x, syn_y
