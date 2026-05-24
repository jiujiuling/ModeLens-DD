from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .losses import pairwise_sq_dists


@dataclass
class ModeEntry:
    layer: str
    cls: int
    centers: torch.Tensor
    probs: torch.Tensor
    tau: float


class ModeBank:
    def __init__(self):
        self.entries: Dict[Tuple[str, int], ModeEntry] = {}

    def add(self, entry: ModeEntry) -> None:
        self.entries[(entry.layer, entry.cls)] = entry

    def get(self, layer: str, cls: int) -> ModeEntry:
        return self.entries[(layer, cls)]

    def keys(self):
        return self.entries.keys()


def _torch_kmeans(x: torch.Tensor, k: int, seed: int = 7, iters: int = 25) -> tuple[torch.Tensor, torch.Tensor]:
    """Small dependency-free KMeans for the demo.

    This avoids heavy sklearn clustering startup cost and keeps the project easy
    to run on a CPU-only machine.
    """
    n = x.shape[0]
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    init_idx = torch.randperm(n, generator=gen, device=x.device)[:k]
    centers = x[init_idx].clone()
    assign = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(iters):
        d = pairwise_sq_dists(x, centers)
        new_assign = d.argmin(dim=1)
        if torch.equal(new_assign, assign):
            break
        assign = new_assign
        new_centers = []
        for j in range(k):
            mask = assign == j
            if mask.any():
                new_centers.append(x[mask].mean(dim=0))
            else:
                # Re-seed empty cluster by the currently worst-represented point.
                farthest = d.min(dim=1).values.argmax()
                new_centers.append(x[farthest])
        centers = torch.stack(new_centers, dim=0)
    return centers, assign


def build_mode_bank(
    features_by_layer: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    num_classes: int,
    modes_per_class: int = 4,
    seed: int = 7,
) -> ModeBank:
    """Build an offline class-layer mode bank using dependency-free KMeans."""
    bank = ModeBank()
    for layer, feats in features_by_layer.items():
        for c in range(num_classes):
            x_c = feats[labels == c].detach()
            k = int(min(max(1, modes_per_class), x_c.shape[0]))
            if k == 1:
                centers = x_c.mean(dim=0, keepdim=True)
                counts = torch.tensor([x_c.shape[0]], dtype=feats.dtype, device=feats.device)
            else:
                centers, assignments = _torch_kmeans(x_c, k=k, seed=seed)
                counts = torch.bincount(assignments, minlength=k).to(feats.dtype)
            probs = counts / counts.sum().clamp_min(1.0)
            if centers.shape[0] > 1:
                d = pairwise_sq_dists(centers, centers)
                vals = d[d > 1e-12]
                tau = float(torch.median(vals).detach().cpu()) if vals.numel() else 1.0
            else:
                tau = 1.0
            bank.add(ModeEntry(layer=layer, cls=c, centers=centers, probs=probs, tau=max(tau, 1e-3)))
    return bank
