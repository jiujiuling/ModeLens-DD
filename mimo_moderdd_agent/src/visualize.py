from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch


def save_distribution_plot(real_x: torch.Tensor, real_y: torch.Tensor, syn_x: torch.Tensor, syn_y: torch.Tensor, path: str, title: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rx, ry = real_x.detach().cpu(), real_y.detach().cpu()
    sx, sy = syn_x.detach().cpu(), syn_y.detach().cpu()
    plt.figure(figsize=(6, 5))
    for c in sorted(ry.unique().tolist()):
        mask = ry == c
        plt.scatter(rx[mask, 0], rx[mask, 1], s=10, alpha=0.25, label=f"real c{int(c)}")
    for c in sorted(sy.unique().tolist()):
        mask = sy == c
        plt.scatter(sx[mask, 0], sx[mask, 1], s=80, marker="x", linewidths=2.0, label=f"synthetic c{int(c)}")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_loss_curve(history: Dict[str, List[float]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for key, values in history.items():
        if key.endswith("loss") or key in {"total", "cs", "coverage", "entropy"}:
            plt.plot(values, label=key)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Mode-aware Rényi distillation loss")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
