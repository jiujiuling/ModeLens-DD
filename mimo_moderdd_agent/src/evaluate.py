from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@dataclass
class EvalResult:
    logistic_acc: float
    mlp_acc: float


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def evaluate_synthetic_set(
    syn_x: torch.Tensor,
    syn_y: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    num_classes: int,
    seed: int = 7,
) -> EvalResult:
    x_np = syn_x.detach().cpu().numpy()
    y_np = syn_y.detach().cpu().numpy()
    xt_np = x_test.detach().cpu().numpy()
    yt_np = y_test.detach().cpu().numpy()

    try:
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(x_np, y_np)
        log_acc = float(accuracy_score(yt_np, clf.predict(xt_np)))
    except Exception:
        log_acc = float("nan")

    torch.manual_seed(seed)
    model = TinyMLP(x_test.shape[1], num_classes).to(x_test.device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(300):
        logits = model(syn_x.detach())
        loss = F.cross_entropy(logits, syn_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(x_test).argmax(dim=1)
        mlp_acc = float((pred == y_test).float().mean().detach().cpu())
    return EvalResult(logistic_acc=log_acc, mlp_acc=mlp_acc)
