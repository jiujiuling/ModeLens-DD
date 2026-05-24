from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F


@dataclass
class EncoderLayer:
    weight: torch.Tensor
    bias: torch.Tensor
    name: str


class FrozenRandomEncoderBank(torch.nn.Module):
    """A tiny frozen multi-layer encoder bank.

    In a full image distillation system this part can be replaced by a random
    ConvNet bank + pretrained semantic encoder. For this standalone MVP, frozen
    random MLP layers provide multiple nonlinear feature spaces while keeping the
    code CPU-friendly.
    """

    def __init__(self, input_dim: int = 2, hidden_dims: List[int] | None = None, seed: int = 7):
        super().__init__()
        hidden_dims = hidden_dims or [16, 32, 32]
        gen = torch.Generator()
        gen.manual_seed(seed)
        dims = [input_dim] + hidden_dims
        self.layers: List[EncoderLayer] = []
        for i in range(len(dims) - 1):
            w = torch.randn(dims[i], dims[i + 1], generator=gen) / (dims[i] ** 0.5)
            b = 0.10 * torch.randn(dims[i + 1], generator=gen)
            self.register_buffer(f"w_{i}", w)
            self.register_buffer(f"b_{i}", b)
            self.layers.append(EncoderLayer(weight=w, bias=b, name=f"rand_mlp_{i+1}"))

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: Dict[str, torch.Tensor] = {"identity": x}
        h = x
        for i, layer in enumerate(self.layers):
            w = getattr(self, f"w_{i}").to(x.device)
            b = getattr(self, f"b_{i}").to(x.device)
            h = torch.tanh(h @ w + b)
            # L2 normalization stabilizes kernel bandwidth and makes layers comparable.
            feats[layer.name] = F.normalize(h, dim=-1)
        return feats
