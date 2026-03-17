"""标准形状隐式场。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class CanonicalField(nn.Module):
    """标准相位域中的 signed distance field。"""

    def __init__(self, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 3
        for _ in range(hidden_layers):
            linear = nn.Linear(in_dim, hidden_dim)
            nn.init.normal_(linear.weight, mean=0.0, std=np.sqrt(2.0 / max(hidden_dim, 1)))
            nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)
            layers.append(nn.Softplus(beta=100.0))
            in_dim = hidden_dim
        final = nn.Linear(in_dim, 1)
        nn.init.normal_(final.weight, mean=np.sqrt(np.pi) / np.sqrt(max(in_dim, 1)), std=1e-4)
        nn.init.constant_(final.bias, -0.1)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.network(points)

    def sdf_and_gradient(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        points = points.requires_grad_(True)
        sdf = self.forward(points)
        grad = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return sdf, grad
