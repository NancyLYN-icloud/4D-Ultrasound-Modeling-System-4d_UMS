"""相位条件形变场。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PhaseEncoder(nn.Module):
    """将标量相位编码为周期特征。"""

    def __init__(self, harmonics: int) -> None:
        super().__init__()
        self.harmonics = max(int(harmonics), 1)
        self.output_dim = 2 * self.harmonics

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        phase = phase.reshape(-1, 1)
        encoded = []
        for k in range(1, self.harmonics + 1):
            encoded.append(torch.sin(2.0 * math.pi * k * phase))
            encoded.append(torch.cos(2.0 * math.pi * k * phase))
        return torch.cat(encoded, dim=-1)


class PhaseConditionedDeformationField(nn.Module):
    """从空间点和相位预测到标准形状域的位移。"""

    def __init__(self, hidden_dim: int, hidden_layers: int, phase_harmonics: int) -> None:
        super().__init__()
        self.phase_encoder = PhaseEncoder(phase_harmonics)
        in_dim = 3 + self.phase_encoder.output_dim
        layers: list[nn.Module] = []
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 3))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, points: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        encoded_phase = self.phase_encoder(phase)
        features = torch.cat([points, encoded_phase], dim=-1)
        return self.network(features)
