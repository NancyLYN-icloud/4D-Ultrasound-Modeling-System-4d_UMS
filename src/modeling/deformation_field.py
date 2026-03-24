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
        linear_layers = [module for module in self.network if isinstance(module, nn.Linear)]
        for module in linear_layers[:-1]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
        if linear_layers:
            nn.init.constant_(linear_layers[-1].weight, 0.0)
            nn.init.constant_(linear_layers[-1].bias, 0.0)

    def forward(self, points: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        encoded_phase = self.phase_encoder(phase)
        features = torch.cat([points, encoded_phase], dim=-1)
        return self.network(features)


class PhaseConditionedBasisCoefficients(nn.Module):
    """Predict low-rank global motion coefficients from phase only."""

    def __init__(self, basis_rank: int, phase_harmonics: int) -> None:
        super().__init__()
        self.phase_encoder = PhaseEncoder(phase_harmonics)
        self.linear = nn.Linear(self.phase_encoder.output_dim, int(basis_rank))
        nn.init.constant_(self.linear.weight, 0.0)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        encoded_phase = self.phase_encoder(phase)
        return self.linear(encoded_phase)


class PhaseConditionedSDFField(nn.Module):
    """Direct spatiotemporal SDF without explicit deformation priors."""

    def __init__(self, hidden_dim: int, hidden_layers: int, phase_harmonics: int) -> None:
        super().__init__()
        self.phase_encoder = PhaseEncoder(phase_harmonics)
        in_dim = 3 + self.phase_encoder.output_dim
        layers: list[nn.Module] = []
        for _ in range(hidden_layers):
            linear = nn.Linear(in_dim, hidden_dim)
            nn.init.normal_(linear.weight, mean=0.0, std=math.sqrt(2.0 / max(hidden_dim, 1)))
            nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)
            layers.append(nn.Softplus(beta=100.0))
            in_dim = hidden_dim
        final = nn.Linear(in_dim, 1)
        nn.init.normal_(final.weight, mean=math.sqrt(math.pi) / math.sqrt(max(in_dim, 1)), std=1e-4)
        nn.init.constant_(final.bias, -0.1)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        encoded_phase = self.phase_encoder(phase)
        features = torch.cat([points, encoded_phase], dim=-1)
        return self.network(features)

    def sdf_and_gradient(self, points: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        points = points.requires_grad_(True)
        sdf = self.forward(points, phase)
        grad = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return sdf, grad


class DecoupledMotionLatentField(nn.Module):
    """Motion network with per-phase latent codes, adapted from the decoupled myocardium model."""

    def __init__(self, latent_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        input_dim = 4 + self.latent_dim
        self.activation = nn.LeakyReLU(0.2)
        self.fc0 = nn.Linear(input_dim, hidden_dim * 16)
        self.fc1 = nn.Linear(input_dim + hidden_dim * 16, hidden_dim * 8)
        self.fc2 = nn.Linear(input_dim + hidden_dim * 8, hidden_dim * 4)
        self.fc3 = nn.Linear(input_dim + hidden_dim * 4, hidden_dim * 2)
        self.fc4 = nn.Linear(input_dim + hidden_dim * 2, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 3)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.fc0, self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
        nn.init.constant_(self.fc5.weight, 0.0)
        nn.init.constant_(self.fc5.bias, 0.0)

    def forward(self, points: torch.Tensor, phase: torch.Tensor, latent_code: torch.Tensor) -> torch.Tensor:
        if phase.dim() == 1:
            phase = phase.unsqueeze(-1)
        base_input = torch.cat([points, phase, latent_code], dim=-1)
        hidden = self.activation(self.fc0(base_input))
        hidden = torch.cat([hidden, base_input], dim=-1)
        hidden = self.activation(self.fc1(hidden))
        hidden = torch.cat([hidden, base_input], dim=-1)
        hidden = self.activation(self.fc2(hidden))
        hidden = torch.cat([hidden, base_input], dim=-1)
        hidden = self.activation(self.fc3(hidden))
        hidden = torch.cat([hidden, base_input], dim=-1)
        hidden = self.activation(self.fc4(hidden))
        return self.fc5(hidden)


class ShapeLatentField(nn.Module):
    """Global shape network with a single latent code for canonical-shape refinement."""

    def __init__(self, latent_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        input_dim = 3 + self.latent_dim
        self.activation = nn.LeakyReLU(0.2)
        self.fc0 = nn.Linear(input_dim, hidden_dim * 16)
        self.fc1 = nn.Linear(input_dim + hidden_dim * 16, hidden_dim * 8)
        self.fc2 = nn.Linear(input_dim + hidden_dim * 8, hidden_dim * 4)
        self.fc3 = nn.Linear(input_dim + hidden_dim * 4, hidden_dim * 2)
        self.fc4 = nn.Linear(input_dim + hidden_dim * 2, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 3)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.fc0, self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
        nn.init.constant_(self.fc5.weight, 0.0)
        nn.init.constant_(self.fc5.bias, 0.0)

    def forward(self, points: torch.Tensor, latent_code: torch.Tensor) -> torch.Tensor:
        base_input = torch.cat([points, latent_code], dim=-1)
        hidden = self.activation(self.fc0(base_input))
        hidden = torch.cat([hidden, base_input], dim=-1)
        hidden = self.activation(self.fc1(hidden))
        hidden = torch.cat([hidden, base_input], dim=-1)
        hidden = self.activation(self.fc2(hidden))
        hidden = torch.cat([hidden, base_input], dim=-1)
        hidden = self.activation(self.fc3(hidden))
        hidden = torch.cat([hidden, base_input], dim=-1)
        hidden = self.activation(self.fc4(hidden))
        return self.fc5(hidden)
