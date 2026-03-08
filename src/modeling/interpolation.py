"""将离散相位体积插值为连续 4D 序列。"""
from __future__ import annotations

from typing import Callable, List

import numpy as np

from ..config import InterpolationConfig, PhaseVolume, TemporalModel, VolumeDescriptor

try:  # pragma: no cover - 可选依赖
    from scipy.interpolate import CubicSpline
except Exception:  # pragma: no cover
    CubicSpline = None


class TemporalInterpolator:
    """以相位为自变量进行体素级插值。"""

    def __init__(self, config: InterpolationConfig) -> None:
        self.config = config

    def _build_linear(
        self,
        phases: np.ndarray,
        volumes: np.ndarray,
        grid: np.ndarray,
    ) -> Callable[[float], VolumeDescriptor]:
        def interpolate(query_phase: float) -> VolumeDescriptor:
            wrapped_phase = query_phase % 1.0
            idx = np.searchsorted(phases, wrapped_phase)
            idx0 = (idx - 1) % len(phases)
            idx1 = idx % len(phases)
            phase0, phase1 = phases[idx0], phases[idx1]
            if phase0 == phase1:
                alpha = 0.0
            else:
                # 处理循环相位
                delta = (phase1 - phase0) % 1.0
                offset = (wrapped_phase - phase0) % 1.0
                alpha = offset / (delta + 1e-8)
            volume = (1 - alpha) * volumes[idx0] + alpha * volumes[idx1]
            return VolumeDescriptor(grid=grid, intensities=volume)

        return interpolate

    def _build_cubic(
        self,
        phases: np.ndarray,
        volumes: np.ndarray,
        grid: np.ndarray,
    ) -> Callable[[float], VolumeDescriptor]:
        flattened = volumes.reshape(volumes.shape[0], -1)
        spline = CubicSpline(np.append(phases, phases[0] + 1.0), np.vstack([flattened, flattened[0]]), bc_type="periodic")

        def interpolate(query_phase: float) -> VolumeDescriptor:
            wrapped_phase = query_phase % 1.0
            values = spline(wrapped_phase)
            volume = values.reshape(volumes.shape[1:])
            return VolumeDescriptor(grid=grid, intensities=volume)

        return interpolate

    def build(self, phase_volumes: List[PhaseVolume]) -> TemporalModel:
        print(f"[Interpolator] build: 相位体积数量 {len(phase_volumes)}")
        phases = np.array([pv.phase for pv in phase_volumes], dtype=float)
        sort_idx = np.argsort(phases)
        phases = phases[sort_idx]
        volumes = np.stack([phase_volumes[i].volume.intensities for i in sort_idx], axis=0)
        grid = phase_volumes[sort_idx[0]].volume.grid
        if self.config.method == "cubic" and CubicSpline is not None and len(phase_volumes) >= 4:
            print("[Interpolator] 使用三次样条插值")
            interpolator = self._build_cubic(phases, volumes, grid)
        else:
            print("[Interpolator] 使用线性插值")
            interpolator = self._build_linear(phases, volumes, grid)
        return TemporalModel(phases=list(phases), volumes=[phase_volumes[i].volume for i in sort_idx], interpolator=interpolator)
