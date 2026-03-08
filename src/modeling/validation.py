"""4D 模型验证与指标计算。"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..config import TemporalModel, ValidationConfig, ValidationReport, VolumeDescriptor


class ModelValidator:
    """提供平滑性、周期一致性和临床参数的计算。"""

    def __init__(self, config: ValidationConfig) -> None:
        self.config = config

    def _finite_difference(self, volumes: Sequence[np.ndarray]) -> float:
        diffs = []
        for v0, v1 in zip(volumes[:-1], volumes[1:]):
            diffs.append(np.linalg.norm(v1 - v0) / (np.linalg.norm(v0) + 1e-8))
        return float(np.mean(diffs)) if diffs else 0.0

    def _estimate_cavity_volume(self, volume: np.ndarray, threshold: float = 0.5) -> float:
        mask = volume > threshold * np.max(volume)
        return float(mask.sum())

    def _estimate_velocity(self, cavity_curve: List[Tuple[float, float]]) -> float:
        if len(cavity_curve) < 2:
            return float("nan")
        times = np.array([t for t, _ in cavity_curve])
        volumes = np.array([v for _, v in cavity_curve])
        gradients = np.gradient(volumes, times)
        return float(np.max(np.abs(gradients)))

    def validate(self, model: TemporalModel, reference_curve: Sequence[float]) -> ValidationReport:
        print(f"[Validator] validate: 参考曲线长度 {len(reference_curve)}")
        sample_phases = np.linspace(0.0, 1.0, num=len(reference_curve))
        sampled_volumes: List[np.ndarray] = []
        cavity_curve: List[Tuple[float, float]] = []
        for phase in sample_phases:
            descriptor = model.interpolator(phase)
            sampled_volumes.append(descriptor.intensities)
            cavity_curve.append((phase, self._estimate_cavity_volume(descriptor.intensities)))
        smoothness = self._finite_difference(sampled_volumes)
        print(f"[Validator] 平滑度得分 {smoothness}")
        jitter = float(np.nanstd(reference_curve))
        peristalsis_velocity = self._estimate_velocity(cavity_curve)
        wall_thickening_rate = float(np.max(reference_curve) - np.min(reference_curve))
        print(f"[Validator] 抖动 {jitter}, 蠕动速度 {peristalsis_velocity}, 厚度变化 {wall_thickening_rate}")
        notes = {
            "smoothness_passed": str(smoothness < self.config.smoothness_threshold),
            "phase_jitter_passed": str(jitter < self.config.max_allowed_phase_jitter),
        }
        report = ValidationReport(
            smoothness_score=smoothness,
            cycle_jitter=jitter,
            cavity_volume_curve=cavity_curve,
            peristalsis_velocity=peristalsis_velocity,
            wall_thickening_rate=wall_thickening_rate,
            notes=notes,
        )
        print(f"[Validator] 验证报告生成")
        return report
