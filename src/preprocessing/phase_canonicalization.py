"""非线性相位标准化。

当前实现采用基于周期峰值位置的单调分段线性 warp：
- 对监测周期提取每个周期的峰值相位位置
- 用稳健统计量估计标准相位域中的目标峰值位置
- 将扫描样本的线性相位映射到标准相位域

这比简单的线性归一化更能适配不同周期中的非对称收缩/舒张速度。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import CycleInfo, PhaseCanonicalizationConfig


@dataclass
class CanonicalizationSummary:
    """相位标准化摘要，用于日志和后续论文分析。"""

    target_peak_phase: float
    reference_peak_phases: list[float]
    cycle_count: int


class NonlinearPhaseCanonicalizer:
    """基于周期峰值位置执行单调相位标准化。"""

    def __init__(self, config: PhaseCanonicalizationConfig) -> None:
        self.config = config

    def _clip_peak_phase(self, value: float) -> float:
        lower = float(self.config.min_peak_phase)
        upper = float(self.config.max_peak_phase)
        if upper <= lower:
            return float(self.config.default_peak_phase)
        return float(np.clip(value, lower, upper))

    def _cycle_peak_phase(self, cycle: CycleInfo) -> float:
        duration = max(float(cycle.duration), 1e-8)
        raw = (float(cycle.peak_time) - float(cycle.start_time)) / duration
        return self._clip_peak_phase(raw)

    def estimate_target_peak_phase(self, reference_cycles: Sequence[CycleInfo]) -> float:
        """估计标准相位域中的峰值位置。"""
        if not reference_cycles:
            return self._clip_peak_phase(float(self.config.default_peak_phase))

        peaks = np.asarray([self._cycle_peak_phase(cycle) for cycle in reference_cycles], dtype=float)
        strategy = str(self.config.template_strategy).lower()
        if strategy == "mean":
            target = float(np.mean(peaks))
        else:
            target = float(np.median(peaks))
        return self._clip_peak_phase(target)

    def _map_cycle_index(self, sample_index: int, sample_cycle_count: int, reference_cycle_count: int) -> int:
        if reference_cycle_count <= 1 or sample_cycle_count <= 1:
            return 0
        ratio = sample_index / max(sample_cycle_count - 1, 1)
        mapped = int(round(ratio * (reference_cycle_count - 1)))
        return int(np.clip(mapped, 0, reference_cycle_count - 1))

    @staticmethod
    def _piecewise_warp(phase: float, source_peak: float, target_peak: float) -> float:
        source_peak = float(np.clip(source_peak, 1e-4, 1.0 - 1e-4))
        target_peak = float(np.clip(target_peak, 1e-4, 1.0 - 1e-4))
        phase = float(np.clip(phase, 0.0, 1.0))
        if phase <= source_peak:
            return float(target_peak * phase / source_peak)
        return float(target_peak + (1.0 - target_peak) * (phase - source_peak) / (1.0 - source_peak))

    def assign_phases(
        self,
        timestamps: Sequence[float],
        sample_cycles: Sequence[CycleInfo],
        reference_cycles: Sequence[CycleInfo],
    ) -> tuple[list[float], CanonicalizationSummary]:
        """将扫描时间戳映射到标准相位域。"""
        target_peak = self.estimate_target_peak_phase(reference_cycles)
        reference_peak_phases = [self._cycle_peak_phase(cycle) for cycle in reference_cycles]
        summary = CanonicalizationSummary(
            target_peak_phase=target_peak,
            reference_peak_phases=reference_peak_phases,
            cycle_count=len(sample_cycles),
        )

        if not sample_cycles:
            return [float("nan") for _ in timestamps], summary

        phases: list[float] = []
        cycle_index = 0
        for timestamp in timestamps:
            while cycle_index < len(sample_cycles) and timestamp > sample_cycles[cycle_index].end_time:
                cycle_index += 1
            if cycle_index >= len(sample_cycles):
                phases.append(float("nan"))
                continue

            cycle = sample_cycles[cycle_index]
            if timestamp < cycle.start_time:
                phases.append(float("nan"))
                continue

            duration = max(float(cycle.duration), 1e-8)
            linear_phase = float(np.clip((float(timestamp) - float(cycle.start_time)) / duration, 0.0, 1.0))
            if reference_peak_phases:
                mapped_index = self._map_cycle_index(cycle_index, len(sample_cycles), len(reference_peak_phases))
                source_peak = reference_peak_phases[mapped_index]
            else:
                source_peak = target_peak
            phases.append(self._piecewise_warp(linear_phase, source_peak, target_peak))

        print(
            "[PhaseCanonicalizer] 非线性相位标准化: "
            f"sample_cycles={len(sample_cycles)}, reference_cycles={len(reference_peak_phases)}, "
            f"target_peak_phase={target_peak:.3f}"
        )
        if reference_peak_phases:
            preview = ", ".join(f"{value:.3f}" for value in reference_peak_phases[:5])
            print(f"[PhaseCanonicalizer] 参考峰值相位(前5个): {preview}")
        return phases, summary
