"""跨周期体素平均。"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..config import AveragingConfig, PhaseBin, PhaseVolume, VolumeDescriptor


class PhaseAverager:
    """对齐后的相位数据执行信噪比加权平均。"""

    def __init__(self, config: AveragingConfig) -> None:
        self.config = config

    def _compute_weights(self, bin_data: PhaseBin) -> np.ndarray:
        weights = np.array([max(sample.snr, self.config.snr_floor) for sample in bin_data.samples], dtype=float)
        weights /= weights.sum() + self.config.epsilon
        return weights

    def average_bin(self, bin_data: PhaseBin, aligned_volumes: Sequence[np.ndarray], reference: VolumeDescriptor) -> PhaseVolume:
        print(f"[Averager] average_bin: 相位 {bin_data.phase_center}, 样本数 {len(bin_data.samples)}, 对齐结果 {len(aligned_volumes)}")
        if not aligned_volumes:
            empty = np.zeros_like(reference.intensities)
            return PhaseVolume(phase=bin_data.phase_center, volume=VolumeDescriptor(grid=reference.grid, intensities=empty))
        stacked = np.stack(aligned_volumes, axis=0)
        weights = self._compute_weights(bin_data)
        weights = weights[: stacked.shape[0]]  # 防止样本数和对齐结果不一致
        weights /= weights.sum() + self.config.epsilon
        averaged = np.tensordot(weights, stacked, axes=(0, 0))
        print(f"[Averager] average_bin 完成")
        return PhaseVolume(
            phase=bin_data.phase_center,
            volume=VolumeDescriptor(grid=reference.grid, intensities=averaged),
        )

    def average_all(self, bins: Sequence[PhaseBin], aligned_collections: Sequence[Sequence[np.ndarray]], reference: VolumeDescriptor) -> List[PhaseVolume]:
        results: List[PhaseVolume] = []
        for bin_data, aligned in zip(bins, aligned_collections):
            results.append(self.average_bin(bin_data, aligned, reference))
        print(f"[Averager] average_all 完成，总共 {len(results)} 个相位体积")
        return results
