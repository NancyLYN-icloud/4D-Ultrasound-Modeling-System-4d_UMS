"""全局参考网格构建。"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from ..config import PhaseBin, VolumeDescriptor


class ReferenceVolumeBuilder:
    """以最具代表性的周期为模板构建 V_ref。"""

    def __init__(self, grid_shape: Tuple[int, int, int] = (64, 64, 64)) -> None:
        self.grid_shape = grid_shape  # 参考网格的体素分辨率

    def _compute_bounds(self, samples: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        positions = np.array(samples)
        return positions.min(axis=0), positions.max(axis=0)  # 计算采样点的最小/最大坐标，用于构建网格范围

    def build(self, bin_data: Sequence[PhaseBin]) -> VolumeDescriptor:
        """从一个相位充足的 bin 中提取模板。"""
        print(f"[Reference] build: 接收 {len(bin_data)} 个 bin")
        # 选取样本最多的 bin，视为代表性周期
        richest = max(bin_data, key=lambda b: len(b.samples))
        print(f"[Reference] 选取样本最多的 bin, 样本数 {len(richest.samples)}")
        if not richest.samples:
            grid = np.zeros((*self.grid_shape, 3), dtype=float)
            volume = np.zeros(self.grid_shape, dtype=float)
            return VolumeDescriptor(grid=grid, intensities=volume)
        positions = [s.position for s in richest.samples]
        p_min, p_max = self._compute_bounds(positions)
        axes = [np.linspace(p_min[i], p_max[i], self.grid_shape[i]) for i in range(3)]  # 在每个轴向上均匀采样
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)  # 构建规则的三维坐标网格
        volume = np.zeros(self.grid_shape, dtype=float)
        counts = np.zeros(self.grid_shape, dtype=float)
        for sample in richest.samples:
            # 将 sample 的平均强度映射到最近网格点
            intensity = float(np.mean(sample.volume_slice))
            idx = [
                int(np.clip(round((sample.position[i] - p_min[i]) / (p_max[i] - p_min[i] + 1e-6) * (self.grid_shape[i] - 1)), 0, self.grid_shape[i] - 1))
                for i in range(3)
            ]
            volume[tuple(idx)] += intensity
            counts[tuple(idx)] += 1
        counts[counts == 0] = 1  # 避免除零
        volume /= counts  # 取平均，得到该网格点的代表性强度
        print(f"[Reference] 参考体积构建完成")
        return VolumeDescriptor(grid=grid, intensities=volume)
