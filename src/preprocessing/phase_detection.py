"""蠕动周期检测与相位归一化。"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..config import CycleInfo, FrameFeature, PhaseAssignment, PhaseDetectionConfig
from ..utils import signals


class PhaseDetector:
    """围绕 2D 监测信号完成周期划分与相位标注。"""

    def __init__(self, config: PhaseDetectionConfig) -> None:
        self.config = config

    def detect_cycles(self, features: Sequence[FrameFeature]) -> List[CycleInfo]:
        """返回所有满足约束的周期。"""
        print(f"[PhaseDetector] detect_cycles: 特征点数量 {len(features)}")
        timestamps = [f.timestamp for f in features]  # 取出每个特征点的时间戳
        values = [f.value for f in features]  # 取出特征值序列（面积/厚度曲线）
        # 自动估计采样率：使用时间戳差的平均倒数
        if len(timestamps) > 1:
            diffs = np.diff(timestamps)
            estimated_rate = 1.0 / (np.mean(diffs) + 1e-8)
        
        print(f"[PhaseDetector] 估计采样率: {estimated_rate:.2f} Hz")
        
        # 自动估计周期范围：基于总时长
        total_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0 
        min_cycle_seconds = total_duration / 10.0  # 假设最多 10 个周期
        max_cycle_seconds = total_duration / 2.0   # 至少 2 个周期
        print(f"[PhaseDetector] 自动周期范围: {min_cycle_seconds:.2f} - {max_cycle_seconds:.2f} s (总时长 {total_duration:.2f} s)")
        
        cycle_bounds = signals.estimate_cycles(
            timestamps=timestamps,
            feature_series=values,
            min_cycle_seconds=min_cycle_seconds,
            max_cycle_seconds=max_cycle_seconds,
            sampling_rate=estimated_rate,
            window_size=self.config.smoothing_window,
            poly_order=self.config.smoothing_poly_order,
        )  # 利用 Savitzky-Golay 平滑 + 峰谷检测，得到周期的起点/峰值/终点索引
        print(f"[PhaseDetector] detect_cycles: 循环边界 {cycle_bounds}")
        cycles: List[CycleInfo] = []
        durations: list[float] = []
        for idx, (start_idx, peak_idx, end_idx) in enumerate(cycle_bounds):
            start_time = timestamps[start_idx]
            peak_time = timestamps[peak_idx]
            end_time = timestamps[end_idx]
            duration = end_time - start_time
            durations.append(duration)
            print(f"[PhaseDetector] 第{idx}周期: start={start_time:.2f}s peak={peak_time:.2f}s end={end_time:.2f}s duration={duration:.2f}s")
            cycles.append(
                CycleInfo(
                    index=idx,  # 周期编号
                    start_time=timestamps[start_idx],  # 周期起点时间
                    peak_time=timestamps[peak_idx],  # 周期峰值时间
                    end_time=timestamps[end_idx],  # 周期终点时间
                )
            )
        if durations:
            avg = float(np.mean(durations))
            print(f"[PhaseDetector] 平均周期时长: {avg:.2f}s")
        print(f"[PhaseDetector] detect_cycles: 生成 CycleInfo 列表，共 {len(cycles)}")
        return cycles

    def assign_phases(self, timestamps: Sequence[float], cycles: Sequence[CycleInfo]) -> List[PhaseAssignment]:
        """将任意时间戳映射到周期编号 + 归一化相位。"""
        cycle_bounds = [(c.start_time, c.end_time) for c in cycles]  # 周期的起止时间列表
        phases = signals.assign_phase(timestamps, cycle_bounds)  # 将时间戳按周期长度线性映射到 [0,1]
        assignments: List[PhaseAssignment] = []
        for ts, phase_value in zip(timestamps, phases):
            cycle_index = -1  # 默认没有匹配到周期
            if not np.isnan(phase_value):
                for cycle in cycles:
                    if cycle.start_time <= ts <= cycle.end_time:
                        cycle_index = cycle.index  # 找到该时间戳属于哪个周期编号
                        break
            assignments.append(PhaseAssignment(cycle_index=cycle_index, normalized_phase=phase_value))
        return assignments

    
        
