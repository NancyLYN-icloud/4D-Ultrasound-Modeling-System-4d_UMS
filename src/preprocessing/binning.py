"""将自由臂扫描数据按相位分箱。"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..config import CycleInfo, PhaseBin, PhaseDetectionConfig, ScanSample
from ..utils import signals


class PhaseBinner:
	"""基于相位归一化结果对 3D 数据进行聚合。

	新增说明的方法：
	- `split_timestamps_into_cycles(timestamps, avg_duration, t0=None)`:
		根据平均周期时长把时间线切分为独立周期（返回 CycleInfo 列表）。
	- `generate_bin_edges_by_time(avg_duration, step_seconds=None)`:
		根据周期时长与相位步长（秒）生成归一化相位的分箱边界（在 [0,1] 上）。
	- `bin_samples_using_duration(samples, avg_duration, step_seconds=None)`:
		将 `samples` 按 `avg_duration` 划分周期、对每个样本计算归一化相位，并按照生成的分箱返回 `PhaseBin` 列表与周期列表。
	"""

	def __init__(self, config: PhaseDetectionConfig) -> None:
		self.config = config
		self.bin_edges = np.asarray(config.bin_edges)
		self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

	def split_timestamps_into_cycles(self, timestamps: Sequence[float], avg_duration: float, t0: float | None = None) -> List[CycleInfo]:
		"""根据 `avg_duration` 将时间戳轴分割为若干个连续周期（CycleInfo 列表）。

		Parameters
		- timestamps: 任意时间戳序列（用于确定总体时段范围）
		- avg_duration: 平均周期时长（秒）
		- t0: 可选的起始时间；若为 None 则使用 timestamps 的最小值
		"""
		if len(timestamps) == 0:
			return []
		arr = np.asarray(timestamps, dtype=float)
		start = float(np.min(arr)) if t0 is None else float(t0)
		end = float(np.max(arr))
		if avg_duration <= 0:
			raise ValueError("avg_duration must be positive")
		span = max(0.0, end - start)
		n_cycles = int(np.ceil((span + 1e-8) / avg_duration)) if end > start else 1
		from ..config import CycleInfo

		cycles: List[CycleInfo] = []
		for i in range(n_cycles):
			s = start + i * avg_duration
			e = min(s + avg_duration, end)
			if e <= s:
				continue
			peak = s + 0.5 * (e - s)
			cycles.append(CycleInfo(index=i, start_time=float(s), peak_time=float(peak), end_time=float(e)))
		if cycles and cycles[-1].duration < avg_duration - 1e-6:
			print(
				"[PhaseBinner] 保留尾部不完整周期: "
				f"末周期 {cycles[-1].duration:.3f}s / 平均周期 {avg_duration:.3f}s"
			)
		return cycles

	def generate_bin_edges_by_time(self, avg_duration: float, step_seconds: float | None = None) -> np.ndarray:
		"""根据周期时长和相位步长（秒）生成归一化相位的分箱边界（闭区间 [0,1]）。

		例如：avg_duration=4s, step_seconds=0.5s => 每 0.5s 一个点，总共 8 个等间隔分段，返回 9 个边界。
		"""
		if step_seconds is None:
			step_seconds = float(self.config.phase_bin_step_seconds)
		if step_seconds <= 0:
			raise ValueError("step_seconds must be positive")
		n_bins = max(1, int(np.ceil(avg_duration / step_seconds)))
		edges = np.linspace(0.0, 1.0, n_bins + 1)
		return edges

	def bin_samples_using_duration(self, samples: Sequence[ScanSample], avg_duration: float, step_seconds: float | None = None) -> tuple[List[PhaseBin], List[CycleInfo]]:
		"""把 `samples`（按时间）基于平均周期时长划分周期、归一化相位，并分箱返回。

		Returns: (phase_bins, cycles)
		- phase_bins: list of `PhaseBin`，每个包含属于该相位窗的 `ScanSample`
		- cycles: 对应用于分箱的 `CycleInfo` 列表
		"""
		if len(samples) == 0:
			return [], []
		timestamps = [s.timestamp for s in samples]
		cycles = self.split_timestamps_into_cycles(timestamps, avg_duration)
		if len(cycles) == 0:
			return [], []

		# create bin edges based on desired temporal step and avg_duration
		if step_seconds is None:
			step_seconds = float(self.config.phase_bin_step_seconds)
		bin_edges = self.generate_bin_edges_by_time(avg_duration, step_seconds)
		bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) 

		# compute normalized phases for each sample using existing signals.assign_phase
		cycle_bounds = [(c.start_time, c.end_time) for c in cycles]
		phases = signals.assign_phase(timestamps, cycle_bounds)

		# build empty bins
		bins: Dict[int, PhaseBin] = {i: PhaseBin(phase_center=float(center)) for i, center in enumerate(bin_centers)}
		for sample, phase_value in zip(samples, phases):
			if np.isnan(phase_value):
				continue
			bin_idx = np.digitize(phase_value, bin_edges) - 1
			bin_idx = int(np.clip(bin_idx, 0, len(bin_centers) - 1))
			bins[bin_idx].samples.append(sample)

		result = [bins[i] for i in range(len(bin_centers))]
		bin_sizes = np.asarray([len(bin_data.samples) for bin_data in result], dtype=np.int32)
		effective_bin_seconds = avg_duration / max(len(result), 1)
		print(
			f"[PhaseBinner] bin_samples_using_duration: samples={len(samples)}, cycles={len(cycles)}, "
			f"avg_duration={avg_duration:.3f}s, step_seconds={step_seconds}, bins={len(result)}, "
			f"effective_bin_seconds={effective_bin_seconds:.3f}s"
		)
		print(
			f"[PhaseBinner] 各bin样本数: min={int(bin_sizes.min())}, max={int(bin_sizes.max())}, "
			f"mean={float(bin_sizes.mean()):.2f}, first10={bin_sizes[:10].tolist()}"
		)
		return result, cycles

	def bin_samples(self, samples: Sequence[ScanSample], cycles: Sequence[CycleInfo]) -> List[PhaseBin]:
		"""输出每个相位窗内的样本集合（使用 config 中的 `bin_edges`）。"""
		print(f"[PhaseBinner] bin_samples: 样本数量 {len(samples)}, 周期数 {len(cycles)}")
		timestamps = [s.timestamp for s in samples]  # 多个 3D 样本的采集时间
		cycle_bounds = [(c.start_time, c.end_time) for c in cycles]  # 来自监测信号的周期时间段
		phases = signals.assign_phase(timestamps, cycle_bounds)  # 把每个样本映射到 [0,1] 相位
		print(f"[PhaseBinner] bin_samples: 计算相位 {phases[:5]} ...")
		bins: Dict[int, PhaseBin] = {i: PhaseBin(phase_center=float(center)) for i, center in enumerate(self.bin_centers)}
		for sample, phase_value in zip(samples, phases):
			if np.isnan(phase_value):
				continue
			bin_idx = np.digitize(phase_value, self.bin_edges) - 1  # 找到相位落入的分箱编号
			bin_idx = int(np.clip(bin_idx, 0, len(self.bin_centers) - 1))
			bins[bin_idx].samples.append(sample)  # 将样本加入对应相位 bin
		result = [bins[i] for i in range(len(self.bin_centers))]
		print(f"[PhaseBinner] bin_samples 完成，各bin大小 {[len(b.samples) for b in result]}")
		return result
