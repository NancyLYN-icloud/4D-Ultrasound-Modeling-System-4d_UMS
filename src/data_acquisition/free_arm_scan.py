"""自由臂扫描阶段的数据结构与工具。"""
from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np

from ..config import AcquisitionConfig, ScanSample


@dataclass
class ProbePose:
    """探头姿态描述，使用 3D 平移 + 旋转矩阵。"""

    position: np.ndarray  # (3,)
    orientation: np.ndarray  # (3, 3)


class FreeArmScanner:
    """缓冲自由臂扫描数据并确保时间戳精确对齐。"""

    def __init__(self, config: AcquisitionConfig) -> None:
        self.config = config
        self.samples: List[ScanSample] = []

    def record(self, timestamps: Iterable[float], poses: Iterable[ProbePose], volumes: Iterable[np.ndarray]) -> None:
        """记录连续扫描数据。"""
        ts_list = list(timestamps)
        print(f"[Scanner] record: 将记录 {len(ts_list)} 个样本")
        for ts, pose, vol in zip(ts_list, poses, volumes):
            self.samples.append(
                ScanSample(
                    timestamp=float(ts),
                    position=np.asarray(pose.position, dtype=float),
                    orientation=np.asarray(pose.orientation, dtype=float),
                    volume_slice=np.asarray(vol, dtype=float),
                    snr=max(float(np.mean(vol) / (np.std(vol) + 1e-6)), 1e-3),
                )
            )

    def jitter_correct(self) -> None:
        """对时间戳进行毫秒级校正，避免跨模块发生漂移。"""
        precision = self.config.timestamp_precision_ms / 1000.0
        print(f"[Scanner] jitter_correct: 精度 {precision}")
        for sample in self.samples:
            old_ts = sample.timestamp
            sample.timestamp = round(sample.timestamp / precision) * precision
            # optionally print for first few
        print(f"[Scanner] jitter_correct 完成")

    def ingest_frame_sequence(
        self,
        frames: Sequence[np.ndarray],
        timestamps: Sequence[float],
        positions: Sequence[np.ndarray],
        orientations: Sequence[np.ndarray],
        *,
        depth_samples: int = 8,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
        slice_thickness: float = 1.0,
        replicate_depth: bool = True,
    ) -> None:
        """将带位姿的 2D 视频帧转为伪 3D 体素块以对接后续算法。"""
        print(f"[Scanner] ingest_frame_sequence: {len(frames)} 帧，深度样本 {depth_samples}")
        for i, (frame, ts, pos, ori) in enumerate(zip(frames, timestamps, positions, orientations)):
            volume = self._frame_to_volume(
                frame=frame,
                depth_samples=depth_samples,
                pixel_spacing=pixel_spacing,
                slice_thickness=slice_thickness,
                replicate_depth=replicate_depth,
            )
            pose = ProbePose(position=np.asarray(pos, dtype=float), orientation=np.asarray(ori, dtype=float))
            self.samples.append(
                ScanSample(
                    timestamp=float(ts),
                    position=pose.position,
                    orientation=pose.orientation,
                    volume_slice=volume,
                    snr=max(float(np.mean(volume) / (np.std(volume) + 1e-6)), 1e-3),
                )
            )
            if i < 2:
                print(f"  第{i}帧 ts={ts}, snr={self.samples[-1].snr:.3f}")
        self.jitter_correct()

    def _frame_to_volume(
        self,
        *,
        frame: np.ndarray,
        depth_samples: int,
        pixel_spacing: Tuple[float, float],
        slice_thickness: float,
        replicate_depth: bool = True,
    ) -> np.ndarray:
        """简单地将单帧图像沿探头法向复制，构造厚度可控的体素块。"""
        array = np.asarray(frame, dtype=float)
        if array.ndim == 3:  # 彩色帧转灰度，兼容 RGB 视频
            array = np.mean(array, axis=-1)
        array -= array.min()
        array /= (array.max() + 1e-6)
        array *= slice_thickness
        if replicate_depth:
            slab = np.repeat(array[np.newaxis, ...], depth_samples, axis=0)
        else:
            # keep single-slice volume (depth=1)
            slab = array[np.newaxis, ...]
        scaling = np.array([slice_thickness, pixel_spacing[0], pixel_spacing[1]], dtype=float)
        slab = slab / (np.linalg.norm(scaling) + 1e-6)
        return slab

    @staticmethod
    def simulate(config: AcquisitionConfig, grid_shape: Tuple[int, int, int] = (32, 32, 32)) -> "FreeArmScanner":
        """生成带有准周期运动的三维数据，用于算法调试。"""
        print(f"[Scanner] 模拟生成 {grid_shape} 形状，点数 {int(config.scan_duration * config.monitoring_fps)}")
        scanner = FreeArmScanner(config)
        total_points = int(config.scan_duration * config.monitoring_fps)
        timestamps = np.linspace(0.0, config.scan_duration, total_points)
        phase = 2 * np.pi * timestamps / config.assumed_cycle
        for ts, ph in zip(timestamps, phase):
            position = np.array([
                20 * np.sin(0.1 * ph),
                20 * np.cos(0.1 * ph),
                10 * np.sin(0.05 * ph),
            ])
            orientation = np.eye(3)
            xx, yy, zz = np.meshgrid(
                np.linspace(-1, 1, grid_shape[0]),
                np.linspace(-1, 1, grid_shape[1]),
                np.linspace(-1, 1, grid_shape[2]),
                indexing="ij",
            )
            dynamic_field = np.exp(-((xx - 0.2 * np.sin(ph)) ** 2 + (yy - 0.2 * np.cos(ph)) ** 2 + zz**2))
            noise = 0.05 * np.random.randn(*grid_shape)
            volume = dynamic_field + noise
            scanner.samples.append(
                ScanSample(
                    timestamp=float(ts),
                    position=position,
                    orientation=orientation,
                    volume_slice=volume,
                    snr=max(float(np.mean(dynamic_field) / (np.std(noise) + 1e-6)), 1e-3),
                )
            )
        scanner.jitter_correct()
        print("[Scanner] 模拟完成")
        return scanner

    @classmethod
    def from_npz(
        cls,
        config: AcquisitionConfig,
        npz_path: Union[str, PathLike],
        *,
        depth_samples: int = 8,
        pixel_spacing: Tuple[float, float] = (1.0, 1.0),
        slice_thickness: float = 1.0,
    ) -> "FreeArmScanner":
        """从 scanner_sequence.npz 等文件构造扫描器实例。"""
        print(f"[Scanner] 正在从文件加载: {npz_path}")
        data = np.load(npz_path)
        scanner = cls(config)
        scanner.ingest_frame_sequence(
            frames=data["frames"],
            timestamps=data["timestamps"],
            positions=data["positions"],
            orientations=data["orientations"],
            depth_samples=depth_samples,
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
            replicate_depth=False,
        )
        print(f"[Scanner] 从文件加载完成，共 {len(scanner.samples)} 样本")
        return scanner
