"""2D 高帧率监测阶段的数据抽象。"""
from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np

from ..config import AcquisitionConfig, FrameFeature
import torch
import torchvision
from torchvision.models import resnet18

@dataclass
class MonitorFrame:
    """监测阶段单帧数据及其时间戳。"""

    timestamp: float
    image: np.ndarray


class UltrasoundMonitor:
    """负责记录监测阶段的帧序列并提取周期特征。"""

    def __init__(self, config: AcquisitionConfig, roi_mask: Optional[np.ndarray] = None) -> None:
        self.config = config
        self.roi_mask = roi_mask
        self.frames: List[MonitorFrame] = []
        self.cached_feature_trace: Optional[np.ndarray] = None
        
        # 在 __init__ 中初始化 CNN 模型
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改第一层适配单通道
        self.cnn.fc = torch.nn.Identity()  # 去掉分类头，用于特征提取
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn.to(self.device)
        self.cnn.eval()  # 推理模式

    def record(self, frames: Iterable[np.ndarray], timestamps: Iterable[float]) -> None:
        """将外部采集的帧流入列入缓冲区。"""
        print(f"[Monitor] 记录 {len(list(frames))} 帧数据")
        for img, ts in zip(frames, timestamps):
            self.frames.append(MonitorFrame(timestamp=ts, image=img))

    @classmethod
    def from_npz(
        cls,
        config: AcquisitionConfig,
        npz_path: Union[str, PathLike],
        roi_mask: Optional[np.ndarray] = None,
    ) -> "UltrasoundMonitor":
        """直接从 monitor_stream.npz 等模拟好的数据文件中构造 Monitor。"""
        print(f"[Monitor] 从文件加载: {npz_path}")
        data = np.load(npz_path)
        monitor = cls(config, roi_mask=roi_mask)
        monitor.record(data["frames"], data["timestamps"])
        if "feature_trace" in data.files:
            monitor.cached_feature_trace = np.asarray(data["feature_trace"], dtype=np.float32)
        print(f"[Monitor] 加载完成，共 {len(monitor.frames)} 帧")
        return monitor
    
    #  计算监视器特征
    def _compute_roi_feature(self, image: np.ndarray) -> float:
        """
        使用 CNN 提取 ROI 区域的嵌入特征（平均池化后取均值）
        image: 单帧图像，形状 (H, W)，类型为 np.ndarray
        """
        # 转为 tensor: 1×1×H×W (batch=1, channel=1)
        tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():  # 推理无需梯度
            feat = self.cnn(tensor)  # 输出: [1, C] 特征向量

        # 返回特征均值（或可改为返回整个特征向量）
        return float(feat.mean().item())
    
    def extract_feature_trace(self) -> List[FrameFeature]:  # 返回值是 FrameFeature 对象的列表，用于存储“时间戳 + 特征值”对
        """将帧序列映射为 Feature-Time 曲线。（把已缓存的所有监测帧转换成一条特征随时间变化的曲线）"""
        print(f"[Monitor] 开始提取特征轨迹，共 {len(self.frames)} 帧")
        if self.cached_feature_trace is not None and len(self.cached_feature_trace) == len(self.frames):
            print("[Monitor] 使用 NPZ 中缓存的 feature_trace")
            features = [
                FrameFeature(timestamp=frame.timestamp, value=float(value))
                for frame, value in zip(self.frames, self.cached_feature_trace)
            ]
            for i, feature in enumerate(features):
                if i < 24 or i == len(features) - 1:
                    print(f"  帧{i} 时间戳{feature.timestamp} 特征值 {feature.value:.4f}")
            print(f"[Monitor] 特征轨迹提取完成，共 {len(features)} 个点")
            return features

        features: List[FrameFeature] = []
        for i, frame in enumerate(self.frames):
            value = self._compute_roi_feature(frame.image) #计算特征值
            if i < 24 or i == len(self.frames) - 1:
                print(f"  帧{i} 时间戳{frame.timestamp} 特征值 {value:.4f}")
            features.append(FrameFeature(timestamp=frame.timestamp, value=value))
        print(f"[Monitor] 特征轨迹提取完成，共 {len(features)} 个点")
        return features

    @staticmethod
    def simulate(config: AcquisitionConfig, roi_size: int = 64) -> "UltrasoundMonitor":
        """方便研究阶段的仿真器：生成准周期信号（多个带时间戳的2D图像帧）。"""
        print(f"[Monitor] 模拟生成 {int(config.monitoring_duration * config.monitoring_fps)} 帧")
        monitor = UltrasoundMonitor(config)
        total_frames = int(config.monitoring_duration * config.monitoring_fps)
        timestamps = np.linspace(0.0, config.monitoring_duration, total_frames)
        base_cycle = 2 * np.pi * timestamps / config.assumed_cycle
        wave = 0.5 * (1 + np.sin(base_cycle) + 0.1 * np.random.randn(total_frames))
        for value, ts in zip(wave, timestamps):
            phantom_img = np.ones((roi_size, roi_size), dtype=float) * value
            monitor.frames.append(MonitorFrame(timestamp=float(ts), image=phantom_img))
        print(f"[Monitor] 模拟完成")
        return monitor
