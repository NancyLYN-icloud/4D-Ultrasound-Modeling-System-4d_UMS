"""项目配置与核心数据结构。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .paths import data_path


@dataclass
class AcquisitionConfig:
    """监测与自由臂扫描阶段的硬件与任务参数。"""

    monitoring_fps: float = 30.0  # 监测阶段的 2D 帧率 (Hz)
    monitoring_duration: float = 60.0  # 监测阶段持续时间 (s)
    scan_duration: float = 180.0  # 自由臂扫描时长 (s)
    assumed_cycle: float = 20.0  # 假设蠕动周期 (s)
    timestamp_precision_ms: float = 0.5  # 时间戳精度 (ms)


@dataclass
class PhaseDetectionConfig:
    """蠕动周期检测与相位归一化参数。"""

    smoothing_window: int = 9  # 奇数窗口长度
    smoothing_poly_order: int = 3  # Savitzky-Golay 多项式阶数
    min_cycle_seconds: float = 10.0  # 最短周期 (s)
    max_cycle_seconds: float = 30.0  # 最长周期 (s)
    phase_bin_step_seconds: float = 0.3  # 相邻 bin 中心在标准周期上的时间步长 (s)
    binning_strategy: str = "disjoint_phase_bin"  # 默认恢复为旧的离散相位分箱；可选 `sliding_time_window`
    sliding_window_seconds: float = 0.9  # 仅在 `sliding_time_window` 时使用
    bin_edges: Sequence[float] = tuple(np.linspace(0.0, 1.0, 11))


@dataclass
class RegistrationConfig:
    """非刚性配准相关参数。"""

    method: str = "b_spline"
    grid_spacing: Tuple[int, int, int] = (20, 20, 20)
    max_iterations: int = 50
    regularization_lambda: float = 1e-2
    multiscale_levels: int = 3


@dataclass
class AveragingConfig:
    """跨周期体素平均策略。"""

    weighting: str = "snr"
    snr_floor: float = 1e-3
    epsilon: float = 1e-6


@dataclass
class InterpolationConfig:
    """时间插值参数。"""

    method: str = "cubic"
    temporal_resolution: int = 100  # 输出 4D 序列的时间采样点数


@dataclass
class PointCloudConfig:
    """点云导出配置：控制如何从分箱样本生成每相位点云。

    - `pixel_spacing`: 像素间距（mm/pixel），用于将像素坐标映射为物理坐标。
    - `slice_thickness`: 切片厚度（mm），用于深度方向标定。
    - `intensity_threshold`: 强度阈值，可为 'auto'（默认：mean+0.5*std）或数值。
    - `min_contour_area`: 轮廓最小面积（像素），低于该值的轮廓会被丢弃。
    - `sample_spacing`: 轮廓上均匀采样的间距（像素）。
    - `max_points_per_phase`: 每相位导出的最大点数（用于下采样避免内存暴涨）。
    """

    pixel_spacing: float = 0.42
    slice_thickness: float = 1.0
    intensity_threshold: float | str = "auto"
    min_contour_area: float = 50.0
    sample_spacing: float = 2.0
    max_points_per_phase: int | None = 200000
    # 点云输出基础目录（每次运行会在该目录下自动新建一个带序号的子文件夹）
    out_dir: str = field(default_factory=lambda: str(data_path("test", "processed")))


@dataclass
class PhaseCanonicalizationConfig:
    """非线性相位标准化参数。"""

    enabled: bool = True
    default_peak_phase: float = 0.5
    min_peak_phase: float = 0.2
    max_peak_phase: float = 0.8
    template_strategy: str = "median"


@dataclass
class SurfaceModelConfig:
    """相位点云表面重建配置。"""

    enabled: bool = True
    out_subdir: str = "meshes"
    voxel_size: float = 2.0
    max_points: int = 4000
    outlier_neighbors: int = 12
    outlier_std_ratio: float = 2.0
    normal_neighbors: int = 24
    normal_offset_scale: float = 0.03
    hidden_dim: int = 128
    hidden_layers: int = 5
    train_steps: int = 80
    learning_rate: float = 0.001
    surface_batch_size: int = 1024
    eikonal_batch_size: int = 1024
    surface_weight: float = 1.0
    eikonal_weight: float = 0.08
    normal_weight: float = 0.15
    bbox_padding: float = 0.12
    mesh_resolution: int = 56
    mesh_threshold: float = 0.0
    eval_batch_size: int = 24576
    smoothing_iterations: int = 10
    taubin_lambda: float = 0.33
    taubin_mu: float = -0.34
    min_face_count: int = 300
    random_seed: int = 7
    use_cuda_if_available: bool = True


@dataclass
class DynamicModelConfig:
    """时空耦合动态隐式建模配置。"""

    enabled: bool = False
    method: str = "shared_topology_vertex_field"
    out_subdir: str = "dynamic_meshes"
    voxel_size: float = 2.0
    max_points_per_phase: int = 3000
    canonical_hidden_dim: int = 128
    canonical_hidden_layers: int = 4
    deformation_hidden_dim: int = 128
    deformation_hidden_layers: int = 3
    phase_harmonics: int = 4
    train_steps: int = 120
    learning_rate: float = 0.001
    surface_batch_size: int = 1024
    eikonal_batch_size: int = 1024
    temporal_batch_size: int = 1024
    normal_batch_size: int = 1024
    eikonal_weight: float = 0.05
    normal_weight: float = 0.15
    temporal_weight: float = 0.1
    temporal_acceleration_weight: float = 0.05
    phase_consistency_weight: float = 0.05
    correspondence_temporal_weight: float = 0.0
    correspondence_acceleration_weight: float = 0.0
    correspondence_phase_consistency_weight: float = 0.0
    correspondence_start_fraction: float = 0.0
    correspondence_ramp_fraction: float = 0.0
    periodicity_weight: float = 0.1
    deformation_weight: float = 0.01
    temporal_delta_phase: float = 0.05
    confidence_floor: float = 0.2
    supervision_binning_strategy: str = "disjoint_phase_bin"
    supervision_step_seconds: float | None = None
    supervision_window_seconds: float | None = None
    overlap_aware_sampling: bool = False
    overlap_loss_min_scale: float = 0.5
    overlap_neighbor_stride_bins: int | None = None
    spatial_smoothness_weight: float = 0.05
    centroid_weight: float = 0.05
    base_mesh_train_steps: int | None = None
    base_mesh_out_subdir: str = "_dynamic_base_mesh_seed"
    bbox_padding: float = 0.12
    mesh_resolution: int = 56
    mesh_threshold: float = 0.0
    eval_batch_size: int = 24576
    min_face_count: int = 300
    export_timeline_meshes: bool = False
    timeline_out_subdir: str = "dynamic_timeline_meshes"
    timeline_stride: int = 1
    timeline_max_exports: int | None = None
    random_seed: int = 7
    use_cuda_if_available: bool = True


@dataclass
class ValidationConfig:
    """模型验证需要的设置。"""

    smoothness_threshold: float = 0.02
    max_allowed_phase_jitter: float = 0.05


@dataclass
class PipelineConfig:
    """整条流水线的统一配置入口。"""

    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    phase_detection: PhaseDetectionConfig = field(default_factory=PhaseDetectionConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    averaging: AveragingConfig = field(default_factory=AveragingConfig)
    interpolation: InterpolationConfig = field(default_factory=InterpolationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    phase_canonicalization: PhaseCanonicalizationConfig = field(default_factory=PhaseCanonicalizationConfig)
    surface_model: SurfaceModelConfig = field(default_factory=SurfaceModelConfig)
    dynamic_model: DynamicModelConfig = field(default_factory=DynamicModelConfig)


@dataclass
class FrameFeature:
    """用于周期检测的 2D 帧特征 (如胃窦面积)。"""

    timestamp: float
    value: float


@dataclass
class CycleInfo:
    """蠕动周期的结构化描述。"""

    index: int
    start_time: float
    peak_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class PhaseAssignment:
    """某个时间点映射到的周期编号与相位。"""

    cycle_index: int
    normalized_phase: float


@dataclass
class ScanSample:
    """自由臂扫描过程中采集的单个 3D 样本。"""

    timestamp: float
    position: np.ndarray  # 3D 空间坐标 (x, y, z)
    orientation: np.ndarray  # 3x3 姿态矩阵
    volume_slice: np.ndarray  # 对应的 3D 体素块或 2.5D 切片
    snr: float = 1.0


@dataclass
class PhaseBin:
    """相位分箱中聚合的数据点。"""

    phase_center: float
    samples: List[ScanSample] = field(default_factory=list)


@dataclass
class PointCloudPhaseSummary:
    """单个相位点云构建阶段的统计摘要。"""

    phase_index: int
    phase_center: float
    sample_count: int
    raw_point_count: int
    exported_point_count: int
    mean_confidence: float
    mean_sample_snr: float
    extracted_slice_ratio: float
    pointcloud_path: Optional[Path] = None


@dataclass
class DynamicMeshBuildResult:
    """动态隐式模型导出的单相位网格摘要。"""

    phase: float
    mesh_path: Path
    vertices: int
    faces: int
    watertight: bool
    method: str


@dataclass
class DynamicTimelineMeshBuildResult:
    """动态隐式模型导出的逐帧时间轴网格摘要。"""

    frame_index: int
    timestamp: float
    phase: float
    mesh_path: Path
    vertices: int
    faces: int
    watertight: bool
    method: str


@dataclass
class VolumeDescriptor:
    """用于描述 3D 容积数据及其元信息。"""

    grid: np.ndarray  # shape (nx, ny, nz, 3)
    intensities: np.ndarray  # shape (nx, ny, nz)


@dataclass
class PhaseVolume:
    """平均后的相位三维容积。"""

    phase: float
    volume: VolumeDescriptor


@dataclass
class ValidationReport:
    """4D 模型验证输出。"""

    smoothness_score: float
    cycle_jitter: float
    cavity_volume_curve: List[Tuple[float, float]]
    peristalsis_velocity: Optional[float]
    wall_thickening_rate: Optional[float]
    notes: Dict[str, str] = field(default_factory=dict)


@dataclass
class TemporalModel:
    """4D 动态模型的抽象表示。"""

    phases: List[float]
    volumes: List[VolumeDescriptor]
    interpolator: Callable[[float], VolumeDescriptor]
