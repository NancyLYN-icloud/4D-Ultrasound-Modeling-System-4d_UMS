"""端到端多周期平均法流水线。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

from ..config import PhaseVolume, PipelineConfig, TemporalModel, ValidationReport
from ..data_acquisition.monitor import UltrasoundMonitor
from ..data_acquisition.free_arm_scan import FreeArmScanner
from ..modeling.interpolation import TemporalInterpolator
from ..modeling.validation import ModelValidator
from ..preprocessing.phase_detection import PhaseDetector
from ..preprocessing.binning import PhaseBinner
from ..preprocessing.pointcloud_builder import build_pointclouds_from_phase_bins
from ..reconstruction.reference import ReferenceVolumeBuilder
from ..reconstruction.registration import NonRigidRegistrar
from ..reconstruction.averaging import PhaseAverager


@dataclass
class PipelineOutput:
    """封装流水线的全部关键产物。"""

    phase_volumes: list[PhaseVolume]
    temporal_model: TemporalModel
    validation_report: ValidationReport


class MulticycleReconstructionPipeline:
    """实现跨时空一致性建模的主流程。"""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.phase_detector = PhaseDetector(config.phase_detection)
        self.binner = PhaseBinner(config.phase_detection)
        self.reference_builder = ReferenceVolumeBuilder()
        self.registrar = NonRigidRegistrar(config.registration)
        self.averager = PhaseAverager(config.averaging)
        self.interpolator = TemporalInterpolator(config.interpolation)
        self.validator = ModelValidator(config.validation)

    def run(self, monitor: UltrasoundMonitor, scanner: FreeArmScanner) -> PipelineOutput:
        """串联执行所有阶段。"""
        # 1. 周期检测
        print("[Pipeline] 开始周期检测")
        features = monitor.extract_feature_trace()  # 将帧序列映射为 Feature-Time 曲线。（把已缓存的所有监测帧转换成一条特征随时间变化的曲线）
        print(f"[Pipeline] 特征点数量: {len(features)}")
        cycles = self.phase_detector.detect_cycles(features) # 检测监测数据的周期，返回周期起点、峰值、终点索引列表
        if cycles:
            durations = [c.end_time - c.start_time for c in cycles]
            avg_duration = float(sum(durations) / len(durations))
            print(f"[Pipeline] 平均周期时长: {avg_duration:.2f}s ({len(cycles)} cycles)")
        else:
            # fallback to assumed cycle length from acquisition config
            avg_duration = float(self.config.acquisition.assumed_cycle)
            print(f"[Pipeline] 未检测到周期，使用假定周期时长: {avg_duration:.2f}s")

        # 2. 相位划分与分箱（使用 PhaseBinner.bin_samples_using_duration）
        print("[Pipeline] 开始相位划分与分箱")
        if not scanner.samples:
            raise RuntimeError("Scanner has no samples to bin")
        # use binner convenience method: returns (phase_bins, cycles)
        phase_bins, scanner_cycles = self.binner.bin_samples_using_duration(scanner.samples, avg_duration, step_seconds=0.5)
        print(f"[Pipeline] 相位分箱完成，共 {len(phase_bins)} 个分箱，基于 {len(scanner_cycles)} 个周期")
        
        # 3. 构建点云
        print("[Pipeline] 构建各相位点云")
        try:
            pc_cfg = self.config.pointcloud
            written = build_pointclouds_from_phase_bins(
                phase_bins,
                pointcloud_config=pc_cfg,
            )
            print(f"[Pipeline] 已导出 {len(written)} 个相位点云")
            print(f"[Pipeline] 点云文件位置: {pc_cfg.out_dir}")
        except Exception as e:
            print(f"[Pipeline] 点云导出失败: {e}")

        # 4. 构建参考网格并执行配准
        print("[Pipeline] 构建参考网格")
        reference = self.reference_builder.build(phase_bins)
        print("[Pipeline] 参考网格构建完成，开始配准")
        try:
            aligned_collections = [self.registrar.register_phase_bin(bin_data, reference) for bin_data in phase_bins]
            print("[Pipeline] 配准完成")
        except Exception as e:
            print(f"[Pipeline] 配准失败，使用未配准体积回退: {e}")
            aligned_collections = [[np.asarray(sample.volume_slice, dtype=float) for sample in bin_data.samples] for bin_data in phase_bins]

        # 5. 加权平均得到相位三维容积
        print("[Pipeline] 开始加权平均")
        phase_volumes = self.averager.average_all(phase_bins, aligned_collections, reference)
        print(f"[Pipeline] 加权平均完成，生成 {len(phase_volumes)} 个相位体积")

        # 6. 插值生成 4D 模型
        print("[Pipeline] 开始时间插值")
        temporal_model = self.interpolator.build(phase_volumes)
        print("[Pipeline] 时间插值完成")
        # 7. 模型验证与报告
        print("[Pipeline] 开始模型验证")
        phase_template = [float(f.value) for f in features]
        validation_report = self.validator.validate(temporal_model, phase_template)
        print("[Pipeline] 模型验证完成")
        return PipelineOutput(phase_volumes=phase_volumes, temporal_model=temporal_model, validation_report=validation_report)
