"""端到端多周期分箱点云/网格重建流水线。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import DynamicMeshBuildResult, DynamicTimelineMeshBuildResult, PipelineConfig, PointCloudPhaseSummary
from ..data_acquisition.monitor import UltrasoundMonitor
from ..data_acquisition.free_arm_scan import FreeArmScanner
from ..modeling.dynamic_surface_reconstruction import reconstruct_dynamic_meshes_from_pointclouds
from ..modeling.surface_reconstruction import MeshBuildResult, reconstruct_meshes_from_pointclouds
from ..preprocessing.phase_detection import PhaseDetector
from ..preprocessing.binning import PhaseBinner
from ..preprocessing.phase_canonicalization import NonlinearPhaseCanonicalizer
from ..preprocessing.pointcloud_builder import build_pointclouds_from_phase_bins


@dataclass
class PipelineOutput:
    """封装流水线的全部关键产物。"""

    pointcloud_paths: list[Path]
    pointcloud_summaries: list[PointCloudPhaseSummary]
    mesh_results: list[MeshBuildResult]
    dynamic_mesh_results: list[DynamicMeshBuildResult]
    dynamic_timeline_mesh_results: list[DynamicTimelineMeshBuildResult]


class MulticycleReconstructionPipeline:
    """主线：周期检测 -> 相位分箱 -> 相位点云 ->（可选）相位表面网格。"""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.phase_detector = PhaseDetector(config.phase_detection)
        self.binner = PhaseBinner(config.phase_detection)
        self.phase_canonicalizer = NonlinearPhaseCanonicalizer(config.phase_canonicalization)
        # 主线：周期检测 -> 非线性相位标准化 -> 点云 ->（可选）相位网格。

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

        # 2. 相位划分与分箱
        print("[Pipeline] 开始相位划分与分箱")
        if not scanner.samples:
            raise RuntimeError("Scanner has no samples to bin")

        scanner_timestamps = [sample.timestamp for sample in scanner.samples]
        scanner_cycles = self.binner.split_timestamps_into_cycles(scanner_timestamps, avg_duration)
        timeline_phases: list[float]
        if self.config.phase_canonicalization.enabled and cycles and scanner_cycles:
            print("[Pipeline] 开始非线性相位标准化")
            canonical_phases, summary = self.phase_canonicalizer.assign_phases(
                timestamps=scanner_timestamps,
                sample_cycles=scanner_cycles,
                reference_cycles=cycles,
            )
            timeline_phases = canonical_phases
            phase_bins = self.binner.bin_samples_with_phases(
                scanner.samples,
                canonical_phases,
                avg_duration,
                step_seconds=self.config.phase_detection.phase_bin_step_seconds,
            )
            print(
                f"[Pipeline] 非线性相位标准化完成，标准峰值相位={summary.target_peak_phase:.3f}，"
                f"共 {summary.cycle_count} 个扫描周期"
            )
        else:
            print("[Pipeline] 使用线性相位分箱回退")
            phase_bins, scanner_cycles = self.binner.bin_samples_using_duration(scanner.samples, avg_duration)
            timeline_phases = [assignment.normalized_phase for assignment in self.phase_detector.assign_phases(scanner_timestamps, scanner_cycles)]
        print(f"[Pipeline] 相位分箱完成，共 {len(phase_bins)} 个分箱，基于 {len(scanner_cycles)} 个周期")
        
        # 3. 构建点云
        print("[Pipeline] 构建各相位点云")
        written: list[Path] = []
        pointcloud_summaries: list[PointCloudPhaseSummary] = []
        try:
            pc_cfg = self.config.pointcloud
            written, pointcloud_summaries = build_pointclouds_from_phase_bins(
                phase_bins,
                pointcloud_config=pc_cfg,
            )
            print(f"[Pipeline] 已导出 {len(written)} 个相位点云")
            print(f"[Pipeline] 点云文件位置: {pc_cfg.out_dir}")
            if pointcloud_summaries:
                mean_conf = sum(item.mean_confidence for item in pointcloud_summaries) / max(len(pointcloud_summaries), 1)
                print(f"[Pipeline] 平均点云置信度: {mean_conf:.3f}")
        except Exception as e:
            print(f"[Pipeline] 点云导出失败: {e}")

        # 4. 基于各相位点云重建平滑水密网格
        mesh_results: list[MeshBuildResult] = []
        dynamic_mesh_results: list[DynamicMeshBuildResult] = []
        dynamic_timeline_mesh_results: list[DynamicTimelineMeshBuildResult] = []
        if self.config.surface_model.enabled and written:
            print("[Pipeline] 构建各相位表面网格")
            try:
                mesh_results = reconstruct_meshes_from_pointclouds(
                    written,
                    config=self.config.surface_model,
                    phase_bin_step_seconds=self.config.phase_detection.phase_bin_step_seconds,
                )
                mesh_dir = written[0].parent / self.config.surface_model.out_subdir
                print(f"[Pipeline] 已导出 {len(mesh_results)} 个相位网格")
                print(f"[Pipeline] 网格文件位置: {mesh_dir}")
            except Exception as e:
                print(f"[Pipeline] 相位网格重建失败: {e}")
        elif self.config.surface_model.enabled:
            print("[Pipeline] 未生成点云，跳过相位网格重建")

        # 5. （可选）共享动态隐式模型重建
        if self.config.dynamic_model.enabled and written:
            print("[Pipeline] 构建共享动态隐式模型")
            try:
                phase_confidences = {
                    summary.pointcloud_path: float(0.7 * summary.mean_confidence + 0.3 * summary.extracted_slice_ratio)
                    for summary in pointcloud_summaries
                    if summary.pointcloud_path is not None and summary.exported_point_count > 0
                }
                phase_summary_map = {
                    summary.pointcloud_path: summary
                    for summary in pointcloud_summaries
                    if summary.pointcloud_path is not None and summary.exported_point_count > 0
                }
                timeline_samples = [
                    (index, float(sample.timestamp), float(phase_value))
                    for index, (sample, phase_value) in enumerate(zip(scanner.samples, timeline_phases))
                    if not np.isnan(phase_value)
                ]
                dynamic_mesh_results, dynamic_timeline_mesh_results = reconstruct_dynamic_meshes_from_pointclouds(
                    written,
                    config=self.config.dynamic_model,
                    phase_confidences=phase_confidences,
                    phase_summaries=phase_summary_map,
                    timeline_samples=timeline_samples,
                )
                dynamic_mesh_dir = written[0].parent / self.config.dynamic_model.out_subdir
                print(f"[Pipeline] 已导出 {len(dynamic_mesh_results)} 个动态网格")
                print(f"[Pipeline] 动态网格文件位置: {dynamic_mesh_dir}")
                if dynamic_timeline_mesh_results:
                    timeline_dir = written[0].parent / self.config.dynamic_model.timeline_out_subdir
                    print(f"[Pipeline] 已导出 {len(dynamic_timeline_mesh_results)} 个逐帧动态网格")
                    print(f"[Pipeline] 时间轴网格文件位置: {timeline_dir}")
            except Exception as e:
                print(f"[Pipeline] 动态隐式模型重建失败: {e}")
        elif self.config.dynamic_model.enabled:
            print("[Pipeline] 未生成点云，跳过动态隐式模型重建")

        return PipelineOutput(
            pointcloud_paths=written,
            pointcloud_summaries=pointcloud_summaries,
            mesh_results=mesh_results,
            dynamic_mesh_results=dynamic_mesh_results,
            dynamic_timeline_mesh_results=dynamic_timeline_mesh_results,
        )
