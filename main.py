"""示例入口：串联运行多周期分箱点云/网格流水线。"""
from __future__ import annotations

import numpy as np

from src.config import PipelineConfig
from src.data_acquisition.monitor import UltrasoundMonitor
from src.data_acquisition.free_arm_scan import FreeArmScanner
from src.pipelines.multicycle_reconstruction import MulticycleReconstructionPipeline
from src.paths import data_path


def main() -> None:
    """研究阶段示例：使用仿真数据验证流程连通性。"""
    config = PipelineConfig()
    config.dynamic_model.enabled = True
    config.dynamic_model.export_timeline_meshes = True
    config.dynamic_model.timeline_stride = 12
    config.dynamic_model.timeline_max_exports = 60
    #monitor = UltrasoundMonitor.simulate(config.acquisition) #应用仿真数据时启用此行
    monitor = UltrasoundMonitor.from_npz(config.acquisition, str(data_path("benchmark", "monitor_stream.npz"))) #使用预录制数据时启用此行
    print(f"加载监测器，包含 {len(monitor.frames)} 帧") 
    if monitor.frames:
        print(f"  第一帧形状: {monitor.frames[0].image.shape}")
        # print(f"  第一帧唯一值: {np.unique(monitor.frames[0].image)}")
        # print(f"  第一帧均值: {np.mean(monitor.frames[0].image):.2f}, 标准差: {np.std(monitor.frames[0].image):.2f}")
    
    #scanner = FreeArmScanner.simulate(config.acquisition) #应用仿真数据时启用此行
    scanner = FreeArmScanner.from_npz(config.acquisition, str(data_path("benchmark", "scanner_sequence.npz"))) #使用预录制数据时启用此行
    print(f"加载扫描器，包含 {len(scanner.samples)} 个样本")
    if scanner.samples:
        print(f"  第一个样本体积形状: {scanner.samples[0].volume_slice.shape}")
        # print(f"  第一个样本体积唯一值: {np.unique(scanner.samples[0].volume_slice)}")
        # print(f"  第一个样本体积均值: {np.mean(scanner.samples[0].volume_slice):.2f}, 标准差: {np.std(scanner.samples[0].volume_slice):.2f}")
        print(f"  第一个样本位置: {scanner.samples[0].position}")
        #print(f"  第一个样本方向:\n{scanner.samples[0].orientation}")
    
    # 基于实际数据计算设置
    print("实际数据设置：")
    if monitor.frames:
        monitor_duration = monitor.frames[-1].timestamp - monitor.frames[0].timestamp
        monitor_fps = len(monitor.frames) / monitor_duration if monitor_duration > 0 else 0
        print(f"  监测 FPS: {monitor_fps:.2f}")
        print(f"  监测持续时间: {monitor_duration:.2f} s")
    if scanner.samples:
        scan_duration = scanner.samples[-1].timestamp - scanner.samples[0].timestamp
        print(f"  扫描持续时间: {scan_duration:.2f} s")
    print(f"  假设周期: {config.acquisition.assumed_cycle} s")
    print(f"  时间戳精度毫秒: {config.acquisition.timestamp_precision_ms}")
    
    pipeline = MulticycleReconstructionPipeline(config)
    print("开始运行流水线...")
    output = pipeline.run(monitor, scanner)
    print("流水线运行完成")

    print(f"相位点云数量: {len(output.pointcloud_paths)}")
    if output.pointcloud_paths:
        print(f"  第一个点云文件: {output.pointcloud_paths[0]}")
    if output.pointcloud_summaries:
        valid_summaries = [item for item in output.pointcloud_summaries if item.exported_point_count > 0]
        if valid_summaries:
            mean_confidence = sum(item.mean_confidence for item in valid_summaries) / len(valid_summaries)
            mean_slice_ratio = sum(item.extracted_slice_ratio for item in valid_summaries) / len(valid_summaries)
            print(f"  平均点云置信度: {mean_confidence:.3f}")
            print(f"  平均切片提取率: {mean_slice_ratio:.3f}")

    print(f"相位网格数量: {len(output.mesh_results)}")
    if output.mesh_results:
        watertight_ratio = sum(int(r.watertight) for r in output.mesh_results) / max(len(output.mesh_results), 1)
        print(f"  水密网格比例: {watertight_ratio:.2%}")
        first = output.mesh_results[0]
        print(f"  第一个网格文件: {first.mesh_path}")
        print(f"  faces={first.faces}, verts={first.vertices}, method={first.method}")

    print(f"动态网格数量: {len(output.dynamic_mesh_results)}")
    if output.dynamic_mesh_results:
        first_dynamic = output.dynamic_mesh_results[0]
        print(f"  第一个动态网格文件: {first_dynamic.mesh_path}")
        print(f"  phase={first_dynamic.phase:.3f}, faces={first_dynamic.faces}, verts={first_dynamic.vertices}")
    print(f"逐帧动态网格数量: {len(output.dynamic_timeline_mesh_results)}")
    if output.dynamic_timeline_mesh_results:
        first_frame = output.dynamic_timeline_mesh_results[0]
        print(f"  第一个逐帧网格文件: {first_frame.mesh_path}")
        print(
            f"  frame={first_frame.frame_index}, ts={first_frame.timestamp:.3f}s, "
            f"phase={first_frame.phase:.3f}, faces={first_frame.faces}, verts={first_frame.vertices}"
        )


if __name__ == "__main__":
    main()
