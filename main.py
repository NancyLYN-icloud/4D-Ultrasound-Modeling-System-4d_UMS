"""示例入口：串联运行多周期平均法流水线。"""
from __future__ import annotations

import numpy as np

from src.config import PipelineConfig
from src.data_acquisition.monitor import UltrasoundMonitor
from src.data_acquisition.free_arm_scan import FreeArmScanner
from src.pipelines.multicycle_reconstruction import MulticycleReconstructionPipeline


def main() -> None:
    """研究阶段示例：使用仿真数据验证流程连通性。"""
    config = PipelineConfig()
    #monitor = UltrasoundMonitor.simulate(config.acquisition) #应用仿真数据时启用此行
    monitor = UltrasoundMonitor.from_npz(config.acquisition, "data/raw/monitor_stream.npz") #使用预录制数据时启用此行
    print(f"加载监测器，包含 {len(monitor.frames)} 帧") 
    if monitor.frames:
        print(f"  第一帧形状: {monitor.frames[0].image.shape}")
        # print(f"  第一帧唯一值: {np.unique(monitor.frames[0].image)}")
        # print(f"  第一帧均值: {np.mean(monitor.frames[0].image):.2f}, 标准差: {np.std(monitor.frames[0].image):.2f}")
    
    #scanner = FreeArmScanner.simulate(config.acquisition) #应用仿真数据时启用此行
    scanner = FreeArmScanner.from_npz(config.acquisition, "data/raw/scanner_sequence.npz") #使用预录制数据时启用此行
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
    
    print(f"相位体积数量: {len(output.phase_volumes)}")
    if output.phase_volumes:
        print(f"  第一个相位体积形状: {output.phase_volumes[0].shape}")
        print(f"  第一个相位体积唯一值: {np.unique(output.phase_volumes[0])}")
        print(f"  第一个相位体积均值: {np.mean(output.phase_volumes[0]):.2f}, 标准差: {np.std(output.phase_volumes[0]):.2f}")
    
    print(f"验证报告平滑度得分: {output.validation_report.smoothness_score}")
    print(f"验证报告蠕动速度: {output.validation_report.peristalsis_velocity}")
    
    # 额外调试验证指标
    print("调试验证指标...")
    if hasattr(output.validation_report, 'debug_info'):
        print(f"  调试信息: {output.validation_report.debug_info}")
    else:
        print("  验证报告中无调试信息")


if __name__ == "__main__":
    main()
