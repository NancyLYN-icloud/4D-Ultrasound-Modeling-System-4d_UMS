# 胃部动态超声多周期平均法建模

该项目实现了“跨时空一致性的动态超声器官建模思路”，通过统一的 Python 架构覆盖监测阶段、相位检测、跨周期配准、三维平均、4D 插值与模型验证。

## 代码结构

- `src/config.py`：集中定义所有模块共享的配置与数据结构。
- `src/data_acquisition/monitor.py`：2D 高帧率监测，负责 ROI 特征提取（蠕动周期模板）。
- `src/data_acquisition/free_arm_scan.py`：自由臂扫描数据管理，包含仿真器、时间戳抖动校正。
- `src/preprocessing/phase_detection.py`：Savitzky-Golay 平滑 + 峰谷检测，输出周期 `C_i` 与相位模板。
- `src/preprocessing/binning.py`：将 3D 样本按归一化相位分箱，形成 {Φ} -> {样本集}。
- `src/reconstruction/reference.py`：根据样本最密集相位构建全局参考网格 `V_ref`。
- `src/reconstruction/registration.py`：两级非刚性配准接口，优先使用 B 样条 (SimpleITK)，退化时回落到质心对齐。
- `src/reconstruction/averaging.py`：基于 SNR 的体素加权平均，生成 `V_φ`。
- `src/modeling/interpolation.py`：对 `{V_φ}` 执行线性/周期三次样条插值，得到 `V(x,y,z,t)`。
- `src/modeling/validation.py`：平滑度、周期抖动、腔体容积曲线等指标验证。
- `src/pipelines/multicycle_reconstruction.py`：封装端到端流程并输出 `PipelineOutput`。
- `main.py`：仿真示例入口，可快速验证流程打通。

## 运行示例

```bash
python main.py
```

默认配置使用仿真数据，因此无需外部输入即可产出代表性结果。若需对接真实设备，可将：

- 高帧率监测阶段的 ROI 图像和时间戳输入 `UltrasoundMonitor.record`；
- 自由臂扫描阶段的 **视频帧 + 时间戳 + 位姿** 输入 `FreeArmScanner.ingest_frame_sequence`（内部会将 2D 帧按照姿态复制成薄层 3D 体素块），或直接调用 `FreeArmScanner.record` 喂入已重建好的 3D 数据。

两类数据准备完毕后，即可像示例那样调用 `MulticycleReconstructionPipeline`。

## 依赖

- Python 3.10+
- 必需：`numpy`
- 可选：`SimpleITK`, `scipy`（若存在，将自动启用 B 样条配准与三次样条插值）

## 扩展建议

1. 在 `FreeArmScanner.record` 内加入真实光学/电磁追踪姿态。
2. 在 `ReferenceVolumeBuilder` 中替换为基于点云插值或体绘制的高质量模板。
3. 将 `ModelValidator` 指标与临床金标准 (如胃压波) 联动，完成交叉验证。
