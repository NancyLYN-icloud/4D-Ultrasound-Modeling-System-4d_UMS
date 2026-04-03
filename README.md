# 4D-Ultrasound-Modeling-System-4D-UMS

本项目当前聚焦于面向自由手胃部超声的时空一致重建，核心主线已经从早期的“多周期平均法”演化为：

监测信号 -> 非线性相位标准化 -> 标准相位分箱点云构建 -> 静态逐相位重建 / `动态共享-全局基残差` 重建 -> 相位级评估与按时间轴导出的 4D 序列评估。

当前推荐作为论文主模型的方法是：

- `动态共享-全局基残差`（`shared_topology_global_basis_residual`）

其核心思想是：

- 用共享拓扑的基准网格表示静态参考形态。
- 用低秩全局运动基表示跨相位的大尺度周期运动。
- 用相位条件残差场补充局部动态细节，并配合 correspondence 时序调度、残差正则与置信度加权联合优化。

## 当前项目能力

项目当前已经具备以下核心能力：

1. 监测流周期检测与非线性相位标准化。
2. 基于标准相位分箱的带质量摘要点云构建。
3. 逐相位静态隐式表面重建。
4. `动态共享-全局基残差` 主方法重建。
5. 基于扫描时间戳的逐帧 4D 时间轴网格导出。
6. Chamfer Distance、HD95、时间平滑度、水密比例等实验指标输出。
7. 方法对比实验与主方法消融实验脚本。

## 目录结构

- [src/config.py](src/config.py)：统一配置入口、数据结构、静态/动态模型超参数。
- [src/data_acquisition/monitor.py](src/data_acquisition/monitor.py)：监测流读取与特征轨迹提取。
- [src/data_acquisition/free_arm_scan.py](src/data_acquisition/free_arm_scan.py)：自由手扫描流读取、时间对齐和伪 3D 组织。
- [src/preprocessing/phase_detection.py](src/preprocessing/phase_detection.py)：周期检测。
- [src/preprocessing/phase_canonicalization.py](src/preprocessing/phase_canonicalization.py)：非线性相位标准化。
- [src/preprocessing/binning.py](src/preprocessing/binning.py)：标准相位分箱，以及可选的滑动窗口分箱实现。
- [src/preprocessing/pointcloud_builder.py](src/preprocessing/pointcloud_builder.py)：置信度感知点云构建与摘要导出。
- [src/modeling/surface_reconstruction.py](src/modeling/surface_reconstruction.py)：逐相位静态隐式表面重建。
- [src/modeling/canonical_field.py](src/modeling/canonical_field.py)：标准形状隐式场。
- [src/modeling/deformation_field.py](src/modeling/deformation_field.py)：相位条件形变场。
- [src/modeling/dynamic_surface_reconstruction.py](src/modeling/dynamic_surface_reconstruction.py)：动态共享重建，当前主线为 `动态共享-全局基残差`。
- [src/modeling/metrics.py](src/modeling/metrics.py)：CD、HD95、时间平滑度等指标。
- [src/pipelines/multicycle_reconstruction.py](src/pipelines/multicycle_reconstruction.py)：端到端主流程。
- [scripts/run_experiments.py](scripts/run_experiments.py)：方法对比与主方法消融实验入口。
- [experiments/METHOD_FRAMEWORK.md](experiments/METHOD_FRAMEWORK.md)：方法框架文档。
- [experiments/DESIGN.md](experiments/DESIGN.md)：实验设计文档。

## 快速开始

### 0. 恢复 Conda 环境

推荐先恢复项目环境，再运行下面的主流程和实验命令：

```bash
conda env create -n modeling_py310 -f environment/modeling_py310.history.yml
conda activate modeling_py310
```

如果最小环境解算后仍缺少依赖，可改用完整导出：

```bash
conda env create -n modeling_py310 -f environment/modeling_py310.environment.yml
conda activate modeling_py310
```

更详细的环境说明见 [environment/README.md](environment/README.md)。

### 1. 运行主流程示例

```bash
python main.py
```

### 2. 运行快速实验

```bash
python scripts/run_experiments.py --mode fast-dev --experiment-set both
```

### 3. 运行正式方法对比

```bash
python scripts/run_experiments.py --mode full-paper --experiment-set method-comparison
```

### 4. 运行正式主方法消融

```bash
python scripts/run_experiments.py --mode full-paper --experiment-set cpd-ablation
```

## 数据路径

默认数据根目录为：

- `/home/liuyanan/data/Research_Data/4D-UMS`

如需切换位置，可设置环境变量：

```bash
export UMS_DATA_ROOT=/your/data/root
```

当前默认测试数据包括：

- `test/monitor_stream.npz`
- `test/scanner_sequence.npz`
- `stomach_pcd/niujiao01.ply`：默认实例 `niujiao01` 的静态参考点云
- `stomach_pcd/*.ply`：多胃部实例参考点云库；其他实例的派生数据默认写入 `benchmark/instances/<instance>/` 与 `simuilate_data/instances/<instance>/`
- `simuilate_data/meshes/`：相位级动态仿真 GT 网格序列（实验评估默认使用）

多实例工作流推荐入口：

- `python scripts/generate_phase_sequence_models.py --batch-all-references`
- `python scripts/generate_scanner_from_phase_models.py --batch-all-references`
- `scripts/build_multi_instance_dataset.sh`
- `python scripts/run_experiments.py --instance-name niujiao01 --mode dynamic-detail --experiment-set both`
- `scripts/run_multi_instance_experiments.sh`
- `scripts/DATA_PIPELINE.md`

## 当前推荐实验模式

`scripts/run_experiments.py` 当前支持两套模式：

- `fast-dev`：快速调试模式，降低训练步数、采样规模和网格分辨率，用于验证实验链路与结果趋势。
- `dynamic-detail`：动态细节诊断模式，提升 `动态共享-全局基残差` 训练预算、采样规模和网格分辨率，并减弱过强的时间平滑约束，用于检查动态网格细节是否被过度抹平。
- `full-paper`：论文正式实验模式，使用更高训练预算生成正式对比结果。

同时支持两类实验集：

- `method-comparison`：静态基线、静态增强与当前主方法三类方法对比。
- `cpd-ablation`：当前主方法关键损失项消融，保留旧实验集名称以兼容历史命令。

## 当前主模型：动态共享-全局基残差

`动态共享-全局基残差` 在当前代码中对应：

- 系数/残差网络：[src/modeling/deformation_field.py](src/modeling/deformation_field.py)
- 动态重建器：[src/modeling/dynamic_surface_reconstruction.py](src/modeling/dynamic_surface_reconstruction.py)
- 方法默认配置：[scripts/run_experiments.py](scripts/run_experiments.py)

目前已实现的关键损失项包括：

- 表面拟合损失
- 法向一致性损失
- 质心一致性与空间平滑损失
- 基系数 bootstrap / 时间 / 加速度 / 周期性正则
- 残差均值与残差基投影约束
- unsupported 区域 anchor / laplacian 正则
- 全局分支 correspondence 时序约束与调度
- 置信度感知的相位采样与损失加权

## 输出结果

主流程和实验脚本当前会导出：

- 相位点云 `.ply`
- 相位点云摘要 `pointcloud_summary.csv`
- 静态网格 `.ply`
- 静态网格摘要 `mesh_summary.csv`
- 动态网格 `.ply`
- 动态网格摘要 `dynamic_mesh_summary.csv`
- 逐帧时间轴动态网格 `.ply`（启用 `dynamic_model.export_timeline_meshes` 时）
- 逐帧时间轴动态网格摘要 `dynamic_timeline_mesh_summary.csv`
- 方法对比结果 `method_comparison.csv/.tex`
- 主方法消融结果 `cpd_ablation.csv/.tex`（沿用历史文件名）

默认情况下，`scripts/run_experiments.py` 会把所有实验输出统一写入 `/home/liuyanan/data/Research_Data/4D-UMS/experiments` 下，并为每次运行自动创建独立归档目录，例如：

- `exp_20260317_154500_fast-dev_both/`
- `run.log`：完整控制台日志
- `run_metadata.json`：命令、输入路径、模式、时间戳等运行元数据
- `configs/*.json`：每个方法/消融对应的完整配置快照
- `artifacts/`：本次运行产生的点云、网格及其摘要
- `method_comparison.csv/.tex`
- `cpd_ablation.csv/.tex`

这样可以保留每次实验的结果、重要参数和日志，便于回溯与论文整理。

## 依赖

- Python 3.10+
- 必需：`numpy`, `torch`, `trimesh`, `mcubes`, `pandas`
- 可选：`opencv-python`, `scipy`
- 推荐直接使用 [environment/README.md](environment/README.md) 中给出的 Conda 环境恢复方式，而不是在系统 Python 中手动补包。

## 文档导航

- 项目架构：[ARCHITECTURE.md](ARCHITECTURE.md)
- 方法框架：[experiments/METHOD_FRAMEWORK.md](experiments/METHOD_FRAMEWORK.md)
- 实验设计：[experiments/DESIGN.md](experiments/DESIGN.md)

## 当前建议的论文主线

如果你计划继续把该项目推进成高水平论文，当前最清晰的主线是：

1. 非线性相位标准化
2. 置信度感知观测融合
3. `动态共享-全局基残差` 动态建模

这三部分已经在代码、文档和实验脚本中形成了相互对应的结构，后续可以围绕它们继续强化方法与实验。
