# 4D-Ultrasound-Modeling-System-4D-UMS

本项目当前聚焦于面向自由手胃部超声的时空一致重建，核心主线已经从早期的“多周期平均法”演化为：

监测信号 -> 非线性相位标准化 -> 标准相位分箱点云构建 -> 静态逐相位重建 / `动态共享-全局基残差` 重建 -> 相位级评估与按时间轴导出的 4D 序列评估。

当前论文实验主线已经进一步收敛为两层入口：

- 批量 benchmark 入口：`scripts/run_benchmark_suite.py`
- 单方法执行入口：`scripts/run_single_dynamic_shared.py`

如果希望直接运行论文主表或补充实验，当前推荐使用：

- `experiments/run_sci_q1_main_table.sh`
- `experiments/run_sci_q1_supplementary_table.sh`

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
6. Chamfer Distance、HD95、表面 MAE、EMD、Dice 等实验指标输出。
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
- [src/modeling/metrics.py](src/modeling/metrics.py)：CD、HD95、表面 MAE、EMD、Dice 等指标。
- [src/pipelines/multicycle_reconstruction.py](src/pipelines/multicycle_reconstruction.py)：端到端主流程。
- [scripts/run_experiments.py](scripts/run_experiments.py)：早期单体实验入口，当前更多用于兼容旧流程与快速调试。
- [scripts/experiment_method_registry.py](scripts/experiment_method_registry.py)：论文方法注册表，统一维护主表方法、补充实验方法、学术命名与 profile。
- [scripts/run_single_dynamic_shared.py](scripts/run_single_dynamic_shared.py)：单方法重建执行入口。
- [scripts/run_benchmark_suite.py](scripts/run_benchmark_suite.py)：当前推荐的批量 benchmark 入口。
- [experiments/run_sci_q1_main_table.sh](experiments/run_sci_q1_main_table.sh)：论文主表一键脚本入口。
- [experiments/run_sci_q1_supplementary_table.sh](experiments/run_sci_q1_supplementary_table.sh)：补充实验一键脚本入口。
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

### 3. 运行旧版正式方法对比

```bash
python scripts/run_experiments.py --mode full-paper --experiment-set method-comparison
```

### 4. 运行旧版正式主方法消融

```bash
python scripts/run_experiments.py --mode full-paper --experiment-set cpd-ablation
```

### 5. 运行当前推荐的论文主实验

当前推荐的主实验入口不是旧的 `run_experiments.py`，而是基于 manifest 的批量 benchmark 入口：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_benchmark_suite.py \
	--manifest /home/tianjun0/liuyanan/data/4D-UMS/Gastro4D-USSim/benchmark/manifests/benchmark_condition_manifest_gpu.csv \
	--data-root /home/tianjun0/liuyanan/data/4D-UMS/Gastro4D-USSim \
	--out-root /home/tianjun0/liuyanan/data/4D-UMS/experiment \
	--suite-name gastro4d_ussim_main_table \
	--split test \
	--conditions Clean \
	--methods main-table \
	--method-profile historical_best_eqbudget \
	--mode full-paper \
	--dynamic-train-steps 10000 \
	--dynamic-mesh-resolution 72 \
	--max-points-per-phase 5000
```

也可以直接使用论文主表包装脚本：

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/home/tianjun0/liuyanan/data/4D-UMS/Gastro4D-USSim \
bash experiments/run_sci_q1_main_table.sh
```

### 6. 运行当前推荐的补充实验

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_benchmark_suite.py \
	--manifest /home/tianjun0/liuyanan/data/4D-UMS/Gastro4D-USSim/benchmark/manifests/benchmark_condition_manifest_gpu.csv \
	--data-root /home/tianjun0/liuyanan/data/4D-UMS/Gastro4D-USSim \
	--out-root /home/tianjun0/liuyanan/data/4D-UMS/experiment \
	--suite-name gastro4d_ussim_supplementary \
	--split test \
	--conditions Clean \
	--methods supplementary-baselines \
	--method-profile historical_best_eqbudget \
	--mode full-paper \
	--dynamic-train-steps 10000 \
	--dynamic-mesh-resolution 72 \
	--max-points-per-phase 5000
```

或直接使用补充实验包装脚本：

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/home/tianjun0/liuyanan/data/4D-UMS/Gastro4D-USSim \
bash experiments/run_sci_q1_supplementary_table.sh
```

### 7. 按方法分片运行主实验

如果希望把主表四个方法拆成独立后台任务，当前推荐入口是：

```bash
CUDA_VISIBLE_DEVICES=0 \
OUT_ROOT=/home/tianjun0/liuyanan/data/4D-UMS/experiment/controlled_observation_robustness_benchmark \
bash experiments/run_sci_q1_shards_by_method.sh
```

## 数据路径

当前生产数据默认根目录为：

- `/home/tianjun0/liuyanan/data/4D-UMS`

当前生产 benchmark 数据集目录为：

- `/home/tianjun0/liuyanan/data/4D-UMS/Gastro4D-USSim`

当前实验输出推荐目录为：

- `/home/tianjun0/liuyanan/data/4D-UMS/experiment`

如需切换位置，可设置环境变量：

```bash
export UMS_DATA_ROOT=/your/data/root
```

当前默认测试数据包括：

- `test/monitor_stream.npz`
- `test/scanner_sequence.npz`
- `stomach_pcd/niujiao01.ply`：默认实例 `niujiao01` 的静态参考点云
- `stomach_pcd/*.ply`：多胃部实例参考点云库；其他实例的派生数据默认写入 `benchmark/instances/<split>/<group>/<instance>/` 与 `simuilate_data/instances/<split>/<group>/<instance>/`
- `simuilate_data/meshes/`：相位级动态仿真 GT 网格序列（实验评估默认使用）

多实例工作流推荐入口：

- `python scripts/generate_phase_sequence_models.py --batch-all-references`
- `python scripts/generate_scanner_from_phase_models.py --batch-all-references`
- `scripts/build_multi_instance_dataset.sh`
- `python scripts/run_experiments.py --instance-name niujiao01 --mode dynamic-detail --experiment-set both`
- `python scripts/run_benchmark_suite.py --manifest <condition_manifest> --data-root <dataset_root> --out-root <experiment_root> --methods main-table`
- `bash experiments/run_sci_q1_main_table.sh`
- `bash experiments/run_sci_q1_supplementary_table.sh`
- `scripts/run_multi_instance_experiments.sh`
- `scripts/DATA_PIPELINE.md`

## 当前推荐实验模式

`scripts/run_experiments.py` 当前主要保留为旧流程兼容与快速调试入口，支持三套模式：

- `fast-dev`：快速调试模式，降低训练步数、采样规模和网格分辨率，用于验证实验链路与结果趋势。
- `dynamic-detail`：动态细节诊断模式，提升 `动态共享-全局基残差` 训练预算、采样规模和网格分辨率，并减弱过强的时间平滑约束，用于检查动态网格细节是否被过度抹平。
- `full-paper`：论文正式实验模式，使用更高训练预算生成正式对比结果。

同时支持两类实验集：

- `method-comparison`：静态基线、静态增强与当前主方法三类方法对比。
- `cpd-ablation`：当前主方法关键损失项消融，保留旧实验集名称以兼容历史命令。

而当前论文正式 benchmark 更推荐使用 `scripts/run_benchmark_suite.py`，其方法集合由 `scripts/experiment_method_registry.py` 统一管理。

当前默认主表方法集合 `main-table` 包括：

- `动态共享-参考对应正则` (`refcorr`)
- `动态共享-连续形变场` (`continuous`)
- `动态共享-解耦运动潜码` (`decoupled_motion`)
- `动态共享-全局基残差` (`global_basis_residual`)

当前补充实验方法集合 `supplementary-baselines` 包括：

- `静态增强` (`phase_normalized_static`)
- `动态共享-无先验4D场` (`prior_free_4d_field`)
- `动态共享-CPD对应点` (`cpd_guided_correspondence`)

当前所有主表与补充实验的模型重建部分都支持 GPU。若要显式指定使用 0 号 GPU，推荐统一写法：

```bash
CUDA_VISIBLE_DEVICES=0 <your_command>
```

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

默认情况下，`scripts/run_experiments.py` 会把所有实验输出统一写入实验输出根目录下，并为每次运行自动创建独立归档目录，例如：

- `exp_20260317_154500_fast-dev_both/`
- `run.log`：完整控制台日志
- `run_metadata.json`：命令、输入路径、模式、时间戳等运行元数据
- `configs/*.json`：每个方法/消融对应的完整配置快照
- `artifacts/`：本次运行产生的点云、网格及其摘要
- `method_comparison.csv/.tex`
- `cpd_ablation.csv/.tex`

而 `scripts/run_benchmark_suite.py` 会按照 suite 结构组织输出，例如：

- `suite_20260404_220000_gastro4d_ussim_main_table/`
- `suite_metadata.json`
- `methods/<method_slug>/method_spec.json`
- `methods/<method_slug>/runs/`
- `aggregated/results_flat.csv`
- `aggregated/results_by_method.csv`

这样可以同时保留单次 run 明细、方法级元信息和最终论文表格聚合结果，便于回溯与论文整理。

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
