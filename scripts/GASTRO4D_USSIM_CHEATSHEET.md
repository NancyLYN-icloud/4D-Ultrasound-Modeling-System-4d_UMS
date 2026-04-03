# Gastro4D-USSim 使用速查

数据集名称：`Gastro4D-USSim`

这份文档对应的是现有的基础版批量生成流程。它适合以下场景：

- 你希望沿用当前已有脚本，不额外切换到新的 grouped GPU 专用目录规范。
- 你希望快速把 `stomach_pcd` 下的数据批量生成成 clean 数据与 benchmark condition 数据。
- 你需要一份最短命令参考，而不是完整设计说明。

如果你现在使用的是按分组组织的 GPU 新流程，也就是 `stomach_pcd/stomachPCD_dev`、`stomach_pcd/stomachPCD_01`、`stomach_pcd/stomachPCD_02` 这种结构，优先看另一份 grouped GPU 文档。

## 推荐数据集根目录

建议先显式指定数据根目录，避免脚本误写回默认数据目录：

```bash
export UMS_DATA_ROOT=/your/data/root/Gastro4D-USSim
```

## 一条命令完整构建

下面这条命令会串行完成：数据集落盘、clean 数据生成、manifest 构建、condition 生成。

```bash
GPU_ID=0 \
SOURCE_DATA_ROOT=/home/liuyanan/data/Research_Data/4D-UMS \
UMS_DATA_ROOT=/home/liuyanan/data/Research_Data/4D-UMS/Gastro4D-USSim \
bash scripts/build_gastro4d_ussim_gpu.sh
```

说明：

- `GPU_ID=0` 用于指定可见 GPU。
- `SOURCE_DATA_ROOT` 是原始数据根目录。
- `UMS_DATA_ROOT` 是输出数据集根目录。
- 这个入口仍然属于旧版批处理思路，不会按 split/group 细分 clean 输出目录。

## 分步执行命令

当你希望逐步检查每一阶段结果时，可以按下面顺序执行。

### 1. 从分组点云落盘出数据集根目录

这一步会创建 `Gastro4D-USSim` 目录，并把 source point cloud 信息写入 manifest。

```bash
/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python \
scripts/materialize_gastro4d_ussim_dataset.py \
  --source-data-root /home/liuyanan/data/Research_Data/4D-UMS \
  --dataset-root /home/liuyanan/data/Research_Data/4D-UMS/Gastro4D-USSim \
  --groups stomachPCD_dev stomachPCD_01 stomachPCD_02
```

### 2. 切换当前流程使用的数据根目录

```bash
export UMS_DATA_ROOT=/home/liuyanan/data/Research_Data/4D-UMS/Gastro4D-USSim
```

### 3. 生成所有 clean 样本资产

这三步分别生成：

- `monitor_stream`
- `phase sequence models`
- `scanner sequence`

```bash
/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python scripts/generate_monitor_stream.py --batch-all-references
/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python scripts/generate_phase_sequence_models.py --batch-all-references
/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python scripts/generate_scanner_from_phase_models.py --batch-all-references --no-png
```

说明：

- `--no-png` 表示不额外导出 scanner PNG，可减少磁盘占用和生成时间。
- 如果你需要检查图像质量，可以去掉这个参数。

### 4. 构建 clean manifest 与 benchmark condition manifest

```bash
/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python \
scripts/build_benchmark_manifest.py \
  --source-manifest "$UMS_DATA_ROOT/benchmark/manifests/source_pointcloud_manifest.csv" \
  --output "$UMS_DATA_ROOT/benchmark/manifests/benchmark_manifest.csv" \
  --skip-incomplete

/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python \
scripts/generate_benchmark_conditions.py \
  --conditions Sparse PoseNoise ImageNoise \
  --source-manifest "$UMS_DATA_ROOT/benchmark/manifests/benchmark_manifest.csv" \
  --condition-manifest "$UMS_DATA_ROOT/benchmark/manifests/benchmark_condition_manifest.csv" \
  --condition-root "$UMS_DATA_ROOT/benchmark/conditions" \
  --overwrite
```

说明：

- `benchmark_manifest.csv` 描述 clean 数据。
- `benchmark_condition_manifest.csv` 描述包含 `Clean/Sparse/PoseNoise/ImageNoise` 在内的实验输入。
- `--skip-incomplete` 会跳过尚未生成完整资产的实例。
- `--overwrite` 会覆盖已存在的条件数据目录。

## 可选：重建 improved clean scanner

如果你想在 clean 数据生成完成后，再用改进版 scanner slicing 逻辑重写 clean scanner，可执行下面这组命令。

```bash
rm -rf "$UMS_DATA_ROOT/benchmark/instances_before"
mkdir -p "$UMS_DATA_ROOT/benchmark/instances_before"
cp -a "$UMS_DATA_ROOT/benchmark/instances/." "$UMS_DATA_ROOT/benchmark/instances_before/"

/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python \
scripts/regenerate_improved_benchmark_instances.py \
  --source-root "$UMS_DATA_ROOT/benchmark/instances_before"
```

说明：

- `instances_before` 是重建前的备份目录。
- 这一步只影响 clean scanner 数据，不会自动重建 condition，必要时需要重新生成条件数据。

## 输出结构

基础版流程的关键输出包括：

- `$UMS_DATA_ROOT/stomach_pcd/`
- `$UMS_DATA_ROOT/simuilate_data/instances/`
- `$UMS_DATA_ROOT/benchmark/instances/`
- `$UMS_DATA_ROOT/benchmark/conditions/`
- `$UMS_DATA_ROOT/benchmark/manifests/source_pointcloud_manifest.csv`
- `$UMS_DATA_ROOT/benchmark/manifests/benchmark_manifest.csv`
- `$UMS_DATA_ROOT/benchmark/manifests/benchmark_condition_manifest.csv`

## 什么时候不要用这份流程

如果你已经明确要求：

- 保留 `stomach_pcd/<group>/<instance>.ply` 的分组结构
- clean 数据输出也按 `split/group/instance` 组织
- condition 输出也按 `condition/split/group/instance` 组织

那么请改用 grouped GPU 版本文档和对应脚本。