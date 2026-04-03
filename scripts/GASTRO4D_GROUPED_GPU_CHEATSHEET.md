# Gastro4D Grouped GPU 使用速查

这份文档对应的是独立的 grouped GPU 专用数据生成流程。

它的设计目标是：

- 不修改、也不替换原有 CPU 脚本。
- 直接适配你现在的分组数据组织方式。
- 让 source、clean、phase model、condition 的目录结构保持一致。
- 更符合科研数据集常见的 `split/group/instance` 组织方式。

## 适用场景

当你的源点云已经按下面这种方式组织时，优先使用这一版：

- `stomach_pcd/stomachPCD_dev/*.ply`
- `stomach_pcd/stomachPCD_01/*.ply`
- `stomach_pcd/stomachPCD_02/*.ply`

这套流程会把“组”和“划分”保留下来，不再像旧流程那样把 clean/condition 输出扁平化。

## 数据组织规范

该流程默认遵循以下目录规范：

- `stomach_pcd/<group>/<instance>.ply`
- `benchmark/instances/<split>/<group>/<instance>/...`
- `simuilate_data/instances/<split>/<group>/<instance>/...`
- `benchmark/conditions/<condition>/<split>/<group>/<instance>/...`

其中：

- `group` 例如 `stomachPCD_dev`、`stomachPCD_01`、`stomachPCD_02`
- `split` 目前按组名自动推断，包含 `dev` 的组归到 `dev`，其余归到 `test`

## 一条命令完整构建

下面这条命令会完成 grouped GPU 流程的全部核心步骤：

- 数据集根目录落盘
- clean monitor 生成
- phase model 生成
- scanner sequence 生成
- clean manifest 构建
- benchmark condition 生成

```bash
GPU_ID=0 \
SOURCE_DATA_ROOT=/home/liuyanan/data/Research_Data/4D-UMS \
UMS_DATA_ROOT=/home/liuyanan/data/Research_Data/4D-UMS/Gastro4D-USSim \
bash scripts/build_gastro4d_ussim_grouped_gpu.sh
```

说明：

- `GPU_ID` 用来指定当前构建使用哪张 GPU。
- `SOURCE_DATA_ROOT` 指向原始数据根目录。
- `UMS_DATA_ROOT` 指向 grouped GPU 流程要写入的数据根目录。
- 这条命令只调用新加的 grouped GPU 脚本，不会把旧 CPU 流程改掉。

## 分步执行

如果你想更细粒度地检查每一步输出，可以按下面顺序执行。

### 1. 指定当前 grouped GPU 流程的数据根目录

```bash
export UMS_DATA_ROOT=/home/liuyanan/data/Research_Data/4D-UMS/Gastro4D-USSim
```

### 2. 从分组点云落盘 grouped 数据集根目录

```bash
/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python \
scripts/materialize_gastro4d_ussim_dataset_gpu.py \
  --source-data-root /home/liuyanan/data/Research_Data/4D-UMS \
  --dataset-root "$UMS_DATA_ROOT" \
  --groups stomachPCD_dev stomachPCD_01 stomachPCD_02
```

说明：

- 这一步会保留 `stomach_pcd/<group>/` 结构。
- 同时会写出 `source_pointcloud_manifest_gpu.csv`，记录实例名、组别、split 和目标相对路径。
- 如果你想减少磁盘复制，可结合脚本里的 `--link-mode symlink` 使用软链接模式。

### 3. 生成 grouped clean 数据

```bash
/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python scripts/generate_monitor_stream_gpu.py
/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python scripts/generate_phase_sequence_models_gpu.py
/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python scripts/generate_scanner_from_phase_models_gpu.py --no-png
```

这三步分别负责：

- 为每个 grouped 实例生成 `monitor_stream`
- 在 grouped phase 目录下生成 `phase_sequence_models_run_*`
- 在 grouped clean 目录下生成 `scanner_sequence`

说明：

- 默认 `--no-png` 可减少磁盘占用。
- 如果你要抽检图像生成质量，可移除 `--no-png`。
- 如果只想处理部分组或部分实例，可以为这些脚本追加 `--groups` 或 `--instances` 参数。

### 4. 生成 grouped manifest 与 grouped benchmark conditions

```bash
/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python \
scripts/build_benchmark_manifest_gpu.py \
  --output "$UMS_DATA_ROOT/benchmark/manifests/benchmark_manifest_gpu.csv" \
  --skip-incomplete

/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python \
scripts/generate_benchmark_conditions_gpu.py \
  --conditions Sparse PoseNoise ImageNoise \
  --source-manifest "$UMS_DATA_ROOT/benchmark/manifests/benchmark_manifest_gpu.csv" \
  --condition-manifest "$UMS_DATA_ROOT/benchmark/manifests/benchmark_condition_manifest_gpu.csv" \
  --condition-root "$UMS_DATA_ROOT/benchmark/conditions" \
  --overwrite
```

说明：

- `benchmark_manifest_gpu.csv` 描述 grouped clean 数据。
- `benchmark_condition_manifest_gpu.csv` 描述 grouped 条件数据。
- `--skip-incomplete` 会跳过未生成完整 clean 资产的实例。
- `--overwrite` 会覆盖现有条件目录，适合重新生成 sparse、pose noise、image noise。

## 输出内容

这套 grouped GPU 流程的关键输出为：

- `$UMS_DATA_ROOT/benchmark/manifests/source_pointcloud_manifest_gpu.csv`
- `$UMS_DATA_ROOT/benchmark/manifests/benchmark_manifest_gpu.csv`
- `$UMS_DATA_ROOT/benchmark/manifests/benchmark_condition_manifest_gpu.csv`
- `$UMS_DATA_ROOT/benchmark/instances/<split>/<group>/<instance>/`
- `$UMS_DATA_ROOT/simuilate_data/instances/<split>/<group>/<instance>/`
- `$UMS_DATA_ROOT/benchmark/conditions/<condition>/<split>/<group>/<instance>/`

## 与旧版流程的区别

相对于旧版 `Gastro4D-USSim` 说明文档，这一版的主要区别是：

- 不再假设 `stomach_pcd` 是扁平目录。
- clean 输出按 `split/group/instance` 组织。
- phase model 输出按 `split/group/instance` 组织。
- condition 输出也按 `condition/split/group/instance` 组织。
- 更适合后续做分组统计、split 管理和批量实验调度。

## 常用补充建议

如果你是在远程 GPU 服务器上批量生成，建议：

- 先用少量实例测试一遍流程，确认目录和 manifest 正确。
- 再放开全部组别批量运行。
- 如果磁盘压力较大，优先使用 `--no-png`。
- 如果源点云不想复制，可把 materialize 阶段改成软链接模式。