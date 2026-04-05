#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/tianjun0/anaconda3/envs/modeling_py310/bin/python}"
DATASET_ROOT="${DATASET_ROOT:?DATASET_ROOT is required}"
SCANNER_FPS="${SCANNER_FPS:-10}"
SCANNER_DURATION_SECONDS="${SCANNER_DURATION_SECONDS:-900}"
SCANNER_GROUPS="${SCANNER_GROUPS:-stomachPCD_dev stomachPCD_01 stomachPCD_02}"
LOG_PREFIX="${LOG_PREFIX:-/tmp/g4d_hifi10fps900s}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python launcher not found: $PYTHON_BIN" >&2
    exit 1
fi

read -r -a GROUP_ARRAY <<< "$SCANNER_GROUPS"

echo "[ResumeScanner] repo=$REPO_ROOT"
echo "[ResumeScanner] dataset_root=$DATASET_ROOT"
echo "[ResumeScanner] groups=${GROUP_ARRAY[*]}"
echo "[ResumeScanner] fps=$SCANNER_FPS duration=$SCANNER_DURATION_SECONDS"
echo "[ResumeScanner] log_prefix=$LOG_PREFIX"

cd "$REPO_ROOT"

declare -a worker_pids=()

for group_name in "${GROUP_ARRAY[@]}"; do
    log_path="${LOG_PREFIX}_scanner_${group_name}.log"
    echo "[ResumeScanner] launch group=$group_name log=$log_path"
    "$PYTHON_BIN" scripts/generate_scanner_from_phase_models_gpu.py \
        --groups "$group_name" \
        --fps "$SCANNER_FPS" \
        --duration-seconds "$SCANNER_DURATION_SECONDS" \
        --no-png \
        > "$log_path" 2>&1 &
    worker_pids+=("$!")
done

for pid in "${worker_pids[@]}"; do
    wait "$pid"
done

echo "[ResumeScanner] scanner_done"

manifest_log="${LOG_PREFIX}_manifest.log"
conditions_log="${LOG_PREFIX}_conditions.log"

"$PYTHON_BIN" scripts/build_benchmark_manifest_gpu.py \
    --groups "${GROUP_ARRAY[@]}" \
    --output "$DATASET_ROOT/benchmark/manifests/benchmark_manifest_gpu.csv" \
    --skip-incomplete \
    > "$manifest_log" 2>&1

echo "[ResumeScanner] manifest_done log=$manifest_log"

"$PYTHON_BIN" scripts/generate_benchmark_conditions_gpu.py \
    --conditions Sparse PoseNoise ImageNoise \
    --source-manifest "$DATASET_ROOT/benchmark/manifests/benchmark_manifest_gpu.csv" \
    --condition-manifest "$DATASET_ROOT/benchmark/manifests/benchmark_condition_manifest_gpu.csv" \
    --condition-root "$DATASET_ROOT/benchmark/conditions" \
    --no-materialize-image-noise-pngs \
    --overwrite \
    > "$conditions_log" 2>&1

echo "[ResumeScanner] conditions_done log=$conditions_log"
echo "[ResumeScanner] all_done"
