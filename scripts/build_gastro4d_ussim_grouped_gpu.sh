#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_ROOT="${CONDA_ROOT:-/home/liuyanan/program/environment/miniconda3}"
CONDA_ENV="${CONDA_ENV:-modeling_py310}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ROOT/envs/$CONDA_ENV/bin/python}"

SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/home/liuyanan/data/Research_Data/4D-UMS}"
DATASET_NAME="${DATASET_NAME:-Gastro4D-USSim}"
UMS_DATA_ROOT="${UMS_DATA_ROOT:-$SOURCE_DATA_ROOT/$DATASET_NAME}"
SOURCE_GROUPS="${SOURCE_GROUPS:-stomachPCD_dev stomachPCD_01 stomachPCD_02}"
CONDITIONS="${CONDITIONS:-Sparse PoseNoise ImageNoise}"
LINK_MODE="${LINK_MODE:-copy}"
WRITE_PNGS="${WRITE_PNGS:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python launcher not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ -n "${GPU_ID:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

read -r -a GROUP_ARRAY <<< "$SOURCE_GROUPS"
read -r -a CONDITION_ARRAY <<< "$CONDITIONS"

echo "[Gastro4D Grouped GPU] repo=$REPO_ROOT"
echo "[Gastro4D Grouped GPU] python=$PYTHON_BIN"
echo "[Gastro4D Grouped GPU] source_data_root=$SOURCE_DATA_ROOT"
echo "[Gastro4D Grouped GPU] dataset_root=$UMS_DATA_ROOT"
echo "[Gastro4D Grouped GPU] groups=${GROUP_ARRAY[*]}"

"$PYTHON_BIN" "$REPO_ROOT/scripts/materialize_gastro4d_ussim_dataset_gpu.py" \
    --source-data-root "$SOURCE_DATA_ROOT" \
    --dataset-root "$UMS_DATA_ROOT" \
    --dataset-name "$DATASET_NAME" \
    --groups "${GROUP_ARRAY[@]}" \
    --link-mode "$LINK_MODE"

export UMS_DATA_ROOT

"$PYTHON_BIN" "$REPO_ROOT/scripts/generate_monitor_stream_gpu.py" --groups "${GROUP_ARRAY[@]}"
"$PYTHON_BIN" "$REPO_ROOT/scripts/generate_phase_sequence_models_gpu.py" --groups "${GROUP_ARRAY[@]}"

if [[ "$WRITE_PNGS" == "1" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/generate_scanner_from_phase_models_gpu.py" --groups "${GROUP_ARRAY[@]}"
else
    "$PYTHON_BIN" "$REPO_ROOT/scripts/generate_scanner_from_phase_models_gpu.py" --groups "${GROUP_ARRAY[@]}" --no-png
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/build_benchmark_manifest_gpu.py" \
    --groups "${GROUP_ARRAY[@]}" \
    --output "$UMS_DATA_ROOT/benchmark/manifests/benchmark_manifest_gpu.csv" \
    --skip-incomplete

"$PYTHON_BIN" "$REPO_ROOT/scripts/generate_benchmark_conditions_gpu.py" \
    --conditions "${CONDITION_ARRAY[@]}" \
    --source-manifest "$UMS_DATA_ROOT/benchmark/manifests/benchmark_manifest_gpu.csv" \
    --condition-manifest "$UMS_DATA_ROOT/benchmark/manifests/benchmark_condition_manifest_gpu.csv" \
    --condition-root "$UMS_DATA_ROOT/benchmark/conditions" \
    --overwrite

echo "[Gastro4D Grouped GPU] completed dataset build under $UMS_DATA_ROOT"
