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
USE_IMPROVED_SCANNER="${USE_IMPROVED_SCANNER:-0}"
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

echo "[Gastro4D-USSim GPU] repo=$REPO_ROOT"
echo "[Gastro4D-USSim GPU] python=$PYTHON_BIN"
echo "[Gastro4D-USSim GPU] source_data_root=$SOURCE_DATA_ROOT"
echo "[Gastro4D-USSim GPU] dataset_root=$UMS_DATA_ROOT"
echo "[Gastro4D-USSim GPU] groups=${GROUP_ARRAY[*]}"

"$PYTHON_BIN" "$REPO_ROOT/scripts/materialize_gastro4d_ussim_dataset.py" \
    --source-data-root "$SOURCE_DATA_ROOT" \
    --dataset-root "$UMS_DATA_ROOT" \
    --dataset-name "$DATASET_NAME" \
    --groups "${GROUP_ARRAY[@]}"

export UMS_DATA_ROOT

"$PYTHON_BIN" "$REPO_ROOT/scripts/generate_monitor_stream.py" --batch-all-references
"$PYTHON_BIN" "$REPO_ROOT/scripts/generate_phase_sequence_models.py" --batch-all-references

if [[ "$WRITE_PNGS" == "1" ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/generate_scanner_from_phase_models.py" --batch-all-references
else
    "$PYTHON_BIN" "$REPO_ROOT/scripts/generate_scanner_from_phase_models.py" --batch-all-references --no-png
fi

if [[ "$USE_IMPROVED_SCANNER" == "1" ]]; then
    rm -rf "$UMS_DATA_ROOT/benchmark/instances_before"
    mkdir -p "$UMS_DATA_ROOT/benchmark/instances_before"
    cp -a "$UMS_DATA_ROOT/benchmark/instances/." "$UMS_DATA_ROOT/benchmark/instances_before/"
    "$PYTHON_BIN" "$REPO_ROOT/scripts/regenerate_improved_benchmark_instances.py" --source-root "$UMS_DATA_ROOT/benchmark/instances_before"
fi

"$PYTHON_BIN" "$REPO_ROOT/scripts/build_benchmark_manifest.py" \
    --source-manifest "$UMS_DATA_ROOT/benchmark/manifests/source_pointcloud_manifest.csv" \
    --output "$UMS_DATA_ROOT/benchmark/manifests/benchmark_manifest.csv" \
    --skip-incomplete

"$PYTHON_BIN" "$REPO_ROOT/scripts/generate_benchmark_conditions.py" \
    --conditions "${CONDITION_ARRAY[@]}" \
    --source-manifest "$UMS_DATA_ROOT/benchmark/manifests/benchmark_manifest.csv" \
    --condition-manifest "$UMS_DATA_ROOT/benchmark/manifests/benchmark_condition_manifest.csv" \
    --condition-root "$UMS_DATA_ROOT/benchmark/conditions" \
    --overwrite

echo "[Gastro4D-USSim GPU] completed dataset build under $UMS_DATA_ROOT"