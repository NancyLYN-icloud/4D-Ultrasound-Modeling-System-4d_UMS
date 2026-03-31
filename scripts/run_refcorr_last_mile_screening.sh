#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/home/liuyanan/data/Research_Data/4D-UMS/experiments}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python}"

run_fresh() {
    local run_name="$1"
    shift
    "$PYTHON_BIN" "$REPO_ROOT/scripts/run_single_dynamic_shared.py" \
        --out-dir "$OUT_DIR" \
        --method "动态共享-参考对应正则" \
        --run-name "$run_name" \
        --dynamic-train-steps 3600 \
        "$@"
}

run_replay() {
    local pointcloud_root="$1"
    local run_name="$2"
    shift 2
    "$PYTHON_BIN" "$REPO_ROOT/scripts/run_single_dynamic_shared.py" \
        --pointcloud-root "$pointcloud_root" \
        --out-dir "$OUT_DIR" \
        --method "动态共享-参考对应正则" \
        --run-name "$run_name" \
        --dynamic-train-steps 3600 \
        "$@"
}

echo "[RefCorrScreen] candidate 1/3: temporal-shift-delayed"
run_fresh "refcorr-temporal-shift-delayed-screen" \
    --temporal-weight 0.015 \
    --temporal-acceleration-weight 0.0075 \
    --phase-consistency-weight 0.0075 \
    --correspondence-temporal-weight 0.01 \
    --correspondence-acceleration-weight 0.005 \
    --correspondence-phase-consistency-weight 0.005 \
    --correspondence-start-fraction 0.35 \
    --correspondence-ramp-fraction 0.25

FIRST_RUN_DIR="$(find "$OUT_DIR" -maxdepth 1 -type d -name 'exp_*_refcorr-temporal-shift-delayed-screen' | sort | tail -n 1)"
if [[ -z "$FIRST_RUN_DIR" ]]; then
    echo "Failed to locate the first screening run directory" >&2
    exit 1
fi

POINTCLOUD_ROOT="$(find "$FIRST_RUN_DIR/artifacts/phase_pointclouds_run_001_single" -maxdepth 1 -type d -name 'phase_pointclouds_run_001_*' | sort | tail -n 1)"
if [[ -z "$POINTCLOUD_ROOT" ]]; then
    echo "Failed to locate replay pointcloud root under $FIRST_RUN_DIR" >&2
    exit 1
fi

echo "[RefCorrScreen] Reusing pointcloud root: $POINTCLOUD_ROOT"

echo "[RefCorrScreen] candidate 2/3: very-light-delayed"
run_replay "$POINTCLOUD_ROOT" "refcorr-very-light-delayed-screen" \
    --temporal-weight 0.02 \
    --temporal-acceleration-weight 0.01 \
    --phase-consistency-weight 0.01 \
    --correspondence-temporal-weight 0.005 \
    --correspondence-acceleration-weight 0.002 \
    --correspondence-phase-consistency-weight 0.002 \
    --correspondence-start-fraction 0.35 \
    --correspondence-ramp-fraction 0.25

echo "[RefCorrScreen] candidate 3/3: smallnet-delayed"
run_replay "$POINTCLOUD_ROOT" "refcorr-smallnet-delayed-screen" \
    --canonical-hidden-dim 128 \
    --canonical-hidden-layers 4 \
    --deformation-hidden-dim 128 \
    --deformation-hidden-layers 3 \
    --confidence-floor 0.6 \
    --correspondence-start-fraction 0.35 \
    --correspondence-ramp-fraction 0.25

echo "[RefCorrScreen] Screening completed. Review the three dynamic_shared_result.csv files under $OUT_DIR."