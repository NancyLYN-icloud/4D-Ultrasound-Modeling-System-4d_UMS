#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <pointcloud_root> [common args for run_single_dynamic_shared.py]" >&2
    echo "Example: $0 /path/to/phase_pointclouds_run_xxx --dynamic-train-steps 6000 --dynamic-mesh-resolution 60" >&2
    exit 1
fi

POINTCLOUD_ROOT="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python}"
OUT_DIR="${OUT_DIR:-}"
INCLUDE_PRIOR_FREE="${INCLUDE_PRIOR_FREE:-0}"

if [[ ! -d "$POINTCLOUD_ROOT" ]]; then
    echo "Pointcloud root not found: $POINTCLOUD_ROOT" >&2
    exit 1
fi

METHODS=(
    "动态共享-参考对应正则::refcorr-main-table"
    "动态共享-连续形变场::continuous-main-table"
    "动态共享-解耦运动潜码::decoupled-motion-main-table"
    "动态共享-全局基残差::global-basis-residual-main-table"
)

if [[ "$INCLUDE_PRIOR_FREE" == "1" ]]; then
    METHODS+=("动态共享-无先验4D场::prior-free-main-table")
fi

for item in "${METHODS[@]}"; do
    method="${item%%::*}"
    run_name="${item##*::}"

    cmd=(
        "$PYTHON_BIN"
        "$REPO_ROOT/scripts/run_single_dynamic_shared.py"
        --pointcloud-root "$POINTCLOUD_ROOT"
        --method "$method"
        --run-name "$run_name"
    )

    if [[ -n "$OUT_DIR" ]]; then
        cmd+=(--out-dir "$OUT_DIR")
    fi

    if [[ $# -gt 0 ]]; then
        cmd+=("$@")
    fi

    echo "[MainTable] Running: $method"
    "${cmd[@]}"
done
