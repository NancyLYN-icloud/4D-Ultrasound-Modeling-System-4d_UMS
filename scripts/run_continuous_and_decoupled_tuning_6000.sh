#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <pointcloud_root> [out_dir]" >&2
    exit 1
fi

POINTCLOUD_ROOT="$1"
OUT_DIR="${2:-/home/liuyanan/data/Research_Data/4D-UMS/experiments}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python}"

if [[ ! -d "$POINTCLOUD_ROOT" ]]; then
    echo "Pointcloud root not found: $POINTCLOUD_ROOT" >&2
    exit 1
fi

run_method() {
    local method="$1"
    local run_name="$2"
    shift 2

    echo "[Tune6000] Running: $method ($run_name)"
    "$PYTHON_BIN" "$REPO_ROOT/scripts/run_single_dynamic_shared.py" \
        --pointcloud-root "$POINTCLOUD_ROOT" \
        --out-dir "$OUT_DIR" \
        --method "$method" \
        --run-name "$run_name" \
        --dynamic-train-steps 6000 \
        "$@"
}

run_method "动态共享-连续形变场" "continuous-history-lite-6000" \
    --dynamic-mesh-resolution 72 \
    --max-points-per-phase 3000 \
    --bootstrap-offset-weight 0.12 \
    --bootstrap-decay-fraction 0.30 \
    --unsupported-anchor-weight 0.03 \
    --unsupported-laplacian-weight 0.08 \
    --temporal-weight 0.02 \
    --temporal-acceleration-weight 0.01 \
    --phase-consistency-weight 0.01 \
    --correspondence-temporal-weight 0.01 \
    --correspondence-acceleration-weight 0.005 \
    --correspondence-phase-consistency-weight 0.005 \
    --correspondence-start-fraction 0.4 \
    --correspondence-ramp-fraction 0.2 \
    --periodicity-weight 0.02 \
    --deformation-weight 0.002 \
    --confidence-floor 1.0 \
    --smoothing-iterations 10

run_method "动态共享-连续形变场" "continuous-history-lite-smallnet-6000" \
    --dynamic-mesh-resolution 72 \
    --max-points-per-phase 3000 \
    --canonical-hidden-dim 96 \
    --canonical-hidden-layers 4 \
    --deformation-hidden-dim 96 \
    --deformation-hidden-layers 3 \
    --bootstrap-offset-weight 0.12 \
    --bootstrap-decay-fraction 0.30 \
    --unsupported-anchor-weight 0.03 \
    --unsupported-laplacian-weight 0.08 \
    --temporal-weight 0.02 \
    --temporal-acceleration-weight 0.01 \
    --phase-consistency-weight 0.01 \
    --correspondence-temporal-weight 0.01 \
    --correspondence-acceleration-weight 0.005 \
    --correspondence-phase-consistency-weight 0.005 \
    --correspondence-start-fraction 0.4 \
    --correspondence-ramp-fraction 0.2 \
    --periodicity-weight 0.02 \
    --deformation-weight 0.002 \
    --confidence-floor 1.0 \
    --smoothing-iterations 10

run_method "动态共享-连续形变场" "continuous-history-lite-delayed-6000" \
    --dynamic-mesh-resolution 72 \
    --max-points-per-phase 3000 \
    --bootstrap-offset-weight 0.12 \
    --bootstrap-decay-fraction 0.30 \
    --unsupported-anchor-weight 0.03 \
    --unsupported-laplacian-weight 0.08 \
    --temporal-weight 0.02 \
    --temporal-acceleration-weight 0.01 \
    --phase-consistency-weight 0.01 \
    --correspondence-temporal-weight 0.01 \
    --correspondence-acceleration-weight 0.005 \
    --correspondence-phase-consistency-weight 0.005 \
    --correspondence-start-fraction 0.45 \
    --correspondence-ramp-fraction 0.25 \
    --periodicity-weight 0.02 \
    --deformation-weight 0.002 \
    --confidence-floor 1.0 \
    --smoothing-iterations 10

run_method "动态共享-解耦运动潜码" "decoupled-motion-balanced-6000" \
    --dynamic-mesh-resolution 72 \
    --max-points-per-phase 3000 \
    --confidence-floor 1.0 \
    --temporal-weight 0.02 \
    --temporal-acceleration-weight 0.01 \
    --phase-consistency-weight 0.01 \
    --correspondence-temporal-weight 0.01 \
    --correspondence-acceleration-weight 0.005 \
    --correspondence-phase-consistency-weight 0.005 \
    --correspondence-start-fraction 0.4 \
    --correspondence-ramp-fraction 0.2 \
    --periodicity-weight 0.02 \
    --deformation-weight 0.002 \
    --bootstrap-offset-weight 0.12 \
    --bootstrap-decay-fraction 0.30 \
    --unsupported-anchor-weight 0.03 \
    --unsupported-laplacian-weight 0.08 \
    --motion-mean-weight 0.04 \
    --motion-lipschitz-weight 0.03 \
    --smoothing-iterations 10

run_method "动态共享-解耦运动潜码" "decoupled-motion-small-latent-6000" \
    --dynamic-mesh-resolution 72 \
    --max-points-per-phase 3000 \
    --motion-latent-dim 32 \
    --canonical-hidden-dim 96 \
    --canonical-hidden-layers 4 \
    --deformation-hidden-dim 96 \
    --deformation-hidden-layers 3 \
    --confidence-floor 1.0 \
    --temporal-weight 0.02 \
    --temporal-acceleration-weight 0.01 \
    --phase-consistency-weight 0.01 \
    --correspondence-temporal-weight 0.01 \
    --correspondence-acceleration-weight 0.005 \
    --correspondence-phase-consistency-weight 0.005 \
    --correspondence-start-fraction 0.4 \
    --correspondence-ramp-fraction 0.2 \
    --periodicity-weight 0.02 \
    --deformation-weight 0.002 \
    --bootstrap-offset-weight 0.12 \
    --bootstrap-decay-fraction 0.30 \
    --unsupported-anchor-weight 0.03 \
    --unsupported-laplacian-weight 0.08 \
    --motion-mean-weight 0.04 \
    --motion-lipschitz-weight 0.03 \
    --smoothing-iterations 10

run_method "动态共享-解耦运动潜码" "decoupled-motion-wide-latent-6000" \
    --dynamic-mesh-resolution 72 \
    --max-points-per-phase 3000 \
    --motion-latent-dim 96 \
    --canonical-hidden-dim 160 \
    --canonical-hidden-layers 4 \
    --deformation-hidden-dim 160 \
    --deformation-hidden-layers 3 \
    --confidence-floor 1.0 \
    --temporal-weight 0.02 \
    --temporal-acceleration-weight 0.01 \
    --phase-consistency-weight 0.01 \
    --correspondence-temporal-weight 0.01 \
    --correspondence-acceleration-weight 0.005 \
    --correspondence-phase-consistency-weight 0.005 \
    --correspondence-start-fraction 0.4 \
    --correspondence-ramp-fraction 0.2 \
    --periodicity-weight 0.02 \
    --deformation-weight 0.002 \
    --bootstrap-offset-weight 0.12 \
    --bootstrap-decay-fraction 0.30 \
    --unsupported-anchor-weight 0.03 \
    --unsupported-laplacian-weight 0.08 \
    --motion-mean-weight 0.04 \
    --motion-lipschitz-weight 0.03 \
    --smoothing-iterations 10