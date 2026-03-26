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

    echo "[HistoricalBest] Running: $method ($run_name)"
    "$PYTHON_BIN" "$REPO_ROOT/scripts/run_single_dynamic_shared.py" \
        --pointcloud-root "$POINTCLOUD_ROOT" \
        --out-dir "$OUT_DIR" \
        --method "$method" \
        --run-name "$run_name" \
        "$@"
}

# 1) RefCorr: exact best-known historical family recovered from exp_20260318_100547_refcorr_base_selection_rerun.
run_method "动态共享-参考对应正则" "refcorr-historical-best-chain" \
    --dynamic-train-steps 140 \
    --dynamic-mesh-resolution 60 \
    --max-points-per-phase 3500 \
    --canonical-hidden-dim 128 \
    --canonical-hidden-layers 4 \
    --deformation-hidden-dim 128 \
    --deformation-hidden-layers 3 \
    --confidence-floor 0.2 \
    --temporal-weight 0.10 \
    --temporal-acceleration-weight 0.05 \
    --phase-consistency-weight 0.05 \
    --correspondence-temporal-weight 0.01 \
    --correspondence-acceleration-weight 0.005 \
    --correspondence-phase-consistency-weight 0.005 \
    --correspondence-start-fraction 0.4 \
    --correspondence-ramp-fraction 0.2 \
    --periodicity-weight 0.10 \
    --deformation-weight 0.01 \
    --unsupported-anchor-weight 0.05 \
    --base-mesh-train-steps 100 \
    --smoothing-iterations 10

# 2) Continuous: best-recoverable historical configuration distilled from earlier quick-validate/search lineage.
run_method "动态共享-连续形变场" "continuous-historical-best-chain" \
    --dynamic-train-steps 10000 \
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

# 3) Decoupled motion: no earlier better run found, so use the current best-known observed configuration.
run_method "动态共享-解耦运动潜码" "decoupled-motion-best-known-chain" \
    --dynamic-train-steps 10000 \
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
    --bootstrap-offset-weight 0.25 \
    --bootstrap-decay-fraction 0.35 \
    --unsupported-anchor-weight 0.05 \
    --unsupported-laplacian-weight 0.12 \
    --motion-mean-weight 0.05 \
    --motion-lipschitz-weight 0.02 \
    --smoothing-iterations 10

# 4) Global basis residual: exact best-known historical run from exp_20260325_064637_basis15000-corrramp015-gated-replay.
run_method "动态共享-全局基残差" "global-basis-historical-best-chain" \
    --dynamic-train-steps 15000 \
    --dynamic-mesh-resolution 72 \
    --max-points-per-phase 3000 \
    --global-motion-basis-rank 6 \
    --confidence-floor 1.0 \
    --temporal-weight 0.02 \
    --temporal-acceleration-weight 0.01 \
    --phase-consistency-weight 0.01 \
    --correspondence-temporal-weight 0.016 \
    --correspondence-acceleration-weight 0.008 \
    --correspondence-phase-consistency-weight 0.008 \
    --correspondence-global-only \
    --correspondence-bootstrap-gate \
    --correspondence-bootstrap-gate-strength 2.0 \
    --correspondence-start-fraction 0.25 \
    --correspondence-ramp-fraction 0.15 \
    --periodicity-weight 0.02 \
    --deformation-weight 0.002 \
    --bootstrap-offset-weight 0.10 \
    --bootstrap-decay-fraction 0.25 \
    --basis-coefficient-bootstrap-weight 0.12 \
    --basis-temporal-weight 0.05 \
    --basis-acceleration-weight 0.025 \
    --basis-periodicity-weight 0.04 \
    --residual-mean-weight 0.02 \
    --residual-basis-projection-weight 0.04 \
    --unsupported-anchor-weight 0.02 \
    --unsupported-laplacian-weight 0.05 \
    --unsupported-propagation-iterations 8 \
    --unsupported-propagation-neighbor-weight 0.75 \
    --smoothing-iterations 8