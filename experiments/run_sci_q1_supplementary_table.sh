#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-/home/tianjun0/anaconda3/envs/modeling_py310/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/tianjun0/liuyanan/data/4D-UMS/Gastro4D-USSim}"
MANIFEST="${MANIFEST:-}"
OUT_ROOT="${OUT_ROOT:-/home/tianjun0/liuyanan/data/4D-UMS/experiment}"
SUITE_NAME="${SUITE_NAME:-gastro4d_ussim_supplementary}"
RUN_LABEL="${RUN_LABEL:-supplementary_gpu0}"
SPLIT="${SPLIT:-test}"
CONDITIONS="${CONDITIONS:-Clean}"
MODE="${MODE:-full-paper}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
MESH_RESOLUTION="${MESH_RESOLUTION:-72}"
MAX_POINTS_PER_PHASE="${MAX_POINTS_PER_PHASE:-5000}"
METHOD_PROFILE="${METHOD_PROFILE:-historical_best_eqbudget}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -z "$MANIFEST" ]]; then
  for candidate in \
    "$DATA_ROOT/benchmark/manifests/benchmark_condition_manifest_gpu.csv" \
    "$DATA_ROOT/benchmark/manifests/benchmark_condition_manifest.csv" \
    "$ROOT_DIR/experiments/benchmark_condition_manifest.csv"; do
    if [[ -f "$candidate" ]]; then
      MANIFEST="$candidate"
      break
    fi
  done
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "[SupplementaryBenchmark] benchmark manifest not found -> $MANIFEST" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[SupplementaryBenchmark] python executable not found -> $PYTHON_BIN" >&2
  exit 1
fi

IFS=',' read -r -a CONDITION_ARGS <<< "$CONDITIONS"

echo "[SupplementaryBenchmark] manifest=$MANIFEST"
echo "[SupplementaryBenchmark] out_root=$OUT_ROOT"
echo "[SupplementaryBenchmark] suite_name=$SUITE_NAME"
echo "[SupplementaryBenchmark] run_label=$RUN_LABEL"
echo "[SupplementaryBenchmark] split=$SPLIT conditions=$CONDITIONS"
echo "[SupplementaryBenchmark] method_profile=$METHOD_PROFILE"
echo "[SupplementaryBenchmark] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" "$ROOT_DIR/scripts/run_benchmark_suite.py" \
  --manifest "$MANIFEST" \
  --data-root "$DATA_ROOT" \
  --out-root "$OUT_ROOT" \
  --suite-name "$SUITE_NAME" \
  --run-label "$RUN_LABEL" \
  --split "$SPLIT" \
  --conditions "${CONDITION_ARGS[@]}" \
  --methods supplementary-baselines \
  --method-profile "$METHOD_PROFILE" \
  --mode "$MODE" \
  --dynamic-train-steps "$TRAIN_STEPS" \
  --dynamic-mesh-resolution "$MESH_RESOLUTION" \
  --max-points-per-phase "$MAX_POINTS_PER_PHASE"