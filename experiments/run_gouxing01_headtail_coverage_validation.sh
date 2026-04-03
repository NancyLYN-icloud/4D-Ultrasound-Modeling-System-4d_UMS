#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/liuyanan/program/project/pcd/getMesh/4D-model/4D-Ultrasound-Modeling-System-4D-UMS-"
PYTHON_BIN="${PYTHON_BIN:-/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/home/liuyanan/data/Research_Data/4D-UMS/experiment/headtail_coverage_validation}"
VALIDATION_ROOT="$OUT_ROOT/${RUN_TIMESTAMP}_gouxing01_headtail_dense"
INSTANCE_NAME="gouxing01"

append_method_specific_args() {
  local method_name="$1"
  local -n out_args_ref=$2

  if [[ "$method_name" == "动态共享-参考对应正则" ]]; then
    out_args_ref+=(
      --canonical-hidden-dim 128 --canonical-hidden-layers 4 --deformation-hidden-dim 128 --deformation-hidden-layers 3
      --confidence-floor 0.2 --temporal-weight 0.10 --temporal-acceleration-weight 0.05 --phase-consistency-weight 0.05
      --correspondence-temporal-weight 0.01 --correspondence-acceleration-weight 0.005 --correspondence-phase-consistency-weight 0.005
      --correspondence-start-fraction 0.4 --correspondence-ramp-fraction 0.2 --periodicity-weight 0.10 --deformation-weight 0.01
      --unsupported-anchor-weight 0.05 --base-mesh-train-steps 100 --smoothing-iterations 10
    )
    return
  fi
  if [[ "$method_name" == "动态共享-解耦运动潜码" ]]; then
    out_args_ref+=(
      --confidence-floor 1.0 --temporal-weight 0.02 --temporal-acceleration-weight 0.01 --phase-consistency-weight 0.01
      --correspondence-temporal-weight 0.01 --correspondence-acceleration-weight 0.005 --correspondence-phase-consistency-weight 0.005
      --correspondence-start-fraction 0.4 --correspondence-ramp-fraction 0.2 --periodicity-weight 0.02 --deformation-weight 0.002
      --bootstrap-offset-weight 0.25 --bootstrap-decay-fraction 0.35 --unsupported-anchor-weight 0.05 --unsupported-laplacian-weight 0.12
      --motion-mean-weight 0.05 --motion-lipschitz-weight 0.02 --smoothing-iterations 10
    )
    return
  fi
  if [[ "$method_name" == "动态共享-全局基残差" ]]; then
    out_args_ref+=(
      --global-motion-basis-rank 6 --confidence-floor 1.0 --temporal-weight 0.02 --temporal-acceleration-weight 0.01 --phase-consistency-weight 0.01
      --correspondence-temporal-weight 0.016 --correspondence-acceleration-weight 0.008 --correspondence-phase-consistency-weight 0.008
      --correspondence-global-only --correspondence-bootstrap-gate --correspondence-bootstrap-gate-strength 2.0
      --correspondence-start-fraction 0.25 --correspondence-ramp-fraction 0.15 --periodicity-weight 0.02 --deformation-weight 0.002
      --bootstrap-offset-weight 0.10 --bootstrap-decay-fraction 0.25 --basis-coefficient-bootstrap-weight 0.12
      --basis-temporal-weight 0.05 --basis-acceleration-weight 0.025 --basis-periodicity-weight 0.04
      --residual-mean-weight 0.02 --residual-basis-projection-weight 0.04 --unsupported-anchor-weight 0.02 --unsupported-laplacian-weight 0.05
      --unsupported-propagation-iterations 8 --unsupported-propagation-neighbor-weight 0.75 --smoothing-iterations 8
    )
    return
  fi
}

echo "[HeadTailValidation] validation_root=$VALIDATION_ROOT"
mkdir -p "$VALIDATION_ROOT"

echo "[HeadTailValidation] Step 1: generate clean head/tail-dense scanner variant"
"$PYTHON_BIN" "$ROOT_DIR/scripts/generate_headtail_dense_scanner_variant.py" \
  --instance-name "$INSTANCE_NAME" \
  --output-root "$VALIDATION_ROOT"

echo "[HeadTailValidation] Step 2: run coverage diagnostic on original vs regenerated clean scanner"
"$PYTHON_BIN" "$ROOT_DIR/scripts/diagnose_scanner_headtail_coverage.py" \
  --instance-name "$INSTANCE_NAME" \
  --baseline-scanner "/home/liuyanan/data/Research_Data/4D-UMS/benchmark/instances/${INSTANCE_NAME}/scanner_sequence.npz" \
  --variant-scanner "$VALIDATION_ROOT/clean/scanner_sequence.npz" \
  --output-dir "$VALIDATION_ROOT/coverage_diagnostics" \
  --baseline-label "original-clean" \
  --variant-label "headtail-dense-clean"

echo "[HeadTailValidation] Step 3: derive sparse and posenoise from the new clean scanner"
"$PYTHON_BIN" - <<'PY' "$VALIDATION_ROOT" "$ROOT_DIR"
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[2])

from scripts.generate_benchmark_conditions import (
    BenchmarkInstanceRow,
    _build_pose_noise_condition,
    _build_sparse_condition,
)
from src.stomach_instance_paths import resolve_instance_paths

validation_root = Path(sys.argv[1])
instance_name = "gouxing01"
instance_paths = resolve_instance_paths(instance_name=instance_name)
phase_model_dir = sorted(path for path in instance_paths.phase_model_base_dir.iterdir() if path.is_dir() and path.name.startswith("phase_sequence_models_run_"))[-1]
clean_root = validation_root / "clean"

monitor_image_dst = clean_root / "image" / "monitor"
monitor_image_src = instance_paths.test_root / "image" / "monitor"
if not monitor_image_dst.exists():
  shutil.copytree(monitor_image_src, monitor_image_dst, dirs_exist_ok=True)

row = BenchmarkInstanceRow(
    instance_name=instance_name,
    shape_family="gouxing",
    split="dev",
    reference_ply=instance_paths.reference_ply,
    monitor_stream=clean_root / "monitor_stream.npz",
    scanner_sequence=clean_root / "scanner_sequence.npz",
    phase_model_dir=phase_model_dir,
    phase_summary=phase_model_dir / "phase_sequence_summary.csv",
)

condition_root = validation_root / "conditions"
_build_sparse_condition(row=row, condition_root=condition_root, sparse_step=4, sparse_offset_mode="hash", overwrite=True)
_build_pose_noise_condition(row=row, condition_root=condition_root, translation_noise_mm=2.5, rotation_noise_deg=4.0, noise_sigma_frames=18.0, seed=20260331, overwrite=True)

print(f"[HeadTailValidation] Clean root: {clean_root}")
print(f"[HeadTailValidation] Sparse root: {condition_root / 'sparse' / 'instances' / instance_name}")
print(f"[HeadTailValidation] PoseNoise root: {condition_root / 'pose_noise' / 'instances' / instance_name}")
PY

run_case() {
  local condition_label="$1"
  local method_name="$2"
  local monitor_path="$3"
  local scanner_path="$4"
  local out_dir="$5"
  local run_name="$6"
  local -a extra_args=()
  append_method_specific_args "$method_name" extra_args
  "$PYTHON_BIN" "$ROOT_DIR/scripts/run_single_dynamic_shared.py" \
    --mode full-paper \
    --method "$method_name" \
    --instance-name "$INSTANCE_NAME" \
    --monitor-path "$monitor_path" \
    --scanner-path "$scanner_path" \
    --out-dir "$out_dir" \
    --run-name "$run_name" \
    --dynamic-train-steps 10000 \
    --dynamic-mesh-resolution 72 \
    --max-points-per-phase 5000 \
    "${extra_args[@]}"
}

CLEAN_ROOT="$VALIDATION_ROOT/clean"
SPARSE_ROOT="$VALIDATION_ROOT/conditions/sparse/instances/$INSTANCE_NAME"
POSENOISE_ROOT="$VALIDATION_ROOT/conditions/pose_noise/instances/$INSTANCE_NAME"
RUN_OUT="$VALIDATION_ROOT/runs"
mkdir -p "$RUN_OUT"

echo "[HeadTailValidation] Step 4: run selected validations"
run_case "clean" "动态共享-全局基残差" "$CLEAN_ROOT/monitor_stream.npz" "$CLEAN_ROOT/scanner_sequence.npz" "$RUN_OUT" "gouxing01_headtaildense_clean_global_basis_residual"
run_case "clean" "动态共享-解耦运动潜码" "$CLEAN_ROOT/monitor_stream.npz" "$CLEAN_ROOT/scanner_sequence.npz" "$RUN_OUT" "gouxing01_headtaildense_clean_decoupled_motion"
run_case "sparse" "动态共享-解耦运动潜码" "$SPARSE_ROOT/monitor_stream.npz" "$SPARSE_ROOT/scanner_sequence.npz" "$RUN_OUT" "gouxing01_headtaildense_sparse_decoupled_motion"
run_case "posenoise" "动态共享-全局基残差" "$POSENOISE_ROOT/monitor_stream.npz" "$POSENOISE_ROOT/scanner_sequence.npz" "$RUN_OUT" "gouxing01_headtaildense_posenoise_global_basis_residual"
run_case "posenoise" "动态共享-参考对应正则" "$POSENOISE_ROOT/monitor_stream.npz" "$POSENOISE_ROOT/scanner_sequence.npz" "$RUN_OUT" "gouxing01_headtaildense_posenoise_refcorr"

echo "[HeadTailValidation] done"