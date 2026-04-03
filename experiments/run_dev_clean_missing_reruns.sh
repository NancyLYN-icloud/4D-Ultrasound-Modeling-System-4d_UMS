#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/liuyanan/program/project/pcd/getMesh/4D-model/4D-Ultrasound-Modeling-System-4D-UMS-"
PYTHON_BIN="${PYTHON_BIN:-/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python}"
MANIFEST="${MANIFEST:-$ROOT_DIR/experiments/benchmark_condition_manifest.csv}"
BATCH_DIR="${BATCH_DIR:-/home/liuyanan/data/Research_Data/4D-UMS/experiment/controlled_observation_robustness_benchmark/20260331_093708_equal_budget_robustness_dev_clean_dev_clean_full_paper}"
MODE="${MODE:-full-paper}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
MESH_RESOLUTION="${MESH_RESOLUTION:-72}"
MAX_POINTS_PER_PHASE="${MAX_POINTS_PER_PHASE:-5000}"
DRY_RUN="${DRY_RUN:-0}"

MISSING_RUNS=(
  "niujiaoxing01|动态共享-连续形变场|continuous"
  "niujiaoxing01|动态共享-解耦运动潜码|decoupled_motion"
  "niujiaoxing01|动态共享-全局基残差|global_basis_residual"
  "pubuxing01|动态共享-参考对应正则|refcorr"
  "pubuxing01|动态共享-连续形变场|continuous"
  "pubuxing01|动态共享-解耦运动潜码|decoupled_motion"
  "pubuxing01|动态共享-全局基残差|global_basis_residual"
)

append_method_specific_args() {
  local method_name="$1"
  local -n out_args_ref=$2

  if [[ "$method_name" == "动态共享-参考对应正则" ]]; then
    out_args_ref+=(
      --canonical-hidden-dim 128
      --canonical-hidden-layers 4
      --deformation-hidden-dim 128
      --deformation-hidden-layers 3
      --confidence-floor 0.2
      --temporal-weight 0.10
      --temporal-acceleration-weight 0.05
      --phase-consistency-weight 0.05
      --correspondence-temporal-weight 0.01
      --correspondence-acceleration-weight 0.005
      --correspondence-phase-consistency-weight 0.005
      --correspondence-start-fraction 0.4
      --correspondence-ramp-fraction 0.2
      --periodicity-weight 0.10
      --deformation-weight 0.01
      --unsupported-anchor-weight 0.05
      --base-mesh-train-steps 100
      --smoothing-iterations 10
    )
    return
  fi

  if [[ "$method_name" == "动态共享-连续形变场" ]]; then
    out_args_ref+=(
      --bootstrap-offset-weight 0.12
      --bootstrap-decay-fraction 0.30
      --unsupported-anchor-weight 0.03
      --unsupported-laplacian-weight 0.08
      --temporal-weight 0.02
      --temporal-acceleration-weight 0.01
      --phase-consistency-weight 0.01
      --correspondence-temporal-weight 0.01
      --correspondence-acceleration-weight 0.005
      --correspondence-phase-consistency-weight 0.005
      --correspondence-start-fraction 0.4
      --correspondence-ramp-fraction 0.2
      --periodicity-weight 0.02
      --deformation-weight 0.002
      --confidence-floor 1.0
      --smoothing-iterations 10
    )
    return
  fi

  if [[ "$method_name" == "动态共享-解耦运动潜码" ]]; then
    out_args_ref+=(
      --confidence-floor 1.0
      --temporal-weight 0.02
      --temporal-acceleration-weight 0.01
      --phase-consistency-weight 0.01
      --correspondence-temporal-weight 0.01
      --correspondence-acceleration-weight 0.005
      --correspondence-phase-consistency-weight 0.005
      --correspondence-start-fraction 0.4
      --correspondence-ramp-fraction 0.2
      --periodicity-weight 0.02
      --deformation-weight 0.002
      --bootstrap-offset-weight 0.25
      --bootstrap-decay-fraction 0.35
      --unsupported-anchor-weight 0.05
      --unsupported-laplacian-weight 0.12
      --motion-mean-weight 0.05
      --motion-lipschitz-weight 0.02
      --smoothing-iterations 10
    )
    return
  fi

  if [[ "$method_name" == "动态共享-全局基残差" ]]; then
    out_args_ref+=(
      --global-motion-basis-rank 6
      --confidence-floor 1.0
      --temporal-weight 0.02
      --temporal-acceleration-weight 0.01
      --phase-consistency-weight 0.01
      --correspondence-temporal-weight 0.016
      --correspondence-acceleration-weight 0.008
      --correspondence-phase-consistency-weight 0.008
      --correspondence-global-only
      --correspondence-bootstrap-gate
      --correspondence-bootstrap-gate-strength 2.0
      --correspondence-start-fraction 0.25
      --correspondence-ramp-fraction 0.15
      --periodicity-weight 0.02
      --deformation-weight 0.002
      --bootstrap-offset-weight 0.10
      --bootstrap-decay-fraction 0.25
      --basis-coefficient-bootstrap-weight 0.12
      --basis-temporal-weight 0.05
      --basis-acceleration-weight 0.025
      --basis-periodicity-weight 0.04
      --residual-mean-weight 0.02
      --residual-basis-projection-weight 0.04
      --unsupported-anchor-weight 0.02
      --unsupported-laplacian-weight 0.05
      --unsupported-propagation-iterations 8
      --unsupported-propagation-neighbor-weight 0.75
      --smoothing-iterations 8
    )
  fi
}

lookup_manifest_row() {
  local instance_name="$1"
  python3 - "$MANIFEST" "$instance_name" <<'PY'
import csv
import sys

manifest_path, instance_name = sys.argv[1], sys.argv[2]
with open(manifest_path, encoding='utf-8') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        if row['instance_name'] == instance_name and row['split'] == 'dev' and row['condition'] == 'Clean':
            print(row['monitor_stream'])
            print(row['scanner_sequence'])
            print(row['phase_model_dir'])
            raise SystemExit(0)
raise SystemExit(f'manifest row not found for {instance_name} dev Clean')
PY
}

for item in "${MISSING_RUNS[@]}"; do
  IFS='|' read -r instance_name method_name method_slug <<< "$item"
  run_name="${instance_name}_${method_slug}_dev_clean_eqbudget"

  if find "$BATCH_DIR" -maxdepth 2 -path "*/dynamic_shared_result.json" | grep -q "_${run_name}/dynamic_shared_result.json\|${run_name}/dynamic_shared_result.json"; then
    echo "[DevCleanRerun] skip completed $run_name"
    continue
  fi

  mapfile -t manifest_fields < <(lookup_manifest_row "$instance_name")
  monitor_stream="${manifest_fields[0]}"
  scanner_sequence="${manifest_fields[1]}"
  phase_model_dir="${manifest_fields[2]}"
  gt_mesh_dir="$phase_model_dir/pointclouds/meshes"

  method_extra_args=()
  append_method_specific_args "$method_name" method_extra_args

  cmd=(
    "$PYTHON_BIN"
    "$ROOT_DIR/scripts/run_single_dynamic_shared.py"
    --mode "$MODE"
    --method "$method_name"
    --instance-name "$instance_name"
    --monitor-path "$monitor_stream"
    --scanner-path "$scanner_sequence"
    --gt-mesh-path "$gt_mesh_dir"
    --out-dir "$BATCH_DIR"
    --run-name "$run_name"
    --dynamic-train-steps "$TRAIN_STEPS"
    --dynamic-mesh-resolution "$MESH_RESOLUTION"
    --max-points-per-phase "$MAX_POINTS_PER_PHASE"
    "${method_extra_args[@]}"
  )

  echo "[DevCleanRerun] run_name=$run_name"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DevCleanRerun][DRY_RUN] %q ' "${cmd[@]}"
    printf '\n'
    continue
  fi

  "${cmd[@]}"
done

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

echo "[DevCleanRerun] aggregating $BATCH_DIR"
"$PYTHON_BIN" "$ROOT_DIR/scripts/aggregate_dynamic_shared_results.py" \
  --runs-root "$BATCH_DIR" \
  --manifest "$MANIFEST" \
  --output-dir "$BATCH_DIR/aggregated"