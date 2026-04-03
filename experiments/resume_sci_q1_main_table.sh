#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/liuyanan/program/project/pcd/getMesh/4D-model/4D-Ultrasound-Modeling-System-4D-UMS-"
PYTHON_BIN="/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python"
MANIFEST="${MANIFEST:-$ROOT_DIR/experiments/benchmark_condition_manifest.csv}"
OUT_ROOT="${OUT_ROOT:-/home/liuyanan/data/Research_Data/4D-UMS/experiment/controlled_observation_robustness_benchmark}"
SPLIT="${SPLIT:-test}"
CONDITIONS="${CONDITIONS:-Clean}"
MODE="${MODE:-full-paper}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
MESH_RESOLUTION="${MESH_RESOLUTION:-72}"
MAX_POINTS_PER_PHASE="${MAX_POINTS_PER_PHASE:-5000}"
RUN_LABEL="${RUN_LABEL:-equal_budget_robustness}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
METHOD_FILTER="${METHOD_FILTER:-all}"
METHOD_PROFILE="${METHOD_PROFILE:-historical_best_eqbudget}"

METHODS=(
  "动态共享-参考对应正则"
  "动态共享-连续形变场"
  "动态共享-解耦运动潜码"
  "动态共享-全局基残差"
)

sanitize_label() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g; s/__*/_/g; s/^_//; s/_$//'
}

SAFE_SPLIT=$(sanitize_label "$SPLIT")
SAFE_CONDITIONS=$(sanitize_label "$CONDITIONS")
SAFE_MODE=$(sanitize_label "$MODE")
SAFE_RUN_LABEL=$(sanitize_label "$RUN_LABEL")

BATCH_DIR="$OUT_ROOT/${RUN_TIMESTAMP}_${SAFE_RUN_LABEL}_${SAFE_SPLIT}_${SAFE_CONDITIONS}_${SAFE_MODE}"

mkdir -p "$BATCH_DIR"

cp "$MANIFEST" "$BATCH_DIR/benchmark_condition_manifest_snapshot.csv"
cat > "$BATCH_DIR/batch_metadata.txt" <<EOF
run_timestamp=$RUN_TIMESTAMP
run_label=$RUN_LABEL
split=$SPLIT
conditions=$CONDITIONS
mode=$MODE
dynamic_train_steps=$TRAIN_STEPS
dynamic_mesh_resolution=$MESH_RESOLUTION
max_points_per_phase=$MAX_POINTS_PER_PHASE
manifest=$MANIFEST
batch_dir=$BATCH_DIR
method_profile=$METHOD_PROFILE
resume_mode=true
EOF

echo "[RobustnessBenchmarkResume] manifest=$MANIFEST"
echo "[RobustnessBenchmarkResume] batch_dir=$BATCH_DIR"
echo "[RobustnessBenchmarkResume] split=$SPLIT conditions=$CONDITIONS"
echo "[RobustnessBenchmarkResume] method_filter=$METHOD_FILTER"

condition_enabled() {
  local value="$1"
  local item
  IFS=',' read -r -a enabled_conditions <<< "$CONDITIONS"
  for item in "${enabled_conditions[@]}"; do
    if [[ "$item" == "$value" ]]; then
      return 0
    fi
  done
  return 1
}

method_slug() {
  printf '%s' "$1" | sed 's/动态共享-//g; s/动态共享/shared/g; s/参考对应正则/refcorr/g; s/连续形变场/continuous/g; s/解耦运动潜码/decoupled_motion/g; s/全局基残差/global_basis_residual/g'
}

method_enabled() {
  local method_name="$1"
  local normalized_filter normalized_name normalized_slug item
  if [[ "$METHOD_FILTER" == "all" ]]; then
    return 0
  fi

  normalized_name=$(sanitize_label "$method_name")
  normalized_slug=$(sanitize_label "$(method_slug "$method_name")")
  IFS=',' read -r -a enabled_methods <<< "$METHOD_FILTER"
  for item in "${enabled_methods[@]}"; do
    normalized_filter=$(sanitize_label "$item")
    if [[ "$normalized_filter" == "$normalized_name" || "$normalized_filter" == "$normalized_slug" ]]; then
      return 0
    fi
  done
  return 1
}

append_method_specific_args() {
  local method_name="$1"
  local -n out_args_ref=$2

  if [[ "$METHOD_PROFILE" != "historical_best_eqbudget" ]]; then
    return
  fi

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

has_completed_result() {
  local run_name="$1"
  find "$BATCH_DIR" -path "*/exp_*_${run_name}/dynamic_shared_result.json" -print -quit | grep -q .
}

tail -n +2 "$MANIFEST" | while IFS=, read -r instance_name shape_family split condition reference_ply monitor_stream scanner_sequence phase_model_dir phase_summary condition_root condition_metadata; do
  if [[ "$split" != "$SPLIT" ]]; then
    continue
  fi
  if ! condition_enabled "$condition"; then
    continue
  fi

  gt_mesh_dir="$phase_model_dir/pointclouds/meshes"
  if [[ ! -d "$gt_mesh_dir" ]]; then
    echo "[SCI-Q1-Resume] Skip $instance_name: GT mesh dir missing -> $gt_mesh_dir" >&2
    continue
  fi
  if [[ ! -f "$monitor_stream" ]]; then
    echo "[SCI-Q1-Resume] Skip $instance_name/$condition: monitor stream missing -> $monitor_stream" >&2
    continue
  fi
  if [[ ! -f "$scanner_sequence" ]]; then
    echo "[SCI-Q1-Resume] Skip $instance_name/$condition: scanner sequence missing -> $scanner_sequence" >&2
    continue
  fi

  safe_condition=$(printf '%s' "$condition" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g')

  for method in "${METHODS[@]}"; do
    if ! method_enabled "$method"; then
      continue
    fi

    safe_method=$(method_slug "$method")
    run_name="${instance_name}_${safe_method}_${SPLIT}_${safe_condition}_eqbudget"

    if has_completed_result "$run_name"; then
      echo "[SCI-Q1-Resume] Skip completed run -> $run_name"
      continue
    fi

    method_extra_args=()
    append_method_specific_args "$method" method_extra_args

    echo "[SCI-Q1-Resume] resume instance=$instance_name split=$split condition=$condition method=$method"
    "$PYTHON_BIN" "$ROOT_DIR/scripts/run_single_dynamic_shared.py" \
      --mode "$MODE" \
      --method "$method" \
      --instance-name "$instance_name" \
      --monitor-path "$monitor_stream" \
      --scanner-path "$scanner_sequence" \
      --gt-mesh-path "$gt_mesh_dir" \
      --out-dir "$BATCH_DIR" \
      --run-name "$run_name" \
      --dynamic-train-steps "$TRAIN_STEPS" \
      --dynamic-mesh-resolution "$MESH_RESOLUTION" \
      --max-points-per-phase "$MAX_POINTS_PER_PHASE" \
      "${method_extra_args[@]}"
  done
done

AGGREGATED_DIR="$BATCH_DIR/aggregated"
echo "[RobustnessBenchmarkResume] aggregating results into $AGGREGATED_DIR"
"$PYTHON_BIN" "$ROOT_DIR/scripts/aggregate_dynamic_shared_results.py" \
  --runs-root "$BATCH_DIR" \
  --manifest "$MANIFEST" \
  --output-dir "$AGGREGATED_DIR"

echo "[RobustnessBenchmarkResume] completed batch_dir=$BATCH_DIR"