#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/liuyanan/program/project/pcd/getMesh/4D-model/4D-Ultrasound-Modeling-System-4D-UMS-"
PYTHON_BIN="${PYTHON_BIN:-/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python}"
MANIFEST="${MANIFEST:-$ROOT_DIR/experiments/benchmark_condition_manifest.csv}"
OUT_ROOT="${OUT_ROOT:-/home/liuyanan/data/Research_Data/4D-UMS/experiment/controlled_observation_robustness_benchmark}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_LABEL="${RUN_LABEL:-continuous_dev_sparse_grid}"
MODE="${MODE:-full-paper}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
MESH_RESOLUTION="${MESH_RESOLUTION:-72}"
MAX_POINTS_PER_PHASE="${MAX_POINTS_PER_PHASE:-5000}"
INSTANCE_FILTER="${INSTANCE_FILTER:-all}"
MAX_CONFIGS="${MAX_CONFIGS:-0}"
DRY_RUN="${DRY_RUN:-0}"

CORR_SCHEDULES_CSV="${CORR_SCHEDULES:-0.40:0.20,0.55:0.30}"
BOOTSTRAP_PROFILES_CSV="${BOOTSTRAP_PROFILES:-base,strong}"
REGULARIZATION_PROFILES_CSV="${REGULARIZATION_PROFILES:-base,strong}"

METHOD_NAME="动态共享-连续形变场"
SAFE_MODE=$(printf '%s' "$MODE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g; s/__*/_/g; s/^_//; s/_$//')
SAFE_RUN_LABEL=$(printf '%s' "$RUN_LABEL" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g; s/__*/_/g; s/^_//; s/_$//')
BATCH_DIR="$OUT_ROOT/${RUN_TIMESTAMP}_${SAFE_RUN_LABEL}_dev_sparse_${SAFE_MODE}"
AGGREGATED_DIR="$BATCH_DIR/aggregated"
GRID_CONFIG_PATH="$BATCH_DIR/grid_configs.csv"

mkdir -p "$BATCH_DIR"
cp "$MANIFEST" "$BATCH_DIR/benchmark_condition_manifest_snapshot.csv"

cat > "$BATCH_DIR/batch_metadata.txt" <<EOF
run_timestamp=$RUN_TIMESTAMP
run_label=$RUN_LABEL
split=dev
conditions=Sparse
mode=$MODE
method=$METHOD_NAME
dynamic_train_steps=$TRAIN_STEPS
dynamic_mesh_resolution=$MESH_RESOLUTION
max_points_per_phase=$MAX_POINTS_PER_PHASE
instance_filter=$INSTANCE_FILTER
max_configs=$MAX_CONFIGS
dry_run=$DRY_RUN
correspondence_schedules=$CORR_SCHEDULES_CSV
bootstrap_profiles=$BOOTSTRAP_PROFILES_CSV
regularization_profiles=$REGULARIZATION_PROFILES_CSV
manifest=$MANIFEST
batch_dir=$BATCH_DIR
EOF

echo "[ContinuousSparseGrid] batch_dir=$BATCH_DIR"
echo "[ContinuousSparseGrid] train_steps=$TRAIN_STEPS mesh_resolution=$MESH_RESOLUTION max_points=$MAX_POINTS_PER_PHASE"

IFS=',' read -r -a CORR_SCHEDULES <<< "$CORR_SCHEDULES_CSV"
IFS=',' read -r -a BOOTSTRAP_PROFILES <<< "$BOOTSTRAP_PROFILES_CSV"
IFS=',' read -r -a REGULARIZATION_PROFILES <<< "$REGULARIZATION_PROFILES_CSV"

instance_enabled() {
  local instance_name="$1"
  local item
  local -a enabled_instances
  if [[ "$INSTANCE_FILTER" == "all" ]]; then
    return 0
  fi
  IFS=',' read -r -a enabled_instances <<< "$INSTANCE_FILTER"
  for item in "${enabled_instances[@]}"; do
    if [[ "$item" == "$instance_name" ]]; then
      return 0
    fi
  done
  return 1
}

sanitize_number() {
  printf '%s' "$1" | sed 's/\./p/g; s/[^0-9A-Za-z_-]//g'
}

append_bootstrap_profile_args() {
  local profile="$1"
  local -n out_args_ref=$2
  case "$profile" in
    base)
      out_args_ref+=(
        --bootstrap-offset-weight 0.12
        --bootstrap-decay-fraction 0.30
      )
      ;;
    strong)
      out_args_ref+=(
        --bootstrap-offset-weight 0.24
        --bootstrap-decay-fraction 0.40
      )
      ;;
    *)
      echo "[ContinuousSparseGrid] Unknown bootstrap profile: $profile" >&2
      exit 1
      ;;
  esac
}

append_regularization_profile_args() {
  local profile="$1"
  local -n out_args_ref=$2
  case "$profile" in
    base)
      out_args_ref+=(
        --unsupported-anchor-weight 0.03
        --unsupported-laplacian-weight 0.08
        --temporal-weight 0.02
        --temporal-acceleration-weight 0.01
        --phase-consistency-weight 0.01
        --periodicity-weight 0.02
        --deformation-weight 0.002
        --smoothing-iterations 10
      )
      ;;
    strong)
      out_args_ref+=(
        --unsupported-anchor-weight 0.08
        --unsupported-laplacian-weight 0.16
        --temporal-weight 0.05
        --temporal-acceleration-weight 0.025
        --phase-consistency-weight 0.03
        --periodicity-weight 0.05
        --deformation-weight 0.0015
        --smoothing-iterations 12
      )
      ;;
    *)
      echo "[ContinuousSparseGrid] Unknown regularization profile: $profile" >&2
      exit 1
      ;;
  esac
}

echo 'config_name,correspondence_start_fraction,correspondence_ramp_fraction,bootstrap_profile,regularization_profile' > "$GRID_CONFIG_PATH"

config_count=0
for corr_schedule in "${CORR_SCHEDULES[@]}"; do
  IFS=':' read -r corr_start corr_ramp <<< "$corr_schedule"
  for bootstrap_profile in "${BOOTSTRAP_PROFILES[@]}"; do
    for regularization_profile in "${REGULARIZATION_PROFILES[@]}"; do
      config_name="cs$(sanitize_number "$corr_start")_cr$(sanitize_number "$corr_ramp")_${bootstrap_profile}_${regularization_profile}"
      echo "$config_name,$corr_start,$corr_ramp,$bootstrap_profile,$regularization_profile" >> "$GRID_CONFIG_PATH"
      config_count=$((config_count + 1))
    done
  done
done

echo "[ContinuousSparseGrid] config_count=$config_count"

run_count=0
config_index=0
while IFS=, read -r config_name corr_start corr_ramp bootstrap_profile regularization_profile; do
  if [[ "$config_name" == "config_name" ]]; then
    continue
  fi
  config_index=$((config_index + 1))
  if (( MAX_CONFIGS > 0 && config_index > MAX_CONFIGS )); then
    break
  fi

  while IFS=, read -r instance_name shape_family split condition reference_ply monitor_stream scanner_sequence phase_model_dir phase_summary condition_root condition_metadata; do
    if [[ "$split" != "dev" || "$condition" != "Sparse" ]]; then
      continue
    fi
    if ! instance_enabled "$instance_name"; then
      continue
    fi

    gt_mesh_dir="$phase_model_dir/pointclouds/meshes"
    [[ -d "$gt_mesh_dir" ]] || continue
    [[ -f "$monitor_stream" ]] || continue
    [[ -f "$scanner_sequence" ]] || continue

    run_name="${instance_name}_continuous_dev_sparse_cfg_${config_name}"
    common_args=(
      --mode "$MODE"
      --method "$METHOD_NAME"
      --instance-name "$instance_name"
      --monitor-path "$monitor_stream"
      --scanner-path "$scanner_sequence"
      --gt-mesh-path "$gt_mesh_dir"
      --out-dir "$BATCH_DIR"
      --run-name "$run_name"
      --dynamic-train-steps "$TRAIN_STEPS"
      --dynamic-mesh-resolution "$MESH_RESOLUTION"
      --max-points-per-phase "$MAX_POINTS_PER_PHASE"
      --confidence-floor 1.0
      --correspondence-temporal-weight 0.01
      --correspondence-acceleration-weight 0.005
      --correspondence-phase-consistency-weight 0.005
      --correspondence-start-fraction "$corr_start"
      --correspondence-ramp-fraction "$corr_ramp"
    )
    append_bootstrap_profile_args "$bootstrap_profile" common_args
    append_regularization_profile_args "$regularization_profile" common_args

    echo "[ContinuousSparseGrid] instance=$instance_name config=$config_name corr=${corr_start}/${corr_ramp} bootstrap=$bootstrap_profile reg=$regularization_profile"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '[ContinuousSparseGrid][DRY_RUN] %q ' "$PYTHON_BIN" "$ROOT_DIR/scripts/run_single_dynamic_shared.py" "${common_args[@]}"
      printf '\n'
      continue
    fi
    "$PYTHON_BIN" "$ROOT_DIR/scripts/run_single_dynamic_shared.py" "${common_args[@]}"
    run_count=$((run_count + 1))
  done < <(tail -n +2 "$MANIFEST")
done < "$GRID_CONFIG_PATH"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[ContinuousSparseGrid] dry run complete"
  exit 0
fi

echo "[ContinuousSparseGrid] completed_runs=$run_count"
"$PYTHON_BIN" "$ROOT_DIR/scripts/aggregate_dynamic_shared_results.py" \
  --runs-root "$BATCH_DIR" \
  --manifest "$MANIFEST" \
  --output-dir "$AGGREGATED_DIR"

"$PYTHON_BIN" - "$AGGREGATED_DIR" "$GRID_CONFIG_PATH" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

agg_dir = Path(sys.argv[1])
grid_config_path = Path(sys.argv[2])
instance_df = pd.read_csv(agg_dir / "instance_level_results.csv")
instance_df = instance_df[instance_df["method"] == "动态共享-连续形变场"].copy()
instance_df["config_name"] = instance_df["run_name"].str.split("_cfg_", n=1).str[-1]
grid_df = pd.read_csv(grid_config_path)
metrics = ["平均CD(mm^2)", "平均HD95(mm)", "时间平滑度(mm/step)", "水密比例", "平均点云置信度", "平均样本SNR", "平均切片提取率"]
summary_df = instance_df.groupby("config_name", dropna=False)[metrics].agg(["mean", "std", "min", "max"]).reset_index()
summary_df.columns = [column if isinstance(column, str) else f"{column[0]}_{column[1]}".strip("_") for column in summary_df.columns.to_flat_index()]
summary_df = grid_df.merge(summary_df, on="config_name", how="left")
summary_df = summary_df.sort_values(["平均CD(mm^2)_mean", "平均HD95(mm)_mean", "时间平滑度(mm/step)_mean"], ascending=[True, True, True]).reset_index(drop=True)
summary_df.insert(0, "rank", summary_df.index + 1)
summary_df.to_csv(agg_dir / "config_level_summary.csv", index=False)
formatted_df = summary_df[["rank", "config_name", "correspondence_start_fraction", "correspondence_ramp_fraction", "bootstrap_profile", "regularization_profile"]].copy()
for metric in metrics:
    formatted_df[metric] = summary_df.apply(lambda row: f"{row[f'{metric}_mean']:.4f} +- {row[f'{metric}_std']:.4f}" if pd.notna(row.get(f"{metric}_std")) else f"{row[f'{metric}_mean']:.4f}", axis=1)
formatted_df.to_csv(agg_dir / "config_level_summary_formatted.csv", index=False)
print(f"[ContinuousSparseGrid] Config summary: {agg_dir / 'config_level_summary.csv'}")
PY

echo "[ContinuousSparseGrid] done"