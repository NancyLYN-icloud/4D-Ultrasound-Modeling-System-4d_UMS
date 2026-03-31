#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/liuyanan/program/project/pcd/getMesh/4D-model/4D-Ultrasound-Modeling-System-4D-UMS-"
BASE_SCRIPT="$ROOT_DIR/experiments/run_sci_q1_main_table.sh"
OUT_ROOT="${OUT_ROOT:-/home/liuyanan/data/Research_Data/4D-UMS/experiment/controlled_observation_robustness_benchmark}"
RUN_LABEL_BASE="${RUN_LABEL_BASE:-equal_budget_robustness}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LAUNCH_MODE="${LAUNCH_MODE:-background}"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/launcher_logs}"

CONDITION_SHARDS=("Clean" "Sparse" "PoseNoise")

mkdir -p "$LOG_ROOT"

echo "[ShardByCondition] out_root=$OUT_ROOT"
echo "[ShardByCondition] run_timestamp=$RUN_TIMESTAMP"
echo "[ShardByCondition] launch_mode=$LAUNCH_MODE"

for condition in "${CONDITION_SHARDS[@]}"; do
  safe_condition=$(printf '%s' "$condition" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g')
  shard_label="${RUN_LABEL_BASE}_${safe_condition}"
  log_path="$LOG_ROOT/${RUN_TIMESTAMP}_${safe_condition}.log"
  cmd=(
    env
    "OUT_ROOT=$OUT_ROOT"
    "RUN_TIMESTAMP=$RUN_TIMESTAMP"
    "RUN_LABEL=$shard_label"
    "CONDITIONS=$condition"
    bash
    "$BASE_SCRIPT"
  )

  echo "[ShardByCondition] shard=$condition log=$log_path"
  if [[ "$LAUNCH_MODE" == "background" ]]; then
    "${cmd[@]}" > "$log_path" 2>&1 &
    echo "[ShardByCondition] started pid=$! condition=$condition"
  else
    "${cmd[@]}" | tee "$log_path"
  fi
done

if [[ "$LAUNCH_MODE" == "background" ]]; then
  echo "[ShardByCondition] background launch complete"
fi