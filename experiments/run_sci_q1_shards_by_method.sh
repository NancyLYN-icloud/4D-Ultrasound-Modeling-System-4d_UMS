#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/liuyanan/program/project/pcd/getMesh/4D-model/4D-Ultrasound-Modeling-System-4D-UMS-"
BASE_SCRIPT="$ROOT_DIR/experiments/run_sci_q1_main_table.sh"
OUT_ROOT="${OUT_ROOT:-/home/liuyanan/data/Research_Data/4D-UMS/experiment/controlled_observation_robustness_benchmark}"
RUN_LABEL_BASE="${RUN_LABEL_BASE:-equal_budget_robustness}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LAUNCH_MODE="${LAUNCH_MODE:-background}"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/launcher_logs}"

METHOD_SHARDS=(
  "refcorr:动态共享-参考对应正则"
  "continuous:动态共享-连续形变场"
  "decoupled_motion:动态共享-解耦运动潜码"
  "global_basis_residual:动态共享-全局基残差"
)

mkdir -p "$LOG_ROOT"

echo "[ShardByMethod] out_root=$OUT_ROOT"
echo "[ShardByMethod] run_timestamp=$RUN_TIMESTAMP"
echo "[ShardByMethod] launch_mode=$LAUNCH_MODE"

for shard in "${METHOD_SHARDS[@]}"; do
  slug="${shard%%:*}"
  method_name="${shard#*:}"
  shard_label="${RUN_LABEL_BASE}_${slug}"
  log_path="$LOG_ROOT/${RUN_TIMESTAMP}_${slug}.log"
  cmd=(
    env
    "OUT_ROOT=$OUT_ROOT"
    "RUN_TIMESTAMP=$RUN_TIMESTAMP"
    "RUN_LABEL=$shard_label"
    "METHOD_FILTER=$slug"
    bash
    "$BASE_SCRIPT"
  )

  echo "[ShardByMethod] shard=$method_name log=$log_path"
  if [[ "$LAUNCH_MODE" == "background" ]]; then
    "${cmd[@]}" > "$log_path" 2>&1 &
    echo "[ShardByMethod] started pid=$! method=$method_name"
  else
    "${cmd[@]}" | tee "$log_path"
  fi
done

if [[ "$LAUNCH_MODE" == "background" ]]; then
  echo "[ShardByMethod] background launch complete"
fi