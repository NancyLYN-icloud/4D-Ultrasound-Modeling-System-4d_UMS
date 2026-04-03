#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python}"
OUT_DIR="${OUT_DIR:-/home/liuyanan/data/Research_Data/4D-UMS/experiments}"
MODE="${MODE:-dynamic-detail}"
EXPERIMENT_SET="${EXPERIMENT_SET:-both}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-multi-instance}"
INCLUDE_PRIOR_FREE="${INCLUDE_PRIOR_FREE:-0}"
INSTANCES=()

usage() {
    cat <<'EOF'
Usage: run_multi_instance_experiments.sh [instance_name ...]

Run scripts/run_experiments.py across all stomach reference instances, or the explicitly listed instances.

Environment overrides:
  PYTHON_BIN          Python launcher, default: modeling_py310 environment python
  OUT_DIR             Experiment output root
  MODE                fast-dev | dynamic-detail | full-paper
  EXPERIMENT_SET      method-comparison | cpd-ablation | both
  RUN_NAME_PREFIX     Prefix used in per-instance run names
  INCLUDE_PRIOR_FREE  Set to 1 to include the prior-free branch
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 0 ]]; then
    INSTANCES=("$@")
fi

resolve_instances() {
    if [[ ${#INSTANCES[@]} -gt 0 ]]; then
        printf '%s\n' "${INSTANCES[@]}"
        return
    fi
    find /home/liuyanan/data/Research_Data/4D-UMS/stomach_pcd -maxdepth 1 -type f -name '*.ply' -printf '%f\n' | sed 's/\.ply$//' | sort
}

while IFS= read -r instance_name; do
    [[ -n "$instance_name" ]] || continue
    cmd=(
        "$PYTHON_BIN" "$REPO_ROOT/scripts/run_experiments.py"
        --instance-name "$instance_name"
        --out-dir "$OUT_DIR"
        --mode "$MODE"
        --experiment-set "$EXPERIMENT_SET"
        --run-name "${RUN_NAME_PREFIX}-${instance_name}"
    )
    if [[ "$INCLUDE_PRIOR_FREE" == "1" ]]; then
        cmd+=(--include-prior-free)
    fi
    echo "[MultiInstanceExp] Running $instance_name"
    "${cmd[@]}"
done < <(resolve_instances)

echo "[MultiInstanceExp] Completed batch experiment execution."