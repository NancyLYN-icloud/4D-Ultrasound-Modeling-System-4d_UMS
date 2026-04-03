#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/liuyanan/program/environment/miniconda3/envs/modeling_py310/bin/python}"
DATA_ROOT="${UMS_DATA_ROOT:-/home/liuyanan/data/Research_Data/4D-UMS}"

PHASE_COUNT="${PHASE_COUNT:-}"
NARROWING_SCALE="${NARROWING_SCALE:-1.0}"
MONITOR_PATH="${MONITOR_PATH:-}"
FPS="${FPS:-}"
DURATION_SECONDS="${DURATION_SECONDS:-}"
WRITE_PNGS="${WRITE_PNGS:-0}"
INSTANCES=()

shared_monitor_source() {
    if [[ -n "$MONITOR_PATH" ]]; then
        printf '%s\n' "$MONITOR_PATH"
    else
        printf '%s\n' "$DATA_ROOT/benchmark/monitor_stream.npz"
    fi
}

instance_monitor_target() {
    local instance_name="$1"
    if [[ "$instance_name" == "niujiao01" ]]; then
        printf '%s\n' "$DATA_ROOT/benchmark/monitor_stream.npz"
    else
        printf '%s\n' "$DATA_ROOT/benchmark/instances/${instance_name}/monitor_stream.npz"
    fi
}

usage() {
    cat <<'EOF'
Usage: build_multi_instance_dataset.sh [instance_name ...]

Batch-build phase models and scanner sequences for all reference point clouds in stomach_pcd,
or for the explicitly listed instance names.

Environment overrides:
    UMS_DATA_ROOT      Dataset root, default: /home/liuyanan/data/Research_Data/4D-UMS
  PYTHON_BIN         Python launcher, default: conda run -n modeling_py310 python
  PHASE_COUNT        Optional explicit phase count
  NARROWING_SCALE    Phase-model narrowing scale, default: 1.0
  MONITOR_PATH       Optional shared monitor stream path
  FPS                Optional scanner FPS for phase-model replay
  DURATION_SECONDS   Optional scanner duration for phase-model replay
    WRITE_PNGS         Set to 1 to export scanner PNGs during replay; default: 0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 0 ]]; then
    INSTANCES=("$@")
fi

run_phase_models() {
    local instance_name="$1"
    local -a cmd=("$REPO_ROOT/scripts/generate_phase_sequence_models.py" "--instance-name" "$instance_name" "--narrowing-scale" "$NARROWING_SCALE")
    if [[ -n "$PHASE_COUNT" ]]; then
        cmd+=("--phase-count" "$PHASE_COUNT")
    fi
    if [[ -n "$MONITOR_PATH" ]]; then
        cmd+=("--monitor-path" "$MONITOR_PATH")
    fi
    echo "[BuildDataset] Generating phase models for $instance_name"
    "$PYTHON_BIN" "${cmd[@]}"
}

sync_monitor_stream() {
    local instance_name="$1"
    local source_path
    local target_path
    source_path="$(shared_monitor_source)"
    target_path="$(instance_monitor_target "$instance_name")"

    if [[ ! -f "$source_path" ]]; then
        echo "Shared monitor stream not found: $source_path" >&2
        exit 1
    fi

    mkdir -p "$(dirname "$target_path")"
    if [[ "$source_path" != "$target_path" ]]; then
        cp -f "$source_path" "$target_path"
    fi
    echo "[BuildDataset] Synced monitor stream for $instance_name -> $target_path"
}

run_scanner_replay() {
    local instance_name="$1"
    local -a cmd=("$REPO_ROOT/scripts/generate_scanner_from_phase_models.py" "--instance-name" "$instance_name")
    if [[ -n "$MONITOR_PATH" ]]; then
        cmd+=("--monitor-path" "$MONITOR_PATH")
    fi
    if [[ -n "$FPS" || -n "$DURATION_SECONDS" ]]; then
        if [[ -z "$FPS" || -z "$DURATION_SECONDS" ]]; then
            echo "FPS and DURATION_SECONDS must be provided together" >&2
            exit 1
        fi
        cmd+=("--fps" "$FPS" "--duration-seconds" "$DURATION_SECONDS")
    fi
    if [[ "$WRITE_PNGS" != "1" ]]; then
        cmd+=("--no-png")
    fi
    echo "[BuildDataset] Replaying scanner sequence for $instance_name"
    "$PYTHON_BIN" "${cmd[@]}"
}

instance_complete() {
    local instance_name="$1"
    local scanner_path
    local mesh_dir
    local image_dir
    if [[ "$instance_name" == "niujiao01" ]]; then
        scanner_path="$DATA_ROOT/benchmark/scanner_sequence.npz"
        mesh_dir="$DATA_ROOT/simuilate_data/meshes"
        image_dir="$DATA_ROOT/benchmark/image/scanner"
    else
        scanner_path="$DATA_ROOT/benchmark/instances/${instance_name}/scanner_sequence.npz"
        mesh_dir="$DATA_ROOT/simuilate_data/instances/${instance_name}/meshes"
        image_dir="$DATA_ROOT/benchmark/instances/${instance_name}/image/scanner"
    fi

    [[ -f "$scanner_path" ]] || return 1
    [[ -d "$image_dir" ]] || return 1
    find "$mesh_dir" -maxdepth 1 -type f -name '*_mesh.ply' | grep -q .
}

resolve_instances() {
    if [[ ${#INSTANCES[@]} -gt 0 ]]; then
        printf '%s\n' "${INSTANCES[@]}"
        return
    fi
    find "$DATA_ROOT/stomach_pcd" -maxdepth 1 -type f -name '*.ply' -printf '%f\n' | sed 's/\.ply$//' | sort
}

while IFS= read -r instance_name; do
    [[ -n "$instance_name" ]] || continue
    sync_monitor_stream "$instance_name"
    if instance_complete "$instance_name"; then
        echo "[BuildDataset] Skipping completed instance $instance_name"
        continue
    fi
    run_phase_models "$instance_name"
    run_scanner_replay "$instance_name"
done < <(resolve_instances)

echo "[BuildDataset] Completed multi-instance dataset generation."