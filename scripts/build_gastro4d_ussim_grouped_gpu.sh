#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_ROOT="${CONDA_ROOT:-/home/liuyanan/program/environment/miniconda3}"
CONDA_ENV="${CONDA_ENV:-modeling_py310}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ROOT/envs/$CONDA_ENV/bin/python}"

SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/home/liuyanan/data/Research_Data/4D-UMS}"
DATASET_NAME="${DATASET_NAME:-Gastro4D-USSim}"
UMS_DATA_ROOT="${UMS_DATA_ROOT:-$SOURCE_DATA_ROOT/$DATASET_NAME}"
SOURCE_GROUPS="${SOURCE_GROUPS:-stomachPCD_dev stomachPCD_01 stomachPCD_02}"
CONDITIONS="${CONDITIONS:-Sparse PoseNoise ImageNoise}"
LINK_MODE="${LINK_MODE:-copy}"
WRITE_PNGS="${WRITE_PNGS:-1}"
MATERIALIZE_IMAGE_NOISE_PNGS="${MATERIALIZE_IMAGE_NOISE_PNGS:-$WRITE_PNGS}"
SCANNER_MODE="${SCANNER_MODE:-improved}"
SCANNER_FPS="${SCANNER_FPS:-15}"
SCANNER_DURATION_SECONDS="${SCANNER_DURATION_SECONDS:-900}"
PARALLEL_GROUP_JOBS="${PARALLEL_GROUP_JOBS:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python launcher not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ -n "${GPU_ID:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

read -r -a GROUP_ARRAY <<< "$SOURCE_GROUPS"
read -r -a CONDITION_ARRAY <<< "$CONDITIONS"

run_stage_per_group() {
    local stage_label="$1"
    local script_path="$2"
    shift 2
    local -a stage_args=("$@")

    if (( PARALLEL_GROUP_JOBS <= 1 || ${#GROUP_ARRAY[@]} <= 1 )); then
        "$PYTHON_BIN" "$script_path" --groups "${GROUP_ARRAY[@]}" "${stage_args[@]}"
        return
    fi

    local -a active_pids=()
    local -a active_groups=()
    local launched_pid=""
    local finished_pid=""
    local exit_code=0

    launch_group_job() {
        local group_name="$1"
        (
            set -euo pipefail
            "$PYTHON_BIN" "$script_path" --groups "$group_name" "${stage_args[@]}"
        ) &
        launched_pid=$!
        active_pids+=("$launched_pid")
        active_groups+=("$group_name")
        echo "[Gastro4D Grouped GPU] stage=$stage_label group=$group_name pid=$launched_pid"
    }

    remove_active_job() {
        local pid_to_remove="$1"
        local -a next_pids=()
        local -a next_groups=()
        local index
        for index in "${!active_pids[@]}"; do
            if [[ "${active_pids[$index]}" == "$pid_to_remove" ]]; then
                continue
            fi
            next_pids+=("${active_pids[$index]}")
            next_groups+=("${active_groups[$index]}")
        done
        active_pids=("${next_pids[@]}")
        active_groups=("${next_groups[@]}")
    }

    wait_for_one_job() {
        finished_pid=""
        if wait -n -p finished_pid; then
            remove_active_job "$finished_pid"
            return 0
        fi
        exit_code=$?
        if [[ -n "$finished_pid" ]]; then
            remove_active_job "$finished_pid"
        fi
        return "$exit_code"
    }

    for group_name in "${GROUP_ARRAY[@]}"; do
        launch_group_job "$group_name"
        if (( ${#active_pids[@]} >= PARALLEL_GROUP_JOBS )); then
            if ! wait_for_one_job; then
                echo "[Gastro4D Grouped GPU] stage=$stage_label failed; terminating remaining group workers" >&2
                for launched_pid in "${active_pids[@]}"; do
                    kill "$launched_pid" 2>/dev/null || true
                done
                wait || true
                return "$exit_code"
            fi
        fi
    done

    for launched_pid in "${active_pids[@]}"; do
        if ! wait "$launched_pid"; then
            exit_code=$?
            echo "[Gastro4D Grouped GPU] stage=$stage_label pid=$launched_pid failed" >&2
            return "$exit_code"
        fi
    done
}

echo "[Gastro4D Grouped GPU] repo=$REPO_ROOT"
echo "[Gastro4D Grouped GPU] python=$PYTHON_BIN"
echo "[Gastro4D Grouped GPU] source_data_root=$SOURCE_DATA_ROOT"
echo "[Gastro4D Grouped GPU] dataset_root=$UMS_DATA_ROOT"
echo "[Gastro4D Grouped GPU] groups=${GROUP_ARRAY[*]}"
echo "[Gastro4D Grouped GPU] parallel_group_jobs=$PARALLEL_GROUP_JOBS"

"$PYTHON_BIN" "$REPO_ROOT/scripts/materialize_gastro4d_ussim_dataset_gpu.py" \
    --source-data-root "$SOURCE_DATA_ROOT" \
    --dataset-root "$UMS_DATA_ROOT" \
    --dataset-name "$DATASET_NAME" \
    --groups "${GROUP_ARRAY[@]}" \
    --link-mode "$LINK_MODE"

export UMS_DATA_ROOT

echo "[Gastro4D Grouped GPU] stage=simulate monitor"
run_stage_per_group "simulate-monitor" "$REPO_ROOT/scripts/generate_monitor_stream_gpu.py"

echo "[Gastro4D Grouped GPU] stage=simulate phase-models"
run_stage_per_group "simulate-phase-models" "$REPO_ROOT/scripts/generate_phase_sequence_models_gpu.py"

if [[ "$WRITE_PNGS" == "1" ]]; then
    echo "[Gastro4D Grouped GPU] stage=benchmark scanner-with-png mode=$SCANNER_MODE"
    run_stage_per_group "benchmark-scanner" "$REPO_ROOT/scripts/generate_scanner_from_phase_models_gpu.py" \
        --scanner-mode "$SCANNER_MODE" \
        --fps "$SCANNER_FPS" \
        --duration-seconds "$SCANNER_DURATION_SECONDS"
else
    echo "[Gastro4D Grouped GPU] stage=benchmark scanner-no-png mode=$SCANNER_MODE"
    run_stage_per_group "benchmark-scanner" "$REPO_ROOT/scripts/generate_scanner_from_phase_models_gpu.py" \
        --scanner-mode "$SCANNER_MODE" \
        --fps "$SCANNER_FPS" \
        --duration-seconds "$SCANNER_DURATION_SECONDS" \
        --no-png
fi

echo "[Gastro4D Grouped GPU] stage=benchmark manifest"
"$PYTHON_BIN" "$REPO_ROOT/scripts/build_benchmark_manifest_gpu.py" \
    --groups "${GROUP_ARRAY[@]}" \
    --output "$UMS_DATA_ROOT/benchmark/manifests/benchmark_manifest_gpu.csv" \
    --skip-incomplete

echo "[Gastro4D Grouped GPU] stage=benchmark conditions"
"$PYTHON_BIN" "$REPO_ROOT/scripts/generate_benchmark_conditions_gpu.py" \
    --conditions "${CONDITION_ARRAY[@]}" \
    --source-manifest "$UMS_DATA_ROOT/benchmark/manifests/benchmark_manifest_gpu.csv" \
    --condition-manifest "$UMS_DATA_ROOT/benchmark/manifests/benchmark_condition_manifest_gpu.csv" \
    --condition-root "$UMS_DATA_ROOT/benchmark/conditions" \
    "--materialize-image-noise-pngs=$MATERIALIZE_IMAGE_NOISE_PNGS" \
    --overwrite

echo "[Gastro4D Grouped GPU] completed dataset build under $UMS_DATA_ROOT"
