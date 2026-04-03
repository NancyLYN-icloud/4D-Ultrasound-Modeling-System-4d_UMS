# Data Pipeline Notes

This note summarizes which maintained scripts in `scripts/` read or write the three dataset roots used for large-scale data generation.

## Data Roots

- `stomach_pcd/`
  Reference stomach point clouds. New instance creation starts here.
- `benchmark/instances/`
  Clean benchmark inputs for each instance, including `monitor_stream.npz`, `scanner_sequence.npz`, and image folders.
- `benchmark/conditions/`
  Derived benchmark conditions such as `sparse`, `pose_noise`, and `image_noise`.
- `simuilate_data/`
  Phase-sequence model runs and GT mesh assets used for supervision and evaluation.

## Script Dependencies

### `simuilate_data/`

- `generate_phase_sequence_models.py`
  Main producer. Generates `phase_sequence_models_run_*` under `simuilate_data/instances/<instance>/` and syncs GT meshes.
- `generate_scanner_from_phase_models.py`
  Consumer. Replays scanner sequences from the latest or specified phase-model run under `simuilate_data/instances/<instance>/`.
- `regenerate_improved_benchmark_instances.py`
  Consumer. Reads the latest phase-model run under `simuilate_data/instances/<instance>/` and regenerates improved clean scanner streams in `benchmark/instances/`.
- `run_experiments.py`
  Consumer. Reads GT meshes from `simuilate_data/meshes/` or `simuilate_data/instances/<instance>/phase_sequence_models_run_*/pointclouds/meshes` for evaluation.
- `run_single_dynamic_shared.py`
  Consumer. Reads GT meshes from the same resolved `simuilate_data` locations when running a single dynamic method.

### `benchmark/instances/`

- `build_multi_instance_dataset.sh`
  Main orchestrator for bulk clean-data generation. Syncs monitor streams into `benchmark/instances/<instance>/`, then calls phase-model generation and scanner replay.
- `generate_scanner_from_phase_models.py`
  Producer. Writes clean `scanner_sequence.npz` and scanner PNGs into `benchmark/instances/<instance>/`.
- `regenerate_improved_benchmark_instances.py`
  Producer. Rewrites `benchmark/instances/<instance>/scanner_sequence.npz` using the improved scanner slicing logic while reusing `benchmark/instances_before/` monitor inputs.
- `generate_benchmark_conditions.py`
  Consumer. Reads clean rows from `experiments/benchmark_manifest.csv`, whose `clean_root` points to `benchmark/instances/<instance>/`.
- `run_experiments.py`
  Consumer. Reads clean monitor/scanner inputs from `benchmark/instances/<instance>/` for method-comparison and ablation runs.

### `benchmark/conditions/`

- `generate_benchmark_conditions.py`
  Main producer. Writes derived condition folders under `benchmark/conditions/{sparse,pose_noise,image_noise}/instances/<instance>/` and updates `experiments/benchmark_condition_manifest.csv`.
- `run_single_dynamic_shared.py`
  Generic consumer. When called with condition-specific `--monitor-path` and `--scanner-path`, it reads directly from `benchmark/conditions/...`.
- `run_experiments.py`
  Generic consumer. Can evaluate condition-specific datasets when passed condition-specific inputs.

## Recommended Workflow For New Data

1. Add or update reference point clouds in `stomach_pcd/`.
2. Run `build_multi_instance_dataset.sh` for initial clean-instance generation.
3. If needed, run `regenerate_improved_benchmark_instances.py` to rewrite clean scanner streams with the improved slicing strategy.
4. Run `generate_benchmark_conditions.py` to derive `Sparse`, `PoseNoise`, and `ImageNoise` benchmark conditions.
5. Run `run_experiments.py` or `run_single_dynamic_shared.py` on the clean or derived condition data.

## Retained Primary Entry Points

- `build_multi_instance_dataset.sh`
- `generate_phase_sequence_models.py`
- `generate_scanner_from_phase_models.py`
- `regenerate_improved_benchmark_instances.py`
- `generate_benchmark_conditions.py`
- `run_experiments.py`
- `run_single_dynamic_shared.py`

## Removed Legacy Scripts

The following legacy generators were removed because they were not referenced by any maintained workflow entry point and overlap with the current multi-instance pipeline.

- `generate_phase_sequence_models_stomach.py`
- `generate_stomach_cycle.py`
- `generate_test_gastric_dataset.py`