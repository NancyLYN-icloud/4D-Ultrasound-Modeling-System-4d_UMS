# Gastro4D-USSim Academic Experiment Framework

This document defines a cleaner experiment structure for journal-oriented benchmarking, especially when the paper narrative distinguishes:

1. the proposed method,
2. multiple external baseline paradigms, and
3. benchmark-scale batch execution on Gastro4D-USSim.

## Goals

The framework is designed to support three recurring needs in the current TMI-oriented workflow:

1. keep paper-facing method taxonomy consistent with code-facing method implementations,
2. make it cheap to add new external baselines without editing multiple shell scripts, and
3. keep outputs organized so main-table, robustness, and ablation experiments remain traceable.

## Core Design

### 1. Unified method registry

The canonical method definitions now live in [scripts/experiment_method_registry.py](/home/tianjun0/liuyanan/program/CV/4D-Ultrasound-Modeling-System-4d_UMS/scripts/experiment_method_registry.py).

Each method entry stores:

1. display name used by the existing reconstruction pipeline,
2. stable slug used in folder names and aggregation tables,
3. academic method name for paper tables,
4. method paradigm for journal writing,
5. paper role such as `ours` or `external-baseline`,
6. implementation identifier, and
7. optional profile-specific CLI overrides.

This eliminates the previous duplication where method names and tuned arguments were scattered across shell scripts and Python entrypoints.

### 2. Manifest-driven batch execution

The preferred batch entrypoint is now [scripts/run_benchmark_suite.py](/home/tianjun0/liuyanan/program/CV/4D-Ultrasound-Modeling-System-4d_UMS/scripts/run_benchmark_suite.py).

It consumes a benchmark manifest or condition manifest and runs a selected method set over:

1. chosen split,
2. chosen conditions, and
3. chosen method group or explicit method list.

The runner organizes outputs as:

```text
suite_<timestamp>_<label>/
  suite_metadata.json
  methods/
    refcorr/
      method_spec.json
      runs/
    continuous/
      method_spec.json
      runs/
    decoupled_motion/
      method_spec.json
      runs/
    global_basis_residual/
      method_spec.json
      runs/
  aggregated/
    results_flat.csv
    results_by_method.csv
```

This structure is intentionally journal-friendly: every method has a clean namespace, and final aggregated tables are separated from raw run folders.

### 3. Single-run metadata made paper-aware

[scripts/run_single_dynamic_shared.py](/home/tianjun0/liuyanan/program/CV/4D-Ultrasound-Modeling-System-4d_UMS/scripts/run_single_dynamic_shared.py) now writes method metadata into every run directory and appends academic fields into `dynamic_shared_result.csv`.

That means every run now carries:

1. `method_slug`,
2. `method_academic_name`,
3. `method_paradigm`,
4. `method_paper_role`, and
5. `method_implementation`.

This makes later aggregation and manuscript table building much cleaner.

## Recommended Usage

### Main table

Run the main journal comparison over a manifest:

```bash
python scripts/run_benchmark_suite.py \
  --manifest experiments/benchmark_condition_manifest.csv \
  --out-root experiments/batches \
  --suite-name gastro4d_ussim_main_table \
  --split test \
  --conditions Clean \
  --methods main-table \
  --mode full-paper \
  --dynamic-train-steps 10000 \
  --dynamic-mesh-resolution 72 \
  --max-points-per-phase 5000
```

### Robustness benchmark

```bash
python scripts/run_benchmark_suite.py \
  --manifest experiments/benchmark_condition_manifest.csv \
  --out-root experiments/batches \
  --suite-name gastro4d_ussim_robustness \
  --split test \
  --conditions Clean Sparse PoseNoise ImageNoise \
  --methods main-table \
  --mode full-paper
```

### Add a new external baseline

When a new baseline is added, the preferred workflow is:

1. add one `MethodSpec` entry in the registry,
2. give it a stable slug and academic name,
3. attach a default profile if tuned overrides are needed,
4. ensure the underlying implementation string is supported by `run_experiments.py`,
5. rerun `run_benchmark_suite.py` with the new method slug or with a method preset.

This prevents further shell-script sprawl.

## Why this is better for a TMI submission

Compared with the old structure, this framework improves three things directly relevant to a journal paper:

1. clearer separation between method taxonomy and implementation details,
2. cleaner batch output hierarchy for reproducibility and supplementary material, and
3. lower friction when expanding the benchmark with additional external methods.

That is important because the paper argument is not only about one better model. It is also about presenting a credible, reproducible benchmark comparison over multiple dynamic-method paradigms on Gastro4D-USSim.