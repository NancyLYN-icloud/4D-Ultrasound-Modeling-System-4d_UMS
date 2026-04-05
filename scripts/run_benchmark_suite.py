from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_method_registry import (
    MethodSpec,
    get_default_main_table_methods,
    get_external_baseline_methods,
    get_method_spec,
    get_supplementary_baseline_methods,
    profile_cli_args,
)


def _sanitize_token(value: str) -> str:
    lowered = value.strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in lowered).strip("_")


def _resolve_methods(method_tokens: list[str]) -> list[MethodSpec]:
    if not method_tokens or method_tokens == ["main-table"]:
        return get_default_main_table_methods()
    if method_tokens == ["external-baselines"]:
        return get_external_baseline_methods()
    if method_tokens == ["supplementary-baselines"]:
        return get_supplementary_baseline_methods()
    return [get_method_spec(token) for token in method_tokens]


def _load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _infer_data_root(manifest_path: Path) -> Path:
    if manifest_path.parent.name == "manifests" and manifest_path.parent.parent.name == "benchmark":
        return manifest_path.parent.parent.parent
    return manifest_path.parent


def _resolve_manifest_path(data_root: Path, value: str | None) -> Path | None:
    raw = (value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return data_root / path


def _resolve_condition(row: dict[str, str]) -> str:
    condition = (row.get("condition") or "Clean").strip()
    return condition or "Clean"


def _resolve_gt_mesh_path(row: dict[str, str], data_root: Path) -> Path:
    direct = _resolve_manifest_path(data_root, row.get("gt_mesh_path"))
    if direct is not None:
        return direct

    phase_model_dir = _resolve_manifest_path(
        data_root,
        row.get("phase_model_dir") or row.get("phase_root_relpath"),
    )
    if phase_model_dir is None:
        return Path()
    preferred = phase_model_dir / "pointclouds" / "meshes"
    if preferred.exists():
        return preferred
    fallback = phase_model_dir / "meshes"
    if fallback.exists():
        return fallback
    return preferred


def _resolve_suite_dir(out_root: Path, suite_name: str, run_label: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _sanitize_token(run_label or suite_name)
    suite_dir = out_root / f"suite_{timestamp}_{suffix}"
    index = 1
    while suite_dir.exists():
        suite_dir = out_root / f"suite_{timestamp}_{suffix}_{index:02d}"
        index += 1
    suite_dir.mkdir(parents=True, exist_ok=False)
    return suite_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _filter_cases(
    rows: list[dict[str, str]],
    data_root: Path,
    selected_split: str | None,
    selected_conditions: set[str],
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for row in rows:
        instance_name = (row.get("instance_name") or "").strip()
        split = (row.get("split") or "").strip()
        condition = _resolve_condition(row)
        if selected_split and split != selected_split:
            continue
        if selected_conditions and condition not in selected_conditions:
            continue

        clean_root = _resolve_manifest_path(data_root, row.get("clean_root_relpath"))
        monitor_path = _resolve_manifest_path(
            data_root,
            row.get("monitor_stream") or row.get("monitor_path") or row.get("monitor_relpath"),
        )
        scanner_path = _resolve_manifest_path(
            data_root,
            row.get("scanner_sequence") or row.get("scanner_path"),
        )
        if scanner_path is None and clean_root is not None:
            scanner_path = clean_root / "scanner_sequence.npz"
        if monitor_path is None or scanner_path is None:
            print(f"[BenchmarkSuite] skip instance={instance_name} condition={condition}: unresolved monitor/scanner paths", file=sys.stderr)
            continue
        gt_mesh_path = _resolve_gt_mesh_path(row, data_root)
        if not (monitor_path.exists() and scanner_path.exists() and gt_mesh_path.exists()):
            print(f"[BenchmarkSuite] skip instance={instance_name} condition={condition}: missing inputs", file=sys.stderr)
            continue

        cases.append(
            {
                "row": row,
                "instance_name": instance_name,
                "split": split,
                "condition": condition,
                "monitor_path": monitor_path,
                "scanner_path": scanner_path,
                "gt_mesh_path": gt_mesh_path,
            }
        )
    return cases


def _build_run_name(instance_name: str, condition: str, spec: MethodSpec, profile_name: str | None) -> str:
    profile_suffix = _sanitize_token(profile_name) if profile_name else "custom"
    return f"{instance_name}_{_sanitize_token(condition)}_{spec.slug}_{profile_suffix}"


def _read_result_csv(run_dir: Path) -> dict[str, str] | None:
    csv_path = run_dir / "dynamic_shared_result.csv"
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return dict(row)
    return None


def _aggregate_results(records: list[dict[str, object]], aggregated_dir: Path) -> None:
    aggregated_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        return

    flat_fields = sorted({key for record in records for key in record.keys()})
    flat_csv = aggregated_dir / "results_flat.csv"
    with flat_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    metrics = ["平均CD(mm^2)", "平均HD95(mm)", "平均表面MAE(mm)", "平均EMD(mm)", "平均Dice"]
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        method_slug = str(record.get("method_slug", "unknown"))
        for metric in metrics:
            value = record.get(metric)
            if value in {None, "", "nan"}:
                continue
            try:
                grouped[method_slug][metric].append(float(value))
            except (TypeError, ValueError):
                continue

    summary_csv = aggregated_dir / "results_by_method.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["method_slug", "run_count", *[f"{metric}_mean" for metric in metrics], *[f"{metric}_std" for metric in metrics]]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method_slug in sorted(grouped):
            row: dict[str, object] = {"method_slug": method_slug, "run_count": len(grouped[method_slug].get(metrics[0], []))}
            for metric in metrics:
                values = grouped[method_slug].get(metric, [])
                if not values:
                    row[f"{metric}_mean"] = ""
                    row[f"{metric}_std"] = ""
                    continue
                mean_value = sum(values) / len(values)
                variance = sum((value - mean_value) ** 2 for value in values) / len(values)
                row[f"{metric}_mean"] = mean_value
                row[f"{metric}_std"] = variance ** 0.5
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark experiments from a manifest using the academic method registry.")
    parser.add_argument("--manifest", type=Path, required=True, help="Benchmark manifest or condition manifest CSV.")
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root used to resolve relative manifest paths. Defaults to the inferred root above benchmark/manifests.")
    parser.add_argument("--out-root", type=Path, required=True, help="Root directory for suite outputs.")
    parser.add_argument("--suite-name", type=str, default="gastro4d_ussim_tmi_suite")
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=["main-table"], help="Method slugs, names, or presets: main-table, external-baselines, supplementary-baselines.")
    parser.add_argument("--method-profile", type=str, default="historical_best_eqbudget")
    parser.add_argument("--mode", choices=["fast-dev", "dynamic-detail", "full-paper"], default="full-paper")
    parser.add_argument("--dynamic-train-steps", type=int, default=10000)
    parser.add_argument("--dynamic-mesh-resolution", type=int, default=72)
    parser.add_argument("--max-points-per-phase", type=int, default=5000)
    parser.add_argument("--quick-profile", choices=["none", "screen", "trend"], default="none")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    methods = _resolve_methods(list(args.methods))
    manifest_path = args.manifest.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root is not None else _infer_data_root(manifest_path)
    rows = _load_manifest_rows(manifest_path)
    suite_dir = _resolve_suite_dir(args.out_root, args.suite_name, args.run_label)
    methods_dir = suite_dir / "methods"
    aggregated_dir = suite_dir / "aggregated"

    selected_conditions = {condition.strip() for condition in (args.conditions or []) if condition.strip()}
    selected_split = args.split.strip() if args.split else None
    cases = _filter_cases(rows, data_root=data_root, selected_split=selected_split, selected_conditions=selected_conditions)

    _write_json(
        suite_dir / "suite_metadata.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "suite_name": args.suite_name,
            "run_label": args.run_label,
            "manifest": str(manifest_path),
            "data_root": str(data_root),
            "split": selected_split,
            "conditions": sorted(selected_conditions),
            "method_profile": args.method_profile,
            "mode": args.mode,
            "dynamic_train_steps": args.dynamic_train_steps,
            "dynamic_mesh_resolution": args.dynamic_mesh_resolution,
            "max_points_per_phase": args.max_points_per_phase,
            "methods": [spec.to_metadata() for spec in methods],
            "python_executable": sys.executable,
        },
    )

    records: list[dict[str, object]] = []
    for spec in methods:
        method_root = methods_dir / spec.slug
        method_runs_dir = method_root / "runs"
        method_runs_dir.mkdir(parents=True, exist_ok=True)
        _write_json(method_root / "method_spec.json", spec.to_metadata())

        for case in cases:
            row = case["row"]
            instance_name = str(case["instance_name"])
            split = str(case["split"])
            condition = str(case["condition"])
            monitor_path = Path(case["monitor_path"])
            scanner_path = Path(case["scanner_path"])
            gt_mesh_path = Path(case["gt_mesh_path"])

            run_name = _build_run_name(instance_name, condition, spec, args.method_profile)
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_single_dynamic_shared.py"),
                "--mode",
                args.mode,
                "--method",
                spec.display_name,
                "--out-dir",
                str(method_runs_dir),
                "--instance-name",
                instance_name,
                "--monitor-path",
                str(monitor_path),
                "--scanner-path",
                str(scanner_path),
                "--gt-mesh-path",
                str(gt_mesh_path),
                "--run-name",
                run_name,
                "--quick-profile",
                args.quick_profile,
                "--dynamic-train-steps",
                str(args.dynamic_train_steps),
                "--dynamic-mesh-resolution",
                str(args.dynamic_mesh_resolution),
                "--max-points-per-phase",
                str(args.max_points_per_phase),
                *profile_cli_args(spec, args.method_profile),
            ]

            print(f"[BenchmarkSuite] method={spec.display_name} instance={instance_name} split={split} condition={condition}")
            if args.dry_run:
                print(" ".join(cmd))
                continue

            subprocess.run(cmd, check=True)
            run_dirs = sorted(method_runs_dir.glob("exp_*"), key=lambda path: path.stat().st_mtime)
            if not run_dirs:
                continue
            latest_run = run_dirs[-1]
            result_row = _read_result_csv(latest_run)
            if result_row is None:
                continue
            result_row.update(
                {
                    "suite_name": args.suite_name,
                    "instance_name": instance_name,
                    "split": split,
                    "condition": condition,
                    "method_slug": spec.slug,
                    "method_academic_name": spec.academic_name,
                    "method_paradigm": spec.paradigm,
                    "method_paper_role": spec.paper_role,
                    "method_execution_family": spec.execution_family,
                    "run_dir": str(latest_run),
                }
            )
            records.append(result_row)

    _aggregate_results(records, aggregated_dir)
    print(f"[BenchmarkSuite] suite_dir={suite_dir}")


if __name__ == "__main__":
    main()