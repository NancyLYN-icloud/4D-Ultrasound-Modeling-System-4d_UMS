from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd


KEY_METRICS = [
    "平均CD(mm^2)",
    "平均HD95(mm)",
    "平均表面MAE(mm)",
    "平均EMD(mm)",
    "平均Dice",
    "平均点云置信度",
    "平均样本SNR",
    "平均切片提取率",
]


def _infer_condition(run_name: str) -> str:
    lowered = run_name.lower()
    if "pose" in lowered and "noise" in lowered:
        return "PoseNoise"
    if "noisy" in lowered and "pose" in lowered:
        return "PoseNoise"
    if "noisy" in lowered:
        return "Noisy"
    if "pose" in lowered:
        return "Pose"
    if "sparse" in lowered:
        return "Sparse"
    if "clean" in lowered:
        return "Clean"
    return "Unknown"


def _normalize_method_slug(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", value.strip())
    return cleaned.strip("_") or "method"


def _load_manifest(manifest_path: Path | None) -> pd.DataFrame:
    if manifest_path is None or not manifest_path.exists():
        return pd.DataFrame()
    return pd.read_csv(manifest_path)


def _extract_instance_name(metadata: dict[str, object], result_path: Path) -> str:
    instance_name = metadata.get("instance_name")
    if isinstance(instance_name, str) and instance_name.strip():
        return instance_name.strip()
    cli_args = metadata.get("cli_args")
    if isinstance(cli_args, dict):
        cli_instance = cli_args.get("instance_name")
        if isinstance(cli_instance, str) and cli_instance.strip():
            return cli_instance.strip()
    return result_path.parent.name


def _load_result_row(result_path: Path) -> dict[str, object]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    result = payload.get("result", payload)
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected result payload in {result_path}")

    run_dir = Path(payload.get("run_dir", result_path.parent))
    metadata_path = run_dir / "run_metadata.json"
    metadata: dict[str, object] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    instance_name = _extract_instance_name(metadata, result_path)
    run_name = str(metadata.get("run_name") or run_dir.name)
    method = str(result.get("方法", "unknown"))
    row: dict[str, object] = {
        "instance_name": instance_name,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "method": method,
        "method_slug": _normalize_method_slug(method),
        "condition": _infer_condition(run_name),
        "mode": metadata.get("mode", "unknown"),
    }
    for key, value in result.items():
        row[key] = value
    return row


def _format_mean_std(mean_value: float, std_value: float) -> str:
    if pd.isna(mean_value):
        return "nan"
    if pd.isna(std_value):
        return f"{mean_value:.4f}"
    return f"{mean_value:.4f} +- {std_value:.4f}"


def aggregate_results(runs_root: Path, manifest_path: Path | None, output_dir: Path) -> tuple[Path, Path, Path]:
    result_paths = sorted(runs_root.glob("**/dynamic_shared_result.json"))
    if not result_paths:
        raise FileNotFoundError(f"No dynamic_shared_result.json found under {runs_root}")

    rows = [_load_result_row(path) for path in result_paths]
    instance_df = pd.DataFrame(rows)

    manifest_df = _load_manifest(manifest_path)
    if not manifest_df.empty:
        merge_columns = [column for column in ["instance_name", "shape_family", "split"] if column in manifest_df.columns]
        instance_df = instance_df.merge(manifest_df[merge_columns], on="instance_name", how="left")

    output_dir.mkdir(parents=True, exist_ok=True)
    instance_path = output_dir / "instance_level_results.csv"
    instance_df.to_csv(instance_path, index=False)

    summary_df = (
        instance_df.groupby(["method", "condition", "split"], dropna=False)[KEY_METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = [
        column if isinstance(column, str) else f"{column[0]}_{column[1]}".strip("_")
        for column in summary_df.columns.to_flat_index()
    ]
    summary_path = output_dir / "summary_results.csv"
    summary_df.to_csv(summary_path, index=False)

    formatted_rows: list[dict[str, object]] = []
    for _, row in summary_df.iterrows():
        formatted: dict[str, object] = {
            "method": row["method"],
            "condition": row["condition"],
            "split": row["split"],
        }
        for metric in KEY_METRICS:
            formatted[metric] = _format_mean_std(row.get(f"{metric}_mean"), row.get(f"{metric}_std"))
        formatted_rows.append(formatted)
    formatted_df = pd.DataFrame(formatted_rows)
    formatted_path = output_dir / "summary_results_formatted.csv"
    formatted_df.to_csv(formatted_path, index=False)

    return instance_path, summary_path, formatted_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate dynamic shared reconstruction runs into instance-level and summary tables")
    parser.add_argument("--runs-root", type=Path, required=True, help="Directory containing experiment run folders")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "benchmark_manifest.csv",
        help="Optional benchmark manifest used to attach split and family labels",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for aggregated CSV outputs")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else args.runs_root / "aggregated"
    instance_path, summary_path, formatted_path = aggregate_results(args.runs_root, args.manifest, output_dir)

    print(f"[AggregateDynamicShared] Instance table: {instance_path}")
    print(f"[AggregateDynamicShared] Summary table: {summary_path}")
    print(f"[AggregateDynamicShared] Formatted summary: {formatted_path}")


if __name__ == "__main__":
    main()