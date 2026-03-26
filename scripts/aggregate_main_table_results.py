from __future__ import annotations

import argparse
import csv
from pathlib import Path


LOWER_IS_BETTER = {
    "平均CD(mm^2)": True,
    "平均HD95(mm)": True,
    "时间平滑度(mm/step)": True,
}

HIGHER_IS_BETTER = {
    "水密比例": True,
}


def _read_single_result(run_dir: Path) -> dict[str, str]:
    csv_path = run_dir / "dynamic_shared_result.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing result file: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
    if row is None:
        raise ValueError(f"Empty result file: {csv_path}")

    row["run_dir"] = str(run_dir)
    row["run_name"] = run_dir.name
    return row


def _rank(rows: list[dict[str, str]], metric: str, descending: bool) -> None:
    ordered = sorted(rows, key=lambda item: float(item[metric]), reverse=descending)
    for index, row in enumerate(ordered, start=1):
        row[f"{metric}_rank"] = index


def _format_float(value: str, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    headers = [
        "方法",
        "平均CD(mm^2)",
        "平均HD95(mm)",
        "时间平滑度(mm/step)",
        "水密比例",
        "run_name",
    ]

    lines = [
        "# Main Table Summary",
        "",
        "| 方法 | CD↓ | HD95↓ | 时间平滑度↓ | 水密比例↑ | run_name |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        lines.append(
            "| {method} | {cd} | {hd95} | {smooth} | {watertight} | {run_name} |".format(
                method=row["方法"],
                cd=_format_float(row["平均CD(mm^2)"], 4),
                hd95=_format_float(row["平均HD95(mm)"], 4),
                smooth=_format_float(row["时间平滑度(mm/step)"], 4),
                watertight=_format_float(row["水密比例"], 4),
                run_name=row["run_name"],
            )
        )

    best_cd = min(rows, key=lambda item: float(item["平均CD(mm^2)"]))
    best_hd95 = min(rows, key=lambda item: float(item["平均HD95(mm)"]))
    best_smooth = min(rows, key=lambda item: float(item["时间平滑度(mm/step)"]))
    best_watertight = max(rows, key=lambda item: float(item["水密比例"]))

    lines.extend(
        [
            "",
            "## Best Metrics",
            "",
            f"- Best CD: {best_cd['方法']} ({_format_float(best_cd['平均CD(mm^2)'], 4)})",
            f"- Best HD95: {best_hd95['方法']} ({_format_float(best_hd95['平均HD95(mm)'], 4)})",
            f"- Best Temporal Smoothness: {best_smooth['方法']} ({_format_float(best_smooth['时间平滑度(mm/step)'], 4)})",
            f"- Best Watertight Ratio: {best_watertight['方法']} ({_format_float(best_watertight['水密比例'], 4)})",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate dynamic main-table run results into CSV and Markdown")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run directories that contain dynamic_shared_result.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for aggregated outputs")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="main_table_dynamic_methods_10000steps",
        help="Prefix for output CSV and Markdown files",
    )
    args = parser.parse_args()

    rows = [_read_single_result(run_dir.resolve()) for run_dir in args.run_dirs]

    for metric in LOWER_IS_BETTER:
        _rank(rows, metric, descending=False)
    for metric in HIGHER_IS_BETTER:
        _rank(rows, metric, descending=True)

    rows.sort(key=lambda item: (float(item["平均CD(mm^2)"]), float(item["平均HD95(mm)"])))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.output_prefix}.csv"
    md_path = args.output_dir / f"{args.output_prefix}.md"

    fieldnames = list(rows[0].keys())
    _write_csv(csv_path, rows, fieldnames)
    _write_markdown(md_path, rows)

    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()