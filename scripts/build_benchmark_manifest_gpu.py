from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gastro4d_gpu_layout import select_grouped_reference_pointclouds
from src.paths import data_path


DEFAULT_OUTPUT = data_path("benchmark", "manifests", "benchmark_manifest_gpu.csv")


def _latest_phase_model_dir(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = sorted(
        path for path in base_dir.iterdir()
        if path.is_dir() and path.name.startswith("phase_sequence_models_run_")
    )
    if not candidates:
        return None
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the grouped GPU benchmark manifest.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--instances", nargs="*", default=None)
    parser.add_argument("--skip-incomplete", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    records = select_grouped_reference_pointclouds(groups=args.groups, instances=args.instances)
    if not records:
        raise FileNotFoundError("No grouped reference point clouds matched the requested filters")

    for record in records:
        phase_model_dir = _latest_phase_model_dir(record.phase_model_base_dir)
        phase_summary = phase_model_dir / "phase_sequence_summary.csv" if phase_model_dir is not None else None
        complete = (
            record.monitor_stream.exists()
            and record.scanner_sequence.exists()
            and phase_model_dir is not None
            and phase_summary is not None
            and phase_summary.exists()
        )
        if args.skip_incomplete and not complete:
            continue

        rows.append(
            {
                "instance_name": record.instance_name,
                "shape_family": record.source_group,
                "split": record.split,
                "reference_ply": str(record.reference_ply),
                "monitor_stream": str(record.monitor_stream),
                "scanner_sequence": str(record.scanner_sequence),
                "phase_model_dir": str(phase_model_dir) if phase_model_dir is not None else "",
                "phase_summary": str(phase_summary) if phase_summary is not None else "",
            }
        )

    rows.sort(key=lambda item: item["instance_name"])
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "instance_name",
                "shape_family",
                "split",
                "reference_ply",
                "monitor_stream",
                "scanner_sequence",
                "phase_model_dir",
                "phase_summary",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[BenchmarkManifestGPU] Wrote {output}")
    print(f"[BenchmarkManifestGPU] Rows: {len(rows)}")


if __name__ == "__main__":
    main()
