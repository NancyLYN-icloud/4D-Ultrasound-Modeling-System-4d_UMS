from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import data_path
from src.stomach_instance_paths import list_reference_pointclouds, resolve_instance_paths


DEFAULT_OUTPUT = data_path("benchmark", "manifests", "benchmark_manifest.csv")
DEFAULT_SOURCE_MANIFEST = data_path("benchmark", "manifests", "source_pointcloud_manifest.csv")


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


def _load_source_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            row["instance_name"].strip(): {
                "shape_family": row["source_group"].strip(),
                "split": row["split"].strip(),
            }
            for row in reader
        }


def _infer_shape_family(instance_name: str) -> str:
    if "-" in instance_name:
        return instance_name.rsplit("-", 1)[0]
    return instance_name


def _infer_split(instance_name: str) -> str:
    if "dev" in instance_name.lower():
        return "dev"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean benchmark manifest for the active UMS_DATA_ROOT dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--skip-incomplete", action="store_true", help="Skip instances missing monitor/scanner/phase-model assets")
    args = parser.parse_args()

    source_meta = _load_source_metadata(args.source_manifest.expanduser().resolve())
    rows: list[dict[str, str]] = []

    for reference_ply in list_reference_pointclouds():
        instance_name = reference_ply.stem
        paths = resolve_instance_paths(instance_name=instance_name, reference_ply=reference_ply)
        phase_model_dir = _latest_phase_model_dir(paths.phase_model_base_dir)
        phase_summary = phase_model_dir / "phase_sequence_summary.csv" if phase_model_dir is not None else None

        complete = (
            paths.monitor_stream.exists()
            and paths.scanner_sequence.exists()
            and phase_model_dir is not None
            and phase_summary is not None
            and phase_summary.exists()
        )
        if args.skip_incomplete and not complete:
            continue

        meta = source_meta.get(instance_name, {})
        rows.append(
            {
                "instance_name": instance_name,
                "shape_family": meta.get("shape_family", _infer_shape_family(instance_name)),
                "split": meta.get("split", _infer_split(instance_name)),
                "reference_ply": str(reference_ply),
                "monitor_stream": str(paths.monitor_stream),
                "scanner_sequence": str(paths.scanner_sequence),
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

    print(f"[BenchmarkManifest] Wrote {output}")
    print(f"[BenchmarkManifest] Rows: {len(rows)}")


if __name__ == "__main__":
    main()