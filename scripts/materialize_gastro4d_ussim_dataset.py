from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path("/home/liuyanan/data/Research_Data/4D-UMS")
DEFAULT_DATASET_NAME = "Gastro4D-USSim"
DEFAULT_GROUPS = ["stomachPCD_dev", "stomachPCD_01", "stomachPCD_02"]


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_optional(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def _split_for_group(group_name: str) -> str:
    lowered = group_name.lower()
    if "dev" in lowered:
        return "dev"
    return "test"


def _collect_group_pointclouds(group_dir: Path) -> list[Path]:
    return sorted(path for path in group_dir.glob("*.ply") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the Gastro4D-USSim dataset root from grouped stomach point-cloud folders.")
    parser.add_argument("--source-data-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=None, help="Target dataset root. Defaults to <source-data-root>/Gastro4D-USSim")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    args = parser.parse_args()

    source_root = args.source_data_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root else (source_root / args.dataset_name)

    stomach_root = dataset_root / "stomach_pcd"
    benchmark_root = dataset_root / "benchmark"
    simulate_root = dataset_root / "simuilate_data"
    manifest_root = benchmark_root / "manifests"

    stomach_root.mkdir(parents=True, exist_ok=True)
    (benchmark_root / "instances").mkdir(parents=True, exist_ok=True)
    (benchmark_root / "instances_before").mkdir(parents=True, exist_ok=True)
    (benchmark_root / "conditions").mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    (simulate_root / "instances").mkdir(parents=True, exist_ok=True)
    (simulate_root / "meshes").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    copied_names: set[str] = set()

    for group_name in args.groups:
        group_dir = source_root / "stomach_pcd" / group_name
        if not group_dir.exists():
            raise FileNotFoundError(f"Grouped point-cloud directory not found: {group_dir}")

        pointclouds = _collect_group_pointclouds(group_dir)
        if not pointclouds:
            raise FileNotFoundError(f"No PLY files found under {group_dir}")

        for pointcloud in pointclouds:
            target_name = pointcloud.name
            if target_name in copied_names:
                raise ValueError(f"Duplicate point-cloud name across groups: {target_name}")
            copied_names.add(target_name)

            target_path = stomach_root / target_name
            _copy_file(pointcloud, target_path)
            rows.append(
                {
                    "instance_name": pointcloud.stem,
                    "source_group": group_name,
                    "split": _split_for_group(group_name),
                    "source_relpath": str(pointcloud.relative_to(source_root)),
                    "target_relpath": str(target_path.relative_to(dataset_root)),
                }
            )

    _copy_optional(source_root / "benchmark" / "monitor_stream.npz", benchmark_root / "monitor_stream.npz")
    _copy_optional(source_root / "benchmark" / "scanner_sequence.npz", benchmark_root / "scanner_sequence.npz")
    _copy_optional(source_root / "benchmark" / "image" / "monitor", benchmark_root / "image" / "monitor")
    _copy_optional(source_root / "benchmark" / "image" / "scanner", benchmark_root / "image" / "scanner")

    manifest_path = manifest_root / "source_pointcloud_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["instance_name", "source_group", "split", "source_relpath", "target_relpath"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset_name": args.dataset_name,
        "dataset_root": str(dataset_root),
        "source_data_root": str(source_root),
        "groups": list(args.groups),
        "instance_count": len(rows),
        "source_manifest": str(manifest_path),
    }
    (manifest_root / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Gastro4D-USSim] Dataset root: {dataset_root}")
    print(f"[Gastro4D-USSim] Copied {len(rows)} point clouds into {stomach_root}")
    print(f"[Gastro4D-USSim] Source manifest: {manifest_path}")


if __name__ == "__main__":
    main()