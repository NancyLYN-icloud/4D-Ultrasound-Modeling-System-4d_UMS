from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path("/home/liuyanan/data/Research_Data/4D-UMS")
DEFAULT_DATASET_NAME = "Gastro4D-USSim"
DEFAULT_GROUPS = ["stomachPCD_dev", "stomachPCD_01", "stomachPCD_02"]


def _copy_or_symlink(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if link_mode == "symlink":
        dst.symlink_to(src, target_is_directory=src.is_dir())
        return
    shutil.copy2(src, dst)


def _split_for_group(group_name: str) -> str:
    return "dev" if "dev" in group_name.lower() else "test"


def _collect_group_pointclouds(group_dir: Path) -> list[Path]:
    return sorted(path for path in group_dir.glob("*.ply") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the grouped Gastro4D-USSim dataset root for the GPU pipeline.")
    parser.add_argument("--source-data-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=None, help="Defaults to <source-data-root>/Gastro4D-USSim")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--link-mode", choices=["copy", "symlink"], default="copy")
    args = parser.parse_args()

    source_root = args.source_data_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root else (source_root / args.dataset_name)
    stomach_root = dataset_root / "stomach_pcd"
    benchmark_root = dataset_root / "benchmark"
    simulate_root = dataset_root / "simuilate_data"
    manifest_root = benchmark_root / "manifests"

    stomach_root.mkdir(parents=True, exist_ok=True)
    (benchmark_root / "instances").mkdir(parents=True, exist_ok=True)
    (benchmark_root / "conditions").mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    (simulate_root / "instances").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    copied_names: set[str] = set()
    for group_name in args.groups:
        group_dir = source_root / "stomach_pcd" / group_name
        if not group_dir.exists():
            raise FileNotFoundError(f"Grouped point-cloud directory not found: {group_dir}")
        target_group_dir = stomach_root / group_name
        target_group_dir.mkdir(parents=True, exist_ok=True)

        pointclouds = _collect_group_pointclouds(group_dir)
        if not pointclouds:
            raise FileNotFoundError(f"No PLY files found under {group_dir}")

        for pointcloud in pointclouds:
            if pointcloud.name in copied_names:
                raise ValueError(f"Duplicate point-cloud name across groups: {pointcloud.name}")
            copied_names.add(pointcloud.name)

            target_path = target_group_dir / pointcloud.name
            _copy_or_symlink(pointcloud, target_path, link_mode=args.link_mode)
            split = _split_for_group(group_name)
            rows.append(
                {
                    "instance_name": pointcloud.stem,
                    "source_group": group_name,
                    "split": split,
                    "source_relpath": str(pointcloud.relative_to(source_root)),
                    "target_relpath": str(target_path.relative_to(dataset_root)),
                    "clean_root_relpath": f"benchmark/instances/{split}/{group_name}/{pointcloud.stem}",
                    "phase_root_relpath": f"simuilate_data/instances/{split}/{group_name}/{pointcloud.stem}",
                }
            )

    manifest_path = manifest_root / "source_pointcloud_manifest_gpu.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "instance_name",
                "source_group",
                "split",
                "source_relpath",
                "target_relpath",
                "clean_root_relpath",
                "phase_root_relpath",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset_name": args.dataset_name,
        "dataset_root": str(dataset_root),
        "source_data_root": str(source_root),
        "groups": list(args.groups),
        "instance_count": len(rows),
        "link_mode": args.link_mode,
        "source_manifest": str(manifest_path),
    }
    (manifest_root / "dataset_summary_gpu.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Gastro4D GPU] Dataset root: {dataset_root}")
    print(f"[Gastro4D GPU] Preserved grouped point clouds under {stomach_root}")
    print(f"[Gastro4D GPU] Source manifest: {manifest_path}")


if __name__ == "__main__":
    main()
