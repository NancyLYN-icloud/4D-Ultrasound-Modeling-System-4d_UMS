from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_experiments import _extract_phase_info, _load_gt_mesh, _match_gt_geometry, _safe_mean
from src.modeling.metrics import compute_chamfer_distance, compute_hausdorff_distance
from src.stomach_instance_paths import resolve_gt_mesh_input_path, resolve_instance_paths


def _update_run(run_dir: Path, gt_mesh: object) -> tuple[int, int, float, float] | None:
    phase_root_candidates = sorted((run_dir / "artifacts" / "phase_pointclouds_run_001_single").glob("phase_pointclouds_run_001_*"))
    if not phase_root_candidates:
        return None

    mesh_dir = phase_root_candidates[-1] / "dynamic_meshes"
    mesh_paths = sorted(mesh_dir.glob("dynamic_phase_*.ply"))
    if not mesh_paths:
        return None

    chamfers: list[float] = []
    hausdorffs: list[float] = []
    for mesh_path in mesh_paths:
        mesh = trimesh.load(mesh_path, force="mesh")
        _, phase = _extract_phase_info(mesh_path)
        matched_gt = _match_gt_geometry(gt_mesh, mesh_path, phase)
        if matched_gt is None:
            continue
        chamfers.append(compute_chamfer_distance(mesh, matched_gt))
        hausdorffs.append(compute_hausdorff_distance(mesh, matched_gt))

    mean_cd = _safe_mean(chamfers)
    mean_hd95 = _safe_mean(hausdorffs)

    result_path = run_dir / "dynamic_shared_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["平均CD(mm^2)"] = mean_cd
    result["平均HD95(mm)"] = mean_hd95
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = run_dir / "dynamic_shared_result.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    return len(mesh_paths), len(chamfers), mean_cd, mean_hd95


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill CD/HD95 for completed dynamic shared runs using stored dynamic meshes.")
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--instance-name", type=str, required=True)
    args = parser.parse_args()

    instance_paths = resolve_instance_paths(args.instance_name)
    gt_mesh_path = resolve_gt_mesh_input_path(instance_paths)
    gt_mesh = _load_gt_mesh(gt_mesh_path)
    if gt_mesh is None:
        raise FileNotFoundError(f"GT mesh unavailable: {gt_mesh_path}")

    print(f"[BackfillMetrics] gt_mesh_path={gt_mesh_path}")
    runs_root = args.validation_root / "runs"
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        updated = _update_run(run_dir, gt_mesh)
        if updated is None:
            print(f"[BackfillMetrics] skip {run_dir.name}: no reusable dynamic meshes")
            continue
        mesh_count, matched_count, mean_cd, mean_hd95 = updated
        print(
            f"[BackfillMetrics] {run_dir.name}: meshes={mesh_count} matched={matched_count} "
            f"mean_cd={mean_cd:.6f} mean_hd95={mean_hd95:.6f}"
        )


if __name__ == "__main__":
    main()