from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from src.config import DynamicModelConfig
from src.modeling.dynamic_surface_reconstruction import reconstruct_dynamic_meshes_from_pointclouds
from src.modeling.metrics import compute_chamfer_distance, compute_hausdorff_distance, compute_temporal_smoothness


PHASE_PATTERNS = [
    re.compile(r"run_\d+_phase_(\d+)_([0-9]+\.[0-9]+)_t"),
    re.compile(r"dynamic_phase_([0-9]+\.[0-9]+)\.ply$"),
]


def _parse_phase(path: Path) -> tuple[int | None, float | None]:
    name = path.name
    match = PHASE_PATTERNS[0].search(name)
    if match:
        return int(match.group(1)), float(match.group(2))
    match = PHASE_PATTERNS[1].search(name)
    if match:
        return None, float(match.group(1))
    return None, None


def _load_sequence(folder: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for path in sorted(folder.glob("*.ply")):
        if path.name == "dynamic_base_mesh.ply":
            continue
        mesh = trimesh.load(path, force="mesh")
        phase_index, phase = _parse_phase(path)
        items.append({"path": path, "mesh": mesh, "phase_index": phase_index, "phase": phase})
    return items


def _sample_points(mesh: trimesh.Trimesh, n: int = 8000) -> np.ndarray:
    if len(mesh.faces) > 0:
        points, _ = trimesh.sample.sample_surface(mesh, n)
        return np.asarray(points, dtype=np.float64)
    return np.asarray(mesh.vertices, dtype=np.float64)


def _match_gt(gt_items: list[dict[str, object]], pred_item: dict[str, object]) -> dict[str, object]:
    phase_index = pred_item["phase_index"]
    if phase_index is not None:
        for gt in gt_items:
            if gt["phase_index"] == phase_index:
                return gt
    pred_phase = float(pred_item["phase"])
    candidates = [gt for gt in gt_items if gt["phase"] is not None]
    return min(candidates, key=lambda gt: abs(float(gt["phase"]) - pred_phase))


def _centroid_stats(meshes: list[trimesh.Trimesh]) -> tuple[float, float]:
    centroids = np.stack([mesh.centroid for mesh in meshes], axis=0)
    radial = np.linalg.norm(centroids - centroids.mean(axis=0, keepdims=True), axis=1)
    return float(radial.mean()), float(radial.max())


def _motion_stats(meshes: list[trimesh.Trimesh]) -> tuple[float, float]:
    base_points = _sample_points(meshes[0])
    base_tree = cKDTree(base_points)
    adjacent: list[float] = []
    from_base: list[float] = []
    for index, mesh in enumerate(meshes):
        points = _sample_points(mesh)
        if index > 0:
            prev_points = _sample_points(meshes[index - 1])
            prev_tree = cKDTree(prev_points)
            dist, _ = prev_tree.query(points, k=1)
            adjacent.append(float(dist.mean()))
        dist_base, _ = base_tree.query(points, k=1)
        from_base.append(float(dist_base.mean()))
    return float(np.mean(adjacent)), float(np.mean(from_base))


def _evaluate(folder: Path, gt_items: list[dict[str, object]]) -> dict[str, float | int]:
    sequence = _load_sequence(folder)
    meshes = [item["mesh"] for item in sequence]
    chamfers: list[float] = []
    hausdorffs: list[float] = []
    for item in sequence:
        gt = _match_gt(gt_items, item)
        chamfers.append(compute_chamfer_distance(item["mesh"], gt["mesh"]))
        hausdorffs.append(compute_hausdorff_distance(item["mesh"], gt["mesh"]))
    centroid_mean, centroid_max = _centroid_stats(meshes)
    adjacent_mean, from_base_mean = _motion_stats(meshes)
    return {
        "mesh_count": len(meshes),
        "mean_cd": float(np.mean(chamfers)),
        "mean_hd95": float(np.mean(hausdorffs)),
        "temporal_smoothness": float(compute_temporal_smoothness(meshes)),
        "centroid_mean": centroid_mean,
        "centroid_max": centroid_max,
        "adjacent_mean": adjacent_mean,
        "from_base_mean": from_base_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune shared-topology reference-correspondence dynamic regularization")
    parser.add_argument("--pointcloud-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--config-name", type=str, default=None)
    args = parser.parse_args()

    pointclouds = sorted(args.pointcloud_root.glob("*.ply"))
    gt_items = _load_sequence(args.gt_dir)
    out_csv = args.pointcloud_root / "refcorr_tuning_summary.csv"
    out_json = args.pointcloud_root / "refcorr_tuning_summary.json"

    configs = [
        {"name": "refcorr_warm_start", "ct": 0.01, "ca": 0.005, "cp": 0.005, "tw": 0.02, "ta": 0.01, "pw": 0.01, "cs": 0.0, "cr": 0.0},
        {"name": "refcorr_light", "ct": 0.0075, "ca": 0.003, "cp": 0.003, "tw": 0.02, "ta": 0.01, "pw": 0.01, "cs": 0.0, "cr": 0.0},
        {"name": "refcorr_very_light", "ct": 0.005, "ca": 0.002, "cp": 0.002, "tw": 0.02, "ta": 0.01, "pw": 0.01, "cs": 0.0, "cr": 0.0},
        {"name": "refcorr_temporal_shift", "ct": 0.01, "ca": 0.003, "cp": 0.003, "tw": 0.015, "ta": 0.0075, "pw": 0.0075, "cs": 0.0, "cr": 0.0},
        {"name": "refcorr_very_light_delayed", "ct": 0.005, "ca": 0.002, "cp": 0.002, "tw": 0.02, "ta": 0.01, "pw": 0.01, "cs": 0.35, "cr": 0.25},
        {"name": "refcorr_warm_start_delayed", "ct": 0.01, "ca": 0.005, "cp": 0.005, "tw": 0.02, "ta": 0.01, "pw": 0.01, "cs": 0.4, "cr": 0.2},
    ]
    if args.config_name is not None:
        configs = [config for config in configs if config["name"] == args.config_name]
        if not configs:
            raise ValueError(f"Unknown config name: {args.config_name}")

    rows: list[dict[str, float | int | str]] = []
    for config in configs:
        out_subdir = f"dynamic_meshes_{config['name']}"
        print(f"=== RUN {config['name']} ===")
        dynamic_config = DynamicModelConfig(
            enabled=True,
            method="shared_topology_vertex_field_reference_correspondence",
            out_subdir=out_subdir,
            max_points_per_phase=3000,
            canonical_hidden_dim=256,
            canonical_hidden_layers=4,
            deformation_hidden_dim=256,
            deformation_hidden_layers=4,
            train_steps=args.train_steps,
            learning_rate=0.001,
            normal_weight=0.15,
            temporal_weight=float(config["tw"]),
            temporal_acceleration_weight=float(config["ta"]),
            phase_consistency_weight=float(config["pw"]),
            correspondence_temporal_weight=float(config["ct"]),
            correspondence_acceleration_weight=float(config["ca"]),
            correspondence_phase_consistency_weight=float(config["cp"]),
            correspondence_start_fraction=float(config["cs"]),
            correspondence_ramp_fraction=float(config["cr"]),
            periodicity_weight=0.02,
            deformation_weight=0.002,
            confidence_floor=1.0,
            centroid_weight=0.05,
            spatial_smoothness_weight=0.05,
            base_mesh_train_steps=80,
            mesh_resolution=72,
            surface_batch_size=1024,
            eikonal_batch_size=1024,
            temporal_batch_size=1024,
        )
        reconstruct_dynamic_meshes_from_pointclouds(pointclouds, config=dynamic_config)
        metrics = _evaluate(args.pointcloud_root / out_subdir, gt_items)
        row = {
            "name": config["name"],
            "temporal_weight": float(config["tw"]),
            "temporal_acceleration_weight": float(config["ta"]),
            "phase_consistency_weight": float(config["pw"]),
            "corr_temporal_weight": float(config["ct"]),
            "corr_acceleration_weight": float(config["ca"]),
            "corr_phase_consistency_weight": float(config["cp"]),
            "corr_start_fraction": float(config["cs"]),
            "corr_ramp_fraction": float(config["cr"]),
            **metrics,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {out_csv}")
    print(f"WROTE {out_json}")


if __name__ == "__main__":
    main()