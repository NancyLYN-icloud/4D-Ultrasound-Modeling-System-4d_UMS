from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DynamicModelConfig, PointCloudPhaseSummary
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


def _load_phase_summaries(pointcloud_root: Path) -> tuple[dict[Path, PointCloudPhaseSummary], dict[Path, float]]:
    summary_path = pointcloud_root / "pointcloud_summary.csv"
    if not summary_path.exists():
        return {}, {}
    lines = summary_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return {}, {}
    header = lines[0].split(",")
    summary_map: dict[Path, PointCloudPhaseSummary] = {}
    confidence_map: dict[Path, float] = {}
    for line in lines[1:]:
        row = dict(zip(header, line.split(",")))
        path = pointcloud_root / row["pointcloud_path"]
        summary = PointCloudPhaseSummary(
            phase_index=int(row["phase_index"]),
            phase_center=float(row["phase_center"]),
            sample_count=int(row["sample_count"]),
            raw_point_count=int(row["raw_point_count"]),
            exported_point_count=int(row["exported_point_count"]),
            mean_confidence=float(row["mean_confidence"]),
            mean_sample_snr=float(row["mean_sample_snr"]),
            extracted_slice_ratio=float(row["extracted_slice_ratio"]),
            pointcloud_path=path,
        )
        summary_map[path] = summary
        confidence_map[path] = float(0.7 * summary.mean_confidence + 0.3 * summary.extracted_slice_ratio)
    return summary_map, confidence_map


def _write_rows(rows: list[dict[str, float | int | str]], out_csv: Path, out_json: Path) -> None:
    if not rows:
        return
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune decoupled shape-motion latent dynamic regularization")
    parser.add_argument("--pointcloud-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=72)
    parser.add_argument("--config-name", type=str, default=None)
    args = parser.parse_args()

    pointclouds = sorted(args.pointcloud_root.glob("*.ply"))
    gt_items = _load_sequence(args.gt_dir)
    phase_summaries, phase_confidences = _load_phase_summaries(args.pointcloud_root)
    out_csv = args.pointcloud_root / "shape_motion_tuning_summary.csv"
    out_json = args.pointcloud_root / "shape_motion_tuning_summary.json"

    configs = [
        {
            "name": "shape_motion_balanced",
            "tw": 0.015,
            "ta": 0.0075,
            "pw": 0.0075,
            "ct": 0.0075,
            "ca": 0.003,
            "cp": 0.003,
            "cs": 0.35,
            "cr": 0.2,
            "shape_offset": 0.01,
            "shape_spatial": 0.03,
            "shape_code": 1e-4,
            "motion_mean": 0.03,
            "motion_lip": 0.02,
            "boot": 0.12,
            "boot_decay": 0.30,
            "spatial": 0.05,
            "deform": 0.003,
            "periodic": 0.05,
        },
        {
            "name": "shape_motion_shape_flexible",
            "tw": 0.015,
            "ta": 0.0075,
            "pw": 0.0075,
            "ct": 0.0075,
            "ca": 0.003,
            "cp": 0.003,
            "cs": 0.35,
            "cr": 0.2,
            "shape_offset": 0.005,
            "shape_spatial": 0.02,
            "shape_code": 5e-5,
            "motion_mean": 0.02,
            "motion_lip": 0.02,
            "boot": 0.08,
            "boot_decay": 0.25,
            "spatial": 0.05,
            "deform": 0.003,
            "periodic": 0.05,
        },
        {
            "name": "shape_motion_motion_strong",
            "tw": 0.02,
            "ta": 0.01,
            "pw": 0.01,
            "ct": 0.01,
            "ca": 0.005,
            "cp": 0.005,
            "cs": 0.4,
            "cr": 0.2,
            "shape_offset": 0.01,
            "shape_spatial": 0.03,
            "shape_code": 1e-4,
            "motion_mean": 0.08,
            "motion_lip": 0.04,
            "boot": 0.12,
            "boot_decay": 0.35,
            "spatial": 0.08,
            "deform": 0.002,
            "periodic": 0.08,
        },
        {
            "name": "shape_motion_bootstrap_guarded",
            "tw": 0.015,
            "ta": 0.0075,
            "pw": 0.0075,
            "ct": 0.005,
            "ca": 0.002,
            "cp": 0.002,
            "cs": 0.45,
            "cr": 0.2,
            "shape_offset": 0.015,
            "shape_spatial": 0.04,
            "shape_code": 1e-4,
            "motion_mean": 0.04,
            "motion_lip": 0.02,
            "boot": 0.20,
            "boot_decay": 0.40,
            "spatial": 0.05,
            "deform": 0.002,
            "periodic": 0.05,
        },
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
            method="shared_topology_decoupled_shape_motion_latent",
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
            shape_latent_dim=64,
            shape_offset_reg_weight=float(config["shape_offset"]),
            shape_spatial_weight=float(config["shape_spatial"]),
            shape_code_reg_weight=float(config["shape_code"]),
            motion_latent_dim=64,
            motion_mean_weight=float(config["motion_mean"]),
            motion_lipschitz_weight=float(config["motion_lip"]),
            bootstrap_offset_weight=float(config["boot"]),
            bootstrap_decay_fraction=float(config["boot_decay"]),
            periodicity_weight=float(config["periodic"]),
            deformation_weight=float(config["deform"]),
            confidence_floor=0.2,
            centroid_weight=0.05,
            spatial_smoothness_weight=float(config["spatial"]),
            observation_support_radius_mm=6.0,
            unsupported_anchor_weight=0.05,
            unsupported_laplacian_weight=0.12,
            base_mesh_train_steps=80,
            mesh_resolution=72,
            surface_batch_size=1024,
            eikonal_batch_size=1024,
            temporal_batch_size=1024,
        )
        reconstruct_dynamic_meshes_from_pointclouds(
            pointclouds,
            config=dynamic_config,
            phase_confidences=phase_confidences,
            phase_summaries=phase_summaries,
        )
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
            "shape_offset_reg_weight": float(config["shape_offset"]),
            "shape_spatial_weight": float(config["shape_spatial"]),
            "shape_code_reg_weight": float(config["shape_code"]),
            "motion_mean_weight": float(config["motion_mean"]),
            "motion_lipschitz_weight": float(config["motion_lip"]),
            "bootstrap_offset_weight": float(config["boot"]),
            "bootstrap_decay_fraction": float(config["boot_decay"]),
            "spatial_smoothness_weight": float(config["spatial"]),
            "deformation_weight": float(config["deform"]),
            "periodicity_weight": float(config["periodic"]),
            **metrics,
        }
        rows.append(row)
        _write_rows(rows, out_csv, out_json)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    _write_rows(rows, out_csv, out_json)
    print(f"WROTE {out_csv}")
    print(f"WROTE {out_json}")


if __name__ == "__main__":
    main()