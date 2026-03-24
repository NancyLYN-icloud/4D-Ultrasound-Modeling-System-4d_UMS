from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
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


def _sample_points(mesh: trimesh.Trimesh, n: int = 8000, random_seed: int = 7) -> np.ndarray:
    if len(mesh.faces) > 0:
        random_state = np.random.get_state()
        np.random.seed(int(random_seed))
        try:
            points, _ = trimesh.sample.sample_surface(mesh, n)
        finally:
            np.random.set_state(random_state)
        return np.asarray(points, dtype=np.float64)
    return np.asarray(mesh.vertices, dtype=np.float64)


def _match_gt(gt_items: list[dict[str, object]], pred_item: dict[str, object]) -> dict[str, object]:
    phase_index = pred_item["phase_index"]
    if phase_index is not None:
        for gt in gt_items:
            if gt["phase_index"] == phase_index:
                return gt
    pred_phase = pred_item["phase"]
    if pred_phase is None:
        return gt_items[0]
    candidates = [gt for gt in gt_items if gt["phase"] is not None]
    if not candidates:
        return gt_items[0]
    return min(candidates, key=lambda gt: abs(float(gt["phase"]) - float(pred_phase)))


def _centroid_stats(meshes: list[trimesh.Trimesh]) -> tuple[float, float]:
    centroids = np.stack([mesh.centroid for mesh in meshes], axis=0)
    radial = np.linalg.norm(centroids - centroids.mean(axis=0, keepdims=True), axis=1)
    return float(radial.mean()), float(radial.max())


def _motion_stats(meshes: list[trimesh.Trimesh]) -> tuple[float, float]:
    base_points = _sample_points(meshes[0], random_seed=7)
    base_tree = cKDTree(base_points)
    adjacent: list[float] = []
    from_base: list[float] = []
    for index, mesh in enumerate(meshes):
        points = _sample_points(mesh, random_seed=13 + index)
        if index > 0:
            prev_points = _sample_points(meshes[index - 1], random_seed=101 + index)
            prev_tree = cKDTree(prev_points)
            dist, _ = prev_tree.query(points, k=1)
            adjacent.append(float(dist.mean()))
        dist_base, _ = base_tree.query(points, k=1)
        from_base.append(float(dist_base.mean()))
    return float(np.mean(adjacent)) if adjacent else 0.0, float(np.mean(from_base))


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


def _evaluate_folder(folder: Path, gt_items: list[dict[str, object]]) -> dict[str, float | int]:
    sequence = _load_sequence(folder)
    meshes = [item["mesh"] for item in sequence]
    if not meshes:
        return {
            "mesh_count": 0,
            "mean_cd": float("inf"),
            "mean_hd95": float("inf"),
            "temporal_smoothness": float("inf"),
            "centroid_mean": float("inf"),
            "centroid_max": float("inf"),
            "adjacent_mean": float("inf"),
            "from_base_mean": float("inf"),
            "watertight_ratio": 0.0,
        }

    chamfers: list[float] = []
    hausdorffs: list[float] = []
    for index, item in enumerate(sequence):
        gt = _match_gt(gt_items, item)
        chamfers.append(compute_chamfer_distance(item["mesh"], gt["mesh"], random_seed=7 + index * 2))
        hausdorffs.append(compute_hausdorff_distance(item["mesh"], gt["mesh"], random_seed=8 + index * 2))

    centroid_mean, centroid_max = _centroid_stats(meshes)
    adjacent_mean, from_base_mean = _motion_stats(meshes)
    watertight_ratio = float(sum(int(mesh.is_watertight) for mesh in meshes) / max(len(meshes), 1))
    return {
        "mesh_count": len(meshes),
        "mean_cd": float(np.mean(chamfers)),
        "mean_hd95": float(np.mean(hausdorffs)),
        "temporal_smoothness": float(compute_temporal_smoothness(meshes, random_seed=97)),
        "centroid_mean": centroid_mean,
        "centroid_max": centroid_max,
        "adjacent_mean": adjacent_mean,
        "from_base_mean": from_base_mean,
        "watertight_ratio": watertight_ratio,
    }


def _overall_score(metrics: dict[str, float | int]) -> float:
    return float(
        1.0 * float(metrics["mean_cd"])
        + 0.12 * float(metrics["mean_hd95"])
        + 0.08 * float(metrics["temporal_smoothness"])
        + 0.04 * float(metrics["centroid_max"])
        + 0.03 * float(metrics["adjacent_mean"])
        + 0.02 * float(metrics["from_base_mean"])
        - 0.5 * float(metrics["watertight_ratio"])
    )


def _write_rows(rows: list[dict[str, float | int | str]], out_csv: Path, out_json: Path) -> None:
    if not rows:
        return
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_dynamic_config(train_steps: int, mesh_resolution: int, max_points_per_phase: int) -> DynamicModelConfig:
    return DynamicModelConfig(
        enabled=True,
        max_points_per_phase=max_points_per_phase,
        canonical_hidden_dim=192,
        canonical_hidden_layers=4,
        deformation_hidden_dim=192,
        deformation_hidden_layers=4,
        train_steps=train_steps,
        learning_rate=0.001,
        surface_batch_size=1024,
        eikonal_batch_size=1024,
        temporal_batch_size=1024,
        normal_weight=0.15,
        temporal_weight=0.015,
        temporal_acceleration_weight=0.0075,
        phase_consistency_weight=0.0075,
        correspondence_temporal_weight=0.0075,
        correspondence_acceleration_weight=0.003,
        correspondence_phase_consistency_weight=0.003,
        correspondence_start_fraction=0.35,
        correspondence_ramp_fraction=0.2,
        periodicity_weight=0.05,
        deformation_weight=0.003,
        confidence_floor=0.2,
        spatial_smoothness_weight=0.05,
        centroid_weight=0.05,
        observation_support_radius_mm=6.0,
        unsupported_anchor_weight=0.02,
        unsupported_laplacian_weight=0.05,
        bootstrap_offset_weight=0.10,
        bootstrap_decay_fraction=0.25,
        base_mesh_train_steps=80,
        mesh_resolution=mesh_resolution,
        smoothing_iterations=8,
    )


def _candidate_configs(train_steps: int, mesh_resolution: int, max_points_per_phase: int, include_prior_free: bool) -> list[tuple[str, DynamicModelConfig]]:
    candidates: list[tuple[str, DynamicModelConfig]] = []

    basis_rank6 = _base_dynamic_config(train_steps, mesh_resolution, max_points_per_phase)
    basis_rank6.method = "shared_topology_global_basis_residual"
    basis_rank6.out_subdir = "search_global_basis_rank6"
    basis_rank6.global_motion_basis_rank = 6
    basis_rank6.bootstrap_offset_weight = 0.08
    basis_rank6.bootstrap_decay_fraction = 0.20
    basis_rank6.unsupported_laplacian_weight = 0.06
    candidates.append(("global_basis_rank6_bootrelax", basis_rank6))

    basis_rank8 = _base_dynamic_config(train_steps, mesh_resolution, max_points_per_phase)
    basis_rank8.method = "shared_topology_global_basis_residual"
    basis_rank8.out_subdir = "search_global_basis_rank8_guarded"
    basis_rank8.global_motion_basis_rank = 8
    basis_rank8.bootstrap_offset_weight = 0.10
    basis_rank8.bootstrap_decay_fraction = 0.25
    basis_rank8.unsupported_anchor_weight = 0.02
    basis_rank8.unsupported_laplacian_weight = 0.05
    candidates.append(("global_basis_rank8_guarded", basis_rank8))

    continuous = _base_dynamic_config(train_steps, mesh_resolution, max_points_per_phase)
    continuous.method = "shared_topology_deformation_field_reference_correspondence"
    continuous.out_subdir = "search_continuous_field"
    continuous.bootstrap_offset_weight = 0.12
    continuous.bootstrap_decay_fraction = 0.30
    continuous.unsupported_anchor_weight = 0.03
    continuous.unsupported_laplacian_weight = 0.08
    candidates.append(("continuous_field_reference", continuous))

    decoupled_motion = _base_dynamic_config(train_steps, mesh_resolution, max_points_per_phase)
    decoupled_motion.method = "shared_topology_decoupled_motion_latent"
    decoupled_motion.out_subdir = "search_decoupled_motion"
    decoupled_motion.motion_latent_dim = 64
    decoupled_motion.motion_mean_weight = 0.04
    decoupled_motion.motion_lipschitz_weight = 0.03
    decoupled_motion.bootstrap_offset_weight = 0.12
    decoupled_motion.bootstrap_decay_fraction = 0.30
    candidates.append(("decoupled_motion_balanced", decoupled_motion))

    decoupled_shape_motion = _base_dynamic_config(train_steps, mesh_resolution, max_points_per_phase)
    decoupled_shape_motion.method = "shared_topology_decoupled_shape_motion_latent"
    decoupled_shape_motion.out_subdir = "search_decoupled_shape_motion"
    decoupled_shape_motion.shape_latent_dim = 64
    decoupled_shape_motion.shape_offset_reg_weight = 0.01
    decoupled_shape_motion.shape_spatial_weight = 0.03
    decoupled_shape_motion.motion_mean_weight = 0.03
    decoupled_shape_motion.motion_lipschitz_weight = 0.02
    decoupled_shape_motion.bootstrap_offset_weight = 0.12
    decoupled_shape_motion.bootstrap_decay_fraction = 0.30
    candidates.append(("decoupled_shape_motion_balanced", decoupled_shape_motion))

    if include_prior_free:
        prior_free = _base_dynamic_config(train_steps, mesh_resolution, max_points_per_phase)
        prior_free.method = "prior_free_4d_field"
        prior_free.out_subdir = "search_prior_free_4d"
        prior_free.canonical_hidden_dim = 224
        prior_free.canonical_hidden_layers = 5
        prior_free.temporal_weight = 0.02
        prior_free.temporal_acceleration_weight = 0.01
        prior_free.phase_consistency_weight = 0.01
        prior_free.periodicity_weight = 0.05
        prior_free.eikonal_weight = 0.08
        candidates.append(("prior_free_4d_field", prior_free))

    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Search dynamic reconstruction routes against GT phase-sequence meshes")
    parser.add_argument("--pointcloud-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=72)
    parser.add_argument("--mesh-resolution", type=int, default=72)
    parser.add_argument("--max-points-per-phase", type=int, default=3000)
    parser.add_argument("--include-prior-free", action="store_true")
    args = parser.parse_args()

    pointclouds = sorted(args.pointcloud_root.glob("*.ply"))
    if not pointclouds:
        raise ValueError(f"No point clouds found in {args.pointcloud_root}")
    gt_items = _load_sequence(args.gt_dir)
    if not gt_items:
        raise ValueError(f"No GT meshes found in {args.gt_dir}")

    phase_summaries, phase_confidences = _load_phase_summaries(args.pointcloud_root)
    rows: list[dict[str, float | int | str]] = []
    best_payload: dict[str, object] | None = None

    for name, config in _candidate_configs(
        train_steps=args.train_steps,
        mesh_resolution=args.mesh_resolution,
        max_points_per_phase=args.max_points_per_phase,
        include_prior_free=bool(args.include_prior_free),
    ):
        print(f"=== RUN {name} ===")
        reconstruct_dynamic_meshes_from_pointclouds(
            pointclouds,
            config=config,
            phase_confidences=phase_confidences,
            phase_summaries=phase_summaries,
        )
        metrics = _evaluate_folder(args.pointcloud_root / config.out_subdir, gt_items)
        score = _overall_score(metrics)
        row: dict[str, float | int | str] = {
            "name": name,
            "method": config.method,
            "overall_score": score,
            "global_motion_basis_rank": int(config.global_motion_basis_rank),
            "bootstrap_offset_weight": float(config.bootstrap_offset_weight),
            "bootstrap_decay_fraction": float(config.bootstrap_decay_fraction),
            "unsupported_anchor_weight": float(config.unsupported_anchor_weight),
            "unsupported_laplacian_weight": float(config.unsupported_laplacian_weight),
            "temporal_weight": float(config.temporal_weight),
            "temporal_acceleration_weight": float(config.temporal_acceleration_weight),
            "phase_consistency_weight": float(config.phase_consistency_weight),
            "corr_temporal_weight": float(config.correspondence_temporal_weight),
            "corr_acceleration_weight": float(config.correspondence_acceleration_weight),
            "corr_phase_consistency_weight": float(config.correspondence_phase_consistency_weight),
            "shape_offset_reg_weight": float(config.shape_offset_reg_weight),
            "shape_spatial_weight": float(config.shape_spatial_weight),
            "motion_mean_weight": float(config.motion_mean_weight),
            "motion_lipschitz_weight": float(config.motion_lipschitz_weight),
            **metrics,
        }
        rows.append(row)
        if best_payload is None or score < float(best_payload["overall_score"]):
            best_payload = {
                "name": name,
                "overall_score": score,
                "config": asdict(config),
                "metrics": metrics,
            }
        print(json.dumps(row, ensure_ascii=False, indent=2))

    rows.sort(key=lambda item: (float(item["overall_score"]), float(item["mean_cd"]), float(item["mean_hd95"])))
    summary_csv = args.pointcloud_root / "gt_sequence_model_search_summary.csv"
    summary_json = args.pointcloud_root / "gt_sequence_model_search_summary.json"
    _write_rows(rows, summary_csv, summary_json)

    if best_payload is not None:
        best_path = args.pointcloud_root / "gt_sequence_best_config.json"
        best_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE {best_path}")
    print(f"WROTE {summary_csv}")
    print(f"WROTE {summary_json}")


if __name__ == "__main__":
    main()