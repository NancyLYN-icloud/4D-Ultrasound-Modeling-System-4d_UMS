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
    match = PHASE_PATTERNS[0].search(path.name)
    if match:
        return int(match.group(1)), float(match.group(2))
    match = PHASE_PATTERNS[1].search(path.name)
    if match:
        return None, float(match.group(1))
    return None, None


def _load_mesh_sequence(folder: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for path in sorted(folder.glob("dynamic_phase_*.ply")):
        mesh = trimesh.load(path, force="mesh", process=False)
        phase_index, phase = _parse_phase(path)
        items.append({"path": path, "mesh": mesh, "phase_index": phase_index, "phase": phase})
    return items


def _extract_vertices(geometry: object) -> np.ndarray:
    if isinstance(geometry, trimesh.Scene):
        vertices = []
        for item in geometry.geometry.values():
            item_vertices = np.asarray(getattr(item, "vertices", np.empty((0, 3))), dtype=np.float64)
            if len(item_vertices) > 0:
                vertices.append(item_vertices)
        if vertices:
            return np.concatenate(vertices, axis=0)
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(getattr(geometry, "vertices", np.empty((0, 3))), dtype=np.float64)


def _load_pointcloud_sequence(folder: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for path in sorted(folder.glob("*.ply")):
        if path.name.startswith("dynamic_"):
            continue
        pointcloud = trimesh.load(path, process=False)
        vertices = _extract_vertices(pointcloud)
        phase_index, phase = _parse_phase(path)
        items.append({"path": path, "points": vertices, "phase_index": phase_index, "phase": phase})
    return items


def _load_gt_sequence(folder: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for path in sorted(folder.glob("*.ply")):
        if path.name == "dynamic_base_mesh.ply":
            continue
        mesh = trimesh.load(path, force="mesh", process=False)
        phase_index, phase = _parse_phase(path)
        items.append({"path": path, "mesh": mesh, "phase_index": phase_index, "phase": phase})
    return items


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


def _match_by_phase(items: list[dict[str, object]], target: dict[str, object], mesh_key: str = "mesh") -> dict[str, object]:
    phase_index = target["phase_index"]
    if phase_index is not None:
        for item in items:
            if item["phase_index"] == phase_index:
                return item
    phase = target["phase"]
    if phase is None:
        return items[0]
    candidates = [item for item in items if item["phase"] is not None]
    if not candidates:
        return items[0]
    return min(candidates, key=lambda item: abs(float(item["phase"]) - float(phase)))


def _load_basis_diagnostics(mesh_dir: Path) -> dict[str, float]:
    path = mesh_dir / "global_basis_diagnostics.csv"
    defaults = {
        "coeff_step_mean": float("inf"),
        "coeff_accel_mean": float("inf"),
        "coeff_periodic_gap": float("inf"),
        "residual_global_ratio_mean": float("inf"),
        "residual_global_ratio_max": float("inf"),
    }
    if not path.exists():
        return defaults
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return defaults
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
    coeff_keys = [key for key in header if key.startswith("coeff_")]
    coeffs = np.asarray([[float(row[key]) for key in coeff_keys] for row in rows], dtype=np.float64)
    ratios = np.asarray([float(row["residual_global_ratio"]) for row in rows], dtype=np.float64)
    coeff_step_mean = float(np.linalg.norm(np.diff(coeffs, axis=0), axis=1).mean()) if len(coeffs) > 1 else 0.0
    coeff_accel_mean = float(np.linalg.norm(coeffs[2:] - 2.0 * coeffs[1:-1] + coeffs[:-2], axis=1).mean()) if len(coeffs) > 2 else 0.0
    coeff_periodic_gap = float(np.linalg.norm(coeffs[0] - coeffs[-1])) if len(coeffs) > 1 else 0.0
    return {
        "coeff_step_mean": coeff_step_mean,
        "coeff_accel_mean": coeff_accel_mean,
        "coeff_periodic_gap": coeff_periodic_gap,
        "residual_global_ratio_mean": float(np.mean(ratios)),
        "residual_global_ratio_max": float(np.max(ratios)),
    }


def _evaluate_unsupervised(pointcloud_root: Path, mesh_dir: Path) -> dict[str, float | int]:
    mesh_items = _load_mesh_sequence(mesh_dir)
    pointcloud_items = _load_pointcloud_sequence(pointcloud_root)
    if not mesh_items or not pointcloud_items:
        return {
            "mesh_count": 0,
            "mean_fit_cd": float("inf"),
            "mean_fit_hd95": float("inf"),
            "temporal_smoothness": float("inf"),
            "watertight_ratio": 0.0,
            **_load_basis_diagnostics(mesh_dir),
        }

    chamfers: list[float] = []
    hausdorffs: list[float] = []
    meshes: list[trimesh.Trimesh] = []
    for index, mesh_item in enumerate(mesh_items):
        pointcloud_item = _match_by_phase(pointcloud_items, mesh_item)
        points = np.asarray(pointcloud_item["points"], dtype=np.float64)
        if len(points) == 0:
            continue
        pointcloud = trimesh.points.PointCloud(points)
        mesh = mesh_item["mesh"]
        meshes.append(mesh)
        chamfers.append(compute_chamfer_distance(mesh, pointcloud, num_samples=6000, random_seed=7 + index * 2))
        hausdorffs.append(compute_hausdorff_distance(mesh, pointcloud, num_samples=6000, random_seed=8 + index * 2))

    diagnostics = _load_basis_diagnostics(mesh_dir)
    watertight_ratio = float(sum(int(mesh.is_watertight) for mesh in meshes) / max(len(meshes), 1)) if meshes else 0.0
    return {
        "mesh_count": len(meshes),
        "mean_fit_cd": float(np.mean(chamfers)) if chamfers else float("inf"),
        "mean_fit_hd95": float(np.mean(hausdorffs)) if hausdorffs else float("inf"),
        "temporal_smoothness": float(compute_temporal_smoothness(meshes, random_seed=97)) if len(meshes) >= 2 else float("inf"),
        "watertight_ratio": watertight_ratio,
        **diagnostics,
    }


def _evaluate_gt(mesh_dir: Path, gt_dir: Path) -> dict[str, float | int]:
    mesh_items = _load_mesh_sequence(mesh_dir)
    gt_items = _load_gt_sequence(gt_dir)
    if not mesh_items or not gt_items:
        return {
            "mean_cd": float("inf"),
            "mean_hd95": float("inf"),
            "gt_temporal_smoothness": float("inf"),
            "centroid_max": float("inf"),
        }
    chamfers: list[float] = []
    hausdorffs: list[float] = []
    meshes: list[trimesh.Trimesh] = []
    centroids: list[np.ndarray] = []
    for index, mesh_item in enumerate(mesh_items):
        gt_item = _match_by_phase(gt_items, mesh_item)
        mesh = mesh_item["mesh"]
        gt_mesh = gt_item["mesh"]
        meshes.append(mesh)
        chamfers.append(compute_chamfer_distance(mesh, gt_mesh, random_seed=101 + index * 2))
        hausdorffs.append(compute_hausdorff_distance(mesh, gt_mesh, random_seed=102 + index * 2))
        centroids.append(np.asarray(mesh.centroid) - np.asarray(gt_mesh.centroid))
    centroid_offsets = np.linalg.norm(np.stack(centroids, axis=0), axis=1) if centroids else np.asarray([float("inf")])
    return {
        "mean_cd": float(np.mean(chamfers)),
        "mean_hd95": float(np.mean(hausdorffs)),
        "gt_temporal_smoothness": float(compute_temporal_smoothness(meshes, random_seed=201)) if len(meshes) >= 2 else float("inf"),
        "centroid_max": float(np.max(centroid_offsets)),
    }


def _selection_score(unsupervised_metrics: dict[str, float | int], gt_metrics: dict[str, float | int] | None) -> float:
    unsupervised_score = float(
        1.0 * float(unsupervised_metrics["mean_fit_cd"])
        + 0.20 * float(unsupervised_metrics["mean_fit_hd95"])
        + 0.15 * float(unsupervised_metrics["temporal_smoothness"])
        + 0.25 * float(unsupervised_metrics["coeff_step_mean"])
        + 0.20 * float(unsupervised_metrics["coeff_accel_mean"])
        + 0.15 * float(unsupervised_metrics["coeff_periodic_gap"])
        + 0.20 * float(unsupervised_metrics["residual_global_ratio_mean"])
        + 0.10 * float(unsupervised_metrics["residual_global_ratio_max"])
        - 0.5 * float(unsupervised_metrics["watertight_ratio"])
    )
    if gt_metrics is None:
        return unsupervised_score
    gt_score = float(
        1.0 * float(gt_metrics["mean_cd"])
        + 0.15 * float(gt_metrics["mean_hd95"])
        + 0.05 * float(gt_metrics["gt_temporal_smoothness"])
        + 0.03 * float(gt_metrics["centroid_max"])
    )
    return 0.8 * gt_score + 0.2 * unsupervised_score


def _write_rows(rows: list[dict[str, float | int | str]], out_csv: Path, out_json: Path) -> None:
    if not rows:
        return
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_config(train_steps: int, mesh_resolution: int, max_points_per_phase: int) -> DynamicModelConfig:
    return DynamicModelConfig(
        enabled=True,
        method="shared_topology_global_basis_residual",
        out_subdir="dynamic_meshes",
        max_points_per_phase=max_points_per_phase,
        canonical_hidden_dim=128,
        canonical_hidden_layers=4,
        deformation_hidden_dim=128,
        deformation_hidden_layers=3,
        train_steps=train_steps,
        learning_rate=0.001,
        surface_batch_size=1024,
        temporal_batch_size=1024,
        normal_weight=0.15,
        temporal_weight=0.10,
        temporal_acceleration_weight=0.05,
        phase_consistency_weight=0.05,
        periodicity_weight=0.10,
        deformation_weight=0.01,
        confidence_floor=0.2,
        spatial_smoothness_weight=0.05,
        centroid_weight=0.05,
        observation_support_radius_mm=6.0,
        unsupported_anchor_weight=0.02,
        unsupported_laplacian_weight=0.05,
        bootstrap_offset_weight=0.10,
        bootstrap_decay_fraction=0.25,
        bootstrap_teacher_weight=0.0,
        bootstrap_teacher_global_only=False,
        bootstrap_teacher_support_aware=False,
        bootstrap_teacher_support_power=1.0,
        bootstrap_teacher_support_floor=0.0,
        bootstrap_teacher_pred_to_target_weight=1.0,
        bootstrap_teacher_target_to_pred_weight=1.0,
        bootstrap_teacher_start_fraction=0.0,
        bootstrap_teacher_ramp_fraction=0.0,
        basis_coefficient_bootstrap_weight=0.08,
        basis_temporal_weight=0.03,
        basis_acceleration_weight=0.015,
        basis_periodicity_weight=0.03,
        residual_mean_weight=0.02,
        residual_basis_projection_weight=0.03,
        residual_locality_weight=0.0,
        residual_locality_budget_scale=1.5,
        residual_locality_global_budget_scale=0.15,
        residual_global_ratio_weight=0.0,
        residual_global_ratio_target=0.85,
        mesh_resolution=mesh_resolution,
        smoothing_iterations=8,
    )


def _candidate_configs(train_steps: int, mesh_resolution: int, max_points_per_phase: int) -> list[tuple[str, DynamicModelConfig]]:
    candidates: list[tuple[str, DynamicModelConfig]] = []

    rank6_balanced = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_balanced.out_subdir = "focused_rank6_balanced"
    rank6_balanced.global_motion_basis_rank = 6
    candidates.append(("rank6_balanced", rank6_balanced))

    rank6_coeff_temporal = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal.out_subdir = "focused_rank6_coeff_temporal"
    rank6_coeff_temporal.global_motion_basis_rank = 6
    rank6_coeff_temporal.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_temporal.basis_temporal_weight = 0.05
    rank6_coeff_temporal.basis_acceleration_weight = 0.02
    rank6_coeff_temporal.basis_periodicity_weight = 0.04
    candidates.append(("rank6_coeff_temporal", rank6_coeff_temporal))

    rank6_residual_split = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_residual_split.out_subdir = "focused_rank6_residual_split"
    rank6_residual_split.global_motion_basis_rank = 6
    rank6_residual_split.residual_mean_weight = 0.03
    rank6_residual_split.residual_basis_projection_weight = 0.05
    rank6_residual_split.unsupported_laplacian_weight = 0.06
    candidates.append(("rank6_residual_split", rank6_residual_split))

    rank6_coeff_residual_balanced = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_residual_balanced.out_subdir = "focused_rank6_coeff_residual_balanced"
    rank6_coeff_residual_balanced.global_motion_basis_rank = 6
    rank6_coeff_residual_balanced.basis_coefficient_bootstrap_weight = 0.10
    rank6_coeff_residual_balanced.basis_temporal_weight = 0.04
    rank6_coeff_residual_balanced.basis_acceleration_weight = 0.018
    rank6_coeff_residual_balanced.basis_periodicity_weight = 0.035
    rank6_coeff_residual_balanced.residual_mean_weight = 0.025
    rank6_coeff_residual_balanced.residual_basis_projection_weight = 0.04
    rank6_coeff_residual_balanced.unsupported_laplacian_weight = 0.055
    candidates.append(("rank6_coeff_residual_balanced", rank6_coeff_residual_balanced))

    rank6_coeff_residual_guarded = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_residual_guarded.out_subdir = "focused_rank6_coeff_residual_guarded"
    rank6_coeff_residual_guarded.global_motion_basis_rank = 6
    rank6_coeff_residual_guarded.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_residual_guarded.basis_temporal_weight = 0.05
    rank6_coeff_residual_guarded.basis_acceleration_weight = 0.02
    rank6_coeff_residual_guarded.basis_periodicity_weight = 0.04
    rank6_coeff_residual_guarded.residual_mean_weight = 0.03
    rank6_coeff_residual_guarded.residual_basis_projection_weight = 0.05
    rank6_coeff_residual_guarded.unsupported_laplacian_weight = 0.06
    candidates.append(("rank6_coeff_residual_guarded", rank6_coeff_residual_guarded))

    rank6_coeff_temporal_light = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal_light.out_subdir = "focused_rank6_coeff_temporal_light"
    rank6_coeff_temporal_light.global_motion_basis_rank = 6
    rank6_coeff_temporal_light.basis_coefficient_bootstrap_weight = 0.09
    rank6_coeff_temporal_light.basis_temporal_weight = 0.04
    rank6_coeff_temporal_light.basis_acceleration_weight = 0.017
    rank6_coeff_temporal_light.basis_periodicity_weight = 0.035
    rank6_coeff_temporal_light.residual_mean_weight = 0.02
    rank6_coeff_temporal_light.residual_basis_projection_weight = 0.035
    candidates.append(("rank6_coeff_temporal_light", rank6_coeff_temporal_light))

    rank8_balanced = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank8_balanced.out_subdir = "focused_rank8_balanced"
    rank8_balanced.global_motion_basis_rank = 8
    candidates.append(("rank8_balanced", rank8_balanced))

    rank8_coeff_temporal = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank8_coeff_temporal.out_subdir = "focused_rank8_coeff_temporal"
    rank8_coeff_temporal.global_motion_basis_rank = 8
    rank8_coeff_temporal.basis_coefficient_bootstrap_weight = 0.14
    rank8_coeff_temporal.basis_temporal_weight = 0.05
    rank8_coeff_temporal.basis_acceleration_weight = 0.025
    rank8_coeff_temporal.basis_periodicity_weight = 0.05
    candidates.append(("rank8_coeff_temporal", rank8_coeff_temporal))

    rank6_coeff_temporal_proj_light = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal_proj_light.out_subdir = "focused_rank6_coeff_temporal_proj_light"
    rank6_coeff_temporal_proj_light.global_motion_basis_rank = 6
    rank6_coeff_temporal_proj_light.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_temporal_proj_light.basis_temporal_weight = 0.05
    rank6_coeff_temporal_proj_light.basis_acceleration_weight = 0.025
    rank6_coeff_temporal_proj_light.basis_periodicity_weight = 0.04
    rank6_coeff_temporal_proj_light.residual_basis_projection_weight = 0.035
    rank6_coeff_temporal_proj_light.unsupported_laplacian_weight = 0.045
    candidates.append(("rank6_coeff_temporal_proj_light", rank6_coeff_temporal_proj_light))

    rank6_coeff_temporal_proj_mid = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal_proj_mid.out_subdir = "focused_rank6_coeff_temporal_proj_mid"
    rank6_coeff_temporal_proj_mid.global_motion_basis_rank = 6
    rank6_coeff_temporal_proj_mid.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_temporal_proj_mid.basis_temporal_weight = 0.05
    rank6_coeff_temporal_proj_mid.basis_acceleration_weight = 0.025
    rank6_coeff_temporal_proj_mid.basis_periodicity_weight = 0.04
    rank6_coeff_temporal_proj_mid.residual_basis_projection_weight = 0.04
    rank6_coeff_temporal_proj_mid.unsupported_laplacian_weight = 0.05
    candidates.append(("rank6_coeff_temporal_proj_mid", rank6_coeff_temporal_proj_mid))

    rank6_coeff_temporal_proj_guarded = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal_proj_guarded.out_subdir = "focused_rank6_coeff_temporal_proj_guarded"
    rank6_coeff_temporal_proj_guarded.global_motion_basis_rank = 6
    rank6_coeff_temporal_proj_guarded.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_temporal_proj_guarded.basis_temporal_weight = 0.05
    rank6_coeff_temporal_proj_guarded.basis_acceleration_weight = 0.025
    rank6_coeff_temporal_proj_guarded.basis_periodicity_weight = 0.04
    rank6_coeff_temporal_proj_guarded.residual_basis_projection_weight = 0.045
    rank6_coeff_temporal_proj_guarded.unsupported_laplacian_weight = 0.055
    candidates.append(("rank6_coeff_temporal_proj_guarded", rank6_coeff_temporal_proj_guarded))

    rank6_coeff_temporal_corr = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal_corr.out_subdir = "focused_rank6_coeff_temporal_corr"
    rank6_coeff_temporal_corr.global_motion_basis_rank = 6
    rank6_coeff_temporal_corr.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_temporal_corr.basis_temporal_weight = 0.05
    rank6_coeff_temporal_corr.basis_acceleration_weight = 0.025
    rank6_coeff_temporal_corr.basis_periodicity_weight = 0.04
    rank6_coeff_temporal_corr.residual_basis_projection_weight = 0.04
    rank6_coeff_temporal_corr.unsupported_laplacian_weight = 0.05
    rank6_coeff_temporal_corr.correspondence_temporal_weight = 0.008
    rank6_coeff_temporal_corr.correspondence_acceleration_weight = 0.004
    rank6_coeff_temporal_corr.correspondence_phase_consistency_weight = 0.004
    rank6_coeff_temporal_corr.correspondence_start_fraction = 0.30
    rank6_coeff_temporal_corr.correspondence_ramp_fraction = 0.25
    candidates.append(("rank6_coeff_temporal_corr", rank6_coeff_temporal_corr))

    rank6_coeff_temporal_corr_global = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal_corr_global.out_subdir = "focused_rank6_coeff_temporal_corr_global"
    rank6_coeff_temporal_corr_global.global_motion_basis_rank = 6
    rank6_coeff_temporal_corr_global.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_temporal_corr_global.basis_temporal_weight = 0.05
    rank6_coeff_temporal_corr_global.basis_acceleration_weight = 0.025
    rank6_coeff_temporal_corr_global.basis_periodicity_weight = 0.04
    rank6_coeff_temporal_corr_global.residual_basis_projection_weight = 0.04
    rank6_coeff_temporal_corr_global.unsupported_laplacian_weight = 0.05
    rank6_coeff_temporal_corr_global.correspondence_temporal_weight = 0.008
    rank6_coeff_temporal_corr_global.correspondence_acceleration_weight = 0.004
    rank6_coeff_temporal_corr_global.correspondence_phase_consistency_weight = 0.004
    rank6_coeff_temporal_corr_global.correspondence_global_only = True
    rank6_coeff_temporal_corr_global.correspondence_start_fraction = 0.30
    rank6_coeff_temporal_corr_global.correspondence_ramp_fraction = 0.25
    candidates.append(("rank6_coeff_temporal_corr_global", rank6_coeff_temporal_corr_global))

    rank6_coeff_temporal_corr_global_strong = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal_corr_global_strong.out_subdir = "focused_rank6_coeff_temporal_corr_global_strong"
    rank6_coeff_temporal_corr_global_strong.global_motion_basis_rank = 6
    rank6_coeff_temporal_corr_global_strong.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_temporal_corr_global_strong.basis_temporal_weight = 0.05
    rank6_coeff_temporal_corr_global_strong.basis_acceleration_weight = 0.025
    rank6_coeff_temporal_corr_global_strong.basis_periodicity_weight = 0.04
    rank6_coeff_temporal_corr_global_strong.residual_basis_projection_weight = 0.04
    rank6_coeff_temporal_corr_global_strong.unsupported_laplacian_weight = 0.05
    rank6_coeff_temporal_corr_global_strong.correspondence_temporal_weight = 0.016
    rank6_coeff_temporal_corr_global_strong.correspondence_acceleration_weight = 0.008
    rank6_coeff_temporal_corr_global_strong.correspondence_phase_consistency_weight = 0.008
    rank6_coeff_temporal_corr_global_strong.correspondence_global_only = True
    rank6_coeff_temporal_corr_global_strong.correspondence_start_fraction = 0.25
    rank6_coeff_temporal_corr_global_strong.correspondence_ramp_fraction = 0.25
    candidates.append(("rank6_coeff_temporal_corr_global_strong", rank6_coeff_temporal_corr_global_strong))

    rank6_coeff_temporal_corr_global_strong_propagated = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal_corr_global_strong_propagated.out_subdir = "focused_rank6_coeff_temporal_corr_global_strong_propagated"
    rank6_coeff_temporal_corr_global_strong_propagated.global_motion_basis_rank = 6
    rank6_coeff_temporal_corr_global_strong_propagated.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_temporal_corr_global_strong_propagated.basis_temporal_weight = 0.05
    rank6_coeff_temporal_corr_global_strong_propagated.basis_acceleration_weight = 0.025
    rank6_coeff_temporal_corr_global_strong_propagated.basis_periodicity_weight = 0.04
    rank6_coeff_temporal_corr_global_strong_propagated.residual_basis_projection_weight = 0.04
    rank6_coeff_temporal_corr_global_strong_propagated.unsupported_laplacian_weight = 0.05
    rank6_coeff_temporal_corr_global_strong_propagated.correspondence_temporal_weight = 0.016
    rank6_coeff_temporal_corr_global_strong_propagated.correspondence_acceleration_weight = 0.008
    rank6_coeff_temporal_corr_global_strong_propagated.correspondence_phase_consistency_weight = 0.008
    rank6_coeff_temporal_corr_global_strong_propagated.correspondence_global_only = True
    rank6_coeff_temporal_corr_global_strong_propagated.correspondence_start_fraction = 0.25
    rank6_coeff_temporal_corr_global_strong_propagated.correspondence_ramp_fraction = 0.25
    rank6_coeff_temporal_corr_global_strong_propagated.unsupported_propagation_iterations = 8
    rank6_coeff_temporal_corr_global_strong_propagated.unsupported_propagation_neighbor_weight = 0.75
    candidates.append(("rank6_coeff_temporal_corr_global_strong_propagated", rank6_coeff_temporal_corr_global_strong_propagated))

    rank6_coeff_temporal_corr_global_strong_propagated_locality_light = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.out_subdir = "focused_rank6_coeff_temporal_corr_global_strong_propagated_locality_light"
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.global_motion_basis_rank = 6
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.basis_temporal_weight = 0.05
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.basis_acceleration_weight = 0.025
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.basis_periodicity_weight = 0.04
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.residual_basis_projection_weight = 0.04
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.unsupported_laplacian_weight = 0.05
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.correspondence_temporal_weight = 0.016
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.correspondence_acceleration_weight = 0.008
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.correspondence_phase_consistency_weight = 0.008
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.correspondence_global_only = True
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.correspondence_start_fraction = 0.25
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.correspondence_ramp_fraction = 0.25
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.unsupported_propagation_iterations = 8
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.unsupported_propagation_neighbor_weight = 0.75
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.residual_locality_weight = 0.015
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.residual_locality_budget_scale = 2.0
    rank6_coeff_temporal_corr_global_strong_propagated_locality_light.residual_locality_global_budget_scale = 0.20
    candidates.append(("rank6_coeff_temporal_corr_global_strong_propagated_locality_light", rank6_coeff_temporal_corr_global_strong_propagated_locality_light))

    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.out_subdir = "focused_rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft"
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.global_motion_basis_rank = 6
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.basis_temporal_weight = 0.05
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.basis_acceleration_weight = 0.025
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.basis_periodicity_weight = 0.04
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.residual_basis_projection_weight = 0.04
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.unsupported_laplacian_weight = 0.05
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.correspondence_temporal_weight = 0.016
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.correspondence_acceleration_weight = 0.008
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.correspondence_phase_consistency_weight = 0.008
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.correspondence_global_only = True
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.correspondence_start_fraction = 0.25
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.correspondence_ramp_fraction = 0.25
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.unsupported_propagation_iterations = 8
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.unsupported_propagation_neighbor_weight = 0.75
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.residual_global_ratio_weight = 0.01
    rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft.residual_global_ratio_target = 1.20
    candidates.append(("rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft", rank6_coeff_temporal_corr_global_strong_propagated_ratio_soft))

    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light.out_subdir = "focused_rank6_coeff_temporal_corr_global_strong_propagated_anchor_light"
    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light.global_motion_basis_rank = 6
    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light.basis_temporal_weight = 0.05
    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light.basis_acceleration_weight = 0.025
    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light.basis_periodicity_weight = 0.04
    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light.residual_basis_projection_weight = 0.04
    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light.correspondence_temporal_weight = 0.016
    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light.correspondence_acceleration_weight = 0.008
    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light.correspondence_phase_consistency_weight = 0.008
    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light.correspondence_global_only = True
    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light.correspondence_start_fraction = 0.25
    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light.correspondence_ramp_fraction = 0.25
    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light.unsupported_propagation_iterations = 8
    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light.unsupported_propagation_neighbor_weight = 0.75
    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light.unsupported_anchor_weight = 0.03
    rank6_coeff_temporal_corr_global_strong_propagated_anchor_light.unsupported_laplacian_weight = 0.06
    candidates.append(("rank6_coeff_temporal_corr_global_strong_propagated_anchor_light", rank6_coeff_temporal_corr_global_strong_propagated_anchor_light))

    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.out_subdir = "focused_rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio"
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.global_motion_basis_rank = 6
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.basis_temporal_weight = 0.05
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.basis_acceleration_weight = 0.025
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.basis_periodicity_weight = 0.04
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.residual_basis_projection_weight = 0.04
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.unsupported_laplacian_weight = 0.05
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.correspondence_temporal_weight = 0.016
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.correspondence_acceleration_weight = 0.008
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.correspondence_phase_consistency_weight = 0.008
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.correspondence_global_only = True
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.correspondence_start_fraction = 0.25
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.correspondence_ramp_fraction = 0.25
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.unsupported_propagation_iterations = 8
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.unsupported_propagation_neighbor_weight = 0.75
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.residual_locality_weight = 0.01
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.residual_locality_budget_scale = 2.2
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.residual_locality_global_budget_scale = 0.20
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.residual_global_ratio_weight = 0.008
    rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio.residual_global_ratio_target = 1.25
    candidates.append(("rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio", rank6_coeff_temporal_corr_global_strong_propagated_locality_ratio))

    rank6_coeff_temporal_corr_global_strong_propagated_gated = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal_corr_global_strong_propagated_gated.out_subdir = "focused_rank6_coeff_temporal_corr_global_strong_propagated_gated"
    rank6_coeff_temporal_corr_global_strong_propagated_gated.global_motion_basis_rank = 6
    rank6_coeff_temporal_corr_global_strong_propagated_gated.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_temporal_corr_global_strong_propagated_gated.basis_temporal_weight = 0.05
    rank6_coeff_temporal_corr_global_strong_propagated_gated.basis_acceleration_weight = 0.025
    rank6_coeff_temporal_corr_global_strong_propagated_gated.basis_periodicity_weight = 0.04
    rank6_coeff_temporal_corr_global_strong_propagated_gated.residual_basis_projection_weight = 0.04
    rank6_coeff_temporal_corr_global_strong_propagated_gated.unsupported_laplacian_weight = 0.05
    rank6_coeff_temporal_corr_global_strong_propagated_gated.correspondence_temporal_weight = 0.016
    rank6_coeff_temporal_corr_global_strong_propagated_gated.correspondence_acceleration_weight = 0.008
    rank6_coeff_temporal_corr_global_strong_propagated_gated.correspondence_phase_consistency_weight = 0.008
    rank6_coeff_temporal_corr_global_strong_propagated_gated.correspondence_global_only = True
    rank6_coeff_temporal_corr_global_strong_propagated_gated.correspondence_bootstrap_gate = True
    rank6_coeff_temporal_corr_global_strong_propagated_gated.correspondence_bootstrap_gate_strength = 2.0
    rank6_coeff_temporal_corr_global_strong_propagated_gated.correspondence_start_fraction = 0.25
    rank6_coeff_temporal_corr_global_strong_propagated_gated.correspondence_ramp_fraction = 0.25
    rank6_coeff_temporal_corr_global_strong_propagated_gated.unsupported_propagation_iterations = 8
    rank6_coeff_temporal_corr_global_strong_propagated_gated.unsupported_propagation_neighbor_weight = 0.75
    candidates.append(("rank6_coeff_temporal_corr_global_strong_propagated_gated", rank6_coeff_temporal_corr_global_strong_propagated_gated))

    rank6_corr_global_strong_propagated_teacher = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_corr_global_strong_propagated_teacher.out_subdir = "focused_rank6_corr_global_strong_propagated_teacher"
    rank6_corr_global_strong_propagated_teacher.global_motion_basis_rank = 6
    rank6_corr_global_strong_propagated_teacher.basis_coefficient_bootstrap_weight = 0.12
    rank6_corr_global_strong_propagated_teacher.basis_temporal_weight = 0.05
    rank6_corr_global_strong_propagated_teacher.basis_acceleration_weight = 0.025
    rank6_corr_global_strong_propagated_teacher.basis_periodicity_weight = 0.04
    rank6_corr_global_strong_propagated_teacher.residual_basis_projection_weight = 0.04
    rank6_corr_global_strong_propagated_teacher.unsupported_laplacian_weight = 0.05
    rank6_corr_global_strong_propagated_teacher.correspondence_temporal_weight = 0.016
    rank6_corr_global_strong_propagated_teacher.correspondence_acceleration_weight = 0.008
    rank6_corr_global_strong_propagated_teacher.correspondence_phase_consistency_weight = 0.008
    rank6_corr_global_strong_propagated_teacher.correspondence_global_only = True
    rank6_corr_global_strong_propagated_teacher.correspondence_start_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher.correspondence_ramp_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher.unsupported_propagation_iterations = 8
    rank6_corr_global_strong_propagated_teacher.unsupported_propagation_neighbor_weight = 0.75
    rank6_corr_global_strong_propagated_teacher.bootstrap_teacher_weight = 0.03
    candidates.append(("rank6_corr_global_strong_propagated_teacher", rank6_corr_global_strong_propagated_teacher))

    rank6_corr_global_strong_propagated_teacher_light = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_corr_global_strong_propagated_teacher_light.out_subdir = "focused_rank6_corr_global_strong_propagated_teacher_light"
    rank6_corr_global_strong_propagated_teacher_light.global_motion_basis_rank = 6
    rank6_corr_global_strong_propagated_teacher_light.basis_coefficient_bootstrap_weight = 0.12
    rank6_corr_global_strong_propagated_teacher_light.basis_temporal_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_light.basis_acceleration_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_light.basis_periodicity_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_light.residual_basis_projection_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_light.unsupported_laplacian_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_light.correspondence_temporal_weight = 0.016
    rank6_corr_global_strong_propagated_teacher_light.correspondence_acceleration_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_light.correspondence_phase_consistency_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_light.correspondence_global_only = True
    rank6_corr_global_strong_propagated_teacher_light.correspondence_start_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_light.correspondence_ramp_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_light.unsupported_propagation_iterations = 8
    rank6_corr_global_strong_propagated_teacher_light.unsupported_propagation_neighbor_weight = 0.75
    rank6_corr_global_strong_propagated_teacher_light.bootstrap_teacher_weight = 0.015
    candidates.append(("rank6_corr_global_strong_propagated_teacher_light", rank6_corr_global_strong_propagated_teacher_light))

    rank6_corr_global_strong_propagated_teacher_delayed = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_corr_global_strong_propagated_teacher_delayed.out_subdir = "focused_rank6_corr_global_strong_propagated_teacher_delayed"
    rank6_corr_global_strong_propagated_teacher_delayed.global_motion_basis_rank = 6
    rank6_corr_global_strong_propagated_teacher_delayed.basis_coefficient_bootstrap_weight = 0.12
    rank6_corr_global_strong_propagated_teacher_delayed.basis_temporal_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_delayed.basis_acceleration_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_delayed.basis_periodicity_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_delayed.residual_basis_projection_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_delayed.unsupported_laplacian_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_delayed.correspondence_temporal_weight = 0.016
    rank6_corr_global_strong_propagated_teacher_delayed.correspondence_acceleration_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_delayed.correspondence_phase_consistency_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_delayed.correspondence_global_only = True
    rank6_corr_global_strong_propagated_teacher_delayed.correspondence_start_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_delayed.correspondence_ramp_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_delayed.unsupported_propagation_iterations = 8
    rank6_corr_global_strong_propagated_teacher_delayed.unsupported_propagation_neighbor_weight = 0.75
    rank6_corr_global_strong_propagated_teacher_delayed.bootstrap_teacher_weight = 0.02
    rank6_corr_global_strong_propagated_teacher_delayed.bootstrap_teacher_start_fraction = 0.35
    rank6_corr_global_strong_propagated_teacher_delayed.bootstrap_teacher_ramp_fraction = 0.20
    candidates.append(("rank6_corr_global_strong_propagated_teacher_delayed", rank6_corr_global_strong_propagated_teacher_delayed))

    rank6_corr_global_strong_propagated_teacher_delayed_mid = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_corr_global_strong_propagated_teacher_delayed_mid.out_subdir = "focused_rank6_corr_global_strong_propagated_teacher_delayed_mid"
    rank6_corr_global_strong_propagated_teacher_delayed_mid.global_motion_basis_rank = 6
    rank6_corr_global_strong_propagated_teacher_delayed_mid.basis_coefficient_bootstrap_weight = 0.12
    rank6_corr_global_strong_propagated_teacher_delayed_mid.basis_temporal_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_delayed_mid.basis_acceleration_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_delayed_mid.basis_periodicity_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_delayed_mid.residual_basis_projection_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_delayed_mid.unsupported_laplacian_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_delayed_mid.correspondence_temporal_weight = 0.016
    rank6_corr_global_strong_propagated_teacher_delayed_mid.correspondence_acceleration_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_delayed_mid.correspondence_phase_consistency_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_delayed_mid.correspondence_global_only = True
    rank6_corr_global_strong_propagated_teacher_delayed_mid.correspondence_start_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_delayed_mid.correspondence_ramp_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_delayed_mid.unsupported_propagation_iterations = 8
    rank6_corr_global_strong_propagated_teacher_delayed_mid.unsupported_propagation_neighbor_weight = 0.75
    rank6_corr_global_strong_propagated_teacher_delayed_mid.bootstrap_teacher_weight = 0.03
    rank6_corr_global_strong_propagated_teacher_delayed_mid.bootstrap_teacher_start_fraction = 0.40
    rank6_corr_global_strong_propagated_teacher_delayed_mid.bootstrap_teacher_ramp_fraction = 0.20
    candidates.append(("rank6_corr_global_strong_propagated_teacher_delayed_mid", rank6_corr_global_strong_propagated_teacher_delayed_mid))

    rank6_corr_global_strong_propagated_teacher_global_delayed = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_corr_global_strong_propagated_teacher_global_delayed.out_subdir = "focused_rank6_corr_global_strong_propagated_teacher_global_delayed"
    rank6_corr_global_strong_propagated_teacher_global_delayed.global_motion_basis_rank = 6
    rank6_corr_global_strong_propagated_teacher_global_delayed.basis_coefficient_bootstrap_weight = 0.12
    rank6_corr_global_strong_propagated_teacher_global_delayed.basis_temporal_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_delayed.basis_acceleration_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_global_delayed.basis_periodicity_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_delayed.residual_basis_projection_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_delayed.unsupported_laplacian_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_delayed.correspondence_temporal_weight = 0.016
    rank6_corr_global_strong_propagated_teacher_global_delayed.correspondence_acceleration_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_delayed.correspondence_phase_consistency_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_delayed.correspondence_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_delayed.correspondence_start_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_delayed.correspondence_ramp_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_delayed.unsupported_propagation_iterations = 8
    rank6_corr_global_strong_propagated_teacher_global_delayed.unsupported_propagation_neighbor_weight = 0.75
    rank6_corr_global_strong_propagated_teacher_global_delayed.bootstrap_teacher_weight = 0.02
    rank6_corr_global_strong_propagated_teacher_global_delayed.bootstrap_teacher_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_delayed.bootstrap_teacher_start_fraction = 0.35
    rank6_corr_global_strong_propagated_teacher_global_delayed.bootstrap_teacher_ramp_fraction = 0.20
    candidates.append(("rank6_corr_global_strong_propagated_teacher_global_delayed", rank6_corr_global_strong_propagated_teacher_global_delayed))

    rank6_corr_global_strong_propagated_teacher_global_delayed_mid = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.out_subdir = "focused_rank6_corr_global_strong_propagated_teacher_global_delayed_mid"
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.global_motion_basis_rank = 6
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.basis_coefficient_bootstrap_weight = 0.12
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.basis_temporal_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.basis_acceleration_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.basis_periodicity_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.residual_basis_projection_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.unsupported_laplacian_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.correspondence_temporal_weight = 0.016
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.correspondence_acceleration_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.correspondence_phase_consistency_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.correspondence_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.correspondence_start_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.correspondence_ramp_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.unsupported_propagation_iterations = 8
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.unsupported_propagation_neighbor_weight = 0.75
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.bootstrap_teacher_weight = 0.03
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.bootstrap_teacher_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.bootstrap_teacher_start_fraction = 0.40
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid.bootstrap_teacher_ramp_fraction = 0.20
    candidates.append(("rank6_corr_global_strong_propagated_teacher_global_delayed_mid", rank6_corr_global_strong_propagated_teacher_global_delayed_mid))

    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.out_subdir = "focused_rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light"
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.global_motion_basis_rank = 6
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.basis_coefficient_bootstrap_weight = 0.12
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.basis_temporal_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.basis_acceleration_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.basis_periodicity_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.residual_basis_projection_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.unsupported_laplacian_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.correspondence_temporal_weight = 0.016
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.correspondence_acceleration_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.correspondence_phase_consistency_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.correspondence_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.correspondence_start_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.correspondence_ramp_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.unsupported_propagation_iterations = 8
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.unsupported_propagation_neighbor_weight = 0.75
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.bootstrap_teacher_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.bootstrap_teacher_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.bootstrap_teacher_start_fraction = 0.40
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light.bootstrap_teacher_ramp_fraction = 0.20
    candidates.append(("rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light", rank6_corr_global_strong_propagated_teacher_global_delayed_mid_light))

    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.out_subdir = "focused_rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft"
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.global_motion_basis_rank = 6
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.basis_coefficient_bootstrap_weight = 0.12
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.basis_temporal_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.basis_acceleration_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.basis_periodicity_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.residual_basis_projection_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.unsupported_laplacian_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.correspondence_temporal_weight = 0.016
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.correspondence_acceleration_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.correspondence_phase_consistency_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.correspondence_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.correspondence_start_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.correspondence_ramp_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.unsupported_propagation_iterations = 8
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.unsupported_propagation_neighbor_weight = 0.75
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.bootstrap_teacher_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.bootstrap_teacher_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.bootstrap_teacher_start_fraction = 0.45
    rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft.bootstrap_teacher_ramp_fraction = 0.15
    candidates.append(("rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft", rank6_corr_global_strong_propagated_teacher_global_delayed_mid_soft))

    rank6_corr_global_strong_propagated_teacher_global_support = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_corr_global_strong_propagated_teacher_global_support.out_subdir = "focused_rank6_corr_global_strong_propagated_teacher_global_support"
    rank6_corr_global_strong_propagated_teacher_global_support.global_motion_basis_rank = 6
    rank6_corr_global_strong_propagated_teacher_global_support.basis_coefficient_bootstrap_weight = 0.12
    rank6_corr_global_strong_propagated_teacher_global_support.basis_temporal_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_support.basis_acceleration_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_global_support.basis_periodicity_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_support.residual_basis_projection_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_support.unsupported_laplacian_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_support.correspondence_temporal_weight = 0.016
    rank6_corr_global_strong_propagated_teacher_global_support.correspondence_acceleration_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_support.correspondence_phase_consistency_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_support.correspondence_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_support.correspondence_start_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_support.correspondence_ramp_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_support.unsupported_propagation_iterations = 8
    rank6_corr_global_strong_propagated_teacher_global_support.unsupported_propagation_neighbor_weight = 0.75
    rank6_corr_global_strong_propagated_teacher_global_support.bootstrap_teacher_weight = 0.03
    rank6_corr_global_strong_propagated_teacher_global_support.bootstrap_teacher_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_support.bootstrap_teacher_support_aware = True
    rank6_corr_global_strong_propagated_teacher_global_support.bootstrap_teacher_support_power = 2.0
    rank6_corr_global_strong_propagated_teacher_global_support.bootstrap_teacher_support_floor = 0.10
    rank6_corr_global_strong_propagated_teacher_global_support.bootstrap_teacher_start_fraction = 0.40
    rank6_corr_global_strong_propagated_teacher_global_support.bootstrap_teacher_ramp_fraction = 0.20
    candidates.append(("rank6_corr_global_strong_propagated_teacher_global_support", rank6_corr_global_strong_propagated_teacher_global_support))

    rank6_corr_global_strong_propagated_teacher_global_support_light = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_corr_global_strong_propagated_teacher_global_support_light.out_subdir = "focused_rank6_corr_global_strong_propagated_teacher_global_support_light"
    rank6_corr_global_strong_propagated_teacher_global_support_light.global_motion_basis_rank = 6
    rank6_corr_global_strong_propagated_teacher_global_support_light.basis_coefficient_bootstrap_weight = 0.12
    rank6_corr_global_strong_propagated_teacher_global_support_light.basis_temporal_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_support_light.basis_acceleration_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_global_support_light.basis_periodicity_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_support_light.residual_basis_projection_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_support_light.unsupported_laplacian_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_support_light.correspondence_temporal_weight = 0.016
    rank6_corr_global_strong_propagated_teacher_global_support_light.correspondence_acceleration_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_support_light.correspondence_phase_consistency_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_support_light.correspondence_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_support_light.correspondence_start_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_support_light.correspondence_ramp_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_support_light.unsupported_propagation_iterations = 8
    rank6_corr_global_strong_propagated_teacher_global_support_light.unsupported_propagation_neighbor_weight = 0.75
    rank6_corr_global_strong_propagated_teacher_global_support_light.bootstrap_teacher_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_global_support_light.bootstrap_teacher_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_support_light.bootstrap_teacher_support_aware = True
    rank6_corr_global_strong_propagated_teacher_global_support_light.bootstrap_teacher_support_power = 2.0
    rank6_corr_global_strong_propagated_teacher_global_support_light.bootstrap_teacher_support_floor = 0.10
    rank6_corr_global_strong_propagated_teacher_global_support_light.bootstrap_teacher_start_fraction = 0.40
    rank6_corr_global_strong_propagated_teacher_global_support_light.bootstrap_teacher_ramp_fraction = 0.20
    candidates.append(("rank6_corr_global_strong_propagated_teacher_global_support_light", rank6_corr_global_strong_propagated_teacher_global_support_light))

    rank6_corr_global_strong_propagated_teacher_global_support_asym = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_corr_global_strong_propagated_teacher_global_support_asym.out_subdir = "focused_rank6_corr_global_strong_propagated_teacher_global_support_asym"
    rank6_corr_global_strong_propagated_teacher_global_support_asym.global_motion_basis_rank = 6
    rank6_corr_global_strong_propagated_teacher_global_support_asym.basis_coefficient_bootstrap_weight = 0.12
    rank6_corr_global_strong_propagated_teacher_global_support_asym.basis_temporal_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_support_asym.basis_acceleration_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_global_support_asym.basis_periodicity_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_support_asym.residual_basis_projection_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_support_asym.unsupported_laplacian_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_support_asym.correspondence_temporal_weight = 0.016
    rank6_corr_global_strong_propagated_teacher_global_support_asym.correspondence_acceleration_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_support_asym.correspondence_phase_consistency_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_support_asym.correspondence_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_support_asym.correspondence_start_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_support_asym.correspondence_ramp_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_support_asym.unsupported_propagation_iterations = 8
    rank6_corr_global_strong_propagated_teacher_global_support_asym.unsupported_propagation_neighbor_weight = 0.75
    rank6_corr_global_strong_propagated_teacher_global_support_asym.bootstrap_teacher_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_global_support_asym.bootstrap_teacher_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_support_asym.bootstrap_teacher_support_aware = True
    rank6_corr_global_strong_propagated_teacher_global_support_asym.bootstrap_teacher_support_power = 2.0
    rank6_corr_global_strong_propagated_teacher_global_support_asym.bootstrap_teacher_support_floor = 0.10
    rank6_corr_global_strong_propagated_teacher_global_support_asym.bootstrap_teacher_pred_to_target_weight = 1.0
    rank6_corr_global_strong_propagated_teacher_global_support_asym.bootstrap_teacher_target_to_pred_weight = 0.25
    rank6_corr_global_strong_propagated_teacher_global_support_asym.bootstrap_teacher_start_fraction = 0.40
    rank6_corr_global_strong_propagated_teacher_global_support_asym.bootstrap_teacher_ramp_fraction = 0.20
    candidates.append(("rank6_corr_global_strong_propagated_teacher_global_support_asym", rank6_corr_global_strong_propagated_teacher_global_support_asym))

    rank6_corr_global_strong_propagated_teacher_global_support_oneway = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.out_subdir = "focused_rank6_corr_global_strong_propagated_teacher_global_support_oneway"
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.global_motion_basis_rank = 6
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.basis_coefficient_bootstrap_weight = 0.12
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.basis_temporal_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.basis_acceleration_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.basis_periodicity_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.residual_basis_projection_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.unsupported_laplacian_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.correspondence_temporal_weight = 0.016
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.correspondence_acceleration_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.correspondence_phase_consistency_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.correspondence_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.correspondence_start_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.correspondence_ramp_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.unsupported_propagation_iterations = 8
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.unsupported_propagation_neighbor_weight = 0.75
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.bootstrap_teacher_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.bootstrap_teacher_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.bootstrap_teacher_support_aware = True
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.bootstrap_teacher_support_power = 2.0
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.bootstrap_teacher_support_floor = 0.10
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.bootstrap_teacher_pred_to_target_weight = 1.0
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.bootstrap_teacher_target_to_pred_weight = 0.0
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.bootstrap_teacher_start_fraction = 0.40
    rank6_corr_global_strong_propagated_teacher_global_support_oneway.bootstrap_teacher_ramp_fraction = 0.20
    candidates.append(("rank6_corr_global_strong_propagated_teacher_global_support_oneway", rank6_corr_global_strong_propagated_teacher_global_support_oneway))

    rank6_corr_global_strong_propagated_teacher_global_late = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_corr_global_strong_propagated_teacher_global_late.out_subdir = "focused_rank6_corr_global_strong_propagated_teacher_global_late"
    rank6_corr_global_strong_propagated_teacher_global_late.global_motion_basis_rank = 6
    rank6_corr_global_strong_propagated_teacher_global_late.basis_coefficient_bootstrap_weight = 0.12
    rank6_corr_global_strong_propagated_teacher_global_late.basis_temporal_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_late.basis_acceleration_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_global_late.basis_periodicity_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_late.residual_basis_projection_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_global_late.unsupported_laplacian_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_global_late.correspondence_temporal_weight = 0.016
    rank6_corr_global_strong_propagated_teacher_global_late.correspondence_acceleration_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_late.correspondence_phase_consistency_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_global_late.correspondence_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_late.correspondence_start_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_late.correspondence_ramp_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_global_late.unsupported_propagation_iterations = 8
    rank6_corr_global_strong_propagated_teacher_global_late.unsupported_propagation_neighbor_weight = 0.75
    rank6_corr_global_strong_propagated_teacher_global_late.bootstrap_teacher_weight = 0.03
    rank6_corr_global_strong_propagated_teacher_global_late.bootstrap_teacher_global_only = True
    rank6_corr_global_strong_propagated_teacher_global_late.bootstrap_teacher_start_fraction = 0.50
    rank6_corr_global_strong_propagated_teacher_global_late.bootstrap_teacher_ramp_fraction = 0.15
    candidates.append(("rank6_corr_global_strong_propagated_teacher_global_late", rank6_corr_global_strong_propagated_teacher_global_late))

    rank6_corr_global_strong_propagated_teacher_late = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_corr_global_strong_propagated_teacher_late.out_subdir = "focused_rank6_corr_global_strong_propagated_teacher_late"
    rank6_corr_global_strong_propagated_teacher_late.global_motion_basis_rank = 6
    rank6_corr_global_strong_propagated_teacher_late.basis_coefficient_bootstrap_weight = 0.12
    rank6_corr_global_strong_propagated_teacher_late.basis_temporal_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_late.basis_acceleration_weight = 0.025
    rank6_corr_global_strong_propagated_teacher_late.basis_periodicity_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_late.residual_basis_projection_weight = 0.04
    rank6_corr_global_strong_propagated_teacher_late.unsupported_laplacian_weight = 0.05
    rank6_corr_global_strong_propagated_teacher_late.correspondence_temporal_weight = 0.016
    rank6_corr_global_strong_propagated_teacher_late.correspondence_acceleration_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_late.correspondence_phase_consistency_weight = 0.008
    rank6_corr_global_strong_propagated_teacher_late.correspondence_global_only = True
    rank6_corr_global_strong_propagated_teacher_late.correspondence_start_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_late.correspondence_ramp_fraction = 0.25
    rank6_corr_global_strong_propagated_teacher_late.unsupported_propagation_iterations = 8
    rank6_corr_global_strong_propagated_teacher_late.unsupported_propagation_neighbor_weight = 0.75
    rank6_corr_global_strong_propagated_teacher_late.bootstrap_teacher_weight = 0.015
    rank6_corr_global_strong_propagated_teacher_late.bootstrap_teacher_start_fraction = 0.50
    rank6_corr_global_strong_propagated_teacher_late.bootstrap_teacher_ramp_fraction = 0.15
    candidates.append(("rank6_corr_global_strong_propagated_teacher_late", rank6_corr_global_strong_propagated_teacher_late))

    rank6_coeff_temporal_corr_propagated = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal_corr_propagated.out_subdir = "focused_rank6_coeff_temporal_corr_propagated"
    rank6_coeff_temporal_corr_propagated.global_motion_basis_rank = 6
    rank6_coeff_temporal_corr_propagated.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_temporal_corr_propagated.basis_temporal_weight = 0.05
    rank6_coeff_temporal_corr_propagated.basis_acceleration_weight = 0.025
    rank6_coeff_temporal_corr_propagated.basis_periodicity_weight = 0.04
    rank6_coeff_temporal_corr_propagated.residual_basis_projection_weight = 0.04
    rank6_coeff_temporal_corr_propagated.unsupported_laplacian_weight = 0.05
    rank6_coeff_temporal_corr_propagated.correspondence_temporal_weight = 0.008
    rank6_coeff_temporal_corr_propagated.correspondence_acceleration_weight = 0.004
    rank6_coeff_temporal_corr_propagated.correspondence_phase_consistency_weight = 0.004
    rank6_coeff_temporal_corr_propagated.correspondence_start_fraction = 0.30
    rank6_coeff_temporal_corr_propagated.correspondence_ramp_fraction = 0.25
    rank6_coeff_temporal_corr_propagated.unsupported_propagation_iterations = 8
    rank6_coeff_temporal_corr_propagated.unsupported_propagation_neighbor_weight = 0.75
    candidates.append(("rank6_coeff_temporal_corr_propagated", rank6_coeff_temporal_corr_propagated))

    rank6_coeff_temporal_ratio_guard = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal_ratio_guard.out_subdir = "focused_rank6_coeff_temporal_ratio_guard"
    rank6_coeff_temporal_ratio_guard.global_motion_basis_rank = 6
    rank6_coeff_temporal_ratio_guard.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_temporal_ratio_guard.basis_temporal_weight = 0.05
    rank6_coeff_temporal_ratio_guard.basis_acceleration_weight = 0.025
    rank6_coeff_temporal_ratio_guard.basis_periodicity_weight = 0.04
    rank6_coeff_temporal_ratio_guard.residual_basis_projection_weight = 0.04
    rank6_coeff_temporal_ratio_guard.unsupported_laplacian_weight = 0.05
    rank6_coeff_temporal_ratio_guard.residual_global_ratio_weight = 0.08
    rank6_coeff_temporal_ratio_guard.residual_global_ratio_target = 0.85
    candidates.append(("rank6_coeff_temporal_ratio_guard", rank6_coeff_temporal_ratio_guard))

    rank6_coeff_temporal_locality_guard = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal_locality_guard.out_subdir = "focused_rank6_coeff_temporal_locality_guard"
    rank6_coeff_temporal_locality_guard.global_motion_basis_rank = 6
    rank6_coeff_temporal_locality_guard.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_temporal_locality_guard.basis_temporal_weight = 0.05
    rank6_coeff_temporal_locality_guard.basis_acceleration_weight = 0.025
    rank6_coeff_temporal_locality_guard.basis_periodicity_weight = 0.04
    rank6_coeff_temporal_locality_guard.residual_basis_projection_weight = 0.04
    rank6_coeff_temporal_locality_guard.unsupported_laplacian_weight = 0.05
    rank6_coeff_temporal_locality_guard.residual_locality_weight = 0.04
    rank6_coeff_temporal_locality_guard.residual_locality_budget_scale = 1.6
    rank6_coeff_temporal_locality_guard.residual_locality_global_budget_scale = 0.12
    candidates.append(("rank6_coeff_temporal_locality_guard", rank6_coeff_temporal_locality_guard))

    rank6_coeff_temporal_locality_tight = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_temporal_locality_tight.out_subdir = "focused_rank6_coeff_temporal_locality_tight"
    rank6_coeff_temporal_locality_tight.global_motion_basis_rank = 6
    rank6_coeff_temporal_locality_tight.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_temporal_locality_tight.basis_temporal_weight = 0.05
    rank6_coeff_temporal_locality_tight.basis_acceleration_weight = 0.025
    rank6_coeff_temporal_locality_tight.basis_periodicity_weight = 0.04
    rank6_coeff_temporal_locality_tight.residual_basis_projection_weight = 0.04
    rank6_coeff_temporal_locality_tight.unsupported_laplacian_weight = 0.05
    rank6_coeff_temporal_locality_tight.residual_locality_weight = 0.08
    rank6_coeff_temporal_locality_tight.residual_locality_budget_scale = 0.9
    rank6_coeff_temporal_locality_tight.residual_locality_global_budget_scale = 0.05
    candidates.append(("rank6_coeff_temporal_locality_tight", rank6_coeff_temporal_locality_tight))

    rank8_residual_split = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank8_residual_split.out_subdir = "focused_rank8_residual_split"
    rank8_residual_split.global_motion_basis_rank = 8
    rank8_residual_split.residual_mean_weight = 0.03
    rank8_residual_split.residual_basis_projection_weight = 0.06
    rank8_residual_split.unsupported_laplacian_weight = 0.06
    candidates.append(("rank8_residual_split", rank8_residual_split))

    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused tuning for shared_topology_global_basis_residual")
    parser.add_argument("--pointcloud-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--train-steps", type=int, default=72)
    parser.add_argument("--mesh-resolution", type=int, default=72)
    parser.add_argument("--max-points-per-phase", type=int, default=3000)
    parser.add_argument("--config-name", type=str, default=None)
    args = parser.parse_args()

    pointclouds = sorted(args.pointcloud_root.glob("*.ply"))
    if not pointclouds:
        raise ValueError(f"No point clouds found in {args.pointcloud_root}")

    phase_summaries, phase_confidences = _load_phase_summaries(args.pointcloud_root)
    candidates = _candidate_configs(args.train_steps, args.mesh_resolution, args.max_points_per_phase)
    if args.config_name is not None:
        candidates = [item for item in candidates if item[0] == args.config_name]
        if not candidates:
            raise ValueError(f"Unknown config name: {args.config_name}")

    gt_available = args.gt_dir is not None and args.gt_dir.exists()
    rows: list[dict[str, float | int | str]] = []
    best_payload: dict[str, object] | None = None
    for name, config in candidates:
        print(f"=== RUN {name} ===")
        reconstruct_dynamic_meshes_from_pointclouds(
            pointclouds,
            config=config,
            phase_confidences=phase_confidences,
            phase_summaries=phase_summaries,
        )
        mesh_dir = args.pointcloud_root / config.out_subdir
        unsupervised_metrics = _evaluate_unsupervised(args.pointcloud_root, mesh_dir)
        gt_metrics = _evaluate_gt(mesh_dir, args.gt_dir) if gt_available and args.gt_dir is not None else None
        score = _selection_score(unsupervised_metrics, gt_metrics)
        row: dict[str, float | int | str] = {
            "name": name,
            "selection_score": score,
            "global_motion_basis_rank": int(config.global_motion_basis_rank),
            "bootstrap_offset_weight": float(config.bootstrap_offset_weight),
            "bootstrap_decay_fraction": float(config.bootstrap_decay_fraction),
            "bootstrap_teacher_weight": float(config.bootstrap_teacher_weight),
            "bootstrap_teacher_global_only": int(bool(config.bootstrap_teacher_global_only)),
            "bootstrap_teacher_support_aware": int(bool(config.bootstrap_teacher_support_aware)),
            "bootstrap_teacher_support_power": float(config.bootstrap_teacher_support_power),
            "bootstrap_teacher_support_floor": float(config.bootstrap_teacher_support_floor),
            "bootstrap_teacher_pred_to_target_weight": float(config.bootstrap_teacher_pred_to_target_weight),
            "bootstrap_teacher_target_to_pred_weight": float(config.bootstrap_teacher_target_to_pred_weight),
            "bootstrap_teacher_start_fraction": float(config.bootstrap_teacher_start_fraction),
            "bootstrap_teacher_ramp_fraction": float(config.bootstrap_teacher_ramp_fraction),
            "basis_coefficient_bootstrap_weight": float(config.basis_coefficient_bootstrap_weight),
            "basis_temporal_weight": float(config.basis_temporal_weight),
            "basis_acceleration_weight": float(config.basis_acceleration_weight),
            "basis_periodicity_weight": float(config.basis_periodicity_weight),
            "correspondence_global_only": int(bool(config.correspondence_global_only)),
            "correspondence_bootstrap_gate": int(bool(config.correspondence_bootstrap_gate)),
            "correspondence_bootstrap_gate_strength": float(config.correspondence_bootstrap_gate_strength),
            "residual_mean_weight": float(config.residual_mean_weight),
            "residual_basis_projection_weight": float(config.residual_basis_projection_weight),
            "residual_locality_weight": float(config.residual_locality_weight),
            "residual_locality_budget_scale": float(config.residual_locality_budget_scale),
            "residual_locality_global_budget_scale": float(config.residual_locality_global_budget_scale),
            "residual_global_ratio_weight": float(config.residual_global_ratio_weight),
            "residual_global_ratio_target": float(config.residual_global_ratio_target),
            "unsupported_anchor_weight": float(config.unsupported_anchor_weight),
            "unsupported_laplacian_weight": float(config.unsupported_laplacian_weight),
            **unsupervised_metrics,
        }
        if gt_metrics is not None:
            row.update(gt_metrics)
        rows.append(row)
        if best_payload is None or score < float(best_payload["selection_score"]):
            best_payload = {
                "name": name,
                "selection_score": score,
                "config": asdict(config),
                "unsupervised_metrics": unsupervised_metrics,
                "gt_metrics": gt_metrics,
            }
        print(json.dumps(row, ensure_ascii=False, indent=2))

    sort_key = ["selection_score", "mean_cd", "mean_fit_cd"] if gt_available else ["selection_score", "mean_fit_cd", "coeff_step_mean"]
    rows.sort(key=lambda item: tuple(float(item.get(key, float("inf"))) for key in sort_key))
    summary_stem = "global_basis_focused_tuning"
    summary_csv = args.pointcloud_root / f"{summary_stem}.csv"
    summary_json = args.pointcloud_root / f"{summary_stem}.json"
    _write_rows(rows, summary_csv, summary_json)
    if best_payload is not None:
        best_path = args.pointcloud_root / f"{summary_stem}_best_config.json"
        best_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE {best_path}")
    print(f"WROTE {summary_csv}")
    print(f"WROTE {summary_json}")


if __name__ == "__main__":
    main()