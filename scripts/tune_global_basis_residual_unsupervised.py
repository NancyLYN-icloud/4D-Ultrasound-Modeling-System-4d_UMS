from __future__ import annotations

import argparse
import csv
import json
import re
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
        mesh = trimesh.load(path, force="mesh")
        phase_index, phase = _parse_phase(path)
        items.append({"path": path, "mesh": mesh, "phase_index": phase_index, "phase": phase})
    return items


def _load_pointcloud_sequence(folder: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for path in sorted(folder.glob("*.ply")):
        if path.name.startswith("dynamic_"):
            continue
        pointcloud = trimesh.load(path, force="mesh")
        vertices = np.asarray(getattr(pointcloud, "vertices", np.empty((0, 3))), dtype=np.float64)
        phase_index, phase = _parse_phase(path)
        items.append({"path": path, "points": vertices, "phase_index": phase_index, "phase": phase})
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


def _match_pointcloud(pointcloud_items: list[dict[str, object]], mesh_item: dict[str, object]) -> dict[str, object]:
    phase_index = mesh_item["phase_index"]
    if phase_index is not None:
        for item in pointcloud_items:
            if item["phase_index"] == phase_index:
                return item
    phase = mesh_item["phase"]
    if phase is None:
        return pointcloud_items[0]
    candidates = [item for item in pointcloud_items if item["phase"] is not None]
    if not candidates:
        return pointcloud_items[0]
    return min(candidates, key=lambda item: abs(float(item["phase"]) - float(phase)))


def _load_basis_diagnostics(mesh_dir: Path) -> dict[str, float]:
    path = mesh_dir / "global_basis_diagnostics.csv"
    if not path.exists():
        return {
            "coeff_step_mean": float("inf"),
            "coeff_accel_mean": float("inf"),
            "coeff_periodic_gap": float("inf"),
            "residual_global_ratio_mean": float("inf"),
            "residual_global_ratio_max": float("inf"),
        }
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return {
            "coeff_step_mean": float("inf"),
            "coeff_accel_mean": float("inf"),
            "coeff_periodic_gap": float("inf"),
            "residual_global_ratio_mean": float("inf"),
            "residual_global_ratio_max": float("inf"),
        }
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
            "coeff_step_mean": float("inf"),
            "coeff_accel_mean": float("inf"),
            "coeff_periodic_gap": float("inf"),
            "residual_global_ratio_mean": float("inf"),
            "residual_global_ratio_max": float("inf"),
        }

    chamfers: list[float] = []
    hausdorffs: list[float] = []
    meshes: list[trimesh.Trimesh] = []
    for index, mesh_item in enumerate(mesh_items):
        pointcloud_item = _match_pointcloud(pointcloud_items, mesh_item)
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


def _overall_score(metrics: dict[str, float | int]) -> float:
    return float(
        1.0 * float(metrics["mean_fit_cd"])
        + 0.20 * float(metrics["mean_fit_hd95"])
        + 0.15 * float(metrics["temporal_smoothness"])
        + 0.25 * float(metrics["coeff_step_mean"])
        + 0.20 * float(metrics["coeff_accel_mean"])
        + 0.15 * float(metrics["coeff_periodic_gap"])
        + 0.20 * float(metrics["residual_global_ratio_mean"])
        + 0.10 * float(metrics["residual_global_ratio_max"])
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
        basis_coefficient_bootstrap_weight=0.08,
        basis_temporal_weight=0.03,
        basis_acceleration_weight=0.015,
        basis_periodicity_weight=0.03,
        residual_mean_weight=0.02,
        residual_basis_projection_weight=0.03,
        mesh_resolution=mesh_resolution,
        smoothing_iterations=8,
    )


def _candidate_configs(train_steps: int, mesh_resolution: int, max_points_per_phase: int) -> list[tuple[str, DynamicModelConfig]]:
    candidates: list[tuple[str, DynamicModelConfig]] = []

    rank6_balanced = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_balanced.out_subdir = "dynamic_meshes_rank6_balanced"
    rank6_balanced.global_motion_basis_rank = 6
    candidates.append(("rank6_balanced", rank6_balanced))

    rank6_coeff_strong = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_coeff_strong.out_subdir = "dynamic_meshes_rank6_coeff_strong"
    rank6_coeff_strong.global_motion_basis_rank = 6
    rank6_coeff_strong.basis_coefficient_bootstrap_weight = 0.12
    rank6_coeff_strong.basis_temporal_weight = 0.04
    rank6_coeff_strong.basis_acceleration_weight = 0.02
    rank6_coeff_strong.basis_periodicity_weight = 0.04
    candidates.append(("rank6_coeff_strong", rank6_coeff_strong))

    rank6_residual_guarded = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank6_residual_guarded.out_subdir = "dynamic_meshes_rank6_residual_guarded"
    rank6_residual_guarded.global_motion_basis_rank = 6
    rank6_residual_guarded.residual_basis_projection_weight = 0.05
    rank6_residual_guarded.residual_mean_weight = 0.03
    rank6_residual_guarded.unsupported_laplacian_weight = 0.06
    candidates.append(("rank6_residual_guarded", rank6_residual_guarded))

    rank8_balanced = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank8_balanced.out_subdir = "dynamic_meshes_rank8_balanced"
    rank8_balanced.global_motion_basis_rank = 8
    candidates.append(("rank8_balanced", rank8_balanced))

    rank8_coeff_guarded = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank8_coeff_guarded.out_subdir = "dynamic_meshes_rank8_coeff_guarded"
    rank8_coeff_guarded.global_motion_basis_rank = 8
    rank8_coeff_guarded.basis_coefficient_bootstrap_weight = 0.14
    rank8_coeff_guarded.basis_temporal_weight = 0.05
    rank8_coeff_guarded.basis_acceleration_weight = 0.025
    rank8_coeff_guarded.basis_periodicity_weight = 0.05
    rank8_coeff_guarded.residual_basis_projection_weight = 0.05
    candidates.append(("rank8_coeff_guarded", rank8_coeff_guarded))

    rank4_flexible = _base_config(train_steps, mesh_resolution, max_points_per_phase)
    rank4_flexible.out_subdir = "dynamic_meshes_rank4_flexible"
    rank4_flexible.global_motion_basis_rank = 4
    rank4_flexible.bootstrap_offset_weight = 0.08
    rank4_flexible.bootstrap_decay_fraction = 0.20
    rank4_flexible.basis_coefficient_bootstrap_weight = 0.06
    rank4_flexible.residual_basis_projection_weight = 0.02
    candidates.append(("rank4_flexible", rank4_flexible))

    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsupervised tuning for shared_topology_global_basis_residual")
    parser.add_argument("--pointcloud-root", type=Path, required=True)
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
        metrics = _evaluate_unsupervised(args.pointcloud_root, args.pointcloud_root / config.out_subdir)
        score = _overall_score(metrics)
        row: dict[str, float | int | str] = {
            "name": name,
            "overall_score": score,
            "global_motion_basis_rank": int(config.global_motion_basis_rank),
            "bootstrap_offset_weight": float(config.bootstrap_offset_weight),
            "bootstrap_decay_fraction": float(config.bootstrap_decay_fraction),
            "basis_coefficient_bootstrap_weight": float(config.basis_coefficient_bootstrap_weight),
            "basis_temporal_weight": float(config.basis_temporal_weight),
            "basis_acceleration_weight": float(config.basis_acceleration_weight),
            "basis_periodicity_weight": float(config.basis_periodicity_weight),
            "residual_mean_weight": float(config.residual_mean_weight),
            "residual_basis_projection_weight": float(config.residual_basis_projection_weight),
            "unsupported_anchor_weight": float(config.unsupported_anchor_weight),
            "unsupported_laplacian_weight": float(config.unsupported_laplacian_weight),
            **metrics,
        }
        rows.append(row)
        if best_payload is None or score < float(best_payload["overall_score"]):
            best_payload = {
                "name": name,
                "overall_score": score,
                "config": config.__dict__,
                "metrics": metrics,
            }
        print(json.dumps(row, ensure_ascii=False, indent=2))

    rows.sort(key=lambda item: (float(item["overall_score"]), float(item["mean_fit_cd"]), float(item["coeff_step_mean"])))
    summary_csv = args.pointcloud_root / "global_basis_unsupervised_tuning_summary.csv"
    summary_json = args.pointcloud_root / "global_basis_unsupervised_tuning_summary.json"
    _write_rows(rows, summary_csv, summary_json)
    if best_payload is not None:
        best_path = args.pointcloud_root / "global_basis_unsupervised_best_config.json"
        best_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE {best_path}")
    print(f"WROTE {summary_csv}")
    print(f"WROTE {summary_json}")


if __name__ == "__main__":
    main()