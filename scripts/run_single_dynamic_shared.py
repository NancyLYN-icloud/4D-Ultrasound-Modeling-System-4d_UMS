from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import shutil
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_experiments import (
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_GT_MESH_PATH,
    DEFAULT_MONITOR_PATH,
    DEFAULT_SCANNER_PATH,
    _build_config,
    _evaluate_output,
    _load_gt_mesh,
    _make_run_dir,
    _run_pipeline,
    _write_json,
)
from src.config import PointCloudPhaseSummary
from src.modeling.dynamic_surface_reconstruction import reconstruct_dynamic_meshes_from_pointclouds


def _load_phase_cache(pointcloud_root: Path) -> tuple[list[Path], dict[Path, PointCloudPhaseSummary], dict[Path, float]]:
    pointclouds = sorted(path for path in pointcloud_root.glob("*.ply") if path.is_file())
    summary_path = pointcloud_root / "pointcloud_summary.csv"
    if not summary_path.exists():
        return pointclouds, {}, {}

    lines = [line for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) <= 1:
        return pointclouds, {}, {}

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
    return pointclouds, summary_map, confidence_map


def _prepare_replay_pointcloud_root(source_root: Path, target_root: Path) -> Path:
    target_root.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(source_root.glob("*.ply")):
        target_path = target_root / source_path.name
        if target_path.exists():
            continue
        try:
            target_path.symlink_to(source_path)
        except OSError:
            shutil.copy2(source_path, target_path)

    summary_path = source_root / "pointcloud_summary.csv"
    if summary_path.exists():
        target_summary = target_root / summary_path.name
        if not target_summary.exists():
            try:
                target_summary.symlink_to(summary_path)
            except OSError:
                shutil.copy2(summary_path, target_summary)
    return target_root


def _apply_quick_profile(args: argparse.Namespace, config: object) -> None:
    if args.quick_profile == "none":
        return

    if args.quick_profile == "screen":
        train_steps = 1200
        mesh_resolution = 48
        max_points = 1600
        base_steps = 120
    elif args.quick_profile == "trend":
        train_steps = 3600
        mesh_resolution = 56
        max_points = 2200
        base_steps = 180
    else:
        raise ValueError(f"Unsupported quick profile: {args.quick_profile}")

    config.train_steps = min(int(config.train_steps), train_steps)
    config.mesh_resolution = min(int(config.mesh_resolution), mesh_resolution)
    config.max_points_per_phase = min(int(config.max_points_per_phase), max_points)
    current_base_steps = int(config.base_mesh_train_steps) if config.base_mesh_train_steps is not None else base_steps
    config.base_mesh_train_steps = min(current_base_steps, base_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run only the tuned dynamic shared-topology method")
    parser.add_argument("--mode", choices=["fast-dev", "dynamic-detail", "full-paper"], default="dynamic-detail")
    parser.add_argument(
        "--method",
        choices=[
            "动态共享",
            "动态共享-全局基残差",
            "动态共享-参考对应正则",
            "动态共享-连续形变场",
            "动态共享-解耦运动潜码",
            "动态共享-解耦形状运动潜码",
            "动态共享-CPD对应点",
            "动态共享-无先验4D场",
        ],
        default="动态共享-全局基残差",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--monitor-path", type=Path, default=DEFAULT_MONITOR_PATH)
    parser.add_argument("--scanner-path", type=Path, default=DEFAULT_SCANNER_PATH)
    parser.add_argument("--gt-mesh-path", type=Path, default=DEFAULT_GT_MESH_PATH)
    parser.add_argument("--pointcloud-root", type=Path, default=None, help="Reuse exported phase pointclouds and skip acquisition/pipeline stages")
    parser.add_argument("--include-surface-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-name", type=str, default="dynamic_shared_single")
    parser.add_argument("--quick-profile", choices=["none", "screen", "trend"], default="none")
    parser.add_argument("--dynamic-train-steps", type=int, default=None)
    parser.add_argument("--dynamic-mesh-resolution", type=int, default=None)
    parser.add_argument("--max-points-per-phase", type=int, default=None)
    parser.add_argument("--canonical-hidden-dim", type=int, default=None)
    parser.add_argument("--canonical-hidden-layers", type=int, default=None)
    parser.add_argument("--deformation-hidden-dim", type=int, default=None)
    parser.add_argument("--deformation-hidden-layers", type=int, default=None)
    parser.add_argument("--confidence-floor", type=float, default=None)
    parser.add_argument("--temporal-weight", type=float, default=None)
    parser.add_argument("--temporal-acceleration-weight", type=float, default=None)
    parser.add_argument("--phase-consistency-weight", type=float, default=None)
    parser.add_argument("--periodicity-weight", type=float, default=None)
    parser.add_argument("--deformation-weight", type=float, default=None)
    parser.add_argument("--correspondence-temporal-weight", type=float, default=None)
    parser.add_argument("--correspondence-acceleration-weight", type=float, default=None)
    parser.add_argument("--correspondence-phase-consistency-weight", type=float, default=None)
    parser.add_argument("--correspondence-global-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--correspondence-bootstrap-gate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--correspondence-bootstrap-gate-strength", type=float, default=None)
    parser.add_argument("--correspondence-start-fraction", type=float, default=None)
    parser.add_argument("--correspondence-ramp-fraction", type=float, default=None)
    parser.add_argument("--bootstrap-offset-weight", type=float, default=None)
    parser.add_argument("--bootstrap-decay-fraction", type=float, default=None)
    parser.add_argument("--bootstrap-teacher-weight", type=float, default=None)
    parser.add_argument("--bootstrap-teacher-global-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bootstrap-teacher-support-aware", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bootstrap-teacher-support-power", type=float, default=None)
    parser.add_argument("--bootstrap-teacher-support-floor", type=float, default=None)
    parser.add_argument("--bootstrap-teacher-pred-to-target-weight", type=float, default=None)
    parser.add_argument("--bootstrap-teacher-target-to-pred-weight", type=float, default=None)
    parser.add_argument("--bootstrap-teacher-start-fraction", type=float, default=None)
    parser.add_argument("--bootstrap-teacher-ramp-fraction", type=float, default=None)
    parser.add_argument("--basis-coefficient-bootstrap-weight", type=float, default=None)
    parser.add_argument("--basis-temporal-weight", type=float, default=None)
    parser.add_argument("--basis-acceleration-weight", type=float, default=None)
    parser.add_argument("--basis-periodicity-weight", type=float, default=None)
    parser.add_argument("--residual-mean-weight", type=float, default=None)
    parser.add_argument("--residual-basis-projection-weight", type=float, default=None)
    parser.add_argument("--residual-basis-projection-support-aware", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--residual-locality-weight", type=float, default=None)
    parser.add_argument("--residual-locality-budget-scale", type=float, default=None)
    parser.add_argument("--residual-locality-global-budget-scale", type=float, default=None)
    parser.add_argument("--residual-wave-band-concentration-weight", type=float, default=None)
    parser.add_argument("--residual-wave-band-target-std", type=float, default=None)
    parser.add_argument("--residual-wave-direction-weight", type=float, default=None)
    parser.add_argument("--residual-wave-direction-band-width", type=float, default=None)
    parser.add_argument("--residual-wave-direction-tangential-weight", type=float, default=None)
    parser.add_argument("--wave-band-data-term-boost-weight", type=float, default=None)
    parser.add_argument("--wave-band-data-term-band-width", type=float, default=None)
    parser.add_argument("--wave-band-data-term-support-power", type=float, default=None)
    parser.add_argument("--residual-global-ratio-weight", type=float, default=None)
    parser.add_argument("--residual-global-ratio-target", type=float, default=None)
    parser.add_argument("--residual-global-ratio-support-aware", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--unsupported-anchor-weight", type=float, default=None)
    parser.add_argument("--unsupported-laplacian-weight", type=float, default=None)
    parser.add_argument("--unsupported-propagation-iterations", type=int, default=None)
    parser.add_argument("--unsupported-propagation-neighbor-weight", type=float, default=None)
    parser.add_argument("--base-mesh-train-steps", type=int, default=None)
    parser.add_argument("--smoothing-iterations", type=int, default=None)
    parser.add_argument("--global-motion-basis-rank", type=int, default=None)
    parser.add_argument("--shape-offset-reg-weight", type=float, default=None)
    parser.add_argument("--shape-spatial-weight", type=float, default=None)
    parser.add_argument("--shape-code-reg-weight", type=float, default=None)
    parser.add_argument("--motion-mean-weight", type=float, default=None)
    parser.add_argument("--motion-lipschitz-weight", type=float, default=None)
    args = parser.parse_args()

    run_dir = _make_run_dir(args.out_dir, args.mode, "single-dynamic-shared", args.run_name)
    artifact_dir = run_dir / "artifacts"
    config_dir = run_dir / "configs"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    pointcloud_out_dir = artifact_dir / "phase_pointclouds_run_001_single"
    pointcloud_out_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config(
        method=args.method,
        mode=args.mode,
        dynamic_train_steps=args.dynamic_train_steps,
        dynamic_mesh_resolution=args.dynamic_mesh_resolution,
        pointcloud_out_dir=pointcloud_out_dir,
    )
    config.surface_model.enabled = bool(args.include_surface_model)

    overrides = {
        "max_points_per_phase": args.max_points_per_phase,
        "canonical_hidden_dim": args.canonical_hidden_dim,
        "canonical_hidden_layers": args.canonical_hidden_layers,
        "deformation_hidden_dim": args.deformation_hidden_dim,
        "deformation_hidden_layers": args.deformation_hidden_layers,
        "confidence_floor": args.confidence_floor,
        "temporal_weight": args.temporal_weight,
        "temporal_acceleration_weight": args.temporal_acceleration_weight,
        "phase_consistency_weight": args.phase_consistency_weight,
        "periodicity_weight": args.periodicity_weight,
        "deformation_weight": args.deformation_weight,
        "correspondence_temporal_weight": args.correspondence_temporal_weight,
        "correspondence_acceleration_weight": args.correspondence_acceleration_weight,
        "correspondence_phase_consistency_weight": args.correspondence_phase_consistency_weight,
        "correspondence_global_only": args.correspondence_global_only,
        "correspondence_bootstrap_gate": args.correspondence_bootstrap_gate,
        "correspondence_bootstrap_gate_strength": args.correspondence_bootstrap_gate_strength,
        "correspondence_start_fraction": args.correspondence_start_fraction,
        "correspondence_ramp_fraction": args.correspondence_ramp_fraction,
        "bootstrap_offset_weight": args.bootstrap_offset_weight,
        "bootstrap_decay_fraction": args.bootstrap_decay_fraction,
        "bootstrap_teacher_weight": args.bootstrap_teacher_weight,
        "bootstrap_teacher_global_only": args.bootstrap_teacher_global_only,
        "bootstrap_teacher_support_aware": args.bootstrap_teacher_support_aware,
        "bootstrap_teacher_support_power": args.bootstrap_teacher_support_power,
        "bootstrap_teacher_support_floor": args.bootstrap_teacher_support_floor,
        "bootstrap_teacher_pred_to_target_weight": args.bootstrap_teacher_pred_to_target_weight,
        "bootstrap_teacher_target_to_pred_weight": args.bootstrap_teacher_target_to_pred_weight,
        "bootstrap_teacher_start_fraction": args.bootstrap_teacher_start_fraction,
        "bootstrap_teacher_ramp_fraction": args.bootstrap_teacher_ramp_fraction,
        "basis_coefficient_bootstrap_weight": args.basis_coefficient_bootstrap_weight,
        "basis_temporal_weight": args.basis_temporal_weight,
        "basis_acceleration_weight": args.basis_acceleration_weight,
        "basis_periodicity_weight": args.basis_periodicity_weight,
        "residual_mean_weight": args.residual_mean_weight,
        "residual_basis_projection_weight": args.residual_basis_projection_weight,
        "residual_basis_projection_support_aware": args.residual_basis_projection_support_aware,
        "residual_locality_weight": args.residual_locality_weight,
        "residual_locality_budget_scale": args.residual_locality_budget_scale,
        "residual_locality_global_budget_scale": args.residual_locality_global_budget_scale,
        "residual_wave_band_concentration_weight": args.residual_wave_band_concentration_weight,
        "residual_wave_band_target_std": args.residual_wave_band_target_std,
        "residual_wave_direction_weight": args.residual_wave_direction_weight,
        "residual_wave_direction_band_width": args.residual_wave_direction_band_width,
        "residual_wave_direction_tangential_weight": args.residual_wave_direction_tangential_weight,
        "wave_band_data_term_boost_weight": args.wave_band_data_term_boost_weight,
        "wave_band_data_term_band_width": args.wave_band_data_term_band_width,
        "wave_band_data_term_support_power": args.wave_band_data_term_support_power,
        "residual_global_ratio_weight": args.residual_global_ratio_weight,
        "residual_global_ratio_target": args.residual_global_ratio_target,
        "residual_global_ratio_support_aware": args.residual_global_ratio_support_aware,
        "unsupported_anchor_weight": args.unsupported_anchor_weight,
        "unsupported_laplacian_weight": args.unsupported_laplacian_weight,
        "unsupported_propagation_iterations": args.unsupported_propagation_iterations,
        "unsupported_propagation_neighbor_weight": args.unsupported_propagation_neighbor_weight,
        "base_mesh_train_steps": args.base_mesh_train_steps,
        "smoothing_iterations": args.smoothing_iterations,
        "global_motion_basis_rank": args.global_motion_basis_rank,
        "shape_offset_reg_weight": args.shape_offset_reg_weight,
        "shape_spatial_weight": args.shape_spatial_weight,
        "shape_code_reg_weight": args.shape_code_reg_weight,
        "motion_mean_weight": args.motion_mean_weight,
        "motion_lipschitz_weight": args.motion_lipschitz_weight,
    }
    for field_name, value in overrides.items():
        if value is not None:
            setattr(config.dynamic_model, field_name, value)

    _apply_quick_profile(args, config.dynamic_model)

    _write_json(config_dir / f"{args.method}.json", asdict(config))

    gt_mesh = _load_gt_mesh(args.gt_mesh_path)
    if args.pointcloud_root is None:
        output = _run_pipeline(
            method=args.method,
            config=config,
            monitor_path=args.monitor_path,
            scanner_path=args.scanner_path,
        )
    else:
        replay_root = _prepare_replay_pointcloud_root(args.pointcloud_root, artifact_dir / "replay_pointclouds")
        pointclouds, phase_summaries, phase_confidences = _load_phase_cache(replay_root)
        if not pointclouds:
            raise FileNotFoundError(f"No pointclouds found under {args.pointcloud_root}")
        dynamic_mesh_results, dynamic_timeline_mesh_results = reconstruct_dynamic_meshes_from_pointclouds(
            pointclouds,
            config=config.dynamic_model,
            phase_confidences=phase_confidences,
            phase_summaries=phase_summaries,
        )
        class _ReplayOutput:
            def __init__(self) -> None:
                self.pointcloud_paths = pointclouds
                self.pointcloud_summaries = list(phase_summaries.values())
                self.mesh_results = []
                self.dynamic_mesh_results = dynamic_mesh_results
                self.dynamic_timeline_mesh_results = dynamic_timeline_mesh_results

        output = _ReplayOutput()
    result = _evaluate_output(args.method, output, gt_mesh)
    result["quick_profile"] = args.quick_profile
    result["reused_pointcloud_cache"] = bool(args.pointcloud_root is not None)
    result_path = run_dir / "dynamic_shared_result.json"
    _write_json(result_path, result)

    csv_path = run_dir / "dynamic_shared_result.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    print(json.dumps({"run_dir": str(run_dir), "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()