from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import csv
import math
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig, SurfaceModelConfig
from src.paths import data_path
from src.modeling.surface_reconstruction import reconstruct_meshes_from_pointclouds
import scripts.regenerate_freehand_scanner_sequence as regen


REFERENCE_PLY = data_path("test", "stomach.ply")
OUTPUT_BASE_DIR = data_path("simulation_mesh")
OUTPUT_PREFIX = "phase_sequence_models_run"
OBSERVATION_ROTATION_DEG = np.array([-35.0, 20.0, -32.0], dtype=np.float64)


@dataclass
class PhaseModelStats:
    phase_index: int
    phase_value: float
    timestamp_s: float
    wave_center_s: float
    max_contraction: float
    pointcloud_path: Path


def _rotation_matrix_xyz(angles_deg: np.ndarray) -> np.ndarray:
    ax, ay, az = np.deg2rad(angles_deg.astype(np.float64))

    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(ax), -math.sin(ax)],
        [0.0, math.sin(ax), math.cos(ax)],
    ], dtype=np.float64)
    rot_y = np.array([
        [math.cos(ay), 0.0, math.sin(ay)],
        [0.0, 1.0, 0.0],
        [-math.sin(ay), 0.0, math.cos(ay)],
    ], dtype=np.float64)
    rot_z = np.array([
        [math.cos(az), -math.sin(az), 0.0],
        [math.sin(az), math.cos(az), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def _apply_observation_transform(points: np.ndarray, center: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return center + ((rotation @ (points - center).T).T)


def _write_observation_transform(
    out_dir: Path,
    center: np.ndarray,
    rotation: np.ndarray,
    phase_bin_step_seconds: float,
) -> Path:
    transform_path = out_dir / "observation_transform.npz"
    np.savez_compressed(
        transform_path,
        center=center.astype(np.float64),
        rotation=rotation.astype(np.float64),
        angles_deg=OBSERVATION_ROTATION_DEG.astype(np.float64),
        phase_bin_step_seconds=np.array(phase_bin_step_seconds, dtype=np.float64),
    )
    return transform_path


def _create_indexed_output_dir(base_dir: Path, prefix: str) -> tuple[Path, str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_indices: list[int] = []
    for child in base_dir.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix + "_"):
            continue
        remainder = child.name[len(prefix) + 1 :]
        run_token = remainder.split("_", 1)[0]
        if run_token.isdigit():
            existing_indices.append(int(run_token))
    next_index = max(existing_indices, default=0) + 1
    run_id = f"{next_index:03d}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = base_dir / f"{prefix}_{run_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_id


def _read_reference_points(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = regen._read_ply_points(path)
    center, basis, canonical = regen._pca_basis(points)
    return points, center, canonical


def _phase_timestamp_seconds(phase_index: int, phase_bin_step_seconds: float) -> float:
    return float(phase_index * phase_bin_step_seconds)


def _format_timestamp_token(timestamp_s: float) -> str:
    return f"t{timestamp_s:07.3f}s"


def _smoothstep(edge0: float, edge1: float, value: float | np.ndarray) -> float | np.ndarray:
    span = max(edge1 - edge0, 1e-8)
    t = np.clip((value - edge0) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _phase_wave(
    s: np.ndarray,
    phase: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float]:
    phase = float(phase % 1.0)

    # One cycle: relaxed start, mid-body initiation, propagation toward the pylorus,
    # distal tightening, then a slow return to the relaxed start shape.
    onset = float(_smoothstep(0.03, 0.18, phase))
    travel = float(_smoothstep(0.12, 0.72, phase))
    release = float(1.0 - _smoothstep(0.82, 0.97, phase))
    pyloric_hold = float(_smoothstep(0.60, 0.75, phase) * (1.0 - _smoothstep(0.80, 0.89, phase)))
    cycle_amp = onset * release

    wave_center = 0.46 - 0.34 * travel
    lead = np.exp(-0.5 * ((s - wave_center) / 0.050) ** 2)
    trail = np.exp(-0.5 * ((s - (wave_center + 0.12)) / 0.108) ** 2)
    recovery = np.exp(-0.5 * ((s - (wave_center + 0.24)) / 0.175) ** 2)

    mid_body_seed = np.exp(-0.5 * ((s - 0.46) / 0.080) ** 2)
    pylorus_peak = np.exp(-0.5 * ((s - 0.10) / 0.055) ** 2)
    distal_gain = 0.60 + 0.62 * np.clip((0.68 - s) / 0.60, 0.0, 1.0)
    body_gate = 1.0 - _smoothstep(0.76, 0.94, s)
    pylorus_focus = 0.96 + 0.12 * pylorus_peak

    contraction = (
        cycle_amp * (0.72 * lead + 0.28 * trail - 0.08 * recovery) * distal_gain * body_gate * pylorus_focus
        + 0.09 * onset * mid_body_seed
        + 0.18 * pyloric_hold * pylorus_peak
    )
    contraction = np.clip(contraction, 0.0, 0.84)
    return (
        contraction.astype(np.float64),
        float(wave_center),
        lead.astype(np.float64),
        trail.astype(np.float64),
        recovery.astype(np.float64),
        cycle_amp,
    )


def _deform_reference_points(
    reference_points: np.ndarray,
    canonical_points: np.ndarray,
    model: regen.GastricReferenceModel,
    phase: float,
    observation_rotation: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    axis = canonical_points[:, 0]
    s = (axis - axis.min()) / max(axis.max() - axis.min(), 1e-8)
    contraction, wave_center, lead, trail, recovery, cycle_amp = _phase_wave(s, phase)

    x_new = np.empty_like(axis)
    y_new = np.empty_like(axis)
    z_new = np.empty_like(axis)

    max_contraction = float(np.max(contraction))
    for idx in range(reference_points.shape[0]):
        si = float(s[idx])
        x0, cy0, cz0, ry0, rz0 = regen._interp_profile(model, si)
        dy = canonical_points[idx, 1] - cy0
        dz = canonical_points[idx, 2] - cz0

        ry_safe = max(ry0, 1e-6)
        rz_safe = max(rz0, 1e-6)
        theta = math.atan2(dz / rz_safe, dy / ry_safe)

        ring_scale_y = 1.0 - (0.58 + 0.09 * (1.0 - si)) * contraction[idx]
        ring_scale_z = 1.0 - (0.54 + 0.08 * (1.0 - si)) * contraction[idx]
        circum_bias = 1.0 - 0.07 * contraction[idx] * math.cos(2.0 * (theta - 0.08))
        pyloric_taper = 1.0 - 0.20 * math.exp(-((si - 0.06) / 0.050) ** 2) * (0.35 + 0.65 * cycle_amp)

        axial_shift = -1.5 * cycle_amp * lead[idx] - 3.1 * contraction[idx] * math.exp(-((si - 0.22) / 0.16) ** 2)
        distal_pull = -7.4 * contraction[idx] * math.exp(-((si - 0.11) / 0.09) ** 2)
        body_sag = 2.5 * contraction[idx] * math.exp(-((si - 0.36) / 0.20) ** 2)
        recovery_lift = 0.85 * recovery[idx] * (1.0 - cycle_amp)
        lumen_shift = 0.15 * contraction[idx] * rz_safe * math.sin(theta - 0.06)
        wall_push = (0.45 * lead[idx] + 0.22 * trail[idx]) * cycle_amp * (0.25 + abs(math.cos(theta)))
        center_y_shift = 0.65 * cycle_amp * (lead[idx] - 0.55 * trail[idx])

        center_x = x0 + axial_shift + distal_pull
        center_y = cy0 + center_y_shift
        center_z = cz0 - body_sag + recovery_lift

        x_new[idx] = center_x + wall_push
        y_new[idx] = center_y + dy * ring_scale_y * circum_bias * pyloric_taper
        z_new[idx] = center_z + dz * ring_scale_z * circum_bias - 0.12 * distal_pull + lumen_shift

    canonical_deformed = np.column_stack([x_new, y_new, z_new])
    world_deformed = model.world_center + (model.world_basis @ canonical_deformed.T).T
    observed_points = _apply_observation_transform(world_deformed, model.world_center, observation_rotation)
    return observed_points.astype(np.float32), wave_center, max_contraction


def _write_pointcloud_ply(points: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a gastric peristaltic phase-sequence model set")
    parser.add_argument("--phase-count", type=int, default=41, help="Number of phase models to generate across one cycle")
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default=str(OUTPUT_BASE_DIR),
        help="Directory where the generated phase-sequence model run folder will be created",
    )
    args = parser.parse_args()

    if not REFERENCE_PLY.exists():
        raise FileNotFoundError(f"Reference stomach point cloud not found: {REFERENCE_PLY}")
    if args.phase_count < 2:
        raise ValueError("phase-count must be at least 2")

    phase_bin_step_seconds = float(PipelineConfig().phase_detection.phase_bin_step_seconds)
    if phase_bin_step_seconds <= 0.0:
        raise ValueError("phase_bin_step_seconds must be positive")

    output_base_dir = Path(args.output_base_dir)
    run_dir, run_id = _create_indexed_output_dir(output_base_dir, OUTPUT_PREFIX)
    points_dir = run_dir / "pointclouds"
    points_dir.mkdir(parents=True, exist_ok=True)

    reference_points, _, canonical_points = _read_reference_points(REFERENCE_PLY)
    model = regen.load_reference_model(REFERENCE_PLY)
    observation_rotation = _rotation_matrix_xyz(OBSERVATION_ROTATION_DEG)
    transform_path = _write_observation_transform(run_dir, model.world_center, observation_rotation, phase_bin_step_seconds)

    phase_values = np.linspace(0.0, 1.0, args.phase_count, endpoint=False, dtype=np.float64)

    pointcloud_paths: list[Path] = []
    stats_rows: list[PhaseModelStats] = []
    for phase_index, phase_value in enumerate(phase_values):
        timestamp_s = _phase_timestamp_seconds(phase_index, phase_bin_step_seconds)
        deformed_points, wave_center, max_contraction = _deform_reference_points(
            reference_points,
            canonical_points,
            model,
            float(phase_value),
            observation_rotation,
        )
        pointcloud_path = points_dir / (
            f"run_{run_id}_phase_{phase_index:03d}_{phase_value:.3f}_{_format_timestamp_token(timestamp_s)}.ply"
        )
        _write_pointcloud_ply(deformed_points, pointcloud_path)
        pointcloud_paths.append(pointcloud_path)
        stats_rows.append(
            PhaseModelStats(
                phase_index=phase_index,
                phase_value=float(phase_value),
                timestamp_s=timestamp_s,
                wave_center_s=float(wave_center),
                max_contraction=float(max_contraction),
                pointcloud_path=pointcloud_path,
            )
        )
        print(
            f"[PhaseModels] phase={phase_index:03d} value={phase_value:.3f} ts={timestamp_s:.3f}s "
            f"wave_center_s={wave_center:.3f} max_contraction={max_contraction:.3f} "
            f"-> {pointcloud_path.name}"
        )

    surface_cfg = SurfaceModelConfig()
    surface_cfg.out_subdir = "meshes"
    meshes = reconstruct_meshes_from_pointclouds(
        pointcloud_paths,
        surface_cfg,
        phase_bin_step_seconds=phase_bin_step_seconds,
    )

    summary_path = run_dir / "phase_sequence_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase_index", "phase_value", "timestamp_s", "wave_center_s", "max_contraction", "pointcloud"])
        for row in stats_rows:
            writer.writerow([
                row.phase_index,
                f"{row.phase_value:.6f}",
                f"{row.timestamp_s:.6f}",
                f"{row.wave_center_s:.6f}",
                f"{row.max_contraction:.6f}",
                row.pointcloud_path.name,
            ])

    print(f"[PhaseModels] Output directory: {run_dir}")
    print(f"[PhaseModels] Phase count: {args.phase_count}")
    print(f"[PhaseModels] Phase bin step: {phase_bin_step_seconds:.6f}s")
    print(f"[PhaseModels] Point clouds: {len(pointcloud_paths)}")
    print(f"[PhaseModels] Meshes: {len(meshes)}")
    print(f"[PhaseModels] Summary: {summary_path}")
    print(f"[PhaseModels] Observation transform: {transform_path}")


if __name__ == "__main__":
    main()