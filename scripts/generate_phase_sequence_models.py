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

from src.config import SurfaceModelConfig
from src.paths import data_path
from src.modeling.surface_reconstruction import reconstruct_meshes_from_pointclouds
import scripts.regenerate_freehand_scanner_sequence as regen


REFERENCE_PLY = data_path("test", "stomach.ply")
OUTPUT_BASE_DIR = data_path("test", "simulation_mesh")
OUTPUT_PREFIX = "phase_sequence_models_run"


@dataclass
class PhaseModelStats:
    phase_index: int
    phase_value: float
    wave_center_s: float
    max_contraction: float
    pointcloud_path: Path


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


def _smoothstep(edge0: float, edge1: float, value: float | np.ndarray) -> float | np.ndarray:
    span = max(edge1 - edge0, 1e-8)
    t = np.clip((value - edge0) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _phase_wave(
    s: np.ndarray,
    phase: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float]:
    phase = float(phase % 1.0)

    # One cycle: a contraction initiates in the gastric body, travels toward the pylorus,
    # peaks distally, then fades back to the relaxed start shape.
    travel = float(_smoothstep(0.02, 0.78, phase))
    wave_center = 0.42 + 0.48 * travel
    lead = np.exp(-0.5 * ((s - wave_center) / 0.052) ** 2)
    trail = np.exp(-0.5 * ((s - (wave_center - 0.12)) / 0.105) ** 2)
    recovery = np.exp(-0.5 * ((s - (wave_center - 0.25)) / 0.16) ** 2)

    cycle_amp = float(np.sin(np.pi * phase) ** 1.15)
    early_seed = np.exp(-0.5 * ((s - 0.44) / 0.085) ** 2)
    distal_gain = 0.45 + 0.92 * np.clip((s - 0.30) / 0.60, 0.0, 1.0)
    body_gate = _smoothstep(0.18, 0.34, s)
    pylorus_focus = 0.88 + 0.18 * np.exp(-0.5 * ((s - 0.90) / 0.07) ** 2)

    contraction = (
        cycle_amp * (0.86 * lead + 0.34 * trail - 0.22 * recovery) * distal_gain * body_gate * pylorus_focus
        + 0.09 * cycle_amp * early_seed
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


def _deform_reference_points(reference_points: np.ndarray, canonical_points: np.ndarray, model: regen.GastricReferenceModel, phase: float) -> tuple[np.ndarray, float, float]:
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

        ring_scale_y = 1.0 - (0.50 + 0.07 * si) * contraction[idx]
        ring_scale_z = 1.0 - (0.46 + 0.05 * si) * contraction[idx]
        circum_bias = 1.0 - 0.12 * contraction[idx] * math.cos(2.0 * (theta - 0.08))
        pyloric_taper = 1.0 - 0.14 * math.exp(-((si - 0.95) / 0.045) ** 2) * (0.3 + 0.7 * cycle_amp)

        axial_shift = 1.2 * cycle_amp * lead[idx] + 2.4 * contraction[idx] * math.exp(-((si - 0.79) / 0.15) ** 2)
        distal_pull = 6.4 * contraction[idx] * math.exp(-((si - 0.87) / 0.08) ** 2)
        body_sag = 2.8 * contraction[idx] * math.exp(-((si - 0.63) / 0.19) ** 2)
        recovery_lift = 0.9 * recovery[idx] * (1.0 - cycle_amp)
        lumen_shift = 0.11 * contraction[idx] * rz_safe * math.sin(theta - 0.06)
        wall_push = (0.45 * lead[idx] + 0.22 * trail[idx]) * cycle_amp * (0.25 + abs(math.cos(theta)))
        center_y_shift = 0.65 * cycle_amp * (lead[idx] - 0.55 * trail[idx])

        center_x = x0 + axial_shift + distal_pull
        center_y = cy0 + center_y_shift
        center_z = cz0 - body_sag + recovery_lift

        x_new[idx] = center_x + wall_push
        y_new[idx] = center_y + dy * ring_scale_y * circum_bias * pyloric_taper
        z_new[idx] = center_z + dz * ring_scale_z * circum_bias - 0.18 * distal_pull + lumen_shift

    canonical_deformed = np.column_stack([x_new, y_new, z_new])
    world_deformed = model.world_center + (model.world_basis @ canonical_deformed.T).T
    return world_deformed.astype(np.float32), wave_center, max_contraction


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

    output_base_dir = Path(args.output_base_dir)
    run_dir, run_id = _create_indexed_output_dir(output_base_dir, OUTPUT_PREFIX)
    points_dir = run_dir / "pointclouds"
    points_dir.mkdir(parents=True, exist_ok=True)

    reference_points, _, canonical_points = _read_reference_points(REFERENCE_PLY)
    model = regen.load_reference_model(REFERENCE_PLY)

    phase_edges = np.linspace(0.0, 1.0, args.phase_count + 1, dtype=np.float64)
    phase_centers = 0.5 * (phase_edges[:-1] + phase_edges[1:])

    pointcloud_paths: list[Path] = []
    stats_rows: list[PhaseModelStats] = []
    for phase_index, phase_value in enumerate(phase_centers):
        deformed_points, wave_center, max_contraction = _deform_reference_points(
            reference_points,
            canonical_points,
            model,
            float(phase_value),
        )
        pointcloud_path = points_dir / f"run_{run_id}_phase_{phase_index:03d}_{phase_value:.3f}.ply"
        _write_pointcloud_ply(deformed_points, pointcloud_path)
        pointcloud_paths.append(pointcloud_path)
        stats_rows.append(
            PhaseModelStats(
                phase_index=phase_index,
                phase_value=float(phase_value),
                wave_center_s=float(wave_center),
                max_contraction=float(max_contraction),
                pointcloud_path=pointcloud_path,
            )
        )
        print(
            f"[PhaseModels] phase={phase_index:03d} value={phase_value:.3f} "
            f"wave_center_s={wave_center:.3f} max_contraction={max_contraction:.3f} "
            f"-> {pointcloud_path.name}"
        )

    surface_cfg = SurfaceModelConfig()
    surface_cfg.out_subdir = "meshes"
    meshes = reconstruct_meshes_from_pointclouds(pointcloud_paths, surface_cfg)

    summary_path = run_dir / "phase_sequence_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase_index", "phase_value", "wave_center_s", "max_contraction", "pointcloud"])
        for row in stats_rows:
            writer.writerow([
                row.phase_index,
                f"{row.phase_value:.6f}",
                f"{row.wave_center_s:.6f}",
                f"{row.max_contraction:.6f}",
                row.pointcloud_path.name,
            ])

    print(f"[PhaseModels] Output directory: {run_dir}")
    print(f"[PhaseModels] Phase count: {args.phase_count}")
    print(f"[PhaseModels] Point clouds: {len(pointcloud_paths)}")
    print(f"[PhaseModels] Meshes: {len(meshes)}")
    print(f"[PhaseModels] Summary: {summary_path}")


if __name__ == "__main__":
    main()