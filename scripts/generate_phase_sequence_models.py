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
from src.modeling.surface_reconstruction import reconstruct_meshes_from_pointclouds
import scripts.regenerate_freehand_scanner_sequence as regen


REFERENCE_PLY = ROOT / "data" / "test" / "stomach.ply"
OUTPUT_BASE_DIR = ROOT / "data" / "test" / "processed"
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


def _phase_wave(s: np.ndarray, phase: float) -> tuple[np.ndarray, float, np.ndarray]:
    wave_center = 0.40 + 0.48 * phase
    lead = np.exp(-0.5 * ((s - wave_center) / 0.050) ** 2)
    trail = np.exp(-0.5 * ((s - (wave_center - 0.12)) / 0.095) ** 2)
    relax = np.exp(-0.5 * ((s - (wave_center - 0.24)) / 0.12) ** 2)
    distal_gain = 0.55 + 0.80 * np.clip((s - 0.28) / 0.62, 0.0, 1.0)
    proximal_gate = np.clip((s - 0.18) / 0.16, 0.0, 1.0)
    pyloric_gate = 1.0 - 0.28 * np.exp(-0.5 * ((s - 0.94) / 0.05) ** 2)
    contraction = np.clip((0.92 * lead + 0.34 * trail - 0.18 * relax) * distal_gain * proximal_gate * pyloric_gate, 0.0, 0.88)
    return contraction.astype(np.float64), float(wave_center), lead.astype(np.float64)


def _deform_reference_points(reference_points: np.ndarray, canonical_points: np.ndarray, model: regen.GastricReferenceModel, phase: float) -> tuple[np.ndarray, float, float]:
    axis = canonical_points[:, 0]
    s = (axis - axis.min()) / max(axis.max() - axis.min(), 1e-8)
    contraction, wave_center, lead = _phase_wave(s, phase)

    x_new = np.empty_like(axis)
    y_new = np.empty_like(axis)
    z_new = np.empty_like(axis)

    max_contraction = float(np.max(contraction))
    for idx in range(reference_points.shape[0]):
        si = float(s[idx])
        x0, cy0, cz0, ry0, rz0 = regen._interp_profile(model, si)
        centerline = regen.canonical_centerline(model, si, phase)
        dy = canonical_points[idx, 1] - cy0
        dz = canonical_points[idx, 2] - cz0

        ry_safe = max(ry0, 1e-6)
        rz_safe = max(rz0, 1e-6)
        theta = math.atan2(dz / rz_safe, dy / ry_safe)

        ring_scale_y = 1.0 - 0.58 * contraction[idx]
        ring_scale_z = 1.0 - 0.52 * contraction[idx]
        circum_bias = 1.0 - 0.10 * contraction[idx] * math.cos(2.0 * (theta - 0.10))
        distal_pull = 5.6 * contraction[idx] * math.exp(-((si - 0.82) / 0.10) ** 2)
        pyloric_taper = 1.0 - 0.10 * math.exp(-((si - 0.93) / 0.05) ** 2)
        body_sag = 2.1 * contraction[idx] * math.exp(-((si - 0.62) / 0.19) ** 2)
        lumen_shift = 0.10 * contraction[idx] * rz_safe * math.sin(theta - 0.08)
        forward_push = 1.4 * lead[idx] * (0.35 + si)

        x_new[idx] = centerline[0] + distal_pull + forward_push * (0.25 + abs(math.cos(theta)))
        y_new[idx] = centerline[1] + dy * ring_scale_y * circum_bias * pyloric_taper
        z_new[idx] = centerline[2] + dz * ring_scale_z * circum_bias - body_sag + lumen_shift

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
    args = parser.parse_args()

    if not REFERENCE_PLY.exists():
        raise FileNotFoundError(f"Reference stomach point cloud not found: {REFERENCE_PLY}")
    if args.phase_count < 2:
        raise ValueError("phase-count must be at least 2")

    run_dir, run_id = _create_indexed_output_dir(OUTPUT_BASE_DIR, OUTPUT_PREFIX)
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