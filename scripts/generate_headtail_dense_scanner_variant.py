from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import trimesh


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_scanner_from_phase_models import _slice_mesh_frame
import scripts.regenerate_freehand_scanner_sequence as regen
from scripts.stomach_peristaltic_axis import axis_u_to_scanner_s, build_peristaltic_axis_model, build_scanner_s_lookup, interpolate_centerline_position, interpolate_centerline_tangent
from src.stomach_instance_paths import resolve_instance_paths, resolve_monitor_input_path, resolve_scanner_template_path


FRAME_SIZE = 512
PHASE_MODEL_PREFIX = "phase_sequence_models_run_"


def _latest_phase_model_dir(base_dir: Path) -> Path:
    candidates = sorted(path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith(PHASE_MODEL_PREFIX))
    if not candidates:
        raise FileNotFoundError(f"No {PHASE_MODEL_PREFIX}* directory found under {base_dir}")
    return candidates[-1]


def _load_observation_transform(phase_model_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    transform_path = phase_model_dir / "observation_transform.npz"
    if not transform_path.exists():
        return None
    with np.load(transform_path) as data:
        center = data["center"].astype(np.float64)
        rotation = data["rotation"].astype(np.float64)
    return center, rotation


def _apply_pose_transform(
    position: np.ndarray,
    orientation: np.ndarray,
    transform: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if transform is None:
        return position, orientation
    center, rotation = transform
    position_out = center + rotation @ (position - center)
    orientation_out = rotation @ orientation
    return position_out.astype(np.float64), orientation_out.astype(np.float64)


def _read_phase_summary(summary_path: Path) -> tuple[np.ndarray, list[Path]]:
    phase_values: list[float] = []
    mesh_paths: list[Path] = []
    mesh_dir = summary_path.parent / "pointclouds" / "meshes"
    with summary_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            phase_values.append(float(row["phase_value"]))
            mesh_paths.append(mesh_dir / row["pointcloud"].replace(".ply", "_mesh.ply"))
    if not phase_values:
        raise RuntimeError(f"No phase rows found in {summary_path}")
    return np.asarray(phase_values, dtype=np.float64), mesh_paths


def _load_meshes(mesh_paths: list[Path]) -> list[trimesh.Trimesh]:
    meshes: list[trimesh.Trimesh] = []
    for mesh_path in mesh_paths:
        mesh = trimesh.load(mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Failed to load mesh: {mesh_path}")
        meshes.append(mesh)
    return meshes


def _nearest_phase_index(phase_values: np.ndarray, phase: float) -> int:
    wrapped = np.minimum(np.abs(phase_values - phase), 1.0 - np.abs(phase_values - phase))
    return int(np.argmin(wrapped))


def _headtail_dense_axis_u(timestamp: float, gastric_period: float) -> float:
    base = timestamp / 7.6
    base += regen.GOLDEN_OFFSET * (timestamp / gastric_period)
    base += 0.09 * np.sin(2.0 * np.pi * timestamp / 31.0 + 0.4)
    base += 0.04 * np.sin(2.0 * np.pi * timestamp / 13.0 - 0.2)
    axis_u = float(np.clip(regen.triangle_wave(np.array([base], dtype=np.float64))[0], 0.0, 1.0))

    if axis_u < 0.12:
        axis_u *= 0.62

    return float(np.clip(axis_u, 0.0, 1.0))


def _normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    return arr / (np.linalg.norm(arr) + 1e-8)


def _headtail_dense_probe_orientation(
    prior_orientation: np.ndarray,
    axis_tangent: np.ndarray,
    axis_u: float,
    phase: float,
    timestamp: float,
) -> np.ndarray:
    prior = np.asarray(prior_orientation, dtype=np.float64)
    prior_x = _normalize(prior[:, 0])
    prior_y = _normalize(prior[:, 1])
    prior_normal = _normalize(prior[:, 2])
    target_normal = _normalize(axis_tangent)

    contraction, _, _ = regen.peristaltic_state(axis_u, phase)
    end_focus = np.exp(-((axis_u - 0.08) / 0.10) ** 2) + np.exp(-((axis_u - 0.92) / 0.10) ** 2)
    end_focus = float(np.clip(end_focus, 0.0, 1.0))

    # Keep the original freehand pose as the main signal and only bias the
    # normal toward the anatomical tangent near the ends.
    tangent_blend = 0.10 + 0.22 * end_focus + 0.08 * contraction
    tangent_blend = float(np.clip(tangent_blend, 0.08, 0.38))
    blended_normal = _normalize((1.0 - tangent_blend) * prior_normal + tangent_blend * target_normal)

    axis_x = prior_x - float(np.dot(prior_x, blended_normal)) * blended_normal
    if np.linalg.norm(axis_x) < 1e-6:
        axis_x = prior_y - float(np.dot(prior_y, blended_normal)) * blended_normal
    if np.linalg.norm(axis_x) < 1e-6:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(fallback, blended_normal))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis_x = fallback - float(np.dot(fallback, blended_normal)) * blended_normal
    axis_x = _normalize(axis_x)
    axis_y = _normalize(np.cross(blended_normal, axis_x))
    axis_x = _normalize(np.cross(axis_y, blended_normal))

    base_orientation = np.column_stack([axis_x, axis_y, blended_normal])

    extra_roll = np.deg2rad(end_focus * 2.8 * np.sin(2.0 * np.pi * timestamp / 19.0 + 0.3))
    extra_yaw = np.deg2rad(end_focus * 3.2 * np.cos(2.0 * np.pi * timestamp / 23.0 - 0.4))
    extra_pitch = np.deg2rad((1.1 + 1.6 * contraction) * np.sin(2.0 * np.pi * timestamp / 17.0 - 0.15))

    rot_roll = np.array([
        [np.cos(extra_roll), -np.sin(extra_roll), 0.0],
        [np.sin(extra_roll), np.cos(extra_roll), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    rot_pitch = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(extra_pitch), -np.sin(extra_pitch)],
        [0.0, np.sin(extra_pitch), np.cos(extra_pitch)],
    ], dtype=np.float64)
    rot_yaw = np.array([
        [np.cos(extra_yaw), 0.0, np.sin(extra_yaw)],
        [0.0, 1.0, 0.0],
        [-np.sin(extra_yaw), 0.0, np.cos(extra_yaw)],
    ], dtype=np.float64)
    return base_orientation @ rot_roll @ rot_pitch @ rot_yaw


def generate_variant(
    instance_name: str,
    phase_model_dir: Path,
    output_root: Path,
    monitor_path: Path | None,
    scanner_template_path: Path | None,
    rewrite_pngs: bool,
) -> None:
    instance_paths = resolve_instance_paths(instance_name=instance_name)
    summary_path = phase_model_dir / "phase_sequence_summary.csv"
    phase_values, mesh_paths = _read_phase_summary(summary_path)
    meshes = _load_meshes(mesh_paths)

    resolved_monitor_path = resolve_monitor_input_path(instance_paths, explicit_path=monitor_path)
    resolved_template_path = resolve_scanner_template_path(instance_paths, explicit_path=scanner_template_path)
    with np.load(resolved_template_path) as data:
        timestamps = data["timestamps"].copy().astype(np.float64)

    gastric_period = regen.detect_monitor_period(resolved_monitor_path)
    reference_model = regen.load_reference_model(instance_paths.reference_ply)
    axis_model = build_peristaltic_axis_model(instance_name=instance_name, reference_ply=instance_paths.reference_ply)
    axis_u_lookup, s_lookup = build_scanner_s_lookup(reference_model, axis_model, phase=0.0)
    observation_transform = _load_observation_transform(phase_model_dir)

    scanner_sequence_path = output_root / "clean" / "scanner_sequence.npz"
    scanner_png_dir = output_root / "clean" / "image" / "scanner"
    monitor_out_path = output_root / "clean" / "monitor_stream.npz"
    monitor_out_path.parent.mkdir(parents=True, exist_ok=True)
    if not monitor_out_path.exists():
        monitor_out_path.symlink_to(resolved_monitor_path)

    frame_count = len(timestamps)
    positions = np.zeros((frame_count, 3), dtype=np.float64)
    orientations = np.zeros((frame_count, 3, 3), dtype=np.float64)

    if rewrite_pngs:
        scanner_png_dir.mkdir(parents=True, exist_ok=True)
        for png_path in scanner_png_dir.glob("scanner_*.png"):
            png_path.unlink()

    tmp_root = scanner_sequence_path.parent
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scanner_headtail_dense_", dir=str(tmp_root)) as tmp_dir:
        frames_path = Path(tmp_dir) / "frames.npy"
        frames = np.lib.format.open_memmap(frames_path, mode="w+", dtype=np.uint8, shape=(frame_count, FRAME_SIZE, FRAME_SIZE))
        for index, timestamp in enumerate(timestamps):
            phase = float((timestamp % gastric_period) / gastric_period)
            phase_index = _nearest_phase_index(phase_values, phase)
            axis_u = _headtail_dense_axis_u(float(timestamp), gastric_period)
            position = interpolate_centerline_position(axis_model, axis_u)
            tangent = interpolate_centerline_tangent(axis_model, axis_u)
            prior_s = float(axis_u_to_scanner_s(axis_u, axis_u_lookup, s_lookup))
            prior_orientation = regen.probe_orientation(reference_model, prior_s, phase, float(timestamp))
            orientation = _headtail_dense_probe_orientation(prior_orientation, tangent, axis_u, phase, float(timestamp))
            position, orientation = _apply_pose_transform(position, orientation, observation_transform)
            frame_uint8 = (np.clip(_slice_mesh_frame(meshes[phase_index], position, orientation, float(timestamp), phase), 0.0, 1.0) * 255.0).astype(np.uint8)

            positions[index] = position
            orientations[index] = orientation
            frames[index] = frame_uint8
            if rewrite_pngs:
                Image.fromarray(frame_uint8, mode="L").save(scanner_png_dir / f"scanner_{index:04d}.png")
            if index < 5 or index % 1000 == 0 or index == frame_count - 1:
                print(f"[HeadTailDenseScanner] frame={index:05d}/{frame_count - 1} ts={timestamp:.3f}s phase={phase:.3f} axis_u_target={axis_u:.3f} prior_s={prior_s:.3f}")

        frames.flush()
        tmp_path = scanner_sequence_path.with_name(scanner_sequence_path.stem + ".tmp.npz")
        np.savez_compressed(
            tmp_path,
            frames=frames,
            timestamps=timestamps.astype(np.float64),
            positions=positions.astype(np.float64),
            orientations=orientations.astype(np.float64),
        )
    final_tmp = tmp_path if tmp_path.exists() else scanner_sequence_path.with_name(scanner_sequence_path.stem + ".tmp.npz")
    final_tmp.replace(scanner_sequence_path)
    print(f"[HeadTailDenseScanner] Wrote clean scanner sequence: {scanner_sequence_path}")
    print(f"[HeadTailDenseScanner] PNG dir: {scanner_png_dir}")
    print(f"[HeadTailDenseScanner] Monitor link: {monitor_out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a head/tail-dense scanner variant for one stomach instance")
    parser.add_argument("--instance-name", required=True, type=str)
    parser.add_argument("--phase-model-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--monitor-path", type=Path, default=None)
    parser.add_argument("--scanner-template-path", type=Path, default=None)
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()

    instance_paths = resolve_instance_paths(instance_name=args.instance_name)
    phase_model_dir = args.phase_model_dir.expanduser().resolve() if args.phase_model_dir else _latest_phase_model_dir(instance_paths.phase_model_base_dir)
    generate_variant(
        instance_name=args.instance_name,
        phase_model_dir=phase_model_dir,
        output_root=args.output_root.expanduser().resolve(),
        monitor_path=args.monitor_path.expanduser().resolve() if args.monitor_path else None,
        scanner_template_path=args.scanner_template_path.expanduser().resolve() if args.scanner_template_path else None,
        rewrite_pngs=not args.no_png,
    )


if __name__ == "__main__":
    main()