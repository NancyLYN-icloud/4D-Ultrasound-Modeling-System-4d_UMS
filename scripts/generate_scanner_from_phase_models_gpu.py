from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.generate_scanner_from_phase_models as base
from scripts.generate_headtail_dense_scanner_variant import (
    _headtail_dense_axis_u,
    _headtail_dense_probe_orientation,
)
from scripts.stomach_peristaltic_axis import (
    axis_u_to_scanner_s,
    build_peristaltic_axis_model,
    build_scanner_s_lookup,
    interpolate_centerline_position,
    interpolate_centerline_tangent,
)
from src.gastro4d_gpu_layout import select_grouped_reference_pointclouds
from src.paths import data_path


def _latest_phase_model_dir(base_dir: Path) -> Path:
    candidates = sorted(path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith(base.PHASE_MODEL_PREFIX))
    if not candidates:
        raise FileNotFoundError(f"No phase_sequence_models_run_* directory found under {base_dir}")
    return candidates[-1]


def _resolve_monitor_path(explicit_path: Path | None, clean_monitor_path: Path) -> Path:
    if explicit_path is not None:
        return explicit_path
    if clean_monitor_path.exists():
        return clean_monitor_path
    shared_path = data_path("benchmark", "monitor_stream.npz")
    if shared_path.exists():
        return shared_path
    if base.RAW_MONITOR_PATH.exists():
        return base.RAW_MONITOR_PATH
    raise FileNotFoundError(f"Monitor stream not found: {clean_monitor_path}")


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def _symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _remove_path(dst)
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
        return
    except OSError:
        pass
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _publish_monitor_assets(sim_monitor_path: Path, sim_monitor_image_dir: Path, benchmark_monitor_path: Path, benchmark_monitor_image_dir: Path) -> None:
    if not sim_monitor_path.exists():
        raise FileNotFoundError(f"Sim monitor stream not found: {sim_monitor_path}")
    if not sim_monitor_image_dir.exists():
        raise FileNotFoundError(f"Sim monitor image dir not found: {sim_monitor_image_dir}")
    _symlink_or_copy(sim_monitor_path, benchmark_monitor_path)
    _symlink_or_copy(sim_monitor_image_dir, benchmark_monitor_image_dir)


def _resolve_scanner_template_path(explicit_path: Path | None, clean_scanner_path: Path) -> Path:
    if explicit_path is not None:
        return explicit_path
    if clean_scanner_path.exists():
        return clean_scanner_path
    shared_path = data_path("benchmark", "scanner_sequence.npz")
    if shared_path.exists():
        return shared_path
    raise FileNotFoundError(
        "Scanner template path is required when the grouped clean scanner sequence does not yet exist"
    )


def _load_timestamps(
    scanner_template_path: Path | None,
    scanner_sequence_path: Path,
    fps: float | None,
    duration_seconds: float | None,
) -> np.ndarray:
    if fps is None or duration_seconds is None:
        template_path = _resolve_scanner_template_path(scanner_template_path, scanner_sequence_path)
        with np.load(template_path) as data:
            return data["timestamps"].copy().astype(np.float64)
    return base._build_timestamps(duration_seconds=duration_seconds, fps=fps)


def _clear_scanner_pngs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for file_path in path.glob("scanner_*.png"):
        file_path.unlink()


def _write_sequence(
    phase_values: np.ndarray,
    meshes: list,
    timestamps: np.ndarray,
    scanner_sequence_path: Path,
    scanner_image_dir: Path,
    rewrite_pngs: bool,
    frame_builder,
) -> None:
    frame_count = len(timestamps)
    positions = np.zeros((frame_count, 3), dtype=np.float64)
    orientations = np.zeros((frame_count, 3, 3), dtype=np.float64)

    scanner_image_dir.mkdir(parents=True, exist_ok=True)
    if rewrite_pngs:
        _clear_scanner_pngs(scanner_image_dir)

    scanner_sequence_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scanner_phase_frames_", dir=str(scanner_sequence_path.parent)) as tmp_dir:
        frames_path = Path(tmp_dir) / "frames.npy"
        frames = np.lib.format.open_memmap(
            frames_path,
            mode="w+",
            dtype=np.uint8,
            shape=(frame_count, base.FRAME_SIZE, base.FRAME_SIZE),
        )
        for index, timestamp in enumerate(timestamps):
            position, orientation, frame_uint8 = frame_builder(index=index, timestamp=float(timestamp), phase_values=phase_values, meshes=meshes)
            positions[index] = position
            orientations[index] = orientation
            frames[index] = frame_uint8

            if rewrite_pngs:
                Image.fromarray(frame_uint8, mode="L").save(scanner_image_dir / f"scanner_{index:04d}.png")

        np.savez_compressed(
            scanner_sequence_path,
            timestamps=timestamps.astype(np.float64),
            positions=positions.astype(np.float64),
            orientations=orientations.astype(np.float64),
            frames=np.asarray(frames, dtype=np.uint8),
        )


def generate_grouped_scanner_sequence(
    instance_name: str,
    reference_ply: Path,
    phase_model_dir: Path,
    monitor_path: Path,
    scanner_template_path: Path | None,
    scanner_sequence_path: Path,
    scanner_image_dir: Path,
    rewrite_pngs: bool,
    fps: float | None,
    duration_seconds: float | None,
    scanner_mode: str,
) -> None:
    summary_path = phase_model_dir / "phase_sequence_summary.csv"
    phase_values, mesh_paths = base._read_phase_summary(summary_path)
    meshes = base._load_meshes(mesh_paths)

    timestamps = _load_timestamps(
        scanner_template_path=scanner_template_path,
        scanner_sequence_path=scanner_sequence_path,
        fps=fps,
        duration_seconds=duration_seconds,
    )

    gastric_period = base.regen.detect_monitor_period(monitor_path)
    reference_model = base.regen.load_reference_model(reference_ply)
    observation_transform = base._load_observation_transform(phase_model_dir)
    if scanner_mode == "standard":
        def build_frame(*, index: int, timestamp: float, phase_values: np.ndarray, meshes: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            phase = float((timestamp % gastric_period) / gastric_period)
            phase_index = base._nearest_phase_index(phase_values, phase)
            sweep_position = base.regen.sweep_coordinate(timestamp, gastric_period)
            position = base.regen.world_centerline(reference_model, sweep_position, phase)
            orientation = base.regen.probe_orientation(reference_model, sweep_position, phase, timestamp)
            position, orientation = base._apply_pose_transform(position, orientation, observation_transform)
            frame_float = base._slice_mesh_frame(meshes[phase_index], position, orientation, timestamp, phase)
            frame_uint8 = (np.clip(frame_float, 0.0, 1.0) * 255.0).astype(np.uint8)
            return position, orientation, frame_uint8

    elif scanner_mode == "improved":
        axis_model = build_peristaltic_axis_model(instance_name=instance_name, reference_ply=reference_ply)
        axis_u_lookup, s_lookup = build_scanner_s_lookup(reference_model, axis_model, phase=0.0)

        def build_frame(*, index: int, timestamp: float, phase_values: np.ndarray, meshes: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            phase = float((timestamp % gastric_period) / gastric_period)
            phase_index = base._nearest_phase_index(phase_values, phase)
            axis_u = _headtail_dense_axis_u(timestamp, gastric_period)
            position = interpolate_centerline_position(axis_model, axis_u)
            tangent = interpolate_centerline_tangent(axis_model, axis_u)
            prior_s = float(axis_u_to_scanner_s(axis_u, axis_u_lookup, s_lookup))
            prior_orientation = base.regen.probe_orientation(reference_model, prior_s, phase, timestamp)
            orientation = _headtail_dense_probe_orientation(prior_orientation, tangent, axis_u, phase, timestamp)
            position, orientation = base._apply_pose_transform(position, orientation, observation_transform)
            frame_float = base._slice_mesh_frame(meshes[phase_index], position, orientation, timestamp, phase)
            frame_uint8 = (np.clip(frame_float, 0.0, 1.0) * 255.0).astype(np.uint8)
            return position, orientation, frame_uint8

    else:
        raise ValueError(f"Unsupported scanner mode: {scanner_mode}")

    _write_sequence(
        phase_values=phase_values,
        meshes=meshes,
        timestamps=timestamps,
        scanner_sequence_path=scanner_sequence_path,
        scanner_image_dir=scanner_image_dir,
        rewrite_pngs=rewrite_pngs,
        frame_builder=build_frame,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grouped scanner sequences for the GPU dataset pipeline.")
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--instances", nargs="*", default=None)
    parser.add_argument("--phase-run-name", type=str, default="", help="Optional run directory name under each grouped phase root")
    parser.add_argument("--scanner-mode", choices=["improved", "standard"], default="improved")
    parser.add_argument("--monitor-path", type=Path, default=None)
    parser.add_argument("--scanner-template-path", type=Path, default=None)
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--duration-seconds", type=float, default=None)
    args = parser.parse_args()

    if (args.fps is None) != (args.duration_seconds is None):
        raise ValueError("--fps and --duration-seconds must be provided together")

    monitor_path = args.monitor_path.expanduser().resolve() if args.monitor_path else None
    scanner_template_path = args.scanner_template_path.expanduser().resolve() if args.scanner_template_path else None
    records = select_grouped_reference_pointclouds(groups=args.groups, instances=args.instances)
    if not records:
        raise FileNotFoundError("No grouped reference point clouds matched the requested filters")

    for record in records:
        if args.phase_run_name:
            phase_model_dir = (record.phase_model_base_dir / args.phase_run_name).expanduser().resolve()
        else:
            phase_model_dir = _latest_phase_model_dir(record.phase_model_base_dir)
        if not phase_model_dir.exists():
            raise FileNotFoundError(f"Phase model directory not found: {phase_model_dir}")

        resolved_monitor_path = _resolve_monitor_path(monitor_path, record.sim_monitor_stream)
        _publish_monitor_assets(
            sim_monitor_path=resolved_monitor_path,
            sim_monitor_image_dir=record.sim_monitor_image_dir,
            benchmark_monitor_path=record.monitor_stream,
            benchmark_monitor_image_dir=record.monitor_image_dir,
        )
        generate_grouped_scanner_sequence(
            instance_name=record.instance_name,
            reference_ply=record.reference_ply,
            phase_model_dir=phase_model_dir,
            monitor_path=resolved_monitor_path,
            scanner_template_path=scanner_template_path,
            scanner_sequence_path=record.scanner_sequence,
            scanner_image_dir=record.scanner_image_dir,
            rewrite_pngs=not args.no_png,
            fps=args.fps,
            duration_seconds=args.duration_seconds,
            scanner_mode=args.scanner_mode,
        )
        print(f"[ScannerGPU] mode={args.scanner_mode} {record.instance_name} -> {record.scanner_sequence}")


if __name__ == "__main__":
    main()
