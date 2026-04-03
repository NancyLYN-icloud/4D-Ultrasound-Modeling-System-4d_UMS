from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.generate_scanner_from_phase_models as base
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


def _clear_scanner_pngs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for file_path in path.glob("scanner_*.png"):
        file_path.unlink()


def generate_grouped_scanner_sequence(
    reference_ply: Path,
    phase_model_dir: Path,
    monitor_path: Path,
    scanner_template_path: Path | None,
    scanner_sequence_path: Path,
    scanner_image_dir: Path,
    rewrite_pngs: bool,
    fps: float | None,
    duration_seconds: float | None,
) -> None:
    summary_path = phase_model_dir / "phase_sequence_summary.csv"
    phase_values, mesh_paths = base._read_phase_summary(summary_path)
    meshes = base._load_meshes(mesh_paths)

    if fps is None or duration_seconds is None:
        template_path = _resolve_scanner_template_path(scanner_template_path, scanner_sequence_path)
        with np.load(template_path) as data:
            timestamps = data["timestamps"].copy().astype(np.float64)
    else:
        timestamps = base._build_timestamps(duration_seconds=duration_seconds, fps=fps)

    gastric_period = base.regen.detect_monitor_period(monitor_path)
    reference_model = base.regen.load_reference_model(reference_ply)
    observation_transform = base._load_observation_transform(phase_model_dir)
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
            phase = float((timestamp % gastric_period) / gastric_period)
            phase_index = base._nearest_phase_index(phase_values, phase)
            sweep_position = base.regen.sweep_coordinate(float(timestamp), gastric_period)
            position = base.regen.world_centerline(reference_model, sweep_position, phase)
            orientation = base.regen.probe_orientation(reference_model, sweep_position, phase, float(timestamp))
            position, orientation = base._apply_pose_transform(position, orientation, observation_transform)

            frame_float = base._slice_mesh_frame(meshes[phase_index], position, orientation, float(timestamp), phase)
            frame_uint8 = (np.clip(frame_float, 0.0, 1.0) * 255.0).astype(np.uint8)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grouped scanner sequences for the GPU dataset pipeline.")
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--instances", nargs="*", default=None)
    parser.add_argument("--phase-run-name", type=str, default="", help="Optional run directory name under each grouped phase root")
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

        resolved_monitor_path = _resolve_monitor_path(monitor_path, record.monitor_stream)
        generate_grouped_scanner_sequence(
            reference_ply=record.reference_ply,
            phase_model_dir=phase_model_dir,
            monitor_path=resolved_monitor_path,
            scanner_template_path=scanner_template_path,
            scanner_sequence_path=record.scanner_sequence,
            scanner_image_dir=record.scanner_image_dir,
            rewrite_pngs=not args.no_png,
            fps=args.fps,
            duration_seconds=args.duration_seconds,
        )
        print(f"[ScannerGPU] {record.instance_name} -> {record.scanner_sequence}")


if __name__ == "__main__":
    main()
