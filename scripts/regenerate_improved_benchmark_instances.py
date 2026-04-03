from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_headtail_dense_scanner_variant import (
    _apply_pose_transform,
    _headtail_dense_axis_u,
    _headtail_dense_probe_orientation,
    _latest_phase_model_dir,
    _load_meshes,
    _nearest_phase_index,
    _read_phase_summary,
)
from scripts.generate_scanner_from_phase_models import _slice_mesh_frame
import scripts.regenerate_freehand_scanner_sequence as regen
from scripts.stomach_peristaltic_axis import (
    axis_u_to_scanner_s,
    build_peristaltic_axis_model,
    build_scanner_s_lookup,
    interpolate_centerline_position,
    interpolate_centerline_tangent,
)
from src.paths import data_path
from src.stomach_instance_paths import list_reference_pointclouds, resolve_instance_paths


FRAME_SIZE = 512
LEGACY_INSTANCE_ROOT = data_path("benchmark", "instances_before")


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


def _legacy_instance_root(instance_name: str, source_root: Path) -> Path:
    return source_root / instance_name


def _sync_monitor_inputs(instance_name: str, target_root: Path, source_root: Path) -> tuple[Path, Path]:
    legacy_root = _legacy_instance_root(instance_name, source_root)
    if not legacy_root.exists():
        raise FileNotFoundError(f"Legacy instance root not found: {legacy_root}")

    legacy_monitor_stream = legacy_root / "monitor_stream.npz"
    legacy_monitor_images = legacy_root / "image" / "monitor"
    if not legacy_monitor_stream.exists():
        raise FileNotFoundError(f"Missing legacy monitor stream: {legacy_monitor_stream}")
    if not legacy_monitor_images.exists():
        raise FileNotFoundError(f"Missing legacy monitor images: {legacy_monitor_images}")

    _symlink_or_copy(legacy_monitor_stream, target_root / "monitor_stream.npz")
    _symlink_or_copy(legacy_monitor_images, target_root / "image" / "monitor")
    return legacy_monitor_stream, legacy_root / "scanner_sequence.npz"


def regenerate_instance(instance_name: str, source_root: Path, rewrite_pngs: bool) -> None:
    instance_paths = resolve_instance_paths(instance_name=instance_name)
    phase_model_dir = _latest_phase_model_dir(instance_paths.phase_model_base_dir)
    summary_path = phase_model_dir / "phase_sequence_summary.csv"
    phase_values, mesh_paths = _read_phase_summary(summary_path)
    meshes = _load_meshes(mesh_paths)

    clean_root = instance_paths.scanner_sequence.parent
    clean_root.mkdir(parents=True, exist_ok=True)
    monitor_path, scanner_template_path = _sync_monitor_inputs(instance_name, clean_root, source_root)

    with np.load(scanner_template_path) as data:
        timestamps = data["timestamps"].copy().astype(np.float64)

    gastric_period = regen.detect_monitor_period(monitor_path)
    reference_model = regen.load_reference_model(instance_paths.reference_ply)
    axis_model = build_peristaltic_axis_model(instance_name=instance_name, reference_ply=instance_paths.reference_ply)
    axis_u_lookup, s_lookup = build_scanner_s_lookup(reference_model, axis_model, phase=0.0)
    observation_transform = None
    transform_path = phase_model_dir / "observation_transform.npz"
    if transform_path.exists():
        with np.load(transform_path) as data:
            observation_transform = (data["center"].astype(np.float64), data["rotation"].astype(np.float64))

    frame_count = len(timestamps)
    positions = np.zeros((frame_count, 3), dtype=np.float64)
    orientations = np.zeros((frame_count, 3, 3), dtype=np.float64)

    scanner_png_dir = instance_paths.scanner_image_dir
    if rewrite_pngs:
        scanner_png_dir.mkdir(parents=True, exist_ok=True)
        for png_path in scanner_png_dir.glob("scanner_*.png"):
            png_path.unlink()

    tmp_root = instance_paths.scanner_sequence.parent
    with tempfile.TemporaryDirectory(prefix=f"improved_scanner_{instance_name}_", dir=str(tmp_root)) as tmp_dir:
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
                print(
                    f"[ImprovedBenchmarkScanner] instance={instance_name} frame={index:05d}/{frame_count - 1} "
                    f"ts={timestamp:.3f}s phase={phase:.3f} axis_u={axis_u:.3f} prior_s={prior_s:.3f}"
                )

        frames.flush()
        tmp_path = instance_paths.scanner_sequence.with_name(instance_paths.scanner_sequence.stem + ".tmp.npz")
        np.savez_compressed(
            tmp_path,
            frames=frames,
            timestamps=timestamps.astype(np.float64),
            positions=positions.astype(np.float64),
            orientations=orientations.astype(np.float64),
        )

    final_tmp = tmp_path if tmp_path.exists() else instance_paths.scanner_sequence.with_name(instance_paths.scanner_sequence.stem + ".tmp.npz")
    final_tmp.replace(instance_paths.scanner_sequence)
    print(f"[ImprovedBenchmarkScanner] Wrote {instance_paths.scanner_sequence}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate benchmark/instances using the improved peristaltic scanner slicing method.")
    parser.add_argument("instance_names", nargs="*", help="Optional specific instance names. Defaults to all reference point clouds.")
    parser.add_argument("--source-root", type=Path, default=LEGACY_INSTANCE_ROOT, help="Legacy benchmark instance root used for monitor/template inputs.")
    parser.add_argument("--no-png", action="store_true", help="Do not rewrite scanner PNGs.")
    args = parser.parse_args()

    if args.instance_names:
        instance_names = args.instance_names
    else:
        instance_names = [path.stem for path in list_reference_pointclouds()]
        if not instance_names:
            raise FileNotFoundError("No reference point clouds found under stomach_pcd")

    for instance_name in instance_names:
        regenerate_instance(instance_name=instance_name, source_root=args.source_root.expanduser().resolve(), rewrite_pngs=not args.no_png)


if __name__ == "__main__":
    main()