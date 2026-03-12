from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
import trimesh


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.regenerate_freehand_scanner_sequence as regen


TEST_SCANNER_PATH = ROOT / "data" / "test" / "scanner_sequence.npz"
TEST_MONITOR_PATH = ROOT / "data" / "test" / "monitor_stream.npz"
RAW_MONITOR_PATH = ROOT / "data" / "raw" / "monitor_stream.npz"
SCANNER_IMG_DIR = ROOT / "data" / "test" / "image" / "scanner"
PHASE_MODEL_PREFIX = "phase_sequence_models_run_"
FRAME_SIZE = 512
PIXEL_SPACING_MM = regen.PIXEL_SPACING_MM


def _latest_phase_model_dir() -> Path:
    candidates = sorted(
        [path for path in (ROOT / "data" / "test" / "processed").iterdir() if path.is_dir() and path.name.startswith(PHASE_MODEL_PREFIX)]
    )
    if not candidates:
        raise FileNotFoundError("No phase_sequence_models_run_* directory found")
    return candidates[-1]


def _read_phase_summary(summary_path: Path) -> tuple[np.ndarray, list[Path]]:
    phase_values: list[float] = []
    mesh_paths: list[Path] = []
    mesh_dir = summary_path.parent / "pointclouds" / "meshes"
    with summary_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            phase_values.append(float(row["phase_value"]))
            pointcloud_name = row["pointcloud"]
            mesh_name = pointcloud_name.replace(".ply", "_mesh.ply")
            mesh_paths.append(mesh_dir / mesh_name)
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


def _project_loop(loop_3d: np.ndarray, origin: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    rel = loop_3d - origin[None, :]
    axis_x = orientation[:, 0]
    axis_y = orientation[:, 1]
    x = rel @ axis_x
    y = rel @ axis_y
    return np.column_stack([x, y]).astype(np.float32)


def _rasterize_loops(projected_loops: list[np.ndarray], frame_size: int, pixel_spacing_mm: float) -> np.ndarray:
    image = Image.new("L", (frame_size, frame_size), 0)
    draw = ImageDraw.Draw(image)
    center = 0.5 * (frame_size - 1)
    for loop in projected_loops:
        if loop.shape[0] < 3:
            continue
        pixels = []
        for x_mm, y_mm in loop:
            px = center + float(x_mm) / pixel_spacing_mm
            py = center + float(y_mm) / pixel_spacing_mm
            pixels.append((px, py))
        draw.polygon(pixels, outline=210, fill=230)
    return np.asarray(image, dtype=np.float32) / 255.0


def _stylize_ultrasound(mask: np.ndarray, timestamp: float, phase: float) -> np.ndarray:
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, mask.shape[0]), np.linspace(-1.0, 1.0, mask.shape[1]), indexing="ij")
    radial = np.sqrt(xx**2 + yy**2)
    gain = 0.92 - 0.25 * radial
    envelope = np.clip(gain, 0.15, 1.0)
    blurred = gaussian_filter(mask, sigma=1.3)
    rng_seed = int(round(timestamp * 1000.0)) % (2**32 - 1)
    rng = np.random.default_rng(rng_seed)
    speckle = rng.normal(loc=1.0, scale=0.08 + 0.04 * phase, size=mask.shape).astype(np.float32)
    background = 0.03 + 0.025 * gaussian_filter(rng.random(mask.shape, dtype=np.float32), sigma=4.0)
    frame = background + np.clip(blurred * envelope * speckle, 0.0, 1.0)
    frame = gaussian_filter(frame, sigma=0.9)
    frame = np.clip(frame, 0.0, 1.0)
    return frame.astype(np.float32)


def _slice_mesh_frame(mesh: trimesh.Trimesh, origin: np.ndarray, orientation: np.ndarray, timestamp: float, phase: float) -> np.ndarray:
    normal = orientation[:, 2]
    offsets = (-0.9, 0.0, 0.9)
    accum = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
    valid = 0
    for offset in offsets:
        plane_origin = origin + normal * offset
        section = mesh.section(plane_origin=plane_origin, plane_normal=normal)
        if section is None:
            continue
        loops = section.discrete
        if not loops:
            continue
        projected_loops = [_project_loop(loop, origin, orientation) for loop in loops if loop.shape[0] >= 3]
        if not projected_loops:
            continue
        accum += _rasterize_loops(projected_loops, FRAME_SIZE, PIXEL_SPACING_MM)
        valid += 1
    if valid == 0:
        return np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
    mask = accum / float(valid)
    return _stylize_ultrasound(mask, timestamp, phase)


def _clear_scanner_pngs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for file_path in path.glob("scanner_*.png"):
        file_path.unlink()


def _save_png(frame: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(path)


def generate_from_phase_models(phase_model_dir: Path, rewrite_pngs: bool = True) -> None:
    summary_path = phase_model_dir / "phase_sequence_summary.csv"
    phase_values, mesh_paths = _read_phase_summary(summary_path)
    meshes = _load_meshes(mesh_paths)

    with np.load(TEST_SCANNER_PATH) as data:
        timestamps = data["timestamps"].copy().astype(np.float64)

    monitor_path = TEST_MONITOR_PATH if TEST_MONITOR_PATH.exists() else RAW_MONITOR_PATH
    gastric_period = regen.detect_monitor_period(monitor_path)
    reference_model = regen.load_reference_model(ROOT / "data" / "test" / "stomach.ply")

    frame_count = len(timestamps)
    frames = np.zeros((frame_count, FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
    positions = np.zeros((frame_count, 3), dtype=np.float64)
    orientations = np.zeros((frame_count, 3, 3), dtype=np.float64)

    if rewrite_pngs:
        _clear_scanner_pngs(SCANNER_IMG_DIR)

    for index, timestamp in enumerate(timestamps):
        phase = float((timestamp % gastric_period) / gastric_period)
        phase_index = _nearest_phase_index(phase_values, phase)
        sweep_position = regen.sweep_coordinate(float(timestamp), gastric_period)
        positions[index] = regen.world_centerline(reference_model, sweep_position, phase)
        orientations[index] = regen.probe_orientation(reference_model, sweep_position, phase, float(timestamp))
        frames[index] = _slice_mesh_frame(meshes[phase_index], positions[index], orientations[index], float(timestamp), phase)
        if rewrite_pngs:
            _save_png(frames[index], SCANNER_IMG_DIR / f"scanner_{index:04d}.png")
        if index < 5 or index % 400 == 0 or index == frame_count - 1:
            print(
                f"[ScannerFromPhaseModels] frame={index:04d}/{frame_count - 1} "
                f"ts={timestamp:.3f}s phase={phase:.3f} mesh_phase={phase_values[phase_index]:.3f}"
            )

    tmp_path = TEST_SCANNER_PATH.with_name(TEST_SCANNER_PATH.stem + ".tmp.npz")
    np.savez_compressed(
        tmp_path,
        frames=frames.astype(np.float32),
        timestamps=timestamps.astype(np.float64),
        positions=positions.astype(np.float64),
        orientations=orientations.astype(np.float64),
    )
    tmp_final = tmp_path if tmp_path.suffix == ".npz" else tmp_path.with_suffix(".npz")
    final_tmp = tmp_final if tmp_final.exists() else tmp_path
    final_tmp.replace(TEST_SCANNER_PATH)

    print(f"[ScannerFromPhaseModels] Rewrote {TEST_SCANNER_PATH}")
    print(f"[ScannerFromPhaseModels] Phase model dir: {phase_model_dir}")
    print(f"[ScannerFromPhaseModels] Monitor path: {monitor_path}")
    print(f"[ScannerFromPhaseModels] Gastric period: {gastric_period:.6f}s")
    print(f"[ScannerFromPhaseModels] Frames: {len(frames)}")
    print(f"[ScannerFromPhaseModels] PNG dir: {SCANNER_IMG_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate test scanner_sequence.npz from phase-sequence stomach models")
    parser.add_argument("--phase-model-dir", type=str, default="", help="Specific phase_sequence_models_run directory to use")
    parser.add_argument("--no-png", action="store_true", help="Do not rewrite scanner PNG images")
    args = parser.parse_args()

    phase_model_dir = Path(args.phase_model_dir) if args.phase_model_dir else _latest_phase_model_dir()
    if not phase_model_dir.exists():
        raise FileNotFoundError(f"Phase model directory not found: {phase_model_dir}")
    generate_from_phase_models(phase_model_dir=phase_model_dir, rewrite_pngs=not args.no_png)


if __name__ == "__main__":
    main()