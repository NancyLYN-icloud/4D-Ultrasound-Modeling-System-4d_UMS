from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import data_path


DEFAULT_SOURCE_MANIFEST = ROOT / "experiments" / "benchmark_manifest.csv"
DEFAULT_CONDITION_MANIFEST = ROOT / "experiments" / "benchmark_condition_manifest.csv"
DEFAULT_CONDITION_ROOT = data_path("benchmark", "conditions")
SPARSE_CONDITION = "Sparse"
POSE_NOISE_CONDITION = "PoseNoise"
IMAGE_NOISE_CONDITION = "ImageNoise"


@dataclass(frozen=True)
class BenchmarkInstanceRow:
    instance_name: str
    shape_family: str
    split: str
    reference_ply: Path
    monitor_stream: Path
    scanner_sequence: Path
    phase_model_dir: Path
    phase_summary: Path

    @property
    def clean_root(self) -> Path:
        return self.scanner_sequence.parent

    @property
    def clean_monitor_image_dir(self) -> Path:
        return self.clean_root / "image" / "monitor"

    @property
    def clean_scanner_image_dir(self) -> Path:
        return self.clean_root / "image" / "scanner"


def _condition_slug(condition: str) -> str:
    lowered = condition.strip().lower()
    if lowered == "sparse":
        return "sparse"
    if lowered in {"posenoise", "pose_noise", "pose-noise", "noisypose", "noisy_pose", "noisy-pose"}:
        return "pose_noise"
    if lowered in {"imagenoise", "image_noise", "image-noise", "noisyimage", "noisy_image", "noisy-image"}:
        return "image_noise"
    raise ValueError(f"Unsupported condition: {condition}")


def _load_manifest(path: Path) -> list[BenchmarkInstanceRow]:
    rows: list[BenchmarkInstanceRow] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                BenchmarkInstanceRow(
                    instance_name=raw["instance_name"].strip(),
                    shape_family=raw["shape_family"].strip(),
                    split=raw["split"].strip(),
                    reference_ply=Path(raw["reference_ply"]).expanduser().resolve(),
                    monitor_stream=Path(raw["monitor_stream"]).expanduser().resolve(),
                    scanner_sequence=Path(raw["scanner_sequence"]).expanduser().resolve(),
                    phase_model_dir=Path(raw["phase_model_dir"]).expanduser().resolve(),
                    phase_summary=Path(raw["phase_summary"]).expanduser().resolve(),
                )
            )
    return rows


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


def _write_npz(path: Path, compressed: bool = True, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), prefix=path.stem + "_", suffix=".npz", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        if compressed:
            np.savez_compressed(tmp_path, **arrays)
        else:
            np.savez(tmp_path, **arrays)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _load_scanner_metadata(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        timestamps = data["timestamps"].astype(np.float64)
        positions = data["positions"].astype(np.float64)
        orientations = data["orientations"].astype(np.float64)
    return timestamps, positions, orientations


def _count_scanner_frames(scanner_dir: Path) -> int:
    return len(sorted(scanner_dir.glob("scanner_*.png")))


def _read_scanner_frame(scanner_dir: Path, index: int) -> np.ndarray:
    frame_path = scanner_dir / f"scanner_{index:04d}.png"
    if not frame_path.exists():
        raise FileNotFoundError(f"Missing scanner frame: {frame_path}")
    return np.asarray(Image.open(frame_path).convert("L"), dtype=np.uint8)


def _build_frames_array(scanner_dir: Path, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        raise ValueError("No scanner frame indices selected")
    first_frame = _read_scanner_frame(scanner_dir, int(indices[0]))
    frames = np.empty((indices.size, first_frame.shape[0], first_frame.shape[1]), dtype=np.uint8)
    frames[0] = first_frame
    for out_idx, src_idx in enumerate(indices[1:], start=1):
        frames[out_idx] = _read_scanner_frame(scanner_dir, int(src_idx))
    return frames


def _write_scanner_pngs(scanner_dir: Path, frames: np.ndarray) -> None:
    _remove_path(scanner_dir)
    scanner_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame.astype(np.uint8), mode="L").save(scanner_dir / f"scanner_{index:04d}.png")


def _create_frame_memmap(frame_count: int, frame_shape: tuple[int, int], prefix: str) -> tuple[Path, np.memmap]:
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".npy", delete=False) as handle:
        tmp_path = Path(handle.name)
    memmap = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.uint8, shape=(frame_count, frame_shape[0], frame_shape[1]))
    return tmp_path, memmap


def _rotation_matrices_from_rotvec(rotvecs: np.ndarray) -> np.ndarray:
    angles = np.linalg.norm(rotvecs, axis=1, keepdims=True)
    axis = np.divide(rotvecs, np.maximum(angles, 1e-12), out=np.zeros_like(rotvecs), where=angles > 1e-12)
    kx = axis[:, 0]
    ky = axis[:, 1]
    kz = axis[:, 2]
    cos_a = np.cos(angles[:, 0])
    sin_a = np.sin(angles[:, 0])
    one_minus = 1.0 - cos_a

    matrices = np.zeros((rotvecs.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = cos_a + kx * kx * one_minus
    matrices[:, 0, 1] = kx * ky * one_minus - kz * sin_a
    matrices[:, 0, 2] = kx * kz * one_minus + ky * sin_a
    matrices[:, 1, 0] = ky * kx * one_minus + kz * sin_a
    matrices[:, 1, 1] = cos_a + ky * ky * one_minus
    matrices[:, 1, 2] = ky * kz * one_minus - kx * sin_a
    matrices[:, 2, 0] = kz * kx * one_minus - ky * sin_a
    matrices[:, 2, 1] = kz * ky * one_minus + kx * sin_a
    matrices[:, 2, 2] = cos_a + kz * kz * one_minus
    zero_angle = angles[:, 0] <= 1e-12
    matrices[zero_angle] = np.eye(3, dtype=np.float64)
    return matrices


def _smooth_noise(rng: np.random.Generator, frame_count: int, dims: int, scale: float, sigma_frames: float) -> np.ndarray:
    raw = rng.normal(0.0, 1.0, size=(frame_count, dims)).astype(np.float64)
    smoothed = gaussian_filter1d(raw, sigma=sigma_frames, axis=0, mode="nearest")
    std = np.std(smoothed, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (smoothed / std) * scale


def _sparse_indices(frame_count: int, step: int, offset: int) -> np.ndarray:
    if step < 1:
        raise ValueError("Sparse step must be >= 1")
    indices = np.arange(offset, frame_count, step, dtype=np.int64)
    if indices.size < 2:
        raise ValueError(f"Sparse selection is too aggressive: frame_count={frame_count}, step={step}, offset={offset}")
    return indices


def _condition_instance_root(condition_root: Path, condition: str, instance_name: str) -> Path:
    return condition_root / _condition_slug(condition) / "instances" / instance_name


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_sparse_condition(
    row: BenchmarkInstanceRow,
    condition_root: Path,
    sparse_step: int,
    sparse_offset_mode: str,
    overwrite: bool,
) -> dict[str, str]:
    target_root = _condition_instance_root(condition_root, SPARSE_CONDITION, row.instance_name)
    if overwrite:
        _remove_path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    timestamps, positions, orientations = _load_scanner_metadata(row.scanner_sequence)
    scanner_dir = row.clean_scanner_image_dir
    frame_count = len(timestamps)
    if sparse_offset_mode == "hash":
        offset = sum(ord(ch) for ch in row.instance_name) % sparse_step
    else:
        offset = 0
    indices = _sparse_indices(frame_count, sparse_step, offset)
    frames = _build_frames_array(scanner_dir, indices)

    sparse_scanner = target_root / "scanner_sequence.npz"
    _write_npz(
        sparse_scanner,
        frames=frames,
        timestamps=timestamps[indices].astype(np.float64),
        positions=positions[indices].astype(np.float64),
        orientations=orientations[indices].astype(np.float64),
    )

    _symlink_or_copy(row.monitor_stream, target_root / "monitor_stream.npz")
    _symlink_or_copy(row.clean_monitor_image_dir, target_root / "image" / "monitor")

    sparse_image_dir = target_root / "image" / "scanner"
    _remove_path(sparse_image_dir)
    sparse_image_dir.mkdir(parents=True, exist_ok=True)
    for new_idx, src_idx in enumerate(indices):
        src = scanner_dir / f"scanner_{int(src_idx):04d}.png"
        dst = sparse_image_dir / f"scanner_{new_idx:04d}.png"
        _symlink_or_copy(src, dst)

    metadata_path = target_root / "condition_metadata.json"
    _write_metadata(
        metadata_path,
        {
            "condition": SPARSE_CONDITION,
            "instance_name": row.instance_name,
            "source_scanner_sequence": str(row.scanner_sequence),
            "source_scanner_image_dir": str(scanner_dir),
            "source_monitor_stream": str(row.monitor_stream),
            "sparse_step": sparse_step,
            "sparse_offset": int(offset),
            "selected_frame_count": int(indices.size),
            "selected_frame_ratio": float(indices.size / frame_count),
        },
    )

    return {
        "instance_name": row.instance_name,
        "shape_family": row.shape_family,
        "split": row.split,
        "condition": SPARSE_CONDITION,
        "reference_ply": str(row.reference_ply),
        "monitor_stream": str(target_root / "monitor_stream.npz"),
        "scanner_sequence": str(sparse_scanner),
        "phase_model_dir": str(row.phase_model_dir),
        "phase_summary": str(row.phase_summary),
        "condition_root": str(target_root),
        "condition_metadata": str(metadata_path),
    }


def _build_pose_noise_condition(
    row: BenchmarkInstanceRow,
    condition_root: Path,
    translation_noise_mm: float,
    rotation_noise_deg: float,
    noise_sigma_frames: float,
    seed: int,
    overwrite: bool,
) -> dict[str, str]:
    target_root = _condition_instance_root(condition_root, POSE_NOISE_CONDITION, row.instance_name)
    if overwrite:
        _remove_path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    timestamps, positions, orientations = _load_scanner_metadata(row.scanner_sequence)
    scanner_dir = row.clean_scanner_image_dir
    frame_indices = np.arange(len(timestamps), dtype=np.int64)
    frames = _build_frames_array(scanner_dir, frame_indices)

    rng = np.random.default_rng(seed + sum(ord(ch) for ch in row.instance_name))
    translation_noise = _smooth_noise(rng, len(timestamps), 3, translation_noise_mm, noise_sigma_frames)
    rotation_noise = _smooth_noise(rng, len(timestamps), 3, np.deg2rad(rotation_noise_deg), noise_sigma_frames)
    delta_rotation = _rotation_matrices_from_rotvec(rotation_noise)
    noisy_positions = positions + translation_noise
    noisy_orientations = np.einsum("nij,njk->nik", delta_rotation, orientations)

    noisy_scanner = target_root / "scanner_sequence.npz"
    _write_npz(
        noisy_scanner,
        frames=frames,
        timestamps=timestamps.astype(np.float64),
        positions=noisy_positions.astype(np.float64),
        orientations=noisy_orientations.astype(np.float64),
    )

    _symlink_or_copy(row.monitor_stream, target_root / "monitor_stream.npz")
    _symlink_or_copy(row.clean_monitor_image_dir, target_root / "image" / "monitor")
    _symlink_or_copy(scanner_dir, target_root / "image" / "scanner")

    metadata_path = target_root / "condition_metadata.json"
    _write_metadata(
        metadata_path,
        {
            "condition": POSE_NOISE_CONDITION,
            "instance_name": row.instance_name,
            "source_scanner_sequence": str(row.scanner_sequence),
            "source_scanner_image_dir": str(scanner_dir),
            "source_monitor_stream": str(row.monitor_stream),
            "translation_noise_mm": translation_noise_mm,
            "rotation_noise_deg": rotation_noise_deg,
            "noise_sigma_frames": noise_sigma_frames,
            "seed": seed,
            "definition": "Pose metadata perturbation only: scanner image frames and monitor stream are unchanged.",
            "image_frames_changed": False,
            "monitor_stream_changed": False,
        },
    )

    return {
        "instance_name": row.instance_name,
        "shape_family": row.shape_family,
        "split": row.split,
        "condition": POSE_NOISE_CONDITION,
        "reference_ply": str(row.reference_ply),
        "monitor_stream": str(target_root / "monitor_stream.npz"),
        "scanner_sequence": str(noisy_scanner),
        "phase_model_dir": str(row.phase_model_dir),
        "phase_summary": str(row.phase_summary),
        "condition_root": str(target_root),
        "condition_metadata": str(metadata_path),
    }


def _build_image_noise_condition(
    row: BenchmarkInstanceRow,
    condition_root: Path,
    image_noise_std: float,
    image_bias_std: float,
    image_gain_std: float,
    image_blur_sigma: float,
    noise_sigma_frames: float,
    seed: int,
    materialize_scanner_pngs: bool,
    overwrite: bool,
) -> dict[str, str]:
    target_root = _condition_instance_root(condition_root, IMAGE_NOISE_CONDITION, row.instance_name)
    if overwrite:
        _remove_path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    timestamps, positions, orientations = _load_scanner_metadata(row.scanner_sequence)
    scanner_dir = row.clean_scanner_image_dir
    frame_count = len(timestamps)
    first_frame = _read_scanner_frame(scanner_dir, 0)

    rng = np.random.default_rng(seed + 10007 + sum(ord(ch) for ch in row.instance_name))
    bias_noise = _smooth_noise(rng, frame_count, 1, image_bias_std, noise_sigma_frames)[:, 0]
    gain_noise = 1.0 + _smooth_noise(rng, frame_count, 1, image_gain_std, noise_sigma_frames)[:, 0]
    frame_memmap_path, noisy_frames = _create_frame_memmap(frame_count, tuple(first_frame.shape), f"{row.instance_name}_image_noise_")

    scanner_png_dir = target_root / "image" / "scanner"
    if materialize_scanner_pngs:
        _remove_path(scanner_png_dir)
        scanner_png_dir.mkdir(parents=True, exist_ok=True)
    else:
        _remove_path(scanner_png_dir)

    try:
        for frame_index in range(frame_count):
            frame = _read_scanner_frame(scanner_dir, frame_index).astype(np.float32)
            frame += rng.normal(0.0, image_noise_std, size=frame.shape).astype(np.float32)
            frame = frame * np.float32(gain_noise[frame_index]) + np.float32(bias_noise[frame_index])
            if image_blur_sigma > 0.0:
                frame = gaussian_filter(frame, sigma=image_blur_sigma, mode="nearest")
            frame_uint8 = np.clip(np.rint(frame), 0.0, 255.0).astype(np.uint8)
            noisy_frames[frame_index] = frame_uint8
            if materialize_scanner_pngs:
                Image.fromarray(frame_uint8, mode="L").save(scanner_png_dir / f"scanner_{frame_index:04d}.png")

        noisy_frames.flush()

        noisy_scanner = target_root / "scanner_sequence.npz"
        _write_npz(
            noisy_scanner,
            compressed=False,
            frames=noisy_frames,
            timestamps=timestamps.astype(np.float64),
            positions=positions.astype(np.float64),
            orientations=orientations.astype(np.float64),
        )
    finally:
        try:
            del noisy_frames
        finally:
            if frame_memmap_path.exists():
                frame_memmap_path.unlink()

    _symlink_or_copy(row.monitor_stream, target_root / "monitor_stream.npz")
    _symlink_or_copy(row.clean_monitor_image_dir, target_root / "image" / "monitor")

    metadata_path = target_root / "condition_metadata.json"
    _write_metadata(
        metadata_path,
        {
            "condition": IMAGE_NOISE_CONDITION,
            "instance_name": row.instance_name,
            "source_scanner_sequence": str(row.scanner_sequence),
            "source_scanner_image_dir": str(scanner_dir),
            "source_monitor_stream": str(row.monitor_stream),
            "image_noise_std": image_noise_std,
            "image_bias_std": image_bias_std,
            "image_gain_std": image_gain_std,
            "image_blur_sigma": image_blur_sigma,
            "noise_sigma_frames": noise_sigma_frames,
            "seed": seed,
            "definition": "Scanner image perturbation only: scanner poses and monitor stream are unchanged.",
            "image_frames_changed": True,
            "monitor_stream_changed": False,
            "scanner_pose_changed": False,
            "scanner_pngs_materialized": materialize_scanner_pngs,
        },
    )

    return {
        "instance_name": row.instance_name,
        "shape_family": row.shape_family,
        "split": row.split,
        "condition": IMAGE_NOISE_CONDITION,
        "reference_ply": str(row.reference_ply),
        "monitor_stream": str(target_root / "monitor_stream.npz"),
        "scanner_sequence": str(noisy_scanner),
        "phase_model_dir": str(row.phase_model_dir),
        "phase_summary": str(row.phase_summary),
        "condition_root": str(target_root),
        "condition_metadata": str(metadata_path),
    }


def _clean_manifest_row(row: BenchmarkInstanceRow) -> dict[str, str]:
    return {
        "instance_name": row.instance_name,
        "shape_family": row.shape_family,
        "split": row.split,
        "condition": "Clean",
        "reference_ply": str(row.reference_ply),
        "monitor_stream": str(row.monitor_stream),
        "scanner_sequence": str(row.scanner_sequence),
        "phase_model_dir": str(row.phase_model_dir),
        "phase_summary": str(row.phase_summary),
        "condition_root": str(row.clean_root),
        "condition_metadata": "",
    }


def _write_condition_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "instance_name",
        "shape_family",
        "split",
        "condition",
        "reference_ply",
        "monitor_stream",
        "scanner_sequence",
        "phase_model_dir",
        "phase_summary",
        "condition_root",
        "condition_metadata",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate derived benchmark conditions from clean benchmark instances")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["Sparse", "PoseNoise", "ImageNoise"],
        choices=["Sparse", "PoseNoise", "ImageNoise"],
        help="Derived conditions to generate",
    )
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--condition-manifest", type=Path, default=DEFAULT_CONDITION_MANIFEST)
    parser.add_argument("--condition-root", type=Path, default=DEFAULT_CONDITION_ROOT)
    parser.add_argument("--instances", nargs="*", default=None, help="Optional instance names to process")
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    parser.add_argument("--sparse-step", type=int, default=2, help="Keep every N-th scanner frame for the Sparse condition")
    parser.add_argument(
        "--sparse-offset-mode",
        choices=["zero", "hash"],
        default="hash",
        help="Use a deterministic per-instance frame offset before sparse subsampling",
    )
    parser.add_argument("--translation-noise-mm", type=float, default=1.0, help="Std target of smooth position noise in mm")
    parser.add_argument("--rotation-noise-deg", type=float, default=1.5, help="Std target of smooth orientation noise in degrees")
    parser.add_argument("--noise-sigma-frames", type=float, default=24.0, help="Temporal smoothing of pose or image noise in frames")
    parser.add_argument("--image-noise-std", type=float, default=6.0, help="Per-pixel Gaussian intensity noise std for the ImageNoise condition")
    parser.add_argument("--image-bias-std", type=float, default=4.0, help="Smooth per-frame brightness bias std for the ImageNoise condition")
    parser.add_argument("--image-gain-std", type=float, default=0.04, help="Smooth per-frame multiplicative gain std for the ImageNoise condition")
    parser.add_argument("--image-blur-sigma", type=float, default=0.6, help="Light Gaussian blur sigma for the ImageNoise condition")
    parser.add_argument(
        "--materialize-image-noise-pngs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to write noisy scanner PNG files for ImageNoise in addition to storing frames in scanner_sequence.npz",
    )
    parser.add_argument("--seed", type=int, default=20260331)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing derived condition directories")
    args = parser.parse_args()

    source_rows = _load_manifest(args.source_manifest)
    selected_instances = set(args.instances or [])
    filtered_rows = [
        row
        for row in source_rows
        if (args.split == "all" or row.split == args.split)
        and (not selected_instances or row.instance_name in selected_instances)
    ]
    if not filtered_rows:
        raise FileNotFoundError("No benchmark instances matched the requested filters")

    for row in filtered_rows:
        if not row.monitor_stream.exists():
            raise FileNotFoundError(f"Missing monitor stream: {row.monitor_stream}")
        if not row.scanner_sequence.exists():
            raise FileNotFoundError(f"Missing scanner sequence: {row.scanner_sequence}")
        if not row.clean_scanner_image_dir.exists():
            raise FileNotFoundError(f"Missing scanner image dir: {row.clean_scanner_image_dir}")
        if _count_scanner_frames(row.clean_scanner_image_dir) == 0:
            raise FileNotFoundError(f"No scanner PNGs found under {row.clean_scanner_image_dir}")

    manifest_rows = [_clean_manifest_row(row) for row in source_rows]

    for row in filtered_rows:
        if SPARSE_CONDITION in args.conditions:
            manifest_rows.append(
                _build_sparse_condition(
                    row=row,
                    condition_root=args.condition_root,
                    sparse_step=args.sparse_step,
                    sparse_offset_mode=args.sparse_offset_mode,
                    overwrite=args.overwrite,
                )
            )
            print(f"[BenchmarkConditions] Generated Sparse for {row.instance_name}")

        if POSE_NOISE_CONDITION in args.conditions:
            manifest_rows.append(
                _build_pose_noise_condition(
                    row=row,
                    condition_root=args.condition_root,
                    translation_noise_mm=args.translation_noise_mm,
                    rotation_noise_deg=args.rotation_noise_deg,
                    noise_sigma_frames=args.noise_sigma_frames,
                    seed=args.seed,
                    overwrite=args.overwrite,
                )
            )
            print(f"[BenchmarkConditions] Generated PoseNoise for {row.instance_name}")

        if IMAGE_NOISE_CONDITION in args.conditions:
            manifest_rows.append(
                _build_image_noise_condition(
                    row=row,
                    condition_root=args.condition_root,
                    image_noise_std=args.image_noise_std,
                    image_bias_std=args.image_bias_std,
                    image_gain_std=args.image_gain_std,
                    image_blur_sigma=args.image_blur_sigma,
                    noise_sigma_frames=args.noise_sigma_frames,
                    seed=args.seed,
                    materialize_scanner_pngs=args.materialize_image_noise_pngs,
                    overwrite=args.overwrite,
                )
            )
            print(f"[BenchmarkConditions] Generated ImageNoise for {row.instance_name}")

    manifest_rows.sort(key=lambda item: (item["instance_name"], item["condition"]))
    _write_condition_manifest(args.condition_manifest, manifest_rows)
    print(f"[BenchmarkConditions] Wrote condition manifest: {args.condition_manifest}")
    print(f"[BenchmarkConditions] Condition root: {args.condition_root}")


if __name__ == "__main__":
    main()