from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.generate_benchmark_conditions as base
from src.gastro4d_gpu_layout import grouped_condition_root
from src.paths import data_path


DEFAULT_SOURCE_MANIFEST = data_path("benchmark", "manifests", "benchmark_manifest_gpu.csv")
DEFAULT_CONDITION_MANIFEST = data_path("benchmark", "manifests", "benchmark_condition_manifest_gpu.csv")
DEFAULT_CONDITION_ROOT = data_path("benchmark", "conditions")


@dataclass(frozen=True)
class BenchmarkInstanceGpuRow:
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
    return base._condition_slug(condition)


def _condition_instance_root(condition_root: Path, condition: str, row: BenchmarkInstanceGpuRow) -> Path:
    return grouped_condition_root(condition_root, _condition_slug(condition), row.split, row.shape_family, row.instance_name)


def _load_manifest(path: Path) -> list[BenchmarkInstanceGpuRow]:
    rows: list[BenchmarkInstanceGpuRow] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                BenchmarkInstanceGpuRow(
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


def _build_sparse_condition(
    row: BenchmarkInstanceGpuRow,
    condition_root: Path,
    sparse_step: int,
    sparse_offset_mode: str,
    overwrite: bool,
) -> dict[str, str]:
    target_root = _condition_instance_root(condition_root, base.SPARSE_CONDITION, row)
    if overwrite:
        base._remove_path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    timestamps, positions, orientations = base._load_scanner_metadata(row.scanner_sequence)
    frame_count = len(timestamps)
    if sparse_offset_mode == "hash":
        offset = sum(ord(ch) for ch in row.instance_name) % sparse_step
    else:
        offset = 0
    indices = base._sparse_indices(frame_count, sparse_step, offset)
    frames = base._build_frames_array(row.clean_scanner_image_dir, indices)

    sparse_scanner = target_root / "scanner_sequence.npz"
    base._write_npz(
        sparse_scanner,
        frames=frames,
        timestamps=timestamps[indices].astype(np.float64),
        positions=positions[indices].astype(np.float64),
        orientations=orientations[indices].astype(np.float64),
    )
    base._symlink_or_copy(row.monitor_stream, target_root / "monitor_stream.npz")
    base._symlink_or_copy(row.clean_monitor_image_dir, target_root / "image" / "monitor")

    sparse_image_dir = target_root / "image" / "scanner"
    base._remove_path(sparse_image_dir)
    sparse_image_dir.mkdir(parents=True, exist_ok=True)
    for new_idx, src_idx in enumerate(indices):
        src = row.clean_scanner_image_dir / f"scanner_{int(src_idx):04d}.png"
        dst = sparse_image_dir / f"scanner_{new_idx:04d}.png"
        base._symlink_or_copy(src, dst)

    metadata_path = target_root / "condition_metadata.json"
    base._write_metadata(
        metadata_path,
        {
            "condition": base.SPARSE_CONDITION,
            "instance_name": row.instance_name,
            "source_scanner_sequence": str(row.scanner_sequence),
            "source_scanner_image_dir": str(row.clean_scanner_image_dir),
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
        "condition": base.SPARSE_CONDITION,
        "reference_ply": str(row.reference_ply),
        "monitor_stream": str(target_root / "monitor_stream.npz"),
        "scanner_sequence": str(sparse_scanner),
        "phase_model_dir": str(row.phase_model_dir),
        "phase_summary": str(row.phase_summary),
        "condition_root": str(target_root),
        "condition_metadata": str(metadata_path),
    }


def _build_pose_noise_condition(
    row: BenchmarkInstanceGpuRow,
    condition_root: Path,
    translation_noise_mm: float,
    rotation_noise_deg: float,
    noise_sigma_frames: float,
    seed: int,
    overwrite: bool,
) -> dict[str, str]:
    target_root = _condition_instance_root(condition_root, base.POSE_NOISE_CONDITION, row)
    if overwrite:
        base._remove_path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    timestamps, positions, orientations = base._load_scanner_metadata(row.scanner_sequence)
    rng = np.random.default_rng(seed + sum(ord(ch) for ch in row.instance_name))
    translation_noise = base._smooth_noise(rng, len(timestamps), 3, translation_noise_mm, noise_sigma_frames)
    rotation_noise = base._smooth_noise(rng, len(timestamps), 3, np.deg2rad(rotation_noise_deg), noise_sigma_frames)
    delta_rotation = base._rotation_matrices_from_rotvec(rotation_noise)
    noisy_positions = positions + translation_noise
    noisy_orientations = np.einsum("nij,njk->nik", delta_rotation, orientations)

    noisy_scanner = target_root / "scanner_sequence.npz"
    frames = base._build_frames_array(row.clean_scanner_image_dir, np.arange(len(timestamps), dtype=np.int64))
    base._write_npz(
        noisy_scanner,
        frames=frames,
        timestamps=timestamps.astype(np.float64),
        positions=noisy_positions.astype(np.float64),
        orientations=noisy_orientations.astype(np.float64),
    )
    base._symlink_or_copy(row.monitor_stream, target_root / "monitor_stream.npz")
    base._symlink_or_copy(row.clean_monitor_image_dir, target_root / "image" / "monitor")
    base._symlink_or_copy(row.clean_scanner_image_dir, target_root / "image" / "scanner")

    metadata_path = target_root / "condition_metadata.json"
    base._write_metadata(
        metadata_path,
        {
            "condition": base.POSE_NOISE_CONDITION,
            "instance_name": row.instance_name,
            "source_scanner_sequence": str(row.scanner_sequence),
            "source_scanner_image_dir": str(row.clean_scanner_image_dir),
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
        "condition": base.POSE_NOISE_CONDITION,
        "reference_ply": str(row.reference_ply),
        "monitor_stream": str(target_root / "monitor_stream.npz"),
        "scanner_sequence": str(noisy_scanner),
        "phase_model_dir": str(row.phase_model_dir),
        "phase_summary": str(row.phase_summary),
        "condition_root": str(target_root),
        "condition_metadata": str(metadata_path),
    }


def _build_image_noise_condition(
    row: BenchmarkInstanceGpuRow,
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
    target_root = _condition_instance_root(condition_root, base.IMAGE_NOISE_CONDITION, row)
    if overwrite:
        base._remove_path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    timestamps, positions, orientations = base._load_scanner_metadata(row.scanner_sequence)
    frame_count = len(timestamps)
    first_frame = base._read_scanner_frame(row.clean_scanner_image_dir, 0)
    rng = np.random.default_rng(seed + 10007 + sum(ord(ch) for ch in row.instance_name))
    bias_noise = base._smooth_noise(rng, frame_count, 1, image_bias_std, noise_sigma_frames)[:, 0]
    gain_noise = 1.0 + base._smooth_noise(rng, frame_count, 1, image_gain_std, noise_sigma_frames)[:, 0]
    frame_memmap_path, noisy_frames = base._create_frame_memmap(frame_count, tuple(first_frame.shape), f"{row.instance_name}_image_noise_")

    scanner_png_dir = target_root / "image" / "scanner"
    if materialize_scanner_pngs:
        base._remove_path(scanner_png_dir)
        scanner_png_dir.mkdir(parents=True, exist_ok=True)
    else:
        base._remove_path(scanner_png_dir)

    try:
        for frame_index in range(frame_count):
            frame = base._read_scanner_frame(row.clean_scanner_image_dir, frame_index).astype(np.float32)
            frame += rng.normal(0.0, image_noise_std, size=frame.shape).astype(np.float32)
            frame = frame * np.float32(gain_noise[frame_index]) + np.float32(bias_noise[frame_index])
            if image_blur_sigma > 0.0:
                frame = base.gaussian_filter(frame, sigma=image_blur_sigma, mode="nearest")
            frame_uint8 = np.clip(np.rint(frame), 0.0, 255.0).astype(np.uint8)
            noisy_frames[frame_index] = frame_uint8
            if materialize_scanner_pngs:
                base.Image.fromarray(frame_uint8, mode="L").save(scanner_png_dir / f"scanner_{frame_index:04d}.png")

        noisy_frames.flush()
        noisy_scanner = target_root / "scanner_sequence.npz"
        base._write_npz(
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

    base._symlink_or_copy(row.monitor_stream, target_root / "monitor_stream.npz")
    base._symlink_or_copy(row.clean_monitor_image_dir, target_root / "image" / "monitor")

    metadata_path = target_root / "condition_metadata.json"
    base._write_metadata(
        metadata_path,
        {
            "condition": base.IMAGE_NOISE_CONDITION,
            "instance_name": row.instance_name,
            "source_scanner_sequence": str(row.scanner_sequence),
            "source_scanner_image_dir": str(row.clean_scanner_image_dir),
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
        "condition": base.IMAGE_NOISE_CONDITION,
        "reference_ply": str(row.reference_ply),
        "monitor_stream": str(target_root / "monitor_stream.npz"),
        "scanner_sequence": str(noisy_scanner),
        "phase_model_dir": str(row.phase_model_dir),
        "phase_summary": str(row.phase_summary),
        "condition_root": str(target_root),
        "condition_metadata": str(metadata_path),
    }


def _clean_manifest_row(row: BenchmarkInstanceGpuRow) -> dict[str, str]:
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
    parser = argparse.ArgumentParser(description="Generate grouped benchmark conditions for the GPU dataset pipeline")
    parser.add_argument("--conditions", nargs="+", default=["Sparse", "PoseNoise", "ImageNoise"], choices=["Sparse", "PoseNoise", "ImageNoise"])
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--condition-manifest", type=Path, default=DEFAULT_CONDITION_MANIFEST)
    parser.add_argument("--condition-root", type=Path, default=DEFAULT_CONDITION_ROOT)
    parser.add_argument("--instances", nargs="*", default=None)
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    parser.add_argument("--sparse-step", type=int, default=2)
    parser.add_argument("--sparse-offset-mode", choices=["zero", "hash"], default="hash")
    parser.add_argument("--translation-noise-mm", type=float, default=1.0)
    parser.add_argument("--rotation-noise-deg", type=float, default=1.5)
    parser.add_argument("--noise-sigma-frames", type=float, default=24.0)
    parser.add_argument("--image-noise-std", type=float, default=6.0)
    parser.add_argument("--image-bias-std", type=float, default=4.0)
    parser.add_argument("--image-gain-std", type=float, default=0.04)
    parser.add_argument("--image-blur-sigma", type=float, default=0.6)
    parser.add_argument("--materialize-image-noise-pngs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=20260331)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_rows = _load_manifest(args.source_manifest.expanduser().resolve())
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
        if base._count_scanner_frames(row.clean_scanner_image_dir) == 0:
            raise FileNotFoundError(f"No scanner PNGs found under {row.clean_scanner_image_dir}")

    manifest_rows = [_clean_manifest_row(row) for row in source_rows]
    for row in filtered_rows:
        if base.SPARSE_CONDITION in args.conditions:
            manifest_rows.append(
                _build_sparse_condition(
                    row=row,
                    condition_root=args.condition_root,
                    sparse_step=args.sparse_step,
                    sparse_offset_mode=args.sparse_offset_mode,
                    overwrite=args.overwrite,
                )
            )
            print(f"[BenchmarkConditionsGPU] Generated Sparse for {row.instance_name}")

        if base.POSE_NOISE_CONDITION in args.conditions:
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
            print(f"[BenchmarkConditionsGPU] Generated PoseNoise for {row.instance_name}")

        if base.IMAGE_NOISE_CONDITION in args.conditions:
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
            print(f"[BenchmarkConditionsGPU] Generated ImageNoise for {row.instance_name}")

    manifest_rows.sort(key=lambda item: (item["instance_name"], item["condition"]))
    _write_condition_manifest(args.condition_manifest.expanduser().resolve(), manifest_rows)
    print(f"[BenchmarkConditionsGPU] Wrote condition manifest: {args.condition_manifest}")
    print(f"[BenchmarkConditionsGPU] Condition root: {args.condition_root}")


if __name__ == "__main__":
    import numpy as np

    main()
