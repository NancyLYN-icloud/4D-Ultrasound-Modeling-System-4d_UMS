"""Generate a synthetic gastric antrum monitor stream and matching PNG frames.

The generated data mimics an ultrasound probe staying on a transverse antrum view
for multiple minutes. The signal intentionally contains mild cycle-to-cycle
variation in period length, peak timing, and contraction strength so that
multi-cycle phase standardization is meaningfully exercised.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import data_path
from src.stomach_instance_paths import default_reference_ply, list_reference_pointclouds, resolve_instance_paths


OUT_NPZ = data_path("benchmark", "monitor_stream.npz")
OUT_IMG_DIR = data_path("benchmark", "image", "monitor")

IMAGE_SIZE = 64
DURATION_SECONDS = 180.0
FRAME_RATE = 3.0
PERISTALSIS_PERIOD = 20.0
RNG_SEED = 20260311


@dataclass(frozen=True)
class MonitorSynthesisProfile:
    instance_name: str
    seed: int
    base_period: float
    period_variation: float
    peak_phase_center: float
    peak_phase_variation: float
    peak_width_center: float
    amplitude_center: float
    amplitude_variation: float
    shoulder_gain: float
    relaxation_gain: float
    early_time_center: float
    peak_time_center: float
    recovery_time_center: float
    quality_floor: float
    quality_variation: float
    quality_noise: float
    frame_drift_scale: float
    lumen_scale: float
    wall_scale: float
    shadow_gain: float
    speckle_scale: float


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value <= 1e-8:
        return np.zeros_like(values, dtype=np.float64)
    return (values - min_value) / (max_value - min_value)


def _read_reference_points(path: Path, max_points: int = 8192) -> np.ndarray:
    vertex_count = 0
    header_lines = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[2])
            if stripped == "end_header":
                break

    if vertex_count <= 0:
        raise ValueError(f"PLY file does not contain vertices: {path}")

    points = np.loadtxt(path, skiprows=header_lines, max_rows=vertex_count, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    points = np.asarray(points[:, :3], dtype=np.float32)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
    return points


def _profile_from_reference(reference_ply: Path | None, instance_name: str | None) -> MonitorSynthesisProfile:
    label = (instance_name or (reference_ply.stem if reference_ply is not None else "shared_monitor")).strip() or "shared_monitor"
    seed_material = label if reference_ply is None else f"{label}|{reference_ply.resolve()}"
    seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "little") ^ RNG_SEED

    if reference_ply is None or not reference_ply.exists():
        return MonitorSynthesisProfile(
            instance_name=label,
            seed=seed,
            base_period=PERISTALSIS_PERIOD,
            period_variation=0.11,
            peak_phase_center=0.58,
            peak_phase_variation=0.08,
            peak_width_center=0.12,
            amplitude_center=0.96,
            amplitude_variation=0.11,
            shoulder_gain=0.32,
            relaxation_gain=0.10,
            early_time_center=0.24,
            peak_time_center=0.58,
            recovery_time_center=0.86,
            quality_floor=0.56,
            quality_variation=0.24,
            quality_noise=0.035,
            frame_drift_scale=1.0,
            lumen_scale=1.0,
            wall_scale=1.0,
            shadow_gain=1.0,
            speckle_scale=1.0,
        )

    points = _read_reference_points(reference_ply)
    lower = points.min(axis=0).astype(np.float64)
    upper = points.max(axis=0).astype(np.float64)
    extents = np.sort(np.maximum(upper - lower, 1e-6))
    extent_norm = _normalize_vector(extents)
    covariance = np.cov((points - points.mean(axis=0)).T)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 1e-8))
    eigen_norm = _normalize_vector(eigenvalues)

    long_ratio = float(extents[-1] / extents[-2]) if extents[-2] > 1e-6 else 1.0
    flat_ratio = float(extents[0] / extents[-1]) if extents[-1] > 1e-6 else 1.0
    point_scale = float(np.clip(np.log10(max(len(points), 100)) / 4.0, 0.4, 1.2))

    return MonitorSynthesisProfile(
        instance_name=label,
        seed=seed,
        base_period=float(np.clip(PERISTALSIS_PERIOD * (0.92 + 0.10 * long_ratio + 0.05 * extent_norm[-1]), 17.0, 25.5)),
        period_variation=float(np.clip(0.08 + 0.05 * (1.0 - flat_ratio) + 0.02 * eigen_norm[-1], 0.06, 0.18)),
        peak_phase_center=float(np.clip(0.53 + 0.10 * extent_norm[1] - 0.05 * flat_ratio, 0.44, 0.72)),
        peak_phase_variation=float(np.clip(0.05 + 0.04 * eigen_norm[1], 0.04, 0.12)),
        peak_width_center=float(np.clip(0.10 + 0.04 * flat_ratio + 0.02 * extent_norm[0], 0.08, 0.18)),
        amplitude_center=float(np.clip(0.88 + 0.12 * extent_norm[-1] + 0.06 * point_scale, 0.78, 1.16)),
        amplitude_variation=float(np.clip(0.08 + 0.04 * (long_ratio - 1.0), 0.06, 0.16)),
        shoulder_gain=float(np.clip(0.26 + 0.10 * extent_norm[0], 0.20, 0.42)),
        relaxation_gain=float(np.clip(0.08 + 0.05 * (1.0 - flat_ratio), 0.06, 0.15)),
        early_time_center=float(np.clip(0.18 + 0.10 * flat_ratio, 0.14, 0.34)),
        peak_time_center=float(np.clip(0.50 + 0.08 * extent_norm[1] + 0.04 * (long_ratio - 1.0), 0.42, 0.70)),
        recovery_time_center=float(np.clip(0.78 + 0.10 * extent_norm[-1], 0.72, 0.93)),
        quality_floor=float(np.clip(0.48 + 0.16 * flat_ratio, 0.38, 0.72)),
        quality_variation=float(np.clip(0.18 + 0.10 * (1.0 - flat_ratio) + 0.04 * extent_norm[1], 0.12, 0.36)),
        quality_noise=float(np.clip(0.020 + 0.025 * (2.0 - point_scale), 0.015, 0.05)),
        frame_drift_scale=float(np.clip(0.85 + 0.25 * extent_norm[-1], 0.75, 1.25)),
        lumen_scale=float(np.clip(0.92 + 0.14 * extent_norm[0], 0.84, 1.12)),
        wall_scale=float(np.clip(0.92 + 0.14 * eigen_norm[1], 0.84, 1.15)),
        shadow_gain=float(np.clip(0.85 + 0.22 * (1.0 - flat_ratio), 0.75, 1.20)),
        speckle_scale=float(np.clip(0.90 + 0.18 * (2.0 - point_scale) + 0.08 * extent_norm[0], 0.82, 1.18)),
    )


def periodic_distance(phase: np.ndarray, center: float) -> np.ndarray:
    """Return wrapped phase distance on [0, 1)."""
    return ((phase - center + 0.5) % 1.0) - 0.5


def build_cycle_schedule(total_duration: float, rng: np.random.Generator, profile: MonitorSynthesisProfile) -> dict[str, np.ndarray]:
    """Create a deterministic but non-stationary cycle schedule."""
    durations: list[float] = []
    start_times: list[float] = []
    peak_phases: list[float] = []
    peak_widths: list[float] = []
    shoulder_phases: list[float] = []
    shoulder_widths: list[float] = []
    amplitudes: list[float] = []
    early_time_fracs: list[float] = []
    peak_time_fracs: list[float] = []
    recovery_time_fracs: list[float] = []
    quality_floors: list[float] = []
    quality_peaks: list[float] = []

    elapsed = 0.0
    cycle_index = 0
    while elapsed < total_duration + PERISTALSIS_PERIOD:
        start_times.append(elapsed)

        duration = profile.base_period * (
            1.0
            + profile.period_variation * np.sin(2.0 * np.pi * cycle_index / 4.0 + 0.2)
            + 0.03 * np.sin(2.0 * np.pi * cycle_index / 7.0 + 1.1)
            + float(rng.normal(0.0, 0.03))
        )
        duration = float(np.clip(duration, 15.5, 26.5))

        peak_phase = (
            profile.peak_phase_center
            + profile.peak_phase_variation * np.sin(2.0 * np.pi * cycle_index / 5.0 + 0.7)
            + float(rng.normal(0.0, 0.02))
        )
        peak_phase = float(np.clip(peak_phase, 0.40, 0.76))

        peak_width = float(np.clip(profile.peak_width_center + float(rng.normal(0.0, 0.015)), 0.08, 0.18))
        shoulder_phase = float(np.clip(peak_phase - 0.22 + float(rng.normal(0.0, 0.025)), 0.18, peak_phase - 0.06))
        shoulder_width = float(np.clip(0.20 + float(rng.normal(0.0, 0.03)), 0.14, 0.28))
        amplitude = float(
            np.clip(
                profile.amplitude_center
                + profile.amplitude_variation * np.sin(2.0 * np.pi * cycle_index / 3.0 + 0.5)
                + float(rng.normal(0.0, 0.04)),
                0.72,
                1.22,
            )
        )
        early_time = float(np.clip(profile.early_time_center + float(rng.normal(0.0, 0.02)), 0.12, peak_phase - 0.10))
        peak_time = float(np.clip(profile.peak_time_center + float(rng.normal(0.0, 0.03)), early_time + 0.08, 0.76))
        recovery_time = float(np.clip(profile.recovery_time_center + float(rng.normal(0.0, 0.03)), peak_time + 0.10, 0.96))
        quality_floor = float(np.clip(profile.quality_floor + float(rng.normal(0.0, 0.03)), 0.28, 0.85))
        quality_peak = float(np.clip(quality_floor + profile.quality_variation + float(rng.normal(0.0, 0.03)), quality_floor + 0.08, 1.0))

        durations.append(duration)
        peak_phases.append(peak_phase)
        peak_widths.append(peak_width)
        shoulder_phases.append(shoulder_phase)
        shoulder_widths.append(shoulder_width)
        amplitudes.append(amplitude)
        early_time_fracs.append(early_time)
        peak_time_fracs.append(peak_time)
        recovery_time_fracs.append(recovery_time)
        quality_floors.append(quality_floor)
        quality_peaks.append(quality_peak)

        elapsed += duration
        cycle_index += 1

    start_arr = np.asarray(start_times, dtype=np.float32)
    duration_arr = np.asarray(durations, dtype=np.float32)
    return {
        "start_times": start_arr,
        "end_times": start_arr + duration_arr,
        "durations": duration_arr,
        "peak_phases": np.asarray(peak_phases, dtype=np.float32),
        "peak_widths": np.asarray(peak_widths, dtype=np.float32),
        "shoulder_phases": np.asarray(shoulder_phases, dtype=np.float32),
        "shoulder_widths": np.asarray(shoulder_widths, dtype=np.float32),
        "amplitudes": np.asarray(amplitudes, dtype=np.float32),
        "early_time_fracs": np.asarray(early_time_fracs, dtype=np.float32),
        "peak_time_fracs": np.asarray(peak_time_fracs, dtype=np.float32),
        "recovery_time_fracs": np.asarray(recovery_time_fracs, dtype=np.float32),
        "quality_floors": np.asarray(quality_floors, dtype=np.float32),
        "quality_peaks": np.asarray(quality_peaks, dtype=np.float32),
    }


def warped_phase_trace(timestamps: np.ndarray, schedule: dict[str, np.ndarray]) -> np.ndarray:
    cycle_indices = np.searchsorted(schedule["end_times"], timestamps, side="right")
    cycle_indices = np.clip(cycle_indices, 0, len(schedule["durations"]) - 1)
    local_time = timestamps - schedule["start_times"][cycle_indices]
    normalized_time = np.clip(local_time / schedule["durations"][cycle_indices], 0.0, 1.0)

    phase = np.empty_like(normalized_time, dtype=np.float32)
    for cycle_index in range(len(schedule["durations"])):
        mask = cycle_indices == cycle_index
        if not np.any(mask):
            continue
        time_knots = np.array(
            [
                0.0,
                float(schedule["early_time_fracs"][cycle_index]),
                float(schedule["peak_time_fracs"][cycle_index]),
                float(schedule["recovery_time_fracs"][cycle_index]),
                1.0,
            ],
            dtype=np.float32,
        )
        phase_knots = np.array(
            [
                0.0,
                max(float(schedule["peak_phases"][cycle_index]) - 0.24, 0.10),
                float(schedule["peak_phases"][cycle_index]),
                0.86,
                1.0,
            ],
            dtype=np.float32,
        )
        phase[mask] = np.interp(normalized_time[mask], time_knots, phase_knots).astype(np.float32)
    return phase


def contraction_waveform(phase: np.ndarray, schedule: dict[str, np.ndarray], cycle_indices: np.ndarray, profile: MonitorSynthesisProfile) -> np.ndarray:
    """Model a smooth gastric contraction pulse with per-cycle variability and non-uniform phase speed."""
    contraction = schedule["amplitudes"][cycle_indices] * np.exp(
        -0.5 * (periodic_distance(phase, schedule["peak_phases"][cycle_indices]) / schedule["peak_widths"][cycle_indices]) ** 2
    )
    shoulder = profile.shoulder_gain * np.exp(
        -0.5
        * (
            periodic_distance(phase, schedule["shoulder_phases"][cycle_indices])
            / schedule["shoulder_widths"][cycle_indices]
        )
        ** 2
    )
    relaxation = profile.relaxation_gain * np.exp(-0.5 * (periodic_distance(phase, 0.90) / 0.16) ** 2)
    waveform = contraction + shoulder + relaxation
    waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min())
    return waveform.astype(np.float32)


def build_quality_trace(
    timestamps: np.ndarray,
    phase: np.ndarray,
    schedule: dict[str, np.ndarray],
    cycle_indices: np.ndarray,
    rng: np.random.Generator,
    profile: MonitorSynthesisProfile,
) -> np.ndarray:
    cycle_floor = schedule["quality_floors"][cycle_indices]
    cycle_peak = schedule["quality_peaks"][cycle_indices]
    wave_gate = np.exp(-0.5 * (periodic_distance(phase, schedule["peak_phases"][cycle_indices]) / 0.18) ** 2)
    slow_drift = 0.08 * np.sin(2.0 * np.pi * timestamps / 57.0 + 0.4) + 0.05 * np.cos(2.0 * np.pi * timestamps / 41.0 + 1.2)
    dropout_gate = np.exp(-0.5 * (periodic_distance(phase, schedule["shoulder_phases"][cycle_indices]) / 0.11) ** 2)
    dropout = (0.10 + 0.10 * rng.random(len(timestamps), dtype=np.float32)) * dropout_gate
    jitter = rng.normal(0.0, profile.quality_noise, size=len(timestamps)).astype(np.float32)

    quality = cycle_floor + (cycle_peak - cycle_floor) * (0.40 + 0.60 * wave_gate)
    quality += slow_drift.astype(np.float32)
    quality -= dropout.astype(np.float32)
    quality += jitter
    return np.clip(quality, 0.22, 1.0).astype(np.float32)


def make_coordinate_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    return np.meshgrid(axis, axis)


def synthesize_frame(
    xx: np.ndarray,
    yy: np.ndarray,
    contraction_level: float,
    quality_level: float,
    timestamp: float,
    rng: np.random.Generator,
    profile: MonitorSynthesisProfile,
) -> np.ndarray:
    """Create one synthetic transverse antrum ultrasound frame."""
    drift_x = 0.03 * profile.frame_drift_scale * np.sin(2.0 * np.pi * timestamp / 45.0)
    drift_y = 0.02 * profile.frame_drift_scale * np.cos(2.0 * np.pi * timestamp / 32.0)
    x = xx - drift_x
    y = yy - drift_y

    radius = np.sqrt((x / 0.92) ** 2 + (y / 0.80) ** 2)
    angle = np.arctan2(y, x)

    lumen_radius = profile.lumen_scale * (0.42 - 0.12 * contraction_level)
    wall_thickness = profile.wall_scale * (0.11 + 0.04 * contraction_level)
    muscularis_radius = lumen_radius + wall_thickness

    quality_soft = float(np.clip(quality_level, 0.0, 1.0))
    background = 0.08 + 0.04 * (yy + 1.0) + 0.03 * (1.0 - quality_soft)
    probe_gain = np.exp(-((yy + 0.8) / 0.65) ** 2) * (0.07 + 0.05 * quality_soft)

    antrum_lumen = np.exp(-((radius / max(lumen_radius, 0.12)) ** 6))
    mucosa_ring = np.exp(-((radius - lumen_radius) / 0.055) ** 2)
    serosa_ring = np.exp(-((radius - muscularis_radius) / 0.07) ** 2)
    fold_texture = np.cos(4.0 * angle + timestamp * 0.35) * np.exp(-((radius - muscularis_radius) / 0.12) ** 2)
    distal_shadow = np.exp(-((x + 0.10) / 0.38) ** 2 - ((y - 0.55) / 0.28) ** 2)
    dropout_shadow = np.exp(-((x - 0.28) / 0.24) ** 2 - ((y + 0.12) / 0.18) ** 2)

    frame = background + probe_gain
    frame -= (0.10 + 0.04 * quality_soft) * antrum_lumen
    frame += (0.26 + 0.22 * quality_soft) * mucosa_ring
    frame += (0.14 + 0.10 * quality_soft) * serosa_ring
    frame += (0.02 + 0.05 * quality_soft) * fold_texture
    frame -= 0.06 * profile.shadow_gain * contraction_level * distal_shadow
    frame -= (1.0 - quality_soft) * 0.18 * dropout_shadow

    speckle = rng.normal(0.0, profile.speckle_scale * (0.015 + 0.04 * (1.0 - quality_soft)), size=frame.shape).astype(np.float32)
    grain = rng.normal(1.0, 0.03 + 0.08 * (1.0 - quality_soft), size=frame.shape).astype(np.float32)
    frame = frame * grain + speckle

    vignette = 1.0 - 0.22 * np.clip(np.sqrt(xx**2 + yy**2) - 0.6, 0.0, 1.0)
    frame *= vignette

    frame = np.clip(frame, 0.0, 1.0)
    return frame.astype(np.float32)


def save_png_sequence(frames: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("monitor_*.png"):
        path.unlink()

    for index, frame in enumerate(frames):
        image = Image.fromarray(np.round(frame * 255.0).astype(np.uint8), mode="L")
        image.save(output_dir / f"monitor_{index:04d}.png")


def generate_monitor_dataset(
    output_npz: Path,
    output_img_dir: Path,
    *,
    instance_name: str | None = None,
    reference_ply: Path | None = None,
) -> None:
    profile = _profile_from_reference(reference_ply=reference_ply, instance_name=instance_name)
    rng = np.random.default_rng(profile.seed)
    total_frames = int(DURATION_SECONDS * FRAME_RATE)
    timestamps = (np.arange(total_frames, dtype=np.float32) / np.float32(FRAME_RATE)).astype(np.float32)
    schedule = build_cycle_schedule(DURATION_SECONDS, rng, profile)
    cycle_indices = np.searchsorted(schedule["end_times"], timestamps, side="right")
    cycle_indices = np.clip(cycle_indices, 0, len(schedule["durations"]) - 1)
    phase = warped_phase_trace(timestamps, schedule)
    contraction = contraction_waveform(phase, schedule, cycle_indices, profile)
    quality = build_quality_trace(timestamps, phase, schedule, cycle_indices, rng, profile)
    xx, yy = make_coordinate_grid(IMAGE_SIZE)

    frames = np.stack(
        [
            synthesize_frame(xx, yy, float(level), float(quality_level), float(ts), rng, profile)
            for ts, level, quality_level in zip(timestamps, contraction, quality)
        ],
        axis=0,
    ).astype(np.float32)

    slow_baseline = 0.008 * np.sin(2.0 * np.pi * timestamps / 95.0 + 0.3) + 0.006 * np.cos(2.0 * np.pi * timestamps / 63.0 + 0.8)
    motion_noise = rng.normal(0.0, 0.004 + 0.010 * (1.0 - quality), size=len(timestamps)).astype(np.float32)
    feature_trace = (0.02 + slow_baseline + (0.10 + 0.07 * quality) * contraction + motion_noise).astype(np.float32)
    feature_trace = np.clip(feature_trace, 0.0, None)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        frames=frames,
        timestamps=timestamps,
        feature_trace=feature_trace,
        latent_phase=phase.astype(np.float32),
        contraction_trace=contraction.astype(np.float32),
        quality_trace=quality.astype(np.float32),
        cycle_start_times=schedule["start_times"],
        cycle_durations=schedule["durations"],
        cycle_peak_phases=schedule["peak_phases"],
        cycle_peak_time_fracs=schedule["peak_time_fracs"],
        cycle_quality_floors=schedule["quality_floors"],
        cycle_quality_peaks=schedule["quality_peaks"],
    )
    save_png_sequence(frames, output_img_dir)

    print(f"Wrote {output_npz} with shape {frames.shape}")
    print(f"Saved {len(frames)} PNG files to {output_img_dir}")
    print(f"Instance profile: {profile.instance_name} seed={profile.seed}")
    print(
        "Feature trace range:",
        f"{float(feature_trace.min()):.4f} .. {float(feature_trace.max()):.4f}",
    )
    print(
        "Quality trace range:",
        f"{float(quality.min()):.3f} .. {float(quality.max()):.3f}",
    )
    print(
        "Cycle count over duration:",
        f"{DURATION_SECONDS / profile.base_period:.1f}",
    )
    print(
        "Scheduled cycles:",
        f"{len(schedule['durations'])} total, mean duration {float(np.mean(schedule['durations'])):.2f}s",
    )
    print(
        "Peak phase range:",
        f"{float(schedule['peak_phases'].min()):.3f} .. {float(schedule['peak_phases'].max()):.3f}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic monitor_stream.npz and monitor PNGs")
    parser.add_argument("--instance-name", type=str, default=None, help="Named stomach instance under stomach_pcd")
    parser.add_argument("--reference-ply", type=str, default=None, help="Explicit reference stomach point cloud path")
    parser.add_argument("--batch-all-references", action="store_true", help="Generate monitor data for all point clouds under stomach_pcd")
    args = parser.parse_args()

    if args.batch_all_references:
        reference_paths = list_reference_pointclouds()
        if not reference_paths:
            raise FileNotFoundError("No reference point clouds found under stomach_pcd")
        for reference_path in reference_paths:
            instance_paths = resolve_instance_paths(instance_name=reference_path.stem, reference_ply=reference_path)
            generate_monitor_dataset(
                output_npz=instance_paths.monitor_stream,
                output_img_dir=instance_paths.test_root / "image" / "monitor",
                instance_name=instance_paths.name,
                reference_ply=reference_path,
            )
        return

    if args.instance_name is not None or args.reference_ply is not None:
        instance_paths = resolve_instance_paths(
            instance_name=args.instance_name,
            reference_ply=Path(args.reference_ply).expanduser().resolve() if args.reference_ply else None,
        )
        generate_monitor_dataset(
            output_npz=instance_paths.monitor_stream,
            output_img_dir=instance_paths.test_root / "image" / "monitor",
            instance_name=instance_paths.name,
            reference_ply=instance_paths.reference_ply,
        )
        return

    reference_paths = list_reference_pointclouds()
    fallback_reference = default_reference_ply() if reference_paths else None
    generate_monitor_dataset(
        output_npz=OUT_NPZ,
        output_img_dir=OUT_IMG_DIR,
        instance_name=fallback_reference.stem if fallback_reference is not None else None,
        reference_ply=fallback_reference,
    )


if __name__ == "__main__":
    main()
