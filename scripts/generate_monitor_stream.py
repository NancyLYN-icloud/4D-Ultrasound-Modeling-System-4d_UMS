"""Generate a synthetic gastric antrum monitor stream and matching PNG frames.

The generated data mimics an ultrasound probe staying on a transverse antrum view
for multiple minutes. The signal intentionally contains mild cycle-to-cycle
variation in period length, peak timing, and contraction strength so that
multi-cycle phase standardization is meaningfully exercised.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import data_path
from src.stomach_instance_paths import list_reference_pointclouds, resolve_instance_paths


OUT_NPZ = data_path("benchmark", "monitor_stream.npz")
OUT_IMG_DIR = data_path("benchmark", "image", "monitor")

IMAGE_SIZE = 64
DURATION_SECONDS = 180.0
FRAME_RATE = 3.0
PERISTALSIS_PERIOD = 20.0
RNG_SEED = 20260311


def periodic_distance(phase: np.ndarray, center: float) -> np.ndarray:
    """Return wrapped phase distance on [0, 1)."""
    return ((phase - center + 0.5) % 1.0) - 0.5


def build_cycle_schedule(total_duration: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Create a deterministic but non-stationary cycle schedule."""
    durations: list[float] = []
    start_times: list[float] = []
    peak_phases: list[float] = []
    peak_widths: list[float] = []
    shoulder_phases: list[float] = []
    shoulder_widths: list[float] = []
    amplitudes: list[float] = []

    elapsed = 0.0
    cycle_index = 0
    while elapsed < total_duration + PERISTALSIS_PERIOD:
        start_times.append(elapsed)

        duration = PERISTALSIS_PERIOD * (
            1.0
            + 0.10 * np.sin(2.0 * np.pi * cycle_index / 4.0 + 0.2)
            + float(rng.normal(0.0, 0.035))
        )
        duration = float(np.clip(duration, 16.5, 24.5))

        peak_phase = 0.58 + 0.08 * np.sin(2.0 * np.pi * cycle_index / 5.0 + 0.7) + float(rng.normal(0.0, 0.02))
        peak_phase = float(np.clip(peak_phase, 0.42, 0.74))

        peak_width = float(np.clip(0.12 + float(rng.normal(0.0, 0.015)), 0.08, 0.17))
        shoulder_phase = float(np.clip(peak_phase - 0.22 + float(rng.normal(0.0, 0.025)), 0.18, peak_phase - 0.06))
        shoulder_width = float(np.clip(0.20 + float(rng.normal(0.0, 0.03)), 0.14, 0.28))
        amplitude = float(
            np.clip(
                0.95 + 0.12 * np.sin(2.0 * np.pi * cycle_index / 3.0 + 0.5) + float(rng.normal(0.0, 0.04)),
                0.78,
                1.18,
            )
        )

        durations.append(duration)
        peak_phases.append(peak_phase)
        peak_widths.append(peak_width)
        shoulder_phases.append(shoulder_phase)
        shoulder_widths.append(shoulder_width)
        amplitudes.append(amplitude)

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
    }


def contraction_waveform(timestamps: np.ndarray, schedule: dict[str, np.ndarray]) -> np.ndarray:
    """Model a smooth gastric contraction pulse with per-cycle variability."""
    cycle_indices = np.searchsorted(schedule["end_times"], timestamps, side="right")
    cycle_indices = np.clip(cycle_indices, 0, len(schedule["durations"]) - 1)
    local_time = timestamps - schedule["start_times"][cycle_indices]
    phase = np.clip(local_time / schedule["durations"][cycle_indices], 0.0, 1.0)

    contraction = schedule["amplitudes"][cycle_indices] * np.exp(
        -0.5 * (periodic_distance(phase, schedule["peak_phases"][cycle_indices]) / schedule["peak_widths"][cycle_indices]) ** 2
    )
    shoulder = 0.32 * np.exp(
        -0.5
        * (
            periodic_distance(phase, schedule["shoulder_phases"][cycle_indices])
            / schedule["shoulder_widths"][cycle_indices]
        )
        ** 2
    )
    relaxation = 0.10 * np.exp(-0.5 * (periodic_distance(phase, 0.90) / 0.16) ** 2)
    waveform = contraction + shoulder + relaxation
    waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min())
    return waveform.astype(np.float32)


def make_coordinate_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    return np.meshgrid(axis, axis)


def synthesize_frame(
    xx: np.ndarray,
    yy: np.ndarray,
    contraction_level: float,
    timestamp: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create one synthetic transverse antrum ultrasound frame."""
    drift_x = 0.03 * np.sin(2.0 * np.pi * timestamp / 45.0)
    drift_y = 0.02 * np.cos(2.0 * np.pi * timestamp / 32.0)
    x = xx - drift_x
    y = yy - drift_y

    radius = np.sqrt((x / 0.92) ** 2 + (y / 0.80) ** 2)
    angle = np.arctan2(y, x)

    lumen_radius = 0.42 - 0.12 * contraction_level
    wall_thickness = 0.11 + 0.04 * contraction_level
    muscularis_radius = lumen_radius + wall_thickness

    background = 0.08 + 0.04 * (yy + 1.0)
    probe_gain = np.exp(-((yy + 0.8) / 0.65) ** 2) * 0.10

    antrum_lumen = np.exp(-((radius / max(lumen_radius, 0.12)) ** 6))
    mucosa_ring = np.exp(-((radius - lumen_radius) / 0.055) ** 2)
    serosa_ring = np.exp(-((radius - muscularis_radius) / 0.07) ** 2)
    fold_texture = np.cos(4.0 * angle + timestamp * 0.35) * np.exp(-((radius - muscularis_radius) / 0.12) ** 2)
    distal_shadow = np.exp(-((x + 0.10) / 0.38) ** 2 - ((y - 0.55) / 0.28) ** 2)

    frame = background + probe_gain
    frame -= 0.12 * antrum_lumen
    frame += 0.42 * mucosa_ring
    frame += 0.22 * serosa_ring
    frame += 0.05 * fold_texture
    frame -= 0.06 * contraction_level * distal_shadow

    speckle = rng.normal(0.0, 0.035, size=frame.shape).astype(np.float32)
    grain = rng.normal(1.0, 0.08, size=frame.shape).astype(np.float32)
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


def generate_monitor_dataset(output_npz: Path, output_img_dir: Path) -> None:
    rng = np.random.default_rng(RNG_SEED)
    total_frames = int(DURATION_SECONDS * FRAME_RATE)
    timestamps = (np.arange(total_frames, dtype=np.float32) / np.float32(FRAME_RATE)).astype(np.float32)
    schedule = build_cycle_schedule(DURATION_SECONDS, rng)
    contraction = contraction_waveform(timestamps, schedule)
    xx, yy = make_coordinate_grid(IMAGE_SIZE)

    frames = np.stack(
        [synthesize_frame(xx, yy, float(level), float(ts), rng) for ts, level in zip(timestamps, contraction)],
        axis=0,
    ).astype(np.float32)

    slow_baseline = 0.008 * np.sin(2.0 * np.pi * timestamps / 95.0 + 0.3)
    feature_trace = (0.03 + slow_baseline + 0.15 * contraction).astype(np.float32)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        frames=frames,
        timestamps=timestamps,
        feature_trace=feature_trace,
    )
    save_png_sequence(frames, output_img_dir)

    print(f"Wrote {output_npz} with shape {frames.shape}")
    print(f"Saved {len(frames)} PNG files to {output_img_dir}")
    print(
        "Feature trace range:",
        f"{float(feature_trace.min()):.4f} .. {float(feature_trace.max()):.4f}",
    )
    print(
        "Cycle count over duration:",
        f"{DURATION_SECONDS / PERISTALSIS_PERIOD:.1f}",
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
        )
        return

    generate_monitor_dataset(output_npz=OUT_NPZ, output_img_dir=OUT_IMG_DIR)


if __name__ == "__main__":
    main()
