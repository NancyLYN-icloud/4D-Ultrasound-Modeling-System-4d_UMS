from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.regenerate_freehand_scanner_sequence as regen

TEST_SCANNER_PATH = ROOT / "data" / "test" / "scanner_sequence.npz"
RAW_MONITOR_PATH = ROOT / "data" / "raw" / "monitor_stream.npz"
REFERENCE_PLY = ROOT / "data" / "test" / "stomach.ply"


def build_segment(
    timestamps: np.ndarray,
    frame_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if timestamps.size == 0:
        return (
            np.empty((0, frame_shape[0], frame_shape[1]), dtype=np.float32),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3, 3), dtype=np.float64),
        )

    with contextlib.redirect_stdout(io.StringIO()):
        gastric_period = regen.detect_monitor_period(RAW_MONITOR_PATH)
    model = regen.load_reference_model(REFERENCE_PLY)

    frames = np.zeros((timestamps.shape[0], frame_shape[0], frame_shape[1]), dtype=np.float32)
    positions = np.zeros((timestamps.shape[0], 3), dtype=np.float64)
    orientations = np.zeros((timestamps.shape[0], 3, 3), dtype=np.float64)

    for index, timestamp in enumerate(timestamps):
        phase = float((timestamp % gastric_period) / gastric_period)
        sweep_position = regen.sweep_coordinate(float(timestamp), gastric_period)
        positions[index] = regen.world_centerline(model, sweep_position, phase)
        orientations[index] = regen.probe_orientation(model, sweep_position, phase, float(timestamp))
        polygon = regen.cross_section_polygon_mm(model, sweep_position, phase)
        frame = regen.rasterize_binary_polygon(polygon, frame_shape[0], regen.PIXEL_SPACING_MM)
        shift_x = int(round(2.0 * math.sin(2.0 * math.pi * timestamp / 39.0)))
        shift_y = int(round(2.0 * math.cos(2.0 * math.pi * timestamp / 35.0)))
        frames[index] = regen.translate_binary_frame(frame, shift_x=shift_x, shift_y=shift_y)

    return frames, positions, orientations


def build_extra_segment(
    timestamps: np.ndarray,
    target_duration: float,
    frame_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if timestamps.size < 2:
        raise ValueError("scanner_sequence.npz must contain at least two timestamps")

    dt = float(np.median(np.diff(timestamps)))
    extra_count = int(np.floor((target_duration - float(timestamps[-1])) / dt))
    if extra_count <= 0:
        return (
            np.empty((0, frame_shape[0], frame_shape[1]), dtype=np.float32),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3, 3), dtype=np.float64),
        )

    extra_timestamps = float(timestamps[-1]) + dt * np.arange(1, extra_count + 1, dtype=np.float64)

    return build_segment(extra_timestamps, frame_shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend data/test/scanner_sequence.npz to a longer duration")
    parser.add_argument("--target-duration", type=float, default=900.0, help="Target duration in seconds")
    parser.add_argument(
        "--rewrite-all",
        action="store_true",
        help="Rebuild all frames, positions, and orientations for the existing timeline before any extension",
    )
    args = parser.parse_args()

    with np.load(TEST_SCANNER_PATH) as data:
        frames = data["frames"].copy().astype(np.float32)
        timestamps = data["timestamps"].copy().astype(np.float64)
        positions = data["positions"].copy().astype(np.float64)
        orientations = data["orientations"].copy().astype(np.float64)

    if args.rewrite_all:
        frames, positions, orientations = build_segment(
            timestamps=timestamps,
            frame_shape=(frames.shape[1], frames.shape[2]),
        )

    if float(timestamps[-1]) >= args.target_duration - 1e-9 and not args.rewrite_all:
        print(f"No extension needed. Existing last timestamp: {float(timestamps[-1]):.6f}s")
        return

    dt = float(np.median(np.diff(timestamps)))
    frames_out = frames
    timestamps_out = timestamps
    positions_out = positions
    orientations_out = orientations

    if float(timestamps[-1]) < args.target_duration - 1e-9:
        extra_frames, extra_positions, extra_orientations = build_extra_segment(
            timestamps=timestamps,
            target_duration=args.target_duration,
            frame_shape=(frames.shape[1], frames.shape[2]),
        )
        extra_timestamps = float(timestamps[-1]) + dt * np.arange(1, extra_frames.shape[0] + 1, dtype=np.float64)
        frames_out = np.concatenate([frames_out, extra_frames], axis=0)
        timestamps_out = np.concatenate([timestamps_out, extra_timestamps], axis=0)
        positions_out = np.concatenate([positions_out, extra_positions], axis=0)
        orientations_out = np.concatenate([orientations_out, extra_orientations], axis=0)

    tmp_path = TEST_SCANNER_PATH.with_name(TEST_SCANNER_PATH.stem + ".tmp.npz")
    np.savez_compressed(
        tmp_path,
        frames=frames_out,
        timestamps=timestamps_out,
        positions=positions_out,
        orientations=orientations_out,
    )
    os.replace(tmp_path, TEST_SCANNER_PATH)

    print(f"Extended {TEST_SCANNER_PATH}")
    print(f"Old frames: {len(timestamps)}")
    print(f"New frames: {len(timestamps_out)}")
    print(f"Last timestamp: {float(timestamps_out[-1]):.6f}s")
    print(f"Median dt: {dt:.12f}s")
    print(f"Rewrite all: {args.rewrite_all}")


if __name__ == "__main__":
    main()