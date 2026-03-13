from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.config import FrameFeature, PipelineConfig
from src.paths import data_path
from src.preprocessing.phase_detection import PhaseDetector


OUT_DIR = data_path("test")
RAW_OUT_DIR = data_path("raw")
SCANNER_IMG_DIR = OUT_DIR / "image" / "scanner"
REFERENCE_PLY = OUT_DIR / "stomach.ply"
MONITOR_STREAM = RAW_OUT_DIR / "monitor_stream.npz"

SCANNER_DURATION = 240.0
SCANNER_FRAMES = 720
SWEEP_PERIOD = 5.7
SCANNER_SIZE = 512
SEED = 20260309
PIXEL_SPACING_MM = 0.42
PROFILE_BINS = 96


@dataclass
class GastricReferenceModel:
	world_center: np.ndarray
	world_basis: np.ndarray
	axis_coord: np.ndarray
	center_y: np.ndarray
	center_z: np.ndarray
	radius_y: np.ndarray
	radius_z: np.ndarray
	centerline_canonical: np.ndarray


def triangle_wave(x: np.ndarray) -> np.ndarray:
	frac = x - np.floor(x)
	return 1.0 - np.abs(2.0 * frac - 1.0)


def detect_monitor_period(monitor_stream: Path) -> float:
	if not monitor_stream.exists():
		raise FileNotFoundError(f"Monitor stream not found: {monitor_stream}")
	data = np.load(monitor_stream)
	timestamps = data["timestamps"]
	feature_trace = data["feature_trace"]
	features = [
		FrameFeature(timestamp=float(ts), value=float(value))
		for ts, value in zip(timestamps, feature_trace)
	]
	cycles = PhaseDetector(PipelineConfig().phase_detection).detect_cycles(features)
	if not cycles:
		raise RuntimeError("No gastric cycles detected from monitor stream")
	durations = np.asarray([cycle.duration for cycle in cycles], dtype=np.float64)
	return float(np.mean(durations))


def save_png(image: np.ndarray, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(path)


def _smooth_1d(values: np.ndarray, passes: int = 3) -> np.ndarray:
	kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64)
	kernel /= kernel.sum()
	out = values.astype(np.float64)
	for _ in range(passes):
		padded = np.pad(out, (2, 2), mode="edge")
		out = np.convolve(padded, kernel, mode="valid")
	return out


def _read_ply_points(path: Path) -> np.ndarray:
	pts = []
	with path.open("r", encoding="utf-8", errors="ignore") as f:
		header = True
		for line in f:
			if header:
				if line.strip() == "end_header":
					header = False
				continue
			parts = line.strip().split()
			if len(parts) < 3:
				continue
			try:
				pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
			except ValueError:
				continue
	points = np.asarray(pts, dtype=np.float64)
	if points.ndim != 2 or points.shape[0] < 100:
		raise ValueError(f"Invalid or too small PLY point cloud: {path}")
	return points


def _pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	center = points.mean(axis=0)
	centered = points - center
	cov = np.cov(centered.T)
	vals, vecs = np.linalg.eigh(cov)
	order = np.argsort(vals)[::-1]
	basis = vecs[:, order]
	if np.linalg.det(basis) < 0:
		basis[:, -1] *= -1.0
	return center, basis, centered @ basis


def load_reference_model(path: Path) -> GastricReferenceModel:
	points = _read_ply_points(path)
	center, basis, canonical = _pca_basis(points)

	axis_coord = canonical[:, 0]
	transverse_y = canonical[:, 1]
	transverse_z = canonical[:, 2]

	x_min = float(axis_coord.min())
	x_max = float(axis_coord.max())
	bins = np.linspace(x_min, x_max, PROFILE_BINS + 1)
	centers = 0.5 * (bins[:-1] + bins[1:])
	cy = np.zeros(PROFILE_BINS, dtype=np.float64)
	cz = np.zeros(PROFILE_BINS, dtype=np.float64)
	ry = np.zeros(PROFILE_BINS, dtype=np.float64)
	rz = np.zeros(PROFILE_BINS, dtype=np.float64)

	bin_index = np.digitize(axis_coord, bins) - 1
	for idx in range(PROFILE_BINS):
		mask = bin_index == idx
		if mask.sum() < 24:
			left = max(0, idx - 1)
			right = min(PROFILE_BINS - 1, idx + 1)
			mask = (bin_index >= left) & (bin_index <= right)
		local_y = transverse_y[mask]
		local_z = transverse_z[mask]
		cy[idx] = float(local_y.mean())
		cz[idx] = float(local_z.mean())
		ry[idx] = float(np.percentile(np.abs(local_y - cy[idx]), 92))
		rz[idx] = float(np.percentile(np.abs(local_z - cz[idx]), 92))

	ry = np.maximum(_smooth_1d(ry), 8.0)
	rz = np.maximum(_smooth_1d(rz), 6.0)
	cy = _smooth_1d(cy)
	cz = _smooth_1d(cz)

	area_head = float(np.mean(ry[:8] * rz[:8]))
	area_tail = float(np.mean(ry[-8:] * rz[-8:]))
	if area_tail > area_head:
		centers = -centers[::-1]
		cy = cy[::-1]
		cz = cz[::-1]
		ry = ry[::-1]
		rz = rz[::-1]
		basis[:, 0] *= -1.0

	centerline_canonical = np.column_stack([centers, cy, cz])
	return GastricReferenceModel(
		world_center=center,
		world_basis=basis,
		axis_coord=centers,
		center_y=cy,
		center_z=cz,
		radius_y=ry,
		radius_z=rz,
		centerline_canonical=centerline_canonical,
	)


def _interp_profile(model: GastricReferenceModel, s: float) -> tuple[float, float, float, float, float]:
	s = float(np.clip(s, 0.0, 1.0))
	x = np.interp(s, np.linspace(0.0, 1.0, model.axis_coord.size), model.axis_coord)
	cy = np.interp(s, np.linspace(0.0, 1.0, model.center_y.size), model.center_y)
	cz = np.interp(s, np.linspace(0.0, 1.0, model.center_z.size), model.center_z)
	ry = np.interp(s, np.linspace(0.0, 1.0, model.radius_y.size), model.radius_y)
	rz = np.interp(s, np.linspace(0.0, 1.0, model.radius_z.size), model.radius_z)
	return float(x), float(cy), float(cz), float(ry), float(rz)


def _wrapped_gaussian(x: float, center: float, sigma: float) -> float:
	deltas = (x - center, x - center - 1.0, x - center + 1.0)
	return max(math.exp(-0.5 * (delta / sigma) ** 2) for delta in deltas)


def peristaltic_state(s: float, phase: float) -> tuple[float, float, float]:
	lead_center = 0.14 + 0.72 * phase
	trail_center = (lead_center - 0.18) % 1.0
	lead = _wrapped_gaussian(s, lead_center, 0.070)
	trail = _wrapped_gaussian(s, trail_center, 0.115)
	distal_gain = 0.30 + 0.95 * (s ** 1.3)
	contraction = 0.18 * distal_gain * lead + 0.05 * distal_gain * trail
	axial_shift = 1.8 * lead - 0.6 * trail
	roll_bias = 0.16 * lead - 0.05 * trail
	return contraction, axial_shift, roll_bias


def canonical_centerline(model: GastricReferenceModel, s: float, phase: float) -> np.ndarray:
	x, cy, cz, _, _ = _interp_profile(model, s)
	contraction, axial_shift, roll_bias = peristaltic_state(s, phase)
	body_pull = 1.4 * contraction * math.exp(-((s - 0.56) / 0.24) ** 2)
	return np.array([
		x + axial_shift,
		cy + 2.2 * roll_bias,
		cz - body_pull,
	], dtype=np.float64)


def world_centerline(model: GastricReferenceModel, s: float, phase: float) -> np.ndarray:
	return model.world_center + model.world_basis @ canonical_centerline(model, s, phase)


def probe_orientation(model: GastricReferenceModel, s: float, phase: float, t: float) -> np.ndarray:
	ds = 1.0 / (model.axis_coord.size - 1)
	p0 = world_centerline(model, max(0.0, s - ds), phase)
	p1 = world_centerline(model, min(1.0, s + ds), phase)
	normal = p1 - p0
	normal /= np.linalg.norm(normal) + 1e-8

	ref_up = model.world_basis[:, 2].copy()
	if abs(float(np.dot(ref_up, normal))) > 0.90:
		ref_up = model.world_basis[:, 1].copy()

	axis_x = np.cross(ref_up, normal)
	axis_x /= np.linalg.norm(axis_x) + 1e-8
	axis_y = np.cross(normal, axis_x)
	axis_y /= np.linalg.norm(axis_y) + 1e-8

	contraction, _, roll_bias = peristaltic_state(s, phase)
	roll = math.radians(7.0 * math.sin(2.0 * math.pi * t / 23.0) + 8.0 * roll_bias)
	pitch = math.radians(5.5 * math.sin(2.0 * math.pi * t / 17.0 + 0.35) + 3.0 * contraction)
	yaw = math.radians(6.5 * math.cos(2.0 * math.pi * t / 29.0 - 0.2) + 2.5 * math.sin(2.0 * math.pi * s))

	rot_roll = np.array([
		[math.cos(roll), -math.sin(roll), 0.0],
		[math.sin(roll), math.cos(roll), 0.0],
		[0.0, 0.0, 1.0],
	], dtype=np.float64)
	rot_pitch = np.array([
		[1.0, 0.0, 0.0],
		[0.0, math.cos(pitch), -math.sin(pitch)],
		[0.0, math.sin(pitch), math.cos(pitch)],
	], dtype=np.float64)
	rot_yaw = np.array([
		[math.cos(yaw), 0.0, math.sin(yaw)],
		[0.0, 1.0, 0.0],
		[-math.sin(yaw), 0.0, math.cos(yaw)],
	], dtype=np.float64)

	return np.column_stack([axis_x, axis_y, normal]) @ rot_roll @ rot_pitch @ rot_yaw


def cross_section_polygon_mm(model: GastricReferenceModel, s: float, phase: float, n_theta: int = 240) -> np.ndarray:
	_, _, _, base_ry, base_rz = _interp_profile(model, s)
	contraction, _, roll_bias = peristaltic_state(s, phase)

	a = max(8.0, base_ry * (1.0 - 0.72 * contraction))
	b = max(7.0, base_rz * (1.0 - 0.58 * contraction))
	body_asym = 0.10 * math.exp(-((s - 0.42) / 0.22) ** 2)
	antrum_notch = 0.22 * contraction + 0.06 * math.exp(-((s - 0.82) / 0.10) ** 2)
	proximal_bulge = 0.08 * math.exp(-((s - 0.16) / 0.18) ** 2)

	theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
	denom = np.sqrt((np.cos(theta) / max(a, 1e-6)) ** 2 + (np.sin(theta) / max(b, 1e-6)) ** 2)
	radius = 1.0 / np.maximum(denom, 1e-6)
	radius *= 1.0 + proximal_bulge * np.cos(theta - 2.4)
	radius *= 1.0 + body_asym * np.cos(2.0 * (theta - 0.35))

	notch = antrum_notch * np.exp(-0.5 * ((np.angle(np.exp(1j * (theta - 0.10)))) / 0.42) ** 2)
	radius *= 1.0 - notch

	x = radius * np.cos(theta)
	y = radius * np.sin(theta)
	x += (0.07 * a * math.sin(2.0 * math.pi * s) + 0.10 * a * roll_bias) * (y / max(b, 1e-6)) ** 2
	y += 0.08 * b * np.sin(theta + 0.6) * np.exp(-((s - 0.70) / 0.18) ** 2)
	return np.column_stack([x, y]).astype(np.float64)


def rasterize_binary_polygon(polygon_mm: np.ndarray, image_size: int, pixel_spacing: float) -> np.ndarray:
	canvas = Image.new("L", (image_size, image_size), 0)
	draw = ImageDraw.Draw(canvas)
	cx = (image_size - 1) / 2.0
	cy = (image_size - 1) / 2.0
	pixels = []
	for x_mm, y_mm in polygon_mm:
		px = cx + x_mm / pixel_spacing
		py = cy + y_mm / pixel_spacing
		pixels.append((float(px), float(py)))
	draw.polygon(pixels, outline=1, fill=1)
	array = np.asarray(canvas, dtype=np.float32)
	return np.clip(array, 0.0, 1.0)


def sweep_coordinate(t: float, gastric_period: float) -> float:
	drift = 0.43 * (t / gastric_period)
	return float(triangle_wave(np.array([t / SWEEP_PERIOD + drift], dtype=np.float64))[0])


def translate_binary_frame(frame: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
	height, width = frame.shape
	translated = np.zeros_like(frame)
	src_x0 = max(0, -shift_x)
	src_x1 = min(width, width - shift_x) if shift_x >= 0 else width
	dst_x0 = max(0, shift_x)
	dst_x1 = dst_x0 + (src_x1 - src_x0)
	src_y0 = max(0, -shift_y)
	src_y1 = min(height, height - shift_y) if shift_y >= 0 else height
	dst_y0 = max(0, shift_y)
	dst_y1 = dst_y0 + (src_y1 - src_y0)
	if src_x1 > src_x0 and src_y1 > src_y0:
		translated[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
	return translated


def generate_scanner_stream(model: GastricReferenceModel, gastric_period: float) -> None:
	timestamps = np.linspace(0.0, SCANNER_DURATION, SCANNER_FRAMES, dtype=np.float64)
	frames = np.zeros((SCANNER_FRAMES, SCANNER_SIZE, SCANNER_SIZE), dtype=np.float32)
	positions = np.zeros((SCANNER_FRAMES, 3), dtype=np.float64)
	orientations = np.zeros((SCANNER_FRAMES, 3, 3), dtype=np.float64)

	for idx, t in enumerate(timestamps):
		phase = float((t % gastric_period) / gastric_period)
		s = sweep_coordinate(float(t), gastric_period)
		positions[idx] = world_centerline(model, s, phase)
		orientations[idx] = probe_orientation(model, s, phase, float(t))

		polygon = cross_section_polygon_mm(model, s, phase)
		frame = rasterize_binary_polygon(polygon, SCANNER_SIZE, PIXEL_SPACING_MM)

		# 轻微抖动仅作用于切面中心，不引入图像噪声，避免重建杂点。
		shift_x = int(round(2.0 * math.sin(2.0 * math.pi * t / 41.0)))
		shift_y = int(round(2.0 * math.cos(2.0 * math.pi * t / 37.0)))
		frame = translate_binary_frame(frame, shift_x=shift_x, shift_y=shift_y)

		frames[idx] = frame
		save_png(frame, SCANNER_IMG_DIR / f"scanner_{idx:04d}.png")

	OUT_DIR.mkdir(parents=True, exist_ok=True)
	RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(
		OUT_DIR / "scanner_sequence.npz",
		frames=frames.astype(np.float32),
		timestamps=timestamps,
		positions=positions,
		orientations=orientations,
	)
	np.savez_compressed(
		RAW_OUT_DIR / "scanner_sequence.npz",
		frames=frames.astype(np.float32),
		timestamps=timestamps,
		positions=positions,
		orientations=orientations,
	)


def clear_existing_scanner_pngs() -> None:
	SCANNER_IMG_DIR.mkdir(parents=True, exist_ok=True)
	for path in SCANNER_IMG_DIR.glob("scanner_*.png"):
		path.unlink()


def main() -> None:
	if not REFERENCE_PLY.exists():
		raise FileNotFoundError(f"Reference stomach point cloud not found: {REFERENCE_PLY}")
	gastric_period = detect_monitor_period(MONITOR_STREAM)
	clear_existing_scanner_pngs()
	model = load_reference_model(REFERENCE_PLY)
	generate_scanner_stream(model, gastric_period)
	print(f"Detected gastric period from monitor stream: {gastric_period:.6f}s")
	print(f"Generated scanner data at {RAW_OUT_DIR / 'scanner_sequence.npz'}")
	print(f"Mirror copy written to {OUT_DIR / 'scanner_sequence.npz'}")
	print(f"Scanner PNGs: {SCANNER_IMG_DIR}")


if __name__ == "__main__":
	main()
