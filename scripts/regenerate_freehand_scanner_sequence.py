from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree, shortest_path
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.config import FrameFeature, PipelineConfig
from src.paths import data_path
from src.preprocessing.phase_detection import PhaseDetector
from src.stomach_instance_paths import list_reference_pointclouds, resolve_instance_paths, resolve_monitor_input_path


DEFAULT_MONITOR_PATH = data_path("benchmark", "monitor_stream.npz")

SCANNER_DURATION = 900.0
SCANNER_FPS = 10.0
SCANNER_FRAMES = int(SCANNER_DURATION * SCANNER_FPS)
SCANNER_SIZE = 512
PIXEL_SPACING_MM = 0.42
PROFILE_BINS = 768
GOLDEN_OFFSET = 0.6180339887498948


@dataclass
class GastricReferenceModel:
	world_center: np.ndarray
	world_basis: np.ndarray
	s_grid: np.ndarray
	centerline_canonical: np.ndarray
	tangent_canonical: np.ndarray
	frame_y: np.ndarray
	frame_z: np.ndarray
	radius_y: np.ndarray
	radius_z: np.ndarray


def triangle_wave(x: np.ndarray) -> np.ndarray:
	frac = x - np.floor(x)
	return 1.0 - np.abs(2.0 * frac - 1.0)


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
	with path.open("r", encoding="utf-8", errors="ignore") as handle:
		header = True
		for line in handle:
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


def _resample_polyline(points: np.ndarray, sample_count: int) -> np.ndarray:
	if points.shape[0] == sample_count:
		return points.astype(np.float64)
	segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
	arc = np.concatenate([[0.0], np.cumsum(segment)])
	if arc[-1] <= 1e-8:
		return np.repeat(points[:1], sample_count, axis=0).astype(np.float64)
	target = np.linspace(0.0, arc[-1], sample_count)
	resampled = np.column_stack([
		np.interp(target, arc, points[:, axis])
		for axis in range(points.shape[1])
	])
	return resampled.astype(np.float64)


def _subsample_points(points: np.ndarray, max_points: int = 4096) -> np.ndarray:
	if points.shape[0] <= max_points:
		return points.astype(np.float64)
	indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
	return points[indices].astype(np.float64)


def _farthest_pair(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	samples = _subsample_points(points, max_points=min(1024, points.shape[0]))
	diff = samples[:, None, :2] - samples[None, :, :2]
	dist2 = np.sum(diff * diff, axis=2)
	start_idx, end_idx = np.unravel_index(int(np.argmax(dist2)), dist2.shape)
	return samples[start_idx].astype(np.float64), samples[end_idx].astype(np.float64)


def _build_surface_graph(points: np.ndarray, k_neighbors: int = 12) -> tuple[np.ndarray, csr_matrix]:
	fit_points = _subsample_points(points, max_points=min(2048, points.shape[0]))
	neighbor_count = min(k_neighbors + 1, fit_points.shape[0])
	tree = cKDTree(fit_points)
	distances, neighbors = tree.query(fit_points, k=neighbor_count)
	if distances.ndim == 1:
		distances = distances[:, None]
		neighbors = neighbors[:, None]

	neighbor_ids = neighbors[:, 1:]
	neighbor_dists = distances[:, 1:]
	local_scale = np.median(neighbor_dists[:, : min(5, neighbor_dists.shape[1])], axis=1)
	local_scale = np.maximum(local_scale, 1e-6)
	mutual = np.zeros((fit_points.shape[0], fit_points.shape[0]), dtype=bool)
	mutual[np.arange(fit_points.shape[0])[:, None], neighbor_ids] = True

	rows: list[int] = []
	cols: list[int] = []
	weights: list[float] = []
	for point_idx in range(fit_points.shape[0]):
		for neighbor_offset, neighbor_idx in enumerate(neighbor_ids[point_idx]):
			neighbor_idx = int(neighbor_idx)
			if neighbor_idx == point_idx:
				continue
			distance = float(neighbor_dists[point_idx, neighbor_offset])
			scale = max(local_scale[point_idx], local_scale[neighbor_idx])
			if not mutual[neighbor_idx, point_idx] and distance > 1.35 * scale:
				continue
			if distance > 1.85 * scale:
				continue
			rows.append(point_idx)
			cols.append(neighbor_idx)
			weights.append(distance)

	if len(weights) < max(8, fit_points.shape[0] // 2):
		rows = []
		cols = []
		weights = []
		for point_idx in range(fit_points.shape[0]):
			for neighbor_offset, neighbor_idx in enumerate(neighbor_ids[point_idx]):
				neighbor_idx = int(neighbor_idx)
				if neighbor_idx == point_idx:
					continue
				distance = float(neighbor_dists[point_idx, neighbor_offset])
				rows.append(point_idx)
				cols.append(neighbor_idx)
				weights.append(distance)

	graph = csr_matrix((weights, (rows, cols)), shape=(fit_points.shape[0], fit_points.shape[0]))
	graph = graph.minimum(graph.T)
	return fit_points, graph


def _extract_tree_diameter_path(graph: csr_matrix) -> np.ndarray:
	if graph.shape[0] < 2:
		return np.arange(graph.shape[0], dtype=np.int64)
	component_count, labels = connected_components(graph, directed=False)
	if component_count > 1:
		component_sizes = np.bincount(labels)
		keep_label = int(np.argmax(component_sizes))
		keep = np.where(labels == keep_label)[0]
		graph = graph[keep][:, keep]
	else:
		keep = np.arange(graph.shape[0], dtype=np.int64)

	tree = minimum_spanning_tree(graph)
	tree = tree + tree.T
	anchor_dist = np.asarray(shortest_path(tree, directed=False, indices=0)).ravel()
	anchor = int(np.nanargmax(np.where(np.isfinite(anchor_dist), anchor_dist, -1.0)))
	end_dist, predecessors = shortest_path(tree, directed=False, indices=anchor, return_predecessors=True)
	end_dist = np.asarray(end_dist).ravel()
	end = int(np.nanargmax(np.where(np.isfinite(end_dist), end_dist, -1.0)))

	path = [end]
	current = end
	while current != anchor and current != -9999:
		current = int(predecessors[current])
		if current == -9999:
			break
		path.append(current)
	path = np.asarray(path[::-1], dtype=np.int64)
	if path.size < 2:
		path = np.array([0, graph.shape[0] - 1], dtype=np.int64)
	return keep[path]


def _fit_elastic_centerline(points: np.ndarray, sample_count: int) -> np.ndarray:
	fit_points, graph = _build_surface_graph(points)
	path_indices = _extract_tree_diameter_path(graph)
	seed_path = fit_points[path_indices]
	if seed_path.shape[0] < 2:
		endpoint_a, endpoint_b = _farthest_pair(fit_points)
		seed_path = np.vstack([endpoint_a, endpoint_b])

	nodes = _resample_polyline(seed_path, sample_count)
	for _ in range(12):
		tangent, frame_y, frame_z = _compute_centerline_frames(nodes)
		s_grid = np.linspace(0.0, 1.0, nodes.shape[0], dtype=np.float64)
		_, assignments, _, _ = _project_to_centerline(points, nodes, frame_y, frame_z, s_grid)
		updated = nodes.copy()
		for node_idx in range(nodes.shape[0]):
			left = max(0, node_idx - 1)
			right = min(nodes.shape[0] - 1, node_idx + 1)
			mask = (assignments >= left) & (assignments <= right)
			if np.any(mask):
				updated[node_idx] = np.mean(points[mask], axis=0)
		updated[0] = nodes[0]
		updated[-1] = nodes[-1]
		for _ in range(2):
			smoothed = updated.copy()
			smoothed[1:-1] = 0.18 * updated[:-2] + 0.64 * updated[1:-1] + 0.18 * updated[2:]
			updated = smoothed
			updated[0] = nodes[0]
			updated[-1] = nodes[-1]
		nodes = _resample_polyline(updated, sample_count)
	return nodes.astype(np.float64)


def _compute_centerline_frames(centerline: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	tangent = np.gradient(centerline, axis=0)
	tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8

	frame_y = np.zeros_like(centerline)
	frame_z = np.zeros_like(centerline)
	ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
	if abs(float(np.dot(ref, tangent[0]))) > 0.90:
		ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

	y0 = ref - float(np.dot(ref, tangent[0])) * tangent[0]
	y0 /= np.linalg.norm(y0) + 1e-8
	z0 = np.cross(tangent[0], y0)
	z0 /= np.linalg.norm(z0) + 1e-8
	frame_y[0] = y0
	frame_z[0] = z0

	for idx in range(1, centerline.shape[0]):
		y = frame_y[idx - 1] - float(np.dot(frame_y[idx - 1], tangent[idx])) * tangent[idx]
		if np.linalg.norm(y) < 1e-6:
			y = frame_z[idx - 1] - float(np.dot(frame_z[idx - 1], tangent[idx])) * tangent[idx]
		y /= np.linalg.norm(y) + 1e-8
		z = np.cross(tangent[idx], y)
		z /= np.linalg.norm(z) + 1e-8
		y = np.cross(z, tangent[idx])
		y /= np.linalg.norm(y) + 1e-8
		frame_y[idx] = y
		frame_z[idx] = z

	return tangent.astype(np.float64), frame_y.astype(np.float64), frame_z.astype(np.float64)


def _project_to_centerline(
	points: np.ndarray,
	centerline: np.ndarray,
	frame_y: np.ndarray,
	frame_z: np.ndarray,
	s_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	segments = centerline[1:] - centerline[:-1]
	segment_length2 = np.sum(segments * segments, axis=1)
	segment_length2 = np.maximum(segment_length2, 1e-8)
	point_rel = points[:, None, :] - centerline[:-1][None, :, :]
	t = np.sum(point_rel * segments[None, :, :], axis=2) / segment_length2[None, :]
	t = np.clip(t, 0.0, 1.0)
	projection = centerline[:-1][None, :, :] + t[..., None] * segments[None, :, :]
	dist2 = np.sum((points[:, None, :] - projection) ** 2, axis=2)
	segment_indices = np.argmin(dist2, axis=1)
	segment_t = t[np.arange(points.shape[0]), segment_indices]
	base_points = centerline[segment_indices]
	next_points = centerline[segment_indices + 1]
	base_y = frame_y[segment_indices]
	next_y = frame_y[segment_indices + 1]
	base_z = frame_z[segment_indices]
	next_z = frame_z[segment_indices + 1]
	interp_points = base_points + (next_points - base_points) * segment_t[:, None]
	interp_y = base_y + (next_y - base_y) * segment_t[:, None]
	interp_z = base_z + (next_z - base_z) * segment_t[:, None]
	interp_y /= np.linalg.norm(interp_y, axis=1, keepdims=True) + 1e-8
	interp_z /= np.linalg.norm(interp_z, axis=1, keepdims=True) + 1e-8
	local = points - interp_points
	coord_y = np.sum(local * interp_y, axis=1)
	coord_z = np.sum(local * interp_z, axis=1)
	s = s_grid[segment_indices] + segment_t * (s_grid[segment_indices + 1] - s_grid[segment_indices])
	indices = np.clip(np.rint(s * (centerline.shape[0] - 1)).astype(np.int64), 0, centerline.shape[0] - 1)
	return s.astype(np.float64), indices, coord_y.astype(np.float64), coord_z.astype(np.float64)


def project_canonical_points(model: GastricReferenceModel, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	return _project_to_centerline(points, model.centerline_canonical, model.frame_y, model.frame_z, model.s_grid)


def load_reference_model(path: Path) -> GastricReferenceModel:
	points = _read_ply_points(path)
	center, basis, canonical = _pca_basis(points)
	centerline = _fit_elastic_centerline(canonical, PROFILE_BINS)
	tangent, frame_y, frame_z = _compute_centerline_frames(centerline)
	s_grid = np.linspace(0.0, 1.0, centerline.shape[0], dtype=np.float64)
	_, indices, coord_y, coord_z = _project_to_centerline(canonical, centerline, frame_y, frame_z, s_grid)

	ry = np.zeros(centerline.shape[0], dtype=np.float64)
	rz = np.zeros(centerline.shape[0], dtype=np.float64)
	for idx in range(centerline.shape[0]):
		mask = indices == idx
		if mask.sum() < 24:
			left = max(0, idx - 1)
			right = min(centerline.shape[0] - 1, idx + 1)
			mask = (indices >= left) & (indices <= right)
		local_y = coord_y[mask]
		local_z = coord_z[mask]
		ry[idx] = float(np.percentile(np.abs(local_y), 92))
		rz[idx] = float(np.percentile(np.abs(local_z), 92))

	ry = np.maximum(_smooth_1d(ry), 8.0)
	rz = np.maximum(_smooth_1d(rz), 6.0)

	area_head = float(np.mean(ry[:8] * rz[:8]))
	area_tail = float(np.mean(ry[-8:] * rz[-8:]))
	if area_tail > area_head:
		centerline = centerline[::-1].copy()
		tangent, frame_y, frame_z = _compute_centerline_frames(centerline)
		ry = ry[::-1].copy()
		rz = rz[::-1].copy()
		s_grid = np.linspace(0.0, 1.0, centerline.shape[0], dtype=np.float64)

	return GastricReferenceModel(
		world_center=center,
		world_basis=basis,
		s_grid=s_grid,
		centerline_canonical=centerline,
		tangent_canonical=tangent,
		frame_y=frame_y,
		frame_z=frame_z,
		radius_y=ry,
		radius_z=rz,
	)


def detect_monitor_period(path: Path) -> float:
	data = np.load(path)
	timestamps = data["timestamps"]
	trace = data["feature_trace"]
	features = [FrameFeature(timestamp=float(ts), value=float(value)) for ts, value in zip(timestamps, trace)]
	cycles = PhaseDetector(PipelineConfig().phase_detection).detect_cycles(features)
	if not cycles:
		raise RuntimeError("No gastric cycle detected from monitor_stream.npz")
	durations = np.asarray([cycle.duration for cycle in cycles], dtype=np.float64)
	return float(np.mean(durations))


def _interp_profile(model: GastricReferenceModel, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
	s = float(np.clip(s, 0.0, 1.0))
	center = np.array([
		np.interp(s, model.s_grid, model.centerline_canonical[:, axis])
		for axis in range(3)
	], dtype=np.float64)
	tangent = np.array([
		np.interp(s, model.s_grid, model.tangent_canonical[:, axis])
		for axis in range(3)
	], dtype=np.float64)
	tangent /= np.linalg.norm(tangent) + 1e-8
	frame_y = np.array([
		np.interp(s, model.s_grid, model.frame_y[:, axis])
		for axis in range(3)
	], dtype=np.float64)
	frame_y -= float(np.dot(frame_y, tangent)) * tangent
	frame_y /= np.linalg.norm(frame_y) + 1e-8
	frame_z = np.cross(tangent, frame_y)
	frame_z /= np.linalg.norm(frame_z) + 1e-8
	frame_y = np.cross(frame_z, tangent)
	frame_y /= np.linalg.norm(frame_y) + 1e-8
	ry = float(np.interp(s, model.s_grid, model.radius_y))
	rz = float(np.interp(s, model.s_grid, model.radius_z))
	return center, tangent, frame_y, float(ry), float(rz)


def _wrapped_gaussian(x: float, center: float, sigma: float) -> float:
	deltas = (x - center, x - center - 1.0, x - center + 1.0)
	return max(math.exp(-0.5 * (delta / sigma) ** 2) for delta in deltas)


def peristaltic_state(s: float, phase: float) -> tuple[float, float, float]:
	# Peristaltic wave starts around the gastric body middle and propagates toward the pylorus.
	lead_center = 0.42 + 0.46 * phase
	trail_center = max(0.20, lead_center - 0.16)
	lead = _wrapped_gaussian(s, lead_center, 0.050)
	trail = _wrapped_gaussian(s, trail_center, 0.085)
	distal_gain = 0.90 + 0.45 * np.clip((s - 0.35) / 0.55, 0.0, 1.0)
	relaxation = 0.14 * _wrapped_gaussian(s, max(0.10, lead_center - 0.28), 0.11)
	contraction = np.clip((0.62 * lead + 0.24 * trail - relaxation) * distal_gain, 0.0, 0.82)
	axial_shift = 6.5 * lead - 1.9 * trail
	roll_bias = 0.26 * lead - 0.08 * trail
	return float(contraction), float(axial_shift), float(roll_bias)


def canonical_centerline(model: GastricReferenceModel, s: float, phase: float) -> np.ndarray:
	center, tangent, frame_y, _, _ = _interp_profile(model, s)
	frame_z = np.cross(tangent, frame_y)
	frame_z /= np.linalg.norm(frame_z) + 1e-8
	contraction, axial_shift, roll_bias = peristaltic_state(s, phase)
	body_pull = 4.8 * contraction * math.exp(-((s - 0.66) / 0.18) ** 2)
	distal_tug = 2.2 * contraction * math.exp(-((s - 0.83) / 0.11) ** 2)
	return (
		center
		+ (axial_shift + distal_tug) * tangent
		+ 4.0 * roll_bias * frame_y
		+ (-body_pull - 0.8 * distal_tug) * frame_z
	).astype(np.float64)


def world_centerline(model: GastricReferenceModel, s: float, phase: float) -> np.ndarray:
	return model.world_center + model.world_basis @ canonical_centerline(model, s, phase)


def probe_orientation(model: GastricReferenceModel, s: float, phase: float, timestamp: float) -> np.ndarray:
	ds = 1.0 / max(model.s_grid.size - 1, 1)
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
	roll = math.radians(6.0 * math.sin(2.0 * math.pi * timestamp / 21.0) + 9.0 * roll_bias)
	pitch = math.radians(4.6 * math.sin(2.0 * math.pi * timestamp / 16.0 + 0.35) + 3.6 * contraction)
	yaw = math.radians(5.8 * math.cos(2.0 * math.pi * timestamp / 27.0 - 0.15) + 2.0 * math.sin(2.0 * math.pi * s))

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

	a = max(6.0, base_ry * (1.0 - 0.84 * contraction))
	b = max(5.0, base_rz * (1.0 - 0.76 * contraction))
	body_asym = 0.16 * math.exp(-((s - 0.48) / 0.18) ** 2)
	antrum_notch = 0.38 * contraction + 0.10 * math.exp(-((s - 0.86) / 0.08) ** 2)
	proximal_bulge = 0.11 * math.exp(-((s - 0.20) / 0.15) ** 2)
	longitudinal_cleft = 0.12 * contraction * math.exp(-((s - 0.70) / 0.16) ** 2)

	theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
	denom = np.sqrt((np.cos(theta) / max(a, 1e-6)) ** 2 + (np.sin(theta) / max(b, 1e-6)) ** 2)
	radius = 1.0 / np.maximum(denom, 1e-6)
	radius *= 1.0 + proximal_bulge * np.cos(theta - 2.4)
	radius *= 1.0 + body_asym * np.cos(2.0 * (theta - 0.32))
	notch = antrum_notch * np.exp(-0.5 * ((np.angle(np.exp(1j * (theta - 0.10)))) / 0.40) ** 2)
	radius *= 1.0 - notch
	radius *= 1.0 - longitudinal_cleft * np.exp(-0.5 * ((np.angle(np.exp(1j * (theta - math.pi / 2.0)))) / 0.30) ** 2)

	x = radius * np.cos(theta)
	y = radius * np.sin(theta)
	x += (0.14 * a * math.sin(2.0 * math.pi * s) + 0.18 * a * roll_bias) * (y / max(b, 1e-6)) ** 2
	y += 0.14 * b * np.sin(theta + 0.6) * np.exp(-((s - 0.72) / 0.16) ** 2)
	y -= 0.10 * b * contraction * np.exp(-0.5 * ((np.angle(np.exp(1j * (theta - 0.05)))) / 0.34) ** 2)
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


def sweep_coordinate(timestamp: float, gastric_period: float) -> float:
	# Keep the freehand sweep quasi-independent from gastric phase so each phase bin
	# sees broad anatomical coverage after long-duration accumulation.
	base = timestamp / 7.6
	base += GOLDEN_OFFSET * (timestamp / gastric_period)
	base += 0.09 * math.sin(2.0 * math.pi * timestamp / 31.0 + 0.4)
	base += 0.04 * math.sin(2.0 * math.pi * timestamp / 13.0 - 0.2)
	return float(np.clip(triangle_wave(np.array([base], dtype=np.float64))[0], 0.0, 1.0))


def clear_scanner_pngs(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)
	for file_path in path.glob("scanner_*.png"):
		file_path.unlink()


def generate_scanner_stream(
	model: GastricReferenceModel,
	gastric_period: float,
	scanner_img_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	timestamps = np.arange(SCANNER_FRAMES, dtype=np.float64) / SCANNER_FPS
	frames = np.zeros((SCANNER_FRAMES, SCANNER_SIZE, SCANNER_SIZE), dtype=np.float32)
	positions = np.zeros((SCANNER_FRAMES, 3), dtype=np.float64)
	orientations = np.zeros((SCANNER_FRAMES, 3, 3), dtype=np.float64)

	for idx, timestamp in enumerate(timestamps):
		phase = float((timestamp % gastric_period) / gastric_period)
		s = sweep_coordinate(float(timestamp), gastric_period)
		positions[idx] = world_centerline(model, s, phase)
		orientations[idx] = probe_orientation(model, s, phase, float(timestamp))

		polygon = cross_section_polygon_mm(model, s, phase)
		frame = rasterize_binary_polygon(polygon, SCANNER_SIZE, PIXEL_SPACING_MM)
		shift_x = int(round(2.0 * math.sin(2.0 * math.pi * timestamp / 39.0)))
		shift_y = int(round(2.0 * math.cos(2.0 * math.pi * timestamp / 35.0)))
		frame = translate_binary_frame(frame, shift_x=shift_x, shift_y=shift_y)
		frames[idx] = frame
		save_png(frame, scanner_img_dir / f"scanner_{idx:04d}.png")

	return frames, timestamps, positions, orientations


def write_outputs(
	frames: np.ndarray,
	timestamps: np.ndarray,
	positions: np.ndarray,
	orientations: np.ndarray,
	output_paths: list[Path],
) -> None:
	for out_path in output_paths:
		out_path.parent.mkdir(parents=True, exist_ok=True)
		np.savez_compressed(
			out_path,
			frames=frames.astype(np.float32),
			timestamps=timestamps.astype(np.float64),
			positions=positions.astype(np.float64),
			orientations=orientations.astype(np.float64),
		)


def generate_scanner_stream_for_instance(
	instance_name: str | None,
	reference_ply: Path | None,
	monitor_path: Path | None,
) -> Path:
	instance_paths = resolve_instance_paths(instance_name=instance_name, reference_ply=reference_ply)
	resolved_monitor_path = resolve_monitor_input_path(instance_paths, explicit_path=monitor_path)
	if not resolved_monitor_path.exists():
		raise FileNotFoundError(f"Monitor stream not found: {resolved_monitor_path}")
	if not instance_paths.reference_ply.exists():
		raise FileNotFoundError(f"Reference stomach point cloud not found: {instance_paths.reference_ply}")

	gastric_period = detect_monitor_period(resolved_monitor_path)
	model = load_reference_model(instance_paths.reference_ply)
	clear_scanner_pngs(instance_paths.scanner_image_dir)
	frames, timestamps, positions, orientations = generate_scanner_stream(
		model,
		gastric_period,
		instance_paths.scanner_image_dir,
	)
	write_outputs(frames, timestamps, positions, orientations, [instance_paths.scanner_sequence])
	print(f"[RegenerateScanner] Instance: {instance_paths.name}")
	print(f"[RegenerateScanner] Reference: {instance_paths.reference_ply}")
	print(f"[RegenerateScanner] Monitor stream: {resolved_monitor_path}")
	print(f"[RegenerateScanner] Detected gastric period: {gastric_period:.6f}s")
	print(f"[RegenerateScanner] Scanner duration: {SCANNER_DURATION:.1f}s, fps: {SCANNER_FPS:.1f}, frames: {SCANNER_FRAMES}")
	print(f"[RegenerateScanner] Generated scanner data at {instance_paths.scanner_sequence}")
	print(f"[RegenerateScanner] Scanner PNGs: {instance_paths.scanner_image_dir}")
	return instance_paths.scanner_sequence


def main() -> None:
	import argparse

	parser = argparse.ArgumentParser(description="Regenerate scanner_sequence.npz for one or more stomach reference instances")
	parser.add_argument("--instance-name", type=str, default=None, help="Named stomach instance under benchmark/stomach_pcd")
	parser.add_argument("--reference-ply", type=str, default=None, help="Explicit reference stomach point cloud path")
	parser.add_argument("--monitor-path", type=str, default=str(DEFAULT_MONITOR_PATH), help="Monitor stream used to derive gastric period")
	parser.add_argument("--batch-all-references", action="store_true", help="Regenerate scanner sequences for all point clouds under benchmark/stomach_pcd")
	args = parser.parse_args()

	monitor_path = Path(args.monitor_path).expanduser().resolve() if args.monitor_path else None
	if args.batch_all_references:
		reference_paths = list_reference_pointclouds()
		if not reference_paths:
			raise FileNotFoundError("No reference point clouds found under benchmark/stomach_pcd")
		for reference_path in reference_paths:
			generate_scanner_stream_for_instance(reference_path.stem, reference_path, monitor_path)
		return

	generate_scanner_stream_for_instance(
		instance_name=args.instance_name,
		reference_ply=Path(args.reference_ply).expanduser().resolve() if args.reference_ply else None,
		monitor_path=monitor_path,
	)


if __name__ == "__main__":
	main()