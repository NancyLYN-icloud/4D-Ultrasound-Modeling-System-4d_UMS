from __future__ import annotations

import argparse
from datetime import datetime
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    gaussian_filter1d,
    label,
)
from skimage.measure import marching_cubes


@dataclass
class Centerline:
    samples: np.ndarray
    tangents: np.ndarray
    u_samples: np.ndarray


@dataclass
class VolumeGeometry:
    solid: np.ndarray
    distance_field: np.ndarray
    voxel_size: float
    min_corner: np.ndarray


@dataclass
class VertexCenterlineMapping:
    u: np.ndarray
    centers: np.ndarray
    tangents: np.ndarray
    curvature: np.ndarray


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    return _build_parser(root).parse_args()


def _build_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a smooth one-cycle peristaltic stomach mesh sequence."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "stomach_data" / "changouxing01.ply",
        help="Input ASCII PLY point cloud.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "simulate_mesh",
        help="Root directory that stores per-run output folders.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional subfolder name for this run. Defaults to a timestamped folder.",
    )
    parser.add_argument(
        "--num-phases",
        type=int,
        default=66,
        help="Number of phases in one closed cycle.",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=144,
        help="Longest-axis voxel resolution used for mesh reconstruction.",
    )
    parser.add_argument(
        "--base-smooth-iterations",
        type=int,
        default=30,
        help="Taubin smoothing iterations applied to the reconstructed rest mesh.",
    )
    parser.add_argument(
        "--centerline-samples",
        type=int,
        default=31,
        help="Number of samples along the fitted stomach centerline.",
    )
    parser.add_argument(
        "--body-contraction",
        type=float,
        default=0.18,
        help="Maximum ring contraction ratio in the gastric body.",
    )
    parser.add_argument(
        "--pylorus-contraction",
        type=float,
        default=0.30,
        help="Maximum contraction ratio near the pylorus.",
    )
    parser.add_argument(
        "--wave-width",
        type=float,
        default=0.09,
        help="Longitudinal width of the traveling peristaltic ring.",
    )
    parser.add_argument(
        "--deformation-smooth-iterations",
        type=int,
        default=11,
        help="Neighbor smoothing iterations applied to the contraction field.",
    )
    parser.add_argument(
        "--deformation-smooth-relax",
        type=float,
        default=0.46,
        help="Relaxation factor for contraction-field smoothing.",
    )
    parser.add_argument(
        "--post-smooth-iterations",
        type=int,
        default=14,
        help="Light Taubin smoothing iterations applied after each phase deformation.",
    )
    parser.add_argument(
        "--post-smooth-neighborhood-order",
        type=int,
        default=8,
        help="Neighborhood ring order used by post-process mesh smoothing.",
    )
    parser.add_argument(
        "--reverse-centerline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reverse the extracted centerline orientation before propagating the wave.",
    )
    parser.add_argument(
        "--wave-start-u",
        type=float,
        default=0.32,
        help="Normalized centerline position where the contraction wave starts.",
    )
    parser.add_argument(
        "--wave-end-u",
        type=float,
        default=0.98,
        help="Normalized centerline position where the contraction wave ends.",
    )
    return parser


def load_ascii_ply_points(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        header = []
        for line in handle:
            header.append(line.strip())
            if line.strip() == "end_header":
                break

        vertex_count = None
        for line in header:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                break

        if vertex_count is None:
            raise ValueError(f"Could not find vertex count in {path}")

        points = np.loadtxt(handle, dtype=np.float64, max_rows=vertex_count)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Unexpected PLY vertex layout in {path}")

    return points[:, :3]


def smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def smootherstep(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def smooth_ramp(alpha: float, start: float, end: float) -> float:
    if end <= start:
        return float(alpha >= end)
    return float(smootherstep((alpha - start) / (end - start)))


def smooth_pulse(alpha: float, start: float, rise_end: float, fall_start: float, end: float) -> float:
    rise = smooth_ramp(alpha, start, rise_end)
    fall = 1.0 - smooth_ramp(alpha, fall_start, end)
    return float(np.clip(rise * fall, 0.0, 1.0))


def normalize_rows(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, eps, None)


def reconstruct_surface_from_points(
    points: np.ndarray,
    grid_resolution: int,
    base_smooth_iterations: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | list[float]], VolumeGeometry]:
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    extent = max_corner - min_corner
    pad = np.maximum(0.08 * extent, 3.0)
    min_corner = min_corner - pad
    max_corner = max_corner + pad
    padded_extent = max_corner - min_corner

    longest = padded_extent.max()
    voxel_size = longest / float(grid_resolution - 1)
    grid_shape = np.maximum(
        np.ceil(padded_extent / voxel_size).astype(int) + 1,
        24,
    )

    occupancy = np.zeros(grid_shape, dtype=np.float32)
    scaled = (points - min_corner) / voxel_size
    indices = np.clip(np.rint(scaled).astype(int), 0, grid_shape - 1)
    occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0

    shell = gaussian_filter(occupancy, sigma=1.1) > 0.035
    shell = binary_closing(shell, iterations=2)
    solid = binary_fill_holes(shell)
    solid = largest_component(solid)
    distance_field = distance_transform_edt(solid) * voxel_size
    field = gaussian_filter(solid.astype(np.float32), sigma=1.0)

    vertices, faces, _, _ = marching_cubes(
        field,
        level=0.5,
        spacing=(voxel_size, voxel_size, voxel_size),
        allow_degenerate=False,
    )
    vertices = vertices + min_corner
    faces = faces.astype(np.int32)
    vertices = taubin_smooth(
        vertices,
        faces,
        iterations=base_smooth_iterations,
        lamb=0.47,
        mu=-0.49,
    )

    meta = {
        "voxel_size": float(voxel_size),
        "grid_shape": grid_shape.astype(int).tolist(),
        "bbox_min": min_corner.tolist(),
        "bbox_max": max_corner.tolist(),
    }
    volume = VolumeGeometry(
        solid=solid,
        distance_field=distance_field,
        voxel_size=float(voxel_size),
        min_corner=min_corner,
    )
    return vertices, faces, meta, volume


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = label(mask)
    if count <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def build_adjacency(num_vertices: int, faces: np.ndarray) -> sparse.csr_matrix:
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    reverse_edges = edges[:, ::-1]
    edges = np.vstack([edges, reverse_edges])
    data = np.ones(edges.shape[0], dtype=np.float64)
    adjacency = sparse.coo_matrix(
        (data, (edges[:, 0], edges[:, 1])),
        shape=(num_vertices, num_vertices),
    ).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


def expand_adjacency(adjacency: sparse.csr_matrix, order: int) -> sparse.csr_matrix:
    if order <= 1:
        return adjacency

    base = adjacency.sign().tocsr()
    expanded = base.copy()
    frontier = base.copy()

    for _ in range(1, order):
        frontier = (frontier @ base).sign().tocsr()
        frontier.setdiag(0)
        frontier.eliminate_zeros()
        expanded = (expanded + frontier).sign().tocsr()

    expanded.setdiag(0)
    expanded.eliminate_zeros()
    return expanded


def taubin_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int,
    lamb: float,
    mu: float,
    adjacency: sparse.csr_matrix | None = None,
) -> np.ndarray:
    if adjacency is None:
        adjacency = build_adjacency(len(vertices), faces)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    degrees = np.clip(degrees, 1.0, None)
    smoothed = vertices.copy()

    for _ in range(iterations):
        neighbor_mean = adjacency @ smoothed / degrees[:, None]
        smoothed = smoothed + lamb * (neighbor_mean - smoothed)
        neighbor_mean = adjacency @ smoothed / degrees[:, None]
        smoothed = smoothed + mu * (neighbor_mean - smoothed)

    return smoothed


def smooth_mesh_field(
    adjacency: sparse.csr_matrix,
    values: np.ndarray,
    iterations: int,
    relax: float,
) -> np.ndarray:
    if iterations <= 0:
        return values

    smoothed = values.copy()
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    degrees = np.clip(degrees, 1.0, None)

    for _ in range(iterations):
        neighbor_mean = adjacency @ smoothed / degrees[:, None] if smoothed.ndim == 2 else (adjacency @ smoothed) / degrees
        smoothed = smoothed + relax * (neighbor_mean - smoothed)

    return smoothed


def voxel_indices_to_world(
    indices: np.ndarray,
    min_corner: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    return min_corner + indices.astype(np.float64) * voxel_size


def build_volume_graph(
    solid: np.ndarray,
    distance_field: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    voxel_coords = np.argwhere(solid)
    node_count = len(voxel_coords)
    index_map = -np.ones(solid.shape, dtype=np.int32)
    index_map[tuple(voxel_coords.T)] = np.arange(node_count, dtype=np.int32)

    rows = []
    cols = []
    weights = []
    forward_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
        if (dx, dy, dz) > (0, 0, 0)
    ]

    for dx, dy, dz in forward_offsets:
        src_slices = (
            slice(max(0, -dx), solid.shape[0] - max(0, dx)),
            slice(max(0, -dy), solid.shape[1] - max(0, dy)),
            slice(max(0, -dz), solid.shape[2] - max(0, dz)),
        )
        dst_slices = (
            slice(max(0, dx), solid.shape[0] - max(0, -dx)),
            slice(max(0, dy), solid.shape[1] - max(0, -dy)),
            slice(max(0, dz), solid.shape[2] - max(0, -dz)),
        )
        src_valid = solid[src_slices]
        dst_valid = solid[dst_slices]
        valid = src_valid & dst_valid
        if not np.any(valid):
            continue

        src_nodes = index_map[src_slices][valid]
        dst_nodes = index_map[dst_slices][valid]
        step_length = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        mean_radius = 0.5 * (
            distance_field[src_slices][valid] + distance_field[dst_slices][valid]
        )
        edge_cost = step_length / np.power(np.clip(mean_radius, 0.5, None), 1.15)

        rows.append(np.concatenate([src_nodes, dst_nodes]))
        cols.append(np.concatenate([dst_nodes, src_nodes]))
        weights.append(np.concatenate([edge_cost, edge_cost]))

    graph = sparse.coo_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(node_count, node_count),
    ).tocsr()
    return graph, voxel_coords, index_map


def trace_predecessor_path(
    predecessors: np.ndarray,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    path = [end_index]
    cursor = end_index
    while cursor != start_index:
        cursor = int(predecessors[cursor])
        if cursor < 0:
            raise ValueError("Could not trace voxel centerline path between the selected endpoints")
        path.append(cursor)
    path.reverse()
    return np.asarray(path, dtype=np.int32)


def resample_polyline(points: np.ndarray, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    diffs = np.diff(points, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total_length = max(cumulative[-1], 1e-6)
    targets = np.linspace(0.0, total_length, sample_count)
    resampled = np.column_stack(
        [np.interp(targets, cumulative, points[:, dim]) for dim in range(3)]
    )
    for dim in range(3):
        resampled[:, dim] = gaussian_filter1d(resampled[:, dim], sigma=1.0, mode="nearest")
    resampled[0] = points[0]
    resampled[-1] = points[-1]
    u_samples = targets / total_length
    return resampled, u_samples


def reverse_centerline(centerline: Centerline) -> Centerline:
    reversed_samples = centerline.samples[::-1].copy()
    reversed_tangents = normalize_rows(np.gradient(reversed_samples, axis=0))
    reversed_u = np.linspace(0.0, 1.0, len(reversed_samples))
    return Centerline(samples=reversed_samples, tangents=reversed_tangents, u_samples=reversed_u)


def farthest_reachable_node(distances: np.ndarray) -> int:
    reachable = np.isfinite(distances)
    if not np.any(reachable):
        raise ValueError("No reachable nodes found in volume graph")
    masked = np.where(reachable, distances, -np.inf)
    return int(np.argmax(masked))


def extract_longest_volume_centerline(
    volume: VolumeGeometry,
    sample_count: int,
) -> tuple[Centerline, dict[str, float | list[float]]]:
    graph, voxel_coords, _ = build_volume_graph(volume.solid, volume.distance_field)
    voxel_coords_world = voxel_indices_to_world(voxel_coords, volume.min_corner, volume.voxel_size)
    distance_values = volume.distance_field[tuple(voxel_coords.T)]

    start_index = int(np.argmax(distance_values))
    distances_a = sparse.csgraph.dijkstra(graph, directed=False, indices=start_index)
    endpoint_a = farthest_reachable_node(distances_a)
    distances_b, predecessors = sparse.csgraph.dijkstra(
        graph,
        directed=False,
        indices=endpoint_a,
        return_predecessors=True,
    )
    endpoint_b = farthest_reachable_node(distances_b)
    path_indices = trace_predecessor_path(predecessors, endpoint_a, endpoint_b)
    path_world = voxel_coords_world[path_indices]
    endpoint_radii = distance_values[[endpoint_a, endpoint_b]]
    neighborhood_radius = max(10.0, 6.0 * volume.voxel_size)
    endpoint_a_dist = np.linalg.norm(voxel_coords_world - voxel_coords_world[endpoint_a], axis=1)
    endpoint_b_dist = np.linalg.norm(voxel_coords_world - voxel_coords_world[endpoint_b], axis=1)
    endpoint_a_mask = endpoint_a_dist <= neighborhood_radius
    endpoint_b_mask = endpoint_b_dist <= neighborhood_radius
    endpoint_a_region_radius = float(distance_values[endpoint_a_mask].mean())
    endpoint_b_region_radius = float(distance_values[endpoint_b_mask].mean())

    if endpoint_a_region_radius >= endpoint_b_region_radius:
        fundus_seed = voxel_coords_world[endpoint_b]
        distal_seed = voxel_coords_world[endpoint_a]
        oriented_path_world = path_world[::-1].copy()
        auto_reversed = True
    else:
        fundus_seed = voxel_coords_world[endpoint_a]
        distal_seed = voxel_coords_world[endpoint_b]
        oriented_path_world = path_world
        auto_reversed = False

    samples, u_samples = resample_polyline(oriented_path_world, sample_count)
    tangents = normalize_rows(np.gradient(samples, axis=0))
    centerline = Centerline(samples=samples, tangents=tangents, u_samples=u_samples)

    centerline_meta = {
        "method": "longest_volume_curve",
        "path_voxel_count": int(len(path_indices)),
        "path_length": float(np.sum(np.linalg.norm(np.diff(oriented_path_world, axis=0), axis=1))),
        "path_end_mode": "volume_graph_diameter",
        "fundus_seed": fundus_seed.tolist(),
        "distal_tip_seed": distal_seed.tolist(),
        "diameter_endpoint_a": voxel_coords_world[endpoint_a].tolist(),
        "diameter_endpoint_b": voxel_coords_world[endpoint_b].tolist(),
        "diameter_endpoint_a_radius": float(endpoint_radii[0]),
        "diameter_endpoint_b_radius": float(endpoint_radii[1]),
        "diameter_endpoint_a_region_radius": endpoint_a_region_radius,
        "diameter_endpoint_b_region_radius": endpoint_b_region_radius,
        "start_seed": voxel_coords_world[start_index].tolist(),
        "auto_oriented_fundus_to_distal": True,
        "auto_reversed": auto_reversed,
    }
    return centerline, centerline_meta


def project_vertices_to_centerline(
    vertices: np.ndarray,
    centerline: Centerline,
) -> VertexCenterlineMapping:
    segment_starts = centerline.samples[:-1]
    segment_ends = centerline.samples[1:]
    segment_vectors = segment_ends - segment_starts
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    segment_lengths = np.clip(segment_lengths, 1e-8, None)
    segment_dirs = segment_vectors / segment_lengths[:, None]
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = max(cumulative[-1], 1e-8)

    relative = vertices[:, None, :] - segment_starts[None, :, :]
    projection_t = np.sum(relative * segment_vectors[None, :, :], axis=2) / (segment_lengths[None, :] ** 2)
    projection_t = np.clip(projection_t, 0.0, 1.0)
    projected = segment_starts[None, :, :] + projection_t[:, :, None] * segment_vectors[None, :, :]
    distances = np.linalg.norm(vertices[:, None, :] - projected, axis=2)
    best_segment = np.argmin(distances, axis=1)
    best_t = projection_t[np.arange(len(vertices)), best_segment]
    centers = projected[np.arange(len(vertices)), best_segment]
    tangents = segment_dirs[best_segment]
    arc_positions = cumulative[best_segment] + best_t * segment_lengths[best_segment]
    u = arc_positions / total_length
    center_distances = distances[np.arange(len(vertices)), best_segment]
    tangent_deltas = np.gradient(centerline.tangents, axis=0)
    arc_step = np.gradient(centerline.u_samples) * total_length
    sample_curvature = np.linalg.norm(tangent_deltas, axis=1) / np.clip(arc_step, 1e-6, None)
    curvature_scale = np.quantile(sample_curvature, 0.95)
    normalized_curvature = np.clip(sample_curvature / max(curvature_scale, 1e-6), 0.0, 1.0)
    curvature = np.interp(u, centerline.u_samples, normalized_curvature)

    return VertexCenterlineMapping(
        u=u,
        centers=centers,
        tangents=tangents,
        curvature=curvature,
    )


def phase_profile(alpha: float) -> dict[str, float]:
    travel_progress = smooth_ramp(alpha, 0.08, 0.84)
    wave_center = 0.32 + 0.66 * travel_progress
    wave_gain = smooth_pulse(alpha, 0.08, 0.24, 0.78, 0.97)
    bend_gain = 0.34 * smooth_pulse(alpha, 0.48, 0.72, 0.86, 1.00)
    distal_gain = 0.26 * smooth_pulse(alpha, 0.58, 0.76, 0.86, 1.00)
    pylorus_gain = 0.52 * smooth_pulse(alpha, 0.66, 0.84, 0.88, 1.00)
    tail_gain = 0.22 * smooth_pulse(alpha, 0.74, 0.90, 0.92, 1.00)
    return {
        "wave_center": float(wave_center),
        "wave_gain": float(wave_gain),
        "bend_gain": float(bend_gain),
        "distal_gain": float(distal_gain),
        "pylorus_gain": float(pylorus_gain),
        "tail_gain": float(tail_gain),
    }


def phase_profile_with_wave_range(alpha: float, wave_start_u: float, wave_end_u: float) -> dict[str, float]:
    params = phase_profile(alpha)
    params["wave_center"] = float(wave_start_u + (wave_end_u - wave_start_u) * smooth_ramp(alpha, 0.08, 0.84))
    return params


def deform_mesh(
    rest_vertices: np.ndarray,
    vertex_mapping: VertexCenterlineMapping,
    alpha: float,
    body_contraction: float,
    pylorus_contraction: float,
    wave_width: float,
    adjacency: sparse.csr_matrix,
    deformation_smooth_iterations: int,
    deformation_smooth_relax: float,
    wave_start_u: float,
    wave_end_u: float,
) -> np.ndarray:
    radial = rest_vertices - vertex_mapping.centers
    tangents = vertex_mapping.tangents
    axial_component = np.sum(radial * tangents, axis=1, keepdims=True)
    circumferential = radial - axial_component * tangents
    circum_radius = np.linalg.norm(circumferential, axis=1, keepdims=True)
    circum_direction = circumferential / np.clip(circum_radius, 1e-8, None)

    params = phase_profile_with_wave_range(alpha, wave_start_u=wave_start_u, wave_end_u=wave_end_u)
    vertex_u = vertex_mapping.u
    active_mask = smoothstep((vertex_u - 0.26) / 0.07)
    body_envelope = np.exp(-0.5 * ((vertex_u - params["wave_center"]) / wave_width) ** 2)
    bend_envelope = np.exp(-0.5 * ((vertex_u - 0.84) / 0.11) ** 2)
    distal_envelope = np.exp(-0.5 * ((vertex_u - 0.90) / 0.08) ** 2)
    pylorus_envelope = np.exp(-0.5 * ((vertex_u - 0.965) / 0.050) ** 2)
    tail_envelope = np.exp(-0.5 * ((vertex_u - 0.992) / 0.030) ** 2)
    distal_bias = smoothstep((vertex_u - 0.74) / 0.24)
    tail_bias = smoothstep((vertex_u - 0.86) / 0.10)
    bend_focus = smoothstep((vertex_mapping.curvature - 0.22) / 0.30)
    local_amplitude = body_contraction + (pylorus_contraction - body_contraction) * distal_bias
    local_amplitude = local_amplitude + 0.02 * tail_bias
    ring_strength = (
        active_mask * params["wave_gain"] * body_envelope
        + params["bend_gain"] * bend_focus * bend_envelope
        + params["distal_gain"] * distal_envelope
        + params["pylorus_gain"] * pylorus_envelope
        + params["tail_gain"] * tail_envelope
    )
    contraction = np.clip(local_amplitude * ring_strength, 0.0, 0.45)
    contraction = smooth_mesh_field(
        adjacency,
        contraction,
        iterations=deformation_smooth_iterations,
        relax=deformation_smooth_relax,
    )

    radial_offset = circum_direction * circum_radius * contraction[:, None]
    return rest_vertices - radial_offset


def write_ascii_mesh_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex in vertices:
            handle.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in faces:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def compute_phase_displacements(phases: list[np.ndarray]) -> dict[str, float]:
    consecutive = [
        np.linalg.norm(phases[index + 1] - phases[index], axis=1)
        for index in range(len(phases) - 1)
    ]
    cycle_close = np.linalg.norm(phases[-1] - phases[0], axis=1)
    return {
        "mean_step_displacement": float(np.mean([step.mean() for step in consecutive])),
        "max_step_displacement": float(np.max([step.max() for step in consecutive])),
        "cycle_closure_mean": float(cycle_close.mean()),
        "cycle_closure_max": float(cycle_close.max()),
    }


def resolve_run_output_dir(output_root: Path, run_name: str) -> tuple[Path, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    if run_name:
        resolved_name = run_name.strip()
    else:
        resolved_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    run_dir = output_root / resolved_name
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{resolved_name}_{suffix:02d}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_dir.name


def generate_cycle(args: argparse.Namespace) -> None:
    points = load_ascii_ply_points(args.input)
    rest_vertices, faces, reconstruction_meta, volume = reconstruct_surface_from_points(
        points,
        grid_resolution=args.grid_resolution,
        base_smooth_iterations=args.base_smooth_iterations,
    )
    centerline, centerline_meta = extract_longest_volume_centerline(
        volume,
        args.centerline_samples,
    )
    if args.reverse_centerline:
        centerline = reverse_centerline(centerline)
    centerline_meta["orientation_reversed"] = bool(args.reverse_centerline)
    vertex_mapping = project_vertices_to_centerline(rest_vertices, centerline)
    adjacency = build_adjacency(len(rest_vertices), faces)
    post_smooth_adjacency = expand_adjacency(adjacency, args.post_smooth_neighborhood_order)
    output_root = args.output_dir
    output_dir, run_name = resolve_run_output_dir(output_root, args.run_name)

    phase_vertices = []
    for phase_index in range(args.num_phases):
        alpha = phase_index / max(args.num_phases - 1, 1)
        deformed_vertices = deform_mesh(
            rest_vertices=rest_vertices,
            vertex_mapping=vertex_mapping,
            alpha=alpha,
            body_contraction=args.body_contraction,
            pylorus_contraction=args.pylorus_contraction,
            wave_width=args.wave_width,
            adjacency=adjacency,
            deformation_smooth_iterations=args.deformation_smooth_iterations,
            deformation_smooth_relax=args.deformation_smooth_relax,
                wave_start_u=args.wave_start_u,
                wave_end_u=args.wave_end_u,
        )
        if args.post_smooth_iterations > 0:
            deformed_vertices = taubin_smooth(
                deformed_vertices,
                faces,
                iterations=args.post_smooth_iterations,
                lamb=0.24,
                mu=-0.25,
                adjacency=post_smooth_adjacency,
            )
        phase_vertices.append(deformed_vertices)
        mesh_name = f"stomach_phase_{phase_index:03d}.ply"
        write_ascii_mesh_ply(output_dir / mesh_name, deformed_vertices, faces)

    metrics = compute_phase_displacements(phase_vertices)
    metadata = {
        "input_point_cloud": str(args.input),
        "output_root": str(output_root),
        "run_name": run_name,
        "run_output_dir": str(output_dir),
        "num_input_points": int(len(points)),
        "num_output_vertices": int(len(rest_vertices)),
        "num_output_faces": int(len(faces)),
        "num_phases": int(args.num_phases),
        "base_smooth_iterations": int(args.base_smooth_iterations),
        "body_contraction": float(args.body_contraction),
        "pylorus_contraction": float(args.pylorus_contraction),
        "wave_width": float(args.wave_width),
        "deformation_smooth_iterations": int(args.deformation_smooth_iterations),
        "deformation_smooth_relax": float(args.deformation_smooth_relax),
        "post_smooth_iterations": int(args.post_smooth_iterations),
        "post_smooth_neighborhood_order": int(args.post_smooth_neighborhood_order),
        "reverse_centerline": bool(args.reverse_centerline),
        "centerline_samples": int(args.centerline_samples),
        "reconstruction": reconstruction_meta,
        "centerline": centerline_meta,
        "smoothness_metrics": metrics,
        "wave_start_u": float(args.wave_start_u),
        "wave_end_u": float(args.wave_end_u),
    }
    (output_dir / "stomach_cycle_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "latest_run.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "run_output_dir": str(output_dir),
                "metadata_file": str(output_dir / "stomach_cycle_metadata.json"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "output_dir": str(output_dir),
                "run_name": run_name,
                "phase_count": args.num_phases,
                "mesh_vertices": len(rest_vertices),
                "mesh_faces": len(faces),
                "smoothness_metrics": metrics,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> None:
    args = parse_args()
    generate_cycle(args)


if __name__ == "__main__":
    main()