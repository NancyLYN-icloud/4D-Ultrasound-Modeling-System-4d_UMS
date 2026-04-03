from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import csv
import math
import shutil
import sys

import numpy as np
from scipy import sparse
import trimesh


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig, SurfaceModelConfig
from src.paths import data_path
from src.modeling.surface_reconstruction import MeshBuildResult, _mesh_export, _read_xyz_ply, reconstruct_meshes_from_pointclouds
from src.preprocessing.phase_detection import PhaseDetector
from src.stomach_instance_paths import list_reference_pointclouds, resolve_instance_paths, resolve_monitor_input_path
import scripts.generate_stomach_cycle as cycle_model
import scripts.regenerate_freehand_scanner_sequence as regen


OUTPUT_BASE_DIR = data_path("simuilate_data")
OUTPUT_PREFIX = "phase_sequence_models_run"
OBSERVATION_ROTATION_DEG = np.array([-35.0, 20.0, -32.0], dtype=np.float64)
SHARED_BASE_VERTEX_SOFT_LIMIT = 14500
SHARED_BASE_VERTEX_HARD_LIMIT = 16000


@dataclass
class PhaseModelStats:
    phase_index: int
    phase_value: float
    timestamp_s: float
    wave_center_s: float
    max_contraction: float
    pointcloud_path: Path


@dataclass
class HybridRemeshConfig:
    enabled: bool
    peak_ratio_threshold: float
    neighbor_span: int


@dataclass
class CycleModelConfig:
    grid_resolution: int
    base_smooth_iterations: int
    centerline_samples: int
    body_contraction: float
    pylorus_contraction: float
    wave_width: float
    wave_start_u: float
    wave_end_u: float
    deformation_smooth_iterations: int
    deformation_smooth_relax: float
    post_smooth_iterations: int
    post_smooth_neighborhood_order: int
    reverse_centerline: bool


def _detect_monitor_aligned_phase_count(monitor_stream: Path, phase_bin_step_seconds: float) -> tuple[int, float]:
    if not monitor_stream.exists():
        raise FileNotFoundError(f"Monitor stream not found: {monitor_stream}")
    with np.load(monitor_stream) as data:
        timestamps = data["timestamps"]
        feature_trace = data["feature_trace"]
        features = [
            regen.FrameFeature(timestamp=float(ts), value=float(value))
            for ts, value in zip(timestamps, feature_trace)
        ]
    cycles = PhaseDetector(PipelineConfig().phase_detection).detect_cycles(features)
    if not cycles:
        raise RuntimeError("No gastric cycles detected from monitor stream")
    avg_duration = float(np.mean([cycle.duration for cycle in cycles]))
    phase_count = max(2, int(np.ceil(avg_duration / max(phase_bin_step_seconds, 1e-8))))
    return phase_count, avg_duration


def _archive_existing_gt_meshes(mesh_dir: Path) -> Path | None:
    existing_files = sorted(mesh_dir.glob("*")) if mesh_dir.exists() else []
    existing_files = [path for path in existing_files if path.is_file()]
    if not existing_files:
        return None

    archive_dir = mesh_dir.parent / f"meshes_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archive_dir.mkdir(parents=True, exist_ok=False)
    for path in existing_files:
        shutil.move(str(path), str(archive_dir / path.name))
    return archive_dir


def _sync_meshes_to_gt(meshes: list[object], mesh_dir: Path, run_dir: Path, summary_path: Path) -> tuple[Path | None, list[Path]]:
    archive_dir = _archive_existing_gt_meshes(mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    synced_paths: list[Path] = []
    for result in meshes:
        source_path = result.mesh_path
        destination = mesh_dir / source_path.name
        shutil.copy2(source_path, destination)
        synced_paths.append(destination)

    shutil.copy2(summary_path, mesh_dir / summary_path.name)
    phase_summary = run_dir / "phase_sequence_summary.csv"
    if phase_summary.exists():
        shutil.copy2(phase_summary, mesh_dir / phase_summary.name)
    transform_path = run_dir / "observation_transform.npz"
    if transform_path.exists():
        shutil.copy2(transform_path, mesh_dir / transform_path.name)
    return archive_dir, synced_paths


def _rotation_matrix_xyz(angles_deg: np.ndarray) -> np.ndarray:
    ax, ay, az = np.deg2rad(angles_deg.astype(np.float64))

    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(ax), -math.sin(ax)],
        [0.0, math.sin(ax), math.cos(ax)],
    ], dtype=np.float64)
    rot_y = np.array([
        [math.cos(ay), 0.0, math.sin(ay)],
        [0.0, 1.0, 0.0],
        [-math.sin(ay), 0.0, math.cos(ay)],
    ], dtype=np.float64)
    rot_z = np.array([
        [math.cos(az), -math.sin(az), 0.0],
        [math.sin(az), math.cos(az), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def _apply_observation_transform(points: np.ndarray, center: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return center + ((rotation @ (points - center).T).T)


def _write_observation_transform(
    out_dir: Path,
    center: np.ndarray,
    rotation: np.ndarray,
    phase_bin_step_seconds: float,
) -> Path:
    transform_path = out_dir / "observation_transform.npz"
    np.savez_compressed(
        transform_path,
        center=center.astype(np.float64),
        rotation=rotation.astype(np.float64),
        angles_deg=OBSERVATION_ROTATION_DEG.astype(np.float64),
        phase_bin_step_seconds=np.array(phase_bin_step_seconds, dtype=np.float64),
    )
    return transform_path

def _create_indexed_output_dir(base_dir: Path, prefix: str) -> tuple[Path, str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_indices: list[int] = []
    for child in base_dir.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix + "_"):
            continue
        remainder = child.name[len(prefix) + 1 :]
        run_token = remainder.split("_", 1)[0]
        if run_token.isdigit():
            existing_indices.append(int(run_token))
    next_index = max(existing_indices, default=0) + 1
    run_id = f"{next_index:03d}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = base_dir / f"{prefix}_{run_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_id


def _read_reference_points(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = regen._read_ply_points(path)
    center, basis, canonical = regen._pca_basis(points)
    return points, center, canonical


def _phase_timestamp_seconds(phase_index: int, phase_bin_step_seconds: float) -> float:
    return float(phase_index * phase_bin_step_seconds)


def _format_timestamp_token(timestamp_s: float) -> str:
    return f"t{timestamp_s:07.3f}s"


def _build_cycle_base_mesh_and_mapping(
    reference_ply: Path,
    cycle_cfg: CycleModelConfig,
    base_mesh_path: Path | None,
) -> tuple[
    np.ndarray,
    trimesh.Trimesh,
    MeshBuildResult,
    cycle_model.VertexCenterlineMapping,
    sparse.csr_matrix,
    sparse.csr_matrix,
    dict[str, float | list[float]],
    dict[str, float | list[float]],
]:
    reference_points = cycle_model.load_ascii_ply_points(reference_ply)
    rest_vertices, faces, reconstruction_meta, volume = cycle_model.reconstruct_surface_from_points(
        reference_points,
        grid_resolution=cycle_cfg.grid_resolution,
        base_smooth_iterations=cycle_cfg.base_smooth_iterations,
    )
    centerline, centerline_meta = cycle_model.extract_longest_volume_centerline(
        volume,
        cycle_cfg.centerline_samples,
    )
    if cycle_cfg.reverse_centerline:
        centerline = cycle_model.reverse_centerline(centerline)
    centerline_meta["orientation_reversed"] = bool(cycle_cfg.reverse_centerline)

    if base_mesh_path is not None:
        mesh = trimesh.load(base_mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Failed to load provided base mesh: {base_mesh_path}")
        base_mesh = mesh.copy()
        base_mesh_result = MeshBuildResult(
            pointcloud_path=reference_ply,
            mesh_path=base_mesh_path,
            timestamp_s=0.0,
            input_points=int(len(reference_points)),
            sampled_points=int(len(reference_points)),
            vertices=int(len(base_mesh.vertices)),
            faces=int(len(base_mesh.faces)),
            watertight=bool(base_mesh.is_watertight),
            method="provided_shared_base_mesh",
        )
    else:
        base_mesh = trimesh.Trimesh(vertices=rest_vertices, faces=faces.astype(np.int64), process=False)
        base_mesh_result = MeshBuildResult(
            pointcloud_path=reference_ply,
            mesh_path=reference_ply,
            timestamp_s=0.0,
            input_points=int(len(reference_points)),
            sampled_points=int(len(reference_points)),
            vertices=int(len(base_mesh.vertices)),
            faces=int(len(base_mesh.faces)),
            watertight=bool(base_mesh.is_watertight),
            method="cycle_volume_reconstruction",
        )

    vertex_mapping = cycle_model.project_vertices_to_centerline(
        np.asarray(base_mesh.vertices, dtype=np.float64),
        centerline,
    )
    adjacency = cycle_model.build_adjacency(len(base_mesh.vertices), np.asarray(base_mesh.faces, dtype=np.int64))
    post_smooth_adjacency = cycle_model.expand_adjacency(adjacency, cycle_cfg.post_smooth_neighborhood_order)
    return (
        reference_points,
        base_mesh,
        base_mesh_result,
        vertex_mapping,
        adjacency,
        post_smooth_adjacency,
        reconstruction_meta,
        centerline_meta,
    )


def _compute_cycle_phase_contraction(
    vertex_mapping: cycle_model.VertexCenterlineMapping,
    alpha: float,
    cycle_cfg: CycleModelConfig,
    adjacency: sparse.csr_matrix,
) -> tuple[np.ndarray, dict[str, float]]:
    params = cycle_model.phase_profile_with_wave_range(
        alpha,
        wave_start_u=cycle_cfg.wave_start_u,
        wave_end_u=cycle_cfg.wave_end_u,
    )
    vertex_u = vertex_mapping.u
    active_mask = cycle_model.smoothstep((vertex_u - 0.26) / 0.07)
    body_envelope = np.exp(-0.5 * ((vertex_u - params["wave_center"]) / cycle_cfg.wave_width) ** 2)
    bend_envelope = np.exp(-0.5 * ((vertex_u - 0.84) / 0.11) ** 2)
    distal_envelope = np.exp(-0.5 * ((vertex_u - 0.90) / 0.08) ** 2)
    pylorus_envelope = np.exp(-0.5 * ((vertex_u - 0.965) / 0.050) ** 2)
    tail_envelope = np.exp(-0.5 * ((vertex_u - 0.992) / 0.030) ** 2)
    distal_bias = cycle_model.smoothstep((vertex_u - 0.74) / 0.24)
    tail_bias = cycle_model.smoothstep((vertex_u - 0.86) / 0.10)
    bend_focus = cycle_model.smoothstep((vertex_mapping.curvature - 0.22) / 0.30)
    local_amplitude = cycle_cfg.body_contraction + (cycle_cfg.pylorus_contraction - cycle_cfg.body_contraction) * distal_bias
    local_amplitude = local_amplitude + 0.02 * tail_bias
    ring_strength = (
        active_mask * params["wave_gain"] * body_envelope
        + params["bend_gain"] * bend_focus * bend_envelope
        + params["distal_gain"] * distal_envelope
        + params["pylorus_gain"] * pylorus_envelope
        + params["tail_gain"] * tail_envelope
    )
    contraction = np.clip(local_amplitude * ring_strength, 0.0, 0.45)
    contraction = cycle_model.smooth_mesh_field(
        adjacency,
        contraction,
        iterations=cycle_cfg.deformation_smooth_iterations,
        relax=cycle_cfg.deformation_smooth_relax,
    )
    return contraction.astype(np.float64), params


def _deform_cycle_phase_vertices(
    base_mesh: trimesh.Trimesh,
    vertex_mapping: cycle_model.VertexCenterlineMapping,
    alpha: float,
    cycle_cfg: CycleModelConfig,
    adjacency: sparse.csr_matrix,
    post_smooth_adjacency: sparse.csr_matrix,
) -> tuple[np.ndarray, float, float]:
    deformed_vertices = cycle_model.deform_mesh(
        rest_vertices=np.asarray(base_mesh.vertices, dtype=np.float64),
        vertex_mapping=vertex_mapping,
        alpha=alpha,
        body_contraction=cycle_cfg.body_contraction,
        pylorus_contraction=cycle_cfg.pylorus_contraction,
        wave_width=cycle_cfg.wave_width,
        adjacency=adjacency,
        deformation_smooth_iterations=cycle_cfg.deformation_smooth_iterations,
        deformation_smooth_relax=cycle_cfg.deformation_smooth_relax,
        wave_start_u=cycle_cfg.wave_start_u,
        wave_end_u=cycle_cfg.wave_end_u,
    )
    if cycle_cfg.post_smooth_iterations > 0:
        deformed_vertices = cycle_model.taubin_smooth(
            deformed_vertices,
            np.asarray(base_mesh.faces, dtype=np.int64),
            iterations=cycle_cfg.post_smooth_iterations,
            lamb=0.24,
            mu=-0.25,
            adjacency=post_smooth_adjacency,
        )
    contraction, params = _compute_cycle_phase_contraction(vertex_mapping, alpha, cycle_cfg, adjacency)
    return deformed_vertices.astype(np.float64), float(params["wave_center"]), float(np.max(contraction))


def _smoothstep(edge0: float, edge1: float, value: float | np.ndarray) -> float | np.ndarray:
    span = max(edge1 - edge0, 1e-8)
    t = np.clip((value - edge0) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _build_principal_axis_profile(canonical_points: np.ndarray, profile_bins: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis_coord = canonical_points[:, 0]
    transverse_y = canonical_points[:, 1]
    transverse_z = canonical_points[:, 2]

    x_min = float(axis_coord.min())
    x_max = float(axis_coord.max())
    bins = np.linspace(x_min, x_max, profile_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    cy = np.zeros(profile_bins, dtype=np.float64)
    cz = np.zeros(profile_bins, dtype=np.float64)
    ry = np.zeros(profile_bins, dtype=np.float64)
    rz = np.zeros(profile_bins, dtype=np.float64)

    bin_index = np.digitize(axis_coord, bins) - 1
    for idx in range(profile_bins):
        mask = bin_index == idx
        if mask.sum() < 24:
            left = max(0, idx - 1)
            right = min(profile_bins - 1, idx + 1)
            mask = (bin_index >= left) & (bin_index <= right)
        local_y = transverse_y[mask]
        local_z = transverse_z[mask]
        cy[idx] = float(local_y.mean())
        cz[idx] = float(local_z.mean())
        ry[idx] = float(np.percentile(np.abs(local_y - cy[idx]), 92))
        rz[idx] = float(np.percentile(np.abs(local_z - cz[idx]), 92))

    cy = regen._smooth_1d(cy)
    cz = regen._smooth_1d(cz)
    ry = np.maximum(regen._smooth_1d(ry), 8.0)
    rz = np.maximum(regen._smooth_1d(rz), 6.0)
    return centers, cy, cz, ry, rz


def _interp_principal_axis_profile(
    axis_coord: np.ndarray,
    center_y: np.ndarray,
    center_z: np.ndarray,
    radius_y: np.ndarray,
    radius_z: np.ndarray,
    s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, axis_coord.size)
    x = np.interp(s, grid, axis_coord)
    cy = np.interp(s, grid, center_y)
    cz = np.interp(s, grid, center_z)
    ry = np.interp(s, grid, radius_y)
    rz = np.interp(s, grid, radius_z)
    return x, cy, cz, ry, rz


def _phase_wave(
    s: np.ndarray,
    phase: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float]:
    phase = float(phase % 1.0)

    onset = float(_smoothstep(0.03, 0.16, phase))
    travel = float(_smoothstep(0.12, 0.70, phase))
    release = float(1.0 - _smoothstep(0.82, 0.995, phase))
    pyloric_hold = float(_smoothstep(0.64, 0.78, phase) * (1.0 - _smoothstep(0.90, 0.995, phase)))
    cycle_amp = onset * release

    pylorus_s = 0.12
    body_origin_s = 0.50
    wave_center = body_origin_s + (pylorus_s - body_origin_s) * travel

    core = np.exp(-0.5 * ((s - wave_center) / 0.026) ** 2)
    lead = np.exp(-0.5 * ((s - wave_center) / 0.036) ** 2)
    trail = np.exp(-0.5 * ((s - (wave_center + 0.078)) / 0.060) ** 2)
    relaxation = np.exp(-0.5 * ((s - pylorus_s) / 0.100) ** 2)

    body_seed = np.exp(-0.5 * ((s - body_origin_s) / 0.078) ** 2)
    pylorus_peak = np.exp(-0.5 * ((s - pylorus_s) / 0.046) ** 2)
    body_gate = 1.0 - _smoothstep(0.78, 0.96, s)

    contraction = (
        cycle_amp * (1.02 * lead + 0.42 * trail + 0.88 * core) * body_gate * (1.04 + 0.24 * body_seed)
        + 0.52 * pyloric_hold * pylorus_peak
    )
    contraction = np.clip(contraction, 0.0, 0.995)
    return (
        contraction.astype(np.float64),
        float(wave_center),
        lead.astype(np.float64),
        trail.astype(np.float64),
        relaxation.astype(np.float64),
        cycle_amp,
    )


def _oriented_arc_coordinate(model: regen.GastricReferenceModel, s: np.ndarray) -> np.ndarray:
    start_area = float(np.mean(model.radius_y[:8] * model.radius_z[:8]))
    end_area = float(np.mean(model.radius_y[-8:] * model.radius_z[-8:]))
    if start_area > end_area:
        return 1.0 - s
    return s


def _interpolate_centerline_geometry(
    model: regen.GastricReferenceModel,
    s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = model.s_grid
    centerline = np.column_stack([
        np.interp(s, grid, model.centerline_canonical[:, axis])
        for axis in range(3)
    ])
    tangent = np.column_stack([
        np.interp(s, grid, model.tangent_canonical[:, axis])
        for axis in range(3)
    ])
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8

    frame_y = np.column_stack([
        np.interp(s, grid, model.frame_y[:, axis])
        for axis in range(3)
    ])
    frame_y -= np.sum(frame_y * tangent, axis=1, keepdims=True) * tangent
    frame_y /= np.linalg.norm(frame_y, axis=1, keepdims=True) + 1e-8

    frame_z = np.cross(tangent, frame_y)
    frame_z /= np.linalg.norm(frame_z, axis=1, keepdims=True) + 1e-8
    frame_y = np.cross(frame_z, tangent)
    frame_y /= np.linalg.norm(frame_y, axis=1, keepdims=True) + 1e-8

    radius_y = np.interp(s, grid, model.radius_y)
    radius_z = np.interp(s, grid, model.radius_z)
    return centerline, tangent, frame_y, frame_z, radius_y, radius_z


def _centerline_total_length(model: regen.GastricReferenceModel) -> float:
    segment = np.linalg.norm(np.diff(model.centerline_canonical, axis=0), axis=1)
    return float(np.sum(segment))


def _rebuild_local_ring_profile(
    s: np.ndarray,
    local_y: np.ndarray,
    local_z: np.ndarray,
    radius_y: np.ndarray,
    radius_z: np.ndarray,
    ring_scale: np.ndarray,
    circular_blend: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.arctan2(local_z * radius_y, local_y * radius_z)
    dir_y = np.cos(theta)
    dir_z = np.sin(theta)

    ellipse_radius = 1.0 / np.sqrt((dir_y ** 2) / (radius_y ** 2) + (dir_z ** 2) / (radius_z ** 2) + 1e-8)
    circular_radius = np.sqrt(radius_y * radius_z)
    target_radius = ((1.0 - circular_blend) * ellipse_radius + circular_blend * circular_radius) * ring_scale

    actual_radius = np.sqrt(local_y ** 2 + local_z ** 2)
    radius_ratio = np.clip(actual_radius / np.maximum(ellipse_radius, 1e-6), 0.0, 1.35)

    point_count = int(max(len(s), 1))
    section_bins = int(np.clip(round(math.sqrt(point_count) * 4.50), 320, 896))
    theta_bins = int(np.clip(round(math.sqrt(point_count) * 7.50), 512, 1536))
    s_pos = np.clip(s * (section_bins - 1), 0.0, float(section_bins - 1))
    s_idx = np.clip(s_pos.astype(np.int64), 0, section_bins - 1)
    theta_unit = ((theta + math.pi) / (2.0 * math.pi)) % 1.0
    theta_pos = theta_unit * theta_bins
    theta_idx = np.mod(theta_pos.astype(np.int64), theta_bins)
    flat_idx = s_idx * theta_bins + theta_idx

    flat_size = section_bins * theta_bins
    radius_sum = np.bincount(flat_idx, weights=target_radius, minlength=flat_size).reshape(section_bins, theta_bins)
    radius_count = np.bincount(flat_idx, minlength=flat_size).reshape(section_bins, theta_bins)
    profile = np.divide(radius_sum, np.maximum(radius_count, 1), dtype=np.float64)
    contraction_sum = np.bincount(flat_idx, weights=np.clip(1.0 - ring_scale, 0.0, 1.0), minlength=flat_size).reshape(section_bins, theta_bins)
    contraction_profile = np.divide(contraction_sum, np.maximum(radius_count, 1), dtype=np.float64)

    default_radius = float(np.mean(target_radius))
    for sec in range(section_bins):
        known = radius_count[sec] > 0
        if np.any(known):
            xp = np.flatnonzero(known).astype(np.float64)
            fp = profile[sec, known]
            profile[sec] = np.interp(np.arange(theta_bins, dtype=np.float64), xp, fp, period=float(theta_bins))
        else:
            profile[sec, :] = default_radius

    for _ in range(8):
        profile = 0.16 * np.roll(profile, 1, axis=1) + 0.68 * profile + 0.16 * np.roll(profile, -1, axis=1)

    section_contraction = np.clip(np.max(contraction_profile, axis=1), 0.0, 1.0)
    axial_mix = np.clip(0.12 - 0.10 * np.power(section_contraction, 0.7), 0.015, 0.12)
    for _ in range(2):
        smooth_profile = profile.copy()
        smooth_profile[1:-1] = (
            axial_mix[1:-1, None] * profile[:-2]
            + (1.0 - 2.0 * axial_mix[1:-1, None]) * profile[1:-1]
            + axial_mix[1:-1, None] * profile[2:]
        )
        profile = smooth_profile

    s0 = np.floor(s_pos).astype(np.int64)
    s1 = np.clip(s0 + 1, 0, section_bins - 1)
    s_frac = s_pos - s0.astype(np.float64)

    theta0 = np.mod(np.floor(theta_pos).astype(np.int64), theta_bins)
    theta1 = np.mod(theta0 + 1, theta_bins)
    theta_frac = theta_pos - np.floor(theta_pos)

    prof00 = profile[s0, theta0]
    prof01 = profile[s0, theta1]
    prof10 = profile[s1, theta0]
    prof11 = profile[s1, theta1]
    prof0 = (1.0 - theta_frac) * prof00 + theta_frac * prof01
    prof1 = (1.0 - theta_frac) * prof10 + theta_frac * prof11
    rebuilt_radius = radius_ratio * ((1.0 - s_frac) * prof0 + s_frac * prof1)
    return rebuilt_radius * dir_y, rebuilt_radius * dir_z


def _deform_reference_points(
    canonical_points: np.ndarray,
    model: regen.GastricReferenceModel,
    phase: float,
    observation_rotation: np.ndarray,
    narrowing_scale: float,
) -> tuple[np.ndarray, float, float]:
    s, _, local_y, local_z = regen.project_canonical_points(model, canonical_points)
    oriented_s = _oriented_arc_coordinate(model, s)
    contraction, wave_center, lead, trail, relaxation, cycle_amp = _phase_wave(oriented_s, phase)
    radial_contraction = np.clip(contraction * float(narrowing_scale), 0.0, 1.0)

    centerline, tangent, frame_y, frame_z, ry0, rz0 = _interpolate_centerline_geometry(model, s)

    max_contraction = float(np.max(radial_contraction))

    ry_safe = np.maximum(ry0, 1e-6)
    rz_safe = np.maximum(rz0, 1e-6)
    radial_level = np.sqrt((local_y / ry_safe) ** 2 + (local_z / rz_safe) ** 2)
    wall_weight = np.clip(0.78 + 0.22 * radial_level, 0.78, 1.0)
    contraction_shape = np.power(radial_contraction, 1.45)
    ring_scale = np.clip(1.0 - 1.16 * contraction_shape * wall_weight, 0.028, 1.0)

    # Strong contractions should tighten into a smoother circular ring instead of
    # preserving the original elliptical section all the way to the lumen minimum.
    circular_blend = np.clip((radial_contraction - 0.04) / 0.07, 0.0, 1.0)
    circular_blend = np.clip(np.power(circular_blend, 0.60), 0.0, 1.0)

    axial_shortening = -2.4 * radial_contraction * np.exp(-0.5 * ((oriented_s - wave_center) / 0.12) ** 2)
    pyloric_shortening = -1.8 * relaxation * (0.35 + 0.65 * (1.0 - cycle_amp)) * np.exp(-0.5 * ((oriented_s - 0.12) / 0.055) ** 2)

    tangent_offset = axial_shortening + pyloric_shortening
    total_length = max(_centerline_total_length(model), 1e-6)
    shifted_s = np.clip(s + tangent_offset / total_length, 0.0, 1.0)
    shifted_centerline, shifted_tangent, shifted_frame_y, shifted_frame_z, _, _ = _interpolate_centerline_geometry(model, shifted_s)
    frame_y_offset, frame_z_offset = _rebuild_local_ring_profile(
        shifted_s,
        local_y,
        local_z,
        ry_safe,
        rz_safe,
        ring_scale,
        circular_blend,
    )

    canonical_deformed = (
        shifted_centerline
        + shifted_frame_y * frame_y_offset[:, None]
        + shifted_frame_z * frame_z_offset[:, None]
    )
    world_deformed = model.world_center + (model.world_basis @ canonical_deformed.T).T
    observed_points = _apply_observation_transform(world_deformed, model.world_center, observation_rotation)
    return observed_points.astype(np.float32), wave_center, max_contraction


def _observed_to_canonical_points(
    observed_points: np.ndarray,
    model: regen.GastricReferenceModel,
    observation_rotation: np.ndarray,
) -> np.ndarray:
    world_points = model.world_center + ((observation_rotation.T @ (observed_points - model.world_center).T).T)
    canonical_points = (model.world_basis.T @ (world_points - model.world_center).T).T
    return canonical_points.astype(np.float64)


def _build_shared_base_mesh(
    relaxed_pointcloud_path: Path,
    surface_cfg: SurfaceModelConfig,
    phase_bin_step_seconds: float,
) -> tuple[trimesh.Trimesh, MeshBuildResult]:
    results = reconstruct_meshes_from_pointclouds(
        [relaxed_pointcloud_path],
        surface_cfg,
        phase_bin_step_seconds=phase_bin_step_seconds,
    )
    if not results:
        raise RuntimeError("Failed to reconstruct relaxed base mesh from phase 0 point cloud")
    result = results[0]
    mesh = trimesh.load(result.mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Failed to load reconstructed base mesh: {result.mesh_path}")
    return mesh, result


def _load_shared_base_mesh(mesh_path: Path, relaxed_pointcloud_path: Path) -> tuple[trimesh.Trimesh, MeshBuildResult]:
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Failed to load provided base mesh: {mesh_path}")
    input_points = int(len(_read_xyz_ply(relaxed_pointcloud_path)))
    return mesh, MeshBuildResult(
        pointcloud_path=relaxed_pointcloud_path,
        mesh_path=mesh_path,
        timestamp_s=0.0,
        input_points=input_points,
        sampled_points=input_points,
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        watertight=bool(mesh.is_watertight),
        method="provided_shared_base_mesh",
    )


def _shared_base_surface_config() -> SurfaceModelConfig:
    cfg = SurfaceModelConfig()
    cfg.out_subdir = "meshes"
    cfg.max_points = 11000
    cfg.train_steps = 220
    cfg.mesh_resolution = 92
    cfg.smoothing_iterations = 36
    cfg.normal_neighbors = 48
    return cfg


def _hybrid_phase_surface_config() -> SurfaceModelConfig:
    cfg = SurfaceModelConfig()
    cfg.out_subdir = "_hybrid_phase_recon"
    cfg.max_points = 9000
    cfg.train_steps = 220
    cfg.mesh_resolution = 84
    cfg.smoothing_iterations = 20
    cfg.normal_neighbors = 36
    return cfg


def _refine_shared_base_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    original = mesh.copy()
    if hasattr(original, "unique_faces"):
        original.update_faces(original.unique_faces())
    if hasattr(original, "nondegenerate_faces"):
        original.update_faces(original.nondegenerate_faces())
    original.remove_unreferenced_vertices()
    original.remove_infinite_values()
    original.fix_normals()

    refined = original.copy()
    target_volume = float(abs(refined.volume)) if refined.is_volume else None
    target_centroid = np.asarray(refined.centroid, dtype=np.float64)
    if hasattr(refined, "unique_faces"):
        refined.update_faces(refined.unique_faces())
    if hasattr(refined, "nondegenerate_faces"):
        refined.update_faces(refined.nondegenerate_faces())
    refined.remove_unreferenced_vertices()

    edge_lengths = np.asarray(getattr(refined, "edges_unique_length", np.array([], dtype=np.float64)), dtype=np.float64)
    if edge_lengths.size > 0 and len(refined.vertices) < SHARED_BASE_VERTEX_SOFT_LIMIT:
        target_edge = float(np.percentile(edge_lengths, 90))
        max_edge = target_edge * 0.93
        if np.isfinite(max_edge) and max_edge > 1e-6 and float(np.max(edge_lengths)) > max_edge * 1.001:
            refined = refined.subdivide_to_size(max_edge=max_edge)
            if hasattr(refined, "unique_faces"):
                refined.update_faces(refined.unique_faces())
            if hasattr(refined, "nondegenerate_faces"):
                refined.update_faces(refined.nondegenerate_faces())
            refined.remove_unreferenced_vertices()

    edge_lengths = np.asarray(getattr(refined, "edges_unique_length", np.array([], dtype=np.float64)), dtype=np.float64)
    if edge_lengths.size > 0 and len(refined.vertices) < SHARED_BASE_VERTEX_SOFT_LIMIT:
        target_edge = float(np.percentile(edge_lengths, 88))
        max_edge = target_edge * 0.96
        if np.isfinite(max_edge) and max_edge > 1e-6 and float(np.max(edge_lengths)) > max_edge * 1.001:
            refined = refined.subdivide_to_size(max_edge=max_edge)
            if hasattr(refined, "unique_faces"):
                refined.update_faces(refined.unique_faces())
            if hasattr(refined, "nondegenerate_faces"):
                refined.update_faces(refined.nondegenerate_faces())
            refined.remove_unreferenced_vertices()
            if len(refined.vertices) > SHARED_BASE_VERTEX_HARD_LIMIT:
                refined = original.copy()
                if hasattr(refined, "unique_faces"):
                    refined.update_faces(refined.unique_faces())
                if hasattr(refined, "nondegenerate_faces"):
                    refined.update_faces(refined.nondegenerate_faces())
                refined.remove_unreferenced_vertices()
                refined.remove_infinite_values()
                refined.fix_normals()

    trimesh.smoothing.filter_taubin(refined, lamb=0.14, nu=-0.15, iterations=48)
    if hasattr(trimesh.smoothing, "filter_humphrey"):
        trimesh.smoothing.filter_humphrey(refined, alpha=0.010, beta=0.12, iterations=18)
    trimesh.smoothing.filter_taubin(refined, lamb=0.12, nu=-0.13, iterations=28)
    refined.remove_infinite_values()
    refined.remove_unreferenced_vertices()
    refined.fix_normals()
    if target_volume is not None and refined.is_volume:
        current_volume = float(abs(refined.volume))
        if current_volume > 1e-8:
            scale = (target_volume / current_volume) ** (1.0 / 3.0)
            refined.vertices = (refined.vertices - target_centroid) * scale + target_centroid
            refined.fix_normals()
    if not refined.is_watertight:
        trimesh.repair.fill_holes(refined)
    if original.is_watertight and not refined.is_watertight:
        return original
    return refined


def _postprocess_shared_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    processed = mesh.copy()
    target_volume = float(abs(processed.volume)) if processed.is_volume else None
    target_centroid = np.asarray(processed.centroid, dtype=np.float64)
    if hasattr(processed, "unique_faces"):
        processed.update_faces(processed.unique_faces())
    if hasattr(processed, "nondegenerate_faces"):
        processed.update_faces(processed.nondegenerate_faces())
    processed.remove_unreferenced_vertices()
    if len(processed.vertices) > 50000:
        trimesh.smoothing.filter_taubin(processed, lamb=0.42, nu=-0.43, iterations=72)
    else:
        trimesh.smoothing.filter_taubin(processed, lamb=0.37, nu=-0.38, iterations=50)
        if hasattr(trimesh.smoothing, "filter_humphrey"):
            trimesh.smoothing.filter_humphrey(processed, alpha=0.07, beta=0.56, iterations=10)
        trimesh.smoothing.filter_taubin(processed, lamb=0.41, nu=-0.42, iterations=28)
    processed.remove_infinite_values()
    processed.remove_unreferenced_vertices()
    processed.fix_normals()
    if target_volume is not None and processed.is_volume:
        current_volume = float(abs(processed.volume))
        if current_volume > 1e-8:
            scale = (target_volume / current_volume) ** (1.0 / 3.0)
            processed.vertices = (processed.vertices - target_centroid) * scale + target_centroid
            processed.fix_normals()
    if not processed.is_watertight:
        trimesh.repair.fill_holes(processed)
    return processed


def _cleanup_shared_phase_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    cleaned = mesh.copy()
    target_volume = float(abs(cleaned.volume)) if cleaned.is_volume else None
    target_centroid = np.asarray(cleaned.centroid, dtype=np.float64)
    if hasattr(cleaned, "unique_faces"):
        cleaned.update_faces(cleaned.unique_faces())
    if hasattr(cleaned, "nondegenerate_faces"):
        cleaned.update_faces(cleaned.nondegenerate_faces())
    if hasattr(cleaned, "merge_vertices"):
        cleaned.merge_vertices(digits_vertex=7)
    cleaned.remove_unreferenced_vertices()
    cleaned.remove_infinite_values()
    cleaned.fix_normals()
    for _ in range(3):
        if cleaned.is_watertight:
            break
        trimesh.repair.fill_holes(cleaned)
        trimesh.repair.fix_winding(cleaned)
        trimesh.repair.fix_inversion(cleaned, multibody=True)
        trimesh.repair.fix_normals(cleaned, multibody=True)
        if hasattr(cleaned, "unique_faces"):
            cleaned.update_faces(cleaned.unique_faces())
        if hasattr(cleaned, "nondegenerate_faces"):
            cleaned.update_faces(cleaned.nondegenerate_faces())
        cleaned.remove_unreferenced_vertices()
        cleaned.remove_infinite_values()
        cleaned.fix_normals()
    if cleaned.is_watertight:
        trimesh.smoothing.filter_taubin(cleaned, lamb=0.14, nu=-0.15, iterations=24)
        if hasattr(trimesh.smoothing, "filter_humphrey"):
            trimesh.smoothing.filter_humphrey(cleaned, alpha=0.018, beta=0.22, iterations=7)
        cleaned.remove_infinite_values()
        cleaned.remove_unreferenced_vertices()
        cleaned.fix_normals()
    if target_volume is not None and cleaned.is_volume:
        current_volume = float(abs(cleaned.volume))
        if current_volume > 1e-8:
            scale = (target_volume / current_volume) ** (1.0 / 3.0)
            cleaned.vertices = (cleaned.vertices - target_centroid) * scale + target_centroid
            cleaned.fix_normals()
    return cleaned


def _select_hybrid_remesh_phase_indices(stats_rows: list[PhaseModelStats], config: HybridRemeshConfig) -> list[int]:
    if not config.enabled or not stats_rows:
        return []
    peak_contraction = max(float(row.max_contraction) for row in stats_rows)
    if peak_contraction <= 1e-8:
        return []

    selected = {
        row.phase_index
        for row in stats_rows
        if float(row.max_contraction) >= peak_contraction * float(config.peak_ratio_threshold)
    }
    if not selected:
        return []

    expanded: set[int] = set()
    phase_count = len(stats_rows)
    for phase_index in selected:
        for offset in range(-config.neighbor_span, config.neighbor_span + 1):
            expanded.add(int(np.clip(phase_index + offset, 0, phase_count - 1)))
    return sorted(expanded)


def _apply_hybrid_phase_reconstruction(
    mesh_dir: Path,
    pointcloud_paths: list[Path],
    results: list[MeshBuildResult],
    selected_phase_indices: list[int],
    phase_bin_step_seconds: float,
) -> list[MeshBuildResult]:
    if not selected_phase_indices:
        return results

    hybrid_cfg = _hybrid_phase_surface_config()
    selected_pointclouds = [pointcloud_paths[idx] for idx in selected_phase_indices]
    temp_results = reconstruct_meshes_from_pointclouds(
        selected_pointclouds,
        hybrid_cfg,
        phase_bin_step_seconds=phase_bin_step_seconds,
    )
    temp_map = {result.pointcloud_path.name: result for result in temp_results}
    updated_results: list[MeshBuildResult] = []
    for result in results:
        override = temp_map.get(result.pointcloud_path.name)
        if override is None:
            updated_results.append(result)
            continue

        override_mesh = trimesh.load(override.mesh_path, force="mesh")
        if not isinstance(override_mesh, trimesh.Trimesh):
            updated_results.append(result)
            continue
        override_mesh = _postprocess_shared_mesh(override_mesh)
        final_mesh_path = mesh_dir / result.mesh_path.name
        _mesh_export(override_mesh, final_mesh_path)
        updated_results.append(
            MeshBuildResult(
                pointcloud_path=result.pointcloud_path,
                mesh_path=final_mesh_path,
                timestamp_s=result.timestamp_s,
                input_points=override.input_points,
                sampled_points=override.sampled_points,
                vertices=int(len(override_mesh.vertices)),
                faces=int(len(override_mesh.faces)),
                watertight=bool(override_mesh.is_watertight),
                method="hybrid_local_phase_reconstruction",
            )
        )
        print(
            f"[PhaseModels][HybridRemesh] phase={result.timestamp_s if result.timestamp_s is not None else -1.0:.3f}s -> {final_mesh_path.name} "
            f"({len(override_mesh.vertices)} verts, {len(override_mesh.faces)} faces, watertight={override_mesh.is_watertight})"
        )

    temp_dir = pointcloud_paths[0].parent / hybrid_cfg.out_subdir
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    return updated_results


def _export_shared_topology_meshes(
    base_mesh: trimesh.Trimesh,
    base_mesh_result: MeshBuildResult,
    pointcloud_paths: list[Path],
    phase_values: np.ndarray,
    model: regen.GastricReferenceModel,
    observation_rotation: np.ndarray,
    mesh_dir: Path,
    phase_bin_step_seconds: float,
    narrowing_scale: float,
) -> list[MeshBuildResult]:
    mesh_dir.mkdir(parents=True, exist_ok=True)
    base_canonical_vertices = _observed_to_canonical_points(
        np.asarray(base_mesh.vertices, dtype=np.float64),
        model,
        observation_rotation,
    )

    results: list[MeshBuildResult] = []
    for phase_index, (pointcloud_path, phase_value) in enumerate(zip(pointcloud_paths, phase_values)):
        deformed_vertices, _, _ = _deform_reference_points(
            base_canonical_vertices,
            model,
            float(phase_value),
            observation_rotation,
            narrowing_scale,
        )
        mesh = trimesh.Trimesh(vertices=deformed_vertices, faces=np.asarray(base_mesh.faces, dtype=np.int64), process=False)
        mesh = _cleanup_shared_phase_mesh(mesh)

        mesh_path = mesh_dir / pointcloud_path.name.replace(".ply", "_mesh.ply")
        _mesh_export(mesh, mesh_path)
        input_points = int(len(_read_xyz_ply(pointcloud_path)))
        sampled_points = int(min(input_points, base_mesh_result.sampled_points))
        timestamp_s = float(_phase_timestamp_seconds(phase_index, phase_bin_step_seconds))
        results.append(
            MeshBuildResult(
                pointcloud_path=pointcloud_path,
                mesh_path=mesh_path,
                timestamp_s=timestamp_s,
                input_points=input_points,
                sampled_points=sampled_points,
                vertices=int(len(mesh.vertices)),
                faces=int(len(mesh.faces)),
                watertight=bool(mesh.is_watertight),
                method="shared_topology_deformation",
            )
        )
        print(
            f"[PhaseModels][SharedMesh] phase={phase_value:.3f} -> {mesh_path.name} "
            f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight})"
        )

    summary_path = mesh_dir / "mesh_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase_pointcloud", "timestamp_s", "mesh", "input_points", "sampled_points", "vertices", "faces", "watertight", "method"])
        for result in results:
            writer.writerow([
                result.pointcloud_path.name,
                "" if result.timestamp_s is None else f"{result.timestamp_s:.6f}",
                result.mesh_path.name,
                result.input_points,
                result.sampled_points,
                result.vertices,
                result.faces,
                int(result.watertight),
                result.method,
            ])
    return results


def _write_pointcloud_ply(points: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f}\n")


def _export_cycle_shared_topology_meshes(
    base_mesh: trimesh.Trimesh,
    phase_vertices: list[np.ndarray],
    pointcloud_paths: list[Path],
    observation_center: np.ndarray,
    observation_rotation: np.ndarray,
    mesh_dir: Path,
    phase_bin_step_seconds: float,
) -> list[MeshBuildResult]:
    mesh_dir.mkdir(parents=True, exist_ok=True)
    results: list[MeshBuildResult] = []
    faces = np.asarray(base_mesh.faces, dtype=np.int64)
    for phase_index, (pointcloud_path, vertices) in enumerate(zip(pointcloud_paths, phase_vertices)):
        observed_vertices = _apply_observation_transform(vertices, observation_center, observation_rotation)
        mesh = trimesh.Trimesh(vertices=observed_vertices, faces=faces, process=False)
        mesh = _cleanup_shared_phase_mesh(mesh)
        mesh_path = mesh_dir / pointcloud_path.name.replace(".ply", "_mesh.ply")
        _mesh_export(mesh, mesh_path)
        input_points = int(len(_read_xyz_ply(pointcloud_path)))
        timestamp_s = float(_phase_timestamp_seconds(phase_index, phase_bin_step_seconds))
        results.append(
            MeshBuildResult(
                pointcloud_path=pointcloud_path,
                mesh_path=mesh_path,
                timestamp_s=timestamp_s,
                input_points=input_points,
                sampled_points=input_points,
                vertices=int(len(mesh.vertices)),
                faces=int(len(mesh.faces)),
                watertight=bool(mesh.is_watertight),
                method="shared_topology_deformation",
            )
        )

    summary_path = mesh_dir / "mesh_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase_pointcloud", "timestamp_s", "mesh", "input_points", "sampled_points", "vertices", "faces", "watertight", "method"])
        for result in results:
            writer.writerow([
                result.pointcloud_path.name,
                "" if result.timestamp_s is None else f"{result.timestamp_s:.6f}",
                result.mesh_path.name,
                result.input_points,
                result.sampled_points,
                result.vertices,
                result.faces,
                int(result.watertight),
                result.method,
            ])
    return results


def generate_phase_models_for_instance(
    instance_name: str | None,
    reference_ply: Path | None,
    monitor_stream: Path | None,
    output_base_dir: Path | None,
    gt_mesh_dir: Path | None,
    base_mesh_path: Path | None,
    preserve_provided_base_mesh: bool,
    phase_count_override: int | None,
    narrowing_scale: float,
    cycle_model_config: CycleModelConfig,
    hybrid_remesh: HybridRemeshConfig,
    sync_gt: bool,
) -> Path:
    instance_paths = resolve_instance_paths(instance_name=instance_name, reference_ply=reference_ply)
    resolved_reference_ply = instance_paths.reference_ply
    resolved_monitor_stream = resolve_monitor_input_path(instance_paths, explicit_path=monitor_stream)
    resolved_output_base_dir = output_base_dir if output_base_dir is not None else instance_paths.phase_model_base_dir
    resolved_gt_mesh_dir = gt_mesh_dir if gt_mesh_dir is not None else instance_paths.gt_mesh_dir

    if not resolved_reference_ply.exists():
        raise FileNotFoundError(f"Reference stomach point cloud not found: {resolved_reference_ply}")
    if not resolved_monitor_stream.exists():
        raise FileNotFoundError(f"Monitor stream not found: {resolved_monitor_stream}")

    phase_bin_step_seconds = float(PipelineConfig().phase_detection.phase_bin_step_seconds)
    if phase_bin_step_seconds <= 0.0:
        raise ValueError("phase_bin_step_seconds must be positive")

    if phase_count_override is None:
        phase_count, avg_duration = _detect_monitor_aligned_phase_count(resolved_monitor_stream, phase_bin_step_seconds)
    else:
        phase_count = int(phase_count_override)
        avg_duration = float("nan")
    if phase_count < 2:
        raise ValueError("phase-count must be at least 2")

    run_dir, run_id = _create_indexed_output_dir(resolved_output_base_dir, OUTPUT_PREFIX)
    points_dir = run_dir / "pointclouds"
    points_dir.mkdir(parents=True, exist_ok=True)

    scaled_cycle_cfg = CycleModelConfig(
        grid_resolution=cycle_model_config.grid_resolution,
        base_smooth_iterations=cycle_model_config.base_smooth_iterations,
        centerline_samples=cycle_model_config.centerline_samples,
        body_contraction=cycle_model_config.body_contraction * narrowing_scale,
        pylorus_contraction=cycle_model_config.pylorus_contraction * narrowing_scale,
        wave_width=cycle_model_config.wave_width,
        wave_start_u=cycle_model_config.wave_start_u,
        wave_end_u=cycle_model_config.wave_end_u,
        deformation_smooth_iterations=cycle_model_config.deformation_smooth_iterations,
        deformation_smooth_relax=cycle_model_config.deformation_smooth_relax,
        post_smooth_iterations=cycle_model_config.post_smooth_iterations,
        post_smooth_neighborhood_order=cycle_model_config.post_smooth_neighborhood_order,
        reverse_centerline=cycle_model_config.reverse_centerline,
    )
    reference_points, base_mesh, base_mesh_result, vertex_mapping, adjacency, post_smooth_adjacency, _, _ = _build_cycle_base_mesh_and_mapping(
        resolved_reference_ply,
        scaled_cycle_cfg,
        base_mesh_path,
    )

    observation_rotation = _rotation_matrix_xyz(OBSERVATION_ROTATION_DEG)
    observation_center = np.mean(reference_points, axis=0).astype(np.float64)
    transform_path = _write_observation_transform(run_dir, observation_center, observation_rotation, phase_bin_step_seconds)

    phase_values = np.linspace(0.0, 1.0, phase_count, endpoint=False, dtype=np.float64)

    pointcloud_paths: list[Path] = []
    stats_rows: list[PhaseModelStats] = []
    phase_vertex_cache: list[np.ndarray] = []
    for phase_index, phase_value in enumerate(phase_values):
        timestamp_s = _phase_timestamp_seconds(phase_index, phase_bin_step_seconds)
        deformed_vertices, wave_center, max_contraction = _deform_cycle_phase_vertices(
            base_mesh,
            vertex_mapping,
            float(phase_value),
            scaled_cycle_cfg,
            adjacency,
            post_smooth_adjacency,
        )
        phase_vertex_cache.append(deformed_vertices)
        observed_points = _apply_observation_transform(deformed_vertices, observation_center, observation_rotation)
        pointcloud_path = points_dir / (
            f"{instance_paths.name}_run_{run_id}_phase_{phase_index:03d}_{phase_value:.3f}_{_format_timestamp_token(timestamp_s)}.ply"
        )
        _write_pointcloud_ply(observed_points.astype(np.float32), pointcloud_path)
        pointcloud_paths.append(pointcloud_path)
        stats_rows.append(
            PhaseModelStats(
                phase_index=phase_index,
                phase_value=float(phase_value),
                timestamp_s=timestamp_s,
                wave_center_s=float(wave_center),
                max_contraction=float(max_contraction),
                pointcloud_path=pointcloud_path,
            )
        )

    observed_reference_points = _apply_observation_transform(reference_points, observation_center, observation_rotation)
    observed_reference_path = points_dir / f"{instance_paths.name}_run_{run_id}_reference_observed.ply"
    _write_pointcloud_ply(observed_reference_points.astype(np.float32), observed_reference_path)

    surface_cfg = _shared_base_surface_config()
    meshes = _export_cycle_shared_topology_meshes(
        base_mesh,
        phase_vertex_cache,
        pointcloud_paths,
        observation_center,
        observation_rotation,
        points_dir / surface_cfg.out_subdir,
        phase_bin_step_seconds,
    )
    hybrid_phase_indices = _select_hybrid_remesh_phase_indices(stats_rows, hybrid_remesh)
    if hybrid_phase_indices:
        meshes = _apply_hybrid_phase_reconstruction(
            points_dir / surface_cfg.out_subdir,
            pointcloud_paths,
            meshes,
            hybrid_phase_indices,
            phase_bin_step_seconds,
        )

    summary_path = run_dir / "phase_sequence_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase_index", "phase_value", "timestamp_s", "wave_center_s", "max_contraction", "pointcloud"])
        for row in stats_rows:
            writer.writerow([
                row.phase_index,
                f"{row.phase_value:.6f}",
                f"{row.timestamp_s:.6f}",
                f"{row.wave_center_s:.6f}",
                f"{row.max_contraction:.6f}",
                row.pointcloud_path.name,
            ])

    archive_dir: Path | None = None
    synced_paths: list[Path] = []
    if sync_gt:
        archive_dir, synced_paths = _sync_meshes_to_gt(meshes, resolved_gt_mesh_dir, run_dir, run_dir / "pointclouds" / "meshes" / "mesh_summary.csv")

    print(f"[PhaseModels] Instance: {instance_paths.name}")
    print(f"[PhaseModels] Reference: {resolved_reference_ply}")
    print(f"[PhaseModels] Monitor stream: {resolved_monitor_stream}")
    print(f"[PhaseModels] Output directory: {run_dir}")
    print(f"[PhaseModels] Phase count: {phase_count}")
    print(f"[PhaseModels] Phase bin step: {phase_bin_step_seconds:.6f}s")
    print(f"[PhaseModels] Narrowing scale: {narrowing_scale:.3f}")
    if not np.isnan(avg_duration):
        print(f"[PhaseModels] Monitor-aligned average cycle duration: {avg_duration:.6f}s")
    print(f"[PhaseModels] Point clouds: {len(pointcloud_paths)}")
    print(f"[PhaseModels] Meshes: {len(meshes)}")
    if hybrid_phase_indices:
        print(f"[PhaseModels] Hybrid remesh phases: {','.join(str(idx) for idx in hybrid_phase_indices)}")
    print(f"[PhaseModels] Summary: {summary_path}")
    print(f"[PhaseModels] Observation transform: {transform_path}")
    if sync_gt:
        print(f"[PhaseModels] Synced GT meshes: {len(synced_paths)} -> {resolved_gt_mesh_dir}")
        if archive_dir is not None:
            print(f"[PhaseModels] Archived previous GT meshes to: {archive_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a gastric peristaltic phase-sequence model set")
    parser.add_argument("--instance-name", type=str, default=None, help="Named stomach instance under stomach_pcd")
    parser.add_argument("--reference-ply", type=str, default=None, help="Explicit reference stomach point cloud path")
    parser.add_argument("--monitor-path", type=str, default=None, help="Monitor stream used to determine phase count")
    parser.add_argument("--base-mesh-path", type=str, default=None, help="Optional existing phase-0 mesh to reuse as the shared base mesh template")
    parser.add_argument(
        "--preserve-provided-base-mesh",
        action="store_true",
        help="When reusing --base-mesh-path, skip base-mesh refinement so the provided mesh topology and smoothness are preserved",
    )
    parser.add_argument("--phase-count", type=int, default=None, help="Number of phase models to generate across one cycle; defaults to monitor-aligned bin count")
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="",
        help="Directory where the generated phase-sequence model run folder will be created; defaults to the instance-specific simuilate_data directory",
    )
    parser.add_argument("--gt-mesh-dir", type=str, default="", help="Optional explicit GT mesh sync directory")
    parser.add_argument("--batch-all-references", action="store_true", help="Generate phase-sequence models for all point clouds under stomach_pcd")
    parser.add_argument("--no-sync-gt", action="store_true", help="Do not sync generated meshes into the instance GT mesh directory")
    parser.add_argument(
        "--narrowing-scale",
        type=float,
        default=1.0,
        help="Global scale factor applied to body and pylorus contraction amplitudes in the migrated centerline-driven cycle model",
    )
    parser.add_argument("--grid-resolution", type=int, default=144, help="Longest-axis voxel resolution used for migrated surface reconstruction")
    parser.add_argument("--base-smooth-iterations", type=int, default=30, help="Taubin smoothing iterations applied to the migrated rest mesh reconstruction")
    parser.add_argument("--centerline-samples", type=int, default=31, help="Number of samples along the migrated fitted centerline")
    parser.add_argument("--body-contraction", type=float, default=0.18, help="Maximum ring contraction ratio in the gastric body")
    parser.add_argument("--pylorus-contraction", type=float, default=0.30, help="Maximum contraction ratio near the pylorus")
    parser.add_argument("--wave-width", type=float, default=0.09, help="Longitudinal width of the traveling peristaltic ring")
    parser.add_argument("--wave-start-u", type=float, default=0.32, help="Normalized centerline position where the contraction wave starts")
    parser.add_argument("--wave-end-u", type=float, default=0.98, help="Normalized centerline position where the contraction wave ends")
    parser.add_argument("--deformation-smooth-iterations", type=int, default=11, help="Neighbor smoothing iterations applied to the migrated contraction field")
    parser.add_argument("--deformation-smooth-relax", type=float, default=0.46, help="Relaxation factor for migrated contraction-field smoothing")
    parser.add_argument("--post-smooth-iterations", type=int, default=14, help="Light Taubin smoothing iterations applied after each migrated phase deformation")
    parser.add_argument("--post-smooth-neighborhood-order", type=int, default=8, help="Neighborhood ring order used by migrated post-process mesh smoothing")
    parser.add_argument(
        "--reverse-centerline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reverse the extracted migrated centerline orientation before propagating the wave",
    )
    parser.add_argument(
        "--hybrid-strong-phase-remesh",
        action="store_true",
        help="For strong-contraction phases, replace shared-topology export with per-phase NeuralGF reconstruction to test a local remeshing route",
    )
    parser.add_argument(
        "--hybrid-remesh-peak-ratio-threshold",
        type=float,
        default=0.72,
        help="Relative peak-contraction threshold used to select strong phases for hybrid remeshing",
    )
    parser.add_argument(
        "--hybrid-remesh-neighbor-span",
        type=int,
        default=2,
        help="Number of neighboring phase bins on each side to include around selected strong phases",
    )
    args = parser.parse_args()

    narrowing_scale = float(args.narrowing_scale)
    if not (0.0 < narrowing_scale <= 1.0):
        raise ValueError("narrowing-scale must be in (0, 1]")
    if not (0.0 <= float(args.wave_start_u) < float(args.wave_end_u) <= 1.0):
        raise ValueError("wave-start-u and wave-end-u must satisfy 0 <= start < end <= 1")
    if not (0.0 < float(args.hybrid_remesh_peak_ratio_threshold) <= 1.0):
        raise ValueError("hybrid-remesh-peak-ratio-threshold must be in (0, 1]")
    if int(args.hybrid_remesh_neighbor_span) < 0:
        raise ValueError("hybrid-remesh-neighbor-span must be >= 0")

    output_base_dir = Path(args.output_base_dir).expanduser().resolve() if args.output_base_dir else None
    gt_mesh_dir = Path(args.gt_mesh_dir).expanduser().resolve() if args.gt_mesh_dir else None
    monitor_stream = Path(args.monitor_path).expanduser().resolve() if args.monitor_path else None
    base_mesh_path = Path(args.base_mesh_path).expanduser().resolve() if args.base_mesh_path else None
    hybrid_remesh = HybridRemeshConfig(
        enabled=bool(args.hybrid_strong_phase_remesh),
        peak_ratio_threshold=float(args.hybrid_remesh_peak_ratio_threshold),
        neighbor_span=int(args.hybrid_remesh_neighbor_span),
    )
    cycle_model_config = CycleModelConfig(
        grid_resolution=int(args.grid_resolution),
        base_smooth_iterations=int(args.base_smooth_iterations),
        centerline_samples=int(args.centerline_samples),
        body_contraction=float(args.body_contraction),
        pylorus_contraction=float(args.pylorus_contraction),
        wave_width=float(args.wave_width),
        wave_start_u=float(args.wave_start_u),
        wave_end_u=float(args.wave_end_u),
        deformation_smooth_iterations=int(args.deformation_smooth_iterations),
        deformation_smooth_relax=float(args.deformation_smooth_relax),
        post_smooth_iterations=int(args.post_smooth_iterations),
        post_smooth_neighborhood_order=int(args.post_smooth_neighborhood_order),
        reverse_centerline=bool(args.reverse_centerline),
    )

    if args.batch_all_references:
        reference_paths = list_reference_pointclouds()
        if not reference_paths:
            raise FileNotFoundError("No reference point clouds found under stomach_pcd")
        for reference_path in reference_paths:
            instance_output_dir = (output_base_dir / reference_path.stem) if output_base_dir is not None else None
            instance_gt_mesh_dir = (gt_mesh_dir / reference_path.stem) if gt_mesh_dir is not None else None
            generate_phase_models_for_instance(
                instance_name=reference_path.stem,
                reference_ply=reference_path,
                monitor_stream=monitor_stream,
                output_base_dir=instance_output_dir,
                gt_mesh_dir=instance_gt_mesh_dir,
                base_mesh_path=base_mesh_path,
                preserve_provided_base_mesh=args.preserve_provided_base_mesh,
                phase_count_override=args.phase_count,
                narrowing_scale=narrowing_scale,
                cycle_model_config=cycle_model_config,
                hybrid_remesh=hybrid_remesh,
                sync_gt=not args.no_sync_gt,
            )
        return

    generate_phase_models_for_instance(
        instance_name=args.instance_name,
        reference_ply=Path(args.reference_ply).expanduser().resolve() if args.reference_ply else None,
        monitor_stream=monitor_stream,
        output_base_dir=output_base_dir,
        gt_mesh_dir=gt_mesh_dir,
        base_mesh_path=base_mesh_path,
        preserve_provided_base_mesh=args.preserve_provided_base_mesh,
        phase_count_override=args.phase_count,
        narrowing_scale=narrowing_scale,
        cycle_model_config=cycle_model_config,
        hybrid_remesh=hybrid_remesh,
        sync_gt=not args.no_sync_gt,
    )


if __name__ == "__main__":
    main()