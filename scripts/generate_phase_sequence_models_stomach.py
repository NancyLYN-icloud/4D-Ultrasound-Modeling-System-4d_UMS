from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import csv
import gc
import math
import shutil
import sys

import numpy as np
from scipy.spatial import cKDTree
import trimesh


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig, SurfaceModelConfig
from src.paths import data_path
from src.modeling.surface_reconstruction import MeshBuildResult, _mesh_export, _read_xyz_ply, reconstruct_meshes_from_pointclouds
from src.preprocessing.phase_detection import PhaseDetector
from src.stomach_instance_paths import resolve_instance_paths, resolve_monitor_input_path
import scripts.regenerate_freehand_scanner_sequence as regen


REFERENCE_PLY = data_path("benchmark", "stomach.ply")
OUTPUT_BASE_DIR = data_path("simulation_mesh")
OUTPUT_PREFIX = "phase_sequence_models_run"
OBSERVATION_ROTATION_DEG = np.array([-35.0, 20.0, -32.0], dtype=np.float64)
SHARED_BASE_VERTEX_SOFT_LIMIT = 13000
SHARED_BASE_VERTEX_HARD_LIMIT = 18000
LOCAL_ADAPTIVE_VERTEX_BUDGET = 2200
GLOBAL_MICRO_DENSIFY_BUDGET = 1600
DEFAULT_PHASE_COUNT = 61
LEGACY_PHASE_COUNT = 41
CONTRACTION_VISUAL_GAIN = 3.0
DEFAULT_CROSS_SECTION_S_VALUES = (0.58, 0.46, 0.34, 0.22, 0.12)
DEFAULT_HYBRID_REMESH_PEAK_RATIO_THRESHOLD = 0.94
DEFAULT_HYBRID_REMESH_NEIGHBOR_SPAN = 0


@dataclass
class PhaseModelStats:
    phase_index: int
    phase_value: float
    timestamp_s: float
    wave_center_s: float
    max_contraction: float
    pointcloud_path: Path


@dataclass
class CrossSectionDiagnostic:
    phase_index: int
    phase_value: float
    timestamp_s: float
    target_s: float
    sample_count: int
    band_half_width: float
    centroid_y: float
    centroid_z: float
    mean_radius: float
    p10_radius: float
    p90_radius: float


@dataclass
class WaveBandMeshDiagnostic:
    phase_index: int
    phase_value: float
    timestamp_s: float
    wave_center_s: float
    band_half_width: float
    section_half_width: float
    band_vertex_count: int
    band_face_count: int
    section_vertex_count: int
    occupied_angle_bins: int
    mean_vertices_per_occupied_bin: float
    max_vertices_in_bin: int
    local_mean_edge_length: float
    local_p90_edge_length: float


@dataclass
class HybridRemeshConfig:
    enabled: bool
    peak_ratio_threshold: float
    neighbor_span: int


@dataclass
class PatchRemeshGuidanceConfig:
    enabled: bool
    peak_ratio_threshold: float
    neighbor_span: int
    band_inner_width: float
    band_outer_width: float
    blend_strength: float


@dataclass
class LocalAdaptiveSubdivisionConfig:
    enabled: bool
    peak_ratio_threshold: float
    band_half_width: float
    face_score_threshold: float
    max_passes: int
    support_band_width: float
    support_weight: float
    edge_percentile: float
    distal_emphasis_threshold: float
    distal_emphasis_gain: float
    boundary_neighbor_rings: int


LOCAL_ADAPTIVE_SUBDIVISION = LocalAdaptiveSubdivisionConfig(
    enabled=True,
    peak_ratio_threshold=0.82,
    band_half_width=0.048,
    face_score_threshold=0.38,
    max_passes=1,
    support_band_width=0.14,
    support_weight=0.16,
    edge_percentile=54.0,
    distal_emphasis_threshold=0.24,
    distal_emphasis_gain=0.32,
    boundary_neighbor_rings=1,
)


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


def _parse_cross_section_s_values(raw_value: str) -> list[float]:
    values: list[float] = []
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        values.append(float(np.clip(value, 0.0, 1.0)))
    unique_values = sorted(set(values), reverse=True)
    if not unique_values:
        raise ValueError("cross-section-s-values must contain at least one numeric s value")
    return unique_values


def _smoothstep(edge0: float, edge1: float, value: float | np.ndarray) -> float | np.ndarray:
    span = max(edge1 - edge0, 1e-8)
    t = np.clip((value - edge0) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _oriented_arc_coordinate(model: regen.GastricReferenceModel, s: np.ndarray) -> np.ndarray:
    start_area = float(np.mean(model.radius_y[:8] * model.radius_z[:8]))
    end_area = float(np.mean(model.radius_y[-8:] * model.radius_z[-8:]))
    if start_area > end_area:
        return 1.0 - s
    return s


def _phase_wave(
    s: np.ndarray,
    phase: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    phase = float(phase % 1.0)

    onset = float(_smoothstep(0.05, 0.18, phase))
    travel = float(_smoothstep(0.16, 0.80, phase))
    wave_release = float(1.0 - _smoothstep(0.80, 0.94, phase))
    pyloric_hold = float(_smoothstep(0.74, 0.84, phase) * (1.0 - _smoothstep(0.90, 0.98, phase)))
    cycle_amp = onset * wave_release
    propagation_gain = 0.38 + 0.76 * travel

    pylorus_s = 0.12
    body_origin_s = 0.58
    wave_center = body_origin_s + (pylorus_s - body_origin_s) * travel

    core = np.exp(-0.5 * ((s - wave_center) / 0.016) ** 2)
    lead = np.exp(-0.5 * ((s - wave_center) / 0.024) ** 2)
    trail = np.exp(-0.5 * ((s - (wave_center + 0.048)) / 0.036) ** 2)
    recovery = np.exp(-0.5 * ((s - pylorus_s) / 0.062) ** 2)

    mid_body_seed = np.exp(-0.5 * ((s - body_origin_s) / 0.072) ** 2)
    pylorus_peak = np.exp(-0.5 * ((s - pylorus_s) / 0.030) ** 2)
    body_gate = 1.0 - _smoothstep(0.82, 0.97, s)
    distal_gain = 0.82 + 0.40 * np.clip((0.44 - s) / 0.34, 0.0, 1.0)
    body_seed_gain = 0.94 + 0.40 * mid_body_seed * (1.0 - 0.55 * travel)

    contraction = (
        cycle_amp
        * propagation_gain
        * (1.16 * core + 0.24 * lead + 0.12 * trail)
        * body_gate
        * body_seed_gain
        * distal_gain
        + 0.92 * pyloric_hold * pylorus_peak
    )
    contraction = np.clip(contraction, 0.0, 1.02)
    return (
        contraction.astype(np.float64),
        float(wave_center),
        core.astype(np.float64),
        lead.astype(np.float64),
        trail.astype(np.float64),
        recovery.astype(np.float64),
        cycle_amp,
    )


def _deform_reference_points(
    canonical_points: np.ndarray,
    model: regen.GastricReferenceModel,
    phase: float,
    observation_rotation: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    s, _, local_y, local_z = regen.project_canonical_points(model, canonical_points)
    oriented_s = _oriented_arc_coordinate(model, s)
    contraction, wave_center, core, lead, trail, recovery, cycle_amp = _phase_wave(oriented_s, phase)
    amplified_contraction = np.clip(contraction * CONTRACTION_VISUAL_GAIN, 0.0, 3.06)
    amplified_cycle_amp = float(min(3.0, cycle_amp * CONTRACTION_VISUAL_GAIN))

    deformed_canonical = np.empty_like(canonical_points, dtype=np.float64)

    max_contraction = float(np.max(amplified_contraction))
    for idx in range(canonical_points.shape[0]):
        si = float(s[idx])
        center, tangent, frame_y, ry0, rz0 = regen._interp_profile(model, si)
        frame_z = np.cross(tangent, frame_y)
        frame_z /= np.linalg.norm(frame_z) + 1e-8
        dy = float(local_y[idx])
        dz = float(local_z[idx])

        local_contraction = float(amplified_contraction[idx])
        local_ring_driver = 1.08 * local_contraction + 0.32 * amplified_cycle_amp * core[idx]
        ring_scale = max(0.035, 1.0 - (1.02 + 0.17 * (1.0 - si)) * local_ring_driver)
        pyloric_taper = 1.0 - 0.42 * math.exp(-((si - 0.06) / 0.042) ** 2) * (0.35 + 0.65 * min(1.35, amplified_cycle_amp))

        axial_shift = -1.7 * amplified_cycle_amp * core[idx] - 3.9 * local_contraction * math.exp(-((si - 0.22) / 0.13) ** 2)
        distal_pull = -10.2 * local_contraction * math.exp(-((si - 0.11) / 0.072) ** 2)
        wall_push = (0.88 * core[idx] + 0.20 * lead[idx] + 0.06 * trail[idx]) * amplified_cycle_amp * 0.30

        axial_offset = axial_shift + distal_pull + wall_push
        lateral_offset = dy * ring_scale * pyloric_taper
        normal_offset = dz * ring_scale * pyloric_taper

        deformed_canonical[idx] = center + axial_offset * tangent + lateral_offset * frame_y + normal_offset * frame_z

    canonical_deformed = deformed_canonical
    world_deformed = model.world_center + (model.world_basis @ canonical_deformed.T).T
    observed_points = _apply_observation_transform(world_deformed, model.world_center, observation_rotation)
    return observed_points.astype(np.float32), wave_center, max_contraction


def _select_diagnostic_phase_indices(stats_rows: list[PhaseModelStats]) -> list[int]:
    if not stats_rows:
        return []
    peak_index = max(range(len(stats_rows)), key=lambda idx: stats_rows[idx].max_contraction)
    candidates = {
        0,
        max(0, peak_index - 8),
        max(0, peak_index - 4),
        peak_index,
        min(len(stats_rows) - 1, peak_index + 4),
        min(len(stats_rows) - 1, peak_index + 8),
        len(stats_rows) - 1,
    }
    return sorted(candidates)


def _collect_cross_section_points(
    canonical_points: np.ndarray,
    model: regen.GastricReferenceModel,
    target_s: float,
) -> tuple[np.ndarray, float]:
    section_s, _, section_y, section_z = regen.project_canonical_points(model, canonical_points)
    selected_points = np.empty((0, 2), dtype=np.float64)
    selected_band = 0.020
    for band_half_width in (0.006, 0.010, 0.014, 0.020):
        mask = np.abs(section_s - target_s) <= band_half_width
        if int(np.count_nonzero(mask)) >= 48:
            selected_points = np.column_stack([section_y[mask], section_z[mask]]).astype(np.float64)
            selected_band = band_half_width
            break
        if int(np.count_nonzero(mask)) > 0:
            selected_points = np.column_stack([section_y[mask], section_z[mask]]).astype(np.float64)
            selected_band = band_half_width
    return selected_points, selected_band


def _build_cross_section_outline(points_2d: np.ndarray, radial_bins: int = 72) -> np.ndarray:
    if len(points_2d) < 8:
        return points_2d.astype(np.float64)
    centroid = np.mean(points_2d, axis=0)
    shifted = points_2d - centroid
    angles = np.arctan2(shifted[:, 1], shifted[:, 0])
    radii = np.linalg.norm(shifted, axis=1)
    bin_edges = np.linspace(-math.pi, math.pi, radial_bins + 1)
    outline_points: list[np.ndarray] = []
    for bin_index in range(radial_bins):
        mask = (angles >= bin_edges[bin_index]) & (angles < bin_edges[bin_index + 1])
        if not np.any(mask):
            continue
        angle_mid = 0.5 * (bin_edges[bin_index] + bin_edges[bin_index + 1])
        radius = float(np.percentile(radii[mask], 90.0))
        outline_points.append(centroid + radius * np.array([math.cos(angle_mid), math.sin(angle_mid)], dtype=np.float64))
    if len(outline_points) < 3:
        return points_2d.astype(np.float64)
    return np.asarray(outline_points, dtype=np.float64)


def _write_cross_section_svg(
    out_path: Path,
    section_payloads: list[tuple[float, np.ndarray, np.ndarray]],
    title: str,
) -> None:
    panel_width = 260
    panel_height = 260
    margin = 28
    title_height = 36
    width = len(section_payloads) * panel_width
    height = title_height + panel_height

    max_extent = 1.0
    for _, points_2d, outline in section_payloads:
        if len(points_2d):
            max_extent = max(max_extent, float(np.max(np.abs(points_2d))))
        if len(outline):
            max_extent = max(max_extent, float(np.max(np.abs(outline))))
    max_extent *= 1.15
    scale = (panel_width * 0.5 - margin) / max(max_extent, 1e-6)

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="16" y="24" font-family="monospace" font-size="15" fill="#111111">{title}</text>',
    ]

    for panel_index, (target_s, points_2d, outline) in enumerate(section_payloads):
        x0 = panel_index * panel_width
        center_x = x0 + panel_width / 2.0
        center_y = title_height + panel_height / 2.0
        lines.append(f'<rect x="{x0 + 8}" y="{title_height + 8}" width="{panel_width - 16}" height="{panel_height - 16}" fill="none" stroke="#d7d7d7" stroke-width="1"/>')
        lines.append(f'<line x1="{x0 + margin}" y1="{center_y:.3f}" x2="{x0 + panel_width - margin}" y2="{center_y:.3f}" stroke="#d0d0d0" stroke-width="1"/>')
        lines.append(f'<line x1="{center_x:.3f}" y1="{title_height + margin}" x2="{center_x:.3f}" y2="{title_height + panel_height - margin}" stroke="#d0d0d0" stroke-width="1"/>')
        lines.append(f'<circle cx="{center_x:.3f}" cy="{center_y:.3f}" r="2.5" fill="#111111"/>')
        lines.append(f'<text x="{x0 + 16}" y="{title_height + 22}" font-family="monospace" font-size="13" fill="#111111">s={target_s:.3f}</text>')

        if len(points_2d):
            point_elements = []
            for y_value, z_value in points_2d:
                px = center_x + y_value * scale
                py = center_y - z_value * scale
                point_elements.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="1.2" fill="#7aa6ff" fill-opacity="0.55"/>')
            lines.extend(point_elements)

        if len(outline) >= 3:
            outline_path = " ".join(
                f'{center_x + point[0] * scale:.3f},{center_y - point[1] * scale:.3f}'
                for point in outline
            )
            lines.append(f'<polygon points="{outline_path}" fill="rgba(34, 77, 160, 0.10)" stroke="#224da0" stroke-width="2"/>')

    lines.append('</svg>')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _export_cross_section_diagnostics(
    run_dir: Path,
    stats_rows: list[PhaseModelStats],
    canonical_points: np.ndarray,
    model: regen.GastricReferenceModel,
    observation_rotation: np.ndarray,
    cross_section_s_values: list[float],
) -> Path:
    diagnostics_dir = run_dir / "cross_section_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[CrossSectionDiagnostic] = []
    for phase_index in _select_diagnostic_phase_indices(stats_rows):
        stats = stats_rows[phase_index]
        deformed_points, _, _ = _deform_reference_points(
            canonical_points,
            model,
            float(stats.phase_value),
            observation_rotation,
        )
        canonical_deformed = _observed_to_canonical_points(deformed_points, model, observation_rotation)

        section_payloads: list[tuple[float, np.ndarray, np.ndarray]] = []
        for target_s in cross_section_s_values:
            section_points, band_half_width = _collect_cross_section_points(canonical_deformed, model, target_s)
            outline = _build_cross_section_outline(section_points)
            if len(section_points):
                centroid = np.mean(section_points, axis=0)
                radii = np.linalg.norm(section_points - centroid, axis=1)
                centroid_y = float(centroid[0])
                centroid_z = float(centroid[1])
                mean_radius = float(np.mean(radii))
                p10_radius = float(np.percentile(radii, 10.0))
                p90_radius = float(np.percentile(radii, 90.0))
            else:
                centroid_y = 0.0
                centroid_z = 0.0
                mean_radius = 0.0
                p10_radius = 0.0
                p90_radius = 0.0
            summary_rows.append(
                CrossSectionDiagnostic(
                    phase_index=stats.phase_index,
                    phase_value=stats.phase_value,
                    timestamp_s=stats.timestamp_s,
                    target_s=float(target_s),
                    sample_count=int(len(section_points)),
                    band_half_width=float(band_half_width),
                    centroid_y=centroid_y,
                    centroid_z=centroid_z,
                    mean_radius=mean_radius,
                    p10_radius=p10_radius,
                    p90_radius=p90_radius,
                )
            )
            section_payloads.append((float(target_s), section_points, outline))

        svg_path = diagnostics_dir / (
            f"phase_{stats.phase_index:03d}_{stats.phase_value:.3f}_{_format_timestamp_token(stats.timestamp_s)}_sections.svg"
        )
        title = (
            f"phase={stats.phase_index:03d} value={stats.phase_value:.3f} "
            f"t={stats.timestamp_s:.3f}s max_contraction={stats.max_contraction:.3f}"
        )
        _write_cross_section_svg(svg_path, section_payloads, title)

    summary_path = diagnostics_dir / "cross_section_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "phase_index",
            "phase_value",
            "timestamp_s",
            "target_s",
            "sample_count",
            "band_half_width",
            "centroid_y",
            "centroid_z",
            "mean_radius",
            "p10_radius",
            "p90_radius",
        ])
        for row in summary_rows:
            writer.writerow([
                row.phase_index,
                f"{row.phase_value:.6f}",
                f"{row.timestamp_s:.6f}",
                f"{row.target_s:.6f}",
                row.sample_count,
                f"{row.band_half_width:.6f}",
                f"{row.centroid_y:.6f}",
                f"{row.centroid_z:.6f}",
                f"{row.mean_radius:.6f}",
                f"{row.p10_radius:.6f}",
                f"{row.p90_radius:.6f}",
            ])
    return diagnostics_dir


def _measure_waveband_mesh_density(
    mesh: trimesh.Trimesh,
    stats: PhaseModelStats,
    model: regen.GastricReferenceModel,
    observation_rotation: np.ndarray,
    band_half_width: float,
    section_half_width: float,
    angular_bins: int,
) -> WaveBandMeshDiagnostic:
    canonical_vertices = _observed_to_canonical_points(
        np.asarray(mesh.vertices, dtype=np.float64),
        model,
        observation_rotation,
    )
    vertex_s, _, local_y, local_z = regen.project_canonical_points(model, canonical_vertices)
    oriented_vertex_s = _oriented_arc_coordinate(model, vertex_s)
    band_delta = np.abs(oriented_vertex_s - float(stats.wave_center_s))
    band_mask = band_delta <= band_half_width
    section_mask = band_delta <= section_half_width

    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces):
        face_s = oriented_vertex_s[faces].mean(axis=1)
        band_face_count = int(np.count_nonzero(np.abs(face_s - float(stats.wave_center_s)) <= band_half_width))
    else:
        band_face_count = 0

    occupied_angle_bins = 0
    mean_vertices_per_occupied_bin = 0.0
    max_vertices_in_bin = 0
    section_vertex_count = int(np.count_nonzero(section_mask))
    if section_vertex_count > 0:
        section_points = np.column_stack([local_y[section_mask], local_z[section_mask]]).astype(np.float64)
        centroid = np.mean(section_points, axis=0)
        shifted = section_points - centroid
        angles = np.arctan2(shifted[:, 1], shifted[:, 0])
        counts, _ = np.histogram(angles, bins=max(angular_bins, 8), range=(-math.pi, math.pi))
        occupied = counts[counts > 0]
        occupied_angle_bins = int(len(occupied))
        if occupied_angle_bins > 0:
            mean_vertices_per_occupied_bin = float(np.mean(occupied))
            max_vertices_in_bin = int(np.max(occupied))

    local_mean_edge_length = 0.0
    local_p90_edge_length = 0.0
    edge_vertices = np.asarray(getattr(mesh, "edges_unique", np.empty((0, 2), dtype=np.int64)), dtype=np.int64)
    edge_lengths = np.asarray(getattr(mesh, "edges_unique_length", np.array([], dtype=np.float64)), dtype=np.float64)
    if edge_vertices.size and edge_lengths.size:
        edge_mid_s = oriented_vertex_s[edge_vertices].mean(axis=1)
        edge_mask = np.abs(edge_mid_s - float(stats.wave_center_s)) <= band_half_width
        local_edges = edge_lengths[edge_mask]
        if local_edges.size:
            local_mean_edge_length = float(np.mean(local_edges))
            local_p90_edge_length = float(np.percentile(local_edges, 90.0))

    return WaveBandMeshDiagnostic(
        phase_index=stats.phase_index,
        phase_value=stats.phase_value,
        timestamp_s=stats.timestamp_s,
        wave_center_s=stats.wave_center_s,
        band_half_width=float(band_half_width),
        section_half_width=float(section_half_width),
        band_vertex_count=int(np.count_nonzero(band_mask)),
        band_face_count=band_face_count,
        section_vertex_count=section_vertex_count,
        occupied_angle_bins=occupied_angle_bins,
        mean_vertices_per_occupied_bin=mean_vertices_per_occupied_bin,
        max_vertices_in_bin=max_vertices_in_bin,
        local_mean_edge_length=local_mean_edge_length,
        local_p90_edge_length=local_p90_edge_length,
    )


def _export_waveband_mesh_diagnostics(
    run_dir: Path,
    mesh_results: list[MeshBuildResult],
    stats_rows: list[PhaseModelStats],
    model: regen.GastricReferenceModel,
    observation_rotation: np.ndarray,
    band_half_width: float = 0.030,
    section_half_width: float = 0.012,
    angular_bins: int = 48,
) -> Path:
    diagnostics_path = run_dir / "waveband_mesh_diagnostics.csv"
    rows: list[WaveBandMeshDiagnostic] = []
    for phase_index in _select_diagnostic_phase_indices(stats_rows):
        mesh_result = mesh_results[phase_index]
        mesh = trimesh.load(mesh_result.mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        rows.append(
            _measure_waveband_mesh_density(
                mesh,
                stats_rows[phase_index],
                model,
                observation_rotation,
                band_half_width=band_half_width,
                section_half_width=section_half_width,
                angular_bins=angular_bins,
            )
        )

    with diagnostics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "phase_index",
            "phase_value",
            "timestamp_s",
            "wave_center_s",
            "band_half_width",
            "section_half_width",
            "band_vertex_count",
            "band_face_count",
            "section_vertex_count",
            "occupied_angle_bins",
            "mean_vertices_per_occupied_bin",
            "max_vertices_in_bin",
            "local_mean_edge_length",
            "local_p90_edge_length",
        ])
        for row in rows:
            writer.writerow([
                row.phase_index,
                f"{row.phase_value:.6f}",
                f"{row.timestamp_s:.6f}",
                f"{row.wave_center_s:.6f}",
                f"{row.band_half_width:.6f}",
                f"{row.section_half_width:.6f}",
                row.band_vertex_count,
                row.band_face_count,
                row.section_vertex_count,
                row.occupied_angle_bins,
                f"{row.mean_vertices_per_occupied_bin:.6f}",
                row.max_vertices_in_bin,
                f"{row.local_mean_edge_length:.6f}",
                f"{row.local_p90_edge_length:.6f}",
            ])
    return diagnostics_path


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


def _observed_to_canonical_points(
    observed_points: np.ndarray,
    model: regen.GastricReferenceModel,
    observation_rotation: np.ndarray,
) -> np.ndarray:
    world_points = model.world_center + ((observation_rotation.T @ (observed_points - model.world_center).T).T)
    canonical_points = (model.world_basis.T @ (world_points - model.world_center).T).T
    return canonical_points.astype(np.float64)


def _shared_base_surface_config() -> SurfaceModelConfig:
    cfg = SurfaceModelConfig()
    cfg.out_subdir = "meshes"
    cfg.max_points = 12800
    cfg.train_steps = 280
    cfg.mesh_resolution = 108
    cfg.smoothing_iterations = 54
    cfg.normal_neighbors = 60
    return cfg


def _hybrid_phase_surface_config() -> SurfaceModelConfig:
    cfg = SurfaceModelConfig()
    cfg.out_subdir = "_hybrid_phase_recon"
    cfg.max_points = 11000
    cfg.train_steps = 240
    cfg.mesh_resolution = 96
    cfg.smoothing_iterations = 16
    cfg.normal_neighbors = 44
    return cfg


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


def _select_local_adaptive_wave_centers(
    stats_rows: list[PhaseModelStats],
    peak_ratio_threshold: float,
    merge_spacing: float,
) -> list[tuple[float, float]]:
    if not stats_rows:
        return []
    peak_contraction = max(float(row.max_contraction) for row in stats_rows)
    if peak_contraction <= 1e-8:
        return []

    candidates = [
        (float(row.wave_center_s), float(row.max_contraction) / peak_contraction)
        for row in stats_rows
        if float(row.max_contraction) >= peak_contraction * peak_ratio_threshold
    ]
    if not candidates:
        return []

    merged: list[tuple[float, float]] = []
    for center_s, weight in sorted(candidates, key=lambda item: item[0], reverse=True):
        if merged and abs(center_s - merged[-1][0]) <= merge_spacing:
            prev_center, prev_weight = merged[-1]
            if weight > prev_weight:
                merged[-1] = (center_s, weight)
            else:
                merged[-1] = (prev_center, max(prev_weight, weight))
            continue
        merged.append((center_s, weight))
    return merged


def _cleanup_mesh_topology(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    cleaned = mesh.copy()
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
        cleaned.remove_unreferenced_vertices()
        cleaned.remove_infinite_values()
        cleaned.fix_normals()
    return cleaned


def _expand_face_selection(mesh: trimesh.Trimesh, face_indices: np.ndarray, neighbor_rings: int) -> np.ndarray:
    if neighbor_rings <= 0 or face_indices.size == 0:
        return np.unique(face_indices.astype(np.int64))

    adjacency = np.asarray(getattr(mesh, "face_adjacency", np.empty((0, 2), dtype=np.int64)), dtype=np.int64)
    if adjacency.size == 0:
        return np.unique(face_indices.astype(np.int64))

    selected = set(int(index) for index in np.unique(face_indices.astype(np.int64)))
    frontier = set(selected)
    for _ in range(neighbor_rings):
        if not frontier:
            break
        expanded: set[int] = set()
        for face_a, face_b in adjacency:
            if face_a in frontier and face_b not in selected:
                expanded.add(int(face_b))
            elif face_b in frontier and face_a not in selected:
                expanded.add(int(face_a))
        if not expanded:
            break
        selected.update(expanded)
        frontier = expanded
    return np.asarray(sorted(selected), dtype=np.int64)


def _apply_local_adaptive_ring_subdivision(
    mesh: trimesh.Trimesh,
    model: regen.GastricReferenceModel,
    observation_rotation: np.ndarray,
    stats_rows: list[PhaseModelStats],
    config: LocalAdaptiveSubdivisionConfig,
) -> trimesh.Trimesh:
    if not config.enabled:
        return mesh

    wave_centers = _select_local_adaptive_wave_centers(
        stats_rows,
        peak_ratio_threshold=config.peak_ratio_threshold,
        merge_spacing=config.band_half_width * 0.8,
    )
    if not wave_centers:
        return mesh

    refined = mesh.copy()
    adaptive_vertex_limit = max(SHARED_BASE_VERTEX_HARD_LIMIT, int(len(refined.vertices)) + LOCAL_ADAPTIVE_VERTEX_BUDGET)
    for _ in range(config.max_passes):
        if len(refined.vertices) >= adaptive_vertex_limit:
            break

        canonical_vertices = _observed_to_canonical_points(
            np.asarray(refined.vertices, dtype=np.float64),
            model,
            observation_rotation,
        )
        vertex_s, _, _, _ = regen.project_canonical_points(model, canonical_vertices)
        oriented_vertex_s = _oriented_arc_coordinate(model, vertex_s)
        faces = np.asarray(refined.faces, dtype=np.int64)
        face_s = oriented_vertex_s[faces].mean(axis=1)

        face_scores = np.zeros(len(faces), dtype=np.float64)
        for center_s, weight in wave_centers:
            distal_ratio = float(
                np.clip(
                    (config.distal_emphasis_threshold - center_s) / max(config.distal_emphasis_threshold, 1e-6),
                    0.0,
                    1.0,
                )
            )
            local_band_width = config.band_half_width * (1.0 - 0.18 * distal_ratio)
            local_weight = min(1.0, weight * (1.0 + config.distal_emphasis_gain * distal_ratio))
            band_score = np.exp(-0.5 * ((face_s - center_s) / max(local_band_width, 1e-6)) ** 2)
            face_scores = np.maximum(face_scores, local_weight * band_score)

        body_support = _smoothstep(0.08, 0.22, face_s) * (1.0 - _smoothstep(0.78, 0.94, face_s))
        broad_center_a = np.exp(-0.5 * ((face_s - 0.56) / max(config.support_band_width, 1e-6)) ** 2)
        broad_center_b = np.exp(-0.5 * ((face_s - 0.34) / max(config.support_band_width * 0.92, 1e-6)) ** 2)
        broad_center_c = np.exp(-0.5 * ((face_s - 0.18) / max(config.support_band_width * 0.78, 1e-6)) ** 2)
        support_score = config.support_weight * np.maximum.reduce([body_support, broad_center_a, broad_center_b, broad_center_c])
        face_scores = np.maximum(face_scores, support_score)

        selected_faces = np.flatnonzero(face_scores >= config.face_score_threshold)
        if selected_faces.size == 0:
            break

        triangles = np.asarray(refined.vertices, dtype=np.float64)[faces]
        edge_lengths = np.stack([
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ], axis=1)
        max_edge_lengths = np.max(edge_lengths, axis=1)
        local_edge_threshold = float(np.percentile(max_edge_lengths[selected_faces], config.edge_percentile))
        selected_faces = selected_faces[max_edge_lengths[selected_faces] >= local_edge_threshold]
        if selected_faces.size == 0:
            break

        remaining_vertex_budget = max(0, adaptive_vertex_limit - int(len(refined.vertices)))
        if remaining_vertex_budget <= 0:
            break

        face_priority = face_scores[selected_faces] * np.maximum(max_edge_lengths[selected_faces], 1e-8)
        max_face_budget = max(96, remaining_vertex_budget // 2)
        if selected_faces.size > max_face_budget:
            keep_order = np.argsort(face_priority)[-max_face_budget:]
            selected_faces = selected_faces[np.sort(keep_order)]
            face_priority = face_priority[np.sort(keep_order)]
        if selected_faces.size == 0:
            break

        priority_order = selected_faces[np.argsort(face_priority)[::-1]]
        before_vertices = int(len(refined.vertices))
        accepted_candidate: trimesh.Trimesh | None = None
        for fraction in (1.0, 0.72, 0.5, 0.35):
            keep_count = max(48, int(math.ceil(len(priority_order) * fraction)))
            candidate_faces = np.asarray(priority_order[:keep_count], dtype=np.int64)
            candidate_faces = _expand_face_selection(refined, candidate_faces, config.boundary_neighbor_rings)
            if candidate_faces.size == 0:
                continue

            candidate = refined.subdivide(face_index=candidate_faces)
            candidate = _cleanup_mesh_topology(candidate)
            if len(candidate.vertices) > adaptive_vertex_limit:
                continue
            if int(len(candidate.vertices)) == before_vertices:
                continue
            if refined.is_watertight and not candidate.is_watertight:
                continue
            accepted_candidate = candidate
            break

        if accepted_candidate is None:
            break
        refined = accepted_candidate

    return refined


def _refine_shared_base_mesh(
    mesh: trimesh.Trimesh,
    model: regen.GastricReferenceModel,
    observation_rotation: np.ndarray,
    stats_rows: list[PhaseModelStats],
) -> trimesh.Trimesh:
    refined = mesh.copy()
    target_volume = float(abs(refined.volume)) if refined.is_volume else None
    target_centroid = np.asarray(refined.centroid, dtype=np.float64)
    refined = _cleanup_mesh_topology(refined)

    if len(refined.vertices) < SHARED_BASE_VERTEX_SOFT_LIMIT:
        subdivision_schedule = ((82, 0.89), (88, 0.92), (93, 0.95))
    else:
        subdivision_schedule = ()

    for percentile, scale in subdivision_schedule:
        if len(refined.vertices) >= SHARED_BASE_VERTEX_HARD_LIMIT:
            break
        edge_lengths = np.asarray(getattr(refined, "edges_unique_length", np.array([], dtype=np.float64)), dtype=np.float64)
        if edge_lengths.size == 0:
            break
        target_edge = float(np.percentile(edge_lengths, percentile))
        max_edge = target_edge * scale
        if np.isfinite(max_edge) and max_edge > 1e-6 and float(np.max(edge_lengths)) > max_edge * 1.001:
            refined = refined.subdivide_to_size(max_edge=max_edge)
            refined = _cleanup_mesh_topology(refined)
            if len(refined.vertices) >= SHARED_BASE_VERTEX_HARD_LIMIT:
                break

    before_local_vertices = int(len(refined.vertices))
    refined = _apply_local_adaptive_ring_subdivision(
        refined,
        model,
        observation_rotation,
        stats_rows,
        LOCAL_ADAPTIVE_SUBDIVISION,
    )

    if int(len(refined.vertices)) == before_local_vertices:
        edge_lengths = np.asarray(getattr(refined, "edges_unique_length", np.array([], dtype=np.float64)), dtype=np.float64)
        if edge_lengths.size:
            micro_target_edge = float(np.percentile(edge_lengths, 96.0)) * 0.965
            if np.isfinite(micro_target_edge) and micro_target_edge > 1e-6:
                micro_candidate = refined.subdivide_to_size(max_edge=micro_target_edge)
                micro_candidate = _cleanup_mesh_topology(micro_candidate)
                if (
                    micro_candidate.is_watertight
                    and int(len(micro_candidate.vertices)) > before_local_vertices
                    and int(len(micro_candidate.vertices)) <= before_local_vertices + GLOBAL_MICRO_DENSIFY_BUDGET
                ):
                    refined = micro_candidate

    trimesh.smoothing.filter_taubin(refined, lamb=0.14, nu=-0.15, iterations=72)
    if hasattr(trimesh.smoothing, "filter_humphrey"):
        trimesh.smoothing.filter_humphrey(refined, alpha=0.009, beta=0.11, iterations=20)
    trimesh.smoothing.filter_taubin(refined, lamb=0.12, nu=-0.13, iterations=46)
    refined = _cleanup_mesh_topology(refined)
    if target_volume is not None and refined.is_volume:
        current_volume = float(abs(refined.volume))
        if current_volume > 1e-8:
            scale = (target_volume / current_volume) ** (1.0 / 3.0)
            refined.vertices = (refined.vertices - target_centroid) * scale + target_centroid
            refined.fix_normals()
    return refined


def _cleanup_shared_phase_mesh(mesh: trimesh.Trimesh, contraction_strength: float = 0.0) -> trimesh.Trimesh:
    cleaned = mesh.copy()
    large_mesh = len(cleaned.vertices) >= 13000
    contraction_ratio = float(np.clip(contraction_strength / 3.06, 0.0, 1.0))
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
    for _ in range(2 if large_mesh else 3):
        if cleaned.is_watertight:
            break
        trimesh.repair.fill_holes(cleaned)
        trimesh.repair.fix_winding(cleaned)
        trimesh.repair.fix_inversion(cleaned, multibody=True)
        trimesh.repair.fix_normals(cleaned, multibody=True)
        cleaned.remove_unreferenced_vertices()
        cleaned.remove_infinite_values()
        cleaned.fix_normals()
    if cleaned.is_watertight:
        smooth_iterations = (24 if large_mesh else 36) + int(round(8.0 * contraction_ratio))
        trimesh.smoothing.filter_taubin(cleaned, lamb=0.12, nu=-0.13, iterations=smooth_iterations)
        if hasattr(trimesh.smoothing, "filter_humphrey"):
            humphrey_iterations = (6 if large_mesh else 10) + int(round(3.0 * contraction_ratio))
            humphrey_alpha = (0.010 if large_mesh else 0.014) + 0.002 * contraction_ratio
            humphrey_beta = (0.12 if large_mesh else 0.16) + 0.02 * contraction_ratio
            trimesh.smoothing.filter_humphrey(
                cleaned,
                alpha=humphrey_alpha,
                beta=humphrey_beta,
                iterations=humphrey_iterations,
            )
        final_taubin_iterations = (14 if large_mesh else 18) + int(round(4.0 * contraction_ratio))
        trimesh.smoothing.filter_taubin(cleaned, lamb=0.10, nu=-0.11, iterations=final_taubin_iterations)
        cleaned.remove_infinite_values()
        cleaned.remove_unreferenced_vertices()
        cleaned.fix_normals()
        if target_volume is not None and cleaned.is_volume:
            current_volume = float(abs(cleaned.volume))
            if current_volume > 1e-8:
                scale = (target_volume / current_volume) ** (1.0 / 3.0)
                cleaned.vertices = (cleaned.vertices - target_centroid) * scale + target_centroid
                cleaned.fix_normals()
    if not cleaned.is_watertight:
        trimesh.repair.fill_holes(cleaned)
        trimesh.repair.fix_winding(cleaned)
        trimesh.repair.fix_inversion(cleaned, multibody=True)
        trimesh.repair.fix_normals(cleaned, multibody=True)
        cleaned.remove_infinite_values()
        cleaned.remove_unreferenced_vertices()
        cleaned.fix_normals()
    return cleaned


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
        trimesh.smoothing.filter_taubin(processed, lamb=0.24, nu=-0.25, iterations=36)
    else:
        trimesh.smoothing.filter_taubin(processed, lamb=0.18, nu=-0.19, iterations=22)
        if hasattr(trimesh.smoothing, "filter_humphrey"):
            trimesh.smoothing.filter_humphrey(processed, alpha=0.018, beta=0.18, iterations=4)
        trimesh.smoothing.filter_taubin(processed, lamb=0.16, nu=-0.17, iterations=12)
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


def _write_mesh_summary(mesh_dir: Path, results: list[MeshBuildResult]) -> Path:
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
    return summary_path


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

    temp_dir = pointcloud_paths[0].parent / hybrid_cfg.out_subdir
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    _write_mesh_summary(mesh_dir, updated_results)
    return updated_results


def _apply_patch_remesh_guidance_to_mesh(
    shared_mesh: trimesh.Trimesh,
    guide_mesh: trimesh.Trimesh,
    stats: PhaseModelStats,
    model: regen.GastricReferenceModel,
    observation_rotation: np.ndarray,
    config: PatchRemeshGuidanceConfig,
) -> trimesh.Trimesh:
    guided = shared_mesh.copy()
    shared_vertices = np.asarray(guided.vertices, dtype=np.float64)
    canonical_vertices = _observed_to_canonical_points(shared_vertices, model, observation_rotation)
    vertex_s, _, _, _ = regen.project_canonical_points(model, canonical_vertices)
    oriented_vertex_s = _oriented_arc_coordinate(model, vertex_s)

    band_delta = np.abs(oriented_vertex_s - float(stats.wave_center_s))
    band_mask = band_delta <= config.band_outer_width
    if not np.any(band_mask):
        return guided

    guide_vertices = np.asarray(guide_mesh.vertices, dtype=np.float64)
    if len(guide_vertices) == 0:
        return guided
    guide_tree = cKDTree(guide_vertices)
    selected_shared = shared_vertices[band_mask]
    _, nearest_indices = guide_tree.query(selected_shared, k=1)
    nearest_vertices = guide_vertices[np.asarray(nearest_indices, dtype=np.int64)]

    band_weights = 1.0 - _smoothstep(config.band_inner_width, config.band_outer_width, band_delta[band_mask])
    blend = np.clip(config.blend_strength * band_weights, 0.0, 1.0)[:, None]
    shared_vertices[band_mask] = selected_shared * (1.0 - blend) + nearest_vertices * blend
    guided.vertices = shared_vertices
    guided = _cleanup_shared_phase_mesh(guided, contraction_strength=stats.max_contraction)
    return guided


def _apply_shared_topology_patch_remesh_guidance(
    mesh_dir: Path,
    pointcloud_paths: list[Path],
    results: list[MeshBuildResult],
    stats_rows: list[PhaseModelStats],
    selected_phase_indices: list[int],
    model: regen.GastricReferenceModel,
    observation_rotation: np.ndarray,
    phase_bin_step_seconds: float,
    config: PatchRemeshGuidanceConfig,
) -> list[MeshBuildResult]:
    if not config.enabled or not selected_phase_indices:
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
        phase_index = next((row.phase_index for row in stats_rows if row.pointcloud_path.name == result.pointcloud_path.name), None)
        if phase_index is None or phase_index not in selected_phase_indices:
            updated_results.append(result)
            continue

        guide_result = temp_map.get(result.pointcloud_path.name)
        if guide_result is None:
            updated_results.append(result)
            continue

        shared_mesh = trimesh.load(result.mesh_path, force="mesh")
        guide_mesh = trimesh.load(guide_result.mesh_path, force="mesh")
        if not isinstance(shared_mesh, trimesh.Trimesh) or not isinstance(guide_mesh, trimesh.Trimesh):
            updated_results.append(result)
            continue

        guided_mesh = _apply_patch_remesh_guidance_to_mesh(
            shared_mesh,
            _postprocess_shared_mesh(guide_mesh),
            stats_rows[phase_index],
            model,
            observation_rotation,
            config,
        )
        if not guided_mesh.is_watertight:
            updated_results.append(result)
            continue

        final_mesh_path = mesh_dir / result.mesh_path.name
        _mesh_export(guided_mesh, final_mesh_path)
        updated_results.append(
            MeshBuildResult(
                pointcloud_path=result.pointcloud_path,
                mesh_path=final_mesh_path,
                timestamp_s=result.timestamp_s,
                input_points=result.input_points,
                sampled_points=result.sampled_points,
                vertices=int(len(guided_mesh.vertices)),
                faces=int(len(guided_mesh.faces)),
                watertight=bool(guided_mesh.is_watertight),
                method="shared_topology_patch_remesh_guidance",
            )
        )

    temp_dir = pointcloud_paths[0].parent / hybrid_cfg.out_subdir
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    _write_mesh_summary(mesh_dir, updated_results)
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
) -> list[MeshBuildResult]:
    mesh_dir.mkdir(parents=True, exist_ok=True)
    base_canonical_vertices = _observed_to_canonical_points(
        np.asarray(base_mesh.vertices, dtype=np.float64),
        model,
        observation_rotation,
    )

    results: list[MeshBuildResult] = []
    for phase_index, (pointcloud_path, phase_value) in enumerate(zip(pointcloud_paths, phase_values)):
        deformed_vertices, _, phase_peak_contraction = _deform_reference_points(
            base_canonical_vertices,
            model,
            float(phase_value),
            observation_rotation,
        )
        mesh = trimesh.Trimesh(vertices=deformed_vertices, faces=np.asarray(base_mesh.faces, dtype=np.int64), process=False)
        mesh = _cleanup_shared_phase_mesh(mesh, contraction_strength=phase_peak_contraction)

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
        del deformed_vertices
        del mesh
        if phase_index % 2 == 1:
            gc.collect()

    _write_mesh_summary(mesh_dir, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a gastric peristaltic phase-sequence model set")
    parser.add_argument("--instance-name", type=str, default=None, help="Named stomach instance under benchmark/stomach_pcd")
    parser.add_argument("--reference-ply", type=str, default=None, help="Explicit reference stomach point cloud path")
    parser.add_argument("--phase-count", type=int, default=None, help="Number of phase models to generate across one cycle")
    parser.add_argument("--monitor-path", type=str, default=None, help="Optional monitor stream used to determine cycle duration")
    parser.add_argument(
        "--shared-topology-patch-remesh-guidance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use strong-phase independent reconstructions as local guidance while keeping shared topology",
    )
    parser.add_argument(
        "--patch-remesh-peak-ratio-threshold",
        type=float,
        default=0.90,
        help="Relative peak-contraction threshold used to select strong phases for shared-topology patch guidance",
    )
    parser.add_argument(
        "--patch-remesh-neighbor-span",
        type=int,
        default=0,
        help="Number of neighboring phase bins on each side to include for shared-topology patch guidance",
    )
    parser.add_argument(
        "--hybrid-strong-phase-remesh",
        action="store_true",
        help="For strong-contraction phases, replace shared-topology export with per-phase NeuralGF reconstruction",
    )
    parser.add_argument(
        "--hybrid-remesh-peak-ratio-threshold",
        type=float,
        default=DEFAULT_HYBRID_REMESH_PEAK_RATIO_THRESHOLD,
        help="Relative peak-contraction threshold used to select strong phases for hybrid remeshing",
    )
    parser.add_argument(
        "--hybrid-remesh-neighbor-span",
        type=int,
        default=DEFAULT_HYBRID_REMESH_NEIGHBOR_SPAN,
        help="Number of neighboring phase bins on each side to include around selected strong phases",
    )
    parser.add_argument(
        "--cross-section-s-values",
        type=str,
        default=",".join(f"{value:.2f}" for value in DEFAULT_CROSS_SECTION_S_VALUES),
        help="Comma-separated canonical s positions for 2D cross-section diagnostics",
    )
    parser.add_argument(
        "--skip-cross-section-diagnostics",
        action="store_true",
        help="Skip exporting 2D cross-section diagnostic SVGs and CSV summaries",
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="",
        help="Directory where the generated phase-sequence model run folder will be created",
    )
    args = parser.parse_args()

    instance_paths = resolve_instance_paths(
        instance_name=args.instance_name,
        reference_ply=Path(args.reference_ply).expanduser().resolve() if args.reference_ply else None,
    )
    reference_ply = instance_paths.reference_ply

    if not reference_ply.exists():
        raise FileNotFoundError(f"Reference stomach point cloud not found: {reference_ply}")
    if args.phase_count is not None and args.phase_count < 2:
        raise ValueError("phase-count must be at least 2")
    if not (0.0 < float(args.patch_remesh_peak_ratio_threshold) <= 1.0):
        raise ValueError("patch-remesh-peak-ratio-threshold must be in (0, 1]")
    if int(args.patch_remesh_neighbor_span) < 0:
        raise ValueError("patch-remesh-neighbor-span must be >= 0")
    if not (0.0 < float(args.hybrid_remesh_peak_ratio_threshold) <= 1.0):
        raise ValueError("hybrid-remesh-peak-ratio-threshold must be in (0, 1]")
    if int(args.hybrid_remesh_neighbor_span) < 0:
        raise ValueError("hybrid-remesh-neighbor-span must be >= 0")
    cross_section_s_values = _parse_cross_section_s_values(args.cross_section_s_values)
    hybrid_remesh = HybridRemeshConfig(
        enabled=bool(args.hybrid_strong_phase_remesh),
        peak_ratio_threshold=float(args.hybrid_remesh_peak_ratio_threshold),
        neighbor_span=int(args.hybrid_remesh_neighbor_span),
    )
    patch_remesh_guidance = PatchRemeshGuidanceConfig(
        enabled=bool(args.shared_topology_patch_remesh_guidance),
        peak_ratio_threshold=float(args.patch_remesh_peak_ratio_threshold),
        neighbor_span=int(args.patch_remesh_neighbor_span),
        band_inner_width=0.020,
        band_outer_width=0.074,
        blend_strength=0.82,
    )

    phase_bin_step_seconds = float(PipelineConfig().phase_detection.phase_bin_step_seconds)
    if phase_bin_step_seconds <= 0.0:
        raise ValueError("phase_bin_step_seconds must be positive")

    default_cycle_duration_s = phase_bin_step_seconds * LEGACY_PHASE_COUNT
    monitor_path = resolve_monitor_input_path(
        instance_paths,
        explicit_path=Path(args.monitor_path).expanduser().resolve() if args.monitor_path else None,
    )
    detected_phase_count: int | None = None
    cycle_duration_s = default_cycle_duration_s
    try:
        detected_phase_count, cycle_duration_s = _detect_monitor_aligned_phase_count(monitor_path, phase_bin_step_seconds)
    except Exception as exc:
        print(f"[PhaseModels] Monitor-aligned phase detection unavailable, using fallback duration {default_cycle_duration_s:.3f}s: {exc}")

    if args.phase_count is None:
        phase_count = max(DEFAULT_PHASE_COUNT, detected_phase_count or 0)
    else:
        phase_count = int(args.phase_count)
    effective_phase_step_seconds = cycle_duration_s / max(phase_count, 1)

    output_base_dir = Path(args.output_base_dir).expanduser().resolve() if args.output_base_dir else instance_paths.phase_model_base_dir
    run_dir, run_id = _create_indexed_output_dir(output_base_dir, OUTPUT_PREFIX)
    points_dir = run_dir / "pointclouds"
    points_dir.mkdir(parents=True, exist_ok=True)

    reference_points, _, canonical_points = _read_reference_points(reference_ply)
    model = regen.load_reference_model(reference_ply)
    observation_rotation = _rotation_matrix_xyz(OBSERVATION_ROTATION_DEG)
    transform_path = _write_observation_transform(run_dir, model.world_center, observation_rotation, effective_phase_step_seconds)

    phase_values = np.linspace(0.0, 1.0, phase_count, endpoint=False, dtype=np.float64)

    pointcloud_paths: list[Path] = []
    stats_rows: list[PhaseModelStats] = []
    for phase_index, phase_value in enumerate(phase_values):
        timestamp_s = _phase_timestamp_seconds(phase_index, effective_phase_step_seconds)
        deformed_points, wave_center, max_contraction = _deform_reference_points(
            canonical_points,
            model,
            float(phase_value),
            observation_rotation,
        )
        pointcloud_path = points_dir / (
            f"{instance_paths.name}_run_{run_id}_phase_{phase_index:03d}_{phase_value:.3f}_{_format_timestamp_token(timestamp_s)}.ply"
        )
        _write_pointcloud_ply(deformed_points, pointcloud_path)
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
        print(
            f"[PhaseModels] phase={phase_index:03d} value={phase_value:.3f} ts={timestamp_s:.3f}s "
            f"wave_center_s={wave_center:.3f} max_contraction={max_contraction:.3f} "
            f"-> {pointcloud_path.name}"
        )

    diagnostics_dir: Path | None = None
    if not args.skip_cross_section_diagnostics:
        diagnostics_dir = _export_cross_section_diagnostics(
            run_dir,
            stats_rows,
            canonical_points,
            model,
            observation_rotation,
            cross_section_s_values,
        )

    surface_cfg = _shared_base_surface_config()
    base_mesh, base_mesh_result = _build_shared_base_mesh(
        pointcloud_paths[0],
        surface_cfg,
        effective_phase_step_seconds,
    )
    base_mesh = _refine_shared_base_mesh(base_mesh, model, observation_rotation, stats_rows)
    meshes = _export_shared_topology_meshes(
        base_mesh,
        base_mesh_result,
        pointcloud_paths,
        phase_values,
        model,
        observation_rotation,
        points_dir / surface_cfg.out_subdir,
        effective_phase_step_seconds,
    )
    patch_phase_indices = _select_hybrid_remesh_phase_indices(
        stats_rows,
        HybridRemeshConfig(
            enabled=patch_remesh_guidance.enabled,
            peak_ratio_threshold=patch_remesh_guidance.peak_ratio_threshold,
            neighbor_span=patch_remesh_guidance.neighbor_span,
        ),
    )
    if patch_phase_indices:
        print(f"[PhaseModels] Shared-topology patch remesh guidance phases: {','.join(str(idx) for idx in patch_phase_indices)}")
        meshes = _apply_shared_topology_patch_remesh_guidance(
            points_dir / surface_cfg.out_subdir,
            pointcloud_paths,
            meshes,
            stats_rows,
            patch_phase_indices,
            model,
            observation_rotation,
            effective_phase_step_seconds,
            patch_remesh_guidance,
        )
    waveband_diagnostics_path = _export_waveband_mesh_diagnostics(
        run_dir,
        meshes,
        stats_rows,
        model,
        observation_rotation,
    )
    hybrid_phase_indices = _select_hybrid_remesh_phase_indices(stats_rows, hybrid_remesh)
    if hybrid_phase_indices:
        print("[PhaseModels] Hybrid remesh overrides shared topology for the selected phases.")
        print(f"[PhaseModels] Hybrid remesh phases: {','.join(str(idx) for idx in hybrid_phase_indices)}")
        meshes = _apply_hybrid_phase_reconstruction(
            points_dir / surface_cfg.out_subdir,
            pointcloud_paths,
            meshes,
            hybrid_phase_indices,
            effective_phase_step_seconds,
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

    print(f"[PhaseModels] Instance: {instance_paths.name}")
    print(f"[PhaseModels] Reference: {reference_ply}")
    if detected_phase_count is not None:
        print(f"[PhaseModels] Monitor stream: {monitor_path}")
        print(f"[PhaseModels] Monitor-aligned phase count: {detected_phase_count}")
        print(f"[PhaseModels] Estimated cycle duration: {cycle_duration_s:.6f}s")
    print(f"[PhaseModels] Output directory: {run_dir}")
    print(f"[PhaseModels] Phase count: {phase_count}")
    print(f"[PhaseModels] Phase bin step: {effective_phase_step_seconds:.6f}s")
    print(f"[PhaseModels] Point clouds: {len(pointcloud_paths)}")
    print(f"[PhaseModels] Meshes: {len(meshes)}")
    print(f"[PhaseModels] Summary: {summary_path}")
    print(f"[PhaseModels] Observation transform: {transform_path}")
    print(f"[PhaseModels] Waveband mesh diagnostics: {waveband_diagnostics_path}")
    if diagnostics_dir is not None:
        print(f"[PhaseModels] Cross-section diagnostics: {diagnostics_dir}")


if __name__ == "__main__":
    main()