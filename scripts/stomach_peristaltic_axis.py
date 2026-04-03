from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.generate_stomach_cycle as cycle_model
from src.stomach_instance_paths import resolve_instance_paths


DEFAULT_CENTERLINE_SAMPLES = 31
DEFAULT_GRID_RESOLUTION = 144
DEFAULT_BASE_SMOOTH_ITERATIONS = 30


@dataclass(frozen=True)
class PeristalticAxisModel:
    centerline: cycle_model.Centerline
    centerline_meta: dict[str, float | list[float] | bool | str]
    reference_ply: Path
    rest_vertices: np.ndarray


def build_peristaltic_axis_model(
    instance_name: str | None = None,
    reference_ply: Path | None = None,
    centerline_samples: int = DEFAULT_CENTERLINE_SAMPLES,
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
    base_smooth_iterations: int = DEFAULT_BASE_SMOOTH_ITERATIONS,
    reverse_centerline: bool = False,
) -> PeristalticAxisModel:
    instance_paths = resolve_instance_paths(instance_name=instance_name, reference_ply=reference_ply)
    reference_points = cycle_model.load_ascii_ply_points(instance_paths.reference_ply)
    rest_vertices, _, _, volume = cycle_model.reconstruct_surface_from_points(
        reference_points,
        grid_resolution=grid_resolution,
        base_smooth_iterations=base_smooth_iterations,
    )
    centerline, centerline_meta = cycle_model.extract_longest_volume_centerline(
        volume,
        centerline_samples,
    )
    if reverse_centerline:
        centerline = cycle_model.reverse_centerline(centerline)
    centerline_meta["orientation_reversed"] = bool(reverse_centerline)
    return PeristalticAxisModel(
        centerline=centerline,
        centerline_meta=centerline_meta,
        reference_ply=instance_paths.reference_ply,
        rest_vertices=np.asarray(rest_vertices, dtype=np.float64),
    )


def project_world_points_to_u(axis_model: PeristalticAxisModel, points_world: np.ndarray) -> np.ndarray:
    mapping = cycle_model.project_vertices_to_centerline(
        np.asarray(points_world, dtype=np.float64),
        axis_model.centerline,
    )
    return np.clip(np.asarray(mapping.u, dtype=np.float64), 0.0, 1.0)


def interpolate_centerline_position(axis_model: PeristalticAxisModel, axis_u: np.ndarray | float) -> np.ndarray:
    axis_u_arr = np.asarray(axis_u, dtype=np.float64)
    centerline = axis_model.centerline.samples
    grid = axis_model.centerline.u_samples
    stacked = np.column_stack([
        np.interp(axis_u_arr, grid, centerline[:, axis])
        for axis in range(3)
    ]).astype(np.float64)
    if np.ndim(axis_u) == 0:
        return stacked[0]
    return stacked


def interpolate_centerline_tangent(axis_model: PeristalticAxisModel, axis_u: np.ndarray | float) -> np.ndarray:
    axis_u_arr = np.asarray(axis_u, dtype=np.float64)
    tangents = axis_model.centerline.tangents
    grid = axis_model.centerline.u_samples
    stacked = np.column_stack([
        np.interp(axis_u_arr, grid, tangents[:, axis])
        for axis in range(3)
    ]).astype(np.float64)
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    stacked /= np.clip(norms, 1e-8, None)
    if np.ndim(axis_u) == 0:
        return stacked[0]
    return stacked


def build_scanner_s_lookup(
    reference_model: object,
    axis_model: PeristalticAxisModel,
    phase: float = 0.0,
    samples: int = 513,
) -> tuple[np.ndarray, np.ndarray]:
    import scripts.regenerate_freehand_scanner_sequence as regen

    s_grid = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    world_points = np.vstack([
        regen.world_centerline(reference_model, float(s_value), float(phase))
        for s_value in s_grid
    ])
    axis_u = project_world_points_to_u(axis_model, world_points)

    order = np.argsort(axis_u)
    axis_u_sorted = axis_u[order]
    s_sorted = s_grid[order]
    unique_mask = np.concatenate([[True], np.diff(axis_u_sorted) > 1e-6])
    axis_u_unique = axis_u_sorted[unique_mask]
    s_unique = s_sorted[unique_mask]
    if axis_u_unique[0] > 0.0:
        axis_u_unique = np.concatenate([[0.0], axis_u_unique])
        s_unique = np.concatenate([[s_unique[0]], s_unique])
    if axis_u_unique[-1] < 1.0:
        axis_u_unique = np.concatenate([axis_u_unique, [1.0]])
        s_unique = np.concatenate([s_unique, [s_unique[-1]]])
    return axis_u_unique.astype(np.float64), s_unique.astype(np.float64)


def axis_u_to_scanner_s(axis_u: np.ndarray | float, axis_u_lookup: np.ndarray, s_lookup: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(axis_u, dtype=np.float64), axis_u_lookup, s_lookup).astype(np.float64)