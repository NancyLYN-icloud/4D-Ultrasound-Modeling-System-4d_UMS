"""Shared-topology dynamic surface reconstruction.

Current implementation uses a shared base mesh plus phase-conditioned vertex
 displacements instead of a canonical implicit field. This keeps the dynamic
 branch aligned with the current shared-topology GT.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import re
from typing import Mapping

import mcubes
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
import trimesh

from ..config import DynamicMeshBuildResult, DynamicModelConfig, DynamicTimelineMeshBuildResult, PointCloudPhaseSummary, SurfaceModelConfig
from .canonical_field import CanonicalField
from .deformation_field import DecoupledMotionLatentField, PhaseConditionedBasisCoefficients, PhaseConditionedDeformationField, PhaseConditionedSDFField, ShapeLatentField
from .surface_reconstruction import _estimate_normals, _mesh_export, _normalize_points, _phase_index_from_path, _read_xyz_ply, _remove_outliers, _sample_points, _voxel_downsample, reconstruct_meshes_from_pointclouds


_PHASE_CENTER_PATTERN = re.compile(r"_phase_\d+_([0-9]+(?:\.[0-9]+)?)")


@dataclass
class PhaseObservation:
    phase: float
    pointcloud_path: Path
    points: np.ndarray
    normals: np.ndarray
    point_weights: np.ndarray
    weight: float
    centroid: np.ndarray
    mean_confidence: float
    extracted_slice_ratio: float
    top_coverage_ratio: float
    vertical_extent_ratio: float
    support_score: float


@dataclass
class SharedTopologyDynamicFit:
    base_vertices: np.ndarray
    base_faces: np.ndarray
    displacements: np.ndarray
    phases: list[float]
    center: np.ndarray
    scale: float
    global_coefficients: np.ndarray | None = None
    global_energy_per_phase: np.ndarray | None = None
    residual_energy_per_phase: np.ndarray | None = None


class CanonicalPhaseDeformationFieldReconstructor:
    """Shared-topology base mesh plus vertex displacement reconstructor."""

    def __init__(self, config: DynamicModelConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config)

    @staticmethod
    def _resolve_device(config: DynamicModelConfig) -> torch.device:
        if config.use_cuda_if_available and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _phase_center_from_path(path: Path) -> float | None:
        match = _PHASE_CENTER_PATTERN.search(path.name)
        if match is None:
            return None
        return float(match.group(1))

    @staticmethod
    def _estimate_point_weights(points: np.ndarray, neighbors: int = 8) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0,), dtype=np.float32)
        if len(points) <= neighbors + 1:
            return np.ones((len(points),), dtype=np.float32)

        tree = cKDTree(points)
        distances, _ = tree.query(points, k=min(neighbors + 1, len(points)))
        local_scale = np.mean(distances[:, 1:], axis=1).astype(np.float32)
        reference = max(float(np.median(local_scale)), 1e-6)
        weights = np.exp(-((local_scale / reference) ** 2)).astype(np.float32)
        return np.clip(weights, 0.1, 1.0)

    def _prepare_phase_observations(
        self,
        pointcloud_paths: list[Path],
        phase_confidences: Mapping[Path, float] | None = None,
        phase_summaries: Mapping[Path, PointCloudPhaseSummary] | None = None,
    ) -> tuple[list[PhaseObservation], np.ndarray, float]:
        observations: list[PhaseObservation] = []
        cleaned_all: list[np.ndarray] = []

        for order, path in enumerate(pointcloud_paths):
            points = _read_xyz_ply(path)
            if len(points) == 0:
                continue
            points = _voxel_downsample(points, self.config.voxel_size)
            points = _remove_outliers(points, neighbors=12, std_ratio=2.0)
            points = _sample_points(points, self.config.max_points_per_phase, self.config.random_seed + order)
            if len(points) < 64:
                continue

            summary = None if phase_summaries is None else phase_summaries.get(path)
            phase_index = _phase_index_from_path(path)
            if summary is not None:
                phase = float(summary.phase_center)
            else:
                parsed_phase = self._phase_center_from_path(path)
                phase = parsed_phase if parsed_phase is not None else float(order / max(len(pointcloud_paths) - 1, 1))
                if phase_index is not None and parsed_phase is None:
                    phase = float(phase_index / max(len(pointcloud_paths), 1))

            if phase_confidences is None:
                weight = 1.0
            else:
                weight = float(phase_confidences.get(path, 1.0))
            weight = float(np.clip(weight, self.config.confidence_floor, 1.0))
            mean_confidence = 1.0 if summary is None else float(summary.mean_confidence)
            extracted_slice_ratio = 1.0 if summary is None else float(summary.extracted_slice_ratio)

            observations.append(
                PhaseObservation(
                    phase=float(np.mod(phase, 1.0)),
                    pointcloud_path=path,
                    points=np.asarray(points, dtype=np.float32),
                    normals=np.empty((0, 3), dtype=np.float32),
                    point_weights=np.empty((0,), dtype=np.float32),
                    weight=weight,
                    centroid=np.mean(points, axis=0).astype(np.float32),
                    mean_confidence=mean_confidence,
                    extracted_slice_ratio=extracted_slice_ratio,
                    top_coverage_ratio=0.0,
                    vertical_extent_ratio=0.0,
                    support_score=0.0,
                )
            )
            cleaned_all.append(points)

        if not observations:
            raise ValueError("No valid phase point clouds available for dynamic reconstruction")

        stacked = np.vstack(cleaned_all).astype(np.float32)
        _, center, scale = _normalize_points(stacked)
        scale = float(scale)

        global_z_values = stacked[:, 2]
        global_z_min = float(np.min(global_z_values))
        global_z_max = float(np.max(global_z_values))
        global_z_span = max(global_z_max - global_z_min, 1e-8)
        global_top_threshold = float(np.quantile(global_z_values, 0.85))

        normalized_observations: list[PhaseObservation] = []
        for observation in observations:
            normalized_points = ((observation.points - center[None, :]) / scale).astype(np.float32)
            normalized_normals = _estimate_normals(normalized_points, neighbors=24)
            point_weights = self._estimate_point_weights(normalized_points)
            normalized_centroid = np.mean(normalized_points, axis=0).astype(np.float32)
            raw_z = observation.points[:, 2]
            top_coverage_ratio = float(np.mean(raw_z >= global_top_threshold))
            vertical_extent_ratio = float(np.clip((float(np.max(raw_z)) - float(np.min(raw_z))) / global_z_span, 0.0, 1.0))
            support_score = float(
                0.35 * observation.weight
                + 0.20 * np.clip(observation.mean_confidence, 0.0, 1.0)
                + 0.15 * np.clip(observation.extracted_slice_ratio, 0.0, 1.0)
                + 0.20 * np.clip(top_coverage_ratio * 6.0, 0.0, 1.0)
                + 0.10 * vertical_extent_ratio
            )
            normalized_observations.append(
                PhaseObservation(
                    phase=observation.phase,
                    pointcloud_path=observation.pointcloud_path,
                    points=normalized_points,
                    normals=normalized_normals,
                    point_weights=point_weights,
                    weight=observation.weight,
                    centroid=normalized_centroid,
                    mean_confidence=observation.mean_confidence,
                    extracted_slice_ratio=observation.extracted_slice_ratio,
                    top_coverage_ratio=top_coverage_ratio,
                    vertical_extent_ratio=vertical_extent_ratio,
                    support_score=support_score,
                )
            )

        normalized_observations.sort(key=lambda item: item.phase)
        return normalized_observations, center.astype(np.float32), scale

    def _select_base_phase_index(self, observations: list[PhaseObservation]) -> int:
        best_index = max(
            range(len(observations)),
            key=lambda index: (
                observations[index].support_score,
                observations[index].top_coverage_ratio,
                observations[index].vertical_extent_ratio,
                observations[index].weight,
                -abs(observations[index].phase - 0.5),
            ),
        )
        best = observations[best_index]
        print(
            "[SharedDynamic] selected base phase "
            f"phase={best.phase:.3f} score={best.support_score:.3f} "
            f"conf={best.mean_confidence:.3f} slice={best.extracted_slice_ratio:.3f} "
            f"top={best.top_coverage_ratio:.3f} extent={best.vertical_extent_ratio:.3f} "
            f"path={best.pointcloud_path.name}"
        )
        return int(best_index)

    def _build_base_mesh(self, base_pointcloud_path: Path) -> trimesh.Trimesh:
        surface_cfg = SurfaceModelConfig()
        surface_cfg.out_subdir = self.config.base_mesh_out_subdir
        surface_cfg.voxel_size = self.config.voxel_size
        surface_cfg.max_points = max(int(self.config.max_points_per_phase), 4000)
        surface_cfg.hidden_dim = int(self.config.canonical_hidden_dim)
        surface_cfg.hidden_layers = int(self.config.canonical_hidden_layers)
        surface_cfg.train_steps = int(self.config.base_mesh_train_steps or max(80, min(self.config.train_steps, 120)))
        surface_cfg.learning_rate = float(self.config.learning_rate)
        surface_cfg.normal_weight = float(self.config.normal_weight)
        surface_cfg.mesh_resolution = int(self.config.mesh_resolution)
        surface_cfg.mesh_threshold = float(self.config.mesh_threshold)
        surface_cfg.eval_batch_size = int(self.config.eval_batch_size)
        surface_cfg.min_face_count = int(self.config.min_face_count)
        surface_cfg.random_seed = int(self.config.random_seed)
        surface_cfg.use_cuda_if_available = bool(self.config.use_cuda_if_available)
        results = reconstruct_meshes_from_pointclouds([base_pointcloud_path], surface_cfg)
        if not results:
            raise RuntimeError("Failed to build shared base mesh from reference phase point cloud")
        mesh = trimesh.load(results[0].mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Failed to load shared base mesh: {results[0].mesh_path}")
        return mesh

    @staticmethod
    def _normalize_base_mesh(mesh: trimesh.Trimesh, center: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
        vertices = ((np.asarray(mesh.vertices, dtype=np.float32) - center[None, :]) / scale).astype(np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        return vertices, faces

    @staticmethod
    def _sample_surface_plan(mesh: trimesh.Trimesh, sample_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        areas = np.asarray(mesh.area_faces, dtype=np.float64)
        area_sum = float(np.sum(areas))
        if area_sum <= 0.0:
            raise ValueError("Base mesh has invalid face areas")
        rng = np.random.default_rng(seed)
        face_indices = rng.choice(len(mesh.faces), size=int(sample_count), replace=True, p=areas / area_sum)
        u = rng.random(sample_count)
        v = rng.random(sample_count)
        sqrt_u = np.sqrt(u)
        barycentric = np.column_stack([1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v]).astype(np.float32)
        return face_indices.astype(np.int64), barycentric

    @staticmethod
    def _sample_reference_points_numpy(
        base_vertices: np.ndarray,
        base_faces: np.ndarray,
        face_indices: np.ndarray,
        barycentric: np.ndarray,
    ) -> np.ndarray:
        tri_vertices = base_vertices[base_faces[face_indices]]
        return (tri_vertices * barycentric[:, :, None]).sum(axis=1).astype(np.float32)

    @staticmethod
    def _sample_reference_offsets_numpy(
        vertex_offsets: np.ndarray,
        base_faces: np.ndarray,
        face_indices: np.ndarray,
        barycentric: np.ndarray,
    ) -> np.ndarray:
        tri_offsets = vertex_offsets[:, base_faces[face_indices]]
        return (tri_offsets * barycentric[None, :, :, None]).sum(axis=2).astype(np.float32)

    @staticmethod
    def _sample_reference_values_numpy(
        vertex_values: np.ndarray,
        base_faces: np.ndarray,
        face_indices: np.ndarray,
        barycentric: np.ndarray,
    ) -> np.ndarray:
        tri_values = vertex_values[base_faces[face_indices]]
        return (tri_values * barycentric).sum(axis=1).astype(np.float32)

    @staticmethod
    def _build_edges(faces: np.ndarray) -> np.ndarray:
        edges = np.vstack([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]).astype(np.int64)
        edges = np.sort(edges, axis=1)
        return np.unique(edges, axis=0)

    def _support_radius_normalized(self, scale: float) -> float:
        radius_mm = max(float(self.config.observation_support_radius_mm), float(self.config.voxel_size))
        return float(radius_mm / max(scale, 1e-8))

    @staticmethod
    def _compute_phase_support_weights(
        reference_points: np.ndarray,
        observations: list[PhaseObservation],
        support_radius: float,
    ) -> np.ndarray:
        if support_radius <= 1e-8:
            return np.ones((len(observations), len(reference_points)), dtype=np.float32)

        phase_support = np.zeros((len(observations), len(reference_points)), dtype=np.float32)
        for index, observation in enumerate(observations):
            tree = cKDTree(observation.points)
            distances, _ = tree.query(reference_points, k=1)
            normalized = np.asarray(distances, dtype=np.float32) / float(support_radius)
            phase_support[index] = np.exp(-(normalized ** 2)).astype(np.float32)
        return np.clip(phase_support, 0.0, 1.0)

    @staticmethod
    def _compute_phase_support_mask(
        reference_points: np.ndarray,
        observations: list[PhaseObservation],
        support_radius: float,
    ) -> np.ndarray:
        if support_radius <= 1e-8:
            return np.ones((len(observations), len(reference_points)), dtype=np.float32)

        phase_support = np.zeros((len(observations), len(reference_points)), dtype=np.float32)
        for index, observation in enumerate(observations):
            tree = cKDTree(observation.points)
            distances, _ = tree.query(reference_points, k=1)
            phase_support[index] = (np.asarray(distances, dtype=np.float32) <= float(support_radius)).astype(np.float32)
        return phase_support

    @staticmethod
    def _initialize_offsets(
        base_vertices: np.ndarray,
        observations: list[PhaseObservation],
        vertex_support_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        offsets = np.zeros((len(observations), len(base_vertices), 3), dtype=np.float32)
        for index, observation in enumerate(observations):
            tree = cKDTree(observation.points)
            _, nn_indices = tree.query(base_vertices, k=1)
            nearest_offsets = observation.points[np.asarray(nn_indices, dtype=np.int64)] - base_vertices
            if vertex_support_weights is None:
                offsets[index] = nearest_offsets
            else:
                support = vertex_support_weights[index][:, None]
                offsets[index] = support * nearest_offsets
        return offsets.astype(np.float32)

    @staticmethod
    def _neighbor_mean_numpy(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        sums = np.zeros_like(values, dtype=np.float32)
        counts = np.zeros((values.shape[0], 1), dtype=np.float32)
        np.add.at(sums, edges[:, 0], values[edges[:, 1]])
        np.add.at(sums, edges[:, 1], values[edges[:, 0]])
        np.add.at(counts, edges[:, 0], 1.0)
        np.add.at(counts, edges[:, 1], 1.0)
        return sums / np.clip(counts, 1.0, None)

    @staticmethod
    def _neighbor_mean_torch(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        if len(edges) == 0:
            return values
        sums = torch.zeros_like(values)
        counts = torch.zeros((values.shape[0], 1), dtype=values.dtype, device=values.device)
        sums.index_add_(0, edges[:, 0], values[edges[:, 1]])
        sums.index_add_(0, edges[:, 1], values[edges[:, 0]])
        counts.index_add_(0, edges[:, 0], torch.ones((len(edges), 1), dtype=values.dtype, device=values.device))
        counts.index_add_(0, edges[:, 1], torch.ones((len(edges), 1), dtype=values.dtype, device=values.device))
        return sums / counts.clamp_min(1.0)

    @staticmethod
    def _principal_axis_coordinates(vertices: np.ndarray) -> np.ndarray:
        if len(vertices) == 0:
            return np.empty((0,), dtype=np.float32)
        centered = vertices - np.mean(vertices, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0].astype(np.float32)
        coordinates = centered @ axis
        coord_min = float(np.min(coordinates))
        coord_max = float(np.max(coordinates))
        coord_span = max(coord_max - coord_min, 1e-6)
        normalized = (coordinates - coord_min) / coord_span
        return normalized.astype(np.float32)

    @staticmethod
    def _principal_axis_frame(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(vertices) == 0:
            empty_scalar = np.empty((0,), dtype=np.float32)
            empty_vector = np.empty((0, 3), dtype=np.float32)
            return empty_scalar, np.zeros((3,), dtype=np.float32), empty_vector
        centered = vertices - np.mean(vertices, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0].astype(np.float32)
        coordinates = centered @ axis
        coord_min = float(np.min(coordinates))
        coord_max = float(np.max(coordinates))
        coord_span = max(coord_max - coord_min, 1e-6)
        normalized = ((coordinates - coord_min) / coord_span).astype(np.float32)
        radial_vectors = centered - coordinates[:, None] * axis[None, :]
        radial_norm = np.linalg.norm(radial_vectors, axis=1, keepdims=True)
        radial_directions = radial_vectors / np.clip(radial_norm, 1e-6, None)
        return normalized, axis, radial_directions.astype(np.float32)

    def _propagate_unsupported_displacements(
        self,
        displacements: np.ndarray,
        vertex_support_weights: np.ndarray,
        edges: np.ndarray,
    ) -> np.ndarray:
        iterations = max(int(self.config.unsupported_propagation_iterations), 0)
        neighbor_weight = float(np.clip(self.config.unsupported_propagation_neighbor_weight, 0.0, 1.0))
        if iterations <= 0 or len(edges) == 0:
            return displacements.astype(np.float32)

        propagated = displacements.astype(np.float32, copy=True)
        for phase_index in range(len(propagated)):
            base_displacement = propagated[phase_index].copy()
            current = propagated[phase_index].copy()
            support = vertex_support_weights[phase_index][:, None].astype(np.float32)
            unsupported = 1.0 - support
            for _ in range(iterations):
                neighbor_mean = self._neighbor_mean_numpy(current, edges)
                unsupported_update = (1.0 - neighbor_weight) * current + neighbor_weight * neighbor_mean
                current = support * base_displacement + unsupported * unsupported_update
            propagated[phase_index] = current
        return propagated.astype(np.float32)

    def _sample_phase_triplet_indices(self, phase_count: int, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        overlap_stride = self._overlap_neighbor_stride(phase_count)
        if overlap_stride is not None:
            stride = int(overlap_stride)
        else:
            stride = max(1, int(round(float(self.config.temporal_delta_phase) * phase_count)))
        centers = rng.choice(phase_count, size=batch_size, replace=True)
        prevs = (centers - stride) % phase_count
        nexts = (centers + stride) % phase_count
        return prevs.astype(np.int64), centers.astype(np.int64), nexts.astype(np.int64)

    def _correspondence_regularization_losses(
        self,
        offsets: torch.Tensor,
        base_vertices: torch.Tensor,
        base_faces: torch.Tensor,
        face_indices: torch.Tensor,
        barycentric: torch.Tensor,
        prev_idx: torch.Tensor,
        center_idx: torch.Tensor,
        next_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prev_vertices = base_vertices.unsqueeze(0) + offsets[prev_idx]
        center_vertices = base_vertices.unsqueeze(0) + offsets[center_idx]
        next_vertices = base_vertices.unsqueeze(0) + offsets[next_idx]

        tri_prev = prev_vertices[:, base_faces[face_indices]]
        tri_center = center_vertices[:, base_faces[face_indices]]
        tri_next = next_vertices[:, base_faces[face_indices]]
        barycentric_expanded = barycentric.unsqueeze(0).unsqueeze(-1)

        pos_prev = (tri_prev * barycentric_expanded).sum(dim=2)
        pos_center = (tri_center * barycentric_expanded).sum(dim=2)
        pos_next = (tri_next * barycentric_expanded).sum(dim=2)

        temporal_loss = 0.5 * (((pos_next - pos_center) ** 2).mean() + ((pos_center - pos_prev) ** 2).mean())
        acceleration_loss = ((pos_next - 2.0 * pos_center + pos_prev) ** 2).mean()
        phase_consistency_loss = ((pos_next - pos_prev) ** 2).mean()
        return temporal_loss, acceleration_loss, phase_consistency_loss

    def _uses_sliding_window_supervision(self) -> bool:
        return (
            self.config.overlap_aware_sampling
            and str(self.config.supervision_binning_strategy).lower() == "sliding_time_window"
        )

    def _overlap_neighbor_stride(self, phase_count: int) -> int | None:
        if not self._uses_sliding_window_supervision() or phase_count < 3:
            return None
        configured_stride = self.config.overlap_neighbor_stride_bins
        if configured_stride is not None:
            stride = max(int(configured_stride), 1)
        else:
            step_seconds = float(self.config.supervision_step_seconds or 0.0)
            window_seconds = float(self.config.supervision_window_seconds or 0.0)
            if step_seconds <= 0.0 or window_seconds <= 0.0:
                return None
            stride = max(int(np.ceil(window_seconds / step_seconds)), 1)
        return min(stride, max(1, phase_count // 2))

    def _overlap_loss_scale(self) -> float:
        if not self._uses_sliding_window_supervision():
            return 1.0
        step_seconds = float(self.config.supervision_step_seconds or 0.0)
        window_seconds = float(self.config.supervision_window_seconds or 0.0)
        if step_seconds <= 0.0 or window_seconds <= 0.0:
            return 1.0
        independence = step_seconds / max(window_seconds, 1e-8)
        return float(np.clip(independence, self.config.overlap_loss_min_scale, 1.0))

    @staticmethod
    def _sample_surface_from_vertices(
        vertices: torch.Tensor,
        faces: torch.Tensor,
        face_indices: torch.Tensor,
        barycentric: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tri_vertices = vertices[faces[face_indices]]
        samples = (tri_vertices * barycentric.unsqueeze(-1)).sum(dim=1)
        face_normals = torch.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0], dim=1)
        face_normals = F.normalize(face_normals, dim=-1)
        return samples, face_normals

    @staticmethod
    def _interpolate_offsets(phases: np.ndarray, displacements: np.ndarray, query_phase: float) -> np.ndarray:
        wrapped = float(query_phase % 1.0)
        if len(phases) == 1:
            return displacements[0]
        idx = int(np.searchsorted(phases, wrapped))
        idx0 = (idx - 1) % len(phases)
        idx1 = idx % len(phases)
        phase0 = float(phases[idx0])
        phase1 = float(phases[idx1])
        delta = (phase1 - phase0) % 1.0
        if delta <= 1e-8:
            alpha = 0.0
        else:
            offset = (wrapped - phase0) % 1.0
            alpha = offset / delta
        return ((1.0 - alpha) * displacements[idx0] + alpha * displacements[idx1]).astype(np.float32)

    def _correspondence_schedule_scale(self, step: int) -> float:
        if self.config.train_steps <= 1:
            return 1.0
        start_fraction = float(np.clip(self.config.correspondence_start_fraction, 0.0, 1.0))
        ramp_fraction = float(np.clip(self.config.correspondence_ramp_fraction, 0.0, 1.0 - start_fraction))
        progress = float(step) / float(max(self.config.train_steps - 1, 1))
        if progress <= start_fraction:
            return 0.0
        if ramp_fraction <= 1e-8:
            return 1.0
        ramp_progress = (progress - start_fraction) / ramp_fraction
        return float(np.clip(ramp_progress, 0.0, 1.0))

    def _bootstrap_schedule_scale(self, step: int) -> float:
        if self.config.train_steps <= 1:
            return 1.0
        decay_fraction = float(np.clip(self.config.bootstrap_decay_fraction, 0.0, 1.0))
        if decay_fraction <= 1e-8:
            return 0.0
        progress = float(step) / float(max(self.config.train_steps - 1, 1))
        return float(np.clip(1.0 - progress / decay_fraction, 0.0, 1.0))

    def _teacher_schedule_scale(self, step: int) -> float:
        if self.config.train_steps <= 1:
            return 1.0
        start_fraction = float(np.clip(self.config.bootstrap_teacher_start_fraction, 0.0, 1.0))
        ramp_fraction = float(np.clip(self.config.bootstrap_teacher_ramp_fraction, 0.0, 1.0 - start_fraction))
        progress = float(step) / float(max(self.config.train_steps - 1, 1))
        if progress <= start_fraction:
            return 0.0
        if ramp_fraction <= 1e-8:
            return 1.0
        ramp_progress = (progress - start_fraction) / ramp_fraction
        return float(np.clip(ramp_progress, 0.0, 1.0))

    @staticmethod
    def _build_low_rank_motion_basis(initial_offsets: np.ndarray, basis_rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phase_count, vertex_count, _ = initial_offsets.shape
        flat_offsets = initial_offsets.reshape(phase_count, -1).astype(np.float32)
        mean_flat = flat_offsets.mean(axis=0, keepdims=True).astype(np.float32)
        centered = flat_offsets - mean_flat
        max_rank = min(max(int(basis_rank), 1), phase_count, centered.shape[1])
        if np.allclose(centered, 0.0):
            basis_flat = np.zeros((max_rank, centered.shape[1]), dtype=np.float32)
            coeff_targets = np.zeros((phase_count, max_rank), dtype=np.float32)
        else:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            basis_flat = vt[:max_rank].astype(np.float32)
            coeff_targets = (centered @ basis_flat.T).astype(np.float32)
        return (
            mean_flat.reshape(vertex_count, 3).astype(np.float32),
            basis_flat.reshape(max_rank, vertex_count, 3).astype(np.float32),
            coeff_targets.astype(np.float32),
        )

    @staticmethod
    def _phase_harmonics_numpy(phases: np.ndarray, harmonics: int) -> np.ndarray:
        phase_values = phases.reshape(-1, 1).astype(np.float32)
        encoded: list[np.ndarray] = []
        for order in range(1, max(int(harmonics), 1) + 1):
            encoded.append(np.sin(2.0 * np.pi * order * phase_values).astype(np.float32))
            encoded.append(np.cos(2.0 * np.pi * order * phase_values).astype(np.float32))
        return np.concatenate(encoded, axis=1).astype(np.float32)

    def _initialize_basis_coefficient_head(
        self,
        coefficient_field: PhaseConditionedBasisCoefficients,
        phases: list[float],
        coeff_targets: np.ndarray,
    ) -> None:
        if coeff_targets.size == 0:
            return
        encoded_phase = self._phase_harmonics_numpy(np.asarray(phases, dtype=np.float32), coefficient_field.phase_encoder.harmonics)
        design = np.concatenate([encoded_phase, np.ones((len(encoded_phase), 1), dtype=np.float32)], axis=1)
        solution, _, _, _ = np.linalg.lstsq(design, coeff_targets.astype(np.float32), rcond=None)
        weight = solution[:-1].T.astype(np.float32)
        bias = solution[-1].astype(np.float32)
        with torch.no_grad():
            coefficient_field.linear.weight.copy_(torch.from_numpy(weight))
            coefficient_field.linear.bias.copy_(torch.from_numpy(bias))

    def _postprocess_dynamic_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            mesh = max(components, key=lambda item: (len(item.faces), item.area))
        if self.config.smoothing_iterations > 0:
            trimesh.smoothing.filter_taubin(
                mesh,
                lamb=self.config.taubin_lambda,
                nu=self.config.taubin_mu,
                iterations=self.config.smoothing_iterations,
            )
        if hasattr(mesh, "unique_faces"):
            mesh.update_faces(mesh.unique_faces())
        if hasattr(mesh, "nondegenerate_faces"):
            mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
        return mesh

    def fit(
        self,
        pointcloud_paths: list[Path],
        phase_confidences: Mapping[Path, float] | None = None,
        phase_summaries: Mapping[Path, PointCloudPhaseSummary] | None = None,
    ) -> SharedTopologyDynamicFit:
        cfg = self.config
        observations, center, scale = self._prepare_phase_observations(
            pointcloud_paths,
            phase_confidences=phase_confidences,
            phase_summaries=phase_summaries,
        )
        base_phase_index = self._select_base_phase_index(observations)
        base_mesh = self._build_base_mesh(observations[base_phase_index].pointcloud_path)
        base_vertices_np, base_faces_np = self._normalize_base_mesh(base_mesh, center, scale)
        sample_count = max(int(cfg.surface_batch_size * 2), 4096)
        face_indices_np, barycentric_np = self._sample_surface_plan(base_mesh, sample_count, cfg.random_seed)
        reference_points_np = self._sample_reference_points_numpy(base_vertices_np, base_faces_np, face_indices_np, barycentric_np)
        support_radius = self._support_radius_normalized(scale)
        phase_support_weights_np = self._compute_phase_support_weights(reference_points_np, observations, support_radius)
        vertex_support_weights_np = self._compute_phase_support_weights(base_vertices_np, observations, support_radius)
        vertex_support_mask_np = self._compute_phase_support_mask(base_vertices_np, observations, support_radius)
        edges_np = self._build_edges(base_faces_np)
        initial_offsets_np = self._initialize_offsets(base_vertices_np, observations, vertex_support_mask_np)

        rng = np.random.default_rng(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_seed)

        base_vertices = torch.from_numpy(base_vertices_np).to(self.device)
        base_faces = torch.from_numpy(base_faces_np).to(self.device)
        face_indices = torch.from_numpy(face_indices_np).to(self.device)
        barycentric = torch.from_numpy(barycentric_np).to(self.device)
        phase_support_weights = torch.from_numpy(phase_support_weights_np).to(self.device)
        vertex_support_weights = torch.from_numpy(vertex_support_weights_np).to(self.device)
        vertex_support_mask = torch.from_numpy(vertex_support_mask_np).to(self.device)
        initial_offsets = torch.from_numpy(initial_offsets_np).to(self.device)
        edges = torch.from_numpy(edges_np).to(self.device)
        phase_points = [torch.from_numpy(item.points).to(self.device) for item in observations]
        phase_normals = [torch.from_numpy(item.normals).to(self.device) for item in observations]
        phase_centroids = [torch.from_numpy(item.centroid).to(self.device) for item in observations]
        phase_weights = np.asarray([item.weight for item in observations], dtype=np.float64)
        sample_probabilities = phase_weights / max(np.sum(phase_weights), 1e-8)
        phase_weight_tensor = torch.tensor(phase_weights, dtype=torch.float32, device=self.device)

        offsets = torch.nn.Parameter(torch.from_numpy(initial_offsets_np).to(self.device))
        optimizer = torch.optim.Adam([offsets], lr=cfg.learning_rate)
        overlap_loss_scale = self._overlap_loss_scale()

        for step in range(cfg.train_steps):
            correspondence_schedule_scale = self._correspondence_schedule_scale(step)
            bootstrap_schedule_scale = self._bootstrap_schedule_scale(step)
            phase_id = int(rng.choice(len(observations), p=sample_probabilities))
            target_points = phase_points[phase_id]
            target_normals = phase_normals[phase_id]
            target_centroid = phase_centroids[phase_id]
            phase_weight = phase_weight_tensor[phase_id]
            support_weights = phase_support_weights[phase_id]
            support_sum = support_weights.sum().clamp_min(1e-6)
            supported_vertex_weights = vertex_support_mask[phase_id]
            supported_vertex_sum = supported_vertex_weights.sum().clamp_min(1e-6)
            vertex_support = vertex_support_weights[phase_id]
            unsupported_vertex_weights = 1.0 - vertex_support_mask[phase_id]
            unsupported_vertex_sum = unsupported_vertex_weights.sum().clamp_min(1e-6)

            vertices = base_vertices + offsets[phase_id]
            pred_samples, pred_normals = self._sample_surface_from_vertices(vertices, base_faces, face_indices, barycentric)

            distances = torch.cdist(pred_samples, target_points)
            pred_to_target, pred_nn = distances.min(dim=1)
            target_to_pred, _ = distances.min(dim=0)
            surface_loss = phase_weight * (((support_weights * pred_to_target.pow(2)).sum() / support_sum) + target_to_pred.pow(2).mean())

            nearest_target_normals = target_normals[pred_nn]
            normal_alignment = 1.0 - torch.abs(torch.sum(pred_normals * nearest_target_normals, dim=-1))
            normal_loss = phase_weight * ((support_weights * normal_alignment).sum() / support_sum)
            supported_pred_centroid = (pred_samples * support_weights.unsqueeze(-1)).sum(dim=0) / support_sum
            centroid_loss = phase_weight * ((supported_pred_centroid - target_centroid) ** 2).mean()
            spatial_loss = ((offsets[phase_id, edges[:, 0]] - offsets[phase_id, edges[:, 1]]) ** 2).mean()
            sample_offset_triangles = offsets[phase_id, base_faces[face_indices]]
            sampled_offsets = (sample_offset_triangles * barycentric.unsqueeze(-1)).sum(dim=1)
            bootstrap_offset_loss = (
                (supported_vertex_weights * (offsets[phase_id] - initial_offsets[phase_id]).pow(2).mean(dim=-1)).sum()
                / supported_vertex_sum
            )
            unsupported_anchor_loss = ((unsupported_vertex_weights * offsets[phase_id].pow(2).mean(dim=-1)).sum() / unsupported_vertex_sum)
            unsupported_neighbor_mean = self._neighbor_mean_torch(offsets[phase_id], edges)
            unsupported_laplacian_loss = (
                (unsupported_vertex_weights * (offsets[phase_id] - unsupported_neighbor_mean).pow(2).mean(dim=-1)).sum()
                / unsupported_vertex_sum
            )

            vertex_count = len(base_vertices_np)
            temporal_count = min(cfg.temporal_batch_size, vertex_count)
            vertex_indices = rng.choice(vertex_count, size=temporal_count, replace=vertex_count < temporal_count)
            vertex_indices_t = torch.from_numpy(vertex_indices.astype(np.int64)).to(self.device)
            prev_idx_np, center_idx_np, next_idx_np = self._sample_phase_triplet_indices(len(observations), temporal_count, rng)
            prev_idx = torch.from_numpy(prev_idx_np).to(self.device)
            center_idx = torch.from_numpy(center_idx_np).to(self.device)
            next_idx = torch.from_numpy(next_idx_np).to(self.device)

            deformation_prev = offsets[prev_idx, vertex_indices_t]
            deformation_now = offsets[center_idx, vertex_indices_t]
            deformation_next = offsets[next_idx, vertex_indices_t]
            temporal_loss = 0.5 * (((deformation_next - deformation_now) ** 2).mean() + ((deformation_now - deformation_prev) ** 2).mean())
            temporal_accel_loss = ((deformation_next - 2.0 * deformation_now + deformation_prev) ** 2).mean()
            phase_consistency_loss = ((deformation_next - deformation_prev) ** 2).mean()
            deformation_loss = (deformation_now ** 2).mean()

            correspondence_temporal_loss = torch.zeros((), device=self.device)
            correspondence_accel_loss = torch.zeros((), device=self.device)
            correspondence_phase_loss = torch.zeros((), device=self.device)
            if (
                correspondence_schedule_scale > 0.0
                and (
                    cfg.correspondence_temporal_weight > 0.0
                    or cfg.correspondence_acceleration_weight > 0.0
                    or cfg.correspondence_phase_consistency_weight > 0.0
                )
            ):
                correspondence_count = min(cfg.temporal_batch_size, len(face_indices_np))
                correspondence_ids = rng.choice(len(face_indices_np), size=correspondence_count, replace=len(face_indices_np) < correspondence_count)
                correspondence_ids_t = torch.from_numpy(correspondence_ids.astype(np.int64)).to(self.device)
                corr_face_indices = face_indices[correspondence_ids_t]
                corr_barycentric = barycentric[correspondence_ids_t]
                correspondence_temporal_loss, correspondence_accel_loss, correspondence_phase_loss = self._correspondence_regularization_losses(
                    offsets,
                    base_vertices,
                    base_faces,
                    corr_face_indices,
                    corr_barycentric,
                    prev_idx,
                    center_idx,
                    next_idx,
                )

            periodic_stride = max(1, int(round(float(cfg.temporal_delta_phase) * len(observations))))
            periodic_vertices = torch.from_numpy(rng.choice(vertex_count, size=temporal_count, replace=vertex_count < temporal_count).astype(np.int64)).to(self.device)
            deformation_zero = offsets[0, periodic_vertices]
            deformation_one = offsets[-1, periodic_vertices]
            deformation_delta = offsets[periodic_stride % len(observations), periodic_vertices]
            deformation_one_minus_delta = offsets[(-periodic_stride) % len(observations), periodic_vertices]
            periodicity_loss = ((deformation_zero - deformation_one) ** 2).mean()
            periodicity_loss = periodicity_loss + 0.5 * (((deformation_delta - deformation_zero) - (deformation_one - deformation_one_minus_delta)) ** 2).mean()

            loss = (
                surface_loss
                + cfg.normal_weight * normal_loss
                + cfg.centroid_weight * centroid_loss
                + cfg.spatial_smoothness_weight * spatial_loss
                + (bootstrap_schedule_scale * cfg.bootstrap_offset_weight) * bootstrap_offset_loss
                + cfg.unsupported_anchor_weight * unsupported_anchor_loss
                + cfg.unsupported_laplacian_weight * unsupported_laplacian_loss
                + (cfg.temporal_weight * overlap_loss_scale) * temporal_loss
                + cfg.temporal_acceleration_weight * temporal_accel_loss
                + (cfg.phase_consistency_weight * overlap_loss_scale) * phase_consistency_loss
                + (correspondence_schedule_scale * cfg.correspondence_temporal_weight * overlap_loss_scale) * correspondence_temporal_loss
                + (correspondence_schedule_scale * cfg.correspondence_acceleration_weight) * correspondence_accel_loss
                + (correspondence_schedule_scale * cfg.correspondence_phase_consistency_weight * overlap_loss_scale) * correspondence_phase_loss
                + cfg.periodicity_weight * periodicity_loss
                + cfg.deformation_weight * deformation_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (step + 1) % max(1, cfg.train_steps // 4) == 0 or step == 0:
                print(
                    "[SharedDynamic] "
                    f"step={step + 1}/{cfg.train_steps} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"surface={float(surface_loss.detach().cpu()):.6f} "
                    f"normal={float(normal_loss.detach().cpu()):.6f} "
                    f"centroid={float(centroid_loss.detach().cpu()):.6f} "
                    f"spatial={float(spatial_loss.detach().cpu()):.6f} "
                    f"boot={float(bootstrap_offset_loss.detach().cpu()):.6f} "
                    f"boot_scale={bootstrap_schedule_scale:.3f} "
                    f"anchor={float(unsupported_anchor_loss.detach().cpu()):.6f} "
                    f"lap={float(unsupported_laplacian_loss.detach().cpu()):.6f} "
                    f"temporal={float(temporal_loss.detach().cpu()):.6f} "
                    f"accel={float(temporal_accel_loss.detach().cpu()):.6f} "
                    f"phase={float(phase_consistency_loss.detach().cpu()):.6f} "
                    f"corr_scale={correspondence_schedule_scale:.3f} "
                    f"corr_temporal={float(correspondence_temporal_loss.detach().cpu()):.6f} "
                    f"corr_accel={float(correspondence_accel_loss.detach().cpu()):.6f} "
                    f"corr_phase={float(correspondence_phase_loss.detach().cpu()):.6f} "
                    f"periodic={float(periodicity_loss.detach().cpu()):.6f} "
                    f"deform={float(deformation_loss.detach().cpu()):.6f}"
                )

        run_dir = observations[0].pointcloud_path.parent
        base_mesh_path = run_dir / cfg.out_subdir / "dynamic_base_mesh.ply"
        base_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        denormalized_base_vertices = base_vertices_np * scale + center[None, :]
        base_mesh_export = trimesh.Trimesh(vertices=denormalized_base_vertices, faces=base_faces_np, process=False)
        _mesh_export(base_mesh_export, base_mesh_path)

        final_displacements = offsets.detach().cpu().numpy().astype(np.float32)
        final_displacements = self._propagate_unsupported_displacements(final_displacements, vertex_support_mask_np, edges_np)

        return SharedTopologyDynamicFit(
            base_vertices=base_vertices_np,
            base_faces=base_faces_np,
            displacements=final_displacements,
            phases=[item.phase for item in observations],
            center=center,
            scale=float(scale),
        )

    def _mesh_from_offsets(self, fit: SharedTopologyDynamicFit, displacements: np.ndarray) -> trimesh.Trimesh:
        vertices = (fit.base_vertices + displacements) * fit.scale + fit.center[None, :]
        mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=fit.base_faces.astype(np.int64), process=False)
        return self._postprocess_dynamic_mesh(mesh)

    def _export_phase_meshes(self, fit: SharedTopologyDynamicFit, mesh_dir: Path) -> list[DynamicMeshBuildResult]:
        results: list[DynamicMeshBuildResult] = []
        for phase, displacement in zip(fit.phases, fit.displacements):
            mesh = self._mesh_from_offsets(fit, displacement)
            if len(mesh.faces) < self.config.min_face_count:
                print(f"[SharedDynamic] phase={phase:.3f}: face count too low ({len(mesh.faces)}), skip")
                continue
            mesh_name = f"dynamic_phase_{phase:.3f}.ply".replace(" ", "")
            mesh_path = mesh_dir / mesh_name
            _mesh_export(mesh, mesh_path)
            print(
                f"[SharedDynamic] wrote {mesh_path} "
                f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight})"
            )
            results.append(
                DynamicMeshBuildResult(
                    phase=float(phase),
                    mesh_path=mesh_path,
                    vertices=int(len(mesh.vertices)),
                    faces=int(len(mesh.faces)),
                    watertight=bool(mesh.is_watertight),
                    method="shared_topology_vertex_field",
                )
            )
        return results

    def _select_timeline_samples(self, timeline_samples: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
        stride = max(int(self.config.timeline_stride), 1)
        selected = list(timeline_samples[::stride])
        max_exports = self.config.timeline_max_exports
        if max_exports is None or len(selected) <= max_exports:
            return selected
        if max_exports <= 1:
            return [selected[0]]
        indices = np.linspace(0, len(selected) - 1, max_exports, dtype=int)
        return [selected[index] for index in indices]

    def _export_timeline_meshes(
        self,
        fit: SharedTopologyDynamicFit,
        mesh_dir: Path,
        timeline_samples: list[tuple[int, float, float]],
    ) -> list[DynamicTimelineMeshBuildResult]:
        selected_samples = self._select_timeline_samples(timeline_samples)
        results: list[DynamicTimelineMeshBuildResult] = []
        phases_np = np.asarray(fit.phases, dtype=np.float32)
        for frame_index, timestamp, phase in selected_samples:
            displacement = self._interpolate_offsets(phases_np, fit.displacements, phase)
            mesh = self._mesh_from_offsets(fit, displacement)
            mesh_name = f"dynamic_timeline_{frame_index:05d}_{phase:.3f}.ply".replace(" ", "")
            mesh_path = mesh_dir / mesh_name
            _mesh_export(mesh, mesh_path)
            results.append(
                DynamicTimelineMeshBuildResult(
                    frame_index=int(frame_index),
                    timestamp=float(timestamp),
                    phase=float(phase),
                    mesh_path=mesh_path,
                    vertices=int(len(mesh.vertices)),
                    faces=int(len(mesh.faces)),
                    watertight=bool(mesh.is_watertight),
                    method="shared_topology_vertex_field_timeline",
                )
            )

        summary_path = mesh_dir / "dynamic_timeline_mesh_summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_index", "timestamp", "phase", "mesh", "vertices", "faces", "watertight", "method"])
            for result in results:
                writer.writerow([
                    result.frame_index,
                    f"{result.timestamp:.6f}",
                    f"{result.phase:.6f}",
                    result.mesh_path.name,
                    result.vertices,
                    result.faces,
                    int(result.watertight),
                    result.method,
                ])
        return results

    def reconstruct(
        self,
        pointcloud_paths: list[Path],
        phase_confidences: Mapping[Path, float] | None = None,
        phase_summaries: Mapping[Path, PointCloudPhaseSummary] | None = None,
        timeline_samples: list[tuple[int, float, float]] | None = None,
    ) -> tuple[list[DynamicMeshBuildResult], list[DynamicTimelineMeshBuildResult]]:
        if not pointcloud_paths:
            return [], []

        fit = self.fit(
            pointcloud_paths,
            phase_confidences=phase_confidences,
            phase_summaries=phase_summaries,
        )
        run_dir = pointcloud_paths[0].parent
        mesh_dir = run_dir / self.config.out_subdir
        mesh_dir.mkdir(parents=True, exist_ok=True)
        results = self._export_phase_meshes(fit, mesh_dir)

        summary_path = mesh_dir / "dynamic_mesh_summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["phase", "mesh", "vertices", "faces", "watertight", "method"])
            for result in results:
                writer.writerow([
                    f"{result.phase:.6f}",
                    result.mesh_path.name,
                    result.vertices,
                    result.faces,
                    int(result.watertight),
                    result.method,
                ])

        timeline_results: list[DynamicTimelineMeshBuildResult] = []
        if self.config.export_timeline_meshes and timeline_samples:
            timeline_dir = run_dir / self.config.timeline_out_subdir
            timeline_dir.mkdir(parents=True, exist_ok=True)
            timeline_results = self._export_timeline_meshes(fit, timeline_dir, timeline_samples)
            print(f"[SharedDynamic] exported {len(timeline_results)} timeline meshes to {timeline_dir}")
        return results, timeline_results


class SharedTopologyReferenceCorrespondenceReconstructor(CanonicalPhaseDeformationFieldReconstructor):
    def _export_phase_meshes(self, fit: SharedTopologyDynamicFit, mesh_dir: Path) -> list[DynamicMeshBuildResult]:
        results: list[DynamicMeshBuildResult] = []
        for phase, displacement in zip(fit.phases, fit.displacements):
            mesh = self._mesh_from_offsets(fit, displacement)
            if len(mesh.faces) < self.config.min_face_count:
                print(f"[SharedDynamicRef] phase={phase:.3f}: face count too low ({len(mesh.faces)}), skip")
                continue
            mesh_name = f"dynamic_phase_{phase:.3f}.ply".replace(" ", "")
            mesh_path = mesh_dir / mesh_name
            _mesh_export(mesh, mesh_path)
            print(
                f"[SharedDynamicRef] wrote {mesh_path} "
                f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight})"
            )
            results.append(
                DynamicMeshBuildResult(
                    phase=float(phase),
                    mesh_path=mesh_path,
                    vertices=int(len(mesh.vertices)),
                    faces=int(len(mesh.faces)),
                    watertight=bool(mesh.is_watertight),
                    method="shared_topology_vertex_field_reference_correspondence",
                )
            )
        return results

    def _export_timeline_meshes(
        self,
        fit: SharedTopologyDynamicFit,
        mesh_dir: Path,
        timeline_samples: list[tuple[int, float, float]],
    ) -> list[DynamicTimelineMeshBuildResult]:
        selected_samples = self._select_timeline_samples(timeline_samples)
        results: list[DynamicTimelineMeshBuildResult] = []
        phases_np = np.asarray(fit.phases, dtype=np.float32)
        for frame_index, timestamp, phase in selected_samples:
            displacement = self._interpolate_offsets(phases_np, fit.displacements, phase)
            mesh = self._mesh_from_offsets(fit, displacement)
            mesh_name = f"dynamic_timeline_{frame_index:05d}_{phase:.3f}.ply".replace(" ", "")
            mesh_path = mesh_dir / mesh_name
            _mesh_export(mesh, mesh_path)
            results.append(
                DynamicTimelineMeshBuildResult(
                    frame_index=int(frame_index),
                    timestamp=float(timestamp),
                    phase=float(phase),
                    mesh_path=mesh_path,
                    vertices=int(len(mesh.vertices)),
                    faces=int(len(mesh.faces)),
                    watertight=bool(mesh.is_watertight),
                    method="shared_topology_vertex_field_reference_correspondence_timeline",
                )
            )

        summary_path = mesh_dir / "dynamic_timeline_mesh_summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_index", "timestamp", "phase", "mesh", "vertices", "faces", "watertight", "method"])
            for result in results:
                writer.writerow([
                    result.frame_index,
                    f"{result.timestamp:.6f}",
                    f"{result.phase:.6f}",
                    result.mesh_path.name,
                    result.vertices,
                    result.faces,
                    int(result.watertight),
                    result.method,
                ])
        return results


class SharedTopologyContinuousFieldReconstructor(CanonicalPhaseDeformationFieldReconstructor):
    """Shared-topology dynamic model with a continuous phase-conditioned deformation field."""

    @staticmethod
    def _phase_tensor(phase: float, count: int, device: torch.device) -> torch.Tensor:
        return torch.full((count,), float(phase), dtype=torch.float32, device=device)

    def fit(
        self,
        pointcloud_paths: list[Path],
        phase_confidences: Mapping[Path, float] | None = None,
        phase_summaries: Mapping[Path, PointCloudPhaseSummary] | None = None,
    ) -> SharedTopologyDynamicFit:
        cfg = self.config
        observations, center, scale = self._prepare_phase_observations(
            pointcloud_paths,
            phase_confidences=phase_confidences,
            phase_summaries=phase_summaries,
        )
        base_phase_index = self._select_base_phase_index(observations)
        base_mesh = self._build_base_mesh(observations[base_phase_index].pointcloud_path)
        base_vertices_np, base_faces_np = self._normalize_base_mesh(base_mesh, center, scale)

        sample_count = max(int(cfg.surface_batch_size * 2), 4096)
        face_indices_np, barycentric_np = self._sample_surface_plan(base_mesh, sample_count, cfg.random_seed)
        reference_points_np = self._sample_reference_points_numpy(base_vertices_np, base_faces_np, face_indices_np, barycentric_np)
        support_radius = self._support_radius_normalized(scale)
        phase_support_weights_np = self._compute_phase_support_weights(reference_points_np, observations, support_radius)
        vertex_support_mask_np = self._compute_phase_support_mask(base_vertices_np, observations, support_radius)
        initial_offsets_np = self._initialize_offsets(base_vertices_np, observations, vertex_support_mask_np)
        bootstrap_reference_offsets_np = self._sample_reference_offsets_numpy(initial_offsets_np, base_faces_np, face_indices_np, barycentric_np)
        edges_np = self._build_edges(base_faces_np)

        rng = np.random.default_rng(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_seed)

        base_vertices = torch.from_numpy(base_vertices_np).to(self.device)
        base_faces = torch.from_numpy(base_faces_np).to(self.device)
        face_indices = torch.from_numpy(face_indices_np).to(self.device)
        barycentric = torch.from_numpy(barycentric_np).to(self.device)
        reference_points = torch.from_numpy(reference_points_np).to(self.device)
        phase_support_weights = torch.from_numpy(phase_support_weights_np).to(self.device)
        vertex_support_mask = torch.from_numpy(vertex_support_mask_np).to(self.device)
        bootstrap_offsets = torch.from_numpy(initial_offsets_np).to(self.device)
        bootstrap_reference_offsets = torch.from_numpy(bootstrap_reference_offsets_np).to(self.device)
        edges = torch.from_numpy(edges_np).to(self.device)
        phase_points = [torch.from_numpy(item.points).to(self.device) for item in observations]
        phase_normals = [torch.from_numpy(item.normals).to(self.device) for item in observations]
        phase_centroids = [torch.from_numpy(item.centroid).to(self.device) for item in observations]
        phase_weights = np.asarray([item.weight for item in observations], dtype=np.float64)
        sample_probabilities = phase_weights / max(np.sum(phase_weights), 1e-8)
        phase_weight_tensor = torch.tensor(phase_weights, dtype=torch.float32, device=self.device)

        deformation_field = PhaseConditionedDeformationField(
            cfg.deformation_hidden_dim,
            cfg.deformation_hidden_layers,
            cfg.phase_harmonics,
        ).to(self.device)
        optimizer = torch.optim.Adam(deformation_field.parameters(), lr=cfg.learning_rate)
        overlap_loss_scale = self._overlap_loss_scale()

        for step in range(cfg.train_steps):
            correspondence_schedule_scale = self._correspondence_schedule_scale(step)
            bootstrap_schedule_scale = self._bootstrap_schedule_scale(step)
            phase_id = int(rng.choice(len(observations), p=sample_probabilities))
            phase_value = observations[phase_id].phase
            target_points = phase_points[phase_id]
            target_normals = phase_normals[phase_id]
            target_centroid = phase_centroids[phase_id]
            phase_weight = phase_weight_tensor[phase_id]
            support_weights = phase_support_weights[phase_id]
            support_sum = support_weights.sum().clamp_min(1e-6)
            unsupported_vertex_weights = 1.0 - vertex_support_mask[phase_id]
            unsupported_vertex_sum = unsupported_vertex_weights.sum().clamp_min(1e-6)

            phase_tensor_vertices = self._phase_tensor(phase_value, len(base_vertices), self.device)
            residual_vertex_offsets = deformation_field(base_vertices, phase_tensor_vertices)
            bootstrap_vertex_offsets = bootstrap_offsets[phase_id]
            vertex_offsets = bootstrap_vertex_offsets + residual_vertex_offsets
            vertices = base_vertices + vertex_offsets
            pred_samples, pred_normals = self._sample_surface_from_vertices(vertices, base_faces, face_indices, barycentric)

            distances = torch.cdist(pred_samples, target_points)
            pred_to_target, pred_nn = distances.min(dim=1)
            target_to_pred, _ = distances.min(dim=0)
            surface_loss = phase_weight * (((support_weights * pred_to_target.pow(2)).sum() / support_sum) + target_to_pred.pow(2).mean())

            nearest_target_normals = target_normals[pred_nn]
            normal_alignment = 1.0 - torch.abs(torch.sum(pred_normals * nearest_target_normals, dim=-1))
            normal_loss = phase_weight * ((support_weights * normal_alignment).sum() / support_sum)
            supported_pred_centroid = (pred_samples * support_weights.unsqueeze(-1)).sum(dim=0) / support_sum
            centroid_loss = phase_weight * ((supported_pred_centroid - target_centroid) ** 2).mean()
            spatial_loss = ((residual_vertex_offsets[edges[:, 0]] - residual_vertex_offsets[edges[:, 1]]) ** 2).mean()
            unsupported_anchor_loss = ((unsupported_vertex_weights * vertex_offsets.pow(2).mean(dim=-1)).sum() / unsupported_vertex_sum)
            unsupported_neighbor_mean = self._neighbor_mean_torch(vertex_offsets, edges)
            unsupported_laplacian_loss = (
                (unsupported_vertex_weights * (vertex_offsets - unsupported_neighbor_mean).pow(2).mean(dim=-1)).sum()
                / unsupported_vertex_sum
            )
            supported_vertex_weights = vertex_support_mask[phase_id]
            supported_vertex_sum = supported_vertex_weights.sum().clamp_min(1e-6)
            bootstrap_offset_loss = (
                (supported_vertex_weights * (vertex_offsets - bootstrap_vertex_offsets).pow(2).mean(dim=-1)).sum()
                / supported_vertex_sum
            )

            temporal_count = min(cfg.temporal_batch_size, len(reference_points_np))
            temporal_idx = rng.choice(len(reference_points_np), size=temporal_count, replace=len(reference_points_np) < temporal_count)
            temporal_idx_t = torch.from_numpy(temporal_idx.astype(np.int64)).to(self.device)
            ref_temporal = reference_points[temporal_idx_t]
            prev_idx_np, center_idx_np, next_idx_np = self._sample_phase_triplet_indices(len(observations), temporal_count, rng)
            prev_idx = torch.from_numpy(prev_idx_np).to(self.device)
            center_idx = torch.from_numpy(center_idx_np).to(self.device)
            next_idx = torch.from_numpy(next_idx_np).to(self.device)
            prev_phase = torch.tensor([observations[index].phase for index in prev_idx_np], dtype=torch.float32, device=self.device)
            center_phase = torch.tensor([observations[index].phase for index in center_idx_np], dtype=torch.float32, device=self.device)
            next_phase = torch.tensor([observations[index].phase for index in next_idx_np], dtype=torch.float32, device=self.device)

            residual_prev = deformation_field(ref_temporal, prev_phase)
            residual_now = deformation_field(ref_temporal, center_phase)
            residual_next = deformation_field(ref_temporal, next_phase)
            bootstrap_prev = bootstrap_reference_offsets[prev_idx, temporal_idx_t]
            bootstrap_now = bootstrap_reference_offsets[center_idx, temporal_idx_t]
            bootstrap_next = bootstrap_reference_offsets[next_idx, temporal_idx_t]
            deformation_prev = bootstrap_prev + residual_prev
            deformation_now = bootstrap_now + residual_now
            deformation_next = bootstrap_next + residual_next
            temporal_loss = 0.5 * (((deformation_next - deformation_now) ** 2).mean() + ((deformation_now - deformation_prev) ** 2).mean())
            temporal_accel_loss = ((deformation_next - 2.0 * deformation_now + deformation_prev) ** 2).mean()
            phase_consistency_loss = ((deformation_next - deformation_prev) ** 2).mean()
            deformation_loss = (residual_now ** 2).mean()

            correspondence_temporal_loss = torch.zeros((), device=self.device)
            correspondence_accel_loss = torch.zeros((), device=self.device)
            correspondence_phase_loss = torch.zeros((), device=self.device)
            if (
                correspondence_schedule_scale > 0.0
                and (
                    cfg.correspondence_temporal_weight > 0.0
                    or cfg.correspondence_acceleration_weight > 0.0
                    or cfg.correspondence_phase_consistency_weight > 0.0
                )
            ):
                correspondence_count = min(cfg.temporal_batch_size, len(reference_points_np))
                correspondence_idx = rng.choice(len(reference_points_np), size=correspondence_count, replace=len(reference_points_np) < correspondence_count)
                correspondence_idx_t = torch.from_numpy(correspondence_idx.astype(np.int64)).to(self.device)
                ref_corr = reference_points[correspondence_idx_t]
                corr_bootstrap_prev = bootstrap_reference_offsets[prev_idx, correspondence_idx_t]
                corr_bootstrap_now = bootstrap_reference_offsets[center_idx, correspondence_idx_t]
                corr_bootstrap_next = bootstrap_reference_offsets[next_idx, correspondence_idx_t]
                corr_prev = ref_corr + corr_bootstrap_prev + deformation_field(ref_corr, prev_phase)
                corr_now = ref_corr + corr_bootstrap_now + deformation_field(ref_corr, center_phase)
                corr_next = ref_corr + corr_bootstrap_next + deformation_field(ref_corr, next_phase)
                correspondence_temporal_loss = 0.5 * (((corr_next - corr_now) ** 2).mean() + ((corr_now - corr_prev) ** 2).mean())
                correspondence_accel_loss = ((corr_next - 2.0 * corr_now + corr_prev) ** 2).mean()
                correspondence_phase_loss = ((corr_next - corr_prev) ** 2).mean()

            periodic_count = min(cfg.temporal_batch_size, len(reference_points_np))
            periodic_idx = rng.choice(len(reference_points_np), size=periodic_count, replace=len(reference_points_np) < periodic_count)
            periodic_idx_t = torch.from_numpy(periodic_idx.astype(np.int64)).to(self.device)
            ref_periodic = reference_points[periodic_idx_t]
            periodic_stride = max(1, int(round(float(cfg.temporal_delta_phase) * len(observations))))
            zero_phase = self._phase_tensor(observations[0].phase, len(ref_periodic), self.device)
            one_phase = self._phase_tensor(observations[-1].phase, len(ref_periodic), self.device)
            delta_phase = self._phase_tensor(observations[periodic_stride % len(observations)].phase, len(ref_periodic), self.device)
            one_minus_delta_phase = self._phase_tensor(observations[(-periodic_stride) % len(observations)].phase, len(ref_periodic), self.device)
            bootstrap_zero = bootstrap_reference_offsets[0, periodic_idx_t]
            bootstrap_one = bootstrap_reference_offsets[-1, periodic_idx_t]
            bootstrap_delta = bootstrap_reference_offsets[periodic_stride % len(observations), periodic_idx_t]
            bootstrap_one_minus_delta = bootstrap_reference_offsets[(-periodic_stride) % len(observations), periodic_idx_t]
            deformation_zero = bootstrap_zero + deformation_field(ref_periodic, zero_phase)
            deformation_one = bootstrap_one + deformation_field(ref_periodic, one_phase)
            deformation_delta = bootstrap_delta + deformation_field(ref_periodic, delta_phase)
            deformation_one_minus_delta = bootstrap_one_minus_delta + deformation_field(ref_periodic, one_minus_delta_phase)
            periodicity_loss = ((deformation_zero - deformation_one) ** 2).mean()
            periodicity_loss = periodicity_loss + 0.5 * (((deformation_delta - deformation_zero) - (deformation_one - deformation_one_minus_delta)) ** 2).mean()

            loss = (
                surface_loss
                + cfg.normal_weight * normal_loss
                + cfg.centroid_weight * centroid_loss
                + cfg.spatial_smoothness_weight * spatial_loss
                + (bootstrap_schedule_scale * cfg.bootstrap_offset_weight) * bootstrap_offset_loss
                + cfg.unsupported_anchor_weight * unsupported_anchor_loss
                + cfg.unsupported_laplacian_weight * unsupported_laplacian_loss
                + (cfg.temporal_weight * overlap_loss_scale) * temporal_loss
                + cfg.temporal_acceleration_weight * temporal_accel_loss
                + (cfg.phase_consistency_weight * overlap_loss_scale) * phase_consistency_loss
                + (correspondence_schedule_scale * cfg.correspondence_temporal_weight * overlap_loss_scale) * correspondence_temporal_loss
                + (correspondence_schedule_scale * cfg.correspondence_acceleration_weight) * correspondence_accel_loss
                + (correspondence_schedule_scale * cfg.correspondence_phase_consistency_weight * overlap_loss_scale) * correspondence_phase_loss
                + cfg.periodicity_weight * periodicity_loss
                + cfg.deformation_weight * deformation_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (step + 1) % max(1, cfg.train_steps // 4) == 0 or step == 0:
                print(
                    "[SharedDynamicCont] "
                    f"step={step + 1}/{cfg.train_steps} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"surface={float(surface_loss.detach().cpu()):.6f} "
                    f"normal={float(normal_loss.detach().cpu()):.6f} "
                    f"centroid={float(centroid_loss.detach().cpu()):.6f} "
                    f"spatial={float(spatial_loss.detach().cpu()):.6f} "
                    f"boot={float(bootstrap_offset_loss.detach().cpu()):.6f} "
                    f"boot_scale={bootstrap_schedule_scale:.3f} "
                    f"anchor={float(unsupported_anchor_loss.detach().cpu()):.6f} "
                    f"lap={float(unsupported_laplacian_loss.detach().cpu()):.6f} "
                    f"temporal={float(temporal_loss.detach().cpu()):.6f} "
                    f"accel={float(temporal_accel_loss.detach().cpu()):.6f} "
                    f"phase={float(phase_consistency_loss.detach().cpu()):.6f} "
                    f"corr_scale={correspondence_schedule_scale:.3f} "
                    f"corr_temporal={float(correspondence_temporal_loss.detach().cpu()):.6f} "
                    f"corr_accel={float(correspondence_accel_loss.detach().cpu()):.6f} "
                    f"corr_phase={float(correspondence_phase_loss.detach().cpu()):.6f} "
                    f"periodic={float(periodicity_loss.detach().cpu()):.6f} "
                    f"deform={float(deformation_loss.detach().cpu()):.6f}"
                )

        with torch.no_grad():
            phase_displacements: list[np.ndarray] = []
            for phase_index, observation in enumerate(observations):
                phase_tensor = self._phase_tensor(observation.phase, len(base_vertices), self.device)
                residual = deformation_field(base_vertices, phase_tensor).detach().cpu().numpy().astype(np.float32)
                displacement = initial_offsets_np[phase_index] + residual
                phase_displacements.append(displacement)

        run_dir = observations[0].pointcloud_path.parent
        base_mesh_path = run_dir / cfg.out_subdir / "dynamic_base_mesh.ply"
        base_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        denormalized_base_vertices = base_vertices_np * scale + center[None, :]
        base_mesh_export = trimesh.Trimesh(vertices=denormalized_base_vertices, faces=base_faces_np, process=False)
        _mesh_export(base_mesh_export, base_mesh_path)

        final_displacements = np.stack(phase_displacements, axis=0)
        final_displacements = self._propagate_unsupported_displacements(final_displacements, vertex_support_mask_np, edges_np)
        return SharedTopologyDynamicFit(
            base_vertices=base_vertices_np,
            base_faces=base_faces_np,
            displacements=final_displacements,
            phases=[item.phase for item in observations],
            center=center,
            scale=float(scale),
        )

    def _export_phase_meshes(self, fit: SharedTopologyDynamicFit, mesh_dir: Path) -> list[DynamicMeshBuildResult]:
        results: list[DynamicMeshBuildResult] = []
        for phase, displacement in zip(fit.phases, fit.displacements):
            mesh = self._mesh_from_offsets(fit, displacement)
            if len(mesh.faces) < self.config.min_face_count:
                print(f"[SharedDynamicCont] phase={phase:.3f}: face count too low ({len(mesh.faces)}), skip")
                continue
            mesh_name = f"dynamic_phase_{phase:.3f}.ply".replace(" ", "")
            mesh_path = mesh_dir / mesh_name
            _mesh_export(mesh, mesh_path)
            print(
                f"[SharedDynamicCont] wrote {mesh_path} "
                f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight})"
            )
            results.append(
                DynamicMeshBuildResult(
                    phase=float(phase),
                    mesh_path=mesh_path,
                    vertices=int(len(mesh.vertices)),
                    faces=int(len(mesh.faces)),
                    watertight=bool(mesh.is_watertight),
                    method="shared_topology_deformation_field_reference_correspondence",
                )
            )
        return results


class SharedTopologyGlobalBasisResidualReconstructor(CanonicalPhaseDeformationFieldReconstructor):
    """Shared-topology dynamic model with low-rank global motion plus local residual offsets."""

    @staticmethod
    def _coefficient_diagnostics_path(mesh_dir: Path) -> Path:
        return mesh_dir / "global_basis_diagnostics.csv"

    def _write_basis_diagnostics(self, fit: SharedTopologyDynamicFit, mesh_dir: Path) -> None:
        if fit.global_coefficients is None:
            return
        summary_path = self._coefficient_diagnostics_path(mesh_dir)
        coeff_count = int(fit.global_coefficients.shape[1]) if fit.global_coefficients.ndim == 2 else 0
        header = ["phase"]
        header.extend([f"coeff_{index:02d}" for index in range(coeff_count)])
        header.extend(["global_energy", "residual_energy", "residual_global_ratio"])
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for index, phase in enumerate(fit.phases):
                global_energy = float(0.0 if fit.global_energy_per_phase is None else fit.global_energy_per_phase[index])
                residual_energy = float(0.0 if fit.residual_energy_per_phase is None else fit.residual_energy_per_phase[index])
                ratio = residual_energy / max(global_energy, 1e-8)
                row = [f"{float(phase):.6f}"]
                row.extend([f"{float(value):.6f}" for value in fit.global_coefficients[index]])
                row.extend([f"{global_energy:.6f}", f"{residual_energy:.6f}", f"{ratio:.6f}"])
                writer.writerow(row)

    @staticmethod
    def _phase_tensor(phase: float, count: int, device: torch.device) -> torch.Tensor:
        return torch.full((count,), float(phase), dtype=torch.float32, device=device)

    def fit(
        self,
        pointcloud_paths: list[Path],
        phase_confidences: Mapping[Path, float] | None = None,
        phase_summaries: Mapping[Path, PointCloudPhaseSummary] | None = None,
    ) -> SharedTopologyDynamicFit:
        cfg = self.config
        observations, center, scale = self._prepare_phase_observations(
            pointcloud_paths,
            phase_confidences=phase_confidences,
            phase_summaries=phase_summaries,
        )
        base_phase_index = self._select_base_phase_index(observations)
        base_mesh = self._build_base_mesh(observations[base_phase_index].pointcloud_path)
        base_vertices_np, base_faces_np = self._normalize_base_mesh(base_mesh, center, scale)

        sample_count = max(int(cfg.surface_batch_size * 2), 4096)
        face_indices_np, barycentric_np = self._sample_surface_plan(base_mesh, sample_count, cfg.random_seed)
        reference_points_np = self._sample_reference_points_numpy(base_vertices_np, base_faces_np, face_indices_np, barycentric_np)
        support_radius = self._support_radius_normalized(scale)
        phase_support_weights_np = self._compute_phase_support_weights(reference_points_np, observations, support_radius)
        vertex_support_mask_np = self._compute_phase_support_mask(base_vertices_np, observations, support_radius)
        initial_offsets_np = self._initialize_offsets(base_vertices_np, observations, vertex_support_mask_np)
        mean_vertex_offset_np, basis_vertex_offsets_np, coeff_targets_np = self._build_low_rank_motion_basis(
            initial_offsets_np,
            cfg.global_motion_basis_rank,
        )
        mean_reference_offset_np = self._sample_reference_offsets_numpy(
            mean_vertex_offset_np[None, ...],
            base_faces_np,
            face_indices_np,
            barycentric_np,
        )[0]
        basis_reference_offsets_np = self._sample_reference_offsets_numpy(
            basis_vertex_offsets_np,
            base_faces_np,
            face_indices_np,
            barycentric_np,
        )
        bootstrap_residual_offsets_np = initial_offsets_np - (
            mean_vertex_offset_np[None, ...]
            + np.einsum("pr,rvj->pvj", coeff_targets_np, basis_vertex_offsets_np, optimize=True)
        )
        bootstrap_residual_reference_offsets_np = self._sample_reference_offsets_numpy(
            bootstrap_residual_offsets_np,
            base_faces_np,
            face_indices_np,
            barycentric_np,
        )
        edges_np = self._build_edges(base_faces_np)

        rng = np.random.default_rng(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_seed)

        base_vertices = torch.from_numpy(base_vertices_np).to(self.device)
        base_faces = torch.from_numpy(base_faces_np).to(self.device)
        face_indices = torch.from_numpy(face_indices_np).to(self.device)
        barycentric = torch.from_numpy(barycentric_np).to(self.device)
        reference_points = torch.from_numpy(reference_points_np).to(self.device)
        phase_support_weights = torch.from_numpy(phase_support_weights_np).to(self.device)
        vertex_support_mask = torch.from_numpy(vertex_support_mask_np).to(self.device)
        bootstrap_residual_offsets = torch.from_numpy(bootstrap_residual_offsets_np).to(self.device)
        mean_vertex_offset = torch.from_numpy(mean_vertex_offset_np).to(self.device)
        basis_vertex_offsets = torch.from_numpy(basis_vertex_offsets_np).to(self.device)
        mean_reference_offset = torch.from_numpy(mean_reference_offset_np).to(self.device)
        basis_reference_offsets = torch.from_numpy(basis_reference_offsets_np).to(self.device)
        bootstrap_residual_reference_offsets = torch.from_numpy(bootstrap_residual_reference_offsets_np).to(self.device)
        coeff_targets = torch.from_numpy(coeff_targets_np).to(self.device)
        edges = torch.from_numpy(edges_np).to(self.device)
        base_axis_coordinates_np, base_axis_direction_np, base_radial_directions_np = self._principal_axis_frame(base_vertices_np)
        reference_axis_coordinates_np = self._sample_reference_values_numpy(
            base_axis_coordinates_np,
            base_faces_np,
            face_indices_np,
            barycentric_np,
        )
        base_vertex_axis_coordinates = torch.from_numpy(base_axis_coordinates_np).to(self.device)
        base_axis_direction = torch.from_numpy(base_axis_direction_np).to(self.device)
        base_vertex_radial_directions = torch.from_numpy(base_radial_directions_np).to(self.device)
        reference_axis_coordinates = torch.from_numpy(reference_axis_coordinates_np).to(self.device)
        phase_points = [torch.from_numpy(item.points).to(self.device) for item in observations]
        phase_normals = [torch.from_numpy(item.normals).to(self.device) for item in observations]
        phase_point_weights = [torch.from_numpy(item.point_weights).to(self.device) for item in observations]
        phase_centroids = [torch.from_numpy(item.centroid).to(self.device) for item in observations]
        phase_weights = np.asarray([item.weight for item in observations], dtype=np.float64)
        sample_probabilities = phase_weights / max(np.sum(phase_weights), 1e-8)
        phase_weight_tensor = torch.tensor(phase_weights, dtype=torch.float32, device=self.device)

        teacher_phase_points: list[torch.Tensor | None] = [None] * len(observations)
        if cfg.bootstrap_teacher_weight > 0.0:
            teacher_dir = pointcloud_paths[0].parent / SurfaceModelConfig().out_subdir
            teacher_mesh_paths = {
                _phase_index_from_path(path): path
                for path in teacher_dir.glob("*.ply")
                if _phase_index_from_path(path) is not None
            }
            for index, observation in enumerate(observations):
                teacher_path = teacher_mesh_paths.get(_phase_index_from_path(observation.pointcloud_path))
                if teacher_path is None:
                    continue
                teacher_mesh = trimesh.load(teacher_path, force="mesh", process=False)
                teacher_vertices = ((np.asarray(teacher_mesh.vertices, dtype=np.float32) - center[None, :]) / scale).astype(np.float32)
                teacher_phase_points[index] = torch.from_numpy(teacher_vertices).to(self.device)

        coefficient_field = PhaseConditionedBasisCoefficients(
            basis_vertex_offsets_np.shape[0],
            cfg.phase_harmonics,
        ).to(self.device)
        self._initialize_basis_coefficient_head(
            coefficient_field,
            [item.phase for item in observations],
            coeff_targets_np,
        )
        residual_field = PhaseConditionedDeformationField(
            cfg.deformation_hidden_dim,
            cfg.deformation_hidden_layers,
            cfg.phase_harmonics,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            list(coefficient_field.parameters()) + list(residual_field.parameters()),
            lr=cfg.learning_rate,
        )
        overlap_loss_scale = self._overlap_loss_scale()

        for step in range(cfg.train_steps):
            correspondence_schedule_scale = self._correspondence_schedule_scale(step)
            bootstrap_schedule_scale = self._bootstrap_schedule_scale(step)
            teacher_schedule_scale = self._teacher_schedule_scale(step)
            phase_id = int(rng.choice(len(observations), p=sample_probabilities))
            phase_value = observations[phase_id].phase
            phase_scalar = torch.tensor([phase_value], dtype=torch.float32, device=self.device)
            target_points = phase_points[phase_id]
            target_normals = phase_normals[phase_id]
            target_point_weights = phase_point_weights[phase_id]
            phase_weight = phase_weight_tensor[phase_id]
            support_weights = phase_support_weights[phase_id]
            support_sum = support_weights.sum().clamp_min(1e-6)
            target_point_weight_sum = target_point_weights.sum().clamp_min(1e-6)
            target_weighted_centroid = (target_points * target_point_weights.unsqueeze(-1)).sum(dim=0) / target_point_weight_sum
            unsupported_vertex_weights = 1.0 - vertex_support_mask[phase_id]
            unsupported_vertex_sum = unsupported_vertex_weights.sum().clamp_min(1e-6)
            supported_vertex_weights = vertex_support_mask[phase_id]
            supported_vertex_sum = supported_vertex_weights.sum().clamp_min(1e-6)

            global_coefficients = coefficient_field(phase_scalar).squeeze(0)
            global_vertex_offsets = mean_vertex_offset + torch.einsum("r,rvj->vj", global_coefficients, basis_vertex_offsets)
            phase_tensor_vertices = self._phase_tensor(phase_value, len(base_vertices), self.device)
            residual_vertex_offsets = residual_field(base_vertices, phase_tensor_vertices)
            global_vertices = base_vertices + global_vertex_offsets
            vertex_offsets = global_vertex_offsets + residual_vertex_offsets
            vertices = base_vertices + vertex_offsets
            residual_vertex_energy = residual_vertex_offsets.pow(2).mean(dim=-1)
            wave_band_weights = supported_vertex_weights * residual_vertex_energy
            wave_band_weight_sum = wave_band_weights.sum().clamp_min(1e-6)
            wave_band_center = (wave_band_weights * base_vertex_axis_coordinates).sum() / wave_band_weight_sum
            global_samples = None
            if cfg.bootstrap_teacher_global_only:
                global_samples, _ = self._sample_surface_from_vertices(global_vertices, base_faces, face_indices, barycentric)
            pred_samples, pred_normals = self._sample_surface_from_vertices(vertices, base_faces, face_indices, barycentric)

            wave_data_band_width = max(float(cfg.wave_band_data_term_band_width), 1e-3)
            wave_data_gate = torch.exp(
                -0.5 * ((reference_axis_coordinates - wave_band_center) / wave_data_band_width).pow(2)
            )
            wave_data_support = support_weights.clamp(0.0, 1.0).pow(float(cfg.wave_band_data_term_support_power))
            wave_data_boost = 1.0 + float(cfg.wave_band_data_term_boost_weight) * wave_data_gate * wave_data_support
            boosted_support_weights = support_weights * wave_data_boost
            boosted_support_sum = boosted_support_weights.sum().clamp_min(1e-6)

            distances = torch.cdist(pred_samples, target_points)
            pred_to_target, pred_nn = distances.min(dim=1)
            target_to_pred, _ = distances.min(dim=0)
            surface_loss = phase_weight * (
                ((boosted_support_weights * pred_to_target.pow(2)).sum() / boosted_support_sum)
                + ((target_point_weights * target_to_pred.pow(2)).sum() / target_point_weight_sum)
            )

            nearest_target_normals = target_normals[pred_nn]
            normal_alignment = 1.0 - torch.abs(torch.sum(pred_normals * nearest_target_normals, dim=-1))
            normal_loss = phase_weight * ((boosted_support_weights * normal_alignment).sum() / boosted_support_sum)
            supported_pred_centroid = (pred_samples * support_weights.unsqueeze(-1)).sum(dim=0) / support_sum
            centroid_loss = phase_weight * ((supported_pred_centroid - target_weighted_centroid) ** 2).mean()
            spatial_loss = ((residual_vertex_offsets[edges[:, 0]] - residual_vertex_offsets[edges[:, 1]]) ** 2).mean()
            unsupported_anchor_loss = (
                (unsupported_vertex_weights * residual_vertex_offsets.pow(2).mean(dim=-1)).sum() / unsupported_vertex_sum
            )
            unsupported_neighbor_mean = self._neighbor_mean_torch(residual_vertex_offsets, edges)
            unsupported_laplacian_loss = (
                (unsupported_vertex_weights * (residual_vertex_offsets - unsupported_neighbor_mean).pow(2).mean(dim=-1)).sum()
                / unsupported_vertex_sum
            )
            bootstrap_residual_vertex_offsets = bootstrap_residual_offsets[phase_id]
            bootstrap_global_vertex_offsets = mean_vertex_offset + torch.einsum(
                "r,rvj->vj",
                coeff_targets[phase_id],
                basis_vertex_offsets,
            )
            bootstrap_offset_loss = (
                (
                    supported_vertex_weights
                    * (residual_vertex_offsets - bootstrap_residual_vertex_offsets).pow(2).mean(dim=-1)
                ).sum()
                / supported_vertex_sum
            )
            bootstrap_teacher_loss = torch.zeros((), device=self.device)
            teacher_points = teacher_phase_points[phase_id]
            if teacher_points is not None and len(teacher_points) > 0:
                teacher_source = global_samples if cfg.bootstrap_teacher_global_only and global_samples is not None else pred_samples
                teacher_distances = torch.cdist(teacher_source, teacher_points)
                teacher_pred_to_target, _ = teacher_distances.min(dim=1)
                teacher_target_to_pred, teacher_target_nn = teacher_distances.min(dim=0)
                pred_to_target_weight = float(cfg.bootstrap_teacher_pred_to_target_weight)
                target_to_pred_weight = float(cfg.bootstrap_teacher_target_to_pred_weight)
                if cfg.bootstrap_teacher_support_aware:
                    teacher_support = support_weights.clamp(0.0, 1.0)
                    teacher_support = teacher_support.pow(float(cfg.bootstrap_teacher_support_power))
                    if float(cfg.bootstrap_teacher_support_floor) > 0.0:
                        teacher_support = torch.clamp(teacher_support, min=float(cfg.bootstrap_teacher_support_floor))
                    teacher_support_sum = teacher_support.sum().clamp_min(1e-6)
                    teacher_target_support = teacher_support[teacher_target_nn]
                    teacher_target_support_sum = teacher_target_support.sum().clamp_min(1e-6)
                    bootstrap_teacher_loss = phase_weight * (
                        pred_to_target_weight * ((teacher_support * teacher_pred_to_target.pow(2)).sum() / teacher_support_sum)
                        + target_to_pred_weight * ((teacher_target_support * teacher_target_to_pred.pow(2)).sum() / teacher_target_support_sum)
                    )
                else:
                    bootstrap_teacher_loss = phase_weight * (
                        pred_to_target_weight * teacher_pred_to_target.pow(2).mean()
                        + target_to_pred_weight * teacher_target_to_pred.pow(2).mean()
                    )
            coefficient_bootstrap_loss = (global_coefficients - coeff_targets[phase_id]).pow(2).mean()
            residual_mean_loss = residual_vertex_offsets.mean(dim=0).pow(2).mean()
            projection_offsets = residual_vertex_offsets
            if cfg.residual_basis_projection_support_aware:
                projection_offsets = projection_offsets * unsupported_vertex_weights.unsqueeze(-1)
            basis_projection = torch.matmul(
                basis_vertex_offsets.reshape(basis_vertex_offsets.shape[0], -1),
                projection_offsets.reshape(-1),
            )
            residual_basis_projection_loss = basis_projection.pow(2).mean() / float(max(len(base_vertices_np), 1))
            bootstrap_residual_energy = bootstrap_residual_vertex_offsets.pow(2).mean(dim=-1)
            if cfg.residual_global_ratio_support_aware:
                global_energy = (supported_vertex_weights * global_vertex_offsets.pow(2).mean(dim=-1)).sum() / supported_vertex_sum
                bootstrap_global_energy = (
                    supported_vertex_weights * bootstrap_global_vertex_offsets.pow(2).mean(dim=-1)
                ).sum() / supported_vertex_sum
                residual_energy = (supported_vertex_weights * residual_vertex_offsets.pow(2).mean(dim=-1)).sum() / supported_vertex_sum
            else:
                global_energy = global_vertex_offsets.pow(2).mean()
                bootstrap_global_energy = bootstrap_global_vertex_offsets.pow(2).mean()
                residual_energy = residual_vertex_offsets.pow(2).mean()
            residual_locality_budget = (
                float(cfg.residual_locality_budget_scale) * bootstrap_residual_energy
                + float(cfg.residual_locality_global_budget_scale) * bootstrap_global_energy
            )
            residual_locality_excess = F.relu(residual_vertex_energy - residual_locality_budget)
            residual_locality_loss = (
                (supported_vertex_weights * residual_locality_excess.pow(2)).sum() / supported_vertex_sum
            )
            wave_band_variance = (
                wave_band_weights * (base_vertex_axis_coordinates - wave_band_center).pow(2)
            ).sum() / wave_band_weight_sum
            residual_wave_band_std = torch.sqrt(wave_band_variance.clamp_min(1e-8))
            residual_wave_band_concentration_loss = F.relu(
                residual_wave_band_std - float(cfg.residual_wave_band_target_std)
            ).pow(2)
            wave_direction_width = max(float(cfg.residual_wave_direction_band_width), 1e-3)
            wave_direction_gate = torch.exp(
                -0.5 * ((base_vertex_axis_coordinates - wave_band_center) / wave_direction_width).pow(2)
            )
            wave_direction_weights = supported_vertex_weights * wave_direction_gate
            wave_direction_weight_sum = wave_direction_weights.sum().clamp_min(1e-6)
            axial_component = torch.einsum("vj,j->v", residual_vertex_offsets, base_axis_direction)
            radial_component = (residual_vertex_offsets * base_vertex_radial_directions).sum(dim=-1)
            residual_total_energy = residual_vertex_offsets.pow(2).sum(dim=-1)
            tangential_energy = (residual_total_energy - axial_component.pow(2) - radial_component.pow(2)).clamp_min(0.0)
            wave_direction_total_energy = (wave_direction_weights * residual_total_energy).sum().clamp_min(1e-6)
            wave_direction_axial_ratio = (wave_direction_weights * axial_component.pow(2)).sum() / wave_direction_total_energy
            wave_direction_outward_ratio = (wave_direction_weights * F.relu(radial_component).pow(2)).sum() / wave_direction_total_energy
            wave_direction_tangential_ratio = (wave_direction_weights * tangential_energy).sum() / wave_direction_total_energy
            residual_wave_direction_loss = (
                wave_direction_axial_ratio
                + wave_direction_outward_ratio
                + float(cfg.residual_wave_direction_tangential_weight) * wave_direction_tangential_ratio
            )
            ratio_denominator = torch.maximum(global_energy, 0.5 * bootstrap_global_energy).clamp_min(1e-6)
            residual_global_ratio = residual_energy / ratio_denominator
            residual_global_ratio_loss = F.relu(
                residual_global_ratio - float(cfg.residual_global_ratio_target)
            ).pow(2)

            temporal_count = min(cfg.temporal_batch_size, len(reference_points_np))
            temporal_idx = rng.choice(len(reference_points_np), size=temporal_count, replace=len(reference_points_np) < temporal_count)
            temporal_idx_t = torch.from_numpy(temporal_idx.astype(np.int64)).to(self.device)
            ref_temporal = reference_points[temporal_idx_t]
            ref_mean_temporal = mean_reference_offset[temporal_idx_t]
            ref_basis_temporal = basis_reference_offsets[:, temporal_idx_t].permute(1, 0, 2)
            prev_idx_np, center_idx_np, next_idx_np = self._sample_phase_triplet_indices(len(observations), temporal_count, rng)
            prev_phase = torch.tensor([observations[index].phase for index in prev_idx_np], dtype=torch.float32, device=self.device)
            center_phase = torch.tensor([observations[index].phase for index in center_idx_np], dtype=torch.float32, device=self.device)
            next_phase = torch.tensor([observations[index].phase for index in next_idx_np], dtype=torch.float32, device=self.device)

            coeff_prev = coefficient_field(prev_phase)
            coeff_now = coefficient_field(center_phase)
            coeff_next = coefficient_field(next_phase)
            coefficient_temporal_loss = 0.5 * (((coeff_next - coeff_now) ** 2).mean() + ((coeff_now - coeff_prev) ** 2).mean())
            coefficient_accel_loss = ((coeff_next - 2.0 * coeff_now + coeff_prev) ** 2).mean()
            global_prev = ref_mean_temporal + torch.einsum("br,brj->bj", coeff_prev, ref_basis_temporal)
            global_now = ref_mean_temporal + torch.einsum("br,brj->bj", coeff_now, ref_basis_temporal)
            global_next = ref_mean_temporal + torch.einsum("br,brj->bj", coeff_next, ref_basis_temporal)
            residual_prev = residual_field(ref_temporal, prev_phase)
            residual_now = residual_field(ref_temporal, center_phase)
            residual_next = residual_field(ref_temporal, next_phase)
            deformation_prev = global_prev + residual_prev
            deformation_now = global_now + residual_now
            deformation_next = global_next + residual_next
            temporal_loss = 0.5 * (((deformation_next - deformation_now) ** 2).mean() + ((deformation_now - deformation_prev) ** 2).mean())
            temporal_accel_loss = ((deformation_next - 2.0 * deformation_now + deformation_prev) ** 2).mean()
            phase_consistency_loss = ((deformation_next - deformation_prev) ** 2).mean()
            deformation_loss = (residual_now ** 2).mean()

            correspondence_temporal_loss = torch.zeros((), device=self.device)
            correspondence_accel_loss = torch.zeros((), device=self.device)
            correspondence_phase_loss = torch.zeros((), device=self.device)
            if (
                correspondence_schedule_scale > 0.0
                and (
                    cfg.correspondence_temporal_weight > 0.0
                    or cfg.correspondence_acceleration_weight > 0.0
                    or cfg.correspondence_phase_consistency_weight > 0.0
                )
            ):
                correspondence_count = min(cfg.temporal_batch_size, len(reference_points_np))
                correspondence_idx = rng.choice(len(reference_points_np), size=correspondence_count, replace=len(reference_points_np) < correspondence_count)
                correspondence_idx_t = torch.from_numpy(correspondence_idx.astype(np.int64)).to(self.device)
                ref_corr = reference_points[correspondence_idx_t]
                ref_mean_corr = mean_reference_offset[correspondence_idx_t]
                ref_basis_corr = basis_reference_offsets[:, correspondence_idx_t].permute(1, 0, 2)
                corr_prev = ref_corr + ref_mean_corr + torch.einsum("br,brj->bj", coeff_prev, ref_basis_corr)
                corr_now = ref_corr + ref_mean_corr + torch.einsum("br,brj->bj", coeff_now, ref_basis_corr)
                corr_next = ref_corr + ref_mean_corr + torch.einsum("br,brj->bj", coeff_next, ref_basis_corr)
                if not cfg.correspondence_global_only:
                    corr_prev = corr_prev + residual_field(ref_corr, prev_phase)
                    corr_now = corr_now + residual_field(ref_corr, center_phase)
                    corr_next = corr_next + residual_field(ref_corr, next_phase)
                corr_temporal_forward = (corr_next - corr_now).pow(2).mean(dim=-1)
                corr_temporal_backward = (corr_now - corr_prev).pow(2).mean(dim=-1)
                corr_accel = (corr_next - 2.0 * corr_now + corr_prev).pow(2).mean(dim=-1)
                corr_phase_gap = (corr_next - corr_prev).pow(2).mean(dim=-1)
                if cfg.correspondence_bootstrap_gate:
                    bootstrap_residual_prev = bootstrap_residual_reference_offsets[prev_idx_np, correspondence_idx_t]
                    bootstrap_residual_now = bootstrap_residual_reference_offsets[center_idx_np, correspondence_idx_t]
                    bootstrap_residual_next = bootstrap_residual_reference_offsets[next_idx_np, correspondence_idx_t]
                    corr_bootstrap_energy = (
                        bootstrap_residual_prev.pow(2).mean(dim=-1)
                        + bootstrap_residual_now.pow(2).mean(dim=-1)
                        + bootstrap_residual_next.pow(2).mean(dim=-1)
                    ) / 3.0
                    corr_bootstrap_scale = corr_bootstrap_energy.mean().clamp_min(1e-6)
                    corr_gate = 1.0 / (
                        1.0
                        + float(cfg.correspondence_bootstrap_gate_strength)
                        * (corr_bootstrap_energy / corr_bootstrap_scale)
                    )
                    corr_gate_sum = corr_gate.sum().clamp_min(1e-6)
                    correspondence_temporal_loss = 0.5 * (
                        (corr_gate * corr_temporal_forward).sum() / corr_gate_sum
                        + (corr_gate * corr_temporal_backward).sum() / corr_gate_sum
                    )
                    correspondence_accel_loss = (corr_gate * corr_accel).sum() / corr_gate_sum
                    correspondence_phase_loss = (corr_gate * corr_phase_gap).sum() / corr_gate_sum
                else:
                    correspondence_temporal_loss = 0.5 * (corr_temporal_forward.mean() + corr_temporal_backward.mean())
                    correspondence_accel_loss = corr_accel.mean()
                    correspondence_phase_loss = corr_phase_gap.mean()

            periodic_count = min(cfg.temporal_batch_size, len(reference_points_np))
            periodic_idx = rng.choice(len(reference_points_np), size=periodic_count, replace=len(reference_points_np) < periodic_count)
            periodic_idx_t = torch.from_numpy(periodic_idx.astype(np.int64)).to(self.device)
            ref_periodic = reference_points[periodic_idx_t]
            ref_mean_periodic = mean_reference_offset[periodic_idx_t]
            ref_basis_periodic = basis_reference_offsets[:, periodic_idx_t].permute(1, 0, 2)
            periodic_stride = max(1, int(round(float(cfg.temporal_delta_phase) * len(observations))))
            zero_phase = self._phase_tensor(observations[0].phase, len(ref_periodic), self.device)
            one_phase = self._phase_tensor(observations[-1].phase, len(ref_periodic), self.device)
            delta_phase = self._phase_tensor(observations[periodic_stride % len(observations)].phase, len(ref_periodic), self.device)
            one_minus_delta_phase = self._phase_tensor(observations[(-periodic_stride) % len(observations)].phase, len(ref_periodic), self.device)
            coeff_zero = coefficient_field(zero_phase)
            coeff_one = coefficient_field(one_phase)
            coeff_delta = coefficient_field(delta_phase)
            coeff_one_minus_delta = coefficient_field(one_minus_delta_phase)
            coefficient_periodicity_loss = ((coeff_zero - coeff_one) ** 2).mean()
            coefficient_periodicity_loss = coefficient_periodicity_loss + 0.5 * (((coeff_delta - coeff_zero) - (coeff_one - coeff_one_minus_delta)) ** 2).mean()
            deformation_zero = ref_mean_periodic + torch.einsum("br,brj->bj", coeff_zero, ref_basis_periodic) + residual_field(ref_periodic, zero_phase)
            deformation_one = ref_mean_periodic + torch.einsum("br,brj->bj", coeff_one, ref_basis_periodic) + residual_field(ref_periodic, one_phase)
            deformation_delta = ref_mean_periodic + torch.einsum("br,brj->bj", coeff_delta, ref_basis_periodic) + residual_field(ref_periodic, delta_phase)
            deformation_one_minus_delta = ref_mean_periodic + torch.einsum("br,brj->bj", coeff_one_minus_delta, ref_basis_periodic) + residual_field(ref_periodic, one_minus_delta_phase)
            periodicity_loss = ((deformation_zero - deformation_one) ** 2).mean()
            periodicity_loss = periodicity_loss + 0.5 * (((deformation_delta - deformation_zero) - (deformation_one - deformation_one_minus_delta)) ** 2).mean()

            loss = (
                surface_loss
                + cfg.normal_weight * normal_loss
                + cfg.centroid_weight * centroid_loss
                + cfg.spatial_smoothness_weight * spatial_loss
                + (bootstrap_schedule_scale * cfg.bootstrap_offset_weight) * bootstrap_offset_loss
                + (teacher_schedule_scale * cfg.bootstrap_teacher_weight) * bootstrap_teacher_loss
                + (bootstrap_schedule_scale * cfg.basis_coefficient_bootstrap_weight) * coefficient_bootstrap_loss
                + cfg.basis_temporal_weight * coefficient_temporal_loss
                + cfg.basis_acceleration_weight * coefficient_accel_loss
                + cfg.basis_periodicity_weight * coefficient_periodicity_loss
                + cfg.residual_mean_weight * residual_mean_loss
                + cfg.residual_basis_projection_weight * residual_basis_projection_loss
                + cfg.residual_locality_weight * residual_locality_loss
                + cfg.residual_wave_band_concentration_weight * residual_wave_band_concentration_loss
                + cfg.residual_wave_direction_weight * residual_wave_direction_loss
                + cfg.residual_global_ratio_weight * residual_global_ratio_loss
                + cfg.unsupported_anchor_weight * unsupported_anchor_loss
                + cfg.unsupported_laplacian_weight * unsupported_laplacian_loss
                + (cfg.temporal_weight * overlap_loss_scale) * temporal_loss
                + cfg.temporal_acceleration_weight * temporal_accel_loss
                + (cfg.phase_consistency_weight * overlap_loss_scale) * phase_consistency_loss
                + (correspondence_schedule_scale * cfg.correspondence_temporal_weight * overlap_loss_scale) * correspondence_temporal_loss
                + (correspondence_schedule_scale * cfg.correspondence_acceleration_weight) * correspondence_accel_loss
                + (correspondence_schedule_scale * cfg.correspondence_phase_consistency_weight * overlap_loss_scale) * correspondence_phase_loss
                + cfg.periodicity_weight * periodicity_loss
                + cfg.deformation_weight * deformation_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (step + 1) % max(1, cfg.train_steps // 4) == 0 or step == 0:
                print(
                    "[SharedDynamicBasis] "
                    f"step={step + 1}/{cfg.train_steps} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"surface={float(surface_loss.detach().cpu()):.6f} "
                    f"normal={float(normal_loss.detach().cpu()):.6f} "
                    f"centroid={float(centroid_loss.detach().cpu()):.6f} "
                    f"spatial={float(spatial_loss.detach().cpu()):.6f} "
                    f"boot={float(bootstrap_offset_loss.detach().cpu()):.6f} "
                    f"teacher={float(bootstrap_teacher_loss.detach().cpu()):.6f} "
                    f"coeff_boot={float(coefficient_bootstrap_loss.detach().cpu()):.6f} "
                    f"coeff_t={float(coefficient_temporal_loss.detach().cpu()):.6f} "
                    f"coeff_a={float(coefficient_accel_loss.detach().cpu()):.6f} "
                    f"coeff_p={float(coefficient_periodicity_loss.detach().cpu()):.6f} "
                    f"res_mean={float(residual_mean_loss.detach().cpu()):.6f} "
                    f"res_proj={float(residual_basis_projection_loss.detach().cpu()):.6f} "
                    f"res_locality={float(residual_locality_loss.detach().cpu()):.6f} "
                    f"res_band_std={float(residual_wave_band_std.detach().cpu()):.6f} "
                    f"res_band_loss={float(residual_wave_band_concentration_loss.detach().cpu()):.6f} "
                    f"res_dir={float(residual_wave_direction_loss.detach().cpu()):.6f} "
                    f"res_dir_axial={float(wave_direction_axial_ratio.detach().cpu()):.6f} "
                    f"res_dir_out={float(wave_direction_outward_ratio.detach().cpu()):.6f} "
                    f"res_dir_tan={float(wave_direction_tangential_ratio.detach().cpu()):.6f} "
                    f"wave_data_boost={float(wave_data_boost.mean().detach().cpu()):.6f} "
                    f"res_ratio={float(residual_global_ratio.detach().cpu()):.6f} "
                    f"res_ratio_loss={float(residual_global_ratio_loss.detach().cpu()):.6f} "
                    f"boot_scale={bootstrap_schedule_scale:.3f} "
                    f"anchor={float(unsupported_anchor_loss.detach().cpu()):.6f} "
                    f"lap={float(unsupported_laplacian_loss.detach().cpu()):.6f} "
                    f"temporal={float(temporal_loss.detach().cpu()):.6f} "
                    f"accel={float(temporal_accel_loss.detach().cpu()):.6f} "
                    f"phase={float(phase_consistency_loss.detach().cpu()):.6f} "
                    f"corr_scale={correspondence_schedule_scale:.3f} "
                    f"corr_temporal={float(correspondence_temporal_loss.detach().cpu()):.6f} "
                    f"corr_accel={float(correspondence_accel_loss.detach().cpu()):.6f} "
                    f"corr_phase={float(correspondence_phase_loss.detach().cpu()):.6f} "
                    f"periodic={float(periodicity_loss.detach().cpu()):.6f} "
                    f"deform={float(deformation_loss.detach().cpu()):.6f}"
                )

        with torch.no_grad():
            phase_displacements: list[np.ndarray] = []
            phase_coefficients: list[np.ndarray] = []
            phase_global_energy: list[float] = []
            phase_residual_energy: list[float] = []
            for observation in observations:
                phase_scalar = torch.tensor([observation.phase], dtype=torch.float32, device=self.device)
                global_coefficients = coefficient_field(phase_scalar).squeeze(0)
                global_offsets = mean_vertex_offset + torch.einsum("r,rvj->vj", global_coefficients, basis_vertex_offsets)
                phase_tensor = self._phase_tensor(observation.phase, len(base_vertices), self.device)
                residual_offsets = residual_field(base_vertices, phase_tensor)
                phase_coefficients.append(global_coefficients.detach().cpu().numpy().astype(np.float32))
                phase_global_energy.append(float(global_offsets.pow(2).mean().detach().cpu()))
                phase_residual_energy.append(float(residual_offsets.pow(2).mean().detach().cpu()))
                phase_displacements.append((global_offsets + residual_offsets).detach().cpu().numpy().astype(np.float32))

        run_dir = observations[0].pointcloud_path.parent
        base_mesh_path = run_dir / cfg.out_subdir / "dynamic_base_mesh.ply"
        base_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        denormalized_base_vertices = base_vertices_np * scale + center[None, :]
        base_mesh_export = trimesh.Trimesh(vertices=denormalized_base_vertices, faces=base_faces_np, process=False)
        _mesh_export(base_mesh_export, base_mesh_path)

        final_displacements = np.stack(phase_displacements, axis=0)
        final_displacements = self._propagate_unsupported_displacements(final_displacements, vertex_support_mask_np, edges_np)
        return SharedTopologyDynamicFit(
            base_vertices=base_vertices_np,
            base_faces=base_faces_np,
            displacements=final_displacements,
            phases=[item.phase for item in observations],
            center=center,
            scale=float(scale),
            global_coefficients=np.stack(phase_coefficients, axis=0),
            global_energy_per_phase=np.asarray(phase_global_energy, dtype=np.float32),
            residual_energy_per_phase=np.asarray(phase_residual_energy, dtype=np.float32),
        )

    def reconstruct(
        self,
        pointcloud_paths: list[Path],
        phase_confidences: Mapping[Path, float] | None = None,
        phase_summaries: Mapping[Path, PointCloudPhaseSummary] | None = None,
        timeline_samples: list[tuple[int, float, float]] | None = None,
    ) -> tuple[list[DynamicMeshBuildResult], list[DynamicTimelineMeshBuildResult]]:
        if not pointcloud_paths:
            return [], []

        fit = self.fit(
            pointcloud_paths,
            phase_confidences=phase_confidences,
            phase_summaries=phase_summaries,
        )
        run_dir = pointcloud_paths[0].parent
        mesh_dir = run_dir / self.config.out_subdir
        mesh_dir.mkdir(parents=True, exist_ok=True)
        results = self._export_phase_meshes(fit, mesh_dir)

        summary_path = mesh_dir / "dynamic_mesh_summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["phase", "mesh", "vertices", "faces", "watertight", "method"])
            for result in results:
                writer.writerow([
                    f"{result.phase:.6f}",
                    result.mesh_path.name,
                    result.vertices,
                    result.faces,
                    int(result.watertight),
                    result.method,
                ])

        self._write_basis_diagnostics(fit, mesh_dir)

        timeline_results: list[DynamicTimelineMeshBuildResult] = []
        if self.config.export_timeline_meshes and timeline_samples:
            timeline_dir = run_dir / self.config.timeline_out_subdir
            timeline_dir.mkdir(parents=True, exist_ok=True)
            timeline_results = self._export_timeline_meshes(fit, timeline_dir, timeline_samples)
            print(f"[SharedDynamicBasis] exported {len(timeline_results)} timeline meshes to {timeline_dir}")
        return results, timeline_results

    def _export_phase_meshes(self, fit: SharedTopologyDynamicFit, mesh_dir: Path) -> list[DynamicMeshBuildResult]:
        results: list[DynamicMeshBuildResult] = []
        for phase, displacement in zip(fit.phases, fit.displacements):
            mesh = self._mesh_from_offsets(fit, displacement)
            if len(mesh.faces) < self.config.min_face_count:
                print(f"[SharedDynamicBasis] phase={phase:.3f}: face count too low ({len(mesh.faces)}), skip")
                continue
            mesh_name = f"dynamic_phase_{phase:.3f}.ply".replace(" ", "")
            mesh_path = mesh_dir / mesh_name
            _mesh_export(mesh, mesh_path)
            print(
                f"[SharedDynamicBasis] wrote {mesh_path} "
                f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight})"
            )
            results.append(
                DynamicMeshBuildResult(
                    phase=float(phase),
                    mesh_path=mesh_path,
                    vertices=int(len(mesh.vertices)),
                    faces=int(len(mesh.faces)),
                    watertight=bool(mesh.is_watertight),
                    method="shared_topology_global_basis_residual",
                )
            )
        return results

    def _export_timeline_meshes(
        self,
        fit: SharedTopologyDynamicFit,
        mesh_dir: Path,
        timeline_samples: list[tuple[int, float, float]],
    ) -> list[DynamicTimelineMeshBuildResult]:
        selected_samples = self._select_timeline_samples(timeline_samples)
        results: list[DynamicTimelineMeshBuildResult] = []
        phases_np = np.asarray(fit.phases, dtype=np.float32)
        for frame_index, timestamp, phase in selected_samples:
            displacement = self._interpolate_offsets(phases_np, fit.displacements, phase)
            mesh = self._mesh_from_offsets(fit, displacement)
            mesh_name = f"dynamic_timeline_{frame_index:05d}_{phase:.3f}.ply".replace(" ", "")
            mesh_path = mesh_dir / mesh_name
            _mesh_export(mesh, mesh_path)
            results.append(
                DynamicTimelineMeshBuildResult(
                    frame_index=int(frame_index),
                    timestamp=float(timestamp),
                    phase=float(phase),
                    mesh_path=mesh_path,
                    vertices=int(len(mesh.vertices)),
                    faces=int(len(mesh.faces)),
                    watertight=bool(mesh.is_watertight),
                    method="shared_topology_global_basis_residual_timeline",
                )
            )

        summary_path = mesh_dir / "dynamic_timeline_mesh_summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_index", "timestamp", "phase", "mesh", "vertices", "faces", "watertight", "method"])
            for result in results:
                writer.writerow([
                    result.frame_index,
                    f"{result.timestamp:.6f}",
                    f"{result.phase:.6f}",
                    result.mesh_path.name,
                    result.vertices,
                    result.faces,
                    int(result.watertight),
                    result.method,
                ])
        return results


class SharedTopologyDecoupledMotionReconstructor(CanonicalPhaseDeformationFieldReconstructor):
    """Shared-topology base shape with a decoupled per-phase latent motion model."""

    @staticmethod
    def _phase_tensor(phase: float, count: int, device: torch.device) -> torch.Tensor:
        return torch.full((count,), float(phase), dtype=torch.float32, device=device)

    @staticmethod
    def _motion_lipschitz_loss(points: torch.Tensor, offsets: torch.Tensor, k: float) -> torch.Tensor:
        if len(points) < 2:
            return torch.zeros((), device=points.device)
        pair_count = min(len(points) // 2, 256)
        if pair_count < 1:
            return torch.zeros((), device=points.device)
        first = points[:pair_count]
        second = points[-pair_count:]
        offset_first = offsets[:pair_count]
        offset_second = offsets[-pair_count:]
        ratio = (offset_first - offset_second).norm(dim=-1) / ((first - second).norm(dim=-1) + 1e-3)
        return F.relu(ratio - float(k)).mean()

    def fit(
        self,
        pointcloud_paths: list[Path],
        phase_confidences: Mapping[Path, float] | None = None,
        phase_summaries: Mapping[Path, PointCloudPhaseSummary] | None = None,
    ) -> SharedTopologyDynamicFit:
        cfg = self.config
        observations, center, scale = self._prepare_phase_observations(
            pointcloud_paths,
            phase_confidences=phase_confidences,
            phase_summaries=phase_summaries,
        )
        base_phase_index = self._select_base_phase_index(observations)
        base_mesh = self._build_base_mesh(observations[base_phase_index].pointcloud_path)
        base_vertices_np, base_faces_np = self._normalize_base_mesh(base_mesh, center, scale)

        sample_count = max(int(cfg.surface_batch_size * 2), 4096)
        face_indices_np, barycentric_np = self._sample_surface_plan(base_mesh, sample_count, cfg.random_seed)
        reference_points_np = self._sample_reference_points_numpy(base_vertices_np, base_faces_np, face_indices_np, barycentric_np)
        support_radius = self._support_radius_normalized(scale)
        phase_support_weights_np = self._compute_phase_support_weights(reference_points_np, observations, support_radius)
        vertex_support_mask_np = self._compute_phase_support_mask(base_vertices_np, observations, support_radius)
        initial_offsets_np = self._initialize_offsets(base_vertices_np, observations, vertex_support_mask_np)
        bootstrap_reference_offsets_np = self._sample_reference_offsets_numpy(initial_offsets_np, base_faces_np, face_indices_np, barycentric_np)
        edges_np = self._build_edges(base_faces_np)

        rng = np.random.default_rng(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_seed)

        base_vertices = torch.from_numpy(base_vertices_np).to(self.device)
        base_faces = torch.from_numpy(base_faces_np).to(self.device)
        face_indices = torch.from_numpy(face_indices_np).to(self.device)
        barycentric = torch.from_numpy(barycentric_np).to(self.device)
        reference_points = torch.from_numpy(reference_points_np).to(self.device)
        phase_support_weights = torch.from_numpy(phase_support_weights_np).to(self.device)
        vertex_support_mask = torch.from_numpy(vertex_support_mask_np).to(self.device)
        bootstrap_offsets = torch.from_numpy(initial_offsets_np).to(self.device)
        bootstrap_reference_offsets = torch.from_numpy(bootstrap_reference_offsets_np).to(self.device)
        edges = torch.from_numpy(edges_np).to(self.device)
        phase_points = [torch.from_numpy(item.points).to(self.device) for item in observations]
        phase_normals = [torch.from_numpy(item.normals).to(self.device) for item in observations]
        phase_centroids = [torch.from_numpy(item.centroid).to(self.device) for item in observations]
        phase_weights = np.asarray([item.weight for item in observations], dtype=np.float64)
        sample_probabilities = phase_weights / max(np.sum(phase_weights), 1e-8)
        phase_weight_tensor = torch.tensor(phase_weights, dtype=torch.float32, device=self.device)

        motion_codes = torch.nn.Embedding(len(observations), cfg.motion_latent_dim).to(self.device)
        torch.nn.init.normal_(motion_codes.weight.data, 0.0, 1.0 / np.sqrt(max(cfg.motion_latent_dim, 1)))
        motion_field = DecoupledMotionLatentField(cfg.motion_latent_dim, hidden_dim=max(16, cfg.deformation_hidden_dim // 8)).to(self.device)
        optimizer = torch.optim.Adam([
            {"params": motion_field.parameters(), "lr": cfg.learning_rate},
            {"params": motion_codes.parameters(), "lr": cfg.learning_rate},
        ])
        overlap_loss_scale = self._overlap_loss_scale()

        for step in range(cfg.train_steps):
            correspondence_schedule_scale = self._correspondence_schedule_scale(step)
            bootstrap_schedule_scale = self._bootstrap_schedule_scale(step)
            phase_id = int(rng.choice(len(observations), p=sample_probabilities))
            phase_value = observations[phase_id].phase
            target_points = phase_points[phase_id]
            target_normals = phase_normals[phase_id]
            target_centroid = phase_centroids[phase_id]
            phase_weight = phase_weight_tensor[phase_id]
            support_weights = phase_support_weights[phase_id]
            support_sum = support_weights.sum().clamp_min(1e-6)
            unsupported_vertex_weights = 1.0 - vertex_support_mask[phase_id]
            unsupported_vertex_sum = unsupported_vertex_weights.sum().clamp_min(1e-6)

            phase_tensor_vertices = self._phase_tensor(phase_value, len(base_vertices), self.device)
            latent_vertices = motion_codes.weight[phase_id].unsqueeze(0).expand(len(base_vertices), -1)
            residual_vertex_offsets = motion_field(base_vertices, phase_tensor_vertices, latent_vertices)
            bootstrap_vertex_offsets = bootstrap_offsets[phase_id]
            vertex_offsets = bootstrap_vertex_offsets + residual_vertex_offsets
            vertices = base_vertices + vertex_offsets
            pred_samples, pred_normals = self._sample_surface_from_vertices(vertices, base_faces, face_indices, barycentric)

            distances = torch.cdist(pred_samples, target_points)
            pred_to_target, pred_nn = distances.min(dim=1)
            target_to_pred, _ = distances.min(dim=0)
            surface_loss = phase_weight * (((support_weights * pred_to_target.pow(2)).sum() / support_sum) + target_to_pred.pow(2).mean())
            nearest_target_normals = target_normals[pred_nn]
            normal_alignment = 1.0 - torch.abs(torch.sum(pred_normals * nearest_target_normals, dim=-1))
            normal_loss = phase_weight * ((support_weights * normal_alignment).sum() / support_sum)
            supported_pred_centroid = (pred_samples * support_weights.unsqueeze(-1)).sum(dim=0) / support_sum
            centroid_loss = phase_weight * ((supported_pred_centroid - target_centroid) ** 2).mean()
            spatial_loss = ((residual_vertex_offsets[edges[:, 0]] - residual_vertex_offsets[edges[:, 1]]) ** 2).mean()
            bootstrap_offset_loss = (((vertex_support_mask[phase_id] * (vertex_offsets - bootstrap_vertex_offsets).pow(2).mean(dim=-1)).sum()) / vertex_support_mask[phase_id].sum().clamp_min(1e-6))
            unsupported_anchor_loss = ((unsupported_vertex_weights * vertex_offsets.pow(2).mean(dim=-1)).sum() / unsupported_vertex_sum)
            unsupported_neighbor_mean = self._neighbor_mean_torch(vertex_offsets, edges)
            unsupported_laplacian_loss = ((unsupported_vertex_weights * (vertex_offsets - unsupported_neighbor_mean).pow(2).mean(dim=-1)).sum() / unsupported_vertex_sum)
            lipschitz_loss = self._motion_lipschitz_loss(base_vertices, residual_vertex_offsets, cfg.motion_lipschitz_k)

            temporal_count = min(cfg.temporal_batch_size, len(reference_points_np))
            temporal_idx = rng.choice(len(reference_points_np), size=temporal_count, replace=len(reference_points_np) < temporal_count)
            temporal_idx_t = torch.from_numpy(temporal_idx.astype(np.int64)).to(self.device)
            ref_temporal = reference_points[temporal_idx_t]
            prev_idx_np, center_idx_np, next_idx_np = self._sample_phase_triplet_indices(len(observations), temporal_count, rng)
            prev_idx = torch.from_numpy(prev_idx_np).to(self.device)
            center_idx = torch.from_numpy(center_idx_np).to(self.device)
            next_idx = torch.from_numpy(next_idx_np).to(self.device)
            prev_phase = torch.tensor([observations[index].phase for index in prev_idx_np], dtype=torch.float32, device=self.device)
            center_phase = torch.tensor([observations[index].phase for index in center_idx_np], dtype=torch.float32, device=self.device)
            next_phase = torch.tensor([observations[index].phase for index in next_idx_np], dtype=torch.float32, device=self.device)
            prev_codes = motion_codes(prev_idx)
            center_codes = motion_codes(center_idx)
            next_codes = motion_codes(next_idx)
            residual_prev = motion_field(ref_temporal, prev_phase, prev_codes)
            residual_now = motion_field(ref_temporal, center_phase, center_codes)
            residual_next = motion_field(ref_temporal, next_phase, next_codes)
            deformation_prev = bootstrap_reference_offsets[prev_idx, temporal_idx_t] + residual_prev
            deformation_now = bootstrap_reference_offsets[center_idx, temporal_idx_t] + residual_now
            deformation_next = bootstrap_reference_offsets[next_idx, temporal_idx_t] + residual_next
            temporal_loss = 0.5 * (((deformation_next - deformation_now) ** 2).mean() + ((deformation_now - deformation_prev) ** 2).mean())
            temporal_accel_loss = ((deformation_next - 2.0 * deformation_now + deformation_prev) ** 2).mean()
            phase_consistency_loss = ((deformation_next - deformation_prev) ** 2).mean()
            deformation_loss = (residual_now ** 2).mean()
            motion_code_loss = motion_codes.weight.pow(2).mean()

            correspondence_temporal_loss = torch.zeros((), device=self.device)
            correspondence_accel_loss = torch.zeros((), device=self.device)
            correspondence_phase_loss = torch.zeros((), device=self.device)
            if correspondence_schedule_scale > 0.0 and (
                cfg.correspondence_temporal_weight > 0.0
                or cfg.correspondence_acceleration_weight > 0.0
                or cfg.correspondence_phase_consistency_weight > 0.0
            ):
                correspondence_count = min(cfg.temporal_batch_size, len(reference_points_np))
                correspondence_idx = rng.choice(len(reference_points_np), size=correspondence_count, replace=len(reference_points_np) < correspondence_count)
                correspondence_idx_t = torch.from_numpy(correspondence_idx.astype(np.int64)).to(self.device)
                ref_corr = reference_points[correspondence_idx_t]
                corr_prev = ref_corr + bootstrap_reference_offsets[prev_idx, correspondence_idx_t] + motion_field(ref_corr, prev_phase, prev_codes)
                corr_now = ref_corr + bootstrap_reference_offsets[center_idx, correspondence_idx_t] + motion_field(ref_corr, center_phase, center_codes)
                corr_next = ref_corr + bootstrap_reference_offsets[next_idx, correspondence_idx_t] + motion_field(ref_corr, next_phase, next_codes)
                correspondence_temporal_loss = 0.5 * (((corr_next - corr_now) ** 2).mean() + ((corr_now - corr_prev) ** 2).mean())
                correspondence_accel_loss = ((corr_next - 2.0 * corr_now + corr_prev) ** 2).mean()
                correspondence_phase_loss = ((corr_next - corr_prev) ** 2).mean()

            periodic_count = min(cfg.temporal_batch_size, len(reference_points_np))
            periodic_idx = rng.choice(len(reference_points_np), size=periodic_count, replace=len(reference_points_np) < periodic_count)
            periodic_idx_t = torch.from_numpy(periodic_idx.astype(np.int64)).to(self.device)
            ref_periodic = reference_points[periodic_idx_t]
            periodic_stride = max(1, int(round(float(cfg.temporal_delta_phase) * len(observations))))
            zero_phase = self._phase_tensor(observations[0].phase, len(ref_periodic), self.device)
            one_phase = self._phase_tensor(observations[-1].phase, len(ref_periodic), self.device)
            delta_phase = self._phase_tensor(observations[periodic_stride % len(observations)].phase, len(ref_periodic), self.device)
            one_minus_delta_phase = self._phase_tensor(observations[(-periodic_stride) % len(observations)].phase, len(ref_periodic), self.device)
            zero_codes = motion_codes.weight[0].unsqueeze(0).expand(len(ref_periodic), -1)
            one_codes = motion_codes.weight[-1].unsqueeze(0).expand(len(ref_periodic), -1)
            delta_codes = motion_codes.weight[periodic_stride % len(observations)].unsqueeze(0).expand(len(ref_periodic), -1)
            one_minus_delta_codes = motion_codes.weight[(-periodic_stride) % len(observations)].unsqueeze(0).expand(len(ref_periodic), -1)
            deformation_zero = bootstrap_reference_offsets[0, periodic_idx_t] + motion_field(ref_periodic, zero_phase, zero_codes)
            deformation_one = bootstrap_reference_offsets[-1, periodic_idx_t] + motion_field(ref_periodic, one_phase, one_codes)
            deformation_delta = bootstrap_reference_offsets[periodic_stride % len(observations), periodic_idx_t] + motion_field(ref_periodic, delta_phase, delta_codes)
            deformation_one_minus_delta = bootstrap_reference_offsets[(-periodic_stride) % len(observations), periodic_idx_t] + motion_field(ref_periodic, one_minus_delta_phase, one_minus_delta_codes)
            periodicity_loss = ((deformation_zero - deformation_one) ** 2).mean()
            periodicity_loss = periodicity_loss + 0.5 * (((deformation_delta - deformation_zero) - (deformation_one - deformation_one_minus_delta)) ** 2).mean()

            loss = (
                surface_loss
                + cfg.normal_weight * normal_loss
                + cfg.centroid_weight * centroid_loss
                + cfg.spatial_smoothness_weight * spatial_loss
                + (bootstrap_schedule_scale * cfg.bootstrap_offset_weight) * bootstrap_offset_loss
                + cfg.unsupported_anchor_weight * unsupported_anchor_loss
                + cfg.unsupported_laplacian_weight * unsupported_laplacian_loss
                + cfg.motion_lipschitz_weight * lipschitz_loss
                + cfg.motion_code_reg_weight * motion_code_loss
                + (cfg.temporal_weight * overlap_loss_scale) * temporal_loss
                + cfg.temporal_acceleration_weight * temporal_accel_loss
                + (cfg.phase_consistency_weight * overlap_loss_scale) * phase_consistency_loss
                + (correspondence_schedule_scale * cfg.correspondence_temporal_weight * overlap_loss_scale) * correspondence_temporal_loss
                + (correspondence_schedule_scale * cfg.correspondence_acceleration_weight) * correspondence_accel_loss
                + (correspondence_schedule_scale * cfg.correspondence_phase_consistency_weight * overlap_loss_scale) * correspondence_phase_loss
                + cfg.periodicity_weight * periodicity_loss
                + cfg.deformation_weight * deformation_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (step + 1) % max(1, cfg.train_steps // 4) == 0 or step == 0:
                print(
                    "[SharedDynamicLatent] "
                    f"step={step + 1}/{cfg.train_steps} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"surface={float(surface_loss.detach().cpu()):.6f} "
                    f"normal={float(normal_loss.detach().cpu()):.6f} "
                    f"centroid={float(centroid_loss.detach().cpu()):.6f} "
                    f"spatial={float(spatial_loss.detach().cpu()):.6f} "
                    f"boot={float(bootstrap_offset_loss.detach().cpu()):.6f} "
                    f"lip={float(lipschitz_loss.detach().cpu()):.6f} "
                    f"code={float(motion_code_loss.detach().cpu()):.6f} "
                    f"temporal={float(temporal_loss.detach().cpu()):.6f} "
                    f"accel={float(temporal_accel_loss.detach().cpu()):.6f} "
                    f"phase={float(phase_consistency_loss.detach().cpu()):.6f} "
                    f"corr_scale={correspondence_schedule_scale:.3f}"
                )

        with torch.no_grad():
            phase_displacements: list[np.ndarray] = []
            for phase_index, observation in enumerate(observations):
                phase_tensor = self._phase_tensor(observation.phase, len(base_vertices), self.device)
                phase_codes = motion_codes.weight[phase_index].unsqueeze(0).expand(len(base_vertices), -1)
                residual = motion_field(base_vertices, phase_tensor, phase_codes).detach().cpu().numpy().astype(np.float32)
                displacement = initial_offsets_np[phase_index] + residual
                phase_displacements.append(displacement)

        run_dir = observations[0].pointcloud_path.parent
        base_mesh_path = run_dir / cfg.out_subdir / "dynamic_base_mesh.ply"
        base_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        denormalized_base_vertices = base_vertices_np * scale + center[None, :]
        base_mesh_export = trimesh.Trimesh(vertices=denormalized_base_vertices, faces=base_faces_np, process=False)
        _mesh_export(base_mesh_export, base_mesh_path)

        return SharedTopologyDynamicFit(
            base_vertices=base_vertices_np,
            base_faces=base_faces_np,
            displacements=np.stack(phase_displacements, axis=0),
            phases=[item.phase for item in observations],
            center=center,
            scale=float(scale),
        )

    def _export_phase_meshes(self, fit: SharedTopologyDynamicFit, mesh_dir: Path) -> list[DynamicMeshBuildResult]:
        results: list[DynamicMeshBuildResult] = []
        for phase, displacement in zip(fit.phases, fit.displacements):
            mesh = self._mesh_from_offsets(fit, displacement)
            if len(mesh.faces) < self.config.min_face_count:
                print(f"[SharedDynamicLatent] phase={phase:.3f}: face count too low ({len(mesh.faces)}), skip")
                continue
            mesh_name = f"dynamic_phase_{phase:.3f}.ply".replace(" ", "")
            mesh_path = mesh_dir / mesh_name
            _mesh_export(mesh, mesh_path)
            results.append(
                DynamicMeshBuildResult(
                    phase=float(phase),
                    mesh_path=mesh_path,
                    vertices=int(len(mesh.vertices)),
                    faces=int(len(mesh.faces)),
                    watertight=bool(mesh.is_watertight),
                    method="shared_topology_decoupled_motion_latent",
                )
            )
        return results


class SharedTopologyDecoupledShapeMotionReconstructor(CanonicalPhaseDeformationFieldReconstructor):
    """Shared-topology reconstructor with a global shape latent and per-phase motion latents."""

    @staticmethod
    def _phase_tensor(phase: float, count: int, device: torch.device) -> torch.Tensor:
        return torch.full((count,), float(phase), dtype=torch.float32, device=device)

    @staticmethod
    def _motion_lipschitz_loss(points: torch.Tensor, offsets: torch.Tensor, k: float) -> torch.Tensor:
        if len(points) < 2:
            return torch.zeros((), device=points.device)
        pair_count = min(len(points) // 2, 256)
        if pair_count < 1:
            return torch.zeros((), device=points.device)
        first = points[:pair_count]
        second = points[-pair_count:]
        offset_first = offsets[:pair_count]
        offset_second = offsets[-pair_count:]
        ratio = (offset_first - offset_second).norm(dim=-1) / ((first - second).norm(dim=-1) + 1e-3)
        return F.relu(ratio - float(k)).mean()

    def fit(
        self,
        pointcloud_paths: list[Path],
        phase_confidences: Mapping[Path, float] | None = None,
        phase_summaries: Mapping[Path, PointCloudPhaseSummary] | None = None,
    ) -> SharedTopologyDynamicFit:
        cfg = self.config
        observations, center, scale = self._prepare_phase_observations(
            pointcloud_paths,
            phase_confidences=phase_confidences,
            phase_summaries=phase_summaries,
        )
        base_phase_index = self._select_base_phase_index(observations)
        base_mesh = self._build_base_mesh(observations[base_phase_index].pointcloud_path)
        base_vertices_np, base_faces_np = self._normalize_base_mesh(base_mesh, center, scale)

        sample_count = max(int(cfg.surface_batch_size * 2), 4096)
        face_indices_np, barycentric_np = self._sample_surface_plan(base_mesh, sample_count, cfg.random_seed)
        reference_points_np = self._sample_reference_points_numpy(base_vertices_np, base_faces_np, face_indices_np, barycentric_np)
        support_radius = self._support_radius_normalized(scale)
        phase_support_weights_np = self._compute_phase_support_weights(reference_points_np, observations, support_radius)
        vertex_support_mask_np = self._compute_phase_support_mask(base_vertices_np, observations, support_radius)
        initial_offsets_np = self._initialize_offsets(base_vertices_np, observations, vertex_support_mask_np)
        bootstrap_reference_offsets_np = self._sample_reference_offsets_numpy(initial_offsets_np, base_faces_np, face_indices_np, barycentric_np)
        edges_np = self._build_edges(base_faces_np)

        rng = np.random.default_rng(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_seed)

        base_vertices = torch.from_numpy(base_vertices_np).to(self.device)
        base_faces = torch.from_numpy(base_faces_np).to(self.device)
        face_indices = torch.from_numpy(face_indices_np).to(self.device)
        barycentric = torch.from_numpy(barycentric_np).to(self.device)
        reference_points = torch.from_numpy(reference_points_np).to(self.device)
        phase_support_weights = torch.from_numpy(phase_support_weights_np).to(self.device)
        vertex_support_mask = torch.from_numpy(vertex_support_mask_np).to(self.device)
        bootstrap_offsets = torch.from_numpy(initial_offsets_np).to(self.device)
        bootstrap_reference_offsets = torch.from_numpy(bootstrap_reference_offsets_np).to(self.device)
        edges = torch.from_numpy(edges_np).to(self.device)
        phase_points = [torch.from_numpy(item.points).to(self.device) for item in observations]
        phase_normals = [torch.from_numpy(item.normals).to(self.device) for item in observations]
        phase_centroids = [torch.from_numpy(item.centroid).to(self.device) for item in observations]
        phase_weights = np.asarray([item.weight for item in observations], dtype=np.float64)
        sample_probabilities = phase_weights / max(np.sum(phase_weights), 1e-8)
        phase_weight_tensor = torch.tensor(phase_weights, dtype=torch.float32, device=self.device)

        shape_code = torch.nn.Parameter(torch.zeros(cfg.shape_latent_dim, dtype=torch.float32, device=self.device))
        motion_codes = torch.nn.Embedding(len(observations), cfg.motion_latent_dim).to(self.device)
        torch.nn.init.normal_(motion_codes.weight.data, 0.0, 1.0 / np.sqrt(max(cfg.motion_latent_dim, 1)))
        shape_field = ShapeLatentField(cfg.shape_latent_dim, hidden_dim=max(16, cfg.canonical_hidden_dim // 8)).to(self.device)
        motion_field = DecoupledMotionLatentField(cfg.motion_latent_dim, hidden_dim=max(16, cfg.deformation_hidden_dim // 8)).to(self.device)
        optimizer = torch.optim.Adam([
            {"params": shape_field.parameters(), "lr": cfg.learning_rate},
            {"params": motion_field.parameters(), "lr": cfg.learning_rate},
            {"params": motion_codes.parameters(), "lr": cfg.learning_rate},
            {"params": [shape_code], "lr": cfg.learning_rate},
        ])
        overlap_loss_scale = self._overlap_loss_scale()

        for step in range(cfg.train_steps):
            correspondence_schedule_scale = self._correspondence_schedule_scale(step)
            bootstrap_schedule_scale = self._bootstrap_schedule_scale(step)
            phase_id = int(rng.choice(len(observations), p=sample_probabilities))
            phase_value = observations[phase_id].phase
            target_points = phase_points[phase_id]
            target_normals = phase_normals[phase_id]
            target_centroid = phase_centroids[phase_id]
            phase_weight = phase_weight_tensor[phase_id]
            support_weights = phase_support_weights[phase_id]
            support_sum = support_weights.sum().clamp_min(1e-6)
            supported_vertex_weights = vertex_support_mask[phase_id]
            supported_vertex_sum = supported_vertex_weights.sum().clamp_min(1e-6)
            unsupported_vertex_weights = 1.0 - supported_vertex_weights
            unsupported_vertex_sum = unsupported_vertex_weights.sum().clamp_min(1e-6)

            shape_code_vertices = shape_code.unsqueeze(0).expand(len(base_vertices), -1)
            shape_vertex_offsets = shape_field(base_vertices, shape_code_vertices)
            canonical_vertices = base_vertices + shape_vertex_offsets
            phase_tensor_vertices = self._phase_tensor(phase_value, len(canonical_vertices), self.device)
            motion_latent_vertices = motion_codes.weight[phase_id].unsqueeze(0).expand(len(canonical_vertices), -1)
            motion_vertex_offsets = motion_field(canonical_vertices, phase_tensor_vertices, motion_latent_vertices)
            predicted_vertices = canonical_vertices + motion_vertex_offsets
            pred_samples, pred_normals = self._sample_surface_from_vertices(predicted_vertices, base_faces, face_indices, barycentric)

            distances = torch.cdist(pred_samples, target_points)
            pred_to_target, pred_nn = distances.min(dim=1)
            target_to_pred, _ = distances.min(dim=0)
            surface_loss = phase_weight * (((support_weights * pred_to_target.pow(2)).sum() / support_sum) + target_to_pred.pow(2).mean())
            nearest_target_normals = target_normals[pred_nn]
            normal_alignment = 1.0 - torch.abs(torch.sum(pred_normals * nearest_target_normals, dim=-1))
            normal_loss = phase_weight * ((support_weights * normal_alignment).sum() / support_sum)
            supported_pred_centroid = (pred_samples * support_weights.unsqueeze(-1)).sum(dim=0) / support_sum
            centroid_loss = phase_weight * ((supported_pred_centroid - target_centroid) ** 2).mean()

            shape_spatial_loss = ((shape_vertex_offsets[edges[:, 0]] - shape_vertex_offsets[edges[:, 1]]) ** 2).mean()
            shape_offset_loss = shape_vertex_offsets.pow(2).mean()
            motion_spatial_loss = ((motion_vertex_offsets[edges[:, 0]] - motion_vertex_offsets[edges[:, 1]]) ** 2).mean()
            bootstrap_target_vertices = base_vertices + bootstrap_offsets[phase_id]
            bootstrap_position_loss = (
                (supported_vertex_weights * (predicted_vertices - bootstrap_target_vertices).pow(2).mean(dim=-1)).sum()
                / supported_vertex_sum
            )
            unsupported_anchor_loss = (
                (unsupported_vertex_weights * motion_vertex_offsets.pow(2).mean(dim=-1)).sum()
                / unsupported_vertex_sum
            )
            unsupported_neighbor_mean = self._neighbor_mean_torch(motion_vertex_offsets, edges)
            unsupported_laplacian_loss = (
                (unsupported_vertex_weights * (motion_vertex_offsets - unsupported_neighbor_mean).pow(2).mean(dim=-1)).sum()
                / unsupported_vertex_sum
            )
            lipschitz_loss = self._motion_lipschitz_loss(canonical_vertices, motion_vertex_offsets, cfg.motion_lipschitz_k)

            shape_code_reference = shape_code.unsqueeze(0).expand(len(reference_points), -1)
            shape_reference_offsets = shape_field(reference_points, shape_code_reference)
            canonical_reference_points = reference_points + shape_reference_offsets

            temporal_count = min(cfg.temporal_batch_size, len(reference_points_np))
            temporal_idx = rng.choice(len(reference_points_np), size=temporal_count, replace=len(reference_points_np) < temporal_count)
            temporal_idx_t = torch.from_numpy(temporal_idx.astype(np.int64)).to(self.device)
            ref_temporal = canonical_reference_points[temporal_idx_t]
            prev_idx_np, center_idx_np, next_idx_np = self._sample_phase_triplet_indices(len(observations), temporal_count, rng)
            prev_idx = torch.from_numpy(prev_idx_np).to(self.device)
            center_idx = torch.from_numpy(center_idx_np).to(self.device)
            next_idx = torch.from_numpy(next_idx_np).to(self.device)
            prev_phase = torch.tensor([observations[index].phase for index in prev_idx_np], dtype=torch.float32, device=self.device)
            center_phase = torch.tensor([observations[index].phase for index in center_idx_np], dtype=torch.float32, device=self.device)
            next_phase = torch.tensor([observations[index].phase for index in next_idx_np], dtype=torch.float32, device=self.device)
            prev_codes = motion_codes(prev_idx)
            center_codes = motion_codes(center_idx)
            next_codes = motion_codes(next_idx)
            motion_prev = motion_field(ref_temporal, prev_phase, prev_codes)
            motion_now = motion_field(ref_temporal, center_phase, center_codes)
            motion_next = motion_field(ref_temporal, next_phase, next_codes)
            deformation_prev = ref_temporal + motion_prev
            deformation_now = ref_temporal + motion_now
            deformation_next = ref_temporal + motion_next
            temporal_loss = 0.5 * (((deformation_next - deformation_now) ** 2).mean() + ((deformation_now - deformation_prev) ** 2).mean())
            temporal_accel_loss = ((deformation_next - 2.0 * deformation_now + deformation_prev) ** 2).mean()
            phase_consistency_loss = ((deformation_next - deformation_prev) ** 2).mean()
            deformation_loss = motion_now.pow(2).mean()
            motion_mean_loss = ((motion_prev + motion_now + motion_next) / 3.0).pow(2).mean()
            shape_code_loss = shape_code.pow(2).mean()
            motion_code_loss = motion_codes.weight.pow(2).mean()

            correspondence_temporal_loss = torch.zeros((), device=self.device)
            correspondence_accel_loss = torch.zeros((), device=self.device)
            correspondence_phase_loss = torch.zeros((), device=self.device)
            if correspondence_schedule_scale > 0.0 and (
                cfg.correspondence_temporal_weight > 0.0
                or cfg.correspondence_acceleration_weight > 0.0
                or cfg.correspondence_phase_consistency_weight > 0.0
            ):
                correspondence_count = min(cfg.temporal_batch_size, len(reference_points_np))
                correspondence_idx = rng.choice(len(reference_points_np), size=correspondence_count, replace=len(reference_points_np) < correspondence_count)
                correspondence_idx_t = torch.from_numpy(correspondence_idx.astype(np.int64)).to(self.device)
                ref_corr = canonical_reference_points[correspondence_idx_t]
                corr_prev = ref_corr + motion_field(ref_corr, prev_phase, prev_codes)
                corr_now = ref_corr + motion_field(ref_corr, center_phase, center_codes)
                corr_next = ref_corr + motion_field(ref_corr, next_phase, next_codes)
                correspondence_temporal_loss = 0.5 * (((corr_next - corr_now) ** 2).mean() + ((corr_now - corr_prev) ** 2).mean())
                correspondence_accel_loss = ((corr_next - 2.0 * corr_now + corr_prev) ** 2).mean()
                correspondence_phase_loss = ((corr_next - corr_prev) ** 2).mean()

            periodic_count = min(cfg.temporal_batch_size, len(reference_points_np))
            periodic_idx = rng.choice(len(reference_points_np), size=periodic_count, replace=len(reference_points_np) < periodic_count)
            periodic_idx_t = torch.from_numpy(periodic_idx.astype(np.int64)).to(self.device)
            ref_periodic = canonical_reference_points[periodic_idx_t]
            periodic_stride = max(1, int(round(float(cfg.temporal_delta_phase) * len(observations))))
            zero_phase = self._phase_tensor(observations[0].phase, len(ref_periodic), self.device)
            one_phase = self._phase_tensor(observations[-1].phase, len(ref_periodic), self.device)
            delta_phase = self._phase_tensor(observations[periodic_stride % len(observations)].phase, len(ref_periodic), self.device)
            one_minus_delta_phase = self._phase_tensor(observations[(-periodic_stride) % len(observations)].phase, len(ref_periodic), self.device)
            zero_codes = motion_codes.weight[0].unsqueeze(0).expand(len(ref_periodic), -1)
            one_codes = motion_codes.weight[-1].unsqueeze(0).expand(len(ref_periodic), -1)
            delta_codes = motion_codes.weight[periodic_stride % len(observations)].unsqueeze(0).expand(len(ref_periodic), -1)
            one_minus_delta_codes = motion_codes.weight[(-periodic_stride) % len(observations)].unsqueeze(0).expand(len(ref_periodic), -1)
            deformation_zero = ref_periodic + motion_field(ref_periodic, zero_phase, zero_codes)
            deformation_one = ref_periodic + motion_field(ref_periodic, one_phase, one_codes)
            deformation_delta = ref_periodic + motion_field(ref_periodic, delta_phase, delta_codes)
            deformation_one_minus_delta = ref_periodic + motion_field(ref_periodic, one_minus_delta_phase, one_minus_delta_codes)
            periodicity_loss = ((deformation_zero - deformation_one) ** 2).mean()
            periodicity_loss = periodicity_loss + 0.5 * (((deformation_delta - deformation_zero) - (deformation_one - deformation_one_minus_delta)) ** 2).mean()

            loss = (
                surface_loss
                + cfg.normal_weight * normal_loss
                + cfg.centroid_weight * centroid_loss
                + cfg.shape_spatial_weight * shape_spatial_loss
                + cfg.shape_offset_reg_weight * shape_offset_loss
                + cfg.spatial_smoothness_weight * motion_spatial_loss
                + (bootstrap_schedule_scale * cfg.bootstrap_offset_weight) * bootstrap_position_loss
                + cfg.unsupported_anchor_weight * unsupported_anchor_loss
                + cfg.unsupported_laplacian_weight * unsupported_laplacian_loss
                + cfg.motion_lipschitz_weight * lipschitz_loss
                + cfg.motion_mean_weight * motion_mean_loss
                + cfg.shape_code_reg_weight * shape_code_loss
                + cfg.motion_code_reg_weight * motion_code_loss
                + (cfg.temporal_weight * overlap_loss_scale) * temporal_loss
                + cfg.temporal_acceleration_weight * temporal_accel_loss
                + (cfg.phase_consistency_weight * overlap_loss_scale) * phase_consistency_loss
                + (correspondence_schedule_scale * cfg.correspondence_temporal_weight * overlap_loss_scale) * correspondence_temporal_loss
                + (correspondence_schedule_scale * cfg.correspondence_acceleration_weight) * correspondence_accel_loss
                + (correspondence_schedule_scale * cfg.correspondence_phase_consistency_weight * overlap_loss_scale) * correspondence_phase_loss
                + cfg.periodicity_weight * periodicity_loss
                + cfg.deformation_weight * deformation_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (step + 1) % max(1, cfg.train_steps // 4) == 0 or step == 0:
                print(
                    "[SharedDynamicShapeMotion] "
                    f"step={step + 1}/{cfg.train_steps} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"surface={float(surface_loss.detach().cpu()):.6f} "
                    f"normal={float(normal_loss.detach().cpu()):.6f} "
                    f"centroid={float(centroid_loss.detach().cpu()):.6f} "
                    f"shape_spatial={float(shape_spatial_loss.detach().cpu()):.6f} "
                    f"shape_off={float(shape_offset_loss.detach().cpu()):.6f} "
                    f"motion_spatial={float(motion_spatial_loss.detach().cpu()):.6f} "
                    f"boot={float(bootstrap_position_loss.detach().cpu()):.6f} "
                    f"lip={float(lipschitz_loss.detach().cpu()):.6f} "
                    f"motion_mean={float(motion_mean_loss.detach().cpu()):.6f} "
                    f"shape_code={float(shape_code_loss.detach().cpu()):.6f} "
                    f"motion_code={float(motion_code_loss.detach().cpu()):.6f} "
                    f"corr_scale={correspondence_schedule_scale:.3f}"
                )

        with torch.no_grad():
            shape_code_vertices = shape_code.unsqueeze(0).expand(len(base_vertices), -1)
            canonical_vertex_offsets = shape_field(base_vertices, shape_code_vertices).detach().cpu().numpy().astype(np.float32)
            canonical_vertices_np = base_vertices_np + canonical_vertex_offsets
            phase_displacements: list[np.ndarray] = []
            canonical_vertices_t = base_vertices + shape_field(base_vertices, shape_code.unsqueeze(0).expand(len(base_vertices), -1))
            for phase_index, observation in enumerate(observations):
                phase_tensor = self._phase_tensor(observation.phase, len(base_vertices), self.device)
                phase_codes = motion_codes.weight[phase_index].unsqueeze(0).expand(len(base_vertices), -1)
                motion_offsets = motion_field(canonical_vertices_t, phase_tensor, phase_codes).detach().cpu().numpy().astype(np.float32)
                phase_displacements.append(motion_offsets)

        final_displacements = np.stack(phase_displacements, axis=0)
        final_displacements = self._propagate_unsupported_displacements(final_displacements, vertex_support_mask_np, edges_np)

        run_dir = observations[0].pointcloud_path.parent
        base_mesh_path = run_dir / cfg.out_subdir / "dynamic_base_mesh.ply"
        base_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        denormalized_base_vertices = canonical_vertices_np * scale + center[None, :]
        base_mesh_export = trimesh.Trimesh(vertices=denormalized_base_vertices, faces=base_faces_np, process=False)
        _mesh_export(base_mesh_export, base_mesh_path)

        return SharedTopologyDynamicFit(
            base_vertices=canonical_vertices_np,
            base_faces=base_faces_np,
            displacements=final_displacements,
            phases=[item.phase for item in observations],
            center=center,
            scale=float(scale),
        )

    def _export_phase_meshes(self, fit: SharedTopologyDynamicFit, mesh_dir: Path) -> list[DynamicMeshBuildResult]:
        results: list[DynamicMeshBuildResult] = []
        for phase, displacement in zip(fit.phases, fit.displacements):
            mesh = self._mesh_from_offsets(fit, displacement)
            if len(mesh.faces) < self.config.min_face_count:
                print(f"[SharedDynamicShapeMotion] phase={phase:.3f}: face count too low ({len(mesh.faces)}), skip")
                continue
            mesh_name = f"dynamic_phase_{phase:.3f}.ply".replace(" ", "")
            mesh_path = mesh_dir / mesh_name
            _mesh_export(mesh, mesh_path)
            results.append(
                DynamicMeshBuildResult(
                    phase=float(phase),
                    mesh_path=mesh_path,
                    vertices=int(len(mesh.vertices)),
                    faces=int(len(mesh.faces)),
                    watertight=bool(mesh.is_watertight),
                    method="shared_topology_decoupled_shape_motion_latent",
                )
            )
        return results


class CPDFieldReferenceCorrespondenceReconstructor(CanonicalPhaseDeformationFieldReconstructor):
    """CPD-Field variant with temporal regularization on shared reference correspondences."""

    @staticmethod
    def _phase_tensor(phase: float, count: int, device: torch.device) -> torch.Tensor:
        return torch.full((count,), float(phase), dtype=torch.float32, device=device)

    def fit(
        self,
        pointcloud_paths: list[Path],
        phase_confidences: Mapping[Path, float] | None = None,
        phase_summaries: Mapping[Path, PointCloudPhaseSummary] | None = None,
    ) -> SharedTopologyDynamicFit:
        cfg = self.config
        observations, center, scale = self._prepare_phase_observations(
            pointcloud_paths,
            phase_confidences=phase_confidences,
            phase_summaries=phase_summaries,
        )
        base_phase_index = self._select_base_phase_index(observations)
        base_mesh = self._build_base_mesh(observations[base_phase_index].pointcloud_path)
        base_vertices_np, base_faces_np = self._normalize_base_mesh(base_mesh, center, scale)

        sample_count = max(int(cfg.surface_batch_size * 3), 4096)
        face_indices_np, barycentric_np = self._sample_surface_plan(base_mesh, sample_count, cfg.random_seed)
        rng = np.random.default_rng(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_seed)

        base_vertices = torch.from_numpy(base_vertices_np).to(self.device)
        base_faces = torch.from_numpy(base_faces_np).to(self.device)
        face_indices = torch.from_numpy(face_indices_np).to(self.device)
        barycentric = torch.from_numpy(barycentric_np).to(self.device)
        reference_points, reference_normals = self._sample_surface_from_vertices(base_vertices, base_faces, face_indices, barycentric)

        phase_points = [torch.from_numpy(item.points).to(self.device) for item in observations]
        phase_normals = [torch.from_numpy(item.normals).to(self.device) for item in observations]
        phase_centroids = [torch.from_numpy(item.centroid).to(self.device) for item in observations]
        phase_weights = np.asarray([item.weight for item in observations], dtype=np.float64)
        sample_probabilities = phase_weights / max(np.sum(phase_weights), 1e-8)
        phase_weight_tensor = torch.tensor(phase_weights, dtype=torch.float32, device=self.device)

        canonical_field = CanonicalField(cfg.canonical_hidden_dim, cfg.canonical_hidden_layers).to(self.device)
        deformation_field = PhaseConditionedDeformationField(
            cfg.deformation_hidden_dim,
            cfg.deformation_hidden_layers,
            cfg.phase_harmonics,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            [
                {"params": canonical_field.parameters(), "lr": cfg.learning_rate},
                {"params": deformation_field.parameters(), "lr": cfg.learning_rate},
            ]
        )
        overlap_loss_scale = self._overlap_loss_scale()
        normal_offset = max(cfg.voxel_size * 0.05, 1e-3)

        for step in range(cfg.train_steps):
            canonical_batch = min(len(reference_points), max(cfg.surface_batch_size, 256))
            canonical_idx = rng.choice(len(reference_points), size=canonical_batch, replace=len(reference_points) < canonical_batch)
            canonical_idx_t = torch.from_numpy(canonical_idx.astype(np.int64)).to(self.device)
            canonical_samples = reference_points[canonical_idx_t].clone().detach().requires_grad_(True)
            canonical_target_normals = reference_normals[canonical_idx_t]
            canonical_sdf, canonical_grad = canonical_field.sdf_and_gradient(canonical_samples)
            canonical_normal_loss = (1.0 - torch.abs(torch.sum(F.normalize(canonical_grad, dim=-1) * canonical_target_normals, dim=-1))).mean()
            canonical_surface_loss = canonical_sdf.abs().mean()

            eikonal_count = min(len(reference_points), max(cfg.eikonal_batch_size, 256))
            eikonal_idx = rng.choice(len(reference_points), size=eikonal_count, replace=len(reference_points) < eikonal_count)
            eikonal_idx_t = torch.from_numpy(eikonal_idx.astype(np.int64)).to(self.device)
            eikonal_base = reference_points[eikonal_idx_t]
            noisy = eikonal_base + torch.randn_like(eikonal_base) * normal_offset
            uniform = torch.empty_like(noisy).uniform_(-0.5 - cfg.bbox_padding, 0.5 + cfg.bbox_padding)
            eikonal_queries = torch.cat([noisy, uniform], dim=0).clone().detach().requires_grad_(True)
            _, eikonal_grad = canonical_field.sdf_and_gradient(eikonal_queries)
            eikonal_loss = ((eikonal_grad.norm(dim=-1) - 1.0) ** 2).mean()

            phase_id = int(rng.choice(len(observations), p=sample_probabilities))
            phase_weight = phase_weight_tensor[phase_id]
            target_points = phase_points[phase_id]
            target_normals = phase_normals[phase_id]
            target_centroid = phase_centroids[phase_id]

            surface_count = min(len(reference_points), max(cfg.surface_batch_size, 512))
            surface_idx = rng.choice(len(reference_points), size=surface_count, replace=len(reference_points) < surface_count)
            surface_idx_t = torch.from_numpy(surface_idx.astype(np.int64)).to(self.device)
            ref_samples = reference_points[surface_idx_t]
            ref_sample_normals = reference_normals[surface_idx_t]
            phase_tensor = self._phase_tensor(observations[phase_id].phase, len(ref_samples), self.device)
            deformed_samples = ref_samples + deformation_field(ref_samples, phase_tensor)

            distances = torch.cdist(deformed_samples, target_points)
            pred_to_target, pred_nn = distances.min(dim=1)
            target_to_pred, _ = distances.min(dim=0)
            surface_loss = phase_weight * (pred_to_target.pow(2).mean() + target_to_pred.pow(2).mean())

            nearest_target_normals = target_normals[pred_nn]
            phase_normal_loss = phase_weight * (1.0 - torch.abs(torch.sum(ref_sample_normals * nearest_target_normals, dim=-1))).mean()
            centroid_loss = phase_weight * ((deformed_samples.mean(dim=0) - target_centroid) ** 2).mean()

            temporal_count = min(len(reference_points), max(cfg.temporal_batch_size, 256))
            temporal_idx = rng.choice(len(reference_points), size=temporal_count, replace=len(reference_points) < temporal_count)
            temporal_idx_t = torch.from_numpy(temporal_idx.astype(np.int64)).to(self.device)
            ref_temporal = reference_points[temporal_idx_t]
            prev_idx_np, center_idx_np, next_idx_np = self._sample_phase_triplet_indices(len(observations), temporal_count, rng)
            prev_phase = torch.tensor([observations[index].phase for index in prev_idx_np], dtype=torch.float32, device=self.device)
            center_phase = torch.tensor([observations[index].phase for index in center_idx_np], dtype=torch.float32, device=self.device)
            next_phase = torch.tensor([observations[index].phase for index in next_idx_np], dtype=torch.float32, device=self.device)

            deformation_prev = deformation_field(ref_temporal, prev_phase)
            deformation_now = deformation_field(ref_temporal, center_phase)
            deformation_next = deformation_field(ref_temporal, next_phase)
            temporal_loss = 0.5 * (((deformation_next - deformation_now) ** 2).mean() + ((deformation_now - deformation_prev) ** 2).mean())
            temporal_accel_loss = ((deformation_next - 2.0 * deformation_now + deformation_prev) ** 2).mean()
            phase_consistency_loss = ((deformation_next - deformation_prev) ** 2).mean()
            deformation_loss = (deformation_now ** 2).mean()

            periodic_count = min(len(reference_points), max(cfg.temporal_batch_size, 256))
            periodic_idx = rng.choice(len(reference_points), size=periodic_count, replace=len(reference_points) < periodic_count)
            periodic_idx_t = torch.from_numpy(periodic_idx.astype(np.int64)).to(self.device)
            ref_periodic = reference_points[periodic_idx_t]
            periodic_stride = max(1, int(round(float(cfg.temporal_delta_phase) * len(observations))))
            zero_phase = self._phase_tensor(observations[0].phase, len(ref_periodic), self.device)
            one_phase = self._phase_tensor(observations[-1].phase, len(ref_periodic), self.device)
            delta_phase = self._phase_tensor(observations[periodic_stride % len(observations)].phase, len(ref_periodic), self.device)
            one_minus_delta_phase = self._phase_tensor(observations[(-periodic_stride) % len(observations)].phase, len(ref_periodic), self.device)
            deformation_zero = deformation_field(ref_periodic, zero_phase)
            deformation_one = deformation_field(ref_periodic, one_phase)
            deformation_delta = deformation_field(ref_periodic, delta_phase)
            deformation_one_minus_delta = deformation_field(ref_periodic, one_minus_delta_phase)
            periodicity_loss = ((deformation_zero - deformation_one) ** 2).mean()
            periodicity_loss = periodicity_loss + 0.5 * (((deformation_delta - deformation_zero) - (deformation_one - deformation_one_minus_delta)) ** 2).mean()

            loss = (
                canonical_surface_loss
                + cfg.eikonal_weight * eikonal_loss
                + cfg.normal_weight * (canonical_normal_loss + phase_normal_loss)
                + surface_loss
                + cfg.centroid_weight * centroid_loss
                + (cfg.temporal_weight * overlap_loss_scale) * temporal_loss
                + cfg.temporal_acceleration_weight * temporal_accel_loss
                + (cfg.phase_consistency_weight * overlap_loss_scale) * phase_consistency_loss
                + cfg.periodicity_weight * periodicity_loss
                + cfg.deformation_weight * deformation_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (step + 1) % max(1, cfg.train_steps // 4) == 0 or step == 0:
                print(
                    "[CPDRef] "
                    f"step={step + 1}/{cfg.train_steps} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"canon={float(canonical_surface_loss.detach().cpu()):.6f} "
                    f"eik={float(eikonal_loss.detach().cpu()):.6f} "
                    f"surface={float(surface_loss.detach().cpu()):.6f} "
                    f"normal={float((canonical_normal_loss + phase_normal_loss).detach().cpu()):.6f} "
                    f"centroid={float(centroid_loss.detach().cpu()):.6f} "
                    f"temporal={float(temporal_loss.detach().cpu()):.6f} "
                    f"accel={float(temporal_accel_loss.detach().cpu()):.6f} "
                    f"phase={float(phase_consistency_loss.detach().cpu()):.6f} "
                    f"periodic={float(periodicity_loss.detach().cpu()):.6f} "
                    f"deform={float(deformation_loss.detach().cpu()):.6f}"
                )

        with torch.no_grad():
            phase_displacements: list[np.ndarray] = []
            for observation in observations:
                phase_tensor = self._phase_tensor(observation.phase, len(base_vertices), self.device)
                displacement = deformation_field(base_vertices, phase_tensor).detach().cpu().numpy().astype(np.float32)
                phase_displacements.append(displacement)

        run_dir = observations[0].pointcloud_path.parent
        base_mesh_path = run_dir / cfg.out_subdir / "dynamic_base_mesh.ply"
        base_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        denormalized_base_vertices = base_vertices_np * scale + center[None, :]
        base_mesh_export = trimesh.Trimesh(vertices=denormalized_base_vertices, faces=base_faces_np, process=False)
        _mesh_export(base_mesh_export, base_mesh_path)

        return SharedTopologyDynamicFit(
            base_vertices=base_vertices_np,
            base_faces=base_faces_np,
            displacements=np.stack(phase_displacements, axis=0),
            phases=[item.phase for item in observations],
            center=center,
            scale=float(scale),
        )

    def _export_phase_meshes(self, fit: SharedTopologyDynamicFit, mesh_dir: Path) -> list[DynamicMeshBuildResult]:
        results: list[DynamicMeshBuildResult] = []
        for phase, displacement in zip(fit.phases, fit.displacements):
            mesh = self._mesh_from_offsets(fit, displacement)
            if len(mesh.faces) < self.config.min_face_count:
                print(f"[CPDRef] phase={phase:.3f}: face count too low ({len(mesh.faces)}), skip")
                continue
            mesh_name = f"dynamic_phase_{phase:.3f}.ply".replace(" ", "")
            mesh_path = mesh_dir / mesh_name
            _mesh_export(mesh, mesh_path)
            print(
                f"[CPDRef] wrote {mesh_path} "
                f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight})"
            )
            results.append(
                DynamicMeshBuildResult(
                    phase=float(phase),
                    mesh_path=mesh_path,
                    vertices=int(len(mesh.vertices)),
                    faces=int(len(mesh.faces)),
                    watertight=bool(mesh.is_watertight),
                    method="cpd_field_reference_correspondence",
                )
            )
        return results

    def _export_timeline_meshes(
        self,
        fit: SharedTopologyDynamicFit,
        mesh_dir: Path,
        timeline_samples: list[tuple[int, float, float]],
    ) -> list[DynamicTimelineMeshBuildResult]:
        selected_samples = self._select_timeline_samples(timeline_samples)
        results: list[DynamicTimelineMeshBuildResult] = []
        phases_np = np.asarray(fit.phases, dtype=np.float32)
        for frame_index, timestamp, phase in selected_samples:
            displacement = self._interpolate_offsets(phases_np, fit.displacements, phase)
            mesh = self._mesh_from_offsets(fit, displacement)
            mesh_name = f"dynamic_timeline_{frame_index:05d}_{phase:.3f}.ply".replace(" ", "")
            mesh_path = mesh_dir / mesh_name
            _mesh_export(mesh, mesh_path)
            results.append(
                DynamicTimelineMeshBuildResult(
                    frame_index=int(frame_index),
                    timestamp=float(timestamp),
                    phase=float(phase),
                    mesh_path=mesh_path,
                    vertices=int(len(mesh.vertices)),
                    faces=int(len(mesh.faces)),
                    watertight=bool(mesh.is_watertight),
                    method="cpd_field_reference_correspondence_timeline",
                )
            )

        summary_path = mesh_dir / "dynamic_timeline_mesh_summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_index", "timestamp", "phase", "mesh", "vertices", "faces", "watertight", "method"])
            for result in results:
                writer.writerow([
                    result.frame_index,
                    f"{result.timestamp:.6f}",
                    f"{result.phase:.6f}",
                    result.mesh_path.name,
                    result.vertices,
                    result.faces,
                    int(result.watertight),
                    result.method,
                ])
        return results


class PriorFreeSpatiotemporalFieldReconstructor(CanonicalPhaseDeformationFieldReconstructor):
    """Observation-driven 4D implicit field without template or deformation priors."""

    @staticmethod
    def _phase_tensor(phase: float, count: int, device: torch.device) -> torch.Tensor:
        return torch.full((count,), float(phase), dtype=torch.float32, device=device)

    def _normalized_bounds(self, observations: list[PhaseObservation]) -> tuple[np.ndarray, np.ndarray]:
        stacked = np.vstack([item.points for item in observations]).astype(np.float32)
        lower = np.min(stacked, axis=0) - float(self.config.bbox_padding)
        upper = np.max(stacked, axis=0) + float(self.config.bbox_padding)
        return lower.astype(np.float32), upper.astype(np.float32)

    def _extract_mesh_for_phase(
        self,
        field: PhaseConditionedSDFField,
        phase: float,
        surface_points: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        center: np.ndarray,
        scale: float,
    ) -> trimesh.Trimesh:
        resolution = int(self.config.mesh_resolution)
        axes = [np.linspace(lower[i], upper[i], resolution, dtype=np.float32) for i in range(3)]
        xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        grid_points = np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).astype(np.float32)

        field_values: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(grid_points), int(self.config.eval_batch_size)):
                stop = min(start + int(self.config.eval_batch_size), len(grid_points))
                batch = torch.from_numpy(grid_points[start:stop]).to(self.device)
                phase_batch = self._phase_tensor(phase, len(batch), self.device)
                field_values.append(field(batch, phase_batch).squeeze(-1).detach().cpu().numpy())
        volume = np.concatenate(field_values, axis=0).reshape(resolution, resolution, resolution)

        with torch.no_grad():
            surface_batch = torch.from_numpy(surface_points.astype(np.float32)).to(self.device)
            surface_phase = self._phase_tensor(phase, len(surface_batch), self.device)
            iso_level = float(field(surface_batch, surface_phase).mean().detach().cpu())
        volume_min = float(np.min(volume))
        volume_max = float(np.max(volume))
        if not (volume_min <= iso_level <= volume_max):
            iso_level = float(np.clip(iso_level, volume_min, volume_max))
        if abs(volume_max - volume_min) <= 1e-8:
            raise ValueError(f"Prior-free field volume collapsed at phase={phase:.3f}")

        vertices, faces = mcubes.marching_cubes(volume, iso_level)
        if len(vertices) == 0 or len(faces) == 0:
            raise ValueError(f"Prior-free field failed to extract a mesh at phase={phase:.3f} with iso={iso_level:.6f}")

        vertices = vertices.astype(np.float32)
        vertices[:, 0] = vertices[:, 0] / (resolution - 1.0) * (upper[0] - lower[0]) + lower[0]
        vertices[:, 1] = vertices[:, 1] / (resolution - 1.0) * (upper[1] - lower[1]) + lower[1]
        vertices[:, 2] = vertices[:, 2] / (resolution - 1.0) * (upper[2] - lower[2]) + lower[2]
        vertices = vertices * float(scale) + center[None, :]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces.astype(np.int64), process=False)
        components = mesh.split(only_watertight=False)
        if components:
            mesh = max(components, key=lambda item: item.area)
        return self._postprocess_dynamic_mesh(mesh)

    def reconstruct(
        self,
        pointcloud_paths: list[Path],
        phase_confidences: Mapping[Path, float] | None = None,
        phase_summaries: Mapping[Path, PointCloudPhaseSummary] | None = None,
        timeline_samples: list[tuple[int, float, float]] | None = None,
    ) -> tuple[list[DynamicMeshBuildResult], list[DynamicTimelineMeshBuildResult]]:
        if not pointcloud_paths:
            return [], []

        cfg = self.config
        observations, center, scale = self._prepare_phase_observations(
            pointcloud_paths,
            phase_confidences=phase_confidences,
            phase_summaries=phase_summaries,
        )
        lower, upper = self._normalized_bounds(observations)

        pooled_points_np = np.vstack([item.points for item in observations]).astype(np.float32)
        phase_points = [torch.from_numpy(item.points).to(self.device) for item in observations]
        phase_normals = [torch.from_numpy(item.normals).to(self.device) for item in observations]
        phase_point_weights = [torch.from_numpy(item.point_weights).to(self.device) for item in observations]
        phase_weights_np = np.asarray([item.weight for item in observations], dtype=np.float64)
        sample_probabilities = phase_weights_np / max(np.sum(phase_weights_np), 1e-8)
        phase_weight_tensor = torch.tensor(phase_weights_np, dtype=torch.float32, device=self.device)
        pooled_points = torch.from_numpy(pooled_points_np).to(self.device)

        field = PhaseConditionedSDFField(
            cfg.canonical_hidden_dim,
            cfg.canonical_hidden_layers,
            cfg.phase_harmonics,
        ).to(self.device)
        optimizer = torch.optim.Adam(field.parameters(), lr=cfg.learning_rate)

        normal_offset = max(float(cfg.voxel_size) / max(float(scale), 1e-8) * 0.5, 1e-3)
        lower_t = torch.from_numpy(lower).to(self.device)
        upper_t = torch.from_numpy(upper).to(self.device)

        rng = np.random.default_rng(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_seed)

        for step in range(int(cfg.train_steps)):
            phase_id = int(rng.choice(len(observations), p=sample_probabilities))
            phase_value = float(observations[phase_id].phase)
            target_points = phase_points[phase_id]
            target_normals = phase_normals[phase_id]
            target_point_weights = phase_point_weights[phase_id]
            phase_weight = phase_weight_tensor[phase_id]

            surface_count = min(int(cfg.surface_batch_size), len(target_points))
            surface_idx = rng.choice(len(target_points), size=surface_count, replace=len(target_points) < surface_count)
            surface_idx_t = torch.from_numpy(surface_idx.astype(np.int64)).to(self.device)
            surface = target_points[surface_idx_t].clone().detach().requires_grad_(True)
            surface_normals = target_normals[surface_idx_t]
            surface_weights = target_point_weights[surface_idx_t]
            surface_weight_sum = surface_weights.sum().clamp_min(1e-6)
            surface_phase = self._phase_tensor(phase_value, len(surface), self.device)

            sdf_surface, grad_surface = field.sdf_and_gradient(surface, surface_phase)
            inner_offset = normal_offset
            outer_offset = normal_offset * 2.0
            outer = surface + surface_normals * inner_offset
            inner = surface - surface_normals * inner_offset
            outer_far = surface + surface_normals * outer_offset
            inner_far = surface - surface_normals * outer_offset
            sdf_outer = field(outer, surface_phase)
            sdf_inner = field(inner, surface_phase)
            sdf_outer_far = field(outer_far, surface_phase)
            sdf_inner_far = field(inner_far, surface_phase)

            surface_loss = phase_weight * (
                (surface_weights * sdf_surface.abs().squeeze(-1)).sum() / surface_weight_sum
                + 0.5 * (surface_weights * (sdf_outer - inner_offset).abs().squeeze(-1)).sum() / surface_weight_sum
                + 0.5 * (surface_weights * (sdf_inner + inner_offset).abs().squeeze(-1)).sum() / surface_weight_sum
                + 0.25 * (surface_weights * (sdf_outer_far - outer_offset).abs().squeeze(-1)).sum() / surface_weight_sum
                + 0.25 * (surface_weights * (sdf_inner_far + outer_offset).abs().squeeze(-1)).sum() / surface_weight_sum
            )
            normal_loss = phase_weight * (
                (surface_weights * (1.0 - torch.abs(torch.sum(F.normalize(grad_surface, dim=-1) * surface_normals, dim=-1)))).sum()
                / surface_weight_sum
            )

            eikonal_count = min(int(cfg.eikonal_batch_size), len(pooled_points_np))
            eikonal_idx = rng.choice(len(pooled_points_np), size=eikonal_count, replace=len(pooled_points_np) < eikonal_count)
            eikonal_idx_t = torch.from_numpy(eikonal_idx.astype(np.int64)).to(self.device)
            eikonal_base = pooled_points[eikonal_idx_t]
            noisy = eikonal_base + torch.randn_like(eikonal_base) * normal_offset
            uniform = torch.rand_like(noisy) * (upper_t - lower_t) + lower_t
            eikonal_queries = torch.cat([noisy, uniform], dim=0).clone().detach().requires_grad_(True)
            eikonal_phases = torch.rand((len(eikonal_queries),), dtype=torch.float32, device=self.device)
            _, grad_queries = field.sdf_and_gradient(eikonal_queries, eikonal_phases)
            eikonal_loss = ((grad_queries.norm(dim=-1) - 1.0) ** 2).mean()

            temporal_count = min(int(cfg.temporal_batch_size), len(pooled_points_np))
            temporal_idx = rng.choice(len(pooled_points_np), size=temporal_count, replace=len(pooled_points_np) < temporal_count)
            temporal_idx_t = torch.from_numpy(temporal_idx.astype(np.int64)).to(self.device)
            temporal_points = pooled_points[temporal_idx_t]
            prev_idx_np, center_idx_np, next_idx_np = self._sample_phase_triplet_indices(len(observations), temporal_count, rng)
            prev_phase = torch.tensor([observations[index].phase for index in prev_idx_np], dtype=torch.float32, device=self.device)
            center_phase = torch.tensor([observations[index].phase for index in center_idx_np], dtype=torch.float32, device=self.device)
            next_phase = torch.tensor([observations[index].phase for index in next_idx_np], dtype=torch.float32, device=self.device)
            sdf_prev = field(temporal_points, prev_phase)
            sdf_now = field(temporal_points, center_phase)
            sdf_next = field(temporal_points, next_phase)
            temporal_loss = 0.5 * (((sdf_next - sdf_now) ** 2).mean() + ((sdf_now - sdf_prev) ** 2).mean())
            temporal_accel_loss = ((sdf_next - 2.0 * sdf_now + sdf_prev) ** 2).mean()
            phase_consistency_loss = ((sdf_next - sdf_prev) ** 2).mean()

            periodic_count = min(int(cfg.temporal_batch_size), len(pooled_points_np))
            periodic_idx = rng.choice(len(pooled_points_np), size=periodic_count, replace=len(pooled_points_np) < periodic_count)
            periodic_idx_t = torch.from_numpy(periodic_idx.astype(np.int64)).to(self.device)
            periodic_points = pooled_points[periodic_idx_t]
            zero_phase = self._phase_tensor(0.0, len(periodic_points), self.device)
            one_phase = self._phase_tensor(1.0, len(periodic_points), self.device)
            periodicity_loss = ((field(periodic_points, zero_phase) - field(periodic_points, one_phase)) ** 2).mean()

            loss = (
                surface_loss
                + cfg.normal_weight * normal_loss
                + cfg.eikonal_weight * eikonal_loss
                + cfg.temporal_weight * temporal_loss
                + cfg.temporal_acceleration_weight * temporal_accel_loss
                + cfg.phase_consistency_weight * phase_consistency_loss
                + cfg.periodicity_weight * periodicity_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (step + 1) % max(1, int(cfg.train_steps) // 4) == 0 or step == 0:
                print(
                    "[PriorFree4D] "
                    f"step={step + 1}/{cfg.train_steps} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"surface={float(surface_loss.detach().cpu()):.6f} "
                    f"normal={float(normal_loss.detach().cpu()):.6f} "
                    f"eikonal={float(eikonal_loss.detach().cpu()):.6f} "
                    f"temporal={float(temporal_loss.detach().cpu()):.6f} "
                    f"accel={float(temporal_accel_loss.detach().cpu()):.6f} "
                    f"phase={float(phase_consistency_loss.detach().cpu()):.6f} "
                    f"periodic={float(periodicity_loss.detach().cpu()):.6f}"
                )

        run_dir = pointcloud_paths[0].parent
        mesh_dir = run_dir / cfg.out_subdir
        mesh_dir.mkdir(parents=True, exist_ok=True)

        results: list[DynamicMeshBuildResult] = []
        for observation in observations:
            try:
                mesh = self._extract_mesh_for_phase(field, observation.phase, observation.points, lower, upper, center, scale)
            except Exception as exc:
                print(f"[PriorFree4D] phase={observation.phase:.3f}: mesh export failed: {exc}")
                continue
            if len(mesh.faces) < cfg.min_face_count:
                print(f"[PriorFree4D] phase={observation.phase:.3f}: face count too low ({len(mesh.faces)}), skip")
                continue
            mesh_name = f"dynamic_phase_{observation.phase:.3f}.ply".replace(" ", "")
            mesh_path = mesh_dir / mesh_name
            _mesh_export(mesh, mesh_path)
            results.append(
                DynamicMeshBuildResult(
                    phase=float(observation.phase),
                    mesh_path=mesh_path,
                    vertices=int(len(mesh.vertices)),
                    faces=int(len(mesh.faces)),
                    watertight=bool(mesh.is_watertight),
                    method="prior_free_4d_field",
                )
            )

        summary_path = mesh_dir / "dynamic_mesh_summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["phase", "mesh", "vertices", "faces", "watertight", "method"])
            for result in results:
                writer.writerow([
                    f"{result.phase:.6f}",
                    result.mesh_path.name,
                    result.vertices,
                    result.faces,
                    int(result.watertight),
                    result.method,
                ])

        timeline_results: list[DynamicTimelineMeshBuildResult] = []
        if cfg.export_timeline_meshes and timeline_samples:
            timeline_dir = run_dir / cfg.timeline_out_subdir
            timeline_dir.mkdir(parents=True, exist_ok=True)
            for frame_index, timestamp, phase in self._select_timeline_samples(timeline_samples):
                nearest_observation = min(observations, key=lambda item: abs(float(item.phase) - float(phase)))
                try:
                    mesh = self._extract_mesh_for_phase(field, phase, nearest_observation.points, lower, upper, center, scale)
                except Exception as exc:
                    print(f"[PriorFree4D] timeline phase={phase:.3f}: mesh export failed: {exc}")
                    continue
                mesh_name = f"dynamic_timeline_{frame_index:05d}_{phase:.3f}.ply".replace(" ", "")
                mesh_path = timeline_dir / mesh_name
                _mesh_export(mesh, mesh_path)
                timeline_results.append(
                    DynamicTimelineMeshBuildResult(
                        frame_index=int(frame_index),
                        timestamp=float(timestamp),
                        phase=float(phase),
                        mesh_path=mesh_path,
                        vertices=int(len(mesh.vertices)),
                        faces=int(len(mesh.faces)),
                        watertight=bool(mesh.is_watertight),
                        method="prior_free_4d_field_timeline",
                    )
                )

            timeline_summary_path = timeline_dir / "dynamic_timeline_mesh_summary.csv"
            with timeline_summary_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frame_index", "timestamp", "phase", "mesh", "vertices", "faces", "watertight", "method"])
                for result in timeline_results:
                    writer.writerow([
                        result.frame_index,
                        f"{result.timestamp:.6f}",
                        f"{result.phase:.6f}",
                        result.mesh_path.name,
                        result.vertices,
                        result.faces,
                        int(result.watertight),
                        result.method,
                    ])
        return results, timeline_results


def reconstruct_dynamic_meshes_from_pointclouds(
    pointcloud_paths: list[Path],
    config: DynamicModelConfig | None = None,
    phase_confidences: Mapping[Path, float] | None = None,
    phase_summaries: Mapping[Path, PointCloudPhaseSummary] | None = None,
    timeline_samples: list[tuple[int, float, float]] | None = None,
) -> tuple[list[DynamicMeshBuildResult], list[DynamicTimelineMeshBuildResult]]:
    resolved_config = config or DynamicModelConfig()
    if resolved_config.method == "prior_free_4d_field":
        reconstructor = PriorFreeSpatiotemporalFieldReconstructor(resolved_config)
    elif resolved_config.method == "shared_topology_global_basis_residual":
        reconstructor = SharedTopologyGlobalBasisResidualReconstructor(resolved_config)
    elif resolved_config.method == "cpd_field_reference_correspondence":
        reconstructor = CPDFieldReferenceCorrespondenceReconstructor(resolved_config)
    elif resolved_config.method == "shared_topology_decoupled_shape_motion_latent":
        reconstructor = SharedTopologyDecoupledShapeMotionReconstructor(resolved_config)
    elif resolved_config.method == "shared_topology_decoupled_motion_latent":
        reconstructor = SharedTopologyDecoupledMotionReconstructor(resolved_config)
    elif resolved_config.method == "shared_topology_deformation_field_reference_correspondence":
        reconstructor = SharedTopologyContinuousFieldReconstructor(resolved_config)
    elif resolved_config.method == "shared_topology_vertex_field_reference_correspondence":
        reconstructor = SharedTopologyReferenceCorrespondenceReconstructor(resolved_config)
    elif resolved_config.method == "shared_topology_vertex_field":
        reconstructor = CanonicalPhaseDeformationFieldReconstructor(resolved_config)
    else:
        raise ValueError(f"Unsupported dynamic reconstruction method: {resolved_config.method}")
    return reconstructor.reconstruct(
        pointcloud_paths,
        phase_confidences=phase_confidences,
        phase_summaries=phase_summaries,
        timeline_samples=timeline_samples,
    )


SpatiotemporalImplicitReconstructor = CanonicalPhaseDeformationFieldReconstructor
