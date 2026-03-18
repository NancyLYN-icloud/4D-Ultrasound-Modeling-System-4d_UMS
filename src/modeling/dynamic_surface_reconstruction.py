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

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
import trimesh

from ..config import DynamicMeshBuildResult, DynamicModelConfig, DynamicTimelineMeshBuildResult, PointCloudPhaseSummary, SurfaceModelConfig
from .canonical_field import CanonicalField
from .deformation_field import PhaseConditionedDeformationField
from .surface_reconstruction import _estimate_normals, _mesh_export, _normalize_points, _phase_index_from_path, _read_xyz_ply, _remove_outliers, _sample_points, _voxel_downsample, reconstruct_meshes_from_pointclouds


_PHASE_CENTER_PATTERN = re.compile(r"_phase_\d+_([0-9]+(?:\.[0-9]+)?)")


@dataclass
class PhaseObservation:
    phase: float
    pointcloud_path: Path
    points: np.ndarray
    normals: np.ndarray
    weight: float
    centroid: np.ndarray


@dataclass
class SharedTopologyDynamicFit:
    base_vertices: np.ndarray
    base_faces: np.ndarray
    displacements: np.ndarray
    phases: list[float]
    center: np.ndarray
    scale: float


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

            observations.append(
                PhaseObservation(
                    phase=float(np.mod(phase, 1.0)),
                    pointcloud_path=path,
                    points=np.asarray(points, dtype=np.float32),
                    normals=np.empty((0, 3), dtype=np.float32),
                    weight=weight,
                    centroid=np.mean(points, axis=0).astype(np.float32),
                )
            )
            cleaned_all.append(points)

        if not observations:
            raise ValueError("No valid phase point clouds available for dynamic reconstruction")

        stacked = np.vstack(cleaned_all).astype(np.float32)
        _, center, scale = _normalize_points(stacked)
        scale = float(scale)

        normalized_observations: list[PhaseObservation] = []
        for observation in observations:
            normalized_points = ((observation.points - center[None, :]) / scale).astype(np.float32)
            normalized_normals = _estimate_normals(normalized_points, neighbors=24)
            normalized_centroid = np.mean(normalized_points, axis=0).astype(np.float32)
            normalized_observations.append(
                PhaseObservation(
                    phase=observation.phase,
                    pointcloud_path=observation.pointcloud_path,
                    points=normalized_points,
                    normals=normalized_normals,
                    weight=observation.weight,
                    centroid=normalized_centroid,
                )
            )

        normalized_observations.sort(key=lambda item: item.phase)
        return normalized_observations, center.astype(np.float32), scale

    def _select_base_phase_index(self, observations: list[PhaseObservation]) -> int:
        phases = np.asarray([item.phase for item in observations], dtype=np.float64)
        return int(np.argmin(phases))

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
    def _build_edges(faces: np.ndarray) -> np.ndarray:
        edges = np.vstack([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]).astype(np.int64)
        edges = np.sort(edges, axis=1)
        return np.unique(edges, axis=0)

    @staticmethod
    def _initialize_offsets(base_vertices: np.ndarray, observations: list[PhaseObservation]) -> np.ndarray:
        offsets = np.zeros((len(observations), len(base_vertices), 3), dtype=np.float32)
        for index, observation in enumerate(observations):
            tree = cKDTree(observation.points)
            _, nn_indices = tree.query(base_vertices, k=1)
            offsets[index] = observation.points[np.asarray(nn_indices, dtype=np.int64)] - base_vertices
        return offsets.astype(np.float32)

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
        edges_np = self._build_edges(base_faces_np)
        initial_offsets_np = self._initialize_offsets(base_vertices_np, observations)

        rng = np.random.default_rng(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_seed)

        base_vertices = torch.from_numpy(base_vertices_np).to(self.device)
        base_faces = torch.from_numpy(base_faces_np).to(self.device)
        face_indices = torch.from_numpy(face_indices_np).to(self.device)
        barycentric = torch.from_numpy(barycentric_np).to(self.device)
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
            phase_id = int(rng.choice(len(observations), p=sample_probabilities))
            target_points = phase_points[phase_id]
            target_normals = phase_normals[phase_id]
            target_centroid = phase_centroids[phase_id]
            phase_weight = phase_weight_tensor[phase_id]

            vertices = base_vertices + offsets[phase_id]
            pred_samples, pred_normals = self._sample_surface_from_vertices(vertices, base_faces, face_indices, barycentric)

            distances = torch.cdist(pred_samples, target_points)
            pred_to_target, pred_nn = distances.min(dim=1)
            target_to_pred, _ = distances.min(dim=0)
            surface_loss = phase_weight * (pred_to_target.pow(2).mean() + target_to_pred.pow(2).mean())

            nearest_target_normals = target_normals[pred_nn]
            normal_loss = phase_weight * (1.0 - torch.abs(torch.sum(pred_normals * nearest_target_normals, dim=-1))).mean()
            centroid_loss = phase_weight * ((pred_samples.mean(dim=0) - target_centroid) ** 2).mean()
            spatial_loss = ((offsets[phase_id, edges[:, 0]] - offsets[phase_id, edges[:, 1]]) ** 2).mean()

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

        return SharedTopologyDynamicFit(
            base_vertices=base_vertices_np,
            base_faces=base_faces_np,
            displacements=offsets.detach().cpu().numpy().astype(np.float32),
            phases=[item.phase for item in observations],
            center=center,
            scale=float(scale),
        )

    def _mesh_from_offsets(self, fit: SharedTopologyDynamicFit, displacements: np.ndarray) -> trimesh.Trimesh:
        vertices = (fit.base_vertices + displacements) * fit.scale + fit.center[None, :]
        mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=fit.base_faces.astype(np.int64), process=False)
        mesh.remove_unreferenced_vertices()
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
        return mesh

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


def reconstruct_dynamic_meshes_from_pointclouds(
    pointcloud_paths: list[Path],
    config: DynamicModelConfig | None = None,
    phase_confidences: Mapping[Path, float] | None = None,
    phase_summaries: Mapping[Path, PointCloudPhaseSummary] | None = None,
    timeline_samples: list[tuple[int, float, float]] | None = None,
) -> tuple[list[DynamicMeshBuildResult], list[DynamicTimelineMeshBuildResult]]:
    resolved_config = config or DynamicModelConfig()
    if resolved_config.method == "cpd_field_reference_correspondence":
        reconstructor = CPDFieldReferenceCorrespondenceReconstructor(resolved_config)
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
