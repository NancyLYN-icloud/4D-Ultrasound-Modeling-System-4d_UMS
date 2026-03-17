"""时空耦合动态隐式表面重建。

当前实现采用更偏论文主模型的 `CPD-Field`（Canonical Phase-Deformation Field）框架：
- 用标准 signed-distance field 表示静态形态
- 用相位条件形变场表示动态位移
- 用法向一致性、相位一致采样与时序平滑联合优化
- 在离散相位上导出连续 4D 网格序列
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import re
from typing import Mapping

import mcubes
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from ..config import DynamicMeshBuildResult, DynamicModelConfig, DynamicTimelineMeshBuildResult, PointCloudPhaseSummary
from .canonical_field import CanonicalField
from .deformation_field import PhaseConditionedDeformationField
from .surface_reconstruction import _estimate_normals, _mesh_export, _normalize_points, _phase_index_from_path, _read_xyz_ply, _remove_outliers, _sample_points, _voxel_downsample


_PHASE_CENTER_PATTERN = re.compile(r"_phase_\d+_([0-9]+(?:\.[0-9]+)?)")


@dataclass
class DynamicFieldFit:
    canonical_field: CanonicalField
    deformation_field: PhaseConditionedDeformationField
    center: np.ndarray
    scale: float
    lower: np.ndarray
    upper: np.ndarray
    phases: list[float]


class CanonicalPhaseDeformationFieldReconstructor:
    """CPD-Field: 标准形状 + 相位条件形变场的动态重建器。"""

    def __init__(self, config: DynamicModelConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config)

    @staticmethod
    def _resolve_device(config: DynamicModelConfig) -> torch.device:
        if config.use_cuda_if_available and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _prepare_phase_points(
        self,
        pointcloud_paths: list[Path],
        phase_confidences: Mapping[Path, float] | None = None,
        phase_summaries: Mapping[Path, PointCloudPhaseSummary] | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[float], np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        phase_points: list[np.ndarray] = []
        phases: list[float] = []
        phase_weights: list[float] = []
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
            phase_points.append(points)
            phases.append(float(np.mod(phase, 1.0)))
            if phase_confidences is None:
                weight = 1.0
            else:
                weight = float(phase_confidences.get(path, 1.0))
            weight = float(np.clip(weight, self.config.confidence_floor, 1.0))
            phase_weights.append(weight)
            cleaned_all.append(points)

        if not cleaned_all:
            raise ValueError("没有可用于动态隐式建模的有效相位点云")

        stacked = np.vstack(cleaned_all)
        normalized_all, center, scale = _normalize_points(stacked)
        lower = np.min(normalized_all, axis=0) - self.config.bbox_padding
        upper = np.max(normalized_all, axis=0) + self.config.bbox_padding

        normalized_phase_points: list[np.ndarray] = []
        normalized_phase_normals: list[np.ndarray] = []
        for points in phase_points:
            normalized = (points - center[None, :]) / scale
            normalized_phase_points.append(normalized.astype(np.float32))
            normalized_phase_normals.append(_estimate_normals(normalized.astype(np.float32), neighbors=24))

        anchor_points = normalized_all.astype(np.float32)
        return normalized_phase_points, normalized_phase_normals, phases, phase_weights, center, float(scale), lower.astype(np.float32), upper.astype(np.float32), anchor_points

    @staticmethod
    def _phase_center_from_path(path: Path) -> float | None:
        match = _PHASE_CENTER_PATTERN.search(path.name)
        if match is None:
            return None
        return float(match.group(1))

    @staticmethod
    def _wrap_phase(phase: torch.Tensor) -> torch.Tensor:
        return torch.remainder(phase, 1.0)

    def _sample_phase_consistent_triplet(
        self,
        phase_values: torch.Tensor,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        overlap_stride = self._overlap_neighbor_stride(len(phase_values))
        if overlap_stride is not None:
            sorted_phases, _ = torch.sort(self._wrap_phase(phase_values))
            center_indices = rng.choice(len(sorted_phases), size=batch_size, replace=True)
            prev_indices = (center_indices - overlap_stride) % len(sorted_phases)
            next_indices = (center_indices + overlap_stride) % len(sorted_phases)
            return sorted_phases[prev_indices], sorted_phases[center_indices], sorted_phases[next_indices]

        delta = float(self.config.temporal_delta_phase)
        base_indices = rng.choice(len(phase_values), size=batch_size, replace=True)
        base = phase_values[base_indices]
        jitter = torch.empty((batch_size,), device=self.device).uniform_(-0.5 * delta, 0.5 * delta)
        center_phase = self._wrap_phase(base + jitter)
        prev_phase = self._wrap_phase(center_phase - delta)
        next_phase = self._wrap_phase(center_phase + delta)
        return prev_phase, center_phase, next_phase

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

    def fit(
        self,
        pointcloud_paths: list[Path],
        phase_confidences: Mapping[Path, float] | None = None,
        phase_summaries: Mapping[Path, PointCloudPhaseSummary] | None = None,
    ) -> DynamicFieldFit:
        cfg = self.config
        phase_points, phase_normals, phases, phase_weights, center, scale, lower, upper, anchor_points = self._prepare_phase_points(
            pointcloud_paths,
            phase_confidences=phase_confidences,
            phase_summaries=phase_summaries,
        )
        rng = np.random.default_rng(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_seed)

        canonical_field = CanonicalField(cfg.canonical_hidden_dim, cfg.canonical_hidden_layers).to(self.device)
        deformation_field = PhaseConditionedDeformationField(
            cfg.deformation_hidden_dim,
            cfg.deformation_hidden_layers,
            cfg.phase_harmonics,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            list(canonical_field.parameters()) + list(deformation_field.parameters()),
            lr=cfg.learning_rate,
        )

        phase_tensors = [torch.from_numpy(points).to(self.device) for points in phase_points]
        phase_normal_tensors = [torch.from_numpy(normals).to(self.device) for normals in phase_normals]
        phase_values = torch.tensor(phases, dtype=torch.float32, device=self.device)
        phase_weight_tensor = torch.tensor(phase_weights, dtype=torch.float32, device=self.device)
        anchor_tensor = torch.from_numpy(anchor_points).to(self.device)
        sample_probabilities = np.asarray(phase_weights, dtype=np.float64)
        sample_probabilities = sample_probabilities / max(np.sum(sample_probabilities), 1e-8)
        overlap_stride = self._overlap_neighbor_stride(len(phase_values))
        overlap_loss_scale = self._overlap_loss_scale()
        if overlap_stride is not None:
            print(
                "[CPD-Field] overlap-aware supervision: "
                f"strategy={cfg.supervision_binning_strategy}, "
                f"step_seconds={cfg.supervision_step_seconds}, "
                f"window_seconds={cfg.supervision_window_seconds}, "
                f"neighbor_stride={overlap_stride}, "
                f"local_loss_scale={overlap_loss_scale:.3f}"
            )

        for step in range(cfg.train_steps):
            phase_id = int(rng.choice(len(phase_tensors), p=sample_probabilities))
            points = phase_tensors[phase_id]
            normals = phase_normal_tensors[phase_id]
            phase_weight = phase_weight_tensor[phase_id]
            surface_batch_size = min(cfg.surface_batch_size, len(points))
            normal_batch_size = min(cfg.normal_batch_size, len(points))
            surface_indices = rng.choice(len(points), size=surface_batch_size, replace=len(points) < surface_batch_size)
            normal_indices = rng.choice(len(points), size=normal_batch_size, replace=len(points) < normal_batch_size)
            surface_points = points[surface_indices].clone().detach()
            normal_points = points[normal_indices].clone().detach()
            surface_normals = normals[normal_indices].clone().detach()
            phase_value = phase_values[phase_id].expand(surface_batch_size)
            normal_phase_value = phase_values[phase_id].expand(normal_batch_size)

            canonical_points = surface_points + deformation_field(surface_points, phase_value)
            normal_canonical_points = normal_points + deformation_field(normal_points, normal_phase_value)
            sdf_surface, _ = canonical_field.sdf_and_gradient(canonical_points)
            _, grad_surface = canonical_field.sdf_and_gradient(normal_canonical_points)
            surface_loss = phase_weight * sdf_surface.abs().mean()
            normal_loss = phase_weight * (1.0 - torch.abs(torch.sum(F.normalize(grad_surface, dim=-1) * surface_normals, dim=-1))).mean()

            random_points = torch.empty((cfg.eikonal_batch_size, 3), device=self.device).uniform_(float(lower.min()), float(upper.max()))
            _, grad_random = canonical_field.sdf_and_gradient(random_points)
            eikonal_loss = ((grad_random.norm(dim=-1) - 1.0) ** 2).mean()

            temporal_count = min(cfg.temporal_batch_size, len(anchor_tensor))
            temporal_indices = rng.choice(len(anchor_tensor), size=temporal_count, replace=len(anchor_tensor) < temporal_count)
            temporal_points = anchor_tensor[temporal_indices].clone().detach()
            prev_phase, center_phase, next_phase = self._sample_phase_consistent_triplet(phase_values, temporal_count, rng)
            deformation_prev = deformation_field(temporal_points, prev_phase)
            deformation_now = deformation_field(temporal_points, center_phase)
            deformation_next = deformation_field(temporal_points, next_phase)
            temporal_loss = 0.5 * (((deformation_next - deformation_now) ** 2).mean() + ((deformation_now - deformation_prev) ** 2).mean())
            temporal_accel_loss = ((deformation_next - 2.0 * deformation_now + deformation_prev) ** 2).mean()
            phase_consistency_loss = ((deformation_next - deformation_prev) ** 2).mean()
            deformation_loss = (deformation_now ** 2).mean()

            periodic_points = anchor_tensor[rng.choice(len(anchor_tensor), size=temporal_count, replace=len(anchor_tensor) < temporal_count)].clone().detach()
            phase_zero = torch.zeros((temporal_count,), dtype=torch.float32, device=self.device)
            phase_one = torch.ones((temporal_count,), dtype=torch.float32, device=self.device)
            phase_delta = torch.full((temporal_count,), float(cfg.temporal_delta_phase), dtype=torch.float32, device=self.device)
            phase_one_minus_delta = torch.full((temporal_count,), float(1.0 - cfg.temporal_delta_phase), dtype=torch.float32, device=self.device)
            deformation_zero = deformation_field(periodic_points, phase_zero)
            deformation_one = deformation_field(periodic_points, phase_one)
            deformation_delta = deformation_field(periodic_points, phase_delta)
            deformation_one_minus_delta = deformation_field(periodic_points, phase_one_minus_delta)
            periodicity_loss = ((deformation_zero - deformation_one) ** 2).mean()
            periodicity_loss = periodicity_loss + 0.5 * (((deformation_delta - deformation_zero) - (deformation_one - deformation_one_minus_delta)) ** 2).mean()

            loss = (
                surface_loss
                + cfg.normal_weight * normal_loss
                + cfg.eikonal_weight * eikonal_loss
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
                    "[CPD-Field] "
                    f"step={step + 1}/{cfg.train_steps} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"surface={float(surface_loss.detach().cpu()):.6f} "
                    f"normal={float(normal_loss.detach().cpu()):.6f} "
                    f"eikonal={float(eikonal_loss.detach().cpu()):.6f} "
                    f"temporal={float(temporal_loss.detach().cpu()):.6f} "
                    f"accel={float(temporal_accel_loss.detach().cpu()):.6f} "
                    f"phase={float(phase_consistency_loss.detach().cpu()):.6f} "
                    f"periodic={float(periodicity_loss.detach().cpu()):.6f} "
                    f"deform={float(deformation_loss.detach().cpu()):.6f}"
                )

        canonical_field.eval()
        deformation_field.eval()
        return DynamicFieldFit(
            canonical_field=canonical_field,
            deformation_field=deformation_field,
            center=center,
            scale=scale,
            lower=lower,
            upper=upper,
            phases=phases,
        )

    def _extract_mesh_for_phase(self, fit: DynamicFieldFit, phase: float) -> trimesh.Trimesh:
        cfg = self.config
        axes = [np.linspace(fit.lower[i], fit.upper[i], cfg.mesh_resolution, dtype=np.float32) for i in range(3)]
        xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        grid_points = np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).astype(np.float32)
        field_values = []
        with torch.no_grad():
            for start in range(0, len(grid_points), cfg.eval_batch_size):
                stop = min(start + cfg.eval_batch_size, len(grid_points))
                batch = torch.from_numpy(grid_points[start:stop]).to(self.device)
                phase_batch = torch.full((len(batch),), float(phase), dtype=torch.float32, device=self.device)
                canonical_points = batch + fit.deformation_field(batch, phase_batch)
                field_values.append(fit.canonical_field(canonical_points).squeeze(-1).detach().cpu().numpy())
        field = np.concatenate(field_values, axis=0).reshape(cfg.mesh_resolution, cfg.mesh_resolution, cfg.mesh_resolution)

        vertices, faces = mcubes.marching_cubes(field, cfg.mesh_threshold)
        if len(vertices) == 0 or len(faces) == 0:
            raise ValueError(f"CPD-Field 在 phase={phase:.3f} 未能提取到网格")

        vertices = vertices.astype(np.float32)
        vertices[:, 0] = vertices[:, 0] / (cfg.mesh_resolution - 1.0) * (fit.upper[0] - fit.lower[0]) + fit.lower[0]
        vertices[:, 1] = vertices[:, 1] / (cfg.mesh_resolution - 1.0) * (fit.upper[1] - fit.lower[1]) + fit.lower[1]
        vertices[:, 2] = vertices[:, 2] / (cfg.mesh_resolution - 1.0) * (fit.upper[2] - fit.lower[2]) + fit.lower[2]
        vertices = vertices * fit.scale + fit.center[None, :]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces.astype(np.int64), process=False)
        components = mesh.split(only_watertight=False)
        if components:
            mesh = max(components, key=lambda item: item.area)
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
        if not mesh.is_watertight:
            print(f"[CPD-Field] phase={phase:.3f}: 网格非水密，保留当前几何以避免凸包抹平细节")
        return mesh

    def _export_phase_meshes(
        self,
        fit: DynamicFieldFit,
        mesh_dir: Path,
    ) -> list[DynamicMeshBuildResult]:
        results: list[DynamicMeshBuildResult] = []

        for phase in fit.phases:
            mesh = self._extract_mesh_for_phase(fit, phase)
            if len(mesh.faces) < self.config.min_face_count:
                print(f"[CPD-Field] phase={phase:.3f}: 面片数过少 ({len(mesh.faces)})，跳过导出")
                continue
            mesh_name = f"dynamic_phase_{phase:.3f}.ply".replace(" ", "")
            mesh_path = mesh_dir / mesh_name
            _mesh_export(mesh, mesh_path)
            print(
                f"[CPD-Field] 写入 {mesh_path} "
                f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight})"
            )
            results.append(
                DynamicMeshBuildResult(
                    phase=float(phase),
                    mesh_path=mesh_path,
                    vertices=int(len(mesh.vertices)),
                    faces=int(len(mesh.faces)),
                    watertight=bool(mesh.is_watertight),
                    method="cpd_field",
                )
            )
        return results

    def _select_timeline_samples(
        self,
        timeline_samples: list[tuple[int, float, float]],
    ) -> list[tuple[int, float, float]]:
        stride = max(int(self.config.timeline_stride), 1)
        selected = list(timeline_samples[::stride])
        max_exports = self.config.timeline_max_exports
        if max_exports is not None and len(selected) > int(max_exports):
            indices = np.linspace(0, len(selected) - 1, int(max_exports), dtype=int)
            selected = [selected[idx] for idx in indices]
        return selected

    def _export_timeline_meshes(
        self,
        fit: DynamicFieldFit,
        mesh_dir: Path,
        timeline_samples: list[tuple[int, float, float]],
    ) -> list[DynamicTimelineMeshBuildResult]:
        selected_samples = self._select_timeline_samples(timeline_samples)
        results: list[DynamicTimelineMeshBuildResult] = []

        for frame_index, timestamp, phase in selected_samples:
            mesh = self._extract_mesh_for_phase(fit, phase)
            if len(mesh.faces) < self.config.min_face_count:
                print(
                    f"[CPD-Field] frame={frame_index} ts={timestamp:.3f}s phase={phase:.3f}: "
                    f"面片数过少 ({len(mesh.faces)})，跳过导出"
                )
                continue
            mesh_name = f"frame_{frame_index:04d}_t_{timestamp:09.3f}_phase_{phase:.3f}.ply".replace(" ", "")
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
                    method="cpd_field_timeline",
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
            print(f"[CPD-Field] 已导出 {len(timeline_results)} 个逐帧时间轴网格到 {timeline_dir}")
        return results, timeline_results


def reconstruct_dynamic_meshes_from_pointclouds(
    pointcloud_paths: list[Path],
    config: DynamicModelConfig | None = None,
    phase_confidences: Mapping[Path, float] | None = None,
    phase_summaries: Mapping[Path, PointCloudPhaseSummary] | None = None,
    timeline_samples: list[tuple[int, float, float]] | None = None,
) -> tuple[list[DynamicMeshBuildResult], list[DynamicTimelineMeshBuildResult]]:
    reconstructor = CanonicalPhaseDeformationFieldReconstructor(config or DynamicModelConfig())
    return reconstructor.reconstruct(
        pointcloud_paths,
        phase_confidences=phase_confidences,
        phase_summaries=phase_summaries,
        timeline_samples=timeline_samples,
    )


SpatiotemporalImplicitReconstructor = CanonicalPhaseDeformationFieldReconstructor
