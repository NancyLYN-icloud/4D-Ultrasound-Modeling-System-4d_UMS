"""NeuralGF-inspired implicit surface reconstruction for phase point clouds."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import re

import mcubes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from scipy.spatial import cKDTree

from ..config import SurfaceModelConfig


@dataclass
class MeshBuildResult:
    pointcloud_path: Path
    mesh_path: Path
    timestamp_s: float | None
    input_points: int
    sampled_points: int
    vertices: int
    faces: int
    watertight: bool
    method: str


_PHASE_INDEX_PATTERN = re.compile(r"_phase_(\d+)_")


def _phase_index_from_path(path: Path) -> int | None:
    match = _PHASE_INDEX_PATTERN.search(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _read_xyz_ply(path: Path) -> np.ndarray:
    vertex_count = 0
    header_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[2])
            if stripped == "end_header":
                break

    if vertex_count <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    points = np.loadtxt(path, skiprows=header_lines, max_rows=vertex_count, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    return np.asarray(points[:, :3], dtype=np.float32)


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0 or voxel_size <= 0:
        return points
    coords = np.floor(points / voxel_size).astype(np.int64)
    unique_voxels, inverse = np.unique(coords, axis=0, return_inverse=True)
    sums = np.zeros((len(unique_voxels), 3), dtype=np.float64)
    counts = np.zeros(len(unique_voxels), dtype=np.int64)
    np.add.at(sums, inverse, points)
    np.add.at(counts, inverse, 1)
    return (sums / np.maximum(counts[:, None], 1)).astype(np.float32)


def _remove_outliers(points: np.ndarray, neighbors: int, std_ratio: float) -> np.ndarray:
    if len(points) < max(32, neighbors + 1):
        return points
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(neighbors + 1, len(points)))
    mean_distances = np.mean(distances[:, 1:], axis=1)
    threshold = float(np.median(mean_distances) + std_ratio * np.std(mean_distances))
    keep = mean_distances <= max(threshold, 1e-6)
    filtered = points[keep]
    return filtered if len(filtered) >= 128 else points


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[np.sort(indices)]


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = 0.5 * (lower + upper)
    scale = float(np.max(upper - lower))
    scale = max(scale, 1e-6)
    normalized = (points - center[None, :]) / scale
    return normalized.astype(np.float32), center.astype(np.float32), scale


def _estimate_normals(points: np.ndarray, neighbors: int) -> np.ndarray:
    tree = cKDTree(points)
    query_k = min(max(neighbors + 1, 4), len(points))
    _, indices = tree.query(points, k=query_k)
    centroid = np.mean(points, axis=0)
    normals = np.zeros_like(points, dtype=np.float32)

    for row_index, neighbor_ids in enumerate(np.atleast_2d(indices)):
        local = points[neighbor_ids]
        local_center = np.mean(local, axis=0)
        cov = np.cov((local - local_center).T)
        _, vectors = np.linalg.eigh(cov)
        normal = vectors[:, 0]
        if np.dot(normal, points[row_index] - centroid) < 0.0:
            normal = -normal
        normals[row_index] = normal / max(np.linalg.norm(normal), 1e-8)

    return normals.astype(np.float32)


def _mesh_export(mesh: trimesh.Trimesh, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path, file_type="ply")


class NeuralGradientField(nn.Module):
    def __init__(self, hidden_dim: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 3
        for _ in range(hidden_layers):
            linear = nn.Linear(in_dim, hidden_dim)
            nn.init.normal_(linear.weight, mean=0.0, std=np.sqrt(2.0 / max(hidden_dim, 1)))
            nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)
            layers.append(nn.Softplus(beta=100.0))
            in_dim = hidden_dim
        final = nn.Linear(in_dim, 1)
        nn.init.normal_(final.weight, mean=np.sqrt(np.pi) / np.sqrt(max(in_dim, 1)), std=1e-4)
        nn.init.constant_(final.bias, -0.1)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.network(points)

    def sdf_and_gradient(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        points = points.requires_grad_(True)
        sdf = self.forward(points)
        grad = torch.autograd.grad(
            outputs=sdf,
            inputs=points,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return sdf, grad


class NeuralGFSurfaceReconstructor:
    def __init__(self, config: SurfaceModelConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config)

    @staticmethod
    def _resolve_device(config: SurfaceModelConfig) -> torch.device:
        if config.use_cuda_if_available and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def reconstruct(self, points: np.ndarray) -> tuple[trimesh.Trimesh, dict[str, float | int | bool | str]]:
        cleaned = _voxel_downsample(points, self.config.voxel_size)
        cleaned = _remove_outliers(cleaned, self.config.outlier_neighbors, self.config.outlier_std_ratio)
        cleaned = _sample_points(cleaned, self.config.max_points, self.config.random_seed)
        if len(cleaned) < 128:
            raise ValueError("点云点数过少，无法进行 NeuralGF 建模")

        normalized, center, scale = _normalize_points(cleaned)
        normals = _estimate_normals(normalized, self.config.normal_neighbors)
        network = self._train_field(normalized, normals)
        mesh = self._extract_mesh(network, normalized, center, scale)

        stats: dict[str, float | int | bool | str] = {
            "input_points": int(len(points)),
            "sampled_points": int(len(cleaned)),
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "method": "neuralgf_implicit",
        }
        return mesh, stats

    def _train_field(self, points: np.ndarray, normals: np.ndarray) -> NeuralGradientField:
        cfg = self.config
        rng = np.random.default_rng(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_seed)
        network = NeuralGradientField(cfg.hidden_dim, cfg.hidden_layers).to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=cfg.learning_rate)

        points_tensor = torch.from_numpy(points).to(self.device)
        normals_tensor = torch.from_numpy(normals).to(self.device)
        normal_offset = max(cfg.normal_offset_scale, 1e-4)

        for step in range(cfg.train_steps):
            surface_idx = rng.choice(len(points), size=min(cfg.surface_batch_size, len(points)), replace=len(points) < cfg.surface_batch_size)
            eikonal_idx = rng.choice(len(points), size=min(cfg.eikonal_batch_size, len(points)), replace=len(points) < cfg.eikonal_batch_size)

            surface = points_tensor[surface_idx].clone().detach().requires_grad_(True)
            surface_normals = normals_tensor[surface_idx]
            eikonal_base = points_tensor[eikonal_idx]
            noisy = eikonal_base + torch.randn_like(eikonal_base) * normal_offset
            uniform = torch.empty_like(noisy).uniform_(
                -0.5 - cfg.bbox_padding,
                0.5 + cfg.bbox_padding,
            )
            queries = torch.cat([noisy, uniform], dim=0).clone().detach().requires_grad_(True)

            sdf_surface, grad_surface = network.sdf_and_gradient(surface)
            sdf_queries, grad_queries = network.sdf_and_gradient(queries)

            grad_surface_n = F.normalize(grad_surface, dim=-1)
            grad_queries_n = F.normalize(grad_queries, dim=-1)
            projected = noisy - sdf_queries[: len(noisy)] * grad_queries_n[: len(noisy)]

            nn_dist = torch.cdist(projected.unsqueeze(0), points_tensor.unsqueeze(0)).amin(dim=-1).mean()
            surface_loss = sdf_surface.abs().mean() + 0.5 * nn_dist
            eikonal_loss = ((grad_queries.norm(dim=-1) - 1.0) ** 2).mean()
            normal_loss = (1.0 - torch.abs(torch.sum(grad_surface_n * surface_normals, dim=-1))).mean()
            loss = (
                cfg.surface_weight * surface_loss
                + cfg.eikonal_weight * eikonal_loss
                + cfg.normal_weight * normal_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (step + 1) % max(1, cfg.train_steps // 4) == 0 or step == 0:
                print(
                    "[Surface][NeuralGF] "
                    f"step={step + 1}/{cfg.train_steps} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"surface={float(surface_loss.detach().cpu()):.6f} "
                    f"eikonal={float(eikonal_loss.detach().cpu()):.6f} "
                    f"normal={float(normal_loss.detach().cpu()):.6f}"
                )

        network.eval()
        return network

    def _extract_mesh(
        self,
        network: NeuralGradientField,
        normalized_points: np.ndarray,
        center: np.ndarray,
        scale: float,
    ) -> trimesh.Trimesh:
        cfg = self.config
        lower = np.min(normalized_points, axis=0) - cfg.bbox_padding
        upper = np.max(normalized_points, axis=0) + cfg.bbox_padding
        axes = [np.linspace(lower[i], upper[i], cfg.mesh_resolution, dtype=np.float32) for i in range(3)]
        xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        grid_points = np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)])

        field_values = []
        with torch.no_grad():
            for start in range(0, len(grid_points), cfg.eval_batch_size):
                stop = min(start + cfg.eval_batch_size, len(grid_points))
                batch = torch.from_numpy(grid_points[start:stop]).to(self.device)
                field_values.append(network(batch).squeeze(-1).detach().cpu().numpy())
        field = np.concatenate(field_values, axis=0).reshape(cfg.mesh_resolution, cfg.mesh_resolution, cfg.mesh_resolution)

        vertices, faces = mcubes.marching_cubes(field, cfg.mesh_threshold)
        if len(vertices) == 0 or len(faces) == 0:
            raise ValueError("NeuralGF 隐式场未能提取到网格")

        vertices = vertices.astype(np.float32)
        vertices[:, 0] = vertices[:, 0] / (cfg.mesh_resolution - 1.0) * (upper[0] - lower[0]) + lower[0]
        vertices[:, 1] = vertices[:, 1] / (cfg.mesh_resolution - 1.0) * (upper[1] - lower[1]) + lower[1]
        vertices[:, 2] = vertices[:, 2] / (cfg.mesh_resolution - 1.0) * (upper[2] - lower[2]) + lower[2]
        vertices = vertices * scale + center[None, :]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces.astype(np.int64), process=False)
        components = mesh.split(only_watertight=False)
        if components:
            mesh = max(components, key=lambda item: item.area)
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
        if not mesh.is_watertight:
            mesh = mesh.convex_hull
        return mesh


def reconstruct_meshes_from_pointclouds(
    pointcloud_paths: list[Path],
    config: SurfaceModelConfig | None = None,
    phase_bin_step_seconds: float | None = None,
) -> list[MeshBuildResult]:
    cfg = config or SurfaceModelConfig()
    if not pointcloud_paths:
        return []

    run_dir = pointcloud_paths[0].parent
    mesh_dir = run_dir / cfg.out_subdir
    mesh_dir.mkdir(parents=True, exist_ok=True)
    reconstructor = NeuralGFSurfaceReconstructor(cfg)
    results: list[MeshBuildResult] = []

    for pointcloud_path in pointcloud_paths:
        points = _read_xyz_ply(pointcloud_path)
        if len(points) == 0:
            print(f"[Surface] 跳过空点云: {pointcloud_path.name}")
            continue

        phase_index = _phase_index_from_path(pointcloud_path)

        mesh, stats = reconstructor.reconstruct(points)
        if int(stats["faces"]) < cfg.min_face_count:
            print(f"[Surface] {pointcloud_path.name}: 面片数过少 ({stats['faces']})，跳过导出")
            continue

        mesh_path = mesh_dir / pointcloud_path.name.replace(".ply", "_mesh.ply")
        _mesh_export(mesh, mesh_path)
        print(
            f"[Surface] 写入 {mesh_path} "
            f"({stats['vertices']} verts, {stats['faces']} faces, watertight={stats['watertight']}, method={stats['method']})"
        )
        results.append(
            MeshBuildResult(
                pointcloud_path=pointcloud_path,
                mesh_path=mesh_path,
                timestamp_s=(
                    None
                    if phase_bin_step_seconds is None or phase_index is None
                    else float(phase_index * phase_bin_step_seconds)
                ),
                input_points=int(stats["input_points"]),
                sampled_points=int(stats["sampled_points"]),
                vertices=int(stats["vertices"]),
                faces=int(stats["faces"]),
                watertight=bool(stats["watertight"]),
                method=str(stats["method"]),
            )
        )

    summary_path = mesh_dir / "mesh_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase_pointcloud", "timestamp_s", "mesh", "input_points", "sampled_points", "vertices", "faces", "watertight", "method"])
        for result in results:
            writer.writerow(
                [
                    result.pointcloud_path.name,
                    "" if result.timestamp_s is None else f"{result.timestamp_s:.6f}",
                    result.mesh_path.name,
                    result.input_points,
                    result.sampled_points,
                    result.vertices,
                    result.faces,
                    int(result.watertight),
                    result.method,
                ]
            )
    return results
