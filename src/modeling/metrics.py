"""几何与时空评价指标计算模块。

移植自 4D-Myocardium 项目并针对本系统进行了适配。
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def _geometry_vertices(geometry: trimesh.Trimesh | trimesh.points.PointCloud) -> np.ndarray:
    vertices = np.asarray(getattr(geometry, "vertices", np.empty((0, 3))), dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float64)
    return vertices


def _sample_geometry_points(
    geometry: trimesh.Trimesh | trimesh.points.PointCloud,
    num_samples: int,
    random_seed: int = 7,
) -> np.ndarray:
    if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0:
        sample_count = min(int(num_samples), max(len(geometry.faces) * 3, 1))
        random_state = np.random.get_state()
        np.random.seed(int(random_seed))
        try:
            points, _ = trimesh.sample.sample_surface(geometry, sample_count)
        finally:
            np.random.set_state(random_state)
        return np.asarray(points, dtype=np.float64)

    vertices = _geometry_vertices(geometry)
    if len(vertices) == 0:
        return vertices
    if len(vertices) <= num_samples:
        return vertices

    rng = np.random.default_rng(int(random_seed))
    indices = rng.choice(len(vertices), size=int(num_samples), replace=False)
    return vertices[indices]


def compute_chamfer_distance(
    mesh_pred: trimesh.Trimesh,
    mesh_gt: trimesh.Trimesh | trimesh.points.PointCloud,
    num_samples: int = 10000,
    random_seed: int = 7,
) -> float:
    """计算对称 Chamfer Distance (L2)。
    
    CD = (1/N) * ( sum(min(dist(p, GT)^2)) + sum(min(dist(q, Pred)^2)) )
    """
    # 采样点
    pred_points = _sample_geometry_points(mesh_pred, num_samples, random_seed=random_seed)
    gt_points = _sample_geometry_points(mesh_gt, num_samples, random_seed=random_seed + 1)
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("nan")

    # Pred -> GT
    gt_tree = cKDTree(gt_points)
    dist_pred_to_gt, _ = gt_tree.query(pred_points, k=1)
    
    # GT -> Pred
    pred_tree = cKDTree(pred_points)
    dist_gt_to_pred, _ = pred_tree.query(gt_points, k=1)

    chamfer_l2 = np.mean(dist_pred_to_gt ** 2) + np.mean(dist_gt_to_pred ** 2)
    return float(chamfer_l2)


def compute_hausdorff_distance(
    mesh_pred: trimesh.Trimesh,
    mesh_gt: trimesh.Trimesh | trimesh.points.PointCloud,
    num_samples: int = 10000,
    percentile: float = 95.0,
    random_seed: int = 7,
) -> float:
    """计算单向 Hausdorff Distance (通常取 Pred->GT 的 HD95)。"""
    pred_points = _sample_geometry_points(mesh_pred, num_samples, random_seed=random_seed)
    gt_points = _sample_geometry_points(mesh_gt, num_samples, random_seed=random_seed + 1)
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("nan")

    gt_tree = cKDTree(gt_points)
    dist_pred_to_gt, _ = gt_tree.query(pred_points, k=1)
    
    return float(np.percentile(dist_pred_to_gt, percentile))


def compute_surface_mae(
    mesh_pred: trimesh.Trimesh,
    mesh_gt: trimesh.Trimesh | trimesh.points.PointCloud,
    num_samples: int = 10000,
    random_seed: int = 7,
) -> float:
    """计算对称平均表面绝对误差。"""
    pred_points = _sample_geometry_points(mesh_pred, num_samples, random_seed=random_seed)
    gt_points = _sample_geometry_points(mesh_gt, num_samples, random_seed=random_seed + 1)
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("nan")

    gt_tree = cKDTree(gt_points)
    dist_pred_to_gt, _ = gt_tree.query(pred_points, k=1)
    pred_tree = cKDTree(pred_points)
    dist_gt_to_pred, _ = pred_tree.query(gt_points, k=1)
    return float(0.5 * (np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)))


def compute_earth_movers_distance(
    mesh_pred: trimesh.Trimesh,
    mesh_gt: trimesh.Trimesh | trimesh.points.PointCloud,
    num_samples: int = 256,
    random_seed: int = 7,
) -> float:
    """计算基于离散最优匹配的近似 EMD。"""
    pred_points = _sample_geometry_points(mesh_pred, num_samples, random_seed=random_seed)
    gt_points = _sample_geometry_points(mesh_gt, num_samples, random_seed=random_seed + 1)
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("nan")

    sample_count = min(len(pred_points), len(gt_points), int(num_samples))
    pred_points = pred_points[:sample_count]
    gt_points = gt_points[:sample_count]
    pairwise_distances = np.linalg.norm(pred_points[:, None, :] - gt_points[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(pairwise_distances)
    return float(np.mean(pairwise_distances[row_ind, col_ind]))


def _quantized_voxel_indices(points: np.ndarray, pitch: float, origin: np.ndarray) -> set[tuple[int, int, int]]:
    if len(points) == 0:
        return set()
    indices = np.floor((points - origin[None, :]) / float(pitch) + 1e-6).astype(np.int64)
    return {tuple(index.tolist()) for index in indices}


def _geometry_voxel_points(
    geometry: trimesh.Trimesh | trimesh.points.PointCloud,
    pitch: float,
    random_seed: int,
) -> np.ndarray:
    if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0:
        try:
            voxel_grid = geometry.voxelized(pitch).fill()
        except Exception:
            voxel_grid = geometry.voxelized(pitch)
        points = np.asarray(voxel_grid.points, dtype=np.float64)
        if len(points) > 0:
            return points
    return _sample_geometry_points(geometry, num_samples=12000, random_seed=random_seed)


def compute_dice_score(
    mesh_pred: trimesh.Trimesh,
    mesh_gt: trimesh.Trimesh | trimesh.points.PointCloud,
    voxel_pitch: float = 2.0,
    random_seed: int = 7,
) -> float:
    """计算共享体素网格上的 Dice 系数。"""
    pred_points = _geometry_voxel_points(mesh_pred, pitch=voxel_pitch, random_seed=random_seed)
    gt_points = _geometry_voxel_points(mesh_gt, pitch=voxel_pitch, random_seed=random_seed + 1)
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("nan")

    origin = np.minimum(pred_points.min(axis=0), gt_points.min(axis=0)) - float(voxel_pitch)
    pred_voxels = _quantized_voxel_indices(pred_points, voxel_pitch, origin)
    gt_voxels = _quantized_voxel_indices(gt_points, voxel_pitch, origin)
    overlap = len(pred_voxels & gt_voxels)
    return float(2.0 * overlap / max(len(pred_voxels) + len(gt_voxels), 1))
