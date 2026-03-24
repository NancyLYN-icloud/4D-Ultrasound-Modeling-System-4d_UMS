"""几何与时空评价指标计算模块。

移植自 4D-Myocardium 项目并针对本系统进行了适配。
"""
from __future__ import annotations

import numpy as np
import trimesh
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


def compute_temporal_smoothness(
    meshes: list[trimesh.Trimesh], num_samples: int = 5000, random_seed: int = 7
) -> float:
    """计算时序网格序列的顶点位移平均速度（用于衡量抖动）。
    注意：这假设网格在拓扑上不一致，因此采用最近点距离近似光流。
    """
    if len(meshes) < 2:
        return 0.0
    
    velocities = []
    for i in range(len(meshes) - 1):
        m0 = meshes[i]
        m1 = meshes[i+1]

        p0 = _sample_geometry_points(m0, num_samples, random_seed=random_seed + i * 2)
        p1 = _sample_geometry_points(m1, num_samples, random_seed=random_seed + i * 2 + 1)
        if len(p0) == 0 or len(p1) == 0:
            continue
        tree = cKDTree(p1)
        
        dists, _ = tree.query(p0, k=1)
        velocities.append(np.mean(dists))

    return float(np.mean(velocities)) if velocities else float("nan")
