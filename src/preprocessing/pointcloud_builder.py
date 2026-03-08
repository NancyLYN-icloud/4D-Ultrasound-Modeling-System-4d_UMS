"""根据相位分箱生成每相位点云。

功能：
- 遍历每个 `PhaseBin`，对其中的 `ScanSample.volume_slice` 提取高强度体素点
- 将体素坐标映射到世界坐标（使用 `ScanSample.position` 和 `ScanSample.orientation`）
- 对每个相位输出一个 PLY 文件：`data/processed/phase_pointclouds/phase_{idx}_{center:.2f}.ply`

该实现参考 `proprecess/ultrasound_to_pointcloud.py` 的思路，但针对项目内 `ScanSample` 数据结构进行简化和适配。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

try:
    import cv2
except Exception:  # OpenCV optional
    cv2 = None
import numpy as np

from ..config import PhaseBin, ScanSample


def _volume_voxel_world_coords(volume_shape: Tuple[int, int, int], pixel_spacing: float, slice_thickness: float) -> np.ndarray:
    """返回体素中心在局部切片坐标系(相对于切片中心)的 (N,3) 坐标数组。

    volume_shape: (D, H, W)
    轴顺序：depth, height, width
    局部坐标系：X 向右 (width)，Y 向下 (height)，Z 指向探头外（depth）
    原点设在切片中心。
    """
    D, H, W = volume_shape
    xs = (np.arange(W) - (W - 1) / 2.0) * pixel_spacing
    ys = (np.arange(H) - (H - 1) / 2.0) * pixel_spacing
    zs = (np.arange(D) - (D - 1) / 2.0) * slice_thickness
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return coords


def _uniform_sample_contour(contour: np.ndarray, spacing: float = 2.0) -> np.ndarray:
    """沿轮廓均匀采样点，返回 (M,2) 的浮点坐标。"""
    if contour.ndim == 1:
        return contour.reshape(1, 2)
    contour_closed = np.vstack([contour, contour[0]])
    segs = np.diff(contour_closed, axis=0)
    seg_lengths = np.linalg.norm(segs, axis=1)
    perimeter = seg_lengths.sum()
    if perimeter <= 0:
        return contour[:1]
    n_samples = max(int(perimeter / max(1e-3, spacing)), 3)
    positions = np.linspace(0, perimeter, n_samples, endpoint=False)
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    sampled = []
    for p in positions:
        idx = np.searchsorted(cum, p) - 1
        idx = int(np.clip(idx, 0, len(segs) - 1))
        seg_start = contour_closed[idx]
        seg = segs[idx]
        seg_len = seg_lengths[idx]
        if seg_len <= 1e-6:
            sampled.append(seg_start)
            continue
        t = (p - cum[idx]) / seg_len
        t = float(np.clip(t, 0.0, 1.0))
        pt = seg_start + t * seg
        sampled.append(pt)
    return np.asarray(sampled, dtype=float)


def samples_to_pointcloud(
    samples: List[ScanSample], *,
    pixel_spacing: float = 1.0,
    slice_thickness: float = 1.0,
    intensity_threshold: float | str = "auto",
    min_contour_area: float = 50.0,
    sample_spacing: float = 2.0,
    max_points: int | None = None,
) -> np.ndarray:
    """基于轮廓提取将若干 `ScanSample` 的 `volume_slice` 合并为 Nx3 点云（世界坐标）。

    对每个 sample 的每个 depth 切片做阈值分割 -> 形态学清理 -> 提取外部轮廓 -> 均匀采样 -> 投影到世界坐标
    """
    world_points = []
    for s in samples:
        vol = np.asarray(s.volume_slice, dtype=float)
        if vol.ndim == 2:
            vol = vol[np.newaxis, ...]
        D, H, W = vol.shape

        # compute threshold per-volume
        if intensity_threshold == "auto":
            thr = float(np.mean(vol) + 0.5 * np.std(vol))
        else:
            thr = float(intensity_threshold)

        # per-slice contour extraction
        for d in range(D):
            frame = vol[d]
            bw = (frame > thr).astype('uint8') * 255
            if bw.sum() == 0:
                continue
            kernel = np.ones((3, 3), np.uint8)
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            for ctr in contours:
                area = cv2.contourArea(ctr)
                if area < min_contour_area:
                    continue
                pts2d = ctr.squeeze().astype(float)
                sampled = _uniform_sample_contour(pts2d, spacing=sample_spacing)
                if sampled.size == 0:
                    continue
                xs = (sampled[:, 0] - (W - 1) / 2.0) * pixel_spacing
                ys = (sampled[:, 1] - (H - 1) / 2.0) * pixel_spacing
                zs = np.full((sampled.shape[0],), (d - (D - 1) / 2.0) * slice_thickness)
                coords_local = np.column_stack([xs, ys, zs])

                R_mat = np.asarray(s.orientation, dtype=float)
                t_vec = np.asarray(s.position, dtype=float)
                pts_world = (R_mat @ coords_local.T).T + t_vec.reshape(1, 3)
                world_points.append(pts_world)

    if not world_points:
        return np.zeros((0, 3), dtype=float)

    pts = np.vstack(world_points)
    if max_points is not None and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts


def write_ply(points: np.ndarray, out_path: str) -> None:
    """Write XYZ ASCII PLY file."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]
    with p.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def build_pointclouds_from_phase_bins(phase_bins: List[PhaseBin], *, out_dir: str | Path = "data/processed/phase_pointclouds", pixel_spacing: float = 1.0, slice_thickness: float = 1.0, intensity_threshold: float | str = "auto", max_points_per_phase: int | None = 200000) -> List[Path]:
    """为每个相位分箱生成点云文件并返回写入路径列表。"""
    out_dir = Path(out_dir)
    written = []
    for idx, bin in enumerate(phase_bins):
        samples = bin.samples
        if not samples:
            print(f"[PointCloud] phase {idx} (center={bin.phase_center:.3f}): no samples, skip")
            continue
        pts = samples_to_pointcloud(samples, pixel_spacing=pixel_spacing, slice_thickness=slice_thickness, intensity_threshold=intensity_threshold, max_points=max_points_per_phase)
        if pts.size == 0:
            print(f"[PointCloud] phase {idx} (center={bin.phase_center:.3f}): no points extracted")
            continue
        out_path = out_dir / f"phase_{idx:03d}_{bin.phase_center:.3f}.ply"
        write_ply(pts, str(out_path))
        print(f"[PointCloud] 写入 {out_path} ({pts.shape[0]} pts)")
        written.append(out_path)
    return written


if __name__ == "__main__":
    print("pointcloud_builder: module provides `build_pointclouds_from_phase_bins` to export per-phase PLY files.")
