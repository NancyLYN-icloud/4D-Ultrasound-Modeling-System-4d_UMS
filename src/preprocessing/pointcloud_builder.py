"""根据相位分箱生成每相位点云。

功能：
- 遍历每个 `PhaseBin`，对其中的 `ScanSample.volume_slice` 提取高强度体素点
- 将体素坐标映射到世界坐标（使用 `ScanSample.position` 和 `ScanSample.orientation`）
- 对每个相位输出一个 PLY 文件到配置指定的数据目录下

该实现参考 `proprecess/ultrasound_to_pointcloud.py` 的思路，但针对项目内 `ScanSample` 数据结构进行简化和适配。
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import sys

try:
    import cv2
    _cv2_import_error: Exception | None = None
except Exception as exc:  # OpenCV optional
    _cv2_import_error = exc
    cv2 = None
import numpy as np

from ..config import PhaseBin, PointCloudConfig, PointCloudPhaseSummary, ScanSample


def _create_indexed_output_dir(base_dir: Path, prefix: str = "phase_pointclouds_run") -> tuple[Path, str]:
    """在基础目录下创建带递增编号与时间戳的新输出目录。"""
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


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2 or not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    eroded = np.ones_like(mask, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            eroded &= padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
    return mask.astype(bool) & ~eroded


def _snr_to_confidence(snr: float) -> float:
    score = np.log1p(max(float(snr), 1e-6)) / np.log1p(10.0)
    return float(np.clip(score, 0.0, 1.0))


def _weighted_choice(confidence: np.ndarray, count: int) -> np.ndarray:
    confidence = np.asarray(confidence, dtype=float)
    confidence = np.clip(confidence, 1e-6, None)
    confidence = confidence / np.sum(confidence)
    return np.random.choice(len(confidence), size=count, replace=False, p=confidence)


def samples_to_pointcloud_with_confidence(
    samples: List[ScanSample], *,
    pixel_spacing: float = 1.0,
    slice_thickness: float = 1.0,
    intensity_threshold: float | str = "auto",
    min_contour_area: float = 50.0,
    sample_spacing: float = 2.0,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """提取点云并返回点级置信度及摘要统计。"""
    world_points = []
    point_confidences = []
    warned_no_cv2 = False
    total_slices = 0
    extracted_slices = 0
    sample_snrs = [float(sample.snr) for sample in samples]

    for s in samples:
        vol = np.asarray(s.volume_slice, dtype=float)
        if vol.ndim == 2:
            vol = vol[np.newaxis, ...]
        D, H, W = vol.shape
        total_slices += D
        snr_score = _snr_to_confidence(s.snr)

        if intensity_threshold == "auto":
            thr = float(np.mean(vol) + 0.5 * np.std(vol))
        else:
            thr = float(intensity_threshold)

        for d in range(D):
            frame = vol[d]
            if cv2 is None:
                if not warned_no_cv2:
                    print(f"[PointCloud] OpenCV(cv2) 不可用，使用边界降级提取 | python={sys.executable} | error={_cv2_import_error!r}")
                    warned_no_cv2 = True
                mask = frame > thr
                boundary = _binary_boundary(mask)
                if not np.any(boundary):
                    continue
                pts2d = np.argwhere(boundary)[:, ::-1].astype(float)
                step = max(1, int(round(sample_spacing)))
                sampled = pts2d[::step]
                sampled_sets = [sampled] if sampled.size != 0 else []
                geometry_score = float(np.clip(np.count_nonzero(boundary) / max(boundary.size * 0.03, 1.0), 0.0, 1.0))
            else:
                bw = (frame > thr).astype("uint8") * 255
                if bw.sum() == 0:
                    continue
                kernel = np.ones((3, 3), np.uint8)
                bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
                bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not contours:
                    continue
                sampled_sets = []
                geometry_score = 0.0
                for ctr in contours:
                    area = cv2.contourArea(ctr)
                    if area < min_contour_area:
                        continue
                    pts2d = ctr.squeeze().astype(float)
                    sampled = _uniform_sample_contour(pts2d, spacing=sample_spacing)
                    if sampled.size != 0:
                        sampled_sets.append(sampled)
                        area_ratio = float(area / max(H * W, 1))
                        geometry_score = max(geometry_score, float(np.clip(area_ratio / 0.2, 0.0, 1.0)))

            if not sampled_sets:
                continue

            extracted_slices += 1
            confidence = float(np.clip(0.7 * snr_score + 0.3 * geometry_score, 0.0, 1.0))
            for sampled in sampled_sets:
                xs = (sampled[:, 0] - (W - 1) / 2.0) * pixel_spacing
                ys = (sampled[:, 1] - (H - 1) / 2.0) * pixel_spacing
                zs = np.full((sampled.shape[0],), (d - (D - 1) / 2.0) * slice_thickness)
                coords_local = np.column_stack([xs, ys, zs])

                R_mat = np.asarray(s.orientation, dtype=float)
                t_vec = np.asarray(s.position, dtype=float)
                pts_world = (R_mat @ coords_local.T).T + t_vec.reshape(1, 3)
                world_points.append(pts_world)
                point_confidences.append(np.full((pts_world.shape[0],), confidence, dtype=float))

    if not world_points:
        stats = {
            "raw_point_count": 0,
            "exported_point_count": 0,
            "mean_confidence": 0.0,
            "mean_sample_snr": float(np.mean(sample_snrs)) if sample_snrs else 0.0,
            "extracted_slice_ratio": 0.0,
        }
        return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float), stats

    pts = np.vstack(world_points)
    confidence = np.concatenate(point_confidences, axis=0)
    raw_point_count = int(pts.shape[0])
    if max_points is not None and pts.shape[0] > max_points:
        idx = _weighted_choice(confidence, max_points)
        pts = pts[idx]
        confidence = confidence[idx]

    stats = {
        "raw_point_count": raw_point_count,
        "exported_point_count": int(pts.shape[0]),
        "mean_confidence": float(np.mean(confidence)) if len(confidence) else 0.0,
        "mean_sample_snr": float(np.mean(sample_snrs)) if sample_snrs else 0.0,
        "extracted_slice_ratio": float(extracted_slices / max(total_slices, 1)),
    }
    return pts, confidence, stats


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
    pts, _, _ = samples_to_pointcloud_with_confidence(
        samples,
        pixel_spacing=pixel_spacing,
        slice_thickness=slice_thickness,
        intensity_threshold=intensity_threshold,
        min_contour_area=min_contour_area,
        sample_spacing=sample_spacing,
        max_points=max_points,
    )
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


def build_pointclouds_from_phase_bins(
    phase_bins: List[PhaseBin],
    *,
    pointcloud_config: PointCloudConfig | None = None,
    out_dir: str | Path | None = None,
    pixel_spacing: float | None = None,
    slice_thickness: float | None = None,
    intensity_threshold: float | str | None = None,
    min_contour_area: float | None = None,
    sample_spacing: float | None = None,
    max_points_per_phase: int | None = None,
) -> tuple[list[Path], list[PointCloudPhaseSummary]]:
    """为每个相位分箱生成点云文件并返回写入路径列表和摘要。"""
    cfg = pointcloud_config or PointCloudConfig()
    base_out_dir = Path(out_dir if out_dir is not None else cfg.out_dir)
    pixel_spacing = float(pixel_spacing if pixel_spacing is not None else cfg.pixel_spacing)
    slice_thickness = float(slice_thickness if slice_thickness is not None else cfg.slice_thickness)
    intensity_threshold = intensity_threshold if intensity_threshold is not None else cfg.intensity_threshold
    min_contour_area = float(min_contour_area if min_contour_area is not None else cfg.min_contour_area)
    sample_spacing = float(sample_spacing if sample_spacing is not None else cfg.sample_spacing)
    max_points_per_phase = max_points_per_phase if max_points_per_phase is not None else cfg.max_points_per_phase

    out_dir, run_id = _create_indexed_output_dir(base_out_dir)
    print(f"[PointCloud] 本次输出目录: {out_dir}")

    written: list[Path] = []
    summaries: list[PointCloudPhaseSummary] = []
    for idx, bin in enumerate(phase_bins):
        samples = bin.samples
        if not samples:
            print(f"[PointCloud] phase {idx} (center={bin.phase_center:.3f}): no samples, skip")
            summaries.append(
                PointCloudPhaseSummary(
                    phase_index=idx,
                    phase_center=float(bin.phase_center),
                    sample_count=0,
                    raw_point_count=0,
                    exported_point_count=0,
                    mean_confidence=0.0,
                    mean_sample_snr=0.0,
                    extracted_slice_ratio=0.0,
                    pointcloud_path=None,
                )
            )
            continue
        pts, _, stats = samples_to_pointcloud_with_confidence(
            samples,
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
            intensity_threshold=intensity_threshold,
            min_contour_area=min_contour_area,
            sample_spacing=sample_spacing,
            max_points=max_points_per_phase,
        )
        if pts.size == 0:
            print(f"[PointCloud] phase {idx} (center={bin.phase_center:.3f}): no points extracted")
            summaries.append(
                PointCloudPhaseSummary(
                    phase_index=idx,
                    phase_center=float(bin.phase_center),
                    sample_count=len(samples),
                    raw_point_count=int(stats["raw_point_count"]),
                    exported_point_count=0,
                    mean_confidence=float(stats["mean_confidence"]),
                    mean_sample_snr=float(stats["mean_sample_snr"]),
                    extracted_slice_ratio=float(stats["extracted_slice_ratio"]),
                    pointcloud_path=None,
                )
            )
            continue
        out_path = out_dir / f"run_{run_id}_phase_{idx:03d}_{bin.phase_center:.3f}.ply"
        write_ply(pts, str(out_path))
        summary = PointCloudPhaseSummary(
            phase_index=idx,
            phase_center=float(bin.phase_center),
            sample_count=len(samples),
            raw_point_count=int(stats["raw_point_count"]),
            exported_point_count=int(stats["exported_point_count"]),
            mean_confidence=float(stats["mean_confidence"]),
            mean_sample_snr=float(stats["mean_sample_snr"]),
            extracted_slice_ratio=float(stats["extracted_slice_ratio"]),
            pointcloud_path=out_path,
        )
        print(
            f"[PointCloud] 写入 {out_path} ({pts.shape[0]} pts, "
            f"conf={summary.mean_confidence:.3f}, slice_ratio={summary.extracted_slice_ratio:.3f})"
        )
        written.append(out_path)
        summaries.append(summary)

    summary_path = out_dir / "pointcloud_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "phase_index",
            "phase_center",
            "sample_count",
            "raw_point_count",
            "exported_point_count",
            "mean_confidence",
            "mean_sample_snr",
            "extracted_slice_ratio",
            "pointcloud_path",
        ])
        for summary in summaries:
            writer.writerow([
                summary.phase_index,
                f"{summary.phase_center:.6f}",
                summary.sample_count,
                summary.raw_point_count,
                summary.exported_point_count,
                f"{summary.mean_confidence:.6f}",
                f"{summary.mean_sample_snr:.6f}",
                f"{summary.extracted_slice_ratio:.6f}",
                "" if summary.pointcloud_path is None else summary.pointcloud_path.name,
            ])
    print(f"[PointCloud] 统计摘要已写入 {summary_path}")
    return written, summaries


if __name__ == "__main__":
    print("pointcloud_builder: module provides `build_pointclouds_from_phase_bins` to export per-phase PLY files.")
