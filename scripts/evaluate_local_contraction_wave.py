from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


PHASE_PATTERNS = [
    re.compile(r"run_\d+_phase_(\d+)_([0-9]+\.[0-9]+)_t"),
    re.compile(r"run_\d+_phase_(\d+)_([0-9]+\.[0-9]+)"),
    re.compile(r"dynamic_phase_([0-9]+\.[0-9]+)\.ply$"),
]


@dataclass
class SequenceItem:
    path: Path
    mesh: trimesh.Trimesh
    phase_index: int | None
    phase: float | None


@dataclass
class WavePhaseSummary:
    phase_index: int
    phase_value: float
    timestamp_s: float
    wave_center_s: float
    max_contraction: float


def _parse_phase(path: Path) -> tuple[int | None, float | None]:
    name = path.name
    match = PHASE_PATTERNS[0].search(name)
    if match:
        return int(match.group(1)), float(match.group(2))
    match = PHASE_PATTERNS[1].search(name)
    if match:
        return int(match.group(1)), float(match.group(2))
    match = PHASE_PATTERNS[2].search(name)
    if match:
        return None, float(match.group(1))
    return None, None


def _load_sequence(folder: Path) -> list[SequenceItem]:
    items: list[SequenceItem] = []
    for path in sorted(folder.glob("*.ply")):
        if path.name == "dynamic_base_mesh.ply":
            continue
        mesh = trimesh.load(path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        phase_index, phase = _parse_phase(path)
        items.append(SequenceItem(path=path, mesh=mesh, phase_index=phase_index, phase=phase))
    return items


def _load_wave_summary(path: Path) -> dict[int, WavePhaseSummary]:
    result: dict[int, WavePhaseSummary] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            phase_index = int(row["phase_index"])
            result[phase_index] = WavePhaseSummary(
                phase_index=phase_index,
                phase_value=float(row["phase_value"]),
                timestamp_s=float(row["timestamp_s"]),
                wave_center_s=float(row["wave_center_s"]),
                max_contraction=float(row["max_contraction"]),
            )
    return result


def _sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    if len(mesh.faces) > 0:
        sample_count = min(int(count), max(len(mesh.faces) * 3, 1))
        random_state = np.random.get_state()
        np.random.seed(int(seed))
        try:
            points, _ = trimesh.sample.sample_surface(mesh, sample_count)
        finally:
            np.random.set_state(random_state)
        return np.asarray(points, dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) <= count:
        return vertices
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(vertices), size=int(count), replace=False)
    return vertices[indices]


def _principal_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projections = centered @ axis
    return center, axis / max(np.linalg.norm(axis), 1e-8), projections


def _normalized_s(points: np.ndarray, center: np.ndarray, axis: np.ndarray, axis_min: float, axis_max: float) -> np.ndarray:
    projections = (points - center[None, :]) @ axis
    denom = max(axis_max - axis_min, 1e-8)
    return np.clip((projections - axis_min) / denom, 0.0, 1.0)


def _radius_profile(mesh: trimesh.Trimesh, center: np.ndarray, axis: np.ndarray, axis_min: float, axis_max: float, bins: int) -> tuple[np.ndarray, np.ndarray]:
    points = _sample_surface(mesh, count=12000, seed=17)
    s = _normalized_s(points, center, axis, axis_min, axis_max)
    projections = (points - center[None, :]) @ axis
    axial_points = center[None, :] + projections[:, None] * axis[None, :]
    radius = np.linalg.norm(points - axial_points, axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.full((bins,), np.nan, dtype=np.float64)
    for index in range(bins):
        mask = (s >= edges[index]) & (s < edges[index + 1])
        if np.any(mask):
            values[index] = float(np.mean(radius[mask]))
    return centers, values


def _pick_axis_orientation(
    gt_items: list[SequenceItem],
    summaries: dict[int, WavePhaseSummary],
    bins: int,
) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, float]]:
    ref_mesh = gt_items[0].mesh
    ref_vertices = np.asarray(ref_mesh.vertices, dtype=np.float64)
    center, axis, projections = _principal_axis(ref_vertices)
    axis_min = float(np.min(projections))
    axis_max = float(np.max(projections))
    peak_items = [item for item in gt_items if item.phase_index in summaries and summaries[item.phase_index].max_contraction >= 0.75]
    if not peak_items:
        return center, axis, axis_min, axis_max, {"orientation_score": 0.0, "flipped": 0.0}

    def orientation_score(candidate_axis: np.ndarray) -> float:
        weighted_centers: list[float] = []
        weights: list[float] = []
        for item in peak_items:
            bin_centers, radius_values = _radius_profile(item.mesh, center, candidate_axis, axis_min, axis_max, bins)
            valid = np.isfinite(radius_values)
            if not np.any(valid):
                continue
            local_centers = bin_centers[valid]
            local_values = radius_values[valid]
            min_index = int(np.argmin(local_values))
            weighted_centers.append(float(local_centers[min_index]))
            weights.append(float(summaries[item.phase_index].max_contraction))
        if not weighted_centers:
            return float("inf")
        weighted_center = float(np.average(np.asarray(weighted_centers), weights=np.asarray(weights)))
        summary_center = float(np.average(
            np.asarray([summaries[item.phase_index].wave_center_s for item in peak_items]),
            weights=np.asarray([summaries[item.phase_index].max_contraction for item in peak_items]),
        ))
        return abs(weighted_center - summary_center)

    direct_score = orientation_score(axis)
    flipped_score = orientation_score(-axis)
    chosen_axis = axis if direct_score <= flipped_score else -axis
    return center, chosen_axis, axis_min, axis_max, {
        "orientation_score": float(min(direct_score, flipped_score)),
        "flipped": float(direct_score > flipped_score),
    }


def _match_gt(gt_items: list[SequenceItem], pred_item: SequenceItem) -> SequenceItem:
    if pred_item.phase_index is not None:
        for gt in gt_items:
            if gt.phase_index == pred_item.phase_index:
                return gt
    if pred_item.phase is not None:
        candidates = [item for item in gt_items if item.phase is not None]
        return min(candidates, key=lambda item: abs(float(item.phase) - float(pred_item.phase)))
    raise ValueError(f"Unable to match GT for {pred_item.path}")


def _local_metrics(pred_points: np.ndarray, gt_points: np.ndarray) -> tuple[float, float]:
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("nan"), float("nan")
    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)
    dist_pred_to_gt, _ = gt_tree.query(pred_points, k=1)
    dist_gt_to_pred, _ = pred_tree.query(gt_points, k=1)
    cd = float(np.mean(dist_pred_to_gt ** 2) + np.mean(dist_gt_to_pred ** 2))
    hd95 = float(np.percentile(dist_pred_to_gt, 95.0))
    return cd, hd95


def _safe_mean(values: list[float]) -> float:
    valid = [value for value in values if np.isfinite(value)]
    return float(np.mean(valid)) if valid else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local geometry near the propagating ring-contraction wave")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--wave-summary", type=Path, default=None)
    parser.add_argument("--window-width", type=float, default=0.12)
    parser.add_argument("--num-samples", type=int, default=12000)
    parser.add_argument("--axis-bins", type=int, default=48)
    parser.add_argument("--high-contraction-threshold", type=float, default=0.75)
    parser.add_argument("--output-prefix", type=str, default="local_contraction_wave_eval")
    args = parser.parse_args()

    pred_items = _load_sequence(args.pred_dir)
    gt_items = _load_sequence(args.gt_dir)
    if not pred_items:
        raise ValueError(f"No predicted meshes found in {args.pred_dir}")
    if not gt_items:
        raise ValueError(f"No GT meshes found in {args.gt_dir}")

    wave_summary_path = args.wave_summary or (args.gt_dir / "phase_sequence_summary.csv")
    summaries = _load_wave_summary(wave_summary_path)
    center, axis, axis_min, axis_max, orientation_meta = _pick_axis_orientation(gt_items, summaries, bins=args.axis_bins)

    rows: list[dict[str, float | int | str]] = []
    all_local_cd: list[float] = []
    all_local_hd95: list[float] = []
    high_local_cd: list[float] = []
    high_local_hd95: list[float] = []

    half_width = 0.5 * float(args.window_width)
    for index, pred_item in enumerate(pred_items):
        gt_item = _match_gt(gt_items, pred_item)
        phase_index = gt_item.phase_index if gt_item.phase_index is not None else pred_item.phase_index
        if phase_index is None or phase_index not in summaries:
            continue
        summary = summaries[phase_index]

        pred_points = _sample_surface(pred_item.mesh, args.num_samples, seed=1000 + index * 2)
        gt_points = _sample_surface(gt_item.mesh, args.num_samples, seed=1001 + index * 2)
        pred_s = _normalized_s(pred_points, center, axis, axis_min, axis_max)
        gt_s = _normalized_s(gt_points, center, axis, axis_min, axis_max)
        low = max(0.0, summary.wave_center_s - half_width)
        high = min(1.0, summary.wave_center_s + half_width)
        pred_local = pred_points[(pred_s >= low) & (pred_s <= high)]
        gt_local = gt_points[(gt_s >= low) & (gt_s <= high)]
        local_cd, local_hd95 = _local_metrics(pred_local, gt_local)

        row = {
            "phase_index": int(phase_index),
            "phase_value": float(summary.phase_value),
            "timestamp_s": float(summary.timestamp_s),
            "wave_center_s": float(summary.wave_center_s),
            "max_contraction": float(summary.max_contraction),
            "band_low_s": float(low),
            "band_high_s": float(high),
            "pred_local_points": int(len(pred_local)),
            "gt_local_points": int(len(gt_local)),
            "local_cd": float(local_cd),
            "local_hd95": float(local_hd95),
            "pred_mesh": pred_item.path.name,
            "gt_mesh": gt_item.path.name,
        }
        rows.append(row)
        all_local_cd.append(local_cd)
        all_local_hd95.append(local_hd95)
        if summary.max_contraction >= float(args.high_contraction_threshold):
            high_local_cd.append(local_cd)
            high_local_hd95.append(local_hd95)

    summary_payload = {
        "pred_dir": str(args.pred_dir),
        "gt_dir": str(args.gt_dir),
        "wave_summary": str(wave_summary_path),
        "window_width": float(args.window_width),
        "num_samples": int(args.num_samples),
        "high_contraction_threshold": float(args.high_contraction_threshold),
        "axis_orientation": orientation_meta,
        "phase_count": int(len(rows)),
        "mean_local_cd": _safe_mean(all_local_cd),
        "mean_local_hd95": _safe_mean(all_local_hd95),
        "high_contraction_phase_count": int(sum(int(row["max_contraction"] >= float(args.high_contraction_threshold)) for row in rows)),
        "high_contraction_mean_local_cd": _safe_mean(high_local_cd),
        "high_contraction_mean_local_hd95": _safe_mean(high_local_hd95),
        "worst_local_cd_phase": max((row for row in rows if np.isfinite(float(row["local_cd"]))), key=lambda row: float(row["local_cd"]), default=None),
        "worst_local_hd95_phase": max((row for row in rows if np.isfinite(float(row["local_hd95"]))), key=lambda row: float(row["local_hd95"]), default=None),
    }

    out_csv = args.pred_dir / f"{args.output_prefix}.csv"
    out_json = args.pred_dir / f"{args.output_prefix}.json"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["phase_index"])
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    print(f"WROTE {out_csv}")
    print(f"WROTE {out_json}")


if __name__ == "__main__":
    main()