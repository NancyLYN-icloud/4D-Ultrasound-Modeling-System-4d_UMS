from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stomach_peristaltic_axis import build_peristaltic_axis_model, project_world_points_to_u
from src.stomach_instance_paths import resolve_instance_paths


HIST_BINS = 24
CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 1000
MARGIN = 70


def _load_positions(scanner_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(scanner_path) as data:
        timestamps = data["timestamps"].astype(np.float64)
        positions = data["positions"].astype(np.float64)
    return timestamps, positions


def _region_labels(axis_u: np.ndarray, head_max: float, tail_min: float) -> np.ndarray:
    labels = np.full(axis_u.shape[0], "body", dtype=object)
    labels[axis_u < head_max] = "head"
    labels[axis_u >= tail_min] = "tail"
    return labels


def _summary_dict(axis_u: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    total = int(axis_u.size)
    quantiles = np.quantile(axis_u, [0.05, 0.25, 0.5, 0.75, 0.95]).astype(float)
    region_counts = {region: int(np.sum(labels == region)) for region in ("head", "body", "tail")}
    region_ratios = {region: float(count / max(total, 1)) for region, count in region_counts.items()}
    return {
        "frame_count": total,
        "mean_u": float(np.mean(axis_u)),
        "std_u": float(np.std(axis_u)),
        "min_u": float(np.min(axis_u)),
        "max_u": float(np.max(axis_u)),
        "quantiles": {
            "p05": quantiles[0],
            "p25": quantiles[1],
            "p50": quantiles[2],
            "p75": quantiles[3],
            "p95": quantiles[4],
        },
        "region_counts": region_counts,
        "region_ratios": region_ratios,
    }


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    draw.text(xy, text, fill=fill, font=font)


def _draw_hist_panel(
    draw: ImageDraw.ImageDraw,
    area: tuple[int, int, int, int],
    baseline_hist: np.ndarray,
    variant_hist: np.ndarray,
    baseline_label: str,
    variant_label: str,
    font: ImageFont.ImageFont,
    title_font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = area
    width = right - left
    height = bottom - top
    axis_y = bottom - 35
    axis_x = left + 55
    usable_w = width - 75
    usable_h = height - 70
    max_count = max(int(np.max(baseline_hist)), int(np.max(variant_hist)), 1)

    draw.rounded_rectangle(area, radius=18, outline=(215, 215, 215), width=2, fill=(252, 252, 252))
    _draw_text(draw, (left + 18, top + 14), "Peristaltic-Axis Slice Distribution", fill=(20, 20, 20), font=title_font)
    draw.line((axis_x, top + 40, axis_x, axis_y), fill=(80, 80, 80), width=2)
    draw.line((axis_x, axis_y, right - 20, axis_y), fill=(80, 80, 80), width=2)

    bin_count = baseline_hist.size
    group_w = usable_w / bin_count
    bar_w = max(4.0, group_w * 0.34)
    for idx in range(bin_count):
        x0 = axis_x + idx * group_w + group_w * 0.14
        base_h = usable_h * (baseline_hist[idx] / max_count)
        var_h = usable_h * (variant_hist[idx] / max_count)
        draw.rectangle((x0, axis_y - base_h, x0 + bar_w, axis_y), fill=(84, 130, 184))
        draw.rectangle((x0 + bar_w + group_w * 0.10, axis_y - var_h, x0 + 2 * bar_w + group_w * 0.10, axis_y), fill=(222, 124, 93))
        if idx < bin_count - 1:
            draw.line((axis_x + (idx + 1) * group_w, top + 40, axis_x + (idx + 1) * group_w, axis_y), fill=(236, 236, 236), width=1)

    for tick in range(5):
        frac = tick / 4.0
        y = axis_y - usable_h * frac
        value = int(round(max_count * frac))
        draw.line((axis_x - 6, y, axis_x, y), fill=(80, 80, 80), width=2)
        _draw_text(draw, (left + 6, int(y - 8)), f"{value}", fill=(80, 80, 80), font=font)

    for tick in range(7):
        frac = tick / 6.0
        x = axis_x + usable_w * frac
        draw.line((x, axis_y, x, axis_y + 6), fill=(80, 80, 80), width=2)
        _draw_text(draw, (int(x - 12), axis_y + 10), f"{frac:.1f}", fill=(80, 80, 80), font=font)

    legend_y = top + 14
    draw.rectangle((right - 265, legend_y + 2, right - 245, legend_y + 22), fill=(84, 130, 184))
    _draw_text(draw, (right - 238, legend_y), baseline_label, fill=(30, 30, 30), font=font)
    draw.rectangle((right - 145, legend_y + 2, right - 125, legend_y + 22), fill=(222, 124, 93))
    _draw_text(draw, (right - 118, legend_y), variant_label, fill=(30, 30, 30), font=font)


def _draw_region_panel(
    draw: ImageDraw.ImageDraw,
    area: tuple[int, int, int, int],
    baseline_counts: dict[str, int],
    variant_counts: dict[str, int],
    baseline_ratios: dict[str, float],
    variant_ratios: dict[str, float],
    baseline_label: str,
    variant_label: str,
    font: ImageFont.ImageFont,
    title_font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = area
    width = right - left
    height = bottom - top
    axis_y = bottom - 55
    axis_x = left + 65
    usable_w = width - 90
    usable_h = height - 110
    max_count = max(max(baseline_counts.values()), max(variant_counts.values()), 1)
    regions = ["head", "body", "tail"]

    draw.rounded_rectangle(area, radius=18, outline=(215, 215, 215), width=2, fill=(252, 252, 252))
    _draw_text(draw, (left + 18, top + 14), "Head / Body / Tail Counts", fill=(20, 20, 20), font=title_font)
    draw.line((axis_x, top + 40, axis_x, axis_y), fill=(80, 80, 80), width=2)
    draw.line((axis_x, axis_y, right - 20, axis_y), fill=(80, 80, 80), width=2)

    group_w = usable_w / len(regions)
    bar_w = min(55.0, group_w * 0.28)
    for idx, region in enumerate(regions):
        x0 = axis_x + idx * group_w + group_w * 0.18
        base_h = usable_h * (baseline_counts[region] / max_count)
        var_h = usable_h * (variant_counts[region] / max_count)
        draw.rectangle((x0, axis_y - base_h, x0 + bar_w, axis_y), fill=(84, 130, 184))
        draw.rectangle((x0 + bar_w + group_w * 0.12, axis_y - var_h, x0 + 2 * bar_w + group_w * 0.12, axis_y), fill=(222, 124, 93))
        _draw_text(draw, (int(x0 + 8), axis_y + 12), region, fill=(40, 40, 40), font=font)
        _draw_text(
            draw,
            (int(x0 - 8), int(axis_y - base_h - 24)),
            f"{baseline_counts[region]} ({baseline_ratios[region] * 100:.1f}%)",
            fill=(56, 92, 134),
            font=font,
        )
        _draw_text(
            draw,
            (int(x0 - 8), int(axis_y - var_h - 42)),
            f"{variant_counts[region]} ({variant_ratios[region] * 100:.1f}%)",
            fill=(163, 82, 55),
            font=font,
        )

    _draw_text(draw, (left + 18, bottom - 35), f"Blue: {baseline_label}", fill=(56, 92, 134), font=font)
    _draw_text(draw, (left + 250, bottom - 35), f"Orange: {variant_label}", fill=(163, 82, 55), font=font)


def _save_plot(
    output_path: Path,
    baseline_hist: np.ndarray,
    variant_hist: np.ndarray,
    baseline_summary: dict[str, object],
    variant_summary: dict[str, object],
    baseline_label: str,
    variant_label: str,
) -> None:
    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), (246, 244, 240))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    _draw_text(draw, (MARGIN, 24), "Scanner Coverage Diagnostic", fill=(18, 18, 18), font=title_font)
    _draw_text(
        draw,
        (MARGIN, 48),
        f"Comparing peristaltic-axis slice density for {baseline_label} vs {variant_label}",
        fill=(72, 72, 72),
        font=font,
    )

    _draw_hist_panel(
        draw,
        (MARGIN, 90, CANVAS_WIDTH - MARGIN, 520),
        baseline_hist,
        variant_hist,
        baseline_label,
        variant_label,
        font,
        title_font,
    )
    _draw_region_panel(
        draw,
        (MARGIN, 560, CANVAS_WIDTH - MARGIN, CANVAS_HEIGHT - MARGIN),
        baseline_summary["region_counts"],
        variant_summary["region_counts"],
        baseline_summary["region_ratios"],
        variant_summary["region_ratios"],
        baseline_label,
        variant_label,
        font,
        title_font,
    )
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old and new scanner head/body/tail coverage")
    parser.add_argument("--instance-name", required=True, type=str)
    parser.add_argument("--baseline-scanner", required=True, type=Path)
    parser.add_argument("--variant-scanner", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-label", type=str, default="baseline")
    parser.add_argument("--variant-label", type=str, default="headtail-dense")
    parser.add_argument("--head-max", type=float, default=0.20)
    parser.add_argument("--tail-min", type=float, default=0.80)
    parser.add_argument("--hist-bins", type=int, default=HIST_BINS)
    args = parser.parse_args()

    resolve_instance_paths(instance_name=args.instance_name)
    axis_model = build_peristaltic_axis_model(instance_name=args.instance_name)

    _, baseline_positions = _load_positions(args.baseline_scanner.expanduser().resolve())
    _, variant_positions = _load_positions(args.variant_scanner.expanduser().resolve())

    baseline_u = project_world_points_to_u(axis_model, baseline_positions)
    variant_u = project_world_points_to_u(axis_model, variant_positions)

    baseline_labels = _region_labels(baseline_u, args.head_max, args.tail_min)
    variant_labels = _region_labels(variant_u, args.head_max, args.tail_min)

    hist_edges = np.linspace(0.0, 1.0, args.hist_bins + 1, dtype=np.float64)
    baseline_hist, _ = np.histogram(baseline_u, bins=hist_edges)
    variant_hist, _ = np.histogram(variant_u, bins=hist_edges)

    baseline_summary = _summary_dict(baseline_u, baseline_labels)
    variant_summary = _summary_dict(variant_u, variant_labels)
    delta_summary = {
        region: float(variant_summary["region_ratios"][region] - baseline_summary["region_ratios"][region])
        for region in ("head", "body", "tail")
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{args.instance_name}_scanner_headtail_coverage.png"
    summary_path = output_dir / f"{args.instance_name}_scanner_headtail_coverage_summary.json"
    hist_path = output_dir / f"{args.instance_name}_scanner_headtail_histogram.csv"

    _save_plot(
        output_path=image_path,
        baseline_hist=baseline_hist,
        variant_hist=variant_hist,
        baseline_summary=baseline_summary,
        variant_summary=variant_summary,
        baseline_label=args.baseline_label,
        variant_label=args.variant_label,
    )

    payload = {
        "instance_name": args.instance_name,
        "baseline_scanner": str(args.baseline_scanner.expanduser().resolve()),
        "variant_scanner": str(args.variant_scanner.expanduser().resolve()),
        "head_max": args.head_max,
        "tail_min": args.tail_min,
        "baseline": baseline_summary,
        "variant": variant_summary,
        "delta_region_ratio": delta_summary,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    hist_lines = ["bin_left,bin_right,baseline_count,variant_count"]
    for idx in range(args.hist_bins):
        hist_lines.append(
            f"{hist_edges[idx]:.6f},{hist_edges[idx + 1]:.6f},{int(baseline_hist[idx])},{int(variant_hist[idx])}"
        )
    hist_path.write_text("\n".join(hist_lines) + "\n", encoding="utf-8")

    print(f"[ScannerCoverage] image: {image_path}")
    print(f"[ScannerCoverage] summary: {summary_path}")
    print(f"[ScannerCoverage] histogram: {hist_path}")
    print(
        "[ScannerCoverage] head/body/tail delta ratio: "
        f"head={delta_summary['head']:+.4f}, body={delta_summary['body']:+.4f}, tail={delta_summary['tail']:+.4f}"
    )


if __name__ == "__main__":
    main()