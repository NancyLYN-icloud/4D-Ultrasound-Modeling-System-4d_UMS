from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_phase_sequence_models import (
    CycleModelConfig,
    HybridRemeshConfig,
    generate_phase_models_for_instance,
)
from src.gastro4d_gpu_layout import select_grouped_reference_pointclouds


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grouped phase-sequence models for the GPU dataset pipeline.")
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--instances", nargs="*", default=None)
    parser.add_argument("--phase-count", type=int, default=None)
    parser.add_argument("--narrowing-scale", type=float, default=1.0)
    parser.add_argument("--monitor-path", type=Path, default=None)
    parser.add_argument("--base-mesh-path", type=Path, default=None)
    parser.add_argument("--preserve-provided-base-mesh", action="store_true")
    parser.add_argument("--no-sync-gt", action="store_true")
    parser.add_argument("--grid-resolution", type=int, default=72)
    parser.add_argument("--base-smooth-iterations", type=int, default=12)
    parser.add_argument("--centerline-samples", type=int, default=200)
    parser.add_argument("--body-contraction", type=float, default=0.18)
    parser.add_argument("--pylorus-contraction", type=float, default=0.10)
    parser.add_argument("--wave-width", type=float, default=0.24)
    parser.add_argument("--wave-start-u", type=float, default=0.18)
    parser.add_argument("--wave-end-u", type=float, default=0.92)
    parser.add_argument("--deformation-smooth-iterations", type=int, default=16)
    parser.add_argument("--deformation-smooth-relax", type=float, default=0.40)
    parser.add_argument("--post-smooth-iterations", type=int, default=10)
    parser.add_argument("--post-smooth-neighborhood-order", type=int, default=2)
    parser.add_argument("--reverse-centerline", action="store_true")
    parser.add_argument("--hybrid-strong-phase-remesh", action="store_true")
    parser.add_argument("--hybrid-remesh-peak-ratio-threshold", type=float, default=0.92)
    parser.add_argument("--hybrid-remesh-neighbor-span", type=int, default=1)
    args = parser.parse_args()

    records = select_grouped_reference_pointclouds(groups=args.groups, instances=args.instances)
    if not records:
        raise FileNotFoundError("No grouped reference point clouds matched the requested filters")

    cycle_model_config = CycleModelConfig(
        grid_resolution=args.grid_resolution,
        base_smooth_iterations=args.base_smooth_iterations,
        centerline_samples=args.centerline_samples,
        body_contraction=args.body_contraction,
        pylorus_contraction=args.pylorus_contraction,
        wave_width=args.wave_width,
        wave_start_u=args.wave_start_u,
        wave_end_u=args.wave_end_u,
        deformation_smooth_iterations=args.deformation_smooth_iterations,
        deformation_smooth_relax=args.deformation_smooth_relax,
        post_smooth_iterations=args.post_smooth_iterations,
        post_smooth_neighborhood_order=args.post_smooth_neighborhood_order,
        reverse_centerline=args.reverse_centerline,
    )
    hybrid_remesh = HybridRemeshConfig(
        enabled=bool(args.hybrid_strong_phase_remesh),
        peak_ratio_threshold=float(args.hybrid_remesh_peak_ratio_threshold),
        neighbor_span=int(args.hybrid_remesh_neighbor_span),
    )

    monitor_path = args.monitor_path.expanduser().resolve() if args.monitor_path else None
    base_mesh_path = args.base_mesh_path.expanduser().resolve() if args.base_mesh_path else None
    for record in records:
        generate_phase_models_for_instance(
            instance_name=record.instance_name,
            reference_ply=record.reference_ply,
            monitor_stream=monitor_path or record.sim_monitor_stream,
            output_base_dir=record.phase_model_base_dir,
            gt_mesh_dir=record.gt_mesh_dir,
            base_mesh_path=base_mesh_path,
            preserve_provided_base_mesh=args.preserve_provided_base_mesh,
            phase_count_override=args.phase_count,
            narrowing_scale=args.narrowing_scale,
            cycle_model_config=cycle_model_config,
            hybrid_remesh=hybrid_remesh,
            sync_gt=not args.no_sync_gt,
        )
        print(f"[PhaseModelsGPU] {record.instance_name} -> {record.phase_model_base_dir}")


if __name__ == "__main__":
    main()
