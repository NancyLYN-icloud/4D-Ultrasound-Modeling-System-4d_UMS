from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_monitor_stream import generate_monitor_dataset
from src.gastro4d_gpu_layout import select_grouped_reference_pointclouds


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grouped clean monitor_stream assets for the GPU dataset pipeline.")
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--instances", nargs="*", default=None)
    args = parser.parse_args()

    records = select_grouped_reference_pointclouds(groups=args.groups, instances=args.instances)
    if not records:
        raise FileNotFoundError("No grouped reference point clouds matched the requested filters")

    for record in records:
        generate_monitor_dataset(
            output_npz=record.monitor_stream,
            output_img_dir=record.monitor_image_dir,
            instance_name=record.instance_name,
            reference_ply=record.reference_ply,
        )
        print(f"[MonitorGPU] {record.instance_name} -> {record.monitor_stream}")


if __name__ == "__main__":
    main()
