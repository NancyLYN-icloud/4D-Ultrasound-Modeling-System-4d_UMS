from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import data_path


REFERENCE_ROOT = data_path("stomach_pcd")
BENCHMARK_ROOT = data_path("benchmark")
SIM_ROOT = data_path("simuilate_data")


@dataclass(frozen=True)
class GroupedReferencePaths:
    instance_name: str
    source_group: str
    split: str
    reference_ply: Path
    clean_root: Path
    phase_model_base_dir: Path
    gt_mesh_dir: Path

    @property
    def monitor_stream(self) -> Path:
        return self.clean_root / "monitor_stream.npz"

    @property
    def legacy_sim_monitor_stream(self) -> Path:
        return self.phase_model_base_dir / "monitor_stream.npz"

    @property
    def scanner_sequence(self) -> Path:
        return self.clean_root / "scanner_sequence.npz"

    @property
    def monitor_image_dir(self) -> Path:
        return self.clean_root / "image" / "monitor"

    @property
    def legacy_sim_monitor_image_dir(self) -> Path:
        return self.phase_model_base_dir / "image" / "monitor"

    @property
    def scanner_image_dir(self) -> Path:
        return self.clean_root / "image" / "scanner"

    @property
    def resolved_phase_monitor_stream(self) -> Path:
        if self.monitor_stream.exists():
            return self.monitor_stream
        return self.legacy_sim_monitor_stream

    @property
    def resolved_phase_monitor_image_dir(self) -> Path:
        if self.monitor_image_dir.exists():
            return self.monitor_image_dir
        return self.legacy_sim_monitor_image_dir


def infer_split(group_name: str) -> str:
    lowered = group_name.lower()
    if "dev" in lowered:
        return "dev"
    return "test"


def grouped_instance_clean_root(split: str, source_group: str, instance_name: str) -> Path:
    return BENCHMARK_ROOT / "instances" / split / source_group / instance_name


def grouped_instance_phase_root(split: str, source_group: str, instance_name: str) -> Path:
    return SIM_ROOT / "instances" / split / source_group / instance_name


def grouped_condition_root(condition_root: Path, condition_slug: str, split: str, source_group: str, instance_name: str) -> Path:
    return condition_root / condition_slug / split / source_group / instance_name


def iter_grouped_reference_pointclouds(reference_root: Path | None = None, groups: list[str] | None = None) -> list[GroupedReferencePaths]:
    root = (reference_root or REFERENCE_ROOT).expanduser().resolve()
    if not root.exists():
        return []

    requested_groups = set(groups) if groups else None
    records: list[GroupedReferencePaths] = []
    for group_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if requested_groups is not None and group_dir.name not in requested_groups:
            continue
        split = infer_split(group_dir.name)
        for reference_ply in sorted(path for path in group_dir.glob("*.ply") if path.is_file()):
            instance_name = reference_ply.stem
            records.append(
                GroupedReferencePaths(
                    instance_name=instance_name,
                    source_group=group_dir.name,
                    split=split,
                    reference_ply=reference_ply.resolve(),
                    clean_root=grouped_instance_clean_root(split, group_dir.name, instance_name),
                    phase_model_base_dir=grouped_instance_phase_root(split, group_dir.name, instance_name),
                    gt_mesh_dir=grouped_instance_phase_root(split, group_dir.name, instance_name) / "meshes",
                )
            )
    return records


def select_grouped_reference_pointclouds(
    groups: list[str] | None = None,
    instances: list[str] | None = None,
    reference_root: Path | None = None,
) -> list[GroupedReferencePaths]:
    instance_filter = set(instances) if instances else None
    records = iter_grouped_reference_pointclouds(reference_root=reference_root, groups=groups)
    if instance_filter is None:
        return records
    return [record for record in records if record.instance_name in instance_filter]
