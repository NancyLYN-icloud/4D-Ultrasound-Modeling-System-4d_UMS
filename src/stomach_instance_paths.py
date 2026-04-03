"""Helpers for resolving single- and multi-instance stomach dataset paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import data_path


DEFAULT_INSTANCE_NAME = "niujiao01"
REFERENCE_DIR = data_path("stomach_pcd")
LEGACY_REFERENCE_PLY = data_path("benchmark", "stomach.ply")
DEFAULT_TEST_ROOT = data_path("benchmark")
INSTANCE_TEST_ROOT = data_path("benchmark", "instances")
DEFAULT_SIM_ROOT = data_path("simuilate_data")
INSTANCE_SIM_ROOT = data_path("simuilate_data", "instances")


@dataclass(frozen=True)
class StomachInstancePaths:
    name: str
    reference_ply: Path
    test_root: Path
    monitor_stream: Path
    scanner_sequence: Path
    scanner_image_dir: Path
    processed_dir: Path
    preview_dir: Path
    phase_model_base_dir: Path
    gt_mesh_dir: Path


def list_reference_pointclouds() -> list[Path]:
    refs = sorted(path for path in REFERENCE_DIR.glob("*.ply") if path.is_file())
    if refs:
        return refs
    if LEGACY_REFERENCE_PLY.exists():
        return [LEGACY_REFERENCE_PLY]
    return []


def normalize_instance_name(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("instance name must not be empty")
    return cleaned


def default_reference_ply() -> Path:
    preferred = REFERENCE_DIR / f"{DEFAULT_INSTANCE_NAME}.ply"
    if preferred.exists():
        return preferred
    if LEGACY_REFERENCE_PLY.exists():
        return LEGACY_REFERENCE_PLY
    refs = list_reference_pointclouds()
    if refs:
        return refs[0]
    raise FileNotFoundError(f"No stomach reference point cloud found under {REFERENCE_DIR}")


def _instance_test_root(name: str) -> Path:
    if name == DEFAULT_INSTANCE_NAME:
        return DEFAULT_TEST_ROOT
    return INSTANCE_TEST_ROOT / name


def _instance_sim_root(name: str) -> Path:
    if name == DEFAULT_INSTANCE_NAME:
        return DEFAULT_SIM_ROOT
    return INSTANCE_SIM_ROOT / name


def resolve_instance_paths(instance_name: str | None = None, reference_ply: Path | None = None) -> StomachInstancePaths:
    if reference_ply is None:
        if instance_name is None:
            reference_ply = default_reference_ply()
        else:
            candidate = REFERENCE_DIR / f"{normalize_instance_name(instance_name)}.ply"
            if candidate.exists():
                reference_ply = candidate
            elif normalize_instance_name(instance_name) == DEFAULT_INSTANCE_NAME and LEGACY_REFERENCE_PLY.exists():
                reference_ply = LEGACY_REFERENCE_PLY
            else:
                raise FileNotFoundError(f"Reference point cloud not found for instance {instance_name}: {candidate}")
    else:
        reference_ply = Path(reference_ply).expanduser().resolve()

    name = normalize_instance_name(instance_name or reference_ply.stem)
    test_root = _instance_test_root(name)
    sim_root = _instance_sim_root(name)
    return StomachInstancePaths(
        name=name,
        reference_ply=reference_ply,
        test_root=test_root,
        monitor_stream=test_root / "monitor_stream.npz",
        scanner_sequence=test_root / "scanner_sequence.npz",
        scanner_image_dir=test_root / "image" / "scanner",
        processed_dir=test_root / "processed",
        preview_dir=test_root / "preview",
        phase_model_base_dir=sim_root,
        gt_mesh_dir=sim_root / "meshes",
    )


def shared_monitor_stream_path() -> Path:
    return DEFAULT_TEST_ROOT / "monitor_stream.npz"


def shared_scanner_sequence_path() -> Path:
    return DEFAULT_TEST_ROOT / "scanner_sequence.npz"


def resolve_monitor_input_path(paths: StomachInstancePaths, explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve()
    if paths.monitor_stream.exists():
        return paths.monitor_stream
    return shared_monitor_stream_path()


def resolve_scanner_template_path(paths: StomachInstancePaths, explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve()
    if paths.scanner_sequence.exists():
        return paths.scanner_sequence
    return shared_scanner_sequence_path()


def resolve_gt_mesh_input_path(paths: StomachInstancePaths, explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve()
    if paths.gt_mesh_dir.exists():
        return paths.gt_mesh_dir

    phase_model_dirs = sorted(
        path for path in paths.phase_model_base_dir.iterdir()
        if path.is_dir() and path.name.startswith("phase_sequence_models_run_")
    ) if paths.phase_model_base_dir.exists() else []
    for phase_model_dir in reversed(phase_model_dirs):
        candidate = phase_model_dir / "pointclouds" / "meshes"
        if candidate.exists():
            return candidate

    return paths.gt_mesh_dir