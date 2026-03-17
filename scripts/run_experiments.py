"""顶会/顶刊实验运行脚本。

自动调用主流程，比较静态基线、静态增强版与共享动态模型，并输出可直接用于论文整理的 CSV 与 LaTeX 表格。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import re
import socket
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import trimesh


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig
from src.data_acquisition.free_arm_scan import FreeArmScanner
from src.data_acquisition.monitor import UltrasoundMonitor
from src.modeling.metrics import compute_chamfer_distance, compute_hausdorff_distance, compute_temporal_smoothness
from src.paths import data_path
from src.pipelines.multicycle_reconstruction import MulticycleReconstructionPipeline, PipelineOutput


DEFAULT_MONITOR_PATH = data_path("test", "monitor_stream.npz")
DEFAULT_SCANNER_PATH = data_path("test", "scanner_sequence.npz")
DEFAULT_GT_MESH_PATH = data_path("simuilate_data", "meshes")
DEFAULT_EXPERIMENT_ROOT = data_path("experiments")

_STATIC_PHASE_PATTERN = re.compile(r"_phase_(\d+)_([0-9]+(?:\.[0-9]+)?)")
_DYNAMIC_PHASE_PATTERN = re.compile(r"dynamic_phase_([0-9]+(?:\.[0-9]+)?)")


FAST_DEV_PRESET = {
    "surface_train_steps": 24,
    "surface_mesh_resolution": 36,
    "surface_max_points": 2000,
    "dynamic_train_steps": 24,
    "dynamic_mesh_resolution": 32,
    "dynamic_max_points_per_phase": 1200,
}

DETAIL_DIAG_PRESET = {
    "surface_train_steps": 80,
    "surface_mesh_resolution": 56,
    "surface_max_points": 4000,
    "dynamic_train_steps": 180,
    "dynamic_mesh_resolution": 72,
    "dynamic_max_points_per_phase": 3000,
    "dynamic_canonical_hidden_dim": 256,
    "dynamic_deformation_hidden_dim": 256,
    "dynamic_temporal_weight": 0.02,
    "dynamic_temporal_acceleration_weight": 0.01,
    "dynamic_phase_consistency_weight": 0.01,
    "dynamic_periodicity_weight": 0.02,
    "dynamic_deformation_weight": 0.002,
    "dynamic_confidence_floor": 1.0,
}

FULL_PAPER_PRESET = {
    "surface_train_steps": 100,
    "surface_mesh_resolution": 64,
    "surface_max_points": 5000,
    "dynamic_train_steps": 140,
    "dynamic_mesh_resolution": 60,
    "dynamic_max_points_per_phase": 3500,
}


class _TeeStream:
    def __init__(self, *streams: object) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "run"


def _make_run_dir(base_dir: Path, mode: str, experiment_set: str, run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _sanitize_name(run_name) if run_name else f"{mode}_{experiment_set}"
    candidate = base_dir / f"exp_{timestamp}_{suffix}"
    index = 1
    while candidate.exists():
        candidate = base_dir / f"exp_{timestamp}_{suffix}_{index:02d}"
        index += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_run_metadata(args: argparse.Namespace, run_dir: Path, log_path: Path) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "monitor_path": str(Path(args.monitor_path).resolve()),
        "scanner_path": str(Path(args.scanner_path).resolve()),
        "gt_mesh_path": str(Path(args.gt_mesh_path).resolve()),
        "mode": args.mode,
        "experiment_set": args.experiment_set,
        "run_name": args.run_name,
        "cli_args": vars(args),
    }
    _write_json(run_dir / "run_metadata.json", metadata)


def _save_config_snapshot(config: PipelineConfig, config_dir: Path, label: str) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{_sanitize_name(label)}.json"
    _write_json(path, asdict(config))


def _safe_mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if not np.isnan(value)]
    return float(np.mean(values)) if values else float("nan")


def _extract_phase_info(path: Path) -> tuple[int | None, float | None]:
    static_match = _STATIC_PHASE_PATTERN.search(path.name)
    if static_match is not None:
        return int(static_match.group(1)), float(static_match.group(2))
    dynamic_match = _DYNAMIC_PHASE_PATTERN.search(path.name)
    if dynamic_match is not None:
        return None, float(dynamic_match.group(1))
    return None, None


def _load_geometry(path: Path) -> trimesh.Trimesh | trimesh.points.PointCloud | None:
    try:
        geometry = trimesh.load(path)
    except Exception as exc:
        print(f"警告：真值几何加载失败 {path}: {exc}")
        return None

    vertices = np.asarray(getattr(geometry, "vertices", np.empty((0, 3))))
    if vertices.size == 0:
        print(f"警告：真值几何为空 {path}，已跳过。")
        return None
    return geometry


def _load_gt_mesh(
    gt_mesh_path: Path,
) -> trimesh.Trimesh | trimesh.points.PointCloud | list[dict[str, object]] | None:
    if not gt_mesh_path.exists():
        print(f"警告：未找到真值网格 {gt_mesh_path}，跳过几何精度计算。")
        return None
    if gt_mesh_path.is_dir():
        sequence: list[dict[str, object]] = []
        for path in sorted(gt_mesh_path.glob("*.ply")):
            geometry = _load_geometry(path)
            if geometry is None:
                continue
            phase_index, phase = _extract_phase_info(path)
            sequence.append(
                {
                    "path": path,
                    "geometry": geometry,
                    "phase_index": phase_index,
                    "phase": phase,
                }
            )
        if not sequence:
            print(f"警告：目录 {gt_mesh_path} 中没有可用的真值网格，跳过几何精度计算。")
            return None
        print(f"[Experiments] 已加载 {len(sequence)} 个相位级 GT 网格: {gt_mesh_path}")
        return sequence

    return _load_geometry(gt_mesh_path)


def _load_mesh_sequence(paths: list[Path]) -> list[trimesh.Trimesh]:
    meshes: list[trimesh.Trimesh] = []
    for path in paths:
        try:
            meshes.append(trimesh.load(path, force="mesh"))
        except Exception as exc:
            print(f"警告：跳过无法加载的网格 {path}: {exc}")
    return meshes


def _match_gt_geometry(
    gt_reference: trimesh.Trimesh | trimesh.points.PointCloud | list[dict[str, object]] | None,
    mesh_path: Path,
    phase: float | None,
) -> trimesh.Trimesh | trimesh.points.PointCloud | None:
    if gt_reference is None:
        return None
    if not isinstance(gt_reference, list):
        return gt_reference

    phase_index, parsed_phase = _extract_phase_info(mesh_path)
    target_phase = phase if phase is not None else parsed_phase

    if phase_index is not None:
        for item in gt_reference:
            if item["phase_index"] == phase_index:
                return item["geometry"]  # type: ignore[return-value]

    if target_phase is not None:
        candidates = [item for item in gt_reference if item["phase"] is not None]
        if candidates:
            best = min(candidates, key=lambda item: abs(float(item["phase"]) - target_phase))
            return best["geometry"]  # type: ignore[return-value]

    return None


def _resolve_mode_preset(mode: str) -> dict[str, int]:
    if mode == "fast-dev":
        return FAST_DEV_PRESET
    if mode == "dynamic-detail":
        return DETAIL_DIAG_PRESET
    if mode == "full-paper":
        return FULL_PAPER_PRESET
    raise ValueError(f"未知模式: {mode}")


def _build_config(
    method: str,
    mode: str,
    dynamic_train_steps: int | None,
    dynamic_mesh_resolution: int | None,
    pointcloud_out_dir: Path,
    dynamic_ablation: dict[str, bool | float] | None = None,
) -> PipelineConfig:
    config = PipelineConfig()
    config.pointcloud.out_dir = str(pointcloud_out_dir)
    config.dynamic_model.supervision_binning_strategy = str(config.phase_detection.binning_strategy)
    config.dynamic_model.supervision_step_seconds = float(config.phase_detection.phase_bin_step_seconds)
    config.dynamic_model.supervision_window_seconds = float(config.phase_detection.sliding_window_seconds)
    preset = _resolve_mode_preset(mode)
    config.surface_model.train_steps = int(preset["surface_train_steps"])
    config.surface_model.mesh_resolution = int(preset["surface_mesh_resolution"])
    config.surface_model.max_points = int(preset["surface_max_points"])
    config.dynamic_model.train_steps = int(preset["dynamic_train_steps"])
    config.dynamic_model.mesh_resolution = int(preset["dynamic_mesh_resolution"])
    config.dynamic_model.max_points_per_phase = int(preset["dynamic_max_points_per_phase"])
    if "dynamic_canonical_hidden_dim" in preset:
        config.dynamic_model.canonical_hidden_dim = int(preset["dynamic_canonical_hidden_dim"])
    if "dynamic_deformation_hidden_dim" in preset:
        config.dynamic_model.deformation_hidden_dim = int(preset["dynamic_deformation_hidden_dim"])
    if "dynamic_temporal_weight" in preset:
        config.dynamic_model.temporal_weight = float(preset["dynamic_temporal_weight"])
    if "dynamic_temporal_acceleration_weight" in preset:
        config.dynamic_model.temporal_acceleration_weight = float(preset["dynamic_temporal_acceleration_weight"])
    if "dynamic_phase_consistency_weight" in preset:
        config.dynamic_model.phase_consistency_weight = float(preset["dynamic_phase_consistency_weight"])
    if "dynamic_periodicity_weight" in preset:
        config.dynamic_model.periodicity_weight = float(preset["dynamic_periodicity_weight"])
    if "dynamic_deformation_weight" in preset:
        config.dynamic_model.deformation_weight = float(preset["dynamic_deformation_weight"])
    if "dynamic_confidence_floor" in preset:
        config.dynamic_model.confidence_floor = float(preset["dynamic_confidence_floor"])

    if dynamic_train_steps is not None:
        config.dynamic_model.train_steps = int(dynamic_train_steps)
    if dynamic_mesh_resolution is not None:
        config.dynamic_model.mesh_resolution = int(dynamic_mesh_resolution)

    if method == "静态基线":
        config.phase_canonicalization.enabled = False
        config.dynamic_model.enabled = False
    elif method == "静态增强":
        config.phase_canonicalization.enabled = True
        config.dynamic_model.enabled = False
    elif method == "动态共享":
        config.phase_canonicalization.enabled = True
        config.dynamic_model.enabled = True
    else:
        raise ValueError(f"未知实验方法: {method}")

    if dynamic_ablation and config.dynamic_model.enabled:
        if bool(dynamic_ablation.get("disable_confidence_weighting", False)):
            config.dynamic_model.confidence_floor = 1.0
        if bool(dynamic_ablation.get("disable_periodicity", False)):
            config.dynamic_model.periodicity_weight = 0.0
        if bool(dynamic_ablation.get("disable_normal", False)):
            config.dynamic_model.normal_weight = 0.0
        if bool(dynamic_ablation.get("disable_acceleration", False)):
            config.dynamic_model.temporal_acceleration_weight = 0.0
        if bool(dynamic_ablation.get("disable_phase_consistency", False)):
            config.dynamic_model.phase_consistency_weight = 0.0

        for key, attr in [
            ("normal_weight", "normal_weight"),
            ("temporal_weight", "temporal_weight"),
            ("temporal_acceleration_weight", "temporal_acceleration_weight"),
            ("phase_consistency_weight", "phase_consistency_weight"),
            ("periodicity_weight", "periodicity_weight"),
            ("deformation_weight", "deformation_weight"),
        ]:
            if key in dynamic_ablation and dynamic_ablation[key] is not None:
                setattr(config.dynamic_model, attr, float(dynamic_ablation[key]))
    return config


def _run_pipeline(method: str, config: PipelineConfig, monitor_path: Path, scanner_path: Path) -> PipelineOutput:
    print(f">>> 运行方法：{method}")
    monitor = UltrasoundMonitor.from_npz(config.acquisition, str(monitor_path))
    scanner = FreeArmScanner.from_npz(config.acquisition, str(scanner_path))
    pipeline = MulticycleReconstructionPipeline(config)
    return pipeline.run(monitor, scanner)


def _evaluate_output(
    method: str,
    output: PipelineOutput,
    gt_mesh: trimesh.Trimesh | trimesh.points.PointCloud | list[dict[str, object]] | None,
) -> dict[str, float | int | str]:
    if output.dynamic_mesh_results:
        mesh_items = [(item.mesh_path, float(item.phase)) for item in output.dynamic_mesh_results]
        mesh_type = "动态共享模型"
    else:
        mesh_items = [(item.mesh_path, None) for item in output.mesh_results]
        mesh_type = "逐相位静态模型"

    meshes: list[trimesh.Trimesh] = []
    chamfers: list[float] = []
    hausdorffs: list[float] = []
    for mesh_path, phase in mesh_items:
        try:
            mesh = trimesh.load(mesh_path, force="mesh")
        except Exception as exc:
            print(f"警告：跳过无法加载的网格 {mesh_path}: {exc}")
            continue
        meshes.append(mesh)
        matched_gt = _match_gt_geometry(gt_mesh, mesh_path, phase)
        if matched_gt is None:
            continue
        chamfers.append(compute_chamfer_distance(mesh, matched_gt))
        hausdorffs.append(compute_hausdorff_distance(mesh, matched_gt))

    valid_summaries = [item for item in output.pointcloud_summaries if item.exported_point_count > 0]
    watertight_ratio = float(
        sum(int(mesh.is_watertight) for mesh in meshes) / max(len(meshes), 1)
    ) if meshes else 0.0

    return {
        "方法": method,
        "模型类型": mesh_type,
        "相位点云数": int(len(output.pointcloud_paths)),
        "网格数": int(len(meshes)),
        "平均点云置信度": _safe_mean(item.mean_confidence for item in valid_summaries),
        "平均样本SNR": _safe_mean(item.mean_sample_snr for item in valid_summaries),
        "平均切片提取率": _safe_mean(item.extracted_slice_ratio for item in valid_summaries),
        "时间平滑度(mm/step)": float(compute_temporal_smoothness(meshes)) if len(meshes) >= 2 else float("nan"),
        "水密比例": watertight_ratio,
        "平均CD(mm^2)": _safe_mean(chamfers),
        "平均HD95(mm)": _safe_mean(hausdorffs),
    }


def run_method_comparison(
    experiment_dir: Path,
    artifact_dir: Path,
    config_dir: Path,
    monitor_path: Path,
    scanner_path: Path,
    gt_mesh_path: Path,
    mode: str,
    dynamic_train_steps: int | None,
    dynamic_mesh_resolution: int | None,
) -> pd.DataFrame:
    """运行静态与动态方法对比实验。"""
    gt_mesh = _load_gt_mesh(gt_mesh_path)
    methods = ["静态基线", "静态增强", "动态共享"]
    results: list[dict[str, float | int | str]] = []

    for method in methods:
        config = _build_config(method, mode, dynamic_train_steps, dynamic_mesh_resolution, artifact_dir)
        _save_config_snapshot(config, config_dir, f"method_comparison_{method}")
        output = _run_pipeline(method, config, monitor_path, scanner_path)
        results.append(_evaluate_output(method, output, gt_mesh))

    df = pd.DataFrame(results)
    df.to_csv(experiment_dir / "method_comparison.csv", index=False)
    print(f"[Experiments] 方法对比结果已写入 {experiment_dir / 'method_comparison.csv'}")
    return df


def run_cpd_ablation(
    experiment_dir: Path,
    artifact_dir: Path,
    config_dir: Path,
    monitor_path: Path,
    scanner_path: Path,
    gt_mesh_path: Path,
    mode: str,
    dynamic_train_steps: int | None,
    dynamic_mesh_resolution: int | None,
    normal_weight: float | None,
    temporal_weight: float | None,
    temporal_acceleration_weight: float | None,
    phase_consistency_weight: float | None,
    periodicity_weight: float | None,
    deformation_weight: float | None,
) -> pd.DataFrame:
    """运行 CPD-Field 关键损失项消融实验。"""
    gt_mesh = _load_gt_mesh(gt_mesh_path)
    ablations: list[tuple[str, dict[str, bool | float | None]]] = [
        (
            "CPD-Field",
            {
                "normal_weight": normal_weight,
                "temporal_weight": temporal_weight,
                "temporal_acceleration_weight": temporal_acceleration_weight,
                "phase_consistency_weight": phase_consistency_weight,
                "periodicity_weight": periodicity_weight,
                "deformation_weight": deformation_weight,
            },
        ),
        ("CPD-Field w/o 周期边界", {"disable_periodicity": True}),
        ("CPD-Field w/o 置信度加权", {"disable_confidence_weighting": True}),
        ("CPD-Field w/o 法向约束", {"disable_normal": True}),
        ("CPD-Field w/o 二阶时间正则", {"disable_acceleration": True}),
        ("CPD-Field w/o 相位一致性", {"disable_phase_consistency": True}),
    ]

    results: list[dict[str, float | int | str]] = []
    for label, ablation in ablations:
        print(f">>> 运行消融：{label}")
        config = _build_config(
            method="动态共享",
            mode=mode,
            dynamic_train_steps=dynamic_train_steps,
            dynamic_mesh_resolution=dynamic_mesh_resolution,
            pointcloud_out_dir=artifact_dir,
            dynamic_ablation=ablation,
        )
        _save_config_snapshot(config, config_dir, f"cpd_ablation_{label}")
        output = _run_pipeline(label, config, monitor_path, scanner_path)
        row = _evaluate_output(label, output, gt_mesh)
        row["模型类型"] = label
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(experiment_dir / "cpd_ablation.csv", index=False)
    print(f"[Experiments] CPD-Field 消融结果已写入 {experiment_dir / 'cpd_ablation.csv'}")
    return df


def emit_tables(df: pd.DataFrame, experiment_dir: Path, stem: str) -> None:
    """导出论文整理所需的表格文件。"""
    latex_path = experiment_dir / f"{stem}.tex"
    latex = df.to_latex(index=False, float_format="%.4f")
    latex_path.write_text(latex, encoding="utf-8")
    print("\n--- LaTeX 表格片段 ---")
    print(latex)
    print("-----------------------\n")
    print(f"[Experiments] LaTeX 表格已写入 {latex_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 4D-UMS 静态/动态模型对比实验")
    parser.add_argument("--mode", choices=["fast-dev", "dynamic-detail", "full-paper"], default="fast-dev", help="实验运行模式：快速调试、动态细节诊断或正式论文配置")
    parser.add_argument("--experiment-set", choices=["method-comparison", "cpd-ablation", "both"], default="both", help="运行方法对比、CPD-Field 消融，或两者都跑")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_EXPERIMENT_ROOT), help="实验归档根目录；每次运行会在该目录下创建独立子目录")
    parser.add_argument("--monitor-path", type=str, default=str(DEFAULT_MONITOR_PATH), help="监测流 NPZ 路径")
    parser.add_argument("--scanner-path", type=str, default=str(DEFAULT_SCANNER_PATH), help="扫描流 NPZ 路径")
    parser.add_argument("--gt-mesh-path", type=str, default=str(DEFAULT_GT_MESH_PATH), help="真值网格路径")
    parser.add_argument("--run-name", type=str, default=None, help="可选运行名称，会写入本次实验归档目录名")
    parser.add_argument("--dynamic-train-steps", type=int, default=None, help="覆盖动态共享模型训练步数")
    parser.add_argument("--dynamic-mesh-resolution", type=int, default=None, help="覆盖动态共享模型导出网格分辨率")
    parser.add_argument("--normal-weight", type=float, default=None, help="覆盖 CPD-Field 法向约束权重")
    parser.add_argument("--temporal-weight", type=float, default=None, help="覆盖 CPD-Field 一阶时间平滑权重")
    parser.add_argument("--temporal-acceleration-weight", type=float, default=None, help="覆盖 CPD-Field 二阶时间正则权重")
    parser.add_argument("--phase-consistency-weight", type=float, default=None, help="覆盖 CPD-Field 相位一致性权重")
    parser.add_argument("--periodicity-weight", type=float, default=None, help="覆盖 CPD-Field 周期边界权重")
    parser.add_argument("--deformation-weight", type=float, default=None, help="覆盖 CPD-Field 形变幅度正则权重")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(out_root, args.mode, args.experiment_set, args.run_name)
    artifact_dir = run_dir / "artifacts"
    config_dir = run_dir / "configs"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    _write_run_metadata(args, run_dir, log_path)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = _TeeStream(original_stdout, log_file)
        tee_stderr = _TeeStream(original_stderr, log_file)
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        try:
            print(f"[Experiments] 本次实验归档目录: {run_dir}")
            print(f"[Experiments] 中间产物目录: {artifact_dir}")
            if args.experiment_set in {"method-comparison", "both"}:
                comparison_df = run_method_comparison(
                    experiment_dir=run_dir,
                    artifact_dir=artifact_dir,
                    config_dir=config_dir,
                    monitor_path=Path(args.monitor_path),
                    scanner_path=Path(args.scanner_path),
                    gt_mesh_path=Path(args.gt_mesh_path),
                    mode=str(args.mode),
                    dynamic_train_steps=args.dynamic_train_steps,
                    dynamic_mesh_resolution=args.dynamic_mesh_resolution,
                )
                emit_tables(comparison_df, run_dir, "method_comparison")

            if args.experiment_set in {"cpd-ablation", "both"}:
                ablation_df = run_cpd_ablation(
                    experiment_dir=run_dir,
                    artifact_dir=artifact_dir,
                    config_dir=config_dir,
                    monitor_path=Path(args.monitor_path),
                    scanner_path=Path(args.scanner_path),
                    gt_mesh_path=Path(args.gt_mesh_path),
                    mode=str(args.mode),
                    dynamic_train_steps=args.dynamic_train_steps,
                    dynamic_mesh_resolution=args.dynamic_mesh_resolution,
                    normal_weight=args.normal_weight,
                    temporal_weight=args.temporal_weight,
                    temporal_acceleration_weight=args.temporal_acceleration_weight,
                    phase_consistency_weight=args.phase_consistency_weight,
                    periodicity_weight=args.periodicity_weight,
                    deformation_weight=args.deformation_weight,
                )
                emit_tables(ablation_df, run_dir, "cpd_ablation")

            print("全部实验执行完成。")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

if __name__ == "__main__":
    main()
