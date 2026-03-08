"""跨周期非刚性配准。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..config import PhaseBin, RegistrationConfig, VolumeDescriptor

try:  # pragma: no cover - 可选依赖
    import SimpleITK as sitk
except Exception:  # pragma: no cover
    sitk = None


@dataclass
class DeformationField:
    """描述从 moving -> reference 的形变场。"""

    displacement: np.ndarray  # shape = (*grid, 3)

    def compose(self, other: "DeformationField") -> "DeformationField":
        """组合两个形变场，返回整体形变。"""
        return DeformationField(displacement=self.displacement + other.displacement)


class NonRigidRegistrar:
    """封装周期内与跨周期的配准逻辑。"""

    def __init__(self, config: RegistrationConfig) -> None:
        self.config = config

    def _to_sitk(self, volume: np.ndarray) -> Optional["sitk.Image"]:
        if sitk is None:
            return None
        image = sitk.GetImageFromArray(volume.astype(np.float32))
        image.SetSpacing((1.0, 1.0, 1.0))
        return image

    def _register_sitk(self, moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        """使用 B 样条非刚性配准 (若可用)。"""
        moving_img = self._to_sitk(moving)
        fixed_img = self._to_sitk(fixed)
        assert moving_img is not None and fixed_img is not None
        transform_domain_mesh_size = [max(1, s // self.config.grid_spacing[i]) for i, s in enumerate(fixed.shape)]
        transform = sitk.BSplineTransformInitializer(fixed_img, transform_domain_mesh_size)
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        registration_method.SetMetricSamplingPercentage(0.2, sitk.sitkWallClock)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=self.config.regularization_lambda)
        registration_method.SetInitialTransform(transform, inPlace=False)
        final_transform = registration_method.Execute(fixed_img, moving_img)
        displacement = sitk.TransformToDisplacementField(final_transform, sitk.sitkVectorFloat64, fixed_img)
        return sitk.GetArrayFromImage(displacement)

    def _register_centroid(self, moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        """退化方案：仅对齐几何质心。"""
        coords = np.stack(np.meshgrid(*[np.arange(s) for s in moving.shape], indexing="ij"), axis=-1)
        moving_center = np.average(coords.reshape(-1, 3), axis=0, weights=moving.flatten())
        fixed_center = np.average(coords.reshape(-1, 3), axis=0, weights=fixed.flatten())
        shift = fixed_center - moving_center
        displacement = np.zeros((*moving.shape, 3), dtype=float)
        displacement[...] = shift
        return displacement

    def register_volume(self, moving: np.ndarray, fixed: np.ndarray) -> DeformationField:
        """对单个三维体执行非刚性配准。"""
        print(f"[Registrar] register_volume: moving shape {moving.shape}, fixed shape {fixed.shape}")
        if sitk is not None:
            try:
                displacement = self._register_sitk(moving, fixed)
            except Exception as e:
                print(f"[Registrar] SITK配准失败: {e}, 使用质心对齐")
                displacement = self._register_centroid(moving, fixed)
        else:
            displacement = self._register_centroid(moving, fixed)
        return DeformationField(displacement=displacement)

    def register_phase_bin(self, bin_data: PhaseBin, reference: VolumeDescriptor) -> Sequence[np.ndarray]:
        """对 bin 中所有样本执行配准并映射到 V_ref。"""
        print(f"[Registrar] register_phase_bin: 样本数 {len(bin_data.samples)}")
        aligned_volumes = []
        for idx, sample in enumerate(bin_data.samples):
            print(f"  对齐样本 {idx} 时间戳 {sample.timestamp}")
            field = self.register_volume(sample.volume_slice, reference.intensities)
            aligned_volumes.append(
                self.apply_field(sample.volume_slice, field)
            )
        return aligned_volumes

    def apply_field(self, volume: np.ndarray, field: DeformationField) -> np.ndarray:
        """应用形变场，将体素插值到参考网格。"""
        grid = np.stack(np.meshgrid(*[np.arange(s) for s in volume.shape], indexing="ij"), axis=-1)
        target = grid + field.displacement
        target = np.clip(target, 0, np.array(volume.shape) - 1)
        # 三线性插值
        floored = np.floor(target).astype(int)
        weight = target - floored
        aligned = np.zeros_like(volume)
        for corner in range(8):
            offset = np.array([(corner >> i) & 1 for i in range(3)])
            idx = tuple(floored[..., i] + offset[i] for i in range(3))
            coeff = np.ones_like(volume, dtype=float)
            for axis in range(3):
                axis_weight = weight[..., axis] if offset[axis] == 1 else (1 - weight[..., axis])
                coeff *= axis_weight
            aligned += coeff * volume[idx]
        return aligned
