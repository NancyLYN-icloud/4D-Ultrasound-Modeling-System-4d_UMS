# data/raw 输入示例

| 文件 | 内容 | 结构 |
| --- | --- | --- |
| `monitor_stream.npz` | 监测阶段的 2D 帧序列与时间戳，供 `UltrasoundMonitor.from_npz` 使用 | `frames`: (N, 64, 64) float32；`timestamps`: (N,) float32；`feature_trace`: (N,) float32 |
| `scanner_sequence.npz` | 自由臂扫描阶段的帧序列、时间戳与位姿，供 `FreeArmScanner.from_npz` 使用 | `frames`: (M, 512, 512) float64（或 float32）；`timestamps`: (M,) float64；`positions`: (M, 3) float64；`orientations`: (M, 3, 3) float64 |

补充说明：仓库内自带的合成 `scanner_sequence.npz` 是按 `0.42 mm/pixel` 栅格化生成的；如果直接导出相位点云，请使用相同的像素间距恢复物理尺度。

## 用法

```python
import numpy as np
from src.data_acquisition.monitor import UltrasoundMonitor
from src.data_acquisition.free_arm_scan import FreeArmScanner
from src.config import PipelineConfig

config = PipelineConfig()
monitor = UltrasoundMonitor.from_npz(config.acquisition, "data/raw/monitor_stream.npz")

scanner = FreeArmScanner.from_npz(config.acquisition, "data/raw/scanner_sequence.npz")
```

之后即可将 `monitor` 与 `scanner` 交给 `MulticycleReconstructionPipeline.run()`，验证端到端流程。