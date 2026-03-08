# 动态超声胃部建模系统架构

本项目采用分层解耦的 Python 模块化结构，将跨时空一致性建模的七个关键环节拆分为若干可复用的子模块：

```
src/
├── config.py                 # 全局参数、数据类定义
├── data_acquisition/
│   ├── monitor.py            # 监测阶段：2D 高帧率信号采集接口
│   └── free_arm_scan.py      # 自由臂扫描阶段：3D 数据采集与同步
├── preprocessing/
│   ├── phase_detection.py    # 蠕动周期检测、相位归一化
│   └── binning.py            # 基于相位的三维数据分箱
├── reconstruction/
│   ├── reference.py          # 全局参考网格 V_ref 构建
│   ├── registration.py       # 周期内/跨周期非刚性配准接口
│   └── averaging.py          # 相位三维容积平均与加权
├── modeling/
│   ├── interpolation.py      # {V_φ} -> 连续 4D 插值
│   └── validation.py         # 4D 模型一致性验证与指标量化
├── pipelines/
│   └── multicycle_reconstruction.py  # 将所有阶段串联的主流程
└── utils/
    └── signals.py            # 信号处理、滤波、峰谷检测等工具函数
```

- **数据获取层** (`data_acquisition`): 模拟或封装真实仪器 API，强调时间同步与探头姿态的标准化记录。
- **预处理层** (`preprocessing`): 负责周期信号分析、相位归一化和多周期分箱，是“多周期平均法”的时间基准核心。
- **重建层** (`reconstruction`): 包含构建参考模板、两级非刚性配准以及加权平均，是空间对齐与降噪的关键。
- **建模与分析层** (`modeling`): 对离散平均容积进行时间插值生成 4D 序列，同时输出临床可解释的量化指标。
- **工具层** (`utils`): 收敛项目中需要反复复用的信号处理小组件，降低耦合度。
- **管线层** (`pipelines`): 提供面向临床工程师的高层 API，一次性调用所有阶段并输出最终 4D 模型与验证报告。

该结构既可以直接连接真实超声硬件，也便于在研究阶段通过仿真数据完成快速验证。后续实现将严格按照此架构落地。