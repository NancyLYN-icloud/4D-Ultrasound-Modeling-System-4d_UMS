# 动态胃部超声建模系统架构

本项目当前已经从早期的“多周期平均法”原型，演化为一套围绕 `动态共享-全局基残差` 的模块化研究框架。其核心主线为：

监测信号 -> 非线性相位标准化 -> 置信度感知点云构建 -> 静态逐相位重建 / `动态共享-全局基残差` 动态共享重建 -> 几何与时序评估。

当前推荐理解的代码结构如下：

```
src/
├── config.py                          # 全局参数、数据结构与模型配置
├── data_acquisition/
│   ├── monitor.py                     # 监测阶段：2D 高帧率信号采集与特征提取
│   └── free_arm_scan.py               # 自由臂扫描阶段：2D/伪3D 数据采集与时间同步
├── preprocessing/
│   ├── phase_detection.py             # 周期检测
│   ├── phase_canonicalization.py      # 非线性相位标准化
│   ├── binning.py                     # 基于标准相位的样本分箱
│   └── pointcloud_builder.py          # 置信度感知点云构建与摘要导出
├── modeling/
│   ├── surface_reconstruction.py      # 逐相位静态隐式表面重建
│   ├── canonical_field.py             # 标准形状隐式场
│   ├── deformation_field.py           # 相位条件形变场
│   ├── dynamic_surface_reconstruction.py # 动态共享重建（当前主线为全局基残差）
│   ├── metrics.py                     # CD、HD95、表面 MAE、EMD、Dice 等指标
│   └── validation.py                  # 传统 4D 指标验证接口（保留）
├── pipelines/
│   └── multicycle_reconstruction.py   # 端到端主流程，统一调度静态/动态模型
└── utils/
        └── signals.py                     # 信号处理、滤波、峰谷检测等工具函数
```

## 1. 架构分层

- **数据获取层** `data_acquisition`
负责管理监测阶段与自由臂扫描阶段的数据，强调时间戳对齐、探头位姿记录与基础预处理兼容性。

- **预处理层** `preprocessing`
负责周期检测、非线性相位标准化、标准相位分箱以及置信度感知点云构建，是整个方法的时间基准与观测质量控制核心。当前主线默认使用旧的离散相位分箱，滑动窗口仅作为可选实验分支保留。

- **建模层** `modeling`
同时包含两类模型分支：
    - 逐相位静态重建分支：用于基线和传统方法比较。
    - `动态共享-全局基残差` 动态分支：用于论文主模型。

- **管线层** `pipelines`
提供高层实验入口，将周期检测、相位标准化、点云构建、静态/动态模型训练与输出组织成统一工作流。

- **评估层** `metrics` / `validation`
输出几何精度、体素重叠和观测质量等结果，用于论文表格、消融实验与可视化分析。

## 2. 当前主模型：动态共享-全局基残差

`动态共享-全局基残差` 是当前项目推荐作为论文主方法的动态建模框架，其思想是：

- 用共享拓扑的基准网格承载静态参考形态。
- 用低秩全局运动基表示跨相位的大尺度周期运动。
- 用相位条件残差场补充局部细节，并在统一标准相位域上共享所有相位点云监督。

代码映射如下：

- 系数与残差场：[src/modeling/deformation_field.py](src/modeling/deformation_field.py)
- 动态训练器：[src/modeling/dynamic_surface_reconstruction.py](src/modeling/dynamic_surface_reconstruction.py)

当前主方法已实现的关键损失项包括：

- 表面拟合损失
- 法向一致性损失
- 质心一致性与空间平滑损失
- 基系数 bootstrap / 时间 / 加速度 / 周期性正则
- 残差均值与残差基投影约束
- unsupported 区域 anchor / laplacian 正则
- 全局分支 correspondence 时序约束
- 置信度感知的相位采样与损失加权

## 3. 当前主流程

[src/pipelines/multicycle_reconstruction.py](src/pipelines/multicycle_reconstruction.py) 的当前主流程可概括为：

1. 从监测流中提取特征轨迹并检测胃蠕动周期。
2. 使用非线性相位标准化将扫描时间映射到统一标准相位域。
3. 将扫描样本按标准相位分箱，并构建带质量摘要的相位点云。
4. 运行逐相位静态重建分支，得到静态网格序列。
5. 运行可选的 `动态共享-全局基残差` 分支，得到动态网格序列。
6. 导出用于实验和论文整理的 mesh/summary/CSV 结果。

## 4. 研究导向的设计原则

当前架构遵循以下研究导向原则：

- **双分支对比**：保留静态逐相位重建，便于与当前主方法做公平对比。
- **最小侵入演化**：新模块以可选方式接入主流程，避免破坏已有结果。
- **配置显式化**：关键损失项与训练参数集中在 [src/config.py](src/config.py) 中，便于消融与复现实验。
- **结果可归档**：各阶段都输出结构化 CSV 摘要，方便生成论文表格。

## 5. 当前实验入口

推荐实验入口为 [scripts/run_experiments.py](scripts/run_experiments.py)，当前已支持：

- `fast-dev`：快速调试模式，用较低训练步数和较低网格分辨率验证趋势。
- `full-paper`：论文正式实验模式，用更高训练预算获取最终结果。
- 方法对比实验：静态基线、静态增强、动态共享。
- 主方法消融实验：去掉周期边界、置信度加权、法向约束、二阶时间正则、相位一致性等关键项。

## 6. 当前状态说明

旧的 `reconstruction/` 与部分 `validation/` 模块仍然保留，用于历史方法对照和后续扩展，但当前论文主线已经不再依赖“参考模板 + 配准 + 平均 + 插值”的旧方案。

当前最应该围绕的主线是：

- 非线性相位标准化
- 置信度感知观测融合
- `动态共享-全局基残差` 动态建模

后续所有论文写作、实验设计与系统扩展，建议以这三部分为主轴继续推进。