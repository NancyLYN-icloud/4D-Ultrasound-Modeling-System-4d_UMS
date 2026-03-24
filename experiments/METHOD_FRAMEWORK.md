# 顶会/顶刊论文方法框架

## 1. 建议的论文定位

当前主流程在 [src/pipelines/multicycle_reconstruction.py](../src/pipelines/multicycle_reconstruction.py) 中可概括为：

监测信号 -> 周期检测 -> 相位分箱 -> 相位点云 -> 逐相位网格重建。

作为工程原型，这条链路是成立的；但如果目标是高水平顶会或顶刊，这套方法仍然更像“模块化流程拼接”，而不是一个具有鲜明方法学主张的模型。论文定位建议从：

- 多周期平均或相位分箱重建

升级为：

- 面向自由手胃部超声的非线性相位标准化与置信度感知时空隐式重建

更有力度的题目方向可以是：

- 面向时空一致 4D 胃部超声重建的非线性相位标准化
- 面向自由手动态胃部超声的置信度感知时空隐式建模
- 基于标准相位域的胃部动态表面重建方法

## 2. 当前方法的技术瓶颈

结合 [src/preprocessing/phase_detection.py](../src/preprocessing/phase_detection.py)、[src/preprocessing/binning.py](../src/preprocessing/binning.py)、[src/preprocessing/pointcloud_builder.py](../src/preprocessing/pointcloud_builder.py) 与 [src/modeling/surface_reconstruction.py](../src/modeling/surface_reconstruction.py) 的现有实现，主要瓶颈包括：

1. 周期标准化仍是线性的。
系统先估计平均周期长度，再把样本时间线性映射到归一化相位。这不能显式处理跨周期速度变化、局部相位漂移，也无法表示收缩与舒张速度不对称。

2. 点云提取仍以启发式规则为主。
当前点云阶段主要依赖阈值、形态学、轮廓提取和均匀采样，没有显式建模超声伪影、探头角度变化、视角缺失和位姿抖动带来的观测不确定性。

3. 重建仍是逐相位独立的，而不是真正的 4D 建模。
当前网格重建阶段对每个相位分别建模，得到的是一组 3D 网格序列，而不是一个共享几何与运动先验的时空连续表示。

4. 解剖先验较弱。
现有网格正则主要是通用几何平滑与水密修补，没有显式编码胃部拓扑、蠕动传播方向或受约束的解剖形变模式，因此临床可解释性还不够强。

## 3. 建议作为论文主贡献的三部分

论文可以围绕以下三个贡献组织。

### 贡献 A：非线性相位标准化

用学习式或优化式的时间扭曲函数替代“平均周期长度 + 线性相位映射”，将每个观测周期映射到统一的标准相位域。

核心价值：

- 消除周期长度变化带来的误差
- 对齐不同周期中的非均匀收缩速度
- 在不依赖硬分箱假设的前提下增强多周期融合稳定性

这是对 [src/preprocessing/binning.py](../src/preprocessing/binning.py) 当前逻辑最直接、最有论文价值的升级。

### 贡献 B：置信度感知观测融合

把点提取阶段从“只输出坐标”升级成“输出带置信度的观测点”，置信度由信号质量、轮廓稳定性、视角与位姿一致性等因素共同决定。

核心价值：

- 提升方法对低质量超声观测的鲁棒性
- 为后续训练或拟合提供可解释的加权机制
- 自然引入不确定性感知损失

这是 [src/preprocessing/pointcloud_builder.py](../src/preprocessing/pointcloud_builder.py) 最自然的强化方向。

### 贡献 C：基于共享拓扑全局基残差的时空耦合表面建模

把“每个相位独立重建一个网格”的策略升级为共享拓扑的时空模型，在网格顶点域与相位域上联合建模器官表面。当前代码中这条主模型已经具体化为：

- `动态共享-全局基残差`（`shared_topology_global_basis_residual`）

其核心思想是：

- 用一个共享拓扑基准网格表示静态参考形态
- 用低秩全局运动基表示跨相位的大尺度周期运动
- 用相位条件残差场补充局部动态细节，并通过 correspondence 调度与残差正则保持时序稳定

形式化表达为：

$$
V(\phi) = V_0 + \bar{\Delta} + \sum_{r=1}^{R} \alpha_r(\phi) B_r + R_\theta(V_0, \phi)
$$

其中 $V_0$ 为共享基准网格，$\bar{\Delta}$ 为全局平均位移，$B_r$ 为第 $r$ 个全局运动基，$\alpha_r(\phi)$ 为相位系数，$R_\theta$ 为相位条件局部残差场。

这种表达兼顾了拓扑稳定性、动态细节承载能力与训练可控性，明显优于单纯“逐相位独立训练”的方法。

## 4. 建议的新主流程

升级后的主流程建议表述为：

监测信号 -> 标准相位对齐 -> 置信度感知点观测 -> 共享拓扑全局基残差动态建模 -> 时空连续网格序列 -> 几何与临床分析。

按模块拆解后，目标架构应包括：

1. 监测信号编码模块
2. 周期标准化模块
3. 观测置信度估计模块
4. 共享拓扑动态重建模块
5. 解剖约束模块
6. 评估与可视化模块

## 5. 方法章节建议结构

方法部分建议按以下顺序组织。

### 5.1 问题定义

给定：

- 来自高帧率 2D 监测的特征时间序列 $m(t)$
- 来自自由手扫描的观测序列 $\{o_i, t_i, T_i\}$

其中 $t_i$ 为时间戳，$T_i$ 为探头位姿，目标是恢复定义在标准相位域 $\phi \in [0,1]$ 上的时空连续 4D 胃部表面。

输出可写为：

$$
\mathcal{S}(\phi), \quad \phi \in [0,1]
$$

或者等价地表示为“标准表面 + 形变场”。

### 5.2 非线性相位标准化

对每个周期 $k$，估计一个单调时间扭曲函数：

$$
\phi = w_k(t), \quad t \in [t_k^{start}, t_k^{end}]
$$

并满足约束：

$$
\frac{d w_k(t)}{dt} > 0, \quad w_k(t_k^{start}) = 0, \quad w_k(t_k^{end}) = 1
$$

适合本项目的三种实现方式：

1. 基于模板的动态时间规整
2. 基于关键点与极值点的单调样条拟合
3. 轻量级神经相位标准化器，并用周期一致性约束训练

建议第一版论文先落地：

- 单调分段样条或分段线性标准化
- 以当前线性相位映射作为基线对比

### 5.3 置信度感知观测模型

每个提取出的观测点 $p_i$ 不再只对应坐标，而是带有置信度权重 $c_i \in [0,1]$：

$$
c_i = g(q_i^{snr}, q_i^{contour}, q_i^{angle}, q_i^{pose})
$$

其中可用的置信度线索包括：

- 局部信噪比或强度对比度
- 轮廓闭合质量
- 轮廓曲率稳定性
- 相对探头方向的入射角
- 短时位姿平滑性

观测损失可以写成：

$$
\mathcal{L}_{obs} = \sum_i c_i \, \rho(F_\theta(p_i, \phi_i))
$$

其中 $\rho$ 可采用 L1 或鲁棒损失。

### 5.4 共享拓扑全局基残差主模型

当前主模型采用 `动态共享-全局基残差`。其核心公式为：

$$
V(\phi) = V_0 + \bar{\Delta} + \sum_{r=1}^{R} \alpha_r(\phi) B_r + R_\theta(V_0, \phi)
$$

其中：

- $V_0$ 为共享拓扑基准网格
- $\bar{\Delta}$ 为全局平均位移
- $B_r$ 为低秩全局运动基
- $\alpha_r(\phi)$ 为相位条件全局系数
- $R_\theta(V_0, \phi)$ 为相位条件局部残差场

这种写法的优势是：

- 将大尺度周期运动与局部细节残差解耦
- 保持共享拓扑，便于控制时序稳定性与网格可解释性
- 便于针对 unsupported 区域、correspondence 时序和局部残差强度施加结构化约束

### 5.5 正则项设计

当前主模型的总损失可写为：

$$
\mathcal{L} = \mathcal{L}_{surf} + \lambda_{nor}\mathcal{L}_{nor} + \lambda_{cen}\mathcal{L}_{cen} + \lambda_{sp}\mathcal{L}_{sp} + \lambda_{boot}\mathcal{L}_{boot} + \lambda_{coef}\mathcal{L}_{coef} + \lambda_{res}\mathcal{L}_{res} + \lambda_{corr}\mathcal{L}_{corr} + \lambda_{per}\mathcal{L}_{per}
$$

其中：

- $\mathcal{L}_{surf}$：置信度加权的表面拟合损失
- $\mathcal{L}_{nor}$：法向一致性损失
- $\mathcal{L}_{cen}$：质心一致性损失
- $\mathcal{L}_{sp}$：残差场空间平滑损失
- $\mathcal{L}_{boot}$：全局基与残差的 bootstrap 对齐损失
- $\mathcal{L}_{coef}$：全局系数的时间、加速度与周期性正则
- $\mathcal{L}_{res}$：残差均值、残差基投影与 unsupported 区域正则
- $\mathcal{L}_{corr}$：global-only correspondence 时序约束
- $\mathcal{L}_{per}$：周期边界一致性约束

其中，观测损失的核心形式为：

$$
\mathcal{L}_{surf} = \sum_i c_i \, d\big(\hat{p}_i(\phi_i), p_i\big)^2
$$

这里 $c_i$ 由点云质量估计得到；在当前代码实现中，相位级训练权重由平均点云置信度和切片提取率共同决定。

一阶与二阶时间正则施加在全局基系数和相位 correspondence 上，用于抑制跨相位高频抖动；周期边界约束用于保证完整蠕动周期首尾连续；unsupported 区域正则则用于防止低支持区域出现不稳定漂移。
- 通过形变正则与自交惩罚维持拓扑稳定

### 5.6 网格提取与临床可解释输出

当 4D 场建模完成后，可在离散相位上查询网格：

$$
\phi_j = \frac{j}{N-1}, \quad j = 0, 1, ..., N-1
$$

随后计算：

- 网格精度指标
- 时间一致性指标
- 胃腔体积变化曲线
- 收缩幅度与传播速度

## 6. 代码改造路线图

当前代码结构已经具备较清晰的模块边界，因此改造建议如下。

### 第一步：加入相位标准化模块

新增文件：

- [src/preprocessing/phase_canonicalization.py](../src/preprocessing/phase_canonicalization.py)

职责：

- 估计标准周期模板
- 拟合每个周期的非线性扭曲函数
- 将扫描时间戳映射到标准相位
- 在需要时输出相位不确定性或 warp 摘要

主要接入点：

- [src/preprocessing/phase_detection.py](../src/preprocessing/phase_detection.py)
- [src/preprocessing/binning.py](../src/preprocessing/binning.py)
- [src/pipelines/multicycle_reconstruction.py](../src/pipelines/multicycle_reconstruction.py)

### 第二步：把点提取升级为带权观测

扩展 [src/preprocessing/pointcloud_builder.py](../src/preprocessing/pointcloud_builder.py)，不仅返回点坐标，还返回：

- 点级置信度
- 来源样本编号
- 来源相位
- 可选的轮廓法向估计

这一步很可能需要在 [src/config.py](../src/config.py) 中新增一个数据类，例如：

- WeightedPointObservation

### 第三步：用共享拓扑全局基残差替换逐相位独立重建

当前代码已经完成了这一方向的第一版实现，核心文件为：

- [src/modeling/canonical_field.py](../src/modeling/canonical_field.py)
- [src/modeling/deformation_field.py](../src/modeling/deformation_field.py)
- [src/modeling/dynamic_surface_reconstruction.py](../src/modeling/dynamic_surface_reconstruction.py)

其中 [src/modeling/dynamic_surface_reconstruction.py](../src/modeling/dynamic_surface_reconstruction.py) 已实现：

- 法向一致性约束
- 相位一致采样
- 一阶和二阶时间正则
- 周期边界闭环约束
- 置信度感知的动态训练加权

### 第四步：重设计主流程输出

[src/pipelines/multicycle_reconstruction.py](../src/pipelines/multicycle_reconstruction.py) 的输出建议从：

- 点云路径
- 逐相位网格

逐步演化为：

- 标准模板表示
- 动态网格序列
- 相位 warp 元数据
- 观测置信度摘要
- 评估报告

## 7. 强消融实验设计

论文中至少应包含以下消融：

1. 线性相位归一化 vs 非线性相位标准化
2. 无置信度加权 vs 有置信度加权
3. 独立逐相位重建 vs 共享拓扑全局基残差模型
4. `动态共享-全局基残差` 去掉周期边界约束 vs 完整模型
5. `动态共享-全局基残差` 去掉置信度加权 vs 完整模型
6. `动态共享-全局基残差` 去掉法向一致性 vs 完整模型
7. 单周期输入 vs 多周期输入

这组消融能直接支撑方法主贡献。

## 8. 图表规划

建议的核心图表包括：

1. 系统总览图
展示当前输入、相位标准化模块、置信度感知融合模块、动态隐式模型与最终输出网格之间的关系。

2. 相位标准化示意图
展示不同原始周期具有不等时长和不等速蠕动时，如何被对齐到统一标准相位域。

3. 重建对比图
对比基线的逐相位独立重建与当前主方法时空一致重建的差异。

4. 误差热力图
在合成数据基准上渲染 mesh-to-GT 的表面误差分布。

5. 真实病例定性图
展示一个完整胃蠕动周期在多个标准相位上的重建结果。

## 9. 优先实现顺序

如果实现时间有限，最值得优先投入的顺序是：

1. 非线性相位标准化
2. 共享式 `动态共享-全局基残差` 动态模型
3. 置信度感知观测加权
4. 解剖先验正则

这个顺序能以最小工程投入换取最大的论文创新收益。

## 10. 当前最合适的下一步

下一步不建议一开始就重写整个系统，而应先实现一个最小但可发表的创新版本：

- 用单调非线性相位标准化替代当前线性相位映射

这一步可以在尽量不打断现有点云与网格流程的前提下接入，并且能直接带来可量化的时间一致性和几何精度提升。

之后的第二个关键里程碑应是：

- 用 `动态共享-全局基残差` 替代逐相位独立重建

这两个创新点的组合，是从当前工程原型走向顶会/顶刊方法论文最可信的路径。
