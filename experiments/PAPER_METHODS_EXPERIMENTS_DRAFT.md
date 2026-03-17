# 论文 Methods / Experiments 中文初稿

## 方法

### 1. 问题定义

本文关注自由手胃部超声场景下的 4D 时空一致重建问题。给定高帧率监测阶段采集的二维超声帧序列及其特征轨迹 $m(t)$，以及自由手扫描阶段采集的带时间戳和位姿信息的观测序列 $\{o_i, t_i, T_i\}$，目标是在统一的标准相位域 $\phi \in [0,1]$ 上恢复连续的动态胃部表面表示。

与传统逐相位独立重建不同，本文旨在同时解决三个关键问题：

1. 不同胃蠕动周期之间存在时长变化与局部相位漂移，简单线性归一化会破坏跨周期一致性。
2. 自由手超声观测存在噪声、遮挡、视角变化与位姿扰动，导致不同相位点云质量不均衡。
3. 逐相位独立建模只能得到离散的 3D 网格集合，难以形成真正时空连续且拓扑稳定的 4D 表示。

为此，本文提出一个由非线性相位标准化、置信度感知观测融合和动态共享隐式建模组成的统一框架。

### 2. 非线性相位标准化

在监测阶段，我们首先从二维超声序列中提取特征时间曲线，并通过周期检测得到若干胃蠕动周期。对于每个周期 $k$，定义其起点、峰值点和终点分别为 $t_k^{start}$、$t_k^{peak}$ 与 $t_k^{end}$。传统方法通常直接采用线性归一化：

$$
\phi = \frac{t - t_k^{start}}{t_k^{end} - t_k^{start}}
$$

但这种做法隐含假设不同周期具有相同的时间演化速度，无法刻画收缩和舒张过程的非对称性。为此，本文采用非线性相位标准化策略，将每个周期映射到统一的标准相位域。对每个周期 $k$，我们估计一个满足单调性的时间扭曲函数：

$$
\phi = w_k(t), \quad t \in [t_k^{start}, t_k^{end}]
$$

并要求：

$$
\frac{d w_k(t)}{dt} > 0, \quad w_k(t_k^{start}) = 0, \quad w_k(t_k^{end}) = 1
$$

当前实现采用基于周期峰值相位位置的分段单调 warp。该策略虽然较为轻量，但能够显式对齐不同周期中的非均匀运动速度，为后续跨周期融合提供更稳定的时间基准。

### 3. 置信度感知点云观测构建

在扫描阶段，本文将每个相位分箱中的超声观测样本转换为世界坐标系下的三维点云。不同于仅保留点坐标的传统做法，本文进一步为每个相位点云估计观测质量摘要，并据此构建置信度感知观测模型。

对于每个观测点 $p_i$，其置信度 $c_i$ 定义为若干质量因素的组合函数：

$$
c_i = g(q_i^{snr}, q_i^{contour}, q_i^{angle}, q_i^{pose})
$$

其中 $q_i^{snr}$ 表示信噪比相关指标，$q_i^{contour}$ 表示轮廓提取质量，$q_i^{angle}$ 表示视角相关质量，$q_i^{pose}$ 表示位姿稳定性。当前实现以相位级别的平均点云置信度和切片提取率作为训练权重来源，并将其用于动态共享模型的相位采样与数据拟合加权。

这一设计的意义在于：一方面可减弱低质量观测对共享动态模型的干扰；另一方面也提升了模型在真实自由手扫描场景中的鲁棒性与可解释性。

### 4. CPD-Field 动态共享隐式模型

为实现时空连续的胃部重建，本文提出 `CPD-Field`（Canonical Phase-Deformation Field）作为核心动态共享模型。该模型由两个部分组成：

1. 标准形状隐式场 $S_0(x)$，用于刻画胃部的静态标准形态。
2. 相位条件形变场 $D_\theta(x, \phi)$，用于描述标准相位域上的动态位移。

其组合形式为：

$$
S(x, \phi) = S_0\big(x + D_\theta(x, \phi)\big)
$$

其中 $x \in \mathbb{R}^3$ 为空间坐标，$\phi \in [0,1]$ 为标准相位。为了增强相位表达能力，我们使用周期编码函数对相位进行嵌入：

$$
\gamma(\phi) = [\sin(2\pi \phi), \cos(2\pi \phi), \ldots, \sin(2K\pi \phi), \cos(2K\pi \phi)]
$$

随后将 $(x, \gamma(\phi))$ 共同输入形变场网络中，以学习更稳定的周期性动态表示。

相较于逐相位独立建模，`CPD-Field` 具有三方面优势：

1. 通过标准场与形变场解耦静态形态和动态运动，增强临床可解释性。
2. 不同相位共享同一个标准形状表示，有助于提升跨相位拓扑一致性。
3. 可以在形变空间中显式加入时间正则和周期边界约束，更适合胃蠕动这种近周期器官运动建模任务。

### 5. 损失函数设计

`CPD-Field` 的总损失定义为：

$$
\mathcal{L} = \mathcal{L}_{obs} + \lambda_{nor} \mathcal{L}_{nor} + \lambda_{eik} \mathcal{L}_{eik} + \lambda_{tmp} \mathcal{L}_{tmp} + \lambda_{acc} \mathcal{L}_{acc} + \lambda_{ph} \mathcal{L}_{ph} + \lambda_{per} \mathcal{L}_{per} + \lambda_{def} \mathcal{L}_{def}
$$

其中各项定义如下。

#### 5.1 观测拟合损失

对于采样得到的观测点 $p_i$，观测拟合损失定义为：

$$
\mathcal{L}_{obs} = \sum_i c_i \left| S\big(p_i, \phi_i\big) \right|
$$

其中 $c_i$ 为观测置信度。该项约束观测点位于隐式表面附近，并通过置信度加权降低低质量观测的负面影响。

#### 5.2 法向一致性损失

对观测点估计出的法向 $n_i$ 与标准场梯度方向进行对齐：

$$
\mathcal{L}_{nor} = \sum_i c_i \left(1 - \left| \left\langle \hat{\nabla} S_0, n_i \right\rangle \right| \right)
$$

该项用于增强局部表面几何稳定性，尤其有助于缓解稀疏点云条件下的表面抖动。

#### 5.3 Eikonal 正则

为保证隐式场接近 signed distance function，本文引入 Eikonal 正则：

$$
\mathcal{L}_{eik} = \mathbb{E}_x \left( \| \nabla S_0(x) \|_2 - 1 \right)^2
$$

#### 5.4 一阶时间平滑损失

为约束相邻相位之间的形变变化平滑，定义：

$$
\mathcal{L}_{tmp} = \mathbb{E}_{x, \phi} \left[ \left\| D_\theta(x, \phi + \Delta \phi) - D_\theta(x, \phi) \right\|_2^2 \right]
$$

#### 5.5 二阶时间加速度损失

为进一步抑制时间上的高频振荡，定义二阶差分正则：

$$
\mathcal{L}_{acc} = \mathbb{E}_{x, \phi} \left[ \left\| D_\theta(x, \phi + \Delta \phi) - 2D_\theta(x, \phi) + D_\theta(x, \phi - \Delta \phi) \right\|_2^2 \right]
$$

该项对于降低表面 flicker 和提升时序平稳性尤为重要。

#### 5.6 相位邻域一致性损失

为了增强局部相位邻域中的形变连续性，本文加入相位一致性约束：

$$
\mathcal{L}_{ph} = \mathbb{E}_{x, \phi} \left[ \left\| D_\theta(x, \phi + \Delta \phi) - D_\theta(x, \phi - \Delta \phi) \right\|_2^2 \right]
$$

#### 5.7 周期边界闭环损失

胃蠕动在一个标准周期上应满足首尾连续性，因此定义周期边界闭环损失：

$$
\mathcal{L}_{per} = \mathbb{E}_x \left[ \left\| D_\theta(x, 0) - D_\theta(x, 1) \right\|_2^2 \right]
$$

同时还可对边界附近的一阶变化趋势施加连续性约束，使相位域两端的局部动态行为保持一致。

#### 5.8 形变幅度正则

为防止形变场出现不合理的大位移，定义：

$$
\mathcal{L}_{def} = \mathbb{E}_{x, \phi} \left[ \left\| D_\theta(x, \phi) \right\|_2^2 \right]
$$

### 6. 网格导出与可解释输出

在模型训练完成后，我们在离散相位序列 $\{\phi_j\}_{j=0}^{N-1}$ 上查询 `CPD-Field`，并通过 marching cubes 导出对应的相位网格。最终输出包括：

1. 静态逐相位网格序列。
2. `CPD-Field` 动态共享网格序列。
3. 点云质量摘要、网格统计摘要和实验评估指标。

这些结果共同构成后续几何精度分析、时间一致性评估和真实数据定性展示的基础。

## 实验

### 1. 实验设置

实验基于项目自带的合成胃部数据与预录制监测/扫描流进行。主实验脚本为 [scripts/run_experiments.py](scripts/run_experiments.py)，支持两种运行模式：

- `fast-dev`：快速调试模式，用于验证实验链路和初步结果趋势。
- `full-paper`：正式论文实验模式，用于生成最终结果。

实验共包含两类：

1. **方法对比实验**：比较静态基线、静态增强与 `CPD-Field` 动态共享模型。
2. **`CPD-Field` 消融实验**：比较完整模型与去掉关键损失项后的性能变化。

### 2. 方法对比实验

我们设计如下三组方法：

1. **静态基线**：关闭非线性相位标准化，关闭动态共享模型，仅保留逐相位静态重建。
2. **静态增强**：开启非线性相位标准化，但仍采用逐相位静态重建。
3. **动态共享**：开启非线性相位标准化，并采用 `CPD-Field` 进行动态共享建模。

该对比实验用于回答两个问题：

1. 非线性相位标准化是否能够在不改变静态建模框架的前提下提升结果质量？
2. 在此基础上，引入 `CPD-Field` 是否能够进一步提升几何精度与时间一致性？

### 3. 消融实验

针对 `CPD-Field`，本文设置以下消融组：

1. 完整 `CPD-Field`
2. 去掉周期边界约束
3. 去掉置信度加权
4. 去掉法向一致性损失
5. 去掉二阶时间加速度损失
6. 去掉相位一致性约束

这些消融用于分析各损失项对几何重建与时序平稳性的贡献，并验证本文设计的必要性。

### 4. 评价指标

本文采用以下指标：

1. **Chamfer Distance (CD)**：衡量预测网格与真值网格之间的平均对称距离。
2. **Hausdorff Distance (HD95)**：衡量表面最大误差边界的 95 分位值。
3. **时间平滑度**：衡量相邻相位网格之间的平均位移，用于评价时序稳定性。
4. **水密比例**：统计导出网格中水密网格所占比例。
5. **平均点云置信度**：用于反映观测质量控制模块的有效性。

### 5. 预期结果与分析重点

对于方法对比实验，我们预期：

1. 静态增强将优于静态基线，说明非线性相位标准化有助于提升跨周期一致性。
2. `CPD-Field` 将在 CD、HD95 和时间平滑度上优于静态方法，说明共享动态建模能够同时改善几何精度与时序稳定性。

对于消融实验，我们预期：

1. 去掉周期边界约束会导致周期首尾处形变不连续，进而影响时间平滑度。
2. 去掉置信度加权会降低模型对低质量观测的鲁棒性。
3. 去掉法向约束会使局部表面质量下降。
4. 去掉二阶时间正则或相位一致性约束会显著增加表面抖动。

### 6. 推荐实验命令

快速验证整套实验流程：

```bash
python scripts/run_experiments.py --mode fast-dev --experiment-set both
```

正式运行方法对比实验：

```bash
python scripts/run_experiments.py --mode full-paper --experiment-set method-comparison
```

正式运行 `CPD-Field` 消融实验：

```bash
python scripts/run_experiments.py --mode full-paper --experiment-set cpd-ablation
```

若需覆盖关键损失项权重，可使用：

```bash
python scripts/run_experiments.py \
  --mode full-paper \
  --experiment-set cpd-ablation \
  --normal-weight 0.15 \
  --temporal-weight 0.10 \
  --temporal-acceleration-weight 0.05 \
  --phase-consistency-weight 0.05 \
  --periodicity-weight 0.10 \
  --deformation-weight 0.01
```