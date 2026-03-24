# 4D 胃部超声重建实验设计

本文档给出 4D Ultrasound Modeling System (4D-UMS) 的系统化实验验证框架。实验目标是从几何精度、时间一致性与真实场景适用性三个层面，验证所提方法的有效性。

当前实验脚本入口为 [scripts/run_experiments.py](scripts/run_experiments.py)，已支持：

- `--mode fast-dev`：快速调试模式，用于打通流程和验证趋势
- `--mode full-paper`：正式论文实验模式，用于生成最终结果
- `--experiment-set method-comparison`：方法对比实验
- `--experiment-set cpd-ablation`：当前主方法消融实验（保留旧名称以兼容历史命令）
- `--experiment-set both`：依次运行上述两类实验

## 1. 实验一：合成数据定量精度评估

**目标**：在可获得真值网格的合成胃部动态数据上，量化不同方法重建 4D 网格的几何精度，并分析从静态逐相位建模到 `动态共享-全局基残差` 主方法的提升。

### 方法设计
- **数据生成**：使用 [scripts/generate_test_gastric_dataset.py](scripts/generate_test_gastric_dataset.py) 构造具有真实蠕动形变的动态胃部合成数据。
- **基线对比**：
    - **静态基线**：关闭非线性相位标准化，关闭动态共享模型。
    - **静态增强**：开启非线性相位标准化，保留逐相位静态重建。
    - **动态共享-全局基残差**：开启当前主方法。
- **控制变量**：
    - **模式切换**：使用 `fast-dev` 快速模式先验证趋势，使用 `full-paper` 模式生成正式结果。
    - **真值比较**：优先将预测网格与 `simuilate_data/meshes/` 中的相位级动态仿真 GT 网格逐相位对齐比较；`test/stomach.ply` 仅作为静态参考点云，不再适合作为动态重建的默认 GT。

### 评价指标
- **Chamfer Distance (CD)**：衡量重建网格与真值网格之间的对称平均距离。
- **Hausdorff Distance (HD95)**：取 95 分位的 Hausdorff 距离，兼顾边界误差与鲁棒性。
- **水密比例**：统计导出网格中 water-tight mesh 的占比。
- **平均点云置信度**：衡量观测质量控制模块的贡献。

### 对应输出文件
- `method_comparison.csv`
- `method_comparison.tex`
- `run.log`
- `run_metadata.json`
- `configs/*.json`
- `artifacts/` 下的点云、网格与摘要文件

### 预期表格：方法对比结果
| 方法 | 模型类型 | 平均 CD | 平均 HD95 | 时间平滑度 | 水密比例 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 静态基线 | 逐相位静态模型 | ... | ... | ... | ... |
| 静态增强 | 逐相位静态模型 | ... | ... | ... | ... |
| 动态共享-全局基残差 | `shared_topology_global_basis_residual` | **...** | **...** | **...** | **...** |

---

## 2. 实验二：时间一致性与消融实验

**目标**：量化当前主方法中关键损失项对 4D 重建稳定性的影响，验证周期边界、法向约束、二阶时间正则与置信度加权的必要性。

### 方法设计
- **消融组设置**：
    1. **完整 `动态共享-全局基残差`**
    2. **去掉周期边界约束**
    3. **去掉置信度加权**
    4. **去掉法向约束**
    5. **去掉二阶时间正则**
    6. **去掉相位一致性约束**
- **评估方式**：比较各变体的时间平滑度、CD、HD95 与水密比例，分析哪些约束对时序稳定和几何精度最关键。

### 评价指标
- **时间平滑度分数**：使用 [src/modeling/metrics.py](src/modeling/metrics.py) 中的时间平滑度指标，计算相邻相位表面点的平均位移，记为 $\bar{v}$。
- **平均 CD / 平均 HD95**：衡量移除某项约束后几何误差的变化。
- **平均点云置信度**：验证置信度模块与动态损失耦合后的收益。

### 对应输出文件
- `cpd_ablation.csv`
- `cpd_ablation.tex`
- `run.log`
- `run_metadata.json`
- `configs/*.json`
- `artifacts/` 下的点云、网格与摘要文件

### 预期表格：主方法消融结果
| 模型变体 | 平均 CD | 平均 HD95 | 时间平滑度 | 水密比例 |
| :--- | :---: | :---: | :---: | :---: |
| `动态共享-全局基残差` | **...** | **...** | **...** | **...** |
| w/o 周期边界 | ... | ... | ... | ... |
| w/o 置信度加权 | ... | ... | ... | ... |
| w/o 法向约束 | ... | ... | ... | ... |
| w/o 二阶时间正则 | ... | ... | ... | ... |
| w/o 相位一致性 | ... | ... | ... | ... |

---

## 3. 实验三：真实数据定性验证

**目标**：在缺乏真值网格的真实自由手胃部超声数据上，验证当前主方法在复杂成像条件下的稳定性、拓扑合理性与解剖可解释性。

### 方法设计
- **可视化对比**：
    - **点云与网格叠加**：展示相位对齐后的原始点云与最终水密网格的对应关系。
    - **静态模型 vs 当前主方法**：展示两类方法在局部连续性与形变稳定性上的差别。
    - **截面或局部放大分析**：展示胃壁边界、胃窦区域及局部形变细节。
- **拓扑检查**：验证重建网格是否保持流形结构、是否闭合、是否具备较好的水密性。

### 预期图示：一个完整蠕动周期的 4D 演化
- 布局：使用 4x4 或 3x5 的相位网格视图展示从 $0\%$ 到 $100\%$ 蠕动周期的形态变化
- 强调重点：胃窦与胃体区域的连续形变，以及各相位之间的平滑过渡

---

## 数据准备与实验流程概览

| 步骤 | 脚本或模块 | 输入 | 输出 |
| :--- | :--- | :--- | :--- |
| 合成数据生成 | [scripts/generate_test_gastric_dataset.py](scripts/generate_test_gastric_dataset.py) | 配置参数 | 真值模型、监测流、扫描流 |
| 主流程处理 | [src/pipelines/multicycle_reconstruction.py](src/pipelines/multicycle_reconstruction.py) | 合成或真实扫描数据 | 相位点云、静态网格、动态网格 |
| 实验调度 | [scripts/run_experiments.py](scripts/run_experiments.py) | 数据路径、模式、实验集 | 方法对比结果、消融结果 |
| 指标计算 | [src/modeling/metrics.py](src/modeling/metrics.py) | 预测网格与真值网格 | CD、HD95、时间平滑度结果 |

## 推荐运行命令

### 1. 快速调试

```bash
python scripts/run_experiments.py --mode fast-dev --experiment-set both
```

### 2. 正式方法对比

```bash
python scripts/run_experiments.py --mode full-paper --experiment-set method-comparison
```

### 3. 正式主方法消融

```bash
python scripts/run_experiments.py --mode full-paper --experiment-set cpd-ablation
```

### 4. 覆盖主方法关键权重

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

## 实验归档约定

默认情况下，实验脚本会把结果统一写入 `/home/liuyanan/data/Research_Data/4D-UMS/experiments`，并为每次运行创建一个独立子目录。目录名包含时间戳以及本次运行模式，便于长期保留、复现实验和整理论文材料。

如需自定义归档根目录，可通过 `--out-dir` 指定；如需给本次实验添加可读标签，可通过 `--run-name` 指定。