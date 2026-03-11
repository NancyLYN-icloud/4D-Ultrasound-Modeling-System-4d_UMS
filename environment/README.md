# 环境打包说明

本目录保存了 `modeling_py310` 的 Conda 环境导出结果，供新设备快速重建。

## 文件说明

- `modeling_py310.history.yml`：只包含手工安装的核心依赖，最适合跨设备、跨时间重新解算依赖。
- `modeling_py310.environment.yml`：包含完整 Conda 依赖但不锁定 build，适合较高还原度的重建。
- `modeling_py310.explicit-linux-64.txt`：精确锁定到当前 Linux 平台的包下载地址，最适合同平台 1:1 复现。

## 推荐恢复方式

### 方式 1：跨设备优先

```bash
conda env create -n modeling_py310 -f environment/modeling_py310.history.yml
conda activate modeling_py310
```

如果只装出最小环境后还缺少次级依赖，可改用完整导出：

```bash
conda env create -n modeling_py310 -f environment/modeling_py310.environment.yml
conda activate modeling_py310
```

### 方式 2：同平台精确复现

```bash
conda create -n modeling_py310 --file environment/modeling_py310.explicit-linux-64.txt
conda activate modeling_py310
```

## 说明

- `explicit-linux-64` 仅建议在 Linux 同架构设备上使用。
- 如果目标机器的 `conda` 源不同，建议优先使用 `history.yml` 或 `environment.yml`。