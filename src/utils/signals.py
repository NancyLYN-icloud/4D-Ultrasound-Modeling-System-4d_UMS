"""信号处理与周期检测辅助函数。"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def _ensure_odd(window_size: int) -> int:
    """确保平滑窗口为奇数，便于对称加权。"""
    if window_size % 2 == 0:
        window_size += 1
    return max(window_size, 3)


def savitzky_golay(signal: Sequence[float], window_size: int, poly_order: int) -> np.ndarray:
    """手写实现的 Savitzky-Golay 滤波器，避免额外依赖。"""
    window_size = _ensure_odd(window_size)
    poly_order = min(poly_order, window_size - 1)
    half_width = window_size // 2
    idx = np.arange(-half_width, half_width + 1, dtype=float)
    # 构造范德蒙德矩阵并求伪逆
    vandermonde = np.vstack([idx ** i for i in range(poly_order + 1)]).T
    pseudo_inv = np.linalg.pinv(vandermonde)
    # 仅需要零阶滤波系数
    coeffs = pseudo_inv[0]
    padded = np.pad(signal, (half_width, half_width), mode="edge")
    smoothed = np.convolve(padded, coeffs[::-1], mode="valid")
    return smoothed

def detrend_signal(signal: Sequence[float]) -> np.ndarray:
    """去除信号的线性趋势，避免趋势干扰FFT周期检测。"""
    arr = np.asarray(signal)
    x = np.arange(len(arr))
    coeffs = np.polyfit(x, arr, 1)
    trend = np.polyval(coeffs, x)
    return arr - trend


def normalize_signal(signal: Sequence[float]) -> np.ndarray:
    """零均值+单位方差归一化，避免数值尺度干扰峰值检测。"""
    arr = np.asarray(signal, dtype=float)
    arr = arr - np.mean(arr)
    std = np.std(arr) + 1e-8
    return arr / std


def detect_peaks(signal: Sequence[float], distance: int) -> List[int]:
    """简单的峰值检测：保留间隔大于 distance 的局部最大值。"""
    arr = np.asarray(signal)
    peaks: List[int] = []
    last_idx = -distance
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] >= arr[i + 1] and (i - last_idx) >= distance:
            peaks.append(i)
            last_idx = i
    return peaks


def detect_valleys(signal: Sequence[float], distance: int) -> List[int]:
    """谷值检测，用于确定周期边界。"""
    return detect_peaks([-x for x in signal], distance)


def autocorr_period(signal: Sequence[float], sampling_rate: float, min_period: float, max_period: float) -> int | None:
    """使用自相关函数估计信号的周期，返回采样点数。"""
    arr = np.asarray(signal)
    # 计算自相关
    corr = np.correlate(arr - np.mean(arr), arr - np.mean(arr), mode='full')
    corr = corr[len(corr)//2:]  # 只取正延迟部分
    # 找到自相关峰值在允许周期范围内
    min_lag = int(min_period * sampling_rate)
    max_lag = int(max_period * sampling_rate)
    if max_lag >= len(corr):
        max_lag = len(corr) - 1
    if min_lag >= max_lag:
        return None
    lags = np.arange(min_lag, max_lag + 1)
    autocorr_values = corr[lags]
    # 找到第一个显著峰值（排除lag=0）
    peak_idx = np.argmax(autocorr_values[1:]) + 1  # +1 因为排除了0
    threshold = 0.1 * autocorr_values[0]  # 降低阈值到10%
    print(f"[signals] 自相关峰值: {autocorr_values[peak_idx]:.3f}, 阈值: {threshold:.3f} (lag={lags[peak_idx]})")
    if autocorr_values[peak_idx] > threshold:
        period_samples = lags[peak_idx]
        print(f"[signals] 自相关检测周期: {period_samples} 样本 ({period_samples / sampling_rate:.2f}s)")
        return period_samples
    print(f"[signals] 自相关峰值不足，峰值 {autocorr_values[peak_idx]:.3f} <= 阈值 {threshold:.3f}")
    return None


def estimate_cycles(
    timestamps: Sequence[float],
    feature_series: Sequence[float],
    min_cycle_seconds: float,
    max_cycle_seconds: float,
    sampling_rate: float,
    window_size: int | None = None,
    poly_order: int = 3,
) -> List[Tuple[int, int, int]]:
    """结合时域和频域的混合判定：用FFT找到最强基频，然后按间隔搜索谷值。"""
    normalized = normalize_signal(feature_series)
    if window_size is None:
        window_size = int(0.3 * sampling_rate) | 1
    smooth = savitzky_golay(normalized, window_size=window_size, poly_order=poly_order)

    # ---------- 频域分析: 通过 FFT 找到最强基频 ----------
    f0: float | None = None
    period_samples: int | None = None
    if len(smooth) > 3:
        # 去除线性趋势，避免趋势干扰周期检测
        detrended = detrend_signal(smooth)
        freqs = np.fft.rfftfreq(len(detrended), d=1.0 / sampling_rate)
        spectrum = np.abs(np.fft.rfft(detrended))
        # 只在允许的周期频率范围内搜索基频
        valid = (freqs >= 1.0 / max_cycle_seconds) & (freqs <= 1.0 / min_cycle_seconds)
        print(f"[signals] FFT: 有效频率范围 {1.0/max_cycle_seconds:.3f}-{1.0/min_cycle_seconds:.3f}Hz, 找到 {np.sum(valid)} 个候选")
        if np.any(valid):
            # 取谱峰最大的频率
            spectrum_valid = spectrum[valid]
            sorted_indices = np.argsort(spectrum_valid)[::-1]  # 从大到小排序
            idx = sorted_indices[0]  # 默认第一个（最大）
            f0 = freqs[valid][idx]
            period_secs = 1.0 / f0 if f0 > 0 else None
            if period_secs and period_secs > 10.0:  # 如果周期太长，选择第二个峰值
                if len(sorted_indices) > 1:
                    idx = sorted_indices[1]
                    f0 = freqs[valid][idx]
                    period_secs = 1.0 / f0 if f0 > 0 else None
                    print(f"[signals] 周期过长，使用第二个峰值 FFT基频 {f0:.3f}Hz, 周期约 {period_secs:.3f}s")
            if period_secs is not None:
                period_samples = int(round(period_secs * sampling_rate))
                print(f"[signals] FFT基频 {f0:.3f}Hz, 周期约 {period_secs:.3f}s (~{period_samples}样本)")
        else:
            print("[signals] FFT: 未找到有效基频，回退到时域方法")
    else:
        print("[signals] FFT: 信号长度不足，回退到时域方法")

    if period_samples is None or period_samples < 1:
        # 回退到原来的时域方法
        print("[signals] 回退到时域峰谷匹配方法")
        min_distance = max(int(min_cycle_seconds * sampling_rate), 1)
        max_distance = max(int(max_cycle_seconds * sampling_rate), min_distance + 1)
        candidate_peaks = detect_peaks(smooth, distance=min_distance)
        candidate_valleys = detect_valleys(smooth, distance=min_distance)
        candidate_valleys.sort()
        cycles: List[Tuple[int, int, int]] = []
        valley_idx = 0
        for peak in candidate_peaks:
            while valley_idx < len(candidate_valleys) and candidate_valleys[valley_idx] < peak:
                start = candidate_valleys[valley_idx]
                valley_idx += 1
                if valley_idx >= len(candidate_valleys):
                    break
                end = candidate_valleys[valley_idx]
                if min_distance <= (end - start) <= max_distance:
                    cycles.append((start, peak, end))
                    break
        return cycles

    # ---------- 时域搜索: 按FFT周期间隔搜索谷值 ----------
    print(f"[signals] 使用FFT周期 {period_samples}样本 搜索谷值")
    tol = max(int(0.2 * period_samples), 1)  # 允许 ±20% 的漂移
    valleys = detect_valleys(smooth, distance=1)  # 检测所有谷值
    cycles: List[Tuple[int, int, int]] = []
    i = 0
    while i < len(valleys) - 1:
        start = valleys[i]
        target = start + period_samples
        # 找到最接近 target 的谷值
        best_end = None
        min_diff = float('inf')
        for j in range(i + 1, len(valleys)):
            diff = abs(valleys[j] - target)
            if diff < min_diff:
                min_diff = diff
                best_end = valleys[j]
        if best_end is not None and min_diff <= tol:
            end = best_end
            # 选取 start 和 end 之间的一个峰值作为周期的峰位置
            peaks_between = [p for p in detect_peaks(smooth, distance=1) if start < p < end]
            peak = peaks_between[0] if peaks_between else (start + end) // 2
            cycles.append((start, peak, end))
            print(f"[signals] 周期 {len(cycles)-1}: 谷值{i}({start}) -> 谷值{valleys.index(end)}({end}), 峰值{peak}, 间隔{end-start}样本")
            i = valleys.index(end)
        else:
            i += 1  # 找不到合适的结束谷值，跳过
    print(f"[signals] 找到 {len(cycles)} 个周期")
    return cycles


def assign_phase(timestamps: Sequence[float], cycle_bounds: List[Tuple[float, float]]) -> List[float]:
    """根据周期起止时间将任意时间戳映射到 [0,1] 相位。"""
    phases: List[float] = []
    cycle_idx = 0
    for t in timestamps:
        while cycle_idx < len(cycle_bounds) and t > cycle_bounds[cycle_idx][1]:
            cycle_idx += 1
        if cycle_idx >= len(cycle_bounds):
            phases.append(float("nan"))
            continue
        start, end = cycle_bounds[cycle_idx]
        if t < start:
            phases.append(float("nan"))
            continue
        duration = end - start
        if duration <= 0:
            phases.append(float("nan"))
        else:
            phases.append(min(max((t - start) / duration, 0.0), 1.0))
    return phases


def resample_to_bins(phases: Sequence[float], values: Sequence[float], bins: Sequence[float]) -> np.ndarray:
    """按相位分箱后求平均，用于生成代表性周期。"""
    phases_arr = np.asarray(phases)
    values_arr = np.asarray(values)
    result = np.zeros(len(bins) - 1, dtype=float)
    for i in range(len(bins) - 1):
        mask = (phases_arr >= bins[i]) & (phases_arr < bins[i + 1])
        if np.any(mask):
            result[i] = float(np.mean(values_arr[mask]))
        else:
            result[i] = float("nan")
    return result
