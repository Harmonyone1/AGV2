"""Support and resistance feature helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import polars as pl


@dataclass
class SupportResistanceConfig:
    window: int = 256
    pivot_span: int = 2
    grid_size: int = 64
    price_range_pct: float = 0.05
    bandwidth: float = 0.02


def sr_heatmap_series(frame: pl.DataFrame, config: SupportResistanceConfig) -> List[List[float]]:
    if frame.height < config.window:
        return [[] for _ in range(frame.height)]
    prices = frame.select(["high", "low", "close"])
    heatmaps: List[List[float]] = [[] for _ in range(frame.height)]
    for end in range(config.window, frame.height + 1):
        window = prices.slice(end - config.window, config.window)
        heatmap = _compute_heatmap(window, config)
        heatmaps[end - 1] = heatmap.tolist()
    return heatmaps


def _compute_heatmap(window: pl.DataFrame, config: SupportResistanceConfig) -> np.ndarray:
    highs = window["high"].to_numpy()
    lows = window["low"].to_numpy()
    closes = window["close"].to_numpy()
    pivot_highs = _fractal_points(highs, config.pivot_span)
    pivot_lows = _fractal_points(lows, config.pivot_span)
    pivots = np.concatenate((pivot_highs, pivot_lows))
    center = closes[-1]
    price_range = max(center * config.price_range_pct, 1e-6)
    grid = np.linspace(center - price_range, center + price_range, config.grid_size)
    if len(pivots) == 0:
        return np.zeros(config.grid_size, dtype=np.float32)
    values = _gaussian_smooth(pivots, grid, config.bandwidth, price_range)
    if values.max() > 0:
        values = values / values.max()
    return values.astype(np.float32)


def _fractal_points(values: np.ndarray, span: int) -> np.ndarray:
    if len(values) < 2 * span + 1:
        return np.array([], dtype=np.float32)
    points: List[float] = []
    for idx in range(span, len(values) - span):
        window = values[idx - span : idx + span + 1]
        mid = values[idx]
        if mid == window.max() or mid == window.min():
            points.append(mid)
    return np.asarray(points, dtype=np.float32)


def _gaussian_smooth(pivots: np.ndarray, grid: np.ndarray, bandwidth: float, scale: float) -> np.ndarray:
    sigma = max(np.std(pivots) * bandwidth, scale * bandwidth, 1e-3)
    diffs = grid[None, :] - pivots[:, None]
    values = np.exp(-0.5 * (diffs / sigma) ** 2).sum(axis=0)
    return values


__all__ = ["SupportResistanceConfig", "sr_heatmap_series"]
