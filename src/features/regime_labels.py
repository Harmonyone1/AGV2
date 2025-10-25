"""Regime classification helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import polars as pl

REGIME_TO_ID: Dict[str, int] = {
    "trend_up": 0,
    "trend_down": 1,
    "consolidation": 2,
    "volatile": 3,
}


@dataclass
class RegimeConfig:
    trend_window: int = 64
    vol_window: int = 128
    slope_threshold: float = 0.0005
    vol_threshold: float = 0.01


def assign_regimes(frame: pl.DataFrame, config: RegimeConfig = RegimeConfig()) -> pl.Series:
    closes = frame["close"].to_numpy()
    log_returns = np.diff(np.log(closes), prepend=np.nan)
    vol = _rolling_std(log_returns, config.vol_window)
    slope = _trend_slope(closes, config.trend_window)
    labels = np.full(frame.height, REGIME_TO_ID["consolidation"], dtype=np.int64)
    labels[(slope > config.slope_threshold) & (vol < config.vol_threshold)] = REGIME_TO_ID["trend_up"]
    labels[(slope < -config.slope_threshold) & (vol < config.vol_threshold)] = REGIME_TO_ID["trend_down"]
    labels[vol >= config.vol_threshold] = REGIME_TO_ID["volatile"]
    return pl.Series("regime_label", labels)


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, 0.0, dtype=np.float64)
    if window <= 1:
        return np.nan_to_num(np.abs(values), nan=0.0)
    for idx in range(window - 1, len(values)):
        segment = values[idx - window + 1 : idx + 1]
        out[idx] = np.nanstd(segment)
    return np.nan_to_num(out, nan=0.0)


def _trend_slope(values: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if window <= 1:
        return out
    for idx in range(window, len(values)):
        prev = values[idx - window]
        denom = max(abs(prev), 1e-6) * window
        out[idx] = (values[idx] - prev) / denom
    return out


__all__ = ["RegimeConfig", "assign_regimes", "REGIME_TO_ID"]
