from pathlib import Path
import textwrap

content = """\
"""Sliding-window utilities for encoder training data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import polars as pl

from features.regime_labels import RegimeConfig, assign_regimes
from features.support_resistance import SupportResistanceConfig, sr_heatmap_series


@dataclass
class WindowConfig:
    window_length: int = 512
    feature_columns: Sequence[str] | None = None
    sr_config: SupportResistanceConfig = field(default_factory=SupportResistanceConfig)
    regime_config: RegimeConfig = field(default_factory=RegimeConfig)
    future_return_horizons: Sequence[int] = (1, 3, 12, 60)


def build_encoder_windows(frame: pl.DataFrame, config: WindowConfig) -> pl.DataFrame:
    if frame.height < config.window_length:
        raise ValueError("Frame shorter than window length")
    feature_cols = list(config.feature_columns or _default_feature_columns(frame))
    if not feature_cols:
        raise ValueError("No numeric feature columns available for windowing")
    if "close" not in frame.columns:
        raise ValueError("Input frame must contain a 'close' column for reward modelling")

    numeric = frame.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    closes = frame["close"].to_numpy()
    timestamps = frame["timestamp"].to_list() if "timestamp" in frame.columns else None
    symbols = frame["symbol"].to_list() if "symbol" in frame.columns else None
    timeframes = frame["timeframe"].to_list() if "timeframe" in frame.columns else None

    sr_heatmaps = sr_heatmap_series(frame, config.sr_config)
    regime_labels = assign_regimes(frame, config.regime_config).to_numpy()
    future_returns = _future_return_matrix(closes, config.future_return_horizons)
    pattern_labels = _pattern_labels(frame)
    max_horizon = max(config.future_return_horizons) if config.future_return_horizons else 0
    limit = frame.height - config.window_length - max_horizon + 1
    if limit <= 0:
        raise ValueError("Not enough rows for requested future horizons")
    rows: List[Dict[str, object]] = []
    for start in range(0, limit):
        end = start + config.window_length
        target_idx = end - 1
        if max_horizon and np.isnan(future_returns[target_idx]).any():
            continue
        window_feats = numeric[start:end]
        row: Dict[str, object] = {
            "features": window_feats.reshape(-1).tolist(),
            "regime_label": int(regime_labels[target_idx]),
            "pattern_label": int(pattern_labels[target_idx]),
            "close": float(closes[target_idx]),
        }
        if timestamps is not None:
            row["timestamp"] = timestamps[target_idx]
        if symbols is not None:
            row["symbol"] = symbols[target_idx]
        if timeframes is not None:
            row["timeframe"] = timeframes[target_idx]
        if sr_heatmaps[target_idx]:
            row["sr_heatmap"] = sr_heatmaps[target_idx]
        if config.future_return_horizons:
            row["future_returns"] = future_returns[target_idx].tolist()
        rows.append(row)
    return pl.DataFrame(rows)


def _default_feature_columns(frame: pl.DataFrame) -> List[str]:
    exclude = {"timestamp", "symbol", "timeframe"}
    cols: List[str] = []
    for col, dtype in zip(frame.columns, frame.dtypes):
        if col in exclude:
            continue
        if hasattr(dtype, "is_numeric") and dtype.is_numeric():
            cols.append(col)
    return cols


def _future_return_matrix(prices: np.ndarray, horizons: Sequence[int]) -> np.ndarray:
    matrix = np.full((len(prices), len(horizons)), np.nan, dtype=np.float32)
    for idx, horizon in enumerate(horizons):
        if horizon <= 0 or horizon >= len(prices):
            continue
        target = prices[:-horizon]
        future = prices[horizon:]
        returns = (future - target) / np.clip(target, 1e-6, None)
        matrix[: len(returns), idx] = returns
    return matrix


def _pattern_labels(frame: pl.DataFrame) -> np.ndarray:
    if "logret_1" not in frame.columns or "logret_3" not in frame.columns:
        return np.zeros(frame.height, dtype=np.int64)
    short = frame["logret_1"].to_numpy()
    mid = frame["logret_3"].to_numpy()
    labels = np.full(frame.height, 2, dtype=np.int64)
    labels[(short > 0) & (mid > 0)] = 0
    labels[(short * mid) < 0] = 1
    return labels


__all__ = ["WindowConfig", "build_encoder_windows"]
"""
Path('src/features/windowing.py').write_text(content)
