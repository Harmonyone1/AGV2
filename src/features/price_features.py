"""Price-action feature engineering utilities."""
from __future__ import annotations

from math import tau
from typing import Dict, Sequence

import polars as pl


DEFAULT_HORIZONS = (1, 3, 12, 60)


def engineer_price_features(
    frame: pl.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    session_config: Dict[str, object] | None = None,
) -> pl.DataFrame:
    """Return a copy of ``frame`` with canonical price-action features appended."""
    if frame.is_empty():
        return frame
    expressions = []
    log_close = pl.col("close").log()
    for horizon in horizons:
        expressions.append((log_close - log_close.shift(horizon)).alias(f"logret_{horizon}"))
    expressions.extend(
        [
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range_pct"),
            ((pl.col("close") - pl.col("open")) / pl.col("close")).alias("body_pct"),
            ((pl.col("high") - pl.max_horizontal(pl.col("open"), pl.col("close"))) / pl.col("close")).alias(
                "upper_wick"
            ),
            ((pl.min_horizontal(pl.col("open"), pl.col("close")) - pl.col("low")) / pl.col("close")).alias(
                "lower_wick"
            ),
            (
                (pl.col("open") - pl.col("close").shift(1)) / pl.col("close").shift(1)
            ).alias("gap_pct"),
            pl.col("timestamp").dt.weekday().alias("dow"),
            pl.col("timestamp").dt.hour().alias("hour"),
        ]
    )
    with_features = frame.sort("timestamp").with_columns(expressions)
    with_features = with_features.with_columns(
        [
            (pl.col("hour") * tau / 24).sin().alias("hour_sin"),
            (pl.col("hour") * tau / 24).cos().alias("hour_cos"),
        ]
    )
    with_features = with_features.drop("hour")
    if session_config and session_config.get("enabled", False):
        with_features = with_features.with_columns(_session_columns(session_config))
    return with_features


def _session_columns(config: Dict[str, object]):
    sessions = config.get("sessions", [])
    expressions = []
    for session in sessions:
        name = str(session.get("name", "session")).lower()
        start_minutes = _parse_minutes(session.get("start", "00:00"))
        end_minutes = _parse_minutes(session.get("end", "23:59"))
        duration = pl.col("timestamp").dt.hour() * 60 + pl.col("timestamp").dt.minute()
        if end_minutes >= start_minutes:
            mask = duration.is_between(start_minutes, end_minutes, closed="both")
        else:
            mask = (duration >= start_minutes) | (duration <= end_minutes)
        expressions.append(mask.cast(pl.Int8).alias(f"session_{name}"))
    return expressions


def _parse_minutes(value: str | object) -> int:
    text = str(value)
    hours, minutes = text.split(":")
    return int(hours) * 60 + int(minutes)


__all__ = ["engineer_price_features", "DEFAULT_HORIZONS"]
