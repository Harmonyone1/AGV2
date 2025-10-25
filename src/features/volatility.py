"""Volatility feature helpers."""
from __future__ import annotations

import numpy as np
import polars as pl


def realized_volatility(window: int = 20, price_col: str = "close") -> pl.Expr:
    log_returns = pl.col(price_col).log().diff()
    return (
        log_returns.pow(2).rolling_mean(window_size=window, min_periods=1).sqrt().alias(f"realized_vol_{window}")
    )


def parkinson_volatility(window: int = 20) -> pl.Expr:
    coef = 1.0 / (4.0 * np.log(2))
    return (
        ((pl.col("high") / pl.col("low")).log().pow(2) * coef)
        .rolling_mean(window_size=window, min_periods=1)
        .sqrt()
        .alias(f"parkinson_vol_{window}")
    )


__all__ = ["realized_volatility", "parkinson_volatility"]
