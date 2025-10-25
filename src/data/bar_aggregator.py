"""Multi-timeframe aggregation utilities."""
from __future__ import annotations

from typing import Sequence

import polars as pl

from data.timeframes import timeframe_to_polars_duration


class BarAggregator:
    """Aggregates base timeframe bars into coarser resolutions using Polars."""

    def __init__(self, *, label: str = "right") -> None:
        self.label = label

    def aggregate(self, frame: pl.DataFrame, targets: Sequence[str]) -> dict[str, pl.DataFrame]:
        if frame.is_empty():
            return {tf: pl.DataFrame() for tf in targets}
        frame = frame.sort("timestamp")
        result: dict[str, pl.DataFrame] = {}
        for tf in targets:
            duration = timeframe_to_polars_duration(tf)
            grouped = frame.group_by_dynamic(
                index_column="timestamp",
                every=duration,
                period=duration,
                closed="left",
                label=self.label,
            ).agg(
                [
                    pl.col("symbol").first().alias("symbol"),
                    pl.lit(tf).alias("timeframe"),
                    pl.col("open").first().alias("open"),
                    pl.col("high").max().alias("high"),
                    pl.col("low").min().alias("low"),
                    pl.col("close").last().alias("close"),
                    pl.col("volume").sum().alias("volume"),
                ]
            )
            result[tf] = grouped.rename({"timestamp": "timestamp"})
        return result


__all__ = ["BarAggregator"]
