"""Data quality checks for aggregated bars."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import polars as pl

from data.timeframes import parse_timeframe


@dataclass
class SeriesQuality:
    symbol: str
    timeframe: str
    total_rows: int
    duplicate_rows: int
    missing_bars: int
    max_gap_bars: int


def run_quality_checks(frame: pl.DataFrame) -> List[SeriesQuality]:
    if frame.is_empty():
        return []
    reports: List[SeriesQuality] = []
    for symbol in frame["symbol"].unique().to_list():
        symbol_frame = frame.filter(pl.col("symbol") == symbol)
        for timeframe in symbol_frame["timeframe"].unique().to_list():
            group = symbol_frame.filter(pl.col("timeframe") == timeframe).sort("timestamp")
            reports.append(_analyze_group(symbol, timeframe, group))
    return reports


def enforce_quality(frame: pl.DataFrame, max_missing: int = 0) -> None:
    for report in run_quality_checks(frame):
        if report.missing_bars > max_missing:
            raise ValueError(
                f"Detected {report.missing_bars} missing bars for {report.symbol}-{report.timeframe}"
            )


def _analyze_group(symbol: str, timeframe: str, group: pl.DataFrame) -> SeriesQuality:
    timestamps = group["timestamp"].to_list()
    total = len(timestamps)
    duplicates = total - len(set(timestamps))
    expected_seconds = parse_timeframe(timeframe).total_seconds()
    missing = 0
    max_gap = 0
    if expected_seconds > 0 and total > 1:
        gaps = []
        for prev, curr in zip(timestamps[:-1], timestamps[1:]):
            delta = (curr - prev).total_seconds()
            if delta > expected_seconds:
                gap_bars = int(round(delta / expected_seconds))
                gaps.append(gap_bars)
        if gaps:
            missing = sum(g - 1 for g in gaps if g > 1)
            max_gap = max(gaps)
    return SeriesQuality(
        symbol=symbol,
        timeframe=timeframe,
        total_rows=total,
        duplicate_rows=int(duplicates),
        missing_bars=int(missing),
        max_gap_bars=int(max_gap),
    )


__all__ = ["SeriesQuality", "run_quality_checks", "enforce_quality"]
