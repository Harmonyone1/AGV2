"""Utility helpers for timeframe parsing and conversion."""
from __future__ import annotations

from datetime import timedelta
from typing import Final

_TIME_UNITS: Final[dict[str, int]] = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_timeframe(value: str) -> timedelta:
    """Convert a timeframe string like ``"5m"`` or ``"1h"`` into a :class:`timedelta`."""
    if not value:
        raise ValueError("Timeframe string is empty")
    value = value.strip().lower()
    unit = value[-1]
    if unit not in _TIME_UNITS:
        raise ValueError(f"Unsupported timeframe suffix '{unit}' in '{value}'")
    try:
        amount = float(value[:-1])
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid timeframe value '{value}'") from exc
    seconds = amount * _TIME_UNITS[unit]
    return timedelta(seconds=seconds)


def timeframe_to_polars_duration(value: str) -> str:
    """Return a duration string Polars understands for dynamic grouping."""
    value = value.strip().lower()
    if not value:
        raise ValueError("Timeframe string is empty")
    unit = value[-1]
    if unit not in _TIME_UNITS:
        raise ValueError(f"Unsupported timeframe suffix '{unit}' in '{value}'")
    return value


def timeframe_to_minutes(value: str) -> float:
    """Return the timeframe expressed in minutes (fractional allowed)."""
    delta = parse_timeframe(value)
    return delta.total_seconds() / 60.0


__all__ = [
    "parse_timeframe",
    "timeframe_to_polars_duration",
    "timeframe_to_minutes",
]
