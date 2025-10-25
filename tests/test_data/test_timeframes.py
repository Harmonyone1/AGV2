import pytest

from data.timeframes import parse_timeframe, timeframe_to_minutes, timeframe_to_polars_duration


def test_parse_timeframe_minutes():
    delta = parse_timeframe("5m")
    assert delta.total_seconds() == 300


def test_timeframe_to_polars_duration():
    assert timeframe_to_polars_duration("15m") == "15m"


def test_timeframe_minutes_hour():
    assert timeframe_to_minutes("1h") == pytest.approx(60.0)
