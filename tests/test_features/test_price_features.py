import datetime as dt

import polars as pl

from features.price_features import engineer_price_features


def test_engineer_price_features_adds_expected_columns():
    timestamps = pl.datetime_range(
        start=dt.datetime(2024, 1, 1, 0, 0, 0),
        end=dt.datetime(2024, 1, 1, 0, 59, 0),
        interval="1m",
        eager=True,
    )
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [float(i) for i in range(len(timestamps))],
            "high": [float(i + 1.5) for i in range(len(timestamps))],
            "low": [float(i - 0.5) for i in range(len(timestamps))],
            "close": [float(i + 0.2) for i in range(len(timestamps))],
            "volume": [1000.0 + i for i in range(len(timestamps))],
        }
    )
    engineered = engineer_price_features(df)
    for column in ("logret_1", "range_pct", "hour_sin"):
        assert column in engineered.columns
