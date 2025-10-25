import datetime as dt

import numpy as np
import polars as pl

from features.windowing import WindowConfig, build_encoder_windows


def _dummy_frame(length: int = 128) -> pl.DataFrame:
    timestamps = pl.datetime_range(
        start=dt.datetime(2024, 1, 1, 0, 0, 0),
        end=dt.datetime(2024, 1, 1, 0, 0, 0) + dt.timedelta(minutes=length - 1),
        interval="1m",
        eager=True,
    )
    prices = np.linspace(100, 110, num=length)
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["TEST"] * length,
            "timeframe": ["1m"] * length,
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices + 0.1,
            "volume": [1000.0 + i for i in range(length)],
        }
    )


def test_build_encoder_windows_returns_expected_count():
    frame = _dummy_frame(80)
    config = WindowConfig(window_length=32, future_return_horizons=(1, 3), feature_columns=("close",))
    windows = build_encoder_windows(frame, config)
    expected = 80 - 32 - 3 + 1
    assert windows.height == expected
    assert windows["features"].len() == expected
    assert all(len(window) == 32 for window in windows["features"])
    assert "close" in windows.columns
    assert "timestamp" in windows.columns
    assert windows["symbol"].unique()[0] == "TEST"
