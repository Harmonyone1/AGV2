import datetime as dt

import numpy as np
import polars as pl

from rl.environment import TradingEnv
from rl.rewards import RewardConfig


def _make_dataset(tmp_path):
    timestamps = pl.datetime_range(
        start=dt.datetime(2024, 1, 1, 0, 0, 0),
        end=dt.datetime(2024, 1, 1, 0, 0, 0) + dt.timedelta(minutes=59),
        interval="1m",
        eager=True,
    )
    closes = [100 + i * 0.1 for i in range(len(timestamps))]
    features = [[float(i), float(i % 5)] for i in range(len(timestamps))]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["TEST"] * len(timestamps),
            "timeframe": ["1m"] * len(timestamps),
            "close": closes,
            "features": features,
        }
    )
    path = tmp_path / "windows.parquet"
    df.write_parquet(path)
    return path


def _make_embedding_dataset(tmp_path):
    timestamps = pl.datetime_range(
        start=dt.datetime(2024, 1, 2, 0, 0, 0),
        end=dt.datetime(2024, 1, 2, 0, 0, 0) + dt.timedelta(minutes=9),
        interval="1m",
        eager=True,
    )
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["EMB"] * len(timestamps),
            "timeframe": ["5m"] * len(timestamps),
            "close": [1.0 + 0.01 * i for i in range(len(timestamps))],
            "features": [[0.0, 0.0]] * len(timestamps),
            "embedding": [[float(i), float(i + 1), float(i + 2), float(i + 3)] for i in range(len(timestamps))],
        }
    )
    path = tmp_path / "windows_with_emb.parquet"
    df.write_parquet(path)
    return path


def test_trading_env_step(tmp_path):
    data_path = _make_dataset(tmp_path)
    env = TradingEnv(data=data_path, episode_length=16, reward_cfg=RewardConfig(trading_cost_bps=0.0))
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    next_obs, reward, terminated, truncated, info = env.step(2)
    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "price_return" in info and isinstance(info["price_return"], float)


def test_trading_env_prefers_embeddings(tmp_path):
    data_path = _make_embedding_dataset(tmp_path)
    env = TradingEnv(data=data_path, episode_length=5)
    obs, _ = env.reset(seed=1)
    assert env.feature_source == "embedding"
    assert env.observation_space.shape == (4,)
    assert np.array_equal(obs, env.features[env.ptr])
