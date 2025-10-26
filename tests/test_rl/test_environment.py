
import datetime as dt

import numpy as np
import polars as pl

from rl.environment import TradingEnv
from rl.rewards import RewardConfig


def _make_dataset(tmp_path, symbol="TEST"):
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
            "symbol": [symbol] * len(timestamps),
            "timeframe": ["1m"] * len(timestamps),
            "close": closes,
            "features": features,
        }
    )
    path = tmp_path / f"{symbol.lower()}_windows.parquet"
    df.write_parquet(path)
    return path


def _make_price_series(tmp_path, prices, symbol="SIM"):
    timestamps = pl.datetime_range(
        start=dt.datetime(2024, 1, 3, 0, 0, 0),
        end=dt.datetime(2024, 1, 3, 0, 0, 0) + dt.timedelta(minutes=len(prices) - 1),
        interval="1m",
        eager=True,
    )
    features = [[float(i), 0.0] for i in range(len(prices))]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": [symbol] * len(prices),
            "timeframe": ["1m"] * len(prices),
            "close": prices,
            "features": features,
        }
    )
    path = tmp_path / f"{symbol.lower()}_series.parquet"
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


def _find_action(env: TradingEnv, position: float, order_type: str = "market", *, limit_offset: float | None = None, tp: float | None = None, sl: float | None = None):
    for idx, (pos, otype, offset, tp_bps, sl_bps) in enumerate(env._action_map):
        if (
            abs(pos - position) < 1e-9
            and otype == order_type
            and (limit_offset is None or offset == limit_offset)
            and (tp is None or tp_bps == tp)
            and (sl is None or sl_bps == sl)
        ):
            return idx
    raise AssertionError("Requested action not found in action map")


def test_trading_env_step(tmp_path):
    data_path = _make_dataset(tmp_path)
    env = TradingEnv(data=data_path, episode_length=16, reward_cfg=RewardConfig(trading_cost_bps=0.0))
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["symbol"] == "TEST"
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


def test_trading_env_tradelocker_costs(tmp_path):
    data_path = _make_dataset(tmp_path, symbol="ETH")
    reward_cfg = RewardConfig(trading_cost_bps=0.0, holding_cost_bps=0.0, reward_clip=None)
    env = TradingEnv(data=data_path, episode_length=8, reward_cfg=reward_cfg)
    env.reset(seed=2)
    _, reward, _, _, info = env.step(2)
    execution = info["execution"]
    assert execution["trade_cost_bps"] > 0.0
    assert execution["spread_bps"] > 0.0
    assert info["symbol"] == "ETH"
    raw_pnl = env.position * info["price_return"]
    assert reward <= raw_pnl


def test_limit_order_behavior(tmp_path):
    data_path = _make_dataset(tmp_path, symbol="SIML")
    reward_cfg = RewardConfig(trading_cost_bps=0.0, holding_cost_bps=0.0, reward_clip=None)
    env = TradingEnv(
        data=data_path,
        symbol="SIML",
        episode_length=16,
        reward_cfg=reward_cfg,
        order_types=["market", "limit"],
        limit_order_fill_bps=0.1,
    )
    env.reset(seed=0)
    limit_long_action = _find_action(env, env.max_position, order_type="limit")
    _, _, _, _, info = env.step(limit_long_action)
    assert info["execution"]["order_type"] == "limit"
    assert info["execution"]["limit_status"] == "limit_blocked"
    assert info["position"] == 0.0


def test_limit_order_hits_when_price_favorable(tmp_path):
    prices = [100.0, 99.8, 99.7, 99.6, 99.5]
    data_path = _make_price_series(tmp_path, prices, symbol="HIT")
    reward_cfg = RewardConfig(trading_cost_bps=0.0, holding_cost_bps=0.0, reward_clip=None)
    env = TradingEnv(
        data=data_path,
        symbol="HIT",
        episode_length=5,
        reward_cfg=reward_cfg,
        order_types=["market", "limit"],
        limit_order_fill_bps=5.0,
    )
    env.reset(options={"start_index": 1})
    limit_long_action = _find_action(env, env.max_position, order_type="limit")
    _, _, _, _, info = env.step(limit_long_action)
    assert info["execution"]["limit_status"] == "limit_hit"
    assert info["position"] == env.max_position


def test_limit_order_offset_selection(tmp_path):
    prices = [100.0, 99.9, 99.8, 99.7]
    data_path = _make_price_series(tmp_path, prices, symbol="OFF")
    reward_cfg = RewardConfig(trading_cost_bps=0.0, holding_cost_bps=0.0, reward_clip=None)
    env = TradingEnv(
        data=data_path,
        symbol="OFF",
        episode_length=4,
        reward_cfg=reward_cfg,
        order_types=["market", "limit"],
        limit_order_fill_bps=5.0,
        limit_offset_bps=[1.0, 8.0],
    )
    limit_action = _find_action(env, env.max_position, order_type="limit", limit_offset=1.0)
    env.reset(options={"start_index": 1})
    _, _, _, _, info = env.step(limit_action)
    assert info["execution"]["limit_offset_bps"] == 1.0


def test_bracket_take_profit(tmp_path):
    prices = [100.0, 100.3, 100.6]
    data_path = _make_price_series(tmp_path, prices, symbol="BTP")
    reward_cfg = RewardConfig(trading_cost_bps=0.0, holding_cost_bps=0.0, reward_clip=None)
    env = TradingEnv(
        data=data_path,
        symbol="BTP",
        episode_length=3,
        reward_cfg=reward_cfg,
        order_types=["market"],
        bracket_take_profit_bps=[5.0],
        position_sizes=[0.0, 1.0],
    )
    env.reset(options={"start_index": 1})
    action = _find_action(env, env.max_position, tp=5.0)
    _, _, _, _, info = env.step(action)
    assert info["execution"]["bracket_trigger"] == "take_profit"
    assert info["position"] == 0.0


def test_bracket_stop_loss(tmp_path):
    prices = [100.0, 99.7, 99.3]
    data_path = _make_price_series(tmp_path, prices, symbol="BSL")
    reward_cfg = RewardConfig(trading_cost_bps=0.0, holding_cost_bps=0.0, reward_clip=None)
    env = TradingEnv(
        data=data_path,
        symbol="BSL",
        episode_length=3,
        reward_cfg=reward_cfg,
        order_types=["market"],
        bracket_stop_loss_bps=[5.0],
        position_sizes=[0.0, 1.0],
    )
    env.reset(options={"start_index": 1})
    action = _find_action(env, env.max_position, sl=5.0)
    _, _, _, _, info = env.step(action)
    assert info["execution"]["bracket_trigger"] == "stop_loss"
    assert info["position"] == 0.0


def test_position_size_levels(tmp_path):
    data_path = _make_dataset(tmp_path, symbol="SIZE")
    env = TradingEnv(
        data=data_path,
        symbol="SIZE",
        episode_length=8,
        reward_cfg=RewardConfig(),
        position_sizes=[-0.5, 0.0, 0.75],
    )
    positions = {spec[0] for spec in env._action_map}
    assert env.max_position * 0.75 in positions
    assert -env.max_position * 0.5 in positions
