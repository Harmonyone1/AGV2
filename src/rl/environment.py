
"""Gymnasium-compatible trading environment with TradeLocker-style execution."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from rl.market_config import BacktestSettings, CostSpec, load_cost_spec, load_market_spec
from rl.rewards import RewardConfig, compute_reward

SessionRule = Tuple[time, time, ZoneInfo]
logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    data_path: str
    symbol: str | None = None
    episode_length: int = 256
    trading_cost_bps: float = 1.0
    max_position: float = 1.0
    reward: RewardConfig = field(default_factory=RewardConfig)
    include_position_in_obs: bool = True  # Include position state in observations


class TradingEnv(gym.Env):
    """Single-symbol trading environment with discrete order semantics."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: str | pl.DataFrame,
        episode_length: int = 256,
        trading_cost_bps: float = 1.0,
        max_position: float = 1.0,
        reward_cfg: RewardConfig | None = None,
        *,
        symbol: str | None = None,
        market_config_path: str = "config/markets.yaml",
        cost_config_path: str = "config/costs.yaml",
        execution_profile: str = "backtesting",
        order_types: Sequence[str] | None = None,
        limit_order_fill_bps: float = 5.0,
        limit_offset_bps: Sequence[float] | None = None,
        bracket_take_profit_bps: Sequence[float] | None = None,
        bracket_stop_loss_bps: Sequence[float] | None = None,
        position_sizes: Sequence[float] | None = None,
        include_position_in_obs: bool = True,
        enable_logging: bool = False,
    ) -> None:
        super().__init__()
        self.enable_logging = enable_logging
        self.frame = self._load_frame(data)
        if "features" not in self.frame.columns or "close" not in self.frame.columns:
            raise ValueError("Dataset must contain 'features' and 'close' columns")

        vector_column = "embedding" if "embedding" in self.frame.columns else "features"
        vectors = [np.asarray(vec, dtype=np.float32) for vec in self.frame[vector_column].to_list()]
        self.features = np.stack(vectors)
        self.feature_source = vector_column
        self.prices = self.frame["close"].to_numpy()
        self.timestamps = self.frame.get_column("timestamp").to_list() if "timestamp" in self.frame.columns else None

        # Check for pre-computed volatility (use default window 30)
        self.precomputed_vol = None
        if "realized_vol_30" in self.frame.columns:
            self.precomputed_vol = self.frame["realized_vol_30"].to_numpy()
        elif "parkinson_vol_30" in self.frame.columns:
            self.precomputed_vol = self.frame["parkinson_vol_30"].to_numpy()

        self.symbol = symbol or self._infer_symbol()
        self.market_spec = None
        self.cost_spec: CostSpec | None = None
        self.execution_settings: BacktestSettings | None = None
        if self.symbol:
            try:
                self.market_spec = load_market_spec(self.symbol, market_config_path)
            except (FileNotFoundError, KeyError):
                self.market_spec = None
            try:
                self.cost_spec, self.execution_settings = load_cost_spec(
                    self.symbol, cost_config_path, profile=execution_profile
                )
            except (FileNotFoundError, KeyError):
                self.cost_spec = None
                self.execution_settings = None

        self.reward_cfg = reward_cfg or RewardConfig(trading_cost_bps=trading_cost_bps)
        self.episode_length = episode_length
        self.trading_cost_bps = trading_cost_bps
        self.max_position = max_position
        self.execution_profile = execution_profile
        self.include_position_in_obs = include_position_in_obs

        self.order_types = tuple(order_types or ["market"])
        self.limit_order_fill_bps = float(limit_order_fill_bps)
        limit_offsets = limit_offset_bps or ([self.limit_order_fill_bps] if "limit" in self.order_types else [])
        self.limit_offsets = tuple(float(x) for x in limit_offsets)
        base_sizes = position_sizes or [-1.0, 0.0, 1.0]
        if not base_sizes:
            raise ValueError("position_sizes must include at least one entry")
        scaled = [float(sz) * self.max_position for sz in base_sizes]
        self._position_levels = tuple(np.clip(scaled, -self.max_position, self.max_position))
        self._tp_levels = tuple(float(x) for x in bracket_take_profit_bps or [])
        self._sl_levels = tuple(float(x) for x in bracket_stop_loss_bps or [])
        self._action_map: list[tuple[float, str, float | None, float | None, float | None]] = []
        self._build_action_map()

        self.action_space = spaces.Discrete(len(self._action_map))
        feature_dim = self.features.shape[1]
        # Add position state dimensions if enabled: [position, equity, steps_in_position]
        obs_dim = feature_dim + 3 if self.include_position_in_obs else feature_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._np_random, _ = gym.utils.seeding.np_random()
        self._cost_noise: Dict[str, float] = {"spread": 1.0, "slippage": 1.0, "commission": 1.0}
        self._step_seconds = self._infer_step_seconds()
        self._steps_per_year = (365.0 * 24.0 * 3600.0) / self._step_seconds if self._step_seconds > 0 else 0.0
        self._session_rules: Sequence[SessionRule] = ()
        self._sessions_enabled = False
        self._gaps_expected = False
        self._init_sessions()
        self._reset_state()

    def _build_action_map(self) -> None:
        base_actions: list[tuple[float, str, float | None]] = []
        for order in self.order_types:
            for pos in self._position_levels:
                if order == "limit" and self.limit_offsets:
                    for offset in self.limit_offsets:
                        base_actions.append((pos, order, offset))
                else:
                    base_actions.append((pos, order, None))
        if self._tp_levels or self._sl_levels:
            tp_candidates = self._tp_levels or (None,)
            sl_candidates = self._sl_levels or (None,)
            for pos, order, offset in base_actions:
                for tp in tp_candidates:
                    for sl in sl_candidates:
                        self._action_map.append((pos, order, offset, tp, sl))
        else:
            for pos, order, offset in base_actions:
                self._action_map.append((pos, order, offset, None, None))

    def _reset_state(self) -> None:
        self.start_idx = 0
        self.ptr = 0
        self.position = 0.0
        self.prev_price = 0.0
        self.equity = 0.0
        self.step_count = 0
        self.steps_in_position = 0
        self._terminated = False
        self._refresh_cost_noise()

    def _build_observation(self, feature_vec: np.ndarray) -> np.ndarray:
        """Augment feature vector with position state if enabled."""
        if not self.include_position_in_obs:
            return feature_vec
        # Append [position, equity, steps_in_position]
        position_state = np.array([
            self.position / max(self.max_position, 1e-6),  # Normalized position
            self.equity / max(self.reward_cfg.initial_equity, 1e-6),  # Normalized equity
            self.steps_in_position / max(self.episode_length, 1.0),  # Normalized time in position
        ], dtype=np.float32)
        return np.concatenate([feature_vec, position_state])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)
        self._reset_state()
        max_start = max(2, len(self.prices) - self.episode_length - 1)
        if options and "start_index" in options:
            candidate = int(options["start_index"])
            self.start_idx = min(max(1, candidate), len(self.prices) - 2)
        else:
            self.start_idx = int(self._np_random.integers(1, max_start)) if max_start > 2 else 1
        self.ptr = self.start_idx
        self.prev_price = float(self.prices[self.ptr - 1])
        observation = self._build_observation(self.features[self.ptr])

        if self.enable_logging:
            logger.info(
                "episode_start",
                extra={
                    "start_idx": self.start_idx,
                    "episode_length": self.episode_length,
                    "timestamp": self._timestamp(self.ptr),
                    "symbol": self.symbol,
                    "initial_price": float(self.prices[self.ptr]),
                },
            )

        info = {
            "position": self.position,
            "timestamp": self._timestamp(self.ptr),
            "symbol": self.symbol,
            "session_open": self._is_session_open(self.ptr),
        }
        return observation, info

    def step(self, action: int):  # type: ignore[override]
        if self._terminated:
            raise RuntimeError("Call reset() before stepping a terminated episode")
        target_position, order_type, limit_offset_bps, tp_bps, sl_bps = self._decode_action(action)
        session_open = self._is_session_open(self.ptr)

        requested_position = target_position
        if not session_open:
            target_position = self.position
        raw_delta = target_position - self.position
        limit_status = "na"
        bracket_trigger = None
        if session_open and order_type == "limit" and abs(raw_delta) > 1e-9:
            if self._limit_order_filled(target_position, raw_delta, limit_offset_bps):
                limit_status = "limit_hit"
            else:
                limit_status = "limit_blocked"
                target_position = self.position
                raw_delta = 0.0
        filled_delta = self._apply_fill_ratio(raw_delta)
        new_position = np.clip(self.position + filled_delta, -self.max_position, self.max_position)

        price = float(self.prices[self.ptr])
        price_return = (price - self.prev_price) / max(self.prev_price, 1e-6)
        realized_vol = self._estimate_realized_vol(self.ptr)
        trade_cost_bps, execution_info = self._compute_transaction_cost_bps(filled_delta, price_return, order_type)
        holding_cost_bps = self._compute_holding_cost_bps(new_position)

        bracket_trigger = self._check_brackets(new_position, tp_bps, sl_bps, price_return)
        reward = compute_reward(
            position=new_position,
            prev_position=self.position,
            price_return=price_return,
            realized_vol=realized_vol,
            cfg=self.reward_cfg,
            trade_cost_bps=trade_cost_bps,
            holding_cost_bps=holding_cost_bps,
            equity=self.equity if self.equity != 0.0 else self.reward_cfg.initial_equity,
        )
        if bracket_trigger:
            new_position = 0.0
            if self.enable_logging:
                logger.info(
                    "bracket_triggered",
                    extra={
                        "trigger_type": bracket_trigger,
                        "position_before": self.position,
                        "price_return": price_return,
                        "equity": self.equity,
                    },
                )

        # Track position changes for logging
        position_changed = abs(new_position - self.position) > 1e-6

        # Track steps in position
        if abs(new_position) > 1e-9:
            self.steps_in_position += 1
        else:
            self.steps_in_position = 0

        self.position = new_position
        self.equity += reward
        self.prev_price = price
        self.ptr += 1
        self.step_count += 1

        # Log position changes
        if self.enable_logging and position_changed:
            logger.info(
                "position_change",
                extra={
                    "step": self.step_count,
                    "timestamp": self._timestamp(self.ptr),
                    "price": price,
                    "new_position": new_position,
                    "position_delta": filled_delta,
                    "order_type": order_type,
                    "limit_status": limit_status,
                    "trade_cost_bps": trade_cost_bps,
                    "reward": reward,
                    "equity": self.equity,
                    "symbol": self.symbol,
                },
            )

        dataset_limit = len(self.prices) - 1
        episode_limit = self.start_idx + self.episode_length
        terminated = self.ptr >= min(dataset_limit, episode_limit)
        self._terminated = terminated

        if self.enable_logging and terminated:
            logger.info(
                "episode_end",
                extra={
                    "final_equity": self.equity,
                    "total_steps": self.step_count,
                    "final_position": self.position,
                    "timestamp": self._timestamp(min(self.ptr, len(self.prices) - 1)),
                },
            )

        obs_index = min(self.ptr, len(self.features) - 1)
        observation = self._build_observation(self.features[obs_index])
        info = {
            "position": self.position,
            "timestamp": self._timestamp(obs_index),
            "price": price,
            "price_return": price_return,
            "equity": self.equity,
            "step": self.step_count,
            "feature_source": self.feature_source,
            "symbol": self.symbol,
            "realized_vol": realized_vol,
            "raw_action": action,
            "session_open": session_open,
            "holding_cost_bps": holding_cost_bps,
        }
        fill_ratio = 0.0 if abs(raw_delta) < 1e-9 else filled_delta / raw_delta
        execution_info.update(
            {
                "trade_cost_bps": trade_cost_bps,
                "requested_position": requested_position,
                "target_position": target_position,
                "fill_ratio": fill_ratio,
                "session_open": session_open,
                "order_type": order_type,
                "limit_status": limit_status,
                "limit_offset_bps": limit_offset_bps,
                "take_profit_bps": tp_bps,
                "stop_loss_bps": sl_bps,
                "bracket_trigger": bracket_trigger,
            }
        )
        info["execution"] = execution_info
        return observation, reward, terminated, False, info

    def render(self):  # pragma: no cover
        ts = self._timestamp(min(self.ptr, len(self.prices) - 1))
        print(f"t={ts} pos={self.position:.2f} equity={self.equity:.5f}")

    def _timestamp(self, idx: int):
        if self.timestamps is None or idx >= len(self.timestamps):
            return idx
        return self.timestamps[idx]

    def _decode_action(self, action: int) -> Tuple[float, str, float | None, float | None, float | None]:
        if action < 0 or action >= len(self._action_map):
            raise ValueError(f"Action index {action} outside valid range")
        return self._action_map[action]

    def _infer_symbol(self) -> str | None:
        if "symbol" not in self.frame.columns:
            return None
        unique_symbols = self.frame["symbol"].unique().to_list()
        if not unique_symbols:
            return None
        if len(unique_symbols) > 1:
            raise ValueError("TradingEnv expects a single symbol per dataset")
        return unique_symbols[0]

    def _init_sessions(self) -> None:
        if not self.market_spec:
            return
        cfg = self.market_spec.session_config or {}
        if not cfg.get("enabled"):
            return
        rules = []
        for session in cfg.get("sessions", []):
            start = self._parse_session_time(session.get("start"))
            end = self._parse_session_time(session.get("end"))
            tz_name = session.get("timezone", "UTC")
            try:
                tzinfo = ZoneInfo(tz_name)
            except ZoneInfoNotFoundError:
                tzinfo = ZoneInfo("UTC")
            rules.append((start, end, tzinfo))
        if rules:
            self._session_rules = tuple(rules)
            self._sessions_enabled = True
            self._gaps_expected = bool(cfg.get("gaps_expected", False))

    def _refresh_cost_noise(self) -> None:
        self._cost_noise = {"spread": 1.0, "slippage": 1.0, "commission": 1.0}
        if not self.execution_settings:
            return
        rand = self.execution_settings.randomization
        if not rand.enabled:
            return
        self._cost_noise["spread"] = max(0.0, float(self._np_random.normal(1.0, rand.spread_noise_std)))
        self._cost_noise["slippage"] = max(0.0, float(self._np_random.normal(1.0, rand.slippage_noise_std)))
        self._cost_noise["commission"] = max(0.0, float(self._np_random.normal(1.0, rand.commission_noise_std)))

    def _apply_fill_ratio(self, delta: float) -> float:
        if abs(delta) < 1e-9 or not self.execution_settings:
            return delta
        ratio = max(0.0, min(1.0, self.execution_settings.fill_ratio))
        if ratio >= 1.0:
            return delta
        if not self.execution_settings.partial_fills:
            return delta if self._np_random.random() < ratio else 0.0
        return delta * ratio

    def _limit_order_filled(self, target_position: float, delta: float, offset_bps: float | None) -> bool:
        threshold_bps = offset_bps if offset_bps is not None else self.limit_order_fill_bps
        threshold = threshold_bps * 1e-4
        price_change = (float(self.prices[self.ptr]) - self.prev_price) / max(self.prev_price, 1e-6)
        if delta > 0:  # buying
            return price_change <= -threshold
        return price_change >= threshold

    def _compute_transaction_cost_bps(self, delta: float, price_return: float, order_type: str):
        if abs(delta) < 1e-9:
            return 0.0, {"spread_bps": 0.0, "slippage_bps": 0.0, "commission_bps": 0.0}
        if not self.cost_spec:
            base = self.reward_cfg.trading_cost_bps
            return base, {"spread_bps": 0.0, "slippage_bps": 0.0, "commission_bps": base}
        spread_bps = self.cost_spec.spread_bps * self._cost_noise["spread"]
        commission_rate = self.cost_spec.maker_bps if order_type == "limit" else self.cost_spec.taker_bps
        commission_bps = commission_rate * self._cost_noise["commission"]
        realized_vol_bps = abs(price_return) * 1e4
        slippage_spec = self.cost_spec.slippage
        slippage_bps = (
            slippage_spec.base_bps
            + slippage_spec.volatility_multiplier * realized_vol_bps
            + slippage_spec.size_impact * abs(delta)
        ) * self._cost_noise["slippage"]
        total = max(0.0, spread_bps + commission_bps + slippage_bps)
        return total, {
            "spread_bps": spread_bps,
            "slippage_bps": slippage_bps,
            "commission_bps": commission_bps,
        }

    def _compute_holding_cost_bps(self, position: float) -> float | None:
        if not self.cost_spec or abs(position) < 1e-12 or self._steps_per_year <= 0:
            return None
        rate = (
            self.cost_spec.financing_long_annual if position > 0 else self.cost_spec.financing_short_annual
        )
        if rate is None:
            return None
        return (rate / self._steps_per_year) * 1e4

    def _estimate_realized_vol(self, idx: int, window: int = 30) -> float:
        """Estimate realized volatility. Uses pre-computed if available for performance."""
        # Use pre-computed volatility if available (much faster!)
        if self.precomputed_vol is not None and idx < len(self.precomputed_vol):
            vol = self.precomputed_vol[idx]
            if not np.isnan(vol):
                return float(vol)

        # Fallback to runtime calculation
        if idx <= 0:
            return 0.0
        start = max(0, idx - window)
        window_prices = self.prices[start : idx + 1]
        if window_prices.size <= 1:
            return 0.0
        returns = np.diff(window_prices) / window_prices[:-1]
        if returns.size == 0:
            return 0.0
        return float(np.std(returns))

    def _is_session_open(self, idx: int) -> bool:
        if not self._sessions_enabled or self.timestamps is None or idx >= len(self.timestamps):
            return True
        ts = self._to_datetime(self.timestamps[idx])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        for start, end, tzinfo in self._session_rules:
            local_time = ts.astimezone(tzinfo).time()
            if start <= end:
                if start <= local_time < end:
                    return True
            else:
                if local_time >= start or local_time < end:
                    return True
        return False

    def _infer_step_seconds(self) -> float:
        if not self.timestamps or len(self.timestamps) < 2:
            return 60.0
        t0 = self._to_datetime(self.timestamps[0])
        t1 = self._to_datetime(self.timestamps[1])
        delta = (t1 - t0).total_seconds()
        return delta if delta > 0 else 60.0

    @staticmethod
    def _parse_session_time(spec: str | None) -> time:
        if not spec:
            return time(0, 0)
        hours, minutes = spec.split(":")
        return time(hour=int(hours), minute=int(minutes))

    @staticmethod
    def _to_datetime(value):
        if isinstance(value, datetime):
            return value
        if isinstance(value, np.datetime64):
            ts_ns = value.astype('datetime64[ns]').astype('int64')
            return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        raise TypeError(f"Unsupported timestamp type: {type(value)!r}")

    def _check_brackets(self, position: float, tp_bps: float | None, sl_bps: float | None, price_return: float) -> str | None:
        if position == 0.0:
            return None
        if tp_bps is not None:
            threshold = tp_bps * 1e-4
            if position > 0 and price_return >= threshold:
                return "take_profit"
            if position < 0 and price_return <= -threshold:
                return "take_profit"
        if sl_bps is not None:
            threshold = sl_bps * 1e-4
            if position > 0 and price_return <= -threshold:
                return "stop_loss"
            if position < 0 and price_return >= threshold:
                return "stop_loss"
        return None

    @staticmethod
    def _load_frame(data: str | pl.DataFrame) -> pl.DataFrame:
        if isinstance(data, pl.DataFrame):
            return data.sort("timestamp") if "timestamp" in data.columns else data
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data}")
        return pl.read_parquet(path)


__all__ = ["TradingEnv", "EnvConfig"]
