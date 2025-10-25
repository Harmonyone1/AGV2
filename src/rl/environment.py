"""Gymnasium-compatible trading environment."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces

from rl.rewards import RewardConfig, compute_reward


@dataclass
class EnvConfig:
    data_path: str
    episode_length: int = 256
    trading_cost_bps: float = 1.0
    max_position: float = 1.0
    reward: RewardConfig = field(default_factory=RewardConfig)


class TradingEnv(gym.Env):
    """Single-symbol trading environment with discrete actions."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: str | pl.DataFrame,
        episode_length: int = 256,
        trading_cost_bps: float = 1.0,
        max_position: float = 1.0,
        reward_cfg: RewardConfig | None = None,
    ) -> None:
        super().__init__()
        self.frame = self._load_frame(data)
        if "features" not in self.frame.columns or "close" not in self.frame.columns:
            raise ValueError("Dataset must contain 'features' and 'close' columns")
        vector_column = "embedding" if "embedding" in self.frame.columns else "features"
        vectors = [np.asarray(vec, dtype=np.float32) for vec in self.frame[vector_column].to_list()]
        self.features = np.stack(vectors)
        self.feature_source = vector_column
        self.prices = self.frame["close"].to_numpy()
        self.timestamps = self.frame.get_column("timestamp").to_list() if "timestamp" in self.frame.columns else None
        self.reward_cfg = reward_cfg or RewardConfig(trading_cost_bps=trading_cost_bps)
        self.episode_length = episode_length
        self.trading_cost_bps = trading_cost_bps
        self.max_position = max_position

        self.action_space = spaces.Discrete(3)
        feature_dim = self.features.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32)

        self._np_random, _ = gym.utils.seeding.np_random()
        self._reset_state()

    def _reset_state(self) -> None:
        self.start_idx = 0
        self.ptr = 0
        self.position = 0.0
        self.prev_price = 0.0
        self.equity = 0.0
        self.step_count = 0
        self._terminated = False

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
        observation = self.features[self.ptr]
        info = {"position": self.position, "timestamp": self._timestamp(self.ptr)}
        return observation, info

    def step(self, action: int):  # type: ignore[override]
        if self._terminated:
            raise RuntimeError("Call reset() before stepping a terminated episode")
        target_position = self._action_to_position(action)
        price = float(self.prices[self.ptr])
        price_return = (price - self.prev_price) / max(self.prev_price, 1e-6)
        reward = compute_reward(
            position=target_position,
            prev_position=self.position,
            price_return=price_return,
            realized_vol=None,
            cfg=self.reward_cfg,
        )
        self.position = target_position
        self.equity += reward
        self.prev_price = price
        self.ptr += 1
        self.step_count += 1

        dataset_limit = len(self.prices) - 1
        episode_limit = self.start_idx + self.episode_length
        terminated = self.ptr >= min(dataset_limit, episode_limit)
        self._terminated = terminated
        obs_index = min(self.ptr, len(self.features) - 1)
        observation = self.features[obs_index]
        info = {
            "position": self.position,
            "timestamp": self._timestamp(obs_index),
            "price": price,
            "price_return": price_return,
            "equity": self.equity,
            "step": self.step_count,
            "feature_source": self.feature_source,
        }
        return observation, reward, terminated, False, info

    def render(self):  # pragma: no cover - debug helper
        ts = self._timestamp(min(self.ptr, len(self.prices) - 1))
        print(f"t={ts} pos={self.position:.2f} equity={self.equity:.5f}")

    def _timestamp(self, idx: int):
        if self.timestamps is None or idx >= len(self.timestamps):
            return idx
        return self.timestamps[idx]

    def _action_to_position(self, action: int) -> float:
        if action == 0:
            return -self.max_position
        if action == 1:
            return 0.0
        return self.max_position

    @staticmethod
    def _load_frame(data: str | pl.DataFrame) -> pl.DataFrame:
        if isinstance(data, pl.DataFrame):
            return data.sort("timestamp") if "timestamp" in data.columns else data
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data}")
        return pl.read_parquet(path)


__all__ = ["TradingEnv", "EnvConfig"]
