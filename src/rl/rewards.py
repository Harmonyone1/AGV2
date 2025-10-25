"""Reward shaping utilities for the trading environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RewardConfig:
    """Weights controlling reward shaping."""

    trading_cost_bps: float = 1.0
    holding_cost_bps: float = 0.2
    risk_aversion: float = 0.0
    reward_clip: Tuple[float, float] | None = (-0.05, 0.05)


def compute_reward(
    position: float,
    prev_position: float,
    price_return: float,
    realized_vol: float | None,
    cfg: RewardConfig,
) -> float:
    """Return reward for the current step given position dynamics."""

    trade_cost = abs(position - prev_position) * cfg.trading_cost_bps * 1e-4
    holding_cost = abs(position) * cfg.holding_cost_bps * 1e-4
    risk_penalty = cfg.risk_aversion * (realized_vol if realized_vol is not None else abs(price_return))
    pnl = position * price_return
    reward = pnl - trade_cost - holding_cost - risk_penalty
    if cfg.reward_clip is not None:
        reward = float(np.clip(reward, cfg.reward_clip[0], cfg.reward_clip[1]))
    return reward


__all__ = ["RewardConfig", "compute_reward"]
