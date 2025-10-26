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
    normalize_pnl: bool = True  # Use log(1 + pnl/equity) normalization
    initial_equity: float = 10000.0  # Initial equity for normalization


def compute_reward(
    position: float,
    prev_position: float,
    price_return: float,
    realized_vol: float | None,
    cfg: RewardConfig,
    trade_cost_bps: float | None = None,
    holding_cost_bps: float | None = None,
    equity: float | None = None,
) -> float:
    """Return reward for the current step given position dynamics.

    Args:
        position: Current position size
        prev_position: Previous position size
        price_return: Price return for this step
        realized_vol: Realized volatility estimate
        cfg: Reward configuration
        trade_cost_bps: Override trading cost (basis points)
        holding_cost_bps: Override holding cost (basis points)
        equity: Current equity for normalization (uses cfg.initial_equity if None)

    Returns:
        Reward value (normalized if cfg.normalize_pnl is True)
    """

    trade_bps = cfg.trading_cost_bps if trade_cost_bps is None else trade_cost_bps
    hold_bps = cfg.holding_cost_bps if holding_cost_bps is None else holding_cost_bps
    trade_cost = abs(position - prev_position) * trade_bps * 1e-4
    holding_cost = abs(position) * hold_bps * 1e-4
    risk_penalty = cfg.risk_aversion * (realized_vol if realized_vol is not None else abs(price_return))
    pnl = position * price_return

    # Apply log normalization if enabled (as per ARCHITECTURE.md)
    if cfg.normalize_pnl:
        current_equity = equity if equity is not None else cfg.initial_equity
        # Use log(1 + pnl/equity) for better scaling across different asset classes
        normalized_pnl = float(np.log1p(pnl / max(current_equity, 1e-6)))
        reward = normalized_pnl - trade_cost - holding_cost - risk_penalty
    else:
        # Raw PnL (legacy mode)
        reward = pnl - trade_cost - holding_cost - risk_penalty

    if cfg.reward_clip is not None:
        reward = float(np.clip(reward, cfg.reward_clip[0], cfg.reward_clip[1]))
    return reward


__all__ = ["RewardConfig", "compute_reward"]
