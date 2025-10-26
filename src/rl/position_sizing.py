"""Position sizing utilities with volatility targeting and Kelly criterion.

Implements:
1. Volatility-targeted position sizing
2. Fractional Kelly criterion
3. Maximum position limits
4. Portfolio heat management
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing."""
    method: str = "volatility_target"  # "volatility_target", "kelly", "fixed"
    target_volatility: float = 0.02  # 2% daily volatility target
    kelly_fraction: float = 0.25  # Conservative Kelly: 0.1-0.3
    max_position: float = 1.0  # Maximum position size (normalized)
    min_position: float = 0.1  # Minimum position size (avoid tiny positions)
    max_portfolio_heat: float = 0.20  # Maximum portfolio risk (20%)


def volatility_targeted_size(
    base_size: float,
    current_vol: float,
    target_vol: float,
    max_size: float = 1.0,
) -> float:
    """Scale position size to achieve target volatility.

    Formula: adjusted_size = base_size * (target_vol / current_vol)

    Args:
        base_size: Base position size from policy
        current_vol: Current realized volatility
        target_vol: Target volatility level
        max_size: Maximum allowed position size

    Returns:
        Adjusted position size
    """
    if current_vol < 1e-6:
        return 0.0  # No position if vol is zero

    scaling_factor = target_vol / max(current_vol, 1e-6)
    adjusted_size = base_size * scaling_factor

    # Clip to max size
    return float(np.clip(adjusted_size, -max_size, max_size))


def kelly_criterion_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_fraction: float = 0.25,
    max_size: float = 1.0,
) -> float:
    """Calculate position size using fractional Kelly criterion.

    Kelly formula: f* = (p * b - q) / b
    where:
        p = win probability
        q = loss probability (1 - p)
        b = avg_win / avg_loss (payoff ratio)

    Args:
        win_rate: Historical win rate (0-1)
        avg_win: Average winning trade size
        avg_loss: Average losing trade size (positive)
        kelly_fraction: Fraction of Kelly to use (0.1-0.5 recommended)
        max_size: Maximum allowed position size

    Returns:
        Kelly-adjusted position size
    """
    if avg_loss < 1e-6 or win_rate < 0.01 or win_rate > 0.99:
        return 0.0  # Invalid parameters

    q = 1.0 - win_rate
    b = avg_win / max(avg_loss, 1e-6)  # Payoff ratio

    # Full Kelly
    kelly_f = (win_rate * b - q) / b

    # Fractional Kelly (conservative)
    fractional_kelly = kelly_f * kelly_fraction

    # Clip to reasonable range
    return float(np.clip(fractional_kelly, 0.0, max_size))


def apply_position_sizing(
    policy_action: float,
    current_vol: float,
    config: PositionSizingConfig,
    win_rate: float | None = None,
    avg_win: float | None = None,
    avg_loss: float | None = None,
) -> float:
    """Apply position sizing rules to policy action.

    Args:
        policy_action: Raw action from policy (-1 to 1)
        current_vol: Current realized volatility
        config: Position sizing configuration
        win_rate: Historical win rate (for Kelly)
        avg_win: Average win size (for Kelly)
        avg_loss: Average loss size (for Kelly)

    Returns:
        Adjusted position size
    """
    if config.method == "fixed":
        # Just clip to max
        return float(np.clip(policy_action, -config.max_position, config.max_position))

    elif config.method == "volatility_target":
        # Scale by volatility
        return volatility_targeted_size(
            policy_action,
            current_vol,
            config.target_volatility,
            config.max_position,
        )

    elif config.method == "kelly":
        # Use Kelly criterion
        if win_rate is None or avg_win is None or avg_loss is None:
            # Fall back to fixed if no stats available
            return float(np.clip(policy_action, -config.max_position, config.max_position))

        kelly_size = kelly_criterion_size(
            win_rate,
            avg_win,
            avg_loss,
            config.kelly_fraction,
            config.max_position,
        )

        # Use Kelly size but keep sign from policy
        return kelly_size * np.sign(policy_action)

    else:
        raise ValueError(f"Unknown position sizing method: {config.method}")


def check_portfolio_heat(
    current_positions: dict[str, float],
    proposed_position: float,
    symbol: str,
    volatilities: dict[str, float],
    max_portfolio_heat: float = 0.20,
) -> tuple[bool, float]:
    """Check if proposed position would exceed portfolio heat limit.

    Portfolio heat = sum of (position_size * volatility) across all positions

    Args:
        current_positions: Dict of symbol -> position size
        proposed_position: Proposed new position for symbol
        symbol: Symbol being traded
        volatilities: Dict of symbol -> current volatility
        max_portfolio_heat: Maximum allowed portfolio heat

    Returns:
        (allowed, current_heat) tuple
    """
    # Calculate current heat
    total_heat = 0.0
    for sym, pos in current_positions.items():
        if sym == symbol:
            continue  # Will be replaced
        vol = volatilities.get(sym, 0.0)
        total_heat += abs(pos) * vol

    # Add proposed position
    proposed_vol = volatilities.get(symbol, 0.0)
    total_heat += abs(proposed_position) * proposed_vol

    allowed = total_heat <= max_portfolio_heat
    return allowed, total_heat


class PositionSizer:
    """Stateful position sizer that tracks historical performance."""

    def __init__(self, config: PositionSizingConfig):
        self.config = config
        self.trade_history: list[float] = []
        self.win_count = 0
        self.loss_count = 0
        self.total_wins = 0.0
        self.total_losses = 0.0

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade for Kelly calculation."""
        self.trade_history.append(pnl)

        if pnl > 0:
            self.win_count += 1
            self.total_wins += pnl
        elif pnl < 0:
            self.loss_count += 1
            self.total_losses += abs(pnl)

    def get_statistics(self) -> tuple[float, float, float]:
        """Get win rate and average win/loss for Kelly."""
        total_trades = self.win_count + self.loss_count

        if total_trades == 0:
            return 0.5, 0.0, 0.0  # No data, use defaults

        win_rate = self.win_count / total_trades
        avg_win = self.total_wins / max(self.win_count, 1)
        avg_loss = self.total_losses / max(self.loss_count, 1)

        return win_rate, avg_win, avg_loss

    def size_position(
        self,
        policy_action: float,
        current_vol: float,
    ) -> float:
        """Size position using configured method and historical stats."""
        win_rate, avg_win, avg_loss = self.get_statistics()

        return apply_position_sizing(
            policy_action,
            current_vol,
            self.config,
            win_rate,
            avg_win,
            avg_loss,
        )


__all__ = [
    "PositionSizingConfig",
    "PositionSizer",
    "volatility_targeted_size",
    "kelly_criterion_size",
    "apply_position_sizing",
    "check_portfolio_heat",
]
