"""Dynamic risk controls and circuit breakers.

Implements:
1. Structure-aware stops (beyond S/R levels)
2. Dynamic trailing stops (volatility bands)
3. Trade throttles (cooldown, min hold time)
4. Daily loss limits and kill-switch
5. Max concurrent positions
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskControlsConfig:
    """Configuration for risk control parameters."""

    # Stop loss settings
    use_structure_aware_stops: bool = True
    stop_loss_atr_multiple: float = 2.0  # Stop at 2x ATR from entry
    min_stop_distance_bps: float = 50.0  # Minimum 50 bps stop

    # Trailing stop settings
    use_trailing_stops: bool = True
    trailing_stop_atr_multiple: float = 1.5
    trail_trigger_bps: float = 100.0  # Start trailing after 100 bps profit

    # Trade throttles
    min_hold_time_bars: int = 5  # Minimum bars to hold position
    trade_cooldown_bars: int = 3  # Bars to wait after exit
    max_trades_per_day: int = 10  # Maximum trades per day

    # Daily limits
    daily_loss_limit_pct: float = 0.05  # 5% daily loss triggers kill-switch
    daily_profit_target_pct: float = 0.10  # 10% daily profit (optional stop)
    enable_kill_switch: bool = True

    # Position limits
    max_concurrent_positions: int = 1  # For single-asset: 1, for multi-asset: higher
    max_concentration_pct: float = 1.0  # 100% max concentration in single asset


@dataclass
class TradeState:
    """State for an active trade."""
    entry_price: float
    entry_bar: int
    entry_time: datetime
    position_size: float
    stop_loss: float
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    highest_price: float = field(default=0.0)
    lowest_price: float = field(default=float("inf"))


class RiskController:
    """Manages risk controls and circuit breakers."""

    def __init__(self, config: RiskControlsConfig):
        self.config = config
        self.active_trades: Dict[str, TradeState] = {}
        self.trade_history: list[Dict] = []
        self.last_exit_bar: int = 0
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.current_date: Optional[datetime] = None
        self.kill_switch_active: bool = False

    def reset_daily_counters(self, current_time: datetime) -> None:
        """Reset daily counters at start of new day."""
        if self.current_date is None or current_time.date() != self.current_date.date():
            logger.info(f"New trading day: {current_time.date()}, resetting counters")
            self.current_date = current_time
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.kill_switch_active = False

    def can_open_position(self, current_bar: int, current_time: datetime) -> tuple[bool, str]:
        """Check if new position is allowed."""

        # Reset daily counters if new day
        self.reset_daily_counters(current_time)

        # Check kill switch
        if self.kill_switch_active:
            return False, "kill_switch_active"

        # Check daily loss limit
        if self.config.enable_kill_switch and self.daily_pnl <= -self.config.daily_loss_limit_pct:
            self.kill_switch_active = True
            logger.warning(
                f"KILL SWITCH ACTIVATED: Daily loss {self.daily_pnl:.2%} exceeds limit {self.config.daily_loss_limit_pct:.2%}"
            )
            return False, "daily_loss_limit_exceeded"

        # Check daily profit target (optional stop)
        if self.daily_pnl >= self.config.daily_profit_target_pct:
            logger.info(f"Daily profit target reached: {self.daily_pnl:.2%}")
            return False, "daily_profit_target_reached"

        # Check trade cooldown
        if current_bar < self.last_exit_bar + self.config.trade_cooldown_bars:
            return False, "trade_cooldown"

        # Check daily trade limit
        if self.daily_trades >= self.config.max_trades_per_day:
            return False, "max_daily_trades_exceeded"

        # Check max concurrent positions
        if len(self.active_trades) >= self.config.max_concurrent_positions:
            return False, "max_concurrent_positions"

        return True, "allowed"

    def open_position(
        self,
        symbol: str,
        entry_price: float,
        position_size: float,
        current_bar: int,
        current_time: datetime,
        atr: float,
        sr_levels: Optional[list[float]] = None,
    ) -> TradeState:
        """Register new position and calculate initial stops."""

        # Calculate stop loss
        if self.config.use_structure_aware_stops and sr_levels:
            # Find nearest S/R level below (for long) or above (for short)
            stop_loss = self._structure_aware_stop(entry_price, position_size, sr_levels, atr)
        else:
            # ATR-based stop
            stop_distance = atr * self.config.stop_loss_atr_multiple
            if position_size > 0:
                stop_loss = entry_price - stop_distance
            else:
                stop_loss = entry_price + stop_distance

        # Ensure minimum stop distance
        min_stop = entry_price * self.config.min_stop_distance_bps * 1e-4
        if position_size > 0:
            stop_loss = min(stop_loss, entry_price - min_stop)
        else:
            stop_loss = max(stop_loss, entry_price + min_stop)

        trade = TradeState(
            entry_price=entry_price,
            entry_bar=current_bar,
            entry_time=current_time,
            position_size=position_size,
            stop_loss=stop_loss,
            highest_price=entry_price,
            lowest_price=entry_price,
        )

        self.active_trades[symbol] = trade
        self.daily_trades += 1

        logger.info(
            "position_opened",
            extra={
                "symbol": symbol,
                "entry_price": entry_price,
                "position_size": position_size,
                "stop_loss": stop_loss,
                "current_bar": current_bar,
            },
        )

        return trade

    def update_position(
        self,
        symbol: str,
        current_price: float,
        current_bar: int,
        atr: float,
    ) -> Optional[str]:
        """Update trailing stops and check for stop-out.

        Returns:
            None if position should remain open
            "stop_loss", "trailing_stop", or "min_hold" if should close
        """
        if symbol not in self.active_trades:
            return None

        trade = self.active_trades[symbol]

        # Update price extremes
        trade.highest_price = max(trade.highest_price, current_price)
        trade.lowest_price = min(trade.lowest_price, current_price)

        # Check minimum hold time
        if current_bar < trade.entry_bar + self.config.min_hold_time_bars:
            return None

        # Check stop loss
        if trade.position_size > 0:  # Long position
            if current_price <= trade.stop_loss:
                logger.info(f"Stop loss hit for {symbol}: {current_price:.4f} <= {trade.stop_loss:.4f}")
                return "stop_loss"
        else:  # Short position
            if current_price >= trade.stop_loss:
                logger.info(f"Stop loss hit for {symbol}: {current_price:.4f} >= {trade.stop_loss:.4f}")
                return "stop_loss"

        # Update trailing stop
        if self.config.use_trailing_stops:
            profit_bps = abs(current_price - trade.entry_price) / trade.entry_price * 1e4

            if profit_bps >= self.config.trail_trigger_bps:
                trail_distance = atr * self.config.trailing_stop_atr_multiple

                if trade.position_size > 0:  # Long position
                    new_trail = trade.highest_price - trail_distance
                    if trade.trailing_stop is None or new_trail > trade.trailing_stop:
                        trade.trailing_stop = new_trail
                        logger.debug(f"Updated trailing stop for {symbol}: {new_trail:.4f}")

                    if current_price <= trade.trailing_stop:
                        logger.info(f"Trailing stop hit for {symbol}: {current_price:.4f} <= {trade.trailing_stop:.4f}")
                        return "trailing_stop"

                else:  # Short position
                    new_trail = trade.lowest_price + trail_distance
                    if trade.trailing_stop is None or new_trail < trade.trailing_stop:
                        trade.trailing_stop = new_trail

                    if current_price >= trade.trailing_stop:
                        logger.info(f"Trailing stop hit for {symbol}: {current_price:.4f} >= {trade.trailing_stop:.4f}")
                        return "trailing_stop"

        return None

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_bar: int,
        exit_reason: str,
    ) -> float:
        """Close position and record PnL."""
        if symbol not in self.active_trades:
            return 0.0

        trade = self.active_trades[symbol]
        pnl = (exit_price - trade.entry_price) / trade.entry_price * trade.position_size

        self.daily_pnl += pnl
        self.last_exit_bar = exit_bar

        # Record trade
        self.trade_history.append({
            "symbol": symbol,
            "entry_price": trade.entry_price,
            "exit_price": exit_price,
            "position_size": trade.position_size,
            "pnl": pnl,
            "entry_bar": trade.entry_bar,
            "exit_bar": exit_bar,
            "duration_bars": exit_bar - trade.entry_bar,
            "exit_reason": exit_reason,
        })

        logger.info(
            "position_closed",
            extra={
                "symbol": symbol,
                "exit_price": exit_price,
                "pnl": pnl,
                "exit_reason": exit_reason,
                "daily_pnl": self.daily_pnl,
            },
        )

        del self.active_trades[symbol]
        return pnl

    def _structure_aware_stop(
        self,
        entry_price: float,
        position_size: float,
        sr_levels: list[float],
        atr: float,
    ) -> float:
        """Calculate stop beyond nearest S/R level."""
        # Find nearest S/R level
        if position_size > 0:  # Long - find support below
            levels_below = [level for level in sr_levels if level < entry_price]
            if levels_below:
                nearest_support = max(levels_below)
                # Place stop below support + buffer
                buffer = atr * 0.5
                return nearest_support - buffer

        else:  # Short - find resistance above
            levels_above = [level for level in sr_levels if level > entry_price]
            if levels_above:
                nearest_resistance = min(levels_above)
                # Place stop above resistance + buffer
                buffer = atr * 0.5
                return nearest_resistance + buffer

        # Fallback to ATR-based
        stop_distance = atr * self.config.stop_loss_atr_multiple
        if position_size > 0:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def get_statistics(self) -> Dict:
        """Get risk control statistics."""
        return {
            "active_positions": len(self.active_trades),
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "kill_switch_active": self.kill_switch_active,
            "total_trades": len(self.trade_history),
            "last_exit_bar": self.last_exit_bar,
        }


__all__ = ["RiskControlsConfig", "RiskController", "TradeState"]
