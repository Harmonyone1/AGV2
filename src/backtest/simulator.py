"""Vectorized backtesting helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class BacktestResult:
    total_return: float
    sharpe: float
    max_drawdown: float
    equity_curve: np.ndarray


class VectorizedBacktester:
    """Applies position signals to price returns for fast evaluation."""

    def __init__(self, returns: Sequence[float], trading_cost_bps: float = 1.0, annualization_factor: float = 252.0) -> None:
        """
        Args:
            returns: Price returns for each step
            trading_cost_bps: Trading cost in basis points
            annualization_factor: Number of periods per year for Sharpe calculation
                                 252 for daily stock/futures (default)
                                 365 for 24/7 crypto markets
                                 ~17520 for hourly data (24*365.25)
        """
        self.returns = np.asarray(returns, dtype=np.float32)
        self.cost = trading_cost_bps * 1e-4
        self.annualization_factor = float(annualization_factor)

    def run(self, signals: Sequence[float]) -> BacktestResult:
        positions = np.clip(np.asarray(signals, dtype=np.float32), -1.0, 1.0)
        if positions.shape[0] != self.returns.shape[0]:
            raise ValueError("Signals and returns must have the same length")
        pnl = positions * self.returns
        costs = np.abs(np.diff(positions, prepend=0.0)) * self.cost
        equity = np.cumsum(pnl - costs)
        total_return = float(equity[-1])
        sharpe = self._sharpe(pnl - costs, self.annualization_factor)
        max_dd = self._max_drawdown(equity)
        return BacktestResult(total_return, sharpe, max_dd, equity)

    @staticmethod
    def _sharpe(pnl: np.ndarray, annualization_factor: float = 252.0) -> float:
        std = np.std(pnl)
        if std == 0:
            return 0.0
        return float(np.mean(pnl) / (std + 1e-9) * np.sqrt(annualization_factor))

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks)
        return float(drawdowns.min())


__all__ = ["VectorizedBacktester", "BacktestResult"]
