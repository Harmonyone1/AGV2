"""Market and cost configuration loaders for TradeLocker-like execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


@dataclass(frozen=True)
class SlippageSpec:
    model: str
    base_bps: float
    volatility_multiplier: float
    size_impact: float


@dataclass(frozen=True)
class CostSpec:
    symbol: str
    maker_bps: float
    taker_bps: float
    spread_bps: float
    slippage: SlippageSpec
    financing_long_annual: float | None = None
    financing_short_annual: float | None = None


@dataclass(frozen=True)
class MarketSpec:
    symbol: str
    tick_size: float
    lot_size: float
    min_notional: float
    session_config: Dict[str, Any] | None = None
    target_volatility: float | None = None


@dataclass(frozen=True)
class CostRandomization:
    enabled: bool = False
    slippage_noise_std: float = 0.0
    spread_noise_std: float = 0.0
    commission_noise_std: float = 0.0


@dataclass(frozen=True)
class BacktestSettings:
    fill_ratio: float = 1.0
    partial_fills: bool = False
    randomization: CostRandomization = field(default_factory=CostRandomization)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=8)
def _cached_markets(path: str) -> Dict[str, Any]:
    return _load_yaml(Path(path))


@lru_cache(maxsize=8)
def _cached_costs(path: str) -> Dict[str, Any]:
    return _load_yaml(Path(path))


def load_market_spec(symbol: str, path: str = "config/markets.yaml") -> MarketSpec:
    data = _cached_markets(str(Path(path).resolve()))
    markets = data.get("markets", {})
    if symbol not in markets:
        available = sorted(markets.keys())
        available_str = ", ".join(available) if available else "none"
        raise KeyError(
            f"Market spec not found for symbol '{symbol}' in {path}.\n"
            f"Available symbols: {available_str}"
        )
    entry = markets[symbol]
    return MarketSpec(
        symbol=symbol,
        tick_size=float(entry.get("tick_size", 0.0)),
        lot_size=float(entry.get("lot_size", 1.0)),
        min_notional=float(entry.get("min_notional", 0.0)),
        session_config=entry.get("session_config"),
        target_volatility=entry.get("target_volatility"),
    )


def load_cost_spec(
    symbol: str,
    path: str = "config/costs.yaml",
    profile: str = "backtesting",
) -> Tuple[CostSpec, BacktestSettings]:
    data = _cached_costs(str(Path(path).resolve()))
    symbol_costs = data.get("costs", {})
    if symbol not in symbol_costs:
        available = sorted(symbol_costs.keys())
        available_str = ", ".join(available) if available else "none"
        raise KeyError(
            f"Cost spec not found for symbol '{symbol}' in {path}.\n"
            f"Available symbols: {available_str}"
        )
    entry = symbol_costs[symbol]
    slippage_entry = entry.get("slippage", {})
    slippage = SlippageSpec(
        model=slippage_entry.get("model", "proportional"),
        base_bps=float(slippage_entry.get("base_bps", 0.0)),
        volatility_multiplier=float(slippage_entry.get("volatility_multiplier", 0.0)),
        size_impact=float(slippage_entry.get("size_impact", 0.0)),
    )
    financing = entry.get("financing", {})
    cost_spec = CostSpec(
        symbol=symbol,
        maker_bps=float(entry.get("maker_bps", 0.0)),
        taker_bps=float(entry.get("taker_bps", 0.0)),
        spread_bps=float(entry.get("spread_bps", 0.0)),
        slippage=slippage,
        financing_long_annual=financing.get("long_rate_annual"),
        financing_short_annual=financing.get("short_rate_annual"),
    )

    profile_data = data.get(profile, {})
    random_data = profile_data.get("randomization", {})
    random_cfg = CostRandomization(
        enabled=bool(random_data.get("enabled", False)),
        slippage_noise_std=float(random_data.get("slippage_noise_std", 0.0)),
        spread_noise_std=float(random_data.get("spread_noise_std", 0.0)),
        commission_noise_std=float(random_data.get("commission_noise_std", 0.0)),
    )
    settings = BacktestSettings(
        fill_ratio=float(profile_data.get("fill_ratio", 1.0)),
        partial_fills=bool(profile_data.get("partial_fills", False)),
        randomization=random_cfg,
    )
    return cost_spec, settings


__all__ = [
    "SlippageSpec",
    "CostSpec",
    "MarketSpec",
    "CostRandomization",
    "BacktestSettings",
    "load_market_spec",
    "load_cost_spec",
]
