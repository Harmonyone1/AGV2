"""Data ingestion + feature pipeline entrypoint for AGV2."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import polars as pl
import yaml

from data import DuckDBStore, PostgresManager, TradeLockerClient, run_quality_checks
from features import (
    WindowConfig,
    build_encoder_windows,
    engineer_price_features,
    parkinson_volatility,
    realized_volatility,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download market data and build encoder windows")
    parser.add_argument("--symbols", default="ETH,XAUUSD", help="Comma-separated list of TradeLocker symbols")
    parser.add_argument("--timeframe", default="1m", help="Base timeframe to download")
    parser.add_argument("--start", help="ISO8601 start timestamp (UTC)")
    parser.add_argument("--end", help="ISO8601 end timestamp (UTC)")
    parser.add_argument("--env", default=".env", help="Path to .env file with credentials")
    parser.add_argument("--markets", default="config/markets.yaml", help="Market configuration YAML")
    parser.add_argument("--duckdb", default="data/processed/market.duckdb", help="DuckDB file path")
    parser.add_argument("--output", default="data/features/encoder_windows.parquet", help="Output parquet for encoder windows")
    parser.add_argument("--window-length", type=int, default=512, help="Encoder window length")
    parser.add_argument("--future-horizons", default="1,3,12,60", help="Comma-separated future return horizons")
    parser.add_argument("--skip-postgres", action="store_true", help="Skip persisting raw bars to Postgres")
    parser.add_argument("--skip-duckdb", action="store_true", help="Skip persisting raw bars to DuckDB")
    parser.add_argument("--quality-threshold", type=int, default=5, help="Maximum missing bars allowed per series")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start, end = _resolve_window(args.start, args.end)
    symbols = [sym.strip() for sym in args.symbols.split(",") if sym.strip()]
    future_horizons = [int(x) for x in args.future_horizons.split(",") if x.strip()]
    markets = _load_market_config(args.markets)

    client = TradeLockerClient.from_env(args.env)
    frames = []
    for symbol in symbols:
        frame = client.fetch_ohlcv(symbol, args.timeframe, start=start, end=end)
        if frame.empty:
            print(f"No data returned for {symbol}")
            continue
        frames.append(frame)
    if not frames:
        raise SystemExit("No data downloaded; aborting")
    raw = pd.concat(frames, ignore_index=True)

    if not args.skip_postgres:
        with PostgresManager.from_env(args.env) as pg:
            for (symbol, timeframe), group in raw.groupby(["symbol", "timeframe"]):
                pg.store_bars(symbol, timeframe, group.to_dict("records"))

    pl_frame = pl.from_pandas(raw)
    if not args.skip_duckdb:
        with DuckDBStore(args.duckdb) as store:
            store.append_bars(pl_frame)

    _run_quality(pl_frame, args.quality_threshold)

    window_frames = []
    for symbol, timeframe, group in _iter_symbol_frames(pl_frame):
        session_cfg = markets.get(symbol, {}).get("session_config")
        engineered = engineer_price_features(group, session_config=session_cfg)
        engineered = engineered.with_columns(
            [
                realized_volatility(window=32),
                realized_volatility(window=96),
                parkinson_volatility(window=64),
            ]
        )
        window_cfg = WindowConfig(
            window_length=args.window_length,
            future_return_horizons=future_horizons,
        )
        try:
            windows = build_encoder_windows(engineered, window_cfg)
        except ValueError as exc:
            print(f"Skipping {symbol}-{timeframe}: {exc}")
            continue
