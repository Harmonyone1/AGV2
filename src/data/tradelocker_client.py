"""Thin TradeLocker client wrapper tailored for AGV2 data ingestion."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

from infrastructure.config_loader import load_config
from infrastructure.tradelocker_api import TLAPI
from data.timeframes import timeframe_to_minutes


@dataclass
class TradeLockerCredentials:
    """Container for the credentials required to hit the TradeLocker API."""

    environment: str
    email: str
    password: str
    server: str
    account_number: int
    log_level: str = "info"


class TradeLockerClient:
    """High-level helper for fetching OHLCV data from TradeLocker."""

    def __init__(self, api: TLAPI, *, logger: Optional[logging.Logger] = None) -> None:
        self.api = api
        self.log = logger or logging.getLogger(self.__class__.__name__)
        self._instrument_cache: Dict[str, int] = {}

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "TradeLockerClient":
        """Instantiate the client by reading credentials from an ``.env`` file."""
        env = load_config(env_path)
        creds = TradeLockerCredentials(
            environment=env["TL_ENVIRONMENT"],
            email=env["TL_EMAIL"],
            password=env["TL_PASSWORD"],
            server=env["TL_SERVER"],
            account_number=int(env["TL_ACC_NUM"]),
            log_level=env.get("TL_LOG_LEVEL", "info").lower(),
        )
        api = TLAPI(
            environment=creds.environment,
            username=creds.email,
            password=creds.password,
            server=creds.server,
            acc_num=creds.account_number,
            log_level=creds.log_level,
        )
        return cls(api)

    def resolve_instrument_id(self, symbol: str) -> int:
        if symbol not in self._instrument_cache:
            instrument_id = int(self.api.get_instrument_id_from_symbol_name(symbol))
            self._instrument_cache[symbol] = instrument_id
        return self._instrument_cache[symbol]

    def _chunk_span(self, timeframe: str) -> timedelta:
        max_rows = getattr(self.api, "max_price_history_rows", lambda: 5000)()
        minutes_per_bar = timeframe_to_minutes(timeframe)
        minutes = max(1, int(max_rows * minutes_per_bar) - 1)
        return timedelta(minutes=minutes)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for ``symbol`` in ``[start, end]``.

        Automatically chunks requests to satisfy TradeLocker's max-row limit and
        returns data with normalized column names.
        """
        if start >= end:
            raise ValueError("start must be before end")
        start = self._ensure_utc(start)
        end = self._ensure_utc(end)
        instrument_id = self.resolve_instrument_id(symbol)
        frames: List[pd.DataFrame] = []
        window_start = start
        chunk = self._chunk_span(timeframe)
        while window_start < end:
            window_end = min(end, window_start + chunk)
            df = self.api.get_price_history(
                instrument_id=instrument_id,
                resolution=timeframe,
                start_timestamp=int(window_start.timestamp() * 1000),
                end_timestamp=int(window_end.timestamp() * 1000),
            )
            if not df.empty:
                frames.append(self._normalize_frame(df, symbol, timeframe))
            window_start = window_end
        if not frames:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe"]
            )
        return pd.concat(frames, ignore_index=True)

    def fetch_many(
        self,
        requests: Iterable[Mapping[str, Any]],
    ) -> pd.DataFrame:
        """Fetch data for multiple requests and concatenate the results."""
        frames: List[pd.DataFrame] = []
        for req in requests:
            start = req.get("start")
            end = req.get("end")
            if not isinstance(start, datetime) or not isinstance(end, datetime):
                raise TypeError("Requests must include datetime 'start' and 'end' keys")
            frames.append(
                self.fetch_ohlcv(
                    symbol=str(req["symbol"]),
                    timeframe=str(req["timeframe"]),
                    start=start,
                    end=end,
                )
            )
        if not frames:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe"]
            )
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _normalize_frame(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        renamed = df.rename(
            columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        renamed = renamed[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        renamed["timestamp"] = pd.to_datetime(renamed["timestamp"], unit="ms", utc=True)
        renamed["symbol"] = symbol
        renamed["timeframe"] = timeframe
        return renamed

    @staticmethod
    def _ensure_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


__all__ = ["TradeLockerClient", "TradeLockerCredentials"]
