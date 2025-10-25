"""DuckDB-backed store for processed bars and feature tensors."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional

import polars as pl

try:  # Optional dependency for analytics storage
    import duckdb  # type: ignore
except ImportError:  # pragma: no cover
    duckdb = None


class DuckDBStore:
    """Lightweight analytics store built on DuckDB."""

    def __init__(self, path: str | Path = "data/processed/market.duckdb") -> None:
        if duckdb is None:  # pragma: no cover - handled in runtime environments lacking duckdb
            raise ImportError(
                "duckdb is not installed. Install it to enable DuckDB persistence, e.g. 'pip install duckdb'."
            )
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bars(
                symbol TEXT,
                timeframe TEXT,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY(symbol, timeframe, timestamp)
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS features(
                symbol TEXT,
                timeframe TEXT,
                timestamp TIMESTAMP,
                window_length INTEGER,
                feature_dim INTEGER,
                features DOUBLE[],
                regime_label INTEGER,
                sr_heatmap DOUBLE[],
                future_returns DOUBLE[],
                pattern_label INTEGER,
                PRIMARY KEY(symbol, timeframe, timestamp, window_length, feature_dim)
            );
            """
        )

    def append_bars(self, frame: pl.DataFrame | Iterable[Mapping[str, object]]) -> int:
        df = self._ensure_polars(frame)
        if df.is_empty():
            return 0
        df = df.select(
            pl.col(["symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume"])
        ).with_columns(pl.col("timestamp").cast(pl.Datetime(time_unit="us", time_zone="UTC")))
        self.conn.register("bars_input", df)
        self.conn.execute("INSERT OR REPLACE INTO bars SELECT * FROM bars_input")
        self.conn.unregister("bars_input")
        return df.height

    def load_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pl.DataFrame:
        conditions = ["symbol = ?", "timeframe = ?"]
        params: list[object] = [symbol, timeframe]
        if start:
            conditions.append("timestamp >= ?")
            params.append(self._bound_value(start))
        if end:
            conditions.append("timestamp <= ?")
            params.append(self._bound_value(end))
        query = "SELECT * FROM bars WHERE " + " AND ".join(conditions) + " ORDER BY timestamp"
        result = self.conn.execute(query, params).pl()
        return result

    def append_features(self, frame: pl.DataFrame | Iterable[Mapping[str, object]]) -> int:
        df = self._ensure_polars(frame)
        if df.is_empty():
            return 0
        required = [
            "symbol",
            "timeframe",
            "timestamp",
            "window_length",
            "feature_dim",
            "features",
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Feature frame missing columns: {missing}")
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime(time_unit="us", time_zone="UTC")))
        self.conn.register("features_input", df)
        self.conn.execute("INSERT OR REPLACE INTO features SELECT * FROM features_input")
        self.conn.unregister("features_input")
        return df.height

    def export_features_parquet(self, output_path: str | Path) -> Path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.conn.execute(
            "COPY (SELECT * FROM features ORDER BY timestamp) TO ? (FORMAT PARQUET, COMPRESSION 'zstd')",
            [str(dest)],
        )
        return dest

    @staticmethod
    def _bound_value(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.isoformat()

    @staticmethod
    def _ensure_polars(frame: pl.DataFrame | Iterable[Mapping[str, object]]) -> pl.DataFrame:
        if isinstance(frame, pl.DataFrame):
            return frame
        return pl.from_dicts(frame)

    def close(self) -> None:
        if duckdb is not None:
            self.conn.close()

    def __enter__(self) -> "DuckDBStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["DuckDBStore"]
