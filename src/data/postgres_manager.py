"""Helpers for persisting market data into PostgreSQL."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import psycopg2
from psycopg2.extensions import connection

from infrastructure import database
from infrastructure.config_loader import load_config


@dataclass
class PostgresSettings:
    name: str
    user: str
    password: str
    host: str
    port: str

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "PostgresSettings":
        env = load_config(env_path)
        return cls(
            name=env["DB_NAME"],
            user=env["DB_USER"],
            password=env["DB_PASSWORD"],
            host=env["DB_HOST"],
            port=str(env["DB_PORT"]),
        )


class PostgresManager:
    """Small wrapper around ``infrastructure.database`` utilities."""

    def __init__(self, settings: PostgresSettings) -> None:
        self.settings = settings
        self._conn: Optional[connection] = None

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "PostgresManager":
        return cls(PostgresSettings.from_env(env_path))

    @property
    def conn(self) -> connection:
        if self._conn is None or self._conn.closed:
            conf = database.DBConf(
                DB_NAME=self.settings.name,
                DB_USER=self.settings.user,
                DB_PASSWORD=self.settings.password,
                DB_HOST=self.settings.host,
                DB_PORT=self.settings.port,
            )
            self._conn = database.connect_db(conf)
        return self._conn

    def ensure_schema(self) -> None:
        database.create_tables(self.conn)

    def recreate_schema(self) -> None:
        database.recreate_tables(self.conn)

    def store_bars(self, symbol: str, timeframe: str, rows: Iterable[Mapping[str, object]]) -> int:
        inserted = 0
        for row in rows:
            database.store_historical_data(self.conn, symbol, timeframe, dict(row))
            inserted += 1
        return inserted

    def fetch_recent(self, symbol: str, timeframe: str, limit: int = 100):
        return database.fetch_price_history(self.conn, symbol, timeframe, limit=limit)

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "PostgresManager":
        _ = self.conn
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["PostgresManager", "PostgresSettings"]
