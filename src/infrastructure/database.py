# Agnostic/database.py

import psycopg2
import psycopg2.extras
from dataclasses import dataclass
from typing import Dict, Any, List
import os
from dotenv import load_dotenv
import argparse


@dataclass
class DBConf:
    """Simple container for database connection settings."""
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: str

def connect_db(config: DBConf):
    """Establish a connection to the PostgreSQL database."""
    return psycopg2.connect(
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        host=config.DB_HOST,
        port=config.DB_PORT,
    )


def drop_tables(conn):
    """Drop all core tables if they exist."""
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS orders")
        cur.execute("DROP TABLE IF EXISTS trade_log")
        cur.execute("DROP TABLE IF EXISTS price_history")
    conn.commit()


def recreate_tables(conn):
    """Recreate core tables by dropping existing ones then calling ``create_tables``."""
    drop_tables(conn)
    create_tables(conn)

def create_tables(conn):
    """
    Creates the core tables and indexes if they do not exist:
      1) price_history + deduplication + unique index on (symbol, timeframe, bar_time)
      2) trade_log
      3) orders
    """
    with conn.cursor() as cur:
        # 1) price_history table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id         SERIAL PRIMARY KEY,
            symbol     TEXT    NOT NULL,
            timeframe  TEXT    NOT NULL,
            bar_time   TIMESTAMPTZ NOT NULL,
            open       DOUBLE PRECISION,
            high       DOUBLE PRECISION,
            low        DOUBLE PRECISION,
            close      DOUBLE PRECISION,
            volume     DOUBLE PRECISION
        );
        """)
        conn.commit()

        # 2) Remove any existing duplicates before unique index
        cur.execute("""
        DELETE FROM price_history a
        USING price_history b
        WHERE a.id < b.id
          AND a.symbol = b.symbol
          AND a.timeframe = b.timeframe
          AND a.bar_time = b.bar_time;
        """)
        conn.commit()

        # 3) Unique index for upserts
        try:
            cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
              price_history_symbol_timeframe_bartime_idx
            ON price_history(symbol, timeframe, bar_time);
            """)
            conn.commit()
        except psycopg2.errors.UniqueViolation:
            conn.rollback()
            # If duplicates snuck in, we've already cleaned above
            pass

        # 4) trade_log table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trade_log (
            id            SERIAL PRIMARY KEY,
            timestamp     TIMESTAMPTZ DEFAULT now(),
            instrument    TEXT    NOT NULL,
            side          TEXT    NOT NULL,
            size          NUMERIC NOT NULL,
            entry_price   NUMERIC NOT NULL,
            exit_price    NUMERIC,
            pnl           NUMERIC,
            confidence    NUMERIC,
            atr           NUMERIC
        );
        """)
        conn.commit()

        # 5) orders table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id        BIGINT PRIMARY KEY,
            account_id      BIGINT    NOT NULL,
            instrument_id   INTEGER   NOT NULL,
            quantity        NUMERIC   NOT NULL,
            side            TEXT      NOT NULL,
            price           NUMERIC,
            type            TEXT      NOT NULL,
            validity        TEXT      NOT NULL,
            status          TEXT      NOT NULL,
            filled_qty      NUMERIC       DEFAULT 0,
            stop_loss       NUMERIC,
            stop_loss_type  TEXT,
            take_profit     NUMERIC,
            take_profit_type TEXT,
            created_at      TIMESTAMPTZ   DEFAULT now(),
            updated_at      TIMESTAMPTZ   DEFAULT now()
        );
        """)
        conn.commit()

def store_historical_data(conn, symbol: str, timeframe: str, row: Dict[str, Any]):
    """
    Inserts one bar into price_history.
    ON CONFLICT on the unique index (symbol, timeframe, bar_time) DO NOTHING.
    Rolls back on error so the connection remains usable.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO price_history(
                    symbol, timeframe, bar_time, open, high, low, close, volume
                ) VALUES (
                    %s, %s, to_timestamp(%s/1000.0), %s, %s, %s, %s, %s
                )
                ON CONFLICT (symbol, timeframe, bar_time)
                DO NOTHING;
            """, (
                symbol, timeframe,
                row.get("t", 0),
                row.get("o"), row.get("h"),
                row.get("l"), row.get("c"),
                row.get("v")
            ))
        conn.commit()
    except Exception:
        conn.rollback()
        raise

def log_trade(
    conn,
    instrument: str,
    side: str,
    size: float,
    entry_price: float,
    exit_price: float = None,
    pnl: float = None,
    confidence: float = None,
    atr: float = None
):
    """
    Inserts a new entry into trade_log at order entry time.
    """
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO trade_log(
                instrument, side, size, entry_price, exit_price, pnl, confidence, atr
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            instrument, side, size,
            entry_price, exit_price, pnl,
            confidence, atr
        ))
    conn.commit()

def fetch_price_history(conn, symbol: str, timeframe: str, limit: int = 100):
    """
    Returns the most recent 'limit' bars from price_history.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("""
            SELECT * FROM price_history
             WHERE symbol = %s AND timeframe = %s
             ORDER BY bar_time DESC
             LIMIT %s;
        """, (symbol, timeframe, limit))
        return cur.fetchall()

# ─── Order persistence helpers ───────────────────────────────────

def save_order(conn, rec: Dict[str, Any]) -> None:
    """
    Persist a new or updated order into the orders table.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO orders(
                order_id, account_id, instrument_id, quantity, side,
                price, type, validity, status, filled_qty,
                stop_loss, stop_loss_type, take_profit, take_profit_type
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (order_id) DO UPDATE
              SET status          = EXCLUDED.status,
                  filled_qty      = EXCLUDED.filled_qty,
                  price           = EXCLUDED.price,
                  type            = EXCLUDED.type,
                  validity        = EXCLUDED.validity,
                  stop_loss       = EXCLUDED.stop_loss,
                  stop_loss_type  = EXCLUDED.stop_loss_type,
                  take_profit     = EXCLUDED.take_profit,
                  take_profit_type= EXCLUDED.take_profit_type,
                  updated_at      = now();
            """,
            (
                rec["order_id"],
                rec["account_id"],
                rec["instrument_id"],
                rec["quantity"],
                rec["side"],
                rec.get("price"),
                rec["type"],
                rec["validity"],
                rec["status"],
                rec.get("filled_qty", 0),
                rec.get("stop_loss"),
                rec.get("stop_loss_type"),
                rec.get("take_profit"),
                rec.get("take_profit_type"),
            ),
        )
    conn.commit()

def update_order(conn, rec: Dict[str, Any]) -> None:
    """
    Update status or filled_qty of an existing order.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE orders
               SET status           = %s,
                   filled_qty       = %s,
                   price            = COALESCE(%s, price),
                   type             = COALESCE(%s, type),
                   validity         = COALESCE(%s, validity),
                   stop_loss        = COALESCE(%s, stop_loss),
                   stop_loss_type   = COALESCE(%s, stop_loss_type),
                   take_profit      = COALESCE(%s, take_profit),
                   take_profit_type = COALESCE(%s, take_profit_type),
                   updated_at       = now()
             WHERE order_id        = %s;
            """,
            (
                rec.get("status"),
                rec.get("filled_qty", 0),
                rec.get("price"),
                rec.get("type"),
                rec.get("validity"),
                rec.get("stop_loss"),
                rec.get("stop_loss_type"),
                rec.get("take_profit"),
                rec.get("take_profit_type"),
                rec["order_id"],
            ),
        )
    conn.commit()

def load_open_orders(conn, account_id: int) -> List[Dict[str, Any]]:
    """
    Load orders that are still OPEN (not FILLED or CANCELED), for bootstrapping.
    Rolls back any aborted transaction and returns empty list if necessary.
    """
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT order_id, instrument_id, quantity, side,
                       price, type, validity, status, filled_qty,
                       stop_loss, stop_loss_type, take_profit, take_profit_type
                  FROM orders
                 WHERE status = 'OPEN' AND account_id = %s;
                """,
                (account_id,)
            )
            rows = cur.fetchall()
    except Exception:
        conn.rollback()
        return []
    return [dict(row) for row in rows]


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Create or reset the core database tables"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop existing tables before creating them",
    )
    args = parser.parse_args()

    db_conf = DBConf(
        DB_NAME=os.getenv("DB_NAME"),
        DB_USER=os.getenv("DB_USER"),
        DB_PASSWORD=os.getenv("DB_PASSWORD"),
        DB_HOST=os.getenv("DB_HOST"),
        DB_PORT=os.getenv("DB_PORT"),
    )

    conn = connect_db(db_conf)
    if args.recreate:
        recreate_tables(conn)
    else:
        create_tables(conn)
    conn.close()
