"""
data.py — read-only loader for the Daily_Cash_Market DuckDB archive.

Single source of truth for the cash-market EOD spine (OHLC + volume + delivery)
and the Nifty index series used by the regime gate. Read-only: this project
never writes to the DCM database.
"""
from __future__ import annotations

import functools

import duckdb
import pandas as pd

from . import config


def _connect() -> duckdb.DuckDBPyConnection:
    if not config.DCM_DUCKDB.exists():
        raise FileNotFoundError(
            f"DCM archive not found at {config.DCM_DUCKDB}. "
            "This system reads the Daily_Cash_Market DuckDB EOD spine."
        )
    try:
        return duckdb.connect(str(config.DCM_DUCKDB), read_only=True)
    except duckdb.IOException as e:
        # DuckDB allows many readers OR one writer — never both. So any process holding the
        # file read-write (the Daily_Cash_Market dashboard, its nightly sync) locks EVERY
        # other process out, even read-only ones. The raw error names a PID and nothing else,
        # which reads like corruption. Say what it actually is and how to clear it.
        if "being used by another process" in str(e) or "lock" in str(e).lower():
            raise RuntimeError(
                "The DCM archive is LOCKED by another process — most likely the "
                "Daily_Cash_Market dashboard (it opens the DuckDB read-write, and DuckDB "
                "permits many readers OR one writer, never both).\n"
                "  FIX: close the Daily_Cash_Market app (or its nightly sync), then ↻ refresh.\n"
                "  This board is READ-ONLY and never writes to that archive.\n"
                f"  ({e})"
            ) from e
        raise


@functools.lru_cache(maxsize=1)
def fno_universe() -> tuple[str, ...]:
    """The liquid single-stock F&O universe (~270 names) — cash equity, no theta."""
    with _connect() as c:
        ph = ",".join("?" * len(config.FNO_INSTRUMENTS))
        rows = c.execute(
            f"select distinct symbol from fno_bhavcopy where instrument in ({ph}) order by 1",
            list(config.FNO_INSTRUMENTS),
        ).fetchall()
    return tuple(r[0] for r in rows)


def load_eod(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Cash EOD (series EQ) for the F&O universe, optionally date-windowed.

    Returns tidy rows sorted by (symbol, trade_date) with valid ranges only.
    """
    syms = fno_universe()
    ph = ",".join("?" * len(syms))
    params = list(syms)
    where = [f"symbol in ({ph})", "series='EQ'", "close_price>0", "high_price>low_price"]
    if start:
        where.append("trade_date>=?"); params.append(start)
    if end:
        where.append("trade_date<=?"); params.append(end)
    sql = f"""
        select trade_date, symbol, prev_close, open_price, high_price, low_price,
               close_price, avg_price, ttl_trd_qnty, deliv_qty, deliv_per,
               turnover_lacs, no_of_trades
        from daily_data
        where {' and '.join(where)}
        order by symbol, trade_date
    """
    with _connect() as c:
        df = c.execute(sql, params).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def load_nifty(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Nifty 50 daily close for the regime gate."""
    where = ["index_name=?"]
    params: list = [config.NIFTY_INDEX_NAME]
    if start:
        where.append("trade_date>=?"); params.append(start)
    if end:
        where.append("trade_date<=?"); params.append(end)
    sql = f"select trade_date, close_val from index_data where {' and '.join(where)} order by trade_date"
    with _connect() as c:
        df = c.execute(sql, params).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


@functools.lru_cache(maxsize=1)
def load_sectors() -> dict[str, str]:
    """symbol -> canonical sector (for the concentration cap). Unmapped names
    (renames/aliases) fall back to their own symbol so they never share a bucket."""
    with _connect() as c:
        rows = c.execute("select symbol, sector from v_sector_master").fetchall()
    return {s: (sec or f"_{s}") for s, sec in rows}


def last_trading_date() -> pd.Timestamp:
    """Most recent date present in the EOD archive."""
    with _connect() as c:
        d = c.execute("select max(trade_date) from daily_data where series='EQ'").fetchone()[0]
    return pd.Timestamp(d)
