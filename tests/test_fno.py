"""F&O positioning columns — the guards that matter, not the labels themselves.

The labels are DCM's; this module only reads them. What can break here is the plumbing:
the archive lock, the as-of discipline, and graceful degradation.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from eqbtst import config, fno


def test_dcm_is_imported_read_only():
    """THE one that must never regress.

    DCM's ConnectionManager opens DuckDB READ-WRITE unless CLOUD_MODE is set, and DuckDB is
    many-readers-OR-one-writer. Importing it without the flag takes the EXCLUSIVE lock on the
    shared archive -- locking out DCM's own dashboard, its nightly sync, and this board's own
    readers. This board must only ever READ.
    """
    fno._dcm()
    assert os.environ.get("CLOUD_MODE", "").lower() == "true"
    if not config.DCM_DUCKDB.exists():
        pytest.skip("archive not reachable")
    import sys
    sys.path.insert(0, str(config.DCM_DUCKDB.parent.parent))
    try:
        from src.data.connection import _read_only
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"DCM not importable: {e}")
    assert _read_only() is True, "DCM would take a WRITE lock on the shared archive"


def test_never_reads_past_the_as_of_date():
    """The board can be pointed at a past close, or Replay. Serving the NEWEST F&O row there
    would put tomorrow's positioning beside today's price."""
    if not config.DCM_DUCKDB.exists():
        pytest.skip("archive not reachable")
    from eqbtst import data
    try:
        last = data.last_trading_date()
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"archive unavailable: {e}")
    back = pd.Timestamp(last) - pd.Timedelta(days=90)
    _, meta = fno.positioning(back)
    if meta["date_used"] is None:
        pytest.skip("no F&O data that far back")
    assert meta["date_used"] <= back.date()


def test_degrades_to_dashes_rather_than_raising():
    """Every failure path -- no archive, pre-retention date, junk input -- must return the
    columns filled with an em dash. A board that raises is worse than one that says nothing."""
    for bad in ["1990-01-01", None, "not-a-date"]:
        frame, meta = fno.positioning(bad)
        assert list(frame.columns) == fno.COLS
        assert meta["date_used"] is None or meta["n"] >= 0
    df = pd.DataFrame({"symbol": ["AAA", "BBB"]})
    out = fno.positioning("1990-01-01")[0]
    assert out.empty
    ann = fno.annotate(df, "1990-01-01")
    assert list(ann[fno.COLS].iloc[0]) == ["—"] * 4


def test_names_without_fno_get_a_dash_not_a_guess():
    """~60 of this board's names have no F&O -- NSE retired the contracts at an expiry.
    A dash is the correct answer; anything else invents positioning that does not exist."""
    if not config.DCM_DUCKDB.exists():
        pytest.skip("archive not reachable")
    from eqbtst import data
    try:
        last = data.last_trading_date()
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"archive unavailable: {e}")
    df = pd.DataFrame({"symbol": ["KOTAKBANK", "ZZZNOTREAL"]})
    out = fno.annotate(df, last)
    assert list(out[out.symbol == "ZZZNOTREAL"][fno.COLS].iloc[0]) == ["—"] * 4
    assert set(out.columns) >= set(fno.COLS)


def test_annotate_preserves_row_count_and_order():
    if not config.DCM_DUCKDB.exists():
        pytest.skip("archive not reachable")
    from eqbtst import data
    try:
        last = data.last_trading_date()
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"archive unavailable: {e}")
    df = pd.DataFrame({"symbol": ["WIPRO", "KOTAKBANK", "ZZZ"], "x": [1, 2, 3]})
    out = fno.annotate(df, last)
    assert len(out) == 3 and list(out["symbol"]) == ["WIPRO", "KOTAKBANK", "ZZZ"]
    assert list(out["x"]) == [1, 2, 3]
