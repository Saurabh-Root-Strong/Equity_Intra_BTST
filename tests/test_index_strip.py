"""The headline index strip in the Intraday board header.

The symbol strings are the whole risk here. Fyers does NOT 404 an unknown index name — it
returns a row with lp=None — so a wrong spelling ships as a blank tile that is
indistinguishable from "market closed". These tests pin the names and pin the behaviour
when a quote does not come back.
"""
from __future__ import annotations

import pytest

from eqbtst import live


def test_the_three_indices_are_the_verified_spellings():
    """NSE:BANKNIFTY-INDEX and NSE:NIFTYFINSERVICE-INDEX are the plausible-looking
    alternatives; both return lp=None from Fyers. Pin the ones that actually quote."""
    assert live.INDEX_STRIP == (
        ("NIFTY 50",          "NSE:NIFTY50-INDEX"),
        ("NIFTY BANK",        "NSE:NIFTYBANK-INDEX"),
        ("NIFTY FIN SERVICE", "NSE:FINNIFTY-INDEX"),
    )


def test_returns_a_row_per_index_in_order(monkeypatch):
    monkeypatch.setattr(live, "_fetch_quotes", lambda syms: {
        "NSE:NIFTY50-INDEX": {"lp": 24080.4, "chp": -0.39},
        "NSE:NIFTYBANK-INDEX": {"lp": 58024.95, "chp": 0.92},
        "NSE:FINNIFTY-INDEX": {"lp": 26293.65, "chp": 0.03},
    })
    live._IDX_CACHE.clear()
    rows, stamp = live.index_quotes()
    assert [r["name"] for r in rows] == ["NIFTY 50", "NIFTY BANK", "NIFTY FIN SERVICE"]
    assert rows[0]["lp"] == pytest.approx(24080.4)
    assert rows[1]["chp"] == pytest.approx(0.92)
    assert stamp is not None


def test_one_index_missing_becomes_none_not_an_exception(monkeypatch):
    """The wrong-symbol case: a row comes back with no lp."""
    monkeypatch.setattr(live, "_fetch_quotes", lambda syms: {
        "NSE:NIFTY50-INDEX": {"lp": 24080.4, "chp": -0.39},
        "NSE:NIFTYBANK-INDEX": {"lp": None, "chp": None},
    })
    live._IDX_CACHE.clear()
    rows, _ = live.index_quotes()
    assert rows[1]["lp"] is None and rows[2]["lp"] is None
    assert rows[0]["lp"] == pytest.approx(24080.4)


def test_a_broker_failure_never_raises(monkeypatch):
    def _boom(syms):
        raise RuntimeError("429 rate limited")
    monkeypatch.setattr(live, "_fetch_quotes", _boom)
    live._IDX_CACHE.clear()
    rows, stamp = live.index_quotes()
    assert len(rows) == 3
    assert all(r["lp"] is None for r in rows)
    assert stamp is None          # nothing quoted -> no misleading timestamp


def test_second_call_is_served_from_the_memo(monkeypatch):
    calls = []
    monkeypatch.setattr(live, "_fetch_quotes",
                        lambda syms: calls.append(1) or {"NSE:NIFTY50-INDEX": {"lp": 1.0, "chp": 0.0}})
    live._IDX_CACHE.clear()
    live.index_quotes()
    live.index_quotes()
    assert len(calls) == 1        # one batched call, not one per render tick
