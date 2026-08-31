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


def test_a_zero_price_is_rejected_not_rendered(monkeypatch):
    """lp <= 0 is not a price. A zero LTP with a -100% change is a known broker failure
    mode off-hours; rendering it would put "0.00 / -100.00%" in the header as a real
    print. It must read as absent instead."""
    monkeypatch.setattr(live, "_fetch_quotes", lambda syms: {
        "NSE:NIFTY50-INDEX": {"lp": 0, "chp": -100.0},
        "NSE:NIFTYBANK-INDEX": {"lp": 58024.95, "chp": 0.92},
        "NSE:FINNIFTY-INDEX": {"lp": -1.0, "chp": -100.0},
    })
    live._IDX_CACHE.clear()
    rows, _ = live.index_quotes()
    assert rows[0]["lp"] is None and rows[0]["chp"] is None
    assert rows[2]["lp"] is None and rows[2]["chp"] is None
    assert rows[1]["lp"] == 58024.95          # the good one survives


def test_a_non_numeric_price_does_not_raise(monkeypatch):
    monkeypatch.setattr(live, "_fetch_quotes", lambda syms: {
        "NSE:NIFTY50-INDEX": {"lp": "n/a", "chp": "n/a"}})
    live._IDX_CACHE.clear()
    rows, stamp = live.index_quotes()
    assert all(r["lp"] is None for r in rows) and stamp is None


# ── points (`ch`) next to the percent ─────────────────────────────────────────────────
def test_points_are_derived_from_the_price_that_is_displayed(monkeypatch):
    """The tile prints lp, so the points must come from lp - prev_close. Deriving keeps
    the three numbers on the tile arithmetically consistent with each other."""
    monkeypatch.setattr(live, "_fetch_quotes", lambda syms: {
        "NSE:NIFTY50-INDEX": {"lp": 24080.4, "prev_close_price": 24175.65,
                              "ch": -95.25, "chp": -0.39},
    })
    live._IDX_CACHE.clear()
    rows, _ = live.index_quotes()
    assert rows[0]["ch"] == pytest.approx(-95.25)
    # chp is re-derived too, so it carries full precision rather than the broker's 2dp
    assert rows[0]["chp"] == pytest.approx((24080.4 / 24175.65 - 1) * 100)


def test_a_disagreeing_broker_change_loses_to_the_derived_one(monkeypatch):
    """If `ch` contradicts lp and prev_close, the version that matches the visible price
    wins — otherwise the tile shows a move that its own numbers do not support."""
    monkeypatch.setattr(live, "_fetch_quotes", lambda syms: {
        "NSE:NIFTY50-INDEX": {"lp": 100.0, "prev_close_price": 90.0,
                              "ch": -999.0, "chp": -999.0},
    })
    live._IDX_CACHE.clear()
    rows, _ = live.index_quotes()
    assert rows[0]["ch"] == pytest.approx(10.0)
    assert rows[0]["chp"] == pytest.approx(11.111111, rel=1e-5)


def test_a_zero_previous_close_drops_the_move_but_keeps_the_level(monkeypatch):
    """prev_close is the denominator: at 0 the percent is meaningless (this is how a
    -100% print appears). The LEVEL is still a real number, so it survives."""
    monkeypatch.setattr(live, "_fetch_quotes", lambda syms: {
        "NSE:NIFTY50-INDEX": {"lp": 24080.4, "prev_close_price": 0, "ch": 0, "chp": -100.0},
    })
    live._IDX_CACHE.clear()
    rows, _ = live.index_quotes()
    assert rows[0]["lp"] == pytest.approx(24080.4)
    assert rows[0]["ch"] is None and rows[0]["chp"] is None


def test_points_absent_when_there_is_no_previous_close(monkeypatch):
    """No prev_close and no broker `ch` — show the percent alone rather than invent one."""
    monkeypatch.setattr(live, "_fetch_quotes", lambda syms: {
        "NSE:NIFTY50-INDEX": {"lp": 24080.4, "chp": -0.39},
    })
    live._IDX_CACHE.clear()
    rows, _ = live.index_quotes()
    assert rows[0]["ch"] is None
    assert rows[0]["chp"] == pytest.approx(-0.39)      # broker value kept as the fallback


def test_a_rejected_price_takes_the_points_with_it(monkeypatch):
    monkeypatch.setattr(live, "_fetch_quotes", lambda syms: {
        "NSE:NIFTY50-INDEX": {"lp": 0, "prev_close_price": 24175.65, "ch": -24175.65,
                              "chp": -100.0},
    })
    live._IDX_CACHE.clear()
    rows, _ = live.index_quotes()
    assert rows[0]["lp"] is None and rows[0]["ch"] is None and rows[0]["chp"] is None
