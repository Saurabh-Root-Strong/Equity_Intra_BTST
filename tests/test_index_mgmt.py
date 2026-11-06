"""Index Management lane — the live Top / Next / Rest decomposition from NSE's feed.

What can go wrong here is quiet: a wrong index name returns an EMPTY list with a 200, a
weight list that does not match the constituents still multiplies, yesterday's open
interest still divides. Each test pins one of those to a visible outcome.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from eqbtst import index_mgmt as im

NOW = dt.datetime(2026, 10, 13, 12, 0, 0)


@pytest.fixture(autouse=True)
def _private_lot_memory(monkeypatch, tmp_path):
    """_lot() remembers proven lot sizes in data/index_mgmt/lots.json. Fixtures with fake
    symbols ('A', 'B') wrote into the REAL file the first time this ran — every test gets
    its own empty memory instead."""
    monkeypatch.setattr(im, "_LOTS_FILE", tmp_path / "lots.json")
    monkeypatch.setattr(im, "_LOTS", {"mem": None})


def _cash(moves: dict, prev_idx=20000.0, idx_pct=1.0, status="Open", ts=NOW):
    return dict(code="NIFTY", ts=ts, status=status, fetched=ts,
                index=dict(last=prev_idx * (1 + idx_pct / 100), prev=prev_idx,
                           open=prev_idx, high=None, low=None),
                stocks={s: dict(ltp=100 * (1 + p / 100), prev=100.0, open=100.0, pct=p,
                                value=5e9, upd=f"{ts:%Y-%m-%d %H:%M:%S}")
                        for s, p in moves.items()})


class _Fake:
    """Stands in for NseClient: path substring -> payload (or an exception to raise)."""
    blocked_until = 0.0

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, path):
        self.calls.append(path)
        for k, v in self.routes.items():
            if k in path:
                if isinstance(v, Exception):
                    raise v
                return v
        raise im.NseError("no route")


# ── transport / parsing ───────────────────────────────────────────────────────────────
def test_the_four_index_names_are_the_ones_nse_actually_quotes():
    """'NIFTY FINANCIAL SERVICES' and 'NIFTY MIDCAP SELECT' answer 200 with an empty list."""
    assert im.NSE_NAME == {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK",
                           "FINNIFTY": "NIFTY FIN SERVICE", "MIDCPNIFTY": "NIFTY MID SELECT"}


def test_an_empty_constituent_list_is_an_error_not_an_empty_index():
    with pytest.raises(im.NseError):
        im.fetch_cash("FINNIFTY", _Fake({"equity-stock-indices": {"data": []}}))


def test_a_zero_price_row_is_dropped_and_the_index_row_is_separated():
    j = {"timestamp": "13-Oct-2026 15:16:27", "marketStatus": {"marketStatus": "Open"},
         "data": [
             {"symbol": "NIFTY 50", "priority": 1, "lastPrice": 22712.2, "previousClose": 22363.35},
             {"symbol": "HDFCBANK", "lastPrice": 714.65, "previousClose": 704.9,
              "totalTradedValue": 1.2e10, "lastUpdateTime": "2026-10-13 15:15:12"},
             {"symbol": "DEAD", "lastPrice": 0, "previousClose": 100}]}
    c = im.fetch_cash("NIFTY", _Fake({"equity-stock-indices": j}))
    assert set(c["stocks"]) == {"HDFCBANK"}
    assert c["index"]["prev"] == pytest.approx(22363.35)
    assert c["stocks"]["HDFCBANK"]["pct"] == pytest.approx((714.65 / 704.9 - 1) * 100)
    assert c["ts"] == dt.datetime(2026, 10, 13, 15, 16, 27)


def test_weights_that_do_not_sum_to_100_are_refused():
    bad = {"data": [{"cmSymbol": "A", "weightage": 10.0}, {"cmSymbol": "B", "weightage": 9.0}]}
    with pytest.raises(im.NseError):
        im.fetch_weights("NIFTY", _Fake({"getConstituents": bad}))


def test_open_interest_is_summed_over_every_expiry_so_a_roll_nets_to_zero():
    """1,000 lots leaving the front month and arriving in the next is not an unwind."""
    rows = [
        {"instrumentType": "FUTSTK", "optionType": "XX", "expiryDate": "27-Oct-2026",
         "openInterest": 9000, "changeinOpenInterest": -1000},
        {"instrumentType": "FUTSTK", "optionType": "XX", "expiryDate": "23-Nov-2026",
         "openInterest": 2000, "changeinOpenInterest": 1000},
        {"instrumentType": "OPTSTK", "optionType": "CE", "expiryDate": "27-Oct-2026",
         "openInterest": 500, "changeinOpenInterest": 50},
        {"instrumentType": "OPTSTK", "optionType": "PE", "expiryDate": "27-Oct-2026",
         "openInterest": 300, "changeinOpenInterest": -30}]
    d = im.fetch_deriv("M&M", f := _Fake({"getSymbolDerivativesData": {"data": rows}}))
    assert d["fut_oi"] == 11000 and d["fut_chg"] == 0
    assert (d["ce_chg"], d["pe_chg"]) == (50, -30)
    assert d["expiries"] == [dt.date(2026, 10, 27), dt.date(2026, 11, 23)]
    assert f.calls[0].endswith("symbol=M%26M")        # '&' must be escaped or NSE reads 'M'


def test_three_failures_open_the_breaker_and_a_success_closes_it(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    class _S:
        def __init__(self): self.ok = False
        def get(self, url, timeout=0):
            class R:
                status_code, text = 200, '{"a": 1}'
                def json(self): return {"a": 1}
            if not self.ok:
                R.status_code = 403
            return R()
    c = im.NseClient(session=(s := _S()), min_gap=0)
    c._prime = lambda: setattr(c, "_primed", True)       # no network in a test
    for _ in range(3):
        with pytest.raises(im.NseError):
            c.get("/api/equity-stock-indices?index=X")
    assert c.blocked_until > 0
    with pytest.raises(im.NseError, match="paused"):
        c.get("/api/equity-stock-indices?index=X")
    # a different family is NOT blocked by the price endpoint's failures
    s.ok = True
    assert c.get(im._DERIV + "RELIANCE") == {"a": 1}
    c.blocked["cash"] = 0.0
    assert c.get("/api/equity-stock-indices?index=X") == {"a": 1} and c.fails["cash"] == 0
    assert (tmp_path / "errors.log").read_text().count("cash") == 3


def test_the_breaker_never_pauses_longer_than_two_minutes(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    class _Bad:
        def get(self, url, timeout=0):
            raise ConnectionError("reset by peer")
    c = im.NseClient(session=_Bad(), min_gap=0)
    c._prime = lambda: setattr(c, "_primed", True)
    import time as _t
    for _ in range(12):
        c.blocked["cash"] = 0.0
        with pytest.raises(im.NseError):
            c.get("/api/equity-stock-indices?index=X")
    assert c.blocked["cash"] - _t.time() <= im.BREAKER_MAX + 1


def test_a_failed_prime_keeps_the_old_session_instead_of_failing_every_call(monkeypatch):
    import requests
    class _Boom(requests.Session):
        def get(self, *a, **k):
            raise requests.ConnectionError("prime page refused")
    monkeypatch.setattr(im.requests, "Session", _Boom)
    old = object()
    c = im.NseClient(session=old, min_gap=0)
    c._primed = False
    c._prime()
    assert c._s is old and c._primed is True          # the API call still gets to try


def test_refresh_button_resets_the_writer_session(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    monkeypatch.setattr(im, "_S", {})
    monkeypatch.setattr(im, "_HEAL", {"at": 0.0, "refresh_seen": 0.0})
    old = im.NseClient(); im._CLIENT[0] = old
    im._st("NIFTY")["cash_at"] = 123.0
    im.request_refresh()
    im._maybe_heal()
    assert im._CLIENT[0] is not old and im._st("NIFTY")["cash_at"] == 0.0
    new = im._CLIENT[0]
    im._maybe_heal()                                   # the same request is honoured once
    assert im._CLIENT[0] is new


def test_stale_prices_in_market_hours_replace_the_session(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    monkeypatch.setattr(im, "_S", {})
    monkeypatch.setattr(im, "_HEAL", {"at": 0.0, "refresh_seen": 0.0})
    monkeypatch.setattr(im, "market_window", lambda now=None: True)
    t0 = dt.datetime(2026, 10, 15, 14, 5)
    monkeypatch.setattr(im, "_now", lambda: t0 + dt.timedelta(minutes=17))
    im._st("NIFTY")["cash"] = dict(status="Open", fetched=t0)
    old = im.NseClient(); im._CLIENT[0] = old
    im._maybe_heal()
    assert im._CLIENT[0] is not old


# ── decomposition ─────────────────────────────────────────────────────────────────────
def test_bucket_points_add_up_to_the_weighted_index_move():
    moves = {"A": 2.0, "B": -1.0, "C": 0.5, "D": 1.5}
    w = {"A": 40.0, "B": 30.0, "C": 20.0, "D": 10.0}
    exp = sum(w[s] / 100 * moves[s] / 100 * 20000 for s in moves)       # 130 points
    snap = im.decompose(_cash(moves, idx_pct=exp / 200), w, {}, "BANKNIFTY", now=NOW)
    assert snap["explained_pts"] == pytest.approx(exp)
    assert sum(b["pts"] for b in snap["buckets"] if b["n_present"]) == pytest.approx(exp)
    assert snap["buckets"][-1]["pts"] is None            # an empty bucket is blank, not 0
    assert snap["residual_pts"] == pytest.approx(0, abs=1e-6)
    assert [b["label"] for b in snap["buckets"]] == ["Top 2", "Next 3", "Rest 0"]
    assert snap["buckets"][0]["pts"] == pytest.approx(0.4 * 0.02 * 20000 - 0.3 * 0.01 * 20000)
    assert snap["spread"] == pytest.approx(exp / 200 - 0.75)


def test_mismatched_weights_hide_points_instead_of_multiplying_a_partial_list():
    snap = im.decompose(_cash({"A": 1.0, "B": 1.0}), {"A": 100.0}, {}, "BANKNIFTY", now=NOW)
    assert snap["have_weights"] is False and snap["explained_pts"] is None
    assert all(b["pts"] is None for b in snap["buckets"])
    assert snap["ew"] == pytest.approx(1.0)                 # needs no weights, still shown
    assert any("disagree" in n for n in snap["notes"])


def test_share_of_move_is_blank_when_the_index_has_barely_moved():
    snap = im.decompose(_cash({"A": 0.05, "B": -0.04}, idx_pct=0.01),
                        {"A": 50.0, "B": 50.0}, {}, "BANKNIFTY", now=NOW)
    assert all(b["share"] is None for b in snap["buckets"] if b["n_present"])


def test_a_stale_feed_during_market_hours_is_said_out_loud():
    old = NOW - dt.timedelta(minutes=9)
    snap = im.decompose(_cash({"A": 1.0}, ts=old), {"A": 100.0}, {}, "BANKNIFTY", now=NOW)
    assert any("minutes old" in n for n in snap["notes"])
    shut = im.decompose(_cash({"A": 1.0}, ts=old, status="Closed"), {"A": 100.0}, {},
                        "BANKNIFTY", now=NOW)
    assert not any("minutes old" in n for n in shut["notes"])


def test_activity_is_not_shown_before_0930():
    assert im.pace_fraction(dt.datetime(2026, 10, 13, 9, 20)) is None
    assert im.pace_fraction(dt.datetime(2026, 10, 13, 12, 2)) == im.config.VOL_PROFILE["12:00"]
    assert im.pace_fraction(dt.datetime(2026, 10, 13, 15, 29)) == 1.0


def test_membership_follows_nse_and_drift_is_reported():
    class _Dcm:
        @staticmethod
        def index_meta(code, day):
            return dict(buckets={"Top 2": ("A", "B"), "Rest 2": ("C", "GONE")},
                        carry_band=0.2, opt_cycle_scale=1.0, source="test")
    rb = im.resolve_buckets("BANKNIFTY", {"A": 40, "B": 30, "C": 20, "NEW": 10},
                            ["A", "B", "C", "NEW"], NOW.date(), _Dcm)
    assert rb["added"] == ["NEW"] and rb["dropped"] == ["GONE"]
    assert rb["buckets"] == {"Top 2": ["A", "B"], "Rest 2": ["C", "NEW"]}


# ── the DCM-backed reads (skipped where the DCM engine is not on this machine) ───────
@pytest.fixture(scope="module")
def dcm():
    m = im._dcm()
    if m is None:
        pytest.skip("Daily_Cash_Market engine not importable here")
    return m


def _deriv(fut_pct, ce_pct=0.0, pe_pct=0.0, expiry=dt.date(2026, 10, 27)):
    return dict(fut_oi=1000 * (1 + fut_pct / 100), fut_chg=10 * fut_pct,
                ce_oi=1000 * (1 + ce_pct / 100), ce_chg=10 * ce_pct,
                pe_oi=1000 * (1 + pe_pct / 100), pe_chg=10 * pe_pct, expiries=[expiry])


def test_the_live_basket_is_dcms_dated_basket(dcm):
    """No copy of the membership lives in this repo — the lane asks DCM for the day's."""
    meta = dcm.index_meta("NIFTY", NOW.date())
    live = [s for mem in meta["buckets"].values() for s in mem]
    rb = im.resolve_buckets("NIFTY", {}, live, NOW.date(), dcm)
    assert rb["buckets"] == {k: list(v) for k, v in meta["buckets"].items()}
    assert not rb["added"] and not rb["dropped"]


def test_futures_words_come_from_dcms_own_vote_rule(dcm):
    meta = dcm.index_meta("BANKNIFTY", NOW.date())
    syms = [s for mem in meta["buckets"].values() for s in mem]
    moves = {s: 1.0 for s in syms}
    w = {s: 100.0 / len(syms) for s in syms}
    snap = im.decompose(_cash(moves), w, {s: _deriv(+2.0) for s in syms}, "BANKNIFTY",
                        dcm=dcm, now=NOW)
    assert all(b["fut_read"] == "long build" for b in snap["buckets"])
    snap = im.decompose(_cash(moves), w, {s: _deriv(-2.0) for s in syms}, "BANKNIFTY",
                        dcm=dcm, now=NOW)
    assert all(b["fut_read"] == "short covering" for b in snap["buckets"])
    # under the 0.5% band nobody votes -> a quorum-less bucket reads mixed, never a side
    snap = im.decompose(_cash(moves), w, {s: _deriv(+0.2) for s in syms}, "BANKNIFTY",
                        dcm=dcm, now=NOW)
    assert all(b["fut_read"] == "mixed" for b in snap["buckets"])
    assert snap["state"] == dcm.STATE_BROAD_UP


def test_options_are_off_on_the_monthly_expiry_day(dcm):
    meta = dcm.index_meta("BANKNIFTY", NOW.date())
    syms = [s for mem in meta["buckets"].values() for s in mem]
    w = {s: 100.0 / len(syms) for s in syms}
    d = {s: _deriv(1.0, ce_pct=-5, pe_pct=5, expiry=NOW.date()) for s in syms}
    snap = im.decompose(_cash({s: 1.0 for s in syms}), w, d, "BANKNIFTY", dcm=dcm, now=NOW)
    assert snap["is_expiry"]
    assert all(b["opt_read"] is None and b["ce_pct"] is None for b in snap["buckets"])
    assert all(b["fut_read"] is not None for b in snap["buckets"])      # futures stay on


def test_a_corporate_action_stock_casts_no_derivatives_vote(dcm):
    meta = dcm.index_meta("BANKNIFTY", NOW.date())
    syms = [s for mem in meta["buckets"].values() for s in mem]
    w = {s: 100.0 / len(syms) for s in syms}
    d = {s: _deriv(2.0) for s in syms}
    snap = im.decompose(_cash({s: 1.0 for s in syms}), w, d, "BANKNIFTY", dcm=dcm,
                        corp={syms[0]: "SPLIT"}, now=NOW)
    row = snap["stocks"].set_index("symbol").loc[syms[0]]
    assert row["fut_oi_pct"] is None or row["fut_oi_pct"] != row["fut_oi_pct"]
    assert any("ex-date today" in n for n in snap["notes"])


# ── the open-interest base (audit 13 Oct 2026) ───────────────────────────────────────
def test_option_change_is_measured_against_yesterdays_full_total_not_the_feeds():
    """NSE's payload lists only contracts traded TODAY. At 09:30 three strikes may have
    printed: the feed's own 'previous total' is then a sliver of yesterday's real one, and
    +100 lots on it reads +10% when the truth is +1%."""
    d = dict(fut_oi=1010.0, fut_chg=10.0, ce_oi=1100.0, ce_chg=100.0,   # feed base: 1,000
             pe_oi=500.0, pe_chg=0.0, fut_prev={dt.date(2026, 10, 27): 1000.0, dt.date(2026, 11, 23): 400.0})
    base = dict(fut=250_000.0, ce=2_500_000.0, pe=1_000_000.0,          # lot 250
                fut_by_exp={dt.date(2026, 10, 27): 250_000.0, dt.date(2026, 11, 23): 100_000.0})
    fut, ce, pe, arch = im.oi_changes(d, base)
    assert arch is True
    assert fut == pytest.approx(1.0)
    assert ce == pytest.approx(1.0)              # 100 on 10,000 contracts, not on 1,000
    assert im.oi_changes(d, None)[1] == pytest.approx(10.0)      # what the feed alone says


def test_a_lot_size_that_is_not_a_whole_number_falls_back_and_says_so():
    """Archive a session behind, or a revised lot: the ratio stops being an integer. Then
    the archive total is NOT yesterday's and must not be used as if it were."""
    d = dict(fut_oi=1010.0, fut_chg=10.0, ce_oi=1100.0, ce_chg=100.0, pe_oi=500.0,
             pe_chg=0.0, fut_prev={dt.date(2026, 10, 27): 1000.0, dt.date(2026, 11, 23): 400.0})
    base = dict(fut=262_000.0, ce=2_500_000.0, pe=1_000_000.0,
                fut_by_exp={dt.date(2026, 10, 27): 262_400.0, dt.date(2026, 11, 23): 104_960.0})          # 262.4 "lots"
    fut, ce, pe, arch = im.oi_changes(d, base)
    assert arch is False and ce == pytest.approx(10.0)


def test_empty_cells_render_blank_not_the_word_none():
    import pandas as pd
    df = im._blank(pd.DataFrame({"Futures OI %": [None, None], "Bucket": ["a", "b"]}),
                   ["Futures OI %"])
    assert str(df["Futures OI %"].dtype) == "Float64" and df["Futures OI %"].isna().all()


# ── the intraday log ──────────────────────────────────────────────────────────────────
def _tape(n=40, step_min=1):
    t0 = dt.datetime(2026, 10, 13, 10, 0)
    return [dict(t=(t0 + dt.timedelta(minutes=i * step_min)).isoformat(), prev=20000.0,
                 last=20000.0 + i, p={"A": i * 0.01, "B": 0.0}) for i in range(n)]


def test_windows_report_points_per_bucket_and_skip_what_the_log_cannot_reach():
    fr = im.tape_frame(_tape(40), {"Top": ["A"], "Rest": ["B"]}, {"A": 50.0, "B": 50.0})
    mv = {r["window"]: r for r in im.window_moves(fr, ["Top", "Rest"])}
    assert mv["last 15 min"]["index"] == pytest.approx(15.0)
    assert mv["last 15 min"]["Top"] == pytest.approx(0.5 * 0.0015 * 20000)
    assert mv["last 30 min"]["Rest"] == pytest.approx(0.0)
    assert "last 60 min" not in mv                    # only 39 minutes logged
    assert any(k.startswith("since 10:00") for k in mv)


def test_a_gap_in_the_log_is_not_passed_off_as_a_window():
    """Dashboard shut 10:05 -> 11:30: 'last 15 min' has no honest anchor."""
    t = _tape(6) + [dict(t="2026-10-13T11:30:00", prev=20000.0, last=20100.0,
                         p={"A": 1.0, "B": 0.0})]
    fr = im.tape_frame(t, {"Top": ["A"], "Rest": ["B"]}, {"A": 50.0, "B": 50.0})
    names = [r["window"] for r in im.window_moves(fr, ["Top", "Rest"])]
    assert names == ["since 10:00 (first log)"]


def test_yesterdays_open_interest_is_never_served_as_todays(monkeypatch):
    monkeypatch.setattr(im, "_D", {"A": dict(day=dt.date(2026, 10, 12), at=1e12, fut_oi=1)})
    assert im._fresh_deriv(["A"], dt.date(2026, 10, 13)) == {}
    assert "A" in im._fresh_deriv(["A"], dt.date(2026, 10, 12))


def test_a_failed_poll_keeps_the_last_good_read(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    monkeypatch.setattr(im, "_S", {})
    j = {"timestamp": "13-Oct-2026 12:00:00", "marketStatus": {"marketStatus": "Open"},
         "data": [{"symbol": "NIFTY BANK", "priority": 1, "lastPrice": 101, "previousClose": 100},
                  {"symbol": "A", "lastPrice": 101, "previousClose": 100}]}
    good = _Fake({"equity-stock-indices": j,
                  "getConstituents": {"data": [{"cmSymbol": "A", "weightage": 100.0}]}})
    assert im.poll_cash("BANKNIFTY", good) is True
    assert im.poll_cash("BANKNIFTY", good) is True        # same NSE timestamp ...
    assert len(im._st("BANKNIFTY")["tape"]) == 1          # ... is logged once, not twice
    assert im.poll_cash("BANKNIFTY", _Fake({})) is False
    s = im._st("BANKNIFTY")
    assert s["cash"] is not None and s["err"]
    assert (tmp_path / "tape_BANKNIFTY_20261013.jsonl").exists()


# ── the "right now" read ─────────────────────────────────────────────────────────────
def _sweeps(spec):
    """spec: [(HH:MM, {sym: (price % since yesterday, fut change vs yesterday)})]"""
    tape, oi = [], []
    for hm, d in spec:
        t = f"2026-10-13T{hm}:00"
        tape.append(dict(t=t, prev=20000.0, last=20000.0, p={s: v[0] for s, v in d.items()}))
        oi.append(dict(t=t, d={s: [1000.0 + v[1], v[1], 0, 0, 0, 0] for s, v in d.items()}))
    return oi, tape


def test_a_fresh_short_build_shows_even_when_the_day_still_reads_long(dcm):
    """Up 2% on the day with OI up = 'long build' in the main table. In the last 30 minutes
    price fell 0.5% while OI rose another 1%: that is fresh shorting, and the window says so."""
    meta = dcm.index_meta("BANKNIFTY", NOW.date())
    buckets = {k: list(v) for k, v in meta["buckets"].items()}
    syms = [s for v in buckets.values() for s in v]
    oi, tape = _sweeps([("13:00", {s: (2.5, 30.0) for s in syms}),
                        ("13:30", {s: (2.0, 40.0) for s in syms})])
    rows = im.recent_flow(oi, tape, buckets, dcm)
    assert {r["window"] for r in rows} == {"last 30 min"}       # 15 and 60 not reachable
    assert all(r["read"] == "short build" for r in rows)
    assert rows[0]["price"] == pytest.approx((1.02 / 1.025 - 1) * 100)
    assert rows[0]["oi"] == pytest.approx(10 / 1030 * 100)


def test_a_contract_printing_its_first_trade_is_not_a_position(dcm):
    """The feed lists only traded contracts, so the OI TOTAL jumps when a far month first
    trades. The window must read the change-vs-yesterday figure, which does not."""
    buckets = {"Top 2": ["A", "B"]}
    tape = [dict(t=f"2026-10-13T{hm}:00", prev=1.0, last=1.0, p={"A": p, "B": p})
            for hm, p in (("13:00", 0.0), ("13:30", 1.0))]
    oi = [dict(t="2026-10-13T13:00:00", d={"A": [1000, 10, 0, 0, 0, 0], "B": [1000, 10, 0, 0, 0, 0]}),
          dict(t="2026-10-13T13:30:00", d={"A": [1500, 10, 0, 0, 0, 0], "B": [1500, 10, 0, 0, 0, 0]})]
    r = im.recent_flow(oi, tape, buckets, dcm)[0]
    assert r["oi"] == pytest.approx(0.0) and r["read"] == "mixed"


def test_a_sweep_is_committed_whole_or_not_at_all(monkeypatch, tmp_path):
    """Mid-sweep the page must not see 7 of 10 names and call the bucket."""
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    monkeypatch.setattr(im, "_S", {})
    monkeypatch.setattr(im, "_D", {})
    s = im._st("BANKNIFTY")
    s["cash"] = _cash({"A": 1.0, "B": 1.0})
    seen = []
    row = {"instrumentType": "FUTSTK", "optionType": "XX", "expiryDate": "27-Oct-2026",
           "openInterest": 100, "changeinOpenInterest": 1}

    class _C(_Fake):
        def get(self, path):
            seen.append(len(im._D))               # what a concurrent page draw would find
            return {"data": [row]}
    assert im.poll_deriv("BANKNIFTY", _C({})) == 2
    assert seen == [0, 0] and len(im._D) == 2


# ── self-tuning from a measured refresh interval ─────────────────────────────────────
def _res(px, oi, n=20):
    return {"price stamp": dict(changes=n, median_s=px, min_s=px, max_s=px),
            "stock OI (futures)": dict(changes=n, median_s=oi, min_s=oi, max_s=oi)}


def test_intervals_follow_the_measurement_but_stay_inside_hard_limits():
    assert im.tuned_intervals(_res(40.0, 90.0)) == (40.0, 90.0)
    assert im.tuned_intervals(_res(3.0, 10.0)) == (15.0, 60.0)       # never hammer NSE
    assert im.tuned_intervals(_res(400.0, 900.0)) == (60.0, 180.0)   # never go blind
    assert im.tuned_intervals(_res(40.0, 90.0, n=2)) == (None, None)  # too few to time


def test_a_saved_measurement_is_adopted_without_a_restart(monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    monkeypatch.setattr(im, "_CADENCE", {"mtime": None, "info": None})
    monkeypatch.setattr(im, "CASH_EVERY", 30.0)
    monkeypatch.setattr(im, "DERIV_EVERY", 120.0)
    assert im.apply_cadence() is None                    # no file: defaults stand
    (tmp_path / "cadence.json").write_text(json.dumps(
        {"measured": "14 Oct 2026 09:51", "result": _res(45.0, 75.0)}), encoding="utf-8")
    info = im.apply_cadence()
    assert (im.CASH_EVERY, im.DERIV_EVERY) == (45.0, 75.0)
    assert info["measured"] == "14 Oct 2026 09:51"


# ── column tooltips ──────────────────────────────────────────────────────────────────
def test_every_column_of_every_table_has_a_plain_english_tooltip():
    """A heading with no explanation gets read as whatever the reader hopes it means."""
    src = (pathlib.Path(__file__).parent.parent / "eqbtst" / "index_mgmt.py").read_text(
        encoding="utf-8")
    draw = src[src.index("def _draw("):src.index("def render_page(")]
    assert draw.count("st.dataframe(") == draw.count("column_config=_cfg("), (
        "a table on the page has no tooltips")
    tables = (im.TIP_MAIN, im.TIP_SOURCE, im.TIP_EOD, im.TIP_WINDOW, im.TIP_FRESH,
              im.TIP_STOCK, im.TIP_BASE)
    for spec in tables:
        for col, (fmt, tip) in spec.items():
            assert len(tip) >= 20, f"{col}: tooltip too thin to explain anything"
    # the columns the page builds must be exactly the columns that are explained
    for col in ("Bucket", "Weight %", "Index pts", "Share of move %", "Return %", "Adv",
                "Activity %", "Value share %", "Usual share %", "Futures OI %", "Futures now", "Stock votes", "Call OI %",
                "Put OI %", "Options now", "F&O names"):
        assert col in im.TIP_MAIN and f'"{col}"' in draw
    assert set(im.TIP_SOURCE) == set(im._provenance(
        dict(ts=NOW, day=NOW.date()), None).columns)
    for spec, cols in ((im.TIP_FRESH, ("Window", "Bucket", "Price %", "Futures OI %",
                                       "Futures in this window", "Stock votes", "Names")),
                       (im.TIP_STOCK, ("Stock", "Bucket", "Weight %", "Move %", "Index pts",
                                       "Futures OI %", "Futures", "Call OI %", "Put OI %",
                                       "Traded ₹ Cr")),
                       (im.TIP_EOD, ("Bucket", "Return %", "Futures at that close",
                                     "Options at that close", "Delivery % (last 4 closes)"))):
        assert tuple(spec) == cols


def test_the_big_tooltips_carry_an_example_and_no_unexplained_jargon():
    for col in ("Index pts", "Share of move %", "Return %", "Activity %", "Futures OI %",
                "Stock votes", "Call OI %", "Put OI %"):
        assert "**Example:**" in im.TIP_MAIN[col][1], f"{col}: no worked example"
    every = " ".join(tip for spec in (im.TIP_MAIN, im.TIP_FRESH, im.TIP_STOCK, im.TIP_EOD,
                                      im.TIP_WINDOW, im.TIP_SOURCE) for _, tip in spec.values())
    for jargon in ("quorum", "winsor", "pro-rat", "cap-weighted", "equal-weight",
                   "decomposition", "roll-immune", "basis point", "z-score"):
        assert jargon not in every.lower(), f"tooltip still says '{jargon}'"
    assert "100 bps = 1%" in im.TIP_BASE["Next open gap (bps)"][1]      # explained where used


# ── one writer, readers follow its log ───────────────────────────────────────────────
def test_only_one_process_writes_and_a_dead_writer_is_replaced(monkeypatch, tmp_path):
    import json, time as _t
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    (tmp_path / "poller.lock").write_text(json.dumps({"pid": -1, "beat": _t.time()}))
    assert im._claim() is False                      # a live writer elsewhere: read only
    (tmp_path / "poller.lock").write_text(json.dumps({"pid": -1, "beat": _t.time() - 120}))
    assert im._claim() is True                       # it stopped refreshing: take over
    im._release()
    assert not (tmp_path / "poller.lock").exists()


def test_a_reader_sees_exactly_what_the_writer_logged(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    monkeypatch.setattr(im, "_S", {})
    monkeypatch.setattr(im, "_D", {})
    j = {"timestamp": "14-Oct-2026 10:58:12", "marketStatus": {"marketStatus": "Open"},
         "data": [{"symbol": "NIFTY BANK", "priority": 1, "lastPrice": 101, "previousClose": 100},
                  {"symbol": "A", "lastPrice": 101, "previousClose": 100}]}
    row = {"instrumentType": "FUTSTK", "optionType": "XX", "expiryDate": "27-Oct-2026",
           "openInterest": 100, "changeinOpenInterest": 1}
    f = _Fake({"equity-stock-indices": j, "getSymbolDerivativesData": {"data": [row]},
               "getConstituents": {"data": [{"cmSymbol": "A", "weightage": 100.0}]}})
    assert im.poll_cash("BANKNIFTY", f) and im.poll_deriv("BANKNIFTY", f) == 1
    writer = im.snapshot("BANKNIFTY")
    # a fresh process: nothing in memory, only the files
    monkeypatch.setattr(im, "_S", {})
    monkeypatch.setattr(im, "_D", {})
    monkeypatch.setattr(im, "_SEEN", {})
    monkeypatch.setattr(im, "_MINE", {"owner": False})
    reader = im.snapshot("BANKNIFTY")
    assert reader["ts"] == writer["ts"] and reader["index"] == writer["index"]
    assert reader["n_deriv"] == 1 and len(reader["tape"]) == len(writer["tape"]) == 1
    assert reader["stocks"]["fut_oi_pct"].tolist() == writer["stocks"]["fut_oi_pct"].tolist()


def test_capture_runs_in_market_hours_only():
    assert im.market_window(dt.datetime(2026, 10, 14, 9, 5))
    assert im.market_window(dt.datetime(2026, 10, 14, 15, 35))
    assert not im.market_window(dt.datetime(2026, 10, 14, 8, 55))
    assert not im.market_window(dt.datetime(2026, 10, 17, 11, 0))      # Saturday


def test_the_board_starts_capture_on_load_not_on_click():
    txt = (pathlib.Path(__file__).parent.parent / "eqbtst" / "dashboard.py").read_text(
        encoding="utf-8")
    head = txt[:txt.index('if tf == "🏛️ Index Management"')]
    assert "index_mgmt.ensure_poller()" in head


# ── audit 14 Oct 2026 ────────────────────────────────────────────────────────────────
def test_the_options_norm_is_not_applied_in_full_before_0930():
    assert im.session_fraction(dt.datetime(2026, 10, 14, 9, 16)) == pytest.approx(
        im.config.VOL_PROFILE["09:15"])
    assert im.session_fraction(dt.datetime(2026, 10, 14, 9, 10)) == 0.0
    assert im.session_fraction(dt.datetime(2026, 10, 14, 15, 28)) == 1.0
    assert im.pace_fraction(dt.datetime(2026, 10, 14, 9, 16)) is None     # Activity stays blank


def test_nothing_is_logged_outside_0915_1530():
    assert im.in_session(dt.datetime(2026, 10, 14, 9, 15))
    assert im.in_session(dt.datetime(2026, 10, 14, 15, 30))
    assert not im.in_session(dt.datetime(2026, 10, 14, 15, 32))     # NSE still said 'Open'
    assert not im.in_session(dt.datetime(2026, 10, 14, 9, 10))


def test_ban_list_is_read_only_for_its_own_trade_date():
    class _R:
        status_code = 200
        text = "Securities in Ban For Trade Date 14-OCT-2026:\n1,AMBUJACEM\n2,KAYNES\n"

    class _Sess:
        headers = {}
        def get(self, url, timeout=0): return _R()
    im._CTX.pop(("ban", dt.date(2026, 10, 14)), None)
    im._CTX.pop(("ban", dt.date(2026, 10, 15)), None)
    assert im.fetch_ban_list(dt.date(2026, 10, 14), _Sess()) == {"AMBUJACEM", "KAYNES"}
    assert im.fetch_ban_list(dt.date(2026, 10, 15), _Sess()) is None    # stale file: ignored


def test_a_banned_stock_casts_no_derivatives_vote(dcm):
    meta = dcm.index_meta("BANKNIFTY", NOW.date())
    syms = [s for mem in meta["buckets"].values() for s in mem]
    w = {s: 100.0 / len(syms) for s in syms}
    snap = im.decompose(_cash({s: 1.0 for s in syms}), w, {s: _deriv(-2.0) for s in syms},
                        "BANKNIFTY", dcm=dcm, corp={syms[0]: "F&O BAN"}, now=NOW)
    row = snap["stocks"].set_index("symbol").loc[syms[0]]
    assert row["quadrant"] is None or row["quadrant"] != row["quadrant"]   # None or NaN
    assert any("F&O ban" in n for n in snap["notes"])


# ── the clocks above the table ───────────────────────────────────────────────────────
def test_the_four_clocks_are_reported_separately_and_graded():
    now = dt.datetime(2026, 10, 15, 9, 43, 48)
    snap = dict(live=True, ts=dt.datetime(2026, 10, 15, 9, 41, 52),          # 116s
                price_read_at=dt.datetime(2026, 10, 15, 9, 43, 36),          # 12s
                oi_nse_to=dt.datetime(2026, 10, 15, 9, 36, 0),               # 468s
                deriv_ts=dt.datetime(2026, 10, 15, 9, 43, 0), stock_lag_s=75.0)
    f = im.freshness(snap, now)
    assert f["price_age"] == pytest.approx(116) and f["price_level"] == "ok"
    assert f["price_read_age"] == pytest.approx(12)
    assert f["oi_level"] == "stale"                      # over 6 minutes: red
    assert f["next_price"] == pytest.approx(im.CASH_EVERY - 12)
    assert f["next_oi"] == pytest.approx(max(0.0, im.DERIV_EVERY - 48))


def test_a_closed_market_is_grey_not_red():
    snap = dict(live=False, ts=dt.datetime(2026, 10, 14, 16, 0), price_read_at=None,
                oi_nse_to=None, deriv_ts=None)
    f = im.freshness(snap, dt.datetime(2026, 10, 15, 8, 0))
    assert f["price_level"] == "closed" and f["oi_level"] == "none"
    assert f["next_price"] is None and f["next_oi"] is None


def test_the_time_strip_sits_above_the_bucket_table():
    src = (pathlib.Path(__file__).parent.parent / "eqbtst" / "index_mgmt.py").read_text(
        encoding="utf-8")
    draw = src[src.index("def _draw("):]
    assert draw.index("_time_strip(st, snap)") < draw.index("_blank(pd.DataFrame(tbl)")


def test_out_of_step_feeds_are_flagged_only_when_they_matter():
    assert im.out_of_step(31.65, 38.06)            # Midcap Select, 15 Oct: 20% miss
    assert not im.out_of_step(256.15, 262.14)      # Nifty: 6 pts on 256 = 2%
    assert not im.out_of_step(0.8, 2.5)            # tiny move, but under 2 points
    assert not im.out_of_step(10.0, None)


def test_the_dashboard_prints_no_stray_value():
    """A bare `f() if x else None` at top level is rendered by Streamlit's magic as 'None'."""
    import ast
    src = (pathlib.Path(__file__).parent.parent / "eqbtst" / "dashboard.py").read_text(
        encoding="utf-8")
    bad = [n.lineno for n in ast.walk(ast.parse(src))
           if isinstance(n, ast.Expr) and isinstance(n.value, ast.IfExp)]
    assert not bad, f"bare conditional expressions Streamlit would print: lines {bad}"


# ── money vs move ────────────────────────────────────────────────────────────────────
def _vtape():
    t0 = dt.datetime(2026, 10, 15, 12, 0)
    rows = []
    for i in range(31):                                   # 12:00 .. 12:30, one row a minute
        rows.append(dict(t=(t0 + dt.timedelta(minutes=i)).isoformat(), prev=1.0, last=1.0,
                         p={"A": -0.02 * i, "B": 0.0},          # A slides 0.6% in 30 min
                         v={"A": 100.0 + 10.0 * i, "B": 50.0 + 1.0 * i}))
    return rows


def test_heavy_selling_is_named_from_price_and_value_against_the_time_of_day_normal():
    norms = {"A": 1000.0, "B": 1000.0}
    rows = im.effort_result(_vtape(), {"Top": ["A"], "Rest": ["B"]}, {"A": 60.0, "B": 40.0},
                            norms, minutes=(30,))
    r30 = {r["bucket"]: r for r in rows if r["window"] == "last 30 min"}
    f = im.session_fraction(dt.datetime(2026, 10, 15, 12, 30)) - im.session_fraction(
        dt.datetime(2026, 10, 15, 12, 0))
    assert r30["Top"]["value_cr"] == pytest.approx(300.0)
    assert r30["Top"]["ratio"] == pytest.approx(300.0 / (1000.0 * f))
    assert r30["Top"]["price"] == pytest.approx(-0.6, abs=1e-6)
    assert r30["Top"]["word"] == "heavy selling"
    assert r30["Rest"]["word"] == "quiet"                 # 30 Cr against ~60 normal, flat
    assert r30["Top"]["value_share"] + r30["Rest"]["value_share"] == pytest.approx(100.0)
    assert {r["window"] for r in rows} == {"last 30 min", "since the open"}


def test_the_effort_words_need_a_real_move_and_scale_with_the_window():
    assert im._effort_word(0.25, 2.0, 30) == "heavy buying"
    assert im._effort_word(0.15, 2.0, 30) == "heavy trade, no progress"
    assert im._effort_word(0.15, 2.0, 15) == "heavy buying"         # 0.14% band at 15 min
    assert im._effort_word(-0.6, 0.5, 0) == "quiet fall"            # day: 0.5% band
    assert im._effort_word(None, 2.0, 30) is None


def test_rows_logged_before_value_was_recorded_are_skipped_not_misread():
    old = [dict(t="2026-10-15T12:00:00", prev=1.0, last=1.0, p={"A": 0.1})]
    assert im.effort_result(old, {"Top": ["A"]}, {"A": 100.0}, {"A": 1.0}) == []


# ── what gets stored ─────────────────────────────────────────────────────────────────
def test_a_price_change_is_logged_even_when_nses_stamp_did_not_move(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    monkeypatch.setattr(im, "_S", {})
    monkeypatch.setattr(im, "_save_meta", lambda code: None)
    def feed(px):
        return {"timestamp": "15-Oct-2026 12:00:00", "marketStatus": {"marketStatus": "Open"},
                "data": [{"symbol": "NIFTY BANK", "priority": 1, "lastPrice": 101, "previousClose": 100},
                         {"symbol": "A", "lastPrice": px, "previousClose": 100,
                          "totalTradedValue": 1e9}]}
    w = {"getConstituents": {"data": [{"cmSymbol": "A", "weightage": 100.0}]}}
    im.poll_cash("BANKNIFTY", _Fake({"equity-stock-indices": feed(101), **w}))
    im.poll_cash("BANKNIFTY", _Fake({"equity-stock-indices": feed(101), **w}))   # nothing new
    import time as _t; _t.sleep(1.1)
    im.poll_cash("BANKNIFTY", _Fake({"equity-stock-indices": feed(102), **w}))   # price moved
    tape = im._st("BANKNIFTY")["tape"]
    assert len(tape) == 2
    assert tape[0]["ns"] == tape[1]["ns"] == "2026-10-15T12:00:00"
    assert tape[1]["t"] > tape[0]["t"]                 # timed by when it was read


def test_per_expiry_detail_is_kept_for_the_log():
    rows = [
        {"instrumentType": "FUTSTK", "optionType": "XX", "expiryDate": "27-Oct-2026",
         "openInterest": 900, "changeinOpenInterest": -10, "totalTradedVolume": 50, "lastPrice": 101.5},
        {"instrumentType": "OPTSTK", "optionType": "CE", "expiryDate": "27-Oct-2026",
         "openInterest": 300, "changeinOpenInterest": 5, "totalTradedVolume": 20},
        {"instrumentType": "OPTSTK", "optionType": "CE", "expiryDate": "27-Oct-2026",
         "openInterest": 200, "changeinOpenInterest": 1, "totalTradedVolume": 7}]
    d = im.fetch_deriv("A", _Fake({"getSymbolDerivativesData": {"data": rows}}))
    assert d["by_exp"]["F"]["2026-10-27"] == [900, -10, 50, 101.5, None]   # no prevClose sent
    assert d["by_exp"]["C"]["2026-10-27"] == [500, 6, 27]


def test_finished_days_are_compressed_and_still_readable(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    old = tmp_path / "tape_NIFTY_20261014.jsonl"
    old.write_text('{"t": "a", "p": {}}\n{"t": "b", "p": {}}\n', encoding="utf-8")
    today = tmp_path / "tape_NIFTY_20261015.jsonl"
    today.write_text('{"t": "c", "p": {}}\n', encoding="utf-8")
    assert im.compress_old_logs(dt.date(2026, 10, 15)) == 1
    assert not old.exists() and today.exists()
    assert [r["t"] for r in im._load_jsonl(old)] == ["a", "b"]


# ── session charts: open interest and futures price ──────────────────────────────────
def _sweep(t, stocks):
    """stocks: {sym: (fut_oi, fut_chg, ce_oi, ce_chg, pe_oi, pe_chg, {exp: [oi, chg, vol, ltp, prev]}, spot)}"""
    return dict(t=t, d={s: list(v[:6]) for s, v in stocks.items()},
                x={s: {"F": v[6], "C": {}, "P": {}, "S": v[7]} for s, v in stocks.items()})


def test_session_series_uses_the_tables_arithmetic_and_the_near_month():
    day = dt.date(2026, 10, 16)
    F = {"2026-10-27": [1010.0, 10.0, 5.0, 101.0, 100.0], "2026-11-23": [500.0, 0.0, 1.0, 102.0, 101.0]}
    sw = [_sweep("2026-10-16T10:00:00", {"A": (1010, 10, 1100, 100, 500, 0, F, 100.5),
                                         "B": (1010, 10, 1100, 100, 500, 0, F, 100.5)}),
          _sweep("2026-10-16T10:02:00", {"A": (1020, 20, 1100, 100, 500, 0, F, 100.5),
                                         "B": (1020, 20, 1100, 100, 500, 0, F, 100.5)})]
    base = {s: dict(fut=250_000.0, ce=2_500_000.0, pe=1_000_000.0,
                    fut_by_exp={dt.date(2026, 10, 27): 250_000.0, dt.date(2026, 11, 23): 125_000.0})
            for s in ("A", "B")}
    ser = im.session_series(sw, [], {"Top": ["A", "B"]}, {"A": 60.0, "B": 40.0}, base, {}, {}, day)
    # lot 250 -> yesterday 1000 fut contracts; options base 10,000 calls (not the feed's 1,000)
    # base["fut"] / lot 250 = 1,000 contracts yesterday
    assert ser["fut_oi"]["Top"].tolist() == pytest.approx([1.0, 2.0])
    assert ser["ce"]["Top"].iloc[0] == pytest.approx(1.0)
    assert ser["fut_px"]["Top"].iloc[0] == pytest.approx(1.0)          # Oct contract 101/100
    assert ser["basis"]["Top"].iloc[0] == pytest.approx((101 / 100.5 - 1) * 1e4)


def test_on_expiry_day_the_next_month_is_followed():
    day = dt.date(2026, 10, 27)
    F = {"2026-10-27": [1000.0, 0.0, 5.0, 100.0, 100.0], "2026-11-23": [500.0, 0.0, 1.0, 103.0, 101.0]}
    sw = [_sweep("2026-10-27T10:00:00", {"A": (1500, 0, 0, 0, 0, 0, F, 100.0)})]
    ser = im.session_series(sw, [], {"Top": ["A"]}, {"A": 100.0}, {}, {}, {}, day)
    assert ser["fut_px"]["Top"].iloc[0] == pytest.approx((103 / 101 - 1) * 100)
    assert ser["basis"]["Top"].iloc[0] == pytest.approx(300.0)


def test_readings_without_a_stored_prev_close_use_the_archive_and_thin_readings_are_dropped():
    day = dt.date(2026, 10, 16)
    F = {"2026-10-27": [1000.0, 0.0, 5.0, 99.0]}                       # no prev kept
    sw = [_sweep("2026-10-16T10:00:00", {"A": (1000, 0, 0, 0, 0, 0, F, None)})]
    ser = im.session_series(sw, [], {"Top": ["A"], "Rest": ["B", "C", "D"]}, {"A": 100.0},
                            {}, {"A": {"2026-10-27": 100.0}}, {}, day)
    assert ser["fut_px"]["Top"].iloc[0] == pytest.approx(-1.0)
    assert pd.isna(ser["fut_px"]["Rest"].iloc[0])                       # 0 of 3 stocks


# ── block-by-block quadrant flow ─────────────────────────────────────────────────────
def _rd(t, chg, ltp, stock="A"):
    """one reading: futures change vs yesterday `chg` contracts, near-month price `ltp`"""
    F = {"2026-10-27": [1000.0 + chg, chg, 10.0, ltp, 100.0],
         "2026-11-23": [400.0, 0.0, 0.0, ltp, 100.0]}
    return dict(t=t, d={stock: [1000.0 + chg, chg, 0, 0, 0, 0]}, x={stock: {"F": F, "S": None}})


_BASE_A = {"A": dict(fut=250_000.0, ce=0.0, pe=0.0, fut_by_exp={dt.date(2026, 10, 27): 250_000.0, dt.date(2026, 11, 23): 100_000.0})}


def test_each_block_puts_rupees_in_the_right_quadrant():
    day = dt.date(2026, 10, 16)
    tape = [_rd("2026-10-16T10:01:00", 0, 100.0),
            _rd("2026-10-16T10:06:00", 40, 101.0),      # +40 contracts, price up  -> long build
            _rd("2026-10-16T10:11:00", 10, 100.0)]      # -30 contracts, price down -> long unwind
    res = im.oi_flow_blocks(tape, {"Top": ["A"]}, {"A": 100.0}, _BASE_A, {}, day, minutes=5)
    f = res["flow"].set_index("quadrant")
    # lot = 250,000 / 1,000 = 250 shares; Rs = contracts x 250 x price / 1e7
    assert f.loc["long build", "cr"] == pytest.approx(40 * 250 * 101.0 / 1e7)
    assert f.loc["long unwind", "cr"] == pytest.approx(-30 * 250 * 100.0 / 1e7)
    assert len(res["flow"]) == 2                        # first reading is the baseline
    assert res["price"]["Top"].iloc[-1] == pytest.approx(0.0)   # 100 vs prev close 100


def test_a_logging_gap_is_not_squeezed_into_one_bar():
    day = dt.date(2026, 10, 16)
    tape = [_rd("2026-10-16T10:01:00", 0, 100.0), _rd("2026-10-16T11:31:00", 400, 103.0)]
    res = im.oi_flow_blocks(tape, {"Top": ["A"]}, {"A": 100.0}, _BASE_A, {}, day, minutes=5)
    assert res["flow"].empty


def test_an_unusual_block_is_flagged_against_the_sessions_typical_block():
    day = dt.date(2026, 10, 16)
    tape, chg, px = [], 0, 100.0
    for i in range(10):                                  # nine ordinary blocks of +4 contracts
        tape.append(_rd(f"2026-10-16T10:{i * 5 + 1:02d}:00", chg, px))
        chg += 4; px += 0.1
    tape.append(_rd("2026-10-16T10:51:00", chg - 60, px - 1.0))   # then -60: a sharp unwind
    res = im.oi_flow_blocks(tape, {"Top": ["A"]}, {"A": 100.0}, _BASE_A, {}, day, minutes=5)
    assert len(res["sharp"]) == 1
    s = res["sharp"][0]
    assert s["main"] == "long unwind" and s["net"] < 0 and s["times"] >= im.FLOW_SHARP_X


def test_options_blocks_value_each_leg_at_the_share_price_and_tag_the_price_move():
    day = dt.date(2026, 10, 16)
    def rd(t, ce_chg, pe_chg, ltp, spot):
        F = {"2026-10-27": [1000.0, 0.0, 10.0, ltp, 100.0], "2026-11-23": [400.0, 0.0, 0.0, ltp, 100.0]}
        return dict(t=t, d={"A": [1000.0, 0.0, 5000.0 + ce_chg, ce_chg, 3000.0 + pe_chg, pe_chg]},
                    x={"A": {"F": F, "S": spot}})
    tape = [rd("2026-10-16T10:01:00", 0, 0, 100.0, 99.8),
            rd("2026-10-16T10:06:00", 0, 200, 101.0, 100.8),     # puts opened, price up
            rd("2026-10-16T10:11:00", 150, 200, 100.0, 99.8)]    # calls opened, price down
    res = im.oi_flow_blocks(tape, {"Top": ["A"]}, {"A": 100.0}, _BASE_A, {}, day, minutes=5)
    o = res["options"].set_index(["leg", "move"])["cr"]
    assert o[("Puts", "price up")] == pytest.approx(200 * 250 * 100.8 / 1e7)
    assert o[("Calls", "price down")] == pytest.approx(150 * 250 * 99.8 / 1e7)
    assert len(res["options"]) == 2
    assert res["flow"].empty                           # no futures positions moved


def test_calls_minus_puts_is_summed_per_bucket_and_a_skipped_stock_keeps_its_value():
    day = dt.date(2026, 10, 16)
    F = {"2026-10-27": [1000.0, 0.0, 10.0, 100.0, 100.0], "2026-11-23": [400.0, 0.0, 0.0, 100.0, 100.0]}
    def rd(t, stocks):
        return dict(t=t, d={s: [1000.0, 0.0, 0.0, ce, 0.0, pe] for s, (ce, pe) in stocks.items()},
                    x={s: {"F": F, "S": 100.0} for s in stocks})
    base = {s: dict(fut=250_000.0, ce=0.0, pe=0.0, fut_by_exp={dt.date(2026, 10, 27): 250_000.0, dt.date(2026, 11, 23): 100_000.0})
            for s in ("A", "B")}
    tape = [rd("2026-10-16T10:00:00", {"A": (100, 20), "B": (0, 50)}),
            rd("2026-10-16T10:02:00", {"A": (140, 20)})]               # B skipped this reading
    cp = im.call_put_series(tape, {"Top": ["A", "B"]}, base, day)
    unit = 250 * 100.0 / 1e7                                          # Rs Cr per contract
    assert cp["Top"].iloc[0] == pytest.approx((80 - 50) * unit)       # calls heavier
    assert cp["Top"].iloc[1] == pytest.approx((120 - 50) * unit)      # B carried, not dropped


def test_bullish_quadrants_plot_up_bearish_down_and_the_pressure_line_adds_them():
    import pandas as pd
    t1, t2 = pd.Timestamp("2026-10-16 10:05"), pd.Timestamp("2026-10-16 10:10")
    flow = pd.DataFrame([
        dict(end=t1, start=t1, bucket="Top", quadrant="long build", cr=30.0, stocks=1),
        dict(end=t1, start=t1, bucket="Top", quadrant="short covering", cr=-20.0, stocks=1),
        dict(end=t1, start=t1, bucket="Top", quadrant="long unwind", cr=-5.0, stocks=1),
        dict(end=t2, start=t1, bucket="Top", quadrant="short build", cr=60.0, stocks=1),
        dict(end=t2, start=t1, bucket="Top", quadrant="price unchanged", cr=9.0, stocks=1)])
    s = im.by_side(flow).set_index("quadrant")["side_cr"]
    assert s["short covering"] == 20.0 and s["long build"] == 30.0     # both UP
    assert s["long unwind"] == -5.0 and s["short build"] == -60.0      # both DOWN
    assert "price unchanged" not in s.index
    p = im.pressure_series(flow, ["Top"])["Top"].tolist()
    assert p == [45.0, -15.0]                                          # 30+20-5, then -60


# ── audit 18 Oct: block-free pressure, gaps, exclusions ─────────────────────────────
def _pl_reading(t, stocks):
    """stocks: {sym: (fut change vs yesterday, near-month ltp)}; prev close 100, lot 250"""
    return dict(t=t, d={s: [1000.0 + c, c, 0, 0, 0, 0] for s, (c, _) in stocks.items()},
                x={s: {"F": {"2026-10-27": [1000.0 + c, c, 1.0, p, 100.0],
                         "2026-11-23": [400.0, 0.0, 0.0, p, 100.0]}, "S": None}
                   for s, (c, p) in stocks.items()})


def test_pressure_level_depends_only_on_end_points_and_matches_the_quadrants():
    day = dt.date(2026, 10, 16)
    base = {s: dict(fut=250_000.0, ce=0.0, pe=0.0, fut_by_exp={dt.date(2026, 10, 27): 250_000.0, dt.date(2026, 11, 23): 100_000.0})
            for s in ("A", "B")}
    tape = [_pl_reading("2026-10-16T10:00:00", {"A": (10, 101.0), "B": (5, 99.0)}),
            _pl_reading("2026-10-16T10:01:00", {"A": (-20, 99.0), "B": (40, 101.0)}),
            _pl_reading("2026-10-16T10:02:00", {"A": (30, 102.0), "B": (-8, 98.0)})]
    lv = im.pressure_level(tape, {"Top": ["A", "B"]}, base, {}, day)
    u = 250 / 1e7
    # last reading: A price up + OI up = long build 30*250*102; B down + OI down = unwind 8*250*98
    assert lv["net"]["Top"].iloc[-1] == pytest.approx(30 * u * 102 - 8 * u * 98)
    # the middle readings cannot affect the last value: drop them, same answer
    lv2 = im.pressure_level([tape[0], tape[-1]], {"Top": ["A", "B"]}, base, {}, day)
    assert lv2["net"]["Top"].iloc[-1] == pytest.approx(lv["net"]["Top"].iloc[-1])
    p = lv["parts"].iloc[-1]
    assert p["long build"] == pytest.approx(30 * u * 102) and p["long unwind"] == pytest.approx(8 * u * 98)


def test_pressure_since_open_subtracts_the_first_reading():
    day = dt.date(2026, 10, 16)
    base = {"A": dict(fut=250_000.0, ce=0.0, pe=0.0, fut_by_exp={dt.date(2026, 10, 27): 250_000.0, dt.date(2026, 11, 23): 100_000.0})}
    tape = [_pl_reading("2026-10-16T09:17:00", {"A": (100, 103.0)}),     # overnight build
            _pl_reading("2026-10-16T10:00:00", {"A": (80, 102.0)})]      # since open: -20, price down
    lv = im.pressure_level(tape, {"Top": ["A"]}, base, {}, day, since="open")
    assert lv["net"]["Top"].iloc[-1] == pytest.approx(-20 * 250 * 102.0 / 1e7)   # long unwind


def test_lines_break_at_a_logging_gap():
    import pandas as pd
    long = pd.DataFrame({"t": pd.to_datetime(["2026-10-16 11:00", "2026-10-16 11:05",
                                              "2026-10-16 13:50", "2026-10-16 13:55"]),
                         "series": ["Top"] * 4, "v": [1, 2, 3, 4]})
    seg = im._with_segments(long, "t", "series")
    assert seg["seg"].tolist() == [0, 0, 1, 1]


def test_withheld_stocks_are_kept_out_of_the_charts_too():
    src = (pathlib.Path(__file__).parent.parent / "eqbtst" / "index_mgmt.py").read_text(encoding="utf-8")
    assert 'snap["withheld"] = set(corp)' in src
    assert "if s not in _w] for l, m in buckets.items()}" in src
    draw = src[src.index("def _draw("):]
    assert "st.line_chart(" not in src, "a line chart that joins across logging gaps is back"


def test_a_ban_list_miss_is_not_refetched_on_every_redraw(monkeypatch):
    calls = []
    class _R:
        status_code = 200
        text = "\n".join(["Securities in Ban For Trade Date 19-OCT-2026:", "1,SAIL", ""])
    class _Sess:
        headers = {}
        def get(self, url, timeout=0):
            calls.append(1); return _R()
    monkeypatch.setattr(im.requests, "Session", _Sess)
    d = dt.date(2026, 10, 16)
    im._CTX.pop(("ban", d), None); im._CTX.pop(("ban-miss", d), None)
    assert im.fetch_ban_list(d) is None and im.fetch_ban_list(d) is None
    assert len(calls) == 1


def test_block_bars_are_steps_of_the_running_line_and_add_up_to_it():
    import pandas as pd
    idx = pd.to_datetime(["2026-10-16 09:21", "2026-10-16 09:23", "2026-10-16 09:26",
                          "2026-10-16 09:31", "2026-10-16 09:36", "2026-10-16 09:42"])
    net = pd.Series([80, 70, -70, -140, -70, 60], index=idx, dtype=float)
    nb = im.net_blocks(net, 5)
    assert nb["cum"].tolist() == [70, -70, -140, -70, 60]
    assert nb["step"].tolist() == [70, -140, -70, 70, 130]
    assert nb["step"].sum() == nb["cum"].iloc[-1]            # bars add up to the line
    nb15 = im.net_blocks(net, 15)
    assert nb15["cum"].iloc[-1] == nb["cum"].iloc[-1]        # block length cannot move the line


def test_no_bar_is_drawn_across_a_logging_gap():
    import pandas as pd
    idx = pd.to_datetime(["2026-10-16 11:00", "2026-10-16 11:05", "2026-10-16 13:50"])
    nb = im.net_blocks(pd.Series([10.0, 20.0, 500.0], index=idx), 5)
    assert pd.isna(nb["step"].iloc[-1]) and nb["cum"].iloc[-1] == 500.0


def test_a_partial_constituent_list_is_rejected_and_the_last_good_read_kept(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    monkeypatch.setattr(im, "_S", {})
    monkeypatch.setattr(im, "_save_meta", lambda code: None)
    def feed(n):
        rows = [{"symbol": "NIFTY BANK", "priority": 1, "lastPrice": 101, "previousClose": 100}]
        rows += [{"symbol": f"S{i}", "lastPrice": 101, "previousClose": 100} for i in range(n)]
        return {"timestamp": "19-Oct-2026 09:15:55", "marketStatus": {"marketStatus": "Open"}, "data": rows}
    w = {"getConstituents": {"data": [{"cmSymbol": f"S{i}", "weightage": 100.0 / 25} for i in range(25)]}}
    assert im.poll_cash("BANKNIFTY", _Fake({"equity-stock-indices": feed(25), **w})) is True
    assert im.poll_cash("BANKNIFTY", _Fake({"equity-stock-indices": feed(18), **w})) is False
    s = im._st("BANKNIFTY")
    assert len(s["cash"]["stocks"]) == 25 and "partial list, 18 of 25" in s["err"]
    assert "partial list" in (tmp_path / "errors.log").read_text(encoding="utf-8")


# ── wiring ────────────────────────────────────────────────────────────────────────────
def test_the_lane_is_wired_above_the_token_gate_and_spends_no_fyers_call():
    root = pathlib.Path(__file__).parent.parent / "eqbtst"
    txt = (root / "dashboard.py").read_text(encoding="utf-8")
    assert '"🏛️ Index Management"' in txt
    m = re.search(r'if tf == "🏛️ Index Management":(.+?)\n\nif tf == "🧮 Arbitrage"', txt, re.S)
    assert m and "index_mgmt.render_page(st)" in m.group(1) and "st.stop()" in m.group(1)
    assert txt.index('if tf == "🏛️ Index Management"') < txt.index("live.token_status()",
                                                                    txt.index('if tf == "Intraday"'))
    code = (root / "index_mgmt.py").read_text(encoding="utf-8")
    for broker in ("FYERS_", "fy_symbol", "_fetch_quotes", "_auth_header", "import live"):
        assert broker not in code, f"index_mgmt reaches for the broker via {broker}"


# ── archive a session behind (26 Oct 2026) ────────────────────────────────────────────
# NSE's 19:30 F & O file came without the November contracts, DCM's sanity gate refused it,
# and the archive stayed on 23 Oct while NSE's live "previous OI" meant 26 Oct. Lots could
# not be proven on 47 of 50 stocks, every rupee chart went blank, and the pressure table
# crashed with KeyError 'bucket'.

_F = {"2026-10-27": [1200.0, 200.0, 50.0, 101.0, 100.0],
      "2026-11-23": [600.0, 100.0, 20.0, 101.5, 100.5]}


def _base(shares_by_exp):
    return dict(fut=sum(shares_by_exp.values()), ce=0.0, pe=0.0, date=dt.date(2026, 10, 26),
                fut_by_exp={dt.date.fromisoformat(k): v for k, v in shares_by_exp.items()})


def test_a_proven_lot_is_remembered_and_reused_when_the_archive_falls_behind(monkeypatch,
                                                                               tmp_path):
    monkeypatch.setattr(im, "_LOTS_FILE", tmp_path / "lots.json")
    monkeypatch.setattr(im, "_LOTS", {"mem": None})
    fresh = _base({"2026-10-27": 1000.0 * 250, "2026-11-23": 500.0 * 250})
    assert im._lot(_F, fresh, "ABC") == 250
    # a stale base (a different session) no longer lands on a whole number ...
    stale = _base({"2026-10-27": 1337.0 * 250, "2026-11-23": 411.0 * 250})
    assert im._lot(_F, stale) is None                         # without a symbol: no guess
    assert im._lot(_F, stale, "ABC") == 250                   # with one: the remembered lot
    # ... and the memory survives a new process (it is on disk, not just in RAM)
    monkeypatch.setattr(im, "_LOTS", {"mem": None})
    assert im._lot(_F, None, "ABC") == 250
    assert im._lot(_F, None, "XYZ") is None                   # never seen: nothing invented


def test_a_remembered_lot_expires(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "_LOTS_FILE", tmp_path / "lots.json")
    old = (dt.date.today() - dt.timedelta(days=im.LOT_CACHE_DAYS + 1)).isoformat()
    monkeypatch.setattr(im, "_LOTS", {"mem": {"ABC": [250, old]}})
    assert im._lot(_F, None, "ABC") is None                   # NSE revises lots; don't trust


def test_the_previous_session_base_is_re_read_not_frozen_for_the_day(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(im.time, "time", lambda: clock[0])
    monkeypatch.setattr(im, "_CTX", {})
    monkeypatch.setattr(im, "_CTX_T", {})
    key = ("oibase", dt.date(2026, 10, 27), ("A",))
    assert not im._ctx_fresh(key)
    im._CTX[key] = {"stale": True}
    assert im._ctx_fresh(key)
    clock[0] += im.ARCHIVE_TTL_S + 1
    assert not im._ctx_fresh(key)                             # the late bhavcopy gets read
    im._CTX[key] = {"fresh": True}
    assert im._ctx_fresh(key)                                 # and is then cached again


def test_pressure_without_any_lot_is_empty_rows_and_the_chart_says_so_instead_of_crashing():
    """No lot for any stock -> pressure_level gives all-empty rows and NO parts. The chart
    guard must see that as 'no data' (it used to pass and crash on parts['bucket'])."""
    tape = [dict(t="2026-10-27T10:00:00", d={"A": [1200, 200, 0, 0, 0, 0]},
                 x={"A": {"F": _F}})]
    lv = im.pressure_level(tape * 3, {"Top 1": ["A"]}, {}, {}, dt.date(2026, 10, 27))
    assert lv["parts"].empty and lv["net"].dropna(how="all").empty
    src = pathlib.Path(im.__file__).read_text(encoding="utf-8")
    body = src[src.index("def _pressure_chart"):src.index("def _call_put_chart")]
    assert 'lv["parts"].groupby' not in body
    assert "parts.empty" in body and 'dropna(how="all")' in body


def test_lot_proof_needs_every_expiry_on_the_same_whole_number():
    # right session: each expiry's ratio is exact
    assert im._proven_lot([(400 * 1000, 1000), (400 * 37, 37), (400 * 5, 5)]) == 400
    # stale session, measured 27 Oct 2026: INFY's median landed on 292.00 (true 400); the
    # other expiries do not agree, so nothing is proven
    assert im._proven_lot([(292.0 * 1000, 1000), (310.37 * 10, 10)]) is None
    # one expiry alone is never proof by itself ...
    assert im._proven_lot([(292.0 * 1000, 1000)]) is None
    # ... unless it lands exactly on the lot proven on an earlier day
    assert im._proven_lot([(400.0 * 1000, 1000)], known=400) == 400
    assert im._proven_lot([(292.0 * 1000, 1000)], known=400) is None
    # an NSE revision, live since the Jan-2027 series: new contracts carry the new lot, every
    # ratio is still whole, and the stock's lot is its most active contract's (HDFCBANK, 29 Oct)
    assert im._proven_lot([(650 * 512872, 512872), (650 * 78839, 78839), (750 * 138, 138)]) == 650
    # ... CHOLAFIN's new lot is SMALLER: the rule must not assume a direction
    assert im._proven_lot([(625 * 19274, 19274), (625 * 248, 248), (300 * 4, 4)]) == 625
    # two expiries that disagree, nothing backing either: could be a stale-day coincidence ...
    assert im._proven_lot([(400 * 90, 90), (300 * 10, 10)]) is None
    # ... unless the near contract's lot is the one already remembered
    assert im._proven_lot([(400 * 90, 90), (300 * 10, 10)], known=400) == 400
    assert im._proven_lot([(400 * 90, 90), (300 * 10, 10)], known=300) is None
    # missing archive rows / untraded contracts are skipped, not divided by zero
    assert im._proven_lot([(None, 10), (400 * 10, 0), (400 * 9, 9), (400 * 3, 3)]) == 400


def test_a_stale_base_can_no_longer_overwrite_a_remembered_lot(monkeypatch):
    fresh = _base({"2026-10-27": 1000.0 * 250, "2026-11-23": 500.0 * 250})
    assert im._lot(_F, fresh, "ABC") == 250
    # stale base whose median happens to land on a whole number (old rule accepted 249)
    stale = _base({"2026-10-27": 1000.0 * 249, "2026-11-23": 500.0 * 251.7})
    assert im._lot(_F, stale, "ABC") == 250                   # memory, not the coincidence
    assert im._lot_cache()["ABC"][0] == 250


def test_oi_percent_never_uses_a_remembered_lot_against_a_wrong_session_base(monkeypatch):
    monkeypatch.setattr(im, "_LOTS", {"mem": {"ABC": [250, dt.date.today().isoformat()]}})
    d = dict(fut_oi=1200.0, fut_chg=200.0, ce_oi=0.0, ce_chg=0.0, pe_oi=0.0, pe_chg=0.0,
             fut_prev={dt.date(2026, 10, 27): 1000.0, dt.date(2026, 11, 23): 500.0})
    stale = dict(fut=1.0, ce=0.0, pe=0.0,
                 fut_by_exp={dt.date(2026, 10, 27): 1337.0 * 250, dt.date(2026, 11, 23): 411.0 * 250})
    *_, used_archive = im.oi_changes(d, stale, "ABC")
    assert used_archive is False                              # feed's own base, not 1337/411


def test_a_locked_archive_on_the_ten_minute_re_read_keeps_the_last_good_base(monkeypatch):
    key = ("oibase", dt.date(2026, 10, 27), ("ABC",))
    monkeypatch.setattr(im, "_CTX", {key: {"ABC": "good"}})
    monkeypatch.setattr(im, "_CTX_T", {key: 0.0})             # long expired

    class Boom:
        def __enter__(self):
            raise RuntimeError("IO Error: Could not set lock on file")

        def __exit__(self, *a):
            return False
    from eqbtst import data
    monkeypatch.setattr(data, "_connect", lambda: Boom())
    assert im._archive_oi_base(("ABC",), dt.date(2026, 10, 27)) == {"ABC": "good"}


# ── 28 Oct 2026: pressure hysteresis, cycle-adjusted calls - puts, nightly OI score ─────
def _hover_tape(prices):
    """One stock, +100 contracts vs yesterday, futures price walking through `prices`."""
    F = lambda p: {"2026-11-24": [1100.0, 100.0, 5.0, p, 100.0],
                   "2026-12-29": [400.0, 0.0, 0.0, p, 100.0]}
    return [dict(t=f"2026-10-28T10:{i:02d}:00", d={"A": [1100.0, 100.0, 0, 0, 0, 0]},
                 x={"A": {"F": F(p), "S": p}}) for i, p in enumerate(prices)]


_HB = {"A": dict(fut=375_000.0, ce=0.0, pe=0.0,
                 fut_by_exp={dt.date(2026, 11, 24): 250_000.0, dt.date(2026, 12, 29): 100_000.0})}


def test_a_stock_hovering_on_yesterdays_close_no_longer_flips_the_line():
    # INFY on 26 Oct: +0.11% / -0.11% / +0.02% / -0.08% around its close, every minute
    tape = _hover_tape([100.2, 99.95, 100.02, 99.92, 100.05])
    flat = im.pressure_level(tape, {"Top": ["A"]}, _HB, {}, dt.date(2026, 10, 28))
    held = im.pressure_level(tape, {"Top": ["A"]}, _HB, {}, dt.date(2026, 10, 28),
                             band=im.PRESSURE_BAND)
    f, h = flat["net"]["Top"].tolist(), held["net"]["Top"].tolist()
    assert sum(1 for a, b in zip(f, f[1:]) if (a > 0) != (b > 0)) == 4   # flipped every read
    assert all(v > 0 for v in h)                                         # side held
    # a real move through the band still changes side
    moved = im.pressure_level(_hover_tape([100.2, 99.85]), {"Top": ["A"]}, _HB, {},
                              dt.date(2026, 10, 28), band=im.PRESSURE_BAND)
    assert moved["net"]["Top"].iloc[-1] < 0


def test_inside_the_band_from_the_start_a_stock_has_no_side_yet():
    lv = im.pressure_level(_hover_tape([100.05, 99.95]), {"Top": ["A"]}, _HB, {},
                           dt.date(2026, 10, 28), band=im.PRESSURE_BAND)
    assert (lv["parts"]["unsided"] > 0).all()


def test_the_usual_cycle_drift_is_rate_times_exposure_pro_rated_through_the_day():
    base = {"A": dict(fut=375_000.0, ce=2_000_000.0, pe=1_000_000.0,
                      fut_by_exp={dt.date(2026, 11, 24): 250_000.0,
                                  dt.date(2026, 12, 29): 100_000.0})}      # lot 250
    F = {"2026-11-24": [1000.0, 0.0, 5.0, 200.0, 200.0], "2026-12-29": [400.0, 0.0, 0.0, 200.0, 200.0]}
    tape = [dict(t="2026-10-28T15:25:00", d={"A": [0, 0, 0, 0, 0, 0]},
                 x={"A": {"F": F, "S": 200.0}})]
    u = im.call_put_usual(tape, {"Top": ["A"]}, base, dt.date(2026, 10, 28), 0.08)
    # full day: 8% x (2.0M + 1.0M) / 2 shares x Rs 200 = Rs 2.4 Cr
    assert u["Top"].iloc[-1] == pytest.approx(2.4)
    early = [dict(tape[0], t="2026-10-28T09:10:00")]          # pre-open: nothing traded yet
    assert im.call_put_usual(early, {"Top": ["A"]}, base, dt.date(2026, 10, 28),
                             0.08)["Top"].iloc[-1] == pytest.approx(0.0)
    assert im.call_put_usual(tape, {"Top": ["A"]}, base, dt.date(2026, 10, 28), None).empty


def test_a_failed_cycle_rate_read_is_retried_later_not_cached_for_the_day(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    monkeypatch.setattr(im, "_CTX", {})
    clock = [1000.0]
    monkeypatch.setattr(im.time, "time", lambda: clock[0])
    calls = []
    from eqbtst import data

    def boom():
        calls.append(1)
        raise RuntimeError("locked")
    monkeypatch.setattr(data, "_connect", boom)
    day = dt.date(2026, 10, 28)
    assert im.cp_cycle_rate(("A",), day) is None
    assert im.cp_cycle_rate(("A",), day) is None and len(calls) == 1   # no hammering
    clock[0] += im.ARCHIVE_TTL_S + 1
    assert im.cp_cycle_rate(("A",), day) is None and len(calls) == 2   # retried


def test_the_cycle_rate_cache_name_is_stable_across_processes():
    src = pathlib.Path(im.__file__).read_text(encoding="utf-8")
    body = src[src.index("def cp_cycle_rate"):src.index("def call_put_usual")]
    assert "hash(symbols)" not in body and "hashlib" in body


def test_reconcile_summary_counts_only_full_days(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    rows = [dict(day="2026-10-15", code="NIFTY", full_day=True, fut=dict(sign_agree=0.42),
                 ce=dict(sign_agree=0.96), pe=dict(sign_agree=0.98)),
            dict(day="2026-10-19", code="NIFTY", full_day=True, fut=dict(sign_agree=0.54),
                 ce=dict(sign_agree=1.0), pe=dict(sign_agree=0.98)),
            dict(day="2026-10-23", code="NIFTY", full_day=False, fut=dict(sign_agree=0.10),
                 ce=dict(sign_agree=0.10), pe=dict(sign_agree=0.10))]
    (tmp_path / im.RECONCILE_FILE).write_text("\n".join(json.dumps(r) for r in rows))
    s = im.reconcile_summary("NIFTY")
    assert s["days"] == 2 and s["fut"] == pytest.approx(0.48) and s["ce"] == pytest.approx(0.98)
    assert im.reconcile_summary("BANKNIFTY") is None



def test_the_usual_drift_counts_only_stocks_the_line_counts(monkeypatch):
    """A stock with no proven lot is not on the calls-puts line, so it must not be in the
    'usual' subtracted from it (up to 11 of 25 Midcap names at 09:17 on 16 Oct)."""
    monkeypatch.setattr(im, "SERIES_QUORUM", 0.5)            # 1 of 2 names may draw
    base = {"A": dict(fut=375_000.0, ce=2_000_000.0, pe=1_000_000.0,
                      fut_by_exp={dt.date(2026, 11, 24): 250_000.0,
                                  dt.date(2026, 12, 29): 100_000.0}),
            "B": dict(fut=1.0, ce=9_000_000.0, pe=9_000_000.0, fut_by_exp={})}
    F = {"2026-11-24": [1000.0, 0.0, 5.0, 200.0, 200.0], "2026-12-29": [400.0, 0.0, 0.0, 200.0, 200.0]}
    tape = [dict(t="2026-10-28T15:25:00", d={"A": [0] * 6, "B": [0] * 6},
                 x={"A": {"F": F, "S": 200.0}, "B": {"F": {}, "S": 200.0}})]
    u = im.call_put_usual(tape, {"Top": ["A", "B"]}, base, dt.date(2026, 10, 28), 0.08)
    assert u["Top"].iloc[-1] == pytest.approx(2.4)            # A only; B would add 144 Cr


def test_a_log_without_per_expiry_detail_is_marked_once_not_re_read(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    day = dt.date(2026, 10, 13)
    (tmp_path / f"oi_NIFTY_{day:%Y%m%d}.jsonl").write_text(
        json.dumps(dict(t="2026-10-13T15:26:00", d={"A": [1, 0, 1, 0, 1, 0]})) + "\n")

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=None):
            if "max(trade_date)" in sql:
                return type("R", (), {"fetchone": lambda s: (dt.date(2026, 10, 12),)})()
            if "count(*)" in sql:
                return type("R", (), {"fetchone": lambda s: (1,)})()
            return type("R", (), {"df": lambda s: pd.DataFrame(
                columns=["symbol", "trade_date", "k", "expiry_date", "oi"])})()
    from eqbtst import data
    monkeypatch.setattr(data, "_connect", lambda: Conn())
    monkeypatch.setattr(im, "NSE_NAME", {"NIFTY": "NIFTY 50"})
    assert im.reconcile_oi(days_back=1, today=dt.date(2026, 10, 14)) == 1
    assert im.reconcile_oi(days_back=1, today=dt.date(2026, 10, 14)) == 0   # not again
    assert im.reconcile_summary("NIFTY") is None                             # never scored


# ── past sessions: rebuilt from the log, with its coverage stated (Nov 2026) ─────────────
_D29 = dt.date(2026, 10, 29)


def _rows(*hhmm):
    return [dict(t=f"2026-10-29T{x}:00") for x in hhmm]


def _every(start, end, step=2):
    t, out = start, []
    while t <= end:
        out.append(f"{t // 60:02d}:{t % 60:02d}")
        t += step
    return _rows(*out)


FULL = _every(9 * 60 + 16, 15 * 60 + 29)


def test_a_day_is_complete_only_when_both_logs_cover_it():
    assert im.session_coverage(FULL, FULL, _D29)["grade"] == "complete"
    # the prices alone are not a full day
    assert im.session_coverage(FULL, [], _D29)["grade"] == "partial"
    assert im.session_coverage([], [], _D29)["grade"] == "none"
    assert im.session_coverage(_rows("15:24", "15:26"), _rows("15:25"), _D29)["grade"] == "thin"


def test_coverage_names_the_hours_that_are_missing():
    morning = _every(9 * 60 + 16, 11 * 60 + 26)
    cov = im.session_coverage(morning, morning, _D29)
    assert cov["grade"] == "partial" and 34 < cov["prices"]["pct"] < 36     # 130 of 375 min
    assert "not logged: 11:26–15:30" in im.coverage_text(cov)
    late = _every(11 * 60 + 9, 15 * 60 + 29)
    assert "09:15–11:09" in im.coverage_text(im.session_coverage(late, late, _D29))
    # a hole in the middle, in only one log, is still a hole
    gap = _every(9 * 60 + 16, 11 * 60) + _every(14 * 60, 15 * 60 + 29)
    assert "11:00–14:00" in im.coverage_text(im.session_coverage(FULL, gap, _D29))


def test_todays_coverage_is_measured_against_the_session_so_far():
    so_far = _every(9 * 60 + 16, 10 * 60)
    now = dt.datetime(2026, 10, 29, 10, 1)
    assert im.session_coverage(so_far, so_far, _D29, now)["grade"] == "complete"
    assert im.session_coverage(so_far, so_far, _D29)["grade"] == "partial"
    # before the open there is no session to cover yet: no division by zero
    early = dt.datetime(2026, 10, 29, 9, 0)
    assert im.session_coverage([], [], _D29, early)["grade"] == "none"


def _write_day(tmp_path, code="NIFTY", day=_D29, **meta_over):
    """A two-stock session on disk, as the poller writes it. A: +1%, B: -2%."""
    tape = [dict(t=f"{day}T15:29:00", ns=f"{day}T15:29:00", last=20040.0, prev=20000.0,
                 hi=20050.0, lo=19990.0, p={"A": 1.0, "B": -2.0}, v={"A": 500.0, "B": 250.0})]
    F = {"2026-11-23": [1100.0, 100.0, 5.0, 101.0, 100.0], "2026-12-29": [400.0, 0.0, 1.0, 101.0, 100.0]}
    oi = [dict(t=f"{day}T15:29:10", d={s: [1100.0, 100.0, 500.0, 50.0, 300.0, 30.0] for s in "AB"},
               x={s: {"F": F, "S": 101.0} for s in "AB"},
               ns={s: f"{day}T15:29:05" for s in "AB"})]
    for kind, rows in (("tape", tape), ("oi", oi)):
        (tmp_path / f"{kind}_{code}_{day:%Y%m%d}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (tmp_path / f"weights_{code}_{day:%Y%m%d}.json").write_text(
        json.dumps({"day": day.isoformat(), "w": {"A": 60.0, "B": 40.0}}), encoding="utf-8")
    meta = dict(day=day.isoformat(), prev={"A": 100.0, "B": 50.0}, ban=[], corp={},
                weights_day=day.isoformat(), weights={"A": 60.0, "B": 40.0})
    meta.update(meta_over)
    (tmp_path / f"meta_{code}_{day:%Y%m%d}.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture
def _offline(monkeypatch, tmp_path):
    """Log files in tmp_path; no archive, no DCM."""
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    monkeypatch.setattr(im, "_REPLAY_CACHE", {})
    monkeypatch.setattr(im, "_COV_CACHE", {})
    monkeypatch.setattr(im, "_dcm", lambda: None)
    monkeypatch.setattr(im, "_archive_oi_base", lambda s, d: {})
    monkeypatch.setattr(im, "_archive_corp", lambda s, d: {})
    monkeypatch.setattr(im, "_archive_norms", lambda s, before=None: {})
    return tmp_path


def test_a_past_session_goes_through_the_same_arithmetic_as_the_live_page(_offline):
    _write_day(_offline)
    snap = im.session_snapshot("NIFTY", _D29)
    cash = dict(code="NIFTY", ts=snap["as_of"], status="Open", fetched=snap["as_of"],
                index=dict(last=20040.0, prev=20000.0, open=None, high=20050.0, low=19990.0),
                stocks={"A": dict(ltp=101.0, prev=100.0, open=None, pct=1.0, value=5e9, upd=""),
                        "B": dict(ltp=49.0, prev=50.0, open=None, pct=-2.0, value=2.5e9, upd="")})
    live = im.decompose(cash, {"A": 60.0, "B": 40.0}, {}, "NIFTY", now=snap["as_of"])
    assert snap["index"] == live["index"] and snap["explained_pts"] == live["explained_pts"]
    assert list(snap["stocks"]["pts"]) == list(live["stocks"]["pts"])
    assert snap["stocks"].set_index("symbol")["value_cr"]["A"] == pytest.approx(500.0)
    # the OI row came through: 100 contracts on a feed base of 1,000 = +10%
    assert snap["stocks"].set_index("symbol")["fut_oi_pct"]["A"] == pytest.approx(10.0)
    assert snap["front_expiry"] == dt.date(2026, 11, 23)
    assert snap["replay"] and not snap["live"] and snap["as_of"].hour == 15


def test_a_partial_day_is_judged_at_its_last_reading_not_against_a_full_day(_offline):
    _write_day(_offline)
    p = _offline / f"tape_NIFTY_{_D29:%Y%m%d}.jsonl"
    p.write_text(p.read_text(encoding="utf-8").replace("T15:29:00", "T11:26:00"), encoding="utf-8")
    snap = im.session_snapshot("NIFTY", _D29)
    assert snap["session_frac"] == im.pace_fraction(dt.datetime(2026, 10, 29, 11, 26)) < 1.0


def test_a_past_session_never_uses_weights_from_after_it(_offline):
    _write_day(_offline)
    (_offline / f"weights_NIFTY_{_D29:%Y%m%d}.json").unlink()
    later = _D29 + dt.timedelta(days=1)
    (_offline / f"weights_NIFTY_{later:%Y%m%d}.json").write_text(
        json.dumps({"day": later.isoformat(), "w": {"A": 99.0, "B": 1.0}}), encoding="utf-8")
    meta = dict(weights={"A": 99.0, "B": 1.0}, weights_day=later.isoformat())
    assert im._weights_for("NIFTY", _D29, meta) == (None, None)
    earlier = _D29 - dt.timedelta(days=2)
    (_offline / f"weights_NIFTY_{earlier:%Y%m%d}.json").write_text(
        json.dumps({"day": earlier.isoformat(), "w": {"A": 55.0, "B": 45.0}}), encoding="utf-8")
    assert im._weights_for("NIFTY", _D29, meta) == ({"A": 55.0, "B": 45.0}, earlier)


def test_a_meta_rewritten_after_its_session_is_flagged_as_having_lost_its_ban_list(_offline):
    _write_day(_offline, weights_day=(_D29 + dt.timedelta(days=1)).isoformat())
    assert im.session_snapshot("NIFTY", _D29)["ban_unknown"] is True
    _write_day(_offline, ban=["A"])
    im._REPLAY_CACHE.clear()
    snap = im.session_snapshot("NIFTY", _D29)
    assert snap["ban_unknown"] is False and "A" in snap["withheld"]


def test_a_finished_days_meta_is_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    monkeypatch.setattr(im, "_dcm", lambda: None)
    monkeypatch.setattr(im, "fetch_ban_list", lambda d: set())
    monkeypatch.setattr(im, "_archive_oi_base", lambda s, d: {})
    monkeypatch.setattr(im, "_archive_corp", lambda s, d: {})
    cash = _cash({"A": 1.0}, ts=dt.datetime(2026, 10, 29, 15, 30))
    s = im._st("NIFTY")
    monkeypatch.setitem(s, "cash", cash)
    monkeypatch.setitem(s, "weights", {"A": 100.0})
    monkeypatch.setitem(s, "weights_day", _D29)
    f = tmp_path / f"meta_NIFTY_{_D29:%Y%m%d}.json"
    monkeypatch.setattr(im, "_now", lambda: dt.datetime(2026, 10, 29, 15, 35))
    im._save_meta("NIFTY")
    first = f.read_text(encoding="utf-8")
    # the next morning NSE still stamps the feed with 29 Oct; new weights must not overwrite
    monkeypatch.setitem(s, "weights", {"A": 99.0})
    monkeypatch.setitem(s, "weights_day", _D29 + dt.timedelta(days=1))
    monkeypatch.setattr(im, "_now", lambda: dt.datetime(2026, 10, 30, 9, 1))
    im._save_meta("NIFTY")
    assert f.read_text(encoding="utf-8") == first
    # but on the day itself it still follows changes
    monkeypatch.setattr(im, "_now", lambda: dt.datetime(2026, 10, 29, 15, 40))
    im._save_meta("NIFTY")
    assert f.read_text(encoding="utf-8") != first


def test_viewing_a_past_session_leaves_the_lot_memory_alone(monkeypatch):
    fresh = _base({"2026-10-27": 1000.0 * 250, "2026-11-23": 500.0 * 250})
    with im._replay():
        assert im._lot(_F, fresh, "ABC") == 250               # proven from that day's archive
    assert "ABC" not in im._lot_cache()                       # but nothing was learned
    im._lot(_F, fresh, "ABC")
    assert im._lot(_F, None, "ABC") == 250                    # the live page still remembers
    with im._replay():
        assert im._lot(_F, None, "ABC") is None               # and a past day cannot borrow it


def test_the_activity_yardstick_of_a_past_day_stops_before_it(monkeypatch):
    seen = {}

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=None):
            seen.update(sql=sql, params=params)
            return type("R", (), {"df": lambda s: pd.DataFrame({"symbol": ["A"], "cr": [10.0]})})()
    from eqbtst import data
    monkeypatch.setattr(data, "_connect", lambda: Conn())
    monkeypatch.setattr(im, "_CTX", {})
    assert im._archive_norms(("A",), before=_D29) == {"A": 10.0}
    assert "trade_date < ?" in seen["sql"] and seen["params"] == ["A", _D29]
    im._archive_norms(("A",))
    assert "trade_date < ?" not in seen["sql"]


def test_a_traded_day_with_no_log_is_listed_as_missing_not_skipped(_offline, monkeypatch):
    _write_day(_offline)
    today = _D29 + dt.timedelta(days=4)
    _write_day(_offline, day=today)                           # today's log: not a past session
    missing = _D29 + dt.timedelta(days=1)

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=None):
            return type("R", (), {"fetchall": lambda s: [(_D29,), (missing,)]})()
    from eqbtst import data
    monkeypatch.setattr(data, "_connect", lambda: Conn())
    monkeypatch.setattr(im, "_CTX", {})
    assert im.list_sessions("NIFTY", today=today) == [missing, _D29]
    assert im._day_coverage("NIFTY", missing)["grade"] == "none"
    assert im.session_snapshot("NIFTY", missing) is None


class _PickerSt:
    def __init__(self, url=None):
        self.session_state, self.query_params = {}, dict(url or {})
        self.column_config = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()

    def selectbox(self, label, opts, format_func=None, key=None, help=None):
        self.labels = [format_func(o) for o in opts]
        return self.session_state[key]

    def expander(self, *a, **k):
        return contextlib.nullcontext()

    def dataframe(self, *a, **k):
        pass

    def caption(self, *a, **k):
        pass


def test_the_url_opens_a_past_session_and_a_bad_one_falls_back_to_live(_offline, monkeypatch):
    _write_day(_offline)
    monkeypatch.setattr(im, "list_sessions", lambda c: [_D29])
    st = _PickerSt({"session": "2026-10-29"})
    assert im._session_picker(st, "NIFTY") == _D29
    assert st.query_params["session"] == "2026-10-29" and "live" not in st.query_params
    assert "complete" in st.labels[1] or "partial" in st.labels[1] or "very little" in st.labels[1]
    for bad in ("2019-01-01", "garbage", "", "2026-13-45"):
        st = _PickerSt({"session": bad})
        assert im._session_picker(st, "NIFTY") is None
        assert "session" not in st.query_params               # the URL says what is on screen
    # switching index to one that lacks the chosen day must not raise
    st = _PickerSt()
    st.session_state["im_session"] = "2026-10-01"
    assert im._session_picker(st, "NIFTY") is None


def test_the_past_view_draws_no_live_clock_and_never_polls():
    src = pathlib.Path(im.__file__).read_text(encoding="utf-8")
    body = src[src.index("def session_snapshot"):src.index("def official_close_read")]
    for live_only in ("fetch_", "poll_", "_client(", "_append(", "_write_json(", "touch("):
        assert live_only not in body, f"{live_only} has no place in a recording"


# ── options: buyers vs writers judged on implied volatility (3 Nov 2026) ────────────────
_NOW = dt.datetime(2026, 11, 3, 11, 0)
_PREV = dt.datetime(2026, 11, 2, 15, 30)
_EXP = "2026-11-23"


def _premium(F, K, when, iv, call=True):
    return float(im.b76([F], [K], [im._trading_years(when, dt.date.fromisoformat(_EXP))], [iv], [call])[0][0])


def test_black76_round_trips_and_refuses_impossible_prices():
    T = im._years(_NOW, dt.date(2026, 11, 23))
    for call in (True, False):
        p = im.b76([1000.0], [1020.0], [T], [0.25], [call])[0]
        assert im.implied_vol(p, [1000.0], [1020.0], [T], [call])[0] == pytest.approx(0.25, abs=1e-4)
    # a call worth less than its intrinsic value has no volatility: NaN, never a guess
    assert np.isnan(im.implied_vol([5.0], [1100.0], [1000.0], [T], [True])[0])
    assert np.isnan(im.implied_vol([10.0], [1000.0], [1000.0], [0.0], [True])[0])


def _flow(rows, F1=1000.0, F0=1000.0, mode="iv", lot=250):
    """rows: (side, strike, last, prev, OI change); one expiry, its future F1 now / F0 then."""
    opts = [(s, k, _EXP, last, prev, doi) for s, k, last, prev, doi in rows]
    return im.option_flow(opts, {_EXP: (F1, F0)}, {_EXP: lot}, _NOW, _PREV, mode)


def _box(out, leg):
    return {q: v[0] for q, v in out[leg].items() if v[0]}


def test_time_decay_alone_is_not_writing():
    """Premium fell only because a day passed: the premium read calls the new calls
    'writing'; the volatility read calls it unclear — nothing was pressed."""
    prev = _premium(1000, 1020, _PREV, 0.25)
    now = _premium(1000, 1020, _NOW, 0.25)
    assert now < prev
    rows = [("CE", 1020, round(now, 2), round(prev, 2), 500)]
    assert _box(_flow(rows, mode="premium"), "CE") == {"writing": 500}
    assert _box(_flow(rows), "CE") == {"unread": 500}


def test_the_stocks_own_move_alone_is_not_buying():
    """Stock +3%: the call premium jumps by delta alone. Premium read: 'long build'. Volatility
    read: unclear — the 99% price echo measured on the archive is exactly this."""
    prev = _premium(1000, 1020, _PREV, 0.25)
    now = _premium(1030, 1020, _NOW, 0.25)
    rows = [("CE", 1020, round(now, 2), round(prev, 2), 400)]
    assert _box(_flow(rows, F1=1030, mode="premium"), "CE") == {"long build": 400}
    assert _box(_flow(rows, F1=1030), "CE") == {"unread": 400}


def test_the_four_boxes_follow_open_interest_and_volatility():
    prev = round(_premium(1000, 1000, _PREV, 0.25), 2)
    up = round(_premium(1000, 1000, _NOW, 0.29), 2)        # bid: volatility +4 points
    dn = round(_premium(1000, 1000, _NOW, 0.21), 2)        # offered
    for side in ("CE", "PE"):
        assert _box(_flow([(side, 1000, up, prev, 100)]), side) == {"long build": 100}
        assert _box(_flow([(side, 1000, dn, prev, 100)]), side) == {"writing": 100}
        assert _box(_flow([(side, 1000, up, prev, -100)]), side) == {"short covering": -100}
        assert _box(_flow([(side, 1000, dn, prev, -100)]), side) == {"long unwind": -100}


def test_rupees_are_delta_weighted_exposure():
    prev = round(_premium(1000, 1000, _PREV, 0.25), 2)
    up = round(_premium(1000, 1000, _NOW, 0.29), 2)
    out = _flow([("CE", 1000, up, prev, 100)], lot=250)
    _, delta = im.b76([1000.0], [1000.0], [im._trading_years(_NOW, dt.date(2026, 11, 23))], [0.29], [True])
    # 100 contracts x 250 x Rs 1,000 x ~0.5 = ~Rs 1.25 Cr, not the Rs 2.5 Cr notional
    assert out["CE"]["long build"][1] == pytest.approx(100 * 250 * 1000 * delta[0] / 1e7, rel=1e-3)
    assert 0.4 < delta[0] < 0.6


def test_a_stale_print_is_caught_by_put_call_parity():
    """INFY 1154 on 3 Nov 2026: yesterday's call close implied 19.7%, the put at the same
    strike 43.5% — one of them is an hours-old last trade. Neither is used."""
    c_prev, p_prev = _premium(1160, 1154, _PREV, 0.197), _premium(1160, 1154, _PREV, 0.435, False)
    c_now, p_now = _premium(1150, 1154, _NOW, 0.31), _premium(1150, 1154, _NOW, 0.31, False)
    rows = [("CE", 1154, round(c_now, 2), round(c_prev, 2), 2000),
            ("PE", 1154, round(p_now, 2), round(p_prev, 2), 300)]
    out = _flow(rows, F1=1150, F0=1160)
    assert _box(out, "CE") == {"unread": 2000} and _box(out, "PE") == {"unread": 300}


def test_a_tiny_premium_is_never_judged():
    assert _box(_flow([("CE", 1300, 0.45, 0.30, 900)]), "CE") == {"unread": 900}


def test_option_lots_follow_the_contract_month():
    F = {"2026-11-23": [521273.0, 8401.0, 0, 719.0, 702.0], "2026-12-29": [80415.0, 1576.0, 0, 722.0, 705.0],
         "2027-01-25": [507.0, 369.0, 0, 726.0, 709.0]}
    base = dict(fut_by_exp={dt.date(2026, 11, 23): 650.0 * 512872, dt.date(2026, 12, 29): 650.0 * 78839,
                            dt.date(2027, 1, 25): 750.0 * 138})
    assert im._lots_by_exp(F, base, "HDFCBANK") == {"2026-11-23": 650, "2026-12-29": 650,
                                                    "2027-01-25": 750}


def _q(**cr):
    """A logged per-stock read with rupees in the given boxes (contracts = 1 each)."""
    out = {}
    for leg in ("CE", "PE"):
        out[leg] = [[1 if cr.get(f"{leg}_{q.replace(' ', '_')}") else 0,
                     cr.get(f"{leg}_{q.replace(' ', '_')}", 0.0)] for q in (*im.OPT_QUADS, "unread")]
    return {"iv": out, "premium": out}


def test_bucket_sums_net_sign_and_a_skipped_stock_keeping_its_read(monkeypatch):
    monkeypatch.setattr(im, "SERIES_QUORUM", 0.5)
    tape = [dict(t="2026-11-03T10:00:00", q={"A": _q(CE_long_build=10.0), "B": _q(PE_writing=5.0)}),
            dict(t="2026-11-03T10:02:00", q={"A": _q(CE_writing=7.0)}),            # B skipped
            dict(t="2026-11-03T10:04:00", q={"B": _q(PE_long_build=3.0)})]
    op = im.option_positioning(tape, {"Top": ["A", "B"]})
    # 10:00 call buying 10 + put writing 5 = +15; 10:02 A now writes calls: -7 + 5 = -2;
    # 10:04 B now buys puts: -7 - 3 = -10
    assert op["net"]["Top"].tolist() == [15.0, -2.0, -10.0]
    assert op["last"]["Top"]["CE"]["writing"] == [1, 7.0]
    assert op["last"]["Top"]["PE"]["long build"] == [1, 3.0]
    assert op["n"] == 2


def test_readings_before_the_option_read_existed_draw_nothing():
    op = im.option_positioning([dict(t="2026-10-29T10:00:00", d={"A": [0] * 6})], {"Top": ["A"]})
    assert op["n"] == 0 and op["net"].empty


def test_the_shared_deriv_file_does_not_carry_every_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    monkeypatch.setattr(im, "_D", {"A": dict(day=dt.date(2026, 11, 3), ts=None, expiries=[],
                                             fut_prev={}, opts=[("CE", 1.0, _EXP, 1, 1, 1)] * 90,
                                             q={"iv": {}})})
    im._save_deriv()
    j = json.loads((tmp_path / "deriv_latest.json").read_text(encoding="utf-8"))
    assert "opts" not in j["A"] and "q" in j["A"]


def test_a_sweep_logs_each_stocks_option_read_and_the_contracts_every_ten_minutes(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    monkeypatch.setattr(im, "_D", {})
    monkeypatch.setattr(im, "_S", {})
    when = dt.datetime(2026, 11, 3, 11, 0)
    monkeypatch.setattr(im, "_now", lambda: when)
    prev = round(_premium(1000, 1000, _PREV, 0.25), 2)
    up = round(_premium(1000, 1000, _NOW, 0.29), 2)

    def fake(sym, client):
        return dict(fut_oi=1100.0, fut_chg=100.0, ce_oi=600.0, ce_chg=100.0, pe_oi=0.0, pe_chg=0.0,
                    expiries=[dt.date(2026, 11, 23), dt.date(2026, 12, 29)],
                    fut_prev={dt.date(2026, 11, 23): 1000.0, dt.date(2026, 12, 29): 400.0},
                    by_exp={"F": {"2026-11-23": [1100.0, 100.0, 5.0, 1000.0, 1000.0],
                                  "2026-12-29": [400.0, 0.0, 1.0, 1003.0, 1003.0]}, "S": 1000.0},
                    opts=[("CE", 1000.0, "2026-11-23", up, prev, 100.0)], ts=when)
    monkeypatch.setattr(im, "fetch_deriv", fake)
    monkeypatch.setattr(im, "_archive_oi_base", lambda s, d: {
        "A": dict(fut=350_000.0, ce=0.0, pe=0.0, date=dt.date(2026, 11, 2),
                  fut_by_exp={dt.date(2026, 11, 23): 250_000.0, dt.date(2026, 12, 29): 100_000.0})})
    s = im._st("NIFTY")
    s["cash"] = _cash({"A": 1.0}, ts=when)
    assert im.poll_deriv("NIFTY", object()) == 1
    e = im._load_jsonl(im._path("oi", "NIFTY", when.date()))[-1]
    assert e["q"]["A"]["iv"]["CE"][0][0] == 100                # long build, 100 contracts
    assert e["o"]["A"] == [["C", 1000.0, "2026-11-23", up, prev, 100.0, None]]
    s["deriv_busy"] = False
    s["deriv_at"] = 0.0
    monkeypatch.setattr(im, "_D", {})
    assert im.poll_deriv("NIFTY", object()) == 1
    e2 = im._load_jsonl(im._path("oi", "NIFTY", when.date()))[-1]
    assert "q" in e2 and "o" not in e2                        # contracts only every 10 minutes


def test_a_failing_option_read_never_costs_the_stock_its_oi_row(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "DATA_DIR", tmp_path)
    monkeypatch.setattr(im, "_D", {})
    monkeypatch.setattr(im, "_S", {})
    when = dt.datetime(2026, 11, 3, 11, 0)
    monkeypatch.setattr(im, "_now", lambda: when)
    monkeypatch.setattr(im, "fetch_deriv", lambda sym, c: dict(
        fut_oi=1.0, fut_chg=0.0, ce_oi=0.0, ce_chg=0.0, pe_oi=0.0, pe_chg=0.0, expiries=[],
        fut_prev={}, by_exp={"F": {}}, opts=[], ts=when))
    monkeypatch.setattr(im, "_archive_oi_base", lambda s, d: {"A": dict(date=dt.date(2026, 11, 2))})

    def boom(*a, **k):
        raise ValueError("bad contract")
    monkeypatch.setattr(im, "stock_option_flow", boom)
    im._st("NIFTY")["cash"] = _cash({"A": 1.0}, ts=when)
    assert im.poll_deriv("NIFTY", object()) == 1
    assert im._load_jsonl(im._path("oi", "NIFTY", when.date()))[-1]["d"]["A"]


def test_option_time_runs_on_the_markets_clock_not_the_calendar(monkeypatch):
    """Calendar time charged the weekend to Monday: 68% of Monday build-ups read as buying
    against 40% on other days (archive). Nights, weekends and holidays now count as nothing."""
    monkeypatch.setattr(im, "_sessions_left", lambda d, e: None)        # weekday calendar
    monkeypatch.setattr(im, "_SESS_CACHE", {})
    exp = dt.date(2026, 11, 23)
    fri_close = im._trading_years(dt.datetime(2026, 11, 6, 15, 30), exp)
    mon_open = im._trading_years(dt.datetime(2026, 11, 9, 9, 15), exp)
    assert mon_open == pytest.approx(fri_close)            # the weekend costs nothing
    mon_mid = im._trading_years(dt.datetime(2026, 11, 9, 12, 22, 30), exp)
    assert fri_close - mon_mid == pytest.approx(0.5 / 252)  # half a session gone
    assert im._trading_years(dt.datetime(2026, 11, 23, 15, 30), exp) == 0.0
    assert im._trading_years(dt.datetime(2026, 11, 24, 10, 0), exp) == 0.0


def test_yesterday_is_the_previous_session_not_the_archives_last_date(monkeypatch):
    monkeypatch.setattr(im, "_SESS_CACHE", {})
    holidays = {dt.date(2026, 11, 24)}

    def sessions(d, e):
        return sum(1 for k in range(1, (e - d).days + 1)
                   if (d + dt.timedelta(days=k)).weekday() < 5
                   and d + dt.timedelta(days=k) not in holidays)
    monkeypatch.setattr(im, "_sessions_left", sessions)
    assert im._prev_session(dt.date(2026, 11, 9)) == dt.date(2026, 11, 6)    # Monday -> Friday
    assert im._prev_session(dt.date(2026, 11, 25)) == dt.date(2026, 11, 23)  # over the holiday
    assert im._prev_session(dt.date(2026, 11, 26)) == dt.date(2026, 11, 25)


def test_the_recorder_steps_aside_when_its_code_changes():
    src = pathlib.Path(im.__file__).read_text(encoding="utf-8")
    body = src[src.index("def capture("):]
    assert "st_mtime != _code_mtime" in body and "break" in body


# ── NSE's stock rows stuck while the index moves (4 Nov 2026) ───────────────────────────
_T = dt.datetime(2026, 11, 4, 9, 34, 39)


def _stuck_cash(status="Open"):
    return dict(code="NIFTY", ts=_T, status=status, fetched=_T,
                index=dict(last=23937.2, prev=23810.8, open=None, high=None, low=None),
                stocks={"HDFCBANK": dict(ltp=725.75, prev=725.5, open=None, pct=0.0345,
                                         value=9e9, upd="2026-11-04 09:15:46"),
                        "SBIN": dict(ltp=1010.0, prev=1000.0, open=None, pct=1.0,
                                     value=5e9, upd="2026-11-04 09:34:30")})


def _quote(S, ts):
    return dict(by_exp={"S": S}, ts=ts)


def test_a_stuck_stock_row_takes_its_derivatives_quote_price():
    deriv = {"HDFCBANK": _quote(738.5, dt.datetime(2026, 11, 4, 9, 34, 56)),
             "SBIN": _quote(1012.0, dt.datetime(2026, 11, 4, 9, 34, 50))}
    out = im._fresh_prices(_stuck_cash(), deriv)
    h = out["stocks"]["HDFCBANK"]
    assert h["ltp"] == 738.5 and h["pct"] == pytest.approx((738.5 / 725.5 - 1) * 100)
    assert h["value"] is None and h["src"] == "deriv"         # frozen value is not used
    assert out["stocks"]["SBIN"]["ltp"] == 1010.0               # current row: left alone
    assert out["stuck"] == [["HDFCBANK"], "09:15:46"]
    json.dumps(out["stuck"])                                    # goes to latest_*.json
    assert _stuck_cash()["stocks"]["HDFCBANK"]["ltp"] == 725.75  # input never mutated


def test_no_swap_after_the_close_or_from_an_older_quote_or_without_one():
    deriv = {"HDFCBANK": _quote(738.5, dt.datetime(2026, 11, 4, 9, 34, 56))}
    assert "stuck" not in im._fresh_prices(_stuck_cash("Closed"), deriv)
    old = {"HDFCBANK": _quote(738.5, dt.datetime(2026, 11, 4, 9, 15, 0))}
    assert im._fresh_prices(_stuck_cash(), old)["stocks"]["HDFCBANK"]["ltp"] == 725.75
    assert im._fresh_prices(_stuck_cash(), {})["stocks"]["HDFCBANK"]["ltp"] == 725.75


def test_half_the_index_not_updating_is_called_a_feed_fault_not_halts():
    moves = {s: 0.1 for s in "ABCD"}
    cash = _cash(moves, ts=dt.datetime(2026, 10, 13, 12, 0))
    for s in "ABC":
        cash["stocks"][s]["upd"] = "2026-10-13 11:40:00"
    notes = im.decompose(cash, {s: 25.0 for s in moves}, {}, "NIFTY",
                         now=dt.datetime(2026, 10, 13, 12, 0))["notes"]
    assert any("feed fault, not a trading halt" in n for n in notes)
    assert not any(n.startswith("No trade for 10+ minutes") for n in notes)


def test_an_entirely_empty_numeric_column_shows_a_dash_not_the_word_none():
    df = im._blank(pd.DataFrame({"Bucket": ["Top 10", "Next 10"], "Activity %": [None, None],
                                 "Weight %": [53.1, None]}), ["Activity %", "Weight %"])
    out, spec = im._dash_empty(df, {"Activity %": ("%+.0f", "tip"), "Weight %": ("%.1f", "tip")})
    assert out["Activity %"].tolist() == ["—", "—"] and spec["Activity %"][0] is None
    assert spec["Weight %"][0] == "%.1f"                    # partly filled: stays a number


# ── trade gate + one chart per side (5 Nov 2026) ─────────────────────────────────────────
def _gate_rows(last, vol, doi=100):
    prev = round(_premium(1000, 1000, _PREV, 0.25), 2)
    return [("CE", 1000.0, _EXP, last, prev, doi, vol)]


def _gate_side(out):
    return next((q for q in im.OPT_QUADS if out["CE"][q][0]), "unread")


def test_an_untraded_contract_keeps_its_side_however_the_stock_moves():
    """Its 'last price' is minutes old: against a moving future it seemed to flip between
    buyer and writer every sweep (14.6% of readings switched on 4 Nov 2026)."""
    mem = {}
    bid = round(_premium(1000, 1000, _NOW, 0.29), 2)
    first = im.option_flow(_gate_rows(bid, 500), {_EXP: (1000.0, 1000.0)}, {_EXP: 250},
                           _NOW, _PREV, "iv", mem)
    assert _gate_side(first) == "long build"
    # the stock rises 1.5%, the option has NOT traded since (same volume, same stale
    # price): without the gate the stale premium now reads as offered
    moved = {_EXP: (1015.0, 1000.0)}
    assert _gate_side(im.option_flow(_gate_rows(bid, 500), moved, {_EXP: 250}, _NOW, _PREV,
                                     "iv", None)) == "writing"
    assert _gate_side(im.option_flow(_gate_rows(bid, 500), moved, {_EXP: 250}, _NOW, _PREV,
                                     "iv", mem)) == "long build"
    # once it trades again (volume up) at an offered price, it is re-judged
    offered = round(_premium(1015, 1000, _NOW, 0.21), 2)
    assert _gate_side(im.option_flow(_gate_rows(offered, 650), moved, {_EXP: 250}, _NOW,
                                     _PREV, "iv", mem)) == "writing"


def test_a_premium_back_inside_the_band_keeps_its_side_instead_of_turning_unclear():
    mem = {}
    bid = round(_premium(1000, 1000, _NOW, 0.29), 2)
    im.option_flow(_gate_rows(bid, 500), {_EXP: (1000.0, 1000.0)}, {_EXP: 250}, _NOW, _PREV,
                   "iv", mem)
    flat = round(_premium(1000, 1000, _NOW, 0.25), 2)      # traded again, at 'no change'
    assert _gate_side(im.option_flow(_gate_rows(flat, 600), {_EXP: (1000.0, 1000.0)},
                                     {_EXP: 250}, _NOW, _PREV, "iv", mem)) == "long build"


def test_the_premium_read_and_past_days_never_use_the_gate():
    mem = {("CE", 1000.0, _EXP): (1, 999)}
    prev = round(_premium(1000, 1000, _PREV, 0.25), 2)
    dn = round(_premium(1000, 1000, _NOW, 0.21), 2)
    out = im.option_flow([("CE", 1000.0, _EXP, dn, prev, 100, 5)], {_EXP: (1000.0, 1000.0)},
                         {_EXP: 250}, _NOW, _PREV, "premium", mem)
    assert out["CE"]["writing"][0] == 100                     # premium read: no memory
    src = pathlib.Path(im.__file__).read_text(encoding="utf-8")
    body = src[src.index("def session_snapshot"):src.index("def official_close_read")]
    assert "stock_option_flow" not in body                    # replays read the logged q


def test_calls_and_puts_each_get_their_own_line(monkeypatch):
    monkeypatch.setattr(im, "SERIES_QUORUM", 0.5)
    tape = [dict(t="2026-11-05T10:00:00", q={"A": _q(CE_writing=10.0, PE_long_build=4.0)}),
            dict(t="2026-11-05T10:02:00", q={"A": _q(CE_long_build=6.0, PE_writing=3.0)})]
    op = im.option_positioning(tape, {"Top": ["A"]})
    assert op["net_leg"]["CE"]["Top"].tolist() == [-10.0, 6.0]
    assert op["net_leg"]["PE"]["Top"].tolist() == [-4.0, 3.0]
    assert op["net"]["Top"].tolist() == [-14.0, 9.0]          # = the two added


def test_the_options_section_has_no_futures_price_panel():
    src = pathlib.Path(im.__file__).read_text(encoding="utf-8")
    body = src[src.index("def _option_positioning_view"):src.index("def _tip_optq")
               if "def _tip_optq" in src[src.index("def _option_positioning_view"):] else None]
    view = body[:body.index("st.caption(\n        \"Every option contract")]
    assert "futures move since yesterday" not in view
    assert view.count("st.altair_chart(") == 1                # one call inside the per-side loop
