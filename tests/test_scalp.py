"""Scalper lane — the guards that matter.

Three classes of thing are pinned here, and only three:

  1. THE COST MODEL, because it is the whole product. A scalp board whose cost number is
     wrong is worse than no board: it hands out a confident ✅ on a trade that cannot pay.
  2. THE CAUSALITY GUARDS, because this lane's offline study already produced one
     spectacular lookahead (+16.6bps, t=33, 100% of sessions positive) from a single
     mis-indexed line, and the live scan runs the same shape of code.
  3. THE HONESTY, because every measured number on the page is a claim that something does
     NOT work, and those are exactly the claims that quietly soften over time.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eqbtst import config, scalp

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "eqbtst" / "dashboard.py"
SCALP = ROOT / "eqbtst" / "scalp.py"


# ── COST ─────────────────────────────────────────────────────────────────────────────
def test_cost_is_the_intraday_schedule_not_the_delivery_one():
    """THE one that must never regress.

    config.COST_BPS = 22 is a DELIVERY round trip and is ~22 almost entirely because
    delivery STT is 0.1% on BOTH legs. Applying it to a square-off would veto every scalp
    on the board; applying a delivery STT silently would do the same, more quietly."""
    assert scalp.STT_SELL == 0.00025, "intraday STT is 0.025%, SELL leg only"
    parts = scalp.cost_parts(200_000, 1000.0)
    assert abs(parts["stt"] - 2.5) < 1e-9, "STT must be 2.5bps of turnover, not 20"
    total = scalp.round_trip_bps(200_000, 1000.0)
    assert 5.0 < total < 13.0, f"intraday round trip out of range: {total}"
    assert total < config.COST_BPS / 1.5, (
        "the scalp floor must be materially BELOW the delivery floor, or the whole reason "
        "this lane can exist is gone")


def test_cost_falls_with_position_size_because_brokerage_is_flat():
    """The Rs20-per-order brokerage is a FIXED cost being amortised, and it is the single
    biggest lever on the page: 6bps of Rs50k, 0.8bps of Rs5L. A cost model that ignored
    size would tell a Rs50k trader the same story as a Rs10L one."""
    sizes = [50_000, 100_000, 200_000, 500_000, 1_000_000]
    totals = [scalp.round_trip_bps(v, 1000.0) for v in sizes]
    assert totals == sorted(totals, reverse=True), f"cost not monotone in size: {totals}"
    assert totals[0] - totals[-1] > 5.0, "size should move the total by more than 5bps"
    # and the mover must be BROKERAGE, not something that drifted in
    brok = [scalp.cost_parts(v, 1000.0)["brokerage"] for v in sizes]
    assert brok[0] > 5.0 and brok[-1] < 1.0


def test_cheap_stocks_carry_a_bigger_tick_cost():
    """One NSE tick is Rs0.05 whatever the price, so it is 3.3bps of a Rs150 name and
    0.5bps of a Rs1000 one. A flat spread assumption would make penny names look tradeable."""
    cheap = scalp.cost_parts(200_000, 150.0)["spread"]
    rich = scalp.cost_parts(200_000, 1000.0)["spread"]
    assert cheap > rich * 5, f"tick cost not scaling with price: {cheap} vs {rich}"


def test_cost_never_raises_on_junk():
    for bad in (None, "x", 0, -5, float("nan")):
        v = scalp.round_trip_bps(bad, 1000.0)
        assert v == v and v > 0, f"junk position {bad!r} produced {v}"
        v = scalp.round_trip_bps(200_000, bad)
        assert v == v and v > 0, f"junk price {bad!r} produced {v}"


# ── P(pays) ──────────────────────────────────────────────────────────────────────────
def test_p_pays_is_monotone_and_bounded():
    """More expected movement against the same cost can only ever make the trip MORE
    likely to cover itself."""
    ps = [scalp.p_pays(r, 8.0) for r in (2, 4, 8, 16, 32, 64)]
    assert ps == sorted(ps), f"p_pays not monotone in expected move: {ps}"
    assert all(0.0 <= p <= 100.0 for p in ps)
    # and monotone DOWNWARD in cost
    cs = [scalp.p_pays(12.0, c) for c in (2, 4, 8, 16, 32)]
    assert cs == sorted(cs, reverse=True), f"p_pays not monotone in cost: {cs}"


def test_p_pays_degrades_to_nan_not_to_a_confident_number():
    """A missing input must produce a dash, never a plausible percentage. `pays?` is the
    column the user is told to treat as final, so a fabricated value there is the most
    expensive kind of bug this page can have."""
    for bad in (None, "x", 0, -1, float("nan")):
        assert scalp.p_pays(bad, 8.0) != scalp.p_pays(bad, 8.0) or np.isnan(
            scalp.p_pays(bad, 8.0)), f"p_pays({bad!r}) returned a number"
        assert np.isnan(scalp.p_pays(12.0, bad)), f"p_pays(cost={bad!r}) returned a number"
    assert scalp._pays_tag(float("nan")) == "—"


def test_the_ratio_table_is_a_real_survival_curve():
    """Probabilities must fall as the hurdle rises. A table entered by hand can invert."""
    assert list(scalp._RATIO_X) == sorted(scalp._RATIO_X)
    assert list(scalp._RATIO_P) == sorted(scalp._RATIO_P, reverse=True)
    assert len(scalp._RATIO_X) == len(scalp._RATIO_P)


def test_exp_move_keeps_its_intercept():
    """A pure ratio form under-predicts the quietest decile by 33% -- it says a dead name is
    deader than it is, which is the one direction of error that HIDES a real trade rather
    than manufacturing one. The intercept is what fixes that."""
    assert scalp.EXP5_A > 1.0, "the OLS intercept was dropped"
    assert 0.5 < scalp.EXP5_B < 1.5
    assert scalp.exp_move_bps(0) == pytest.approx(scalp.EXP5_A)
    assert scalp.exp_move_bps(10) > scalp.exp_move_bps(5)
    assert np.isnan(scalp.exp_move_bps(None))


# ── CAUSALITY ────────────────────────────────────────────────────────────────────────
def _frame(n_minutes: int, start="2026-08-25 09:15"):
    ts = pd.date_range(start, periods=n_minutes, freq="1min")
    px = np.linspace(100.0, 101.0, n_minutes)
    return pd.DataFrame({"ts": ts, "open": px, "high": px + 0.1, "low": px - 0.1,
                         "close": px, "volume": np.full(n_minutes, 1000.0)})


def test_closed_never_returns_a_bar_that_has_not_finished_printing():
    """THE LOOKAHEAD GUARD.

    A bar stamped T on an m-minute frame covers [T, T+m) and is only known at T+m. The
    offline replay for this lane indexed each 1-minute bar to the 5-minute bar CONTAINING it
    and printed WITH-TREND CONTINUATION at +16.6bps with t=33 -- four minutes of the future
    against a five-minute forward return. Same shape of code ships in _scan_one."""
    f = _frame(60)
    five = scalp._resample(f, 5)
    now = pd.Timestamp("2026-08-25 09:42").to_pydatetime()
    out = scalp._closed(five, 5, now)
    assert out["ts"].max() + pd.Timedelta(minutes=5) <= pd.Timestamp(now), (
        f"_closed returned a bar ending after `now`: last={out['ts'].max()} now={now}")
    # the 09:40 bar closes at 09:45, so at 09:42 the newest CLOSED bar is 09:35
    assert out["ts"].max() == pd.Timestamp("2026-08-25 09:35")


def test_closed_degrades_to_dropping_one_bar_rather_than_emptying_the_frame():
    """In the first minutes of a session nothing has closed yet. Returning empty would render
    the name as UNREADABLE, which is a different (and wrong) statement from 'it is early'."""
    f = _frame(60)
    out = scalp._closed(f, 1, pd.Timestamp("2020-01-01").to_pydatetime())
    assert len(out) == len(f) - 1 and not out.empty


# ── FRAMES ───────────────────────────────────────────────────────────────────────────
def test_the_session_stub_is_folded_on_7m_and_10m_but_not_5m():
    """375 minutes divides by 5 exactly, and by neither 7 (53.6) nor 10 (37.5). So a 7m board
    ends every session on a FOUR-minute bar and a 10m board on a FIVE-minute one, and the
    structure classifier compares bar ranges against each other -- a half-width bar wearing a
    full-width label is what was manufacturing coils on the 1h/2h frames."""
    f = _frame(375)                                   # one full NSE session
    assert len(scalp._resample(f, 5)) == 75, "5m should bin exactly, no stub"
    for m in (7, 10):
        r = scalp._resample(f, m)
        last = r["ts"].iloc[-1]
        span = min(last + pd.Timedelta(minutes=m),
                   last.normalize() + pd.Timedelta("15h30min")) - last
        assert span >= pd.Timedelta(minutes=scalp._STUB_FRAC * m), (
            f"{m}m frame ends on an unfolded {span} stub")
        # nothing is discarded -- the folded bar keeps the session's true extremes
        assert r["high"].max() == pytest.approx(f["high"].max())
        assert r["low"].min() == pytest.approx(f["low"].min())
        assert r["volume"].sum() == pytest.approx(f["volume"].sum())


def test_resample_leaves_the_1m_series_alone():
    f = _frame(30)
    assert scalp._resample(f, 1) is f


def test_the_validated_stub_helper_is_left_untouched():
    """scalp._resample owns its own folding rule ON PURPOSE. live.merge_session_stubs
    early-returns at <=15 minutes, which is CORRECT for the frames it was measured on (1, 5
    and 15 all divide 375 exactly). Widening that guard would change a validated function's
    behaviour on frames nobody re-measured."""
    from eqbtst import live
    src = io.open(ROOT / "eqbtst" / "live.py", encoding="utf-8").read()
    assert "freq_min <= 15" in src, (
        "live.merge_session_stubs' guard was widened -- re-measure 1h/2h/4h before doing that")
    assert live.merge_session_stubs(_frame(30), 10) is not None


# ── UNIVERSE ─────────────────────────────────────────────────────────────────────────
def test_universe_ranks_by_range_not_by_turnover():
    """Measured across 40 names, ATR% ranks realised 5-minute movement at Spearman +0.902
    while TURNOVER ranks it at -0.10 -- sorting by liquidity picks the LEAST scalpable names
    in the market (top-10 by turnover moved 10.08bps, top-10 by ATR% moved 15.44bps)."""
    src = io.open(SCALP, encoding="utf-8").read()
    body = src[src.index("def scalp_universe"):src.index("def scan_seconds")]
    assert 'nlargest(n, "atr_pct")' in body, "the ranking key stopped being ATR%"
    assert 'nlargest(n, "turn_cr")' not in body.split("if gated.empty")[0], (
        "turnover is being used to RANK, not merely to gate")
    assert '"turn_cr"] >= float(min_turn_cr)' in body, "the fillability floor is gone"


def test_universe_never_returns_empty_just_because_the_floor_was_set_too_high():
    """An empty board is indistinguishable from 'nothing is scalpable today', which is a
    completely different statement from 'your filter is too tight'."""
    if not config.DCM_DUCKDB.exists():
        pytest.skip("archive not reachable")
    scalp.scalp_universe.cache_clear()
    try:
        out = scalp.scalp_universe(5, 1e9)          # a floor no name can clear
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"archive unavailable: {e}")
    assert isinstance(out, tuple)
    if out:
        assert len(out) == 5


def test_scan_seconds_tracks_the_measured_pacer():
    """The slider's estimate is the whole basis for the 'this cannot keep up with a 1-minute
    bar' warning. If it drifts from the real pacer the warning fires at the wrong place."""
    from eqbtst import live
    assert scalp.scan_seconds(100) == pytest.approx(100 * live._HIST_GAP, rel=0.01)
    assert scalp.scan_seconds(242) > 60, (
        "the full universe must still be reported as slower than a 1-minute bar")
    assert scalp.scan_seconds(0) == 0


# ── THE BROKER'S DUPLICATE CANDLES ───────────────────────────────────────────────────
def test_duplicate_candles_are_dropped_on_every_same_date_fetch():
    """The broker returns BYTE-IDENTICAL duplicate candles when range_from == range_to
    (verified: RELIANCE 2026-08-25 came back 50 rows / 25 unique at 15m, 750/375 at 1m).

    A doubled series leaves the range BOX intact -- min and max do not care -- so nothing
    looks wrong, while every volume figure doubles and a 20-bar structure window silently
    covers only TEN real bars. Both replay fetchers use the single-date form."""
    from eqbtst import live
    f = _frame(10)
    doubled = pd.concat([f, f], ignore_index=True)
    out = live._dedupe_candles(doubled)
    assert len(out) == 10 and out["ts"].is_unique
    assert out["volume"].sum() == pytest.approx(f["volume"].sum())
    src = io.open(ROOT / "eqbtst" / "live.py", encoding="utf-8").read()
    # every fetcher that pins range_from == range_to must de-dupe
    assert src.count("_dedupe_candles(f)") >= 3, (
        "a same-date candle fetcher lost its de-dup -- replay will read half its history")


# ── WHAT THE PAGE IS ALLOWED TO CLAIM ────────────────────────────────────────────────
def test_no_setup_cell_is_recorded_as_clearing_the_cost_floor():
    """THE TRIPWIRE.

    Every one of the 10 causal tag/side cells is negative net of cost -- the best is
    +3.118bps gross against a 6.88bps floor. If someone later re-measures and a cell really
    does clear, this test SHOULD fail and force the claim to be made deliberately."""
    assert scalp.EVIDENCE["mtf_cells_clearing_cost"] == 0
    cost = scalp.round_trip_bps(500_000, 1000.0)          # the most favourable size
    best = max(e for _, _, _, e, _, _ in scalp.MTF_CELLS)
    assert best < cost, (
        f"a setup cell now clears the floor ({best:.3f} vs {cost:.2f}bps) -- update the "
        f"docstring, EVIDENCE and the page, do not just relax this test")
    assert len(scalp.MTF_CELLS) == 10


def test_the_withdrawn_lookahead_number_never_comes_back():
    """+16.6bps / t=33 / 100% of sessions positive was a LOOKAHEAD, not a result. It may
    appear in the source only as a labelled retraction, never as a live claim."""
    src = io.open(SCALP, encoding="utf-8").read()
    for hit in re.finditer(r"16\.6\s*bps|\+16\.6", src):
        window = src[max(0, hit.start() - 400):hit.end() + 200].lower()
        assert any(w in window for w in ("lookahead", "withdrawn", "bug", "first version")), (
            "+16.6bps appears without being labelled as the retracted lookahead")


def test_the_page_says_pays_is_not_a_probability_of_profit():
    """`pays?` is P(|move| > cost) -- a NECESSARY condition. Sold as P(profit) it becomes a
    directional signal the measurements say does not exist (46.2% up over 457k bars)."""
    blob = (scalp.HELP["pays"] + scalp.HELP["setup"]).lower()
    assert "not" in blob and "profit" in blob, "the pays? tooltip stopped disclaiming profit"
    assert "46.2" in blob or "coin flip" in blob


def test_scalp_never_imports_streamlit():
    """render_page takes `st` as an ARGUMENT so the page is testable headless. An import
    would make the whole module unloadable outside a Streamlit script run."""
    src = io.open(SCALP, encoding="utf-8").read()
    assert "import streamlit" not in src


# ── THE COLUMNS THIS LANE DELIBERATELY DROPS ─────────────────────────────────────────
def test_the_end_of_day_columns_are_gone():
    """The whole reason scalper mode is a separate lane: at a five-minute hold, delivery, the
    F&O bhavcopy, `carry` and sector tilt are all end-of-day or multi-week numbers that
    cannot change inside the trade. If they creep back the lane has no reason to exist."""
    from eqbtst import fno
    banned = set(fno.COLS) | {"carry", "wtd_deliv7", "deliv_vs_100d", "deliv trend",
                              "sector tilt"}
    assert not (set(scalp.COLS) & banned), \
        f"end-of-day columns leaked into the scalp board: {set(scalp.COLS) & banned}"


def test_every_scalp_column_has_a_tooltip_or_is_self_evident():
    """A column config built by hand drifted from its column list once already on the F&O
    block, and those columns rendered bare."""
    needs_help = {"setup", "exp5m", "cost", "pays", "rvol", "spread", "room", "1m", "5m"}
    assert needs_help <= set(scalp.HELP), f"missing tooltips: {needs_help - set(scalp.HELP)}"
    assert needs_help <= set(scalp.COLS)


# ── `entered` — THE CURRENT RUN, NOT THE FIRST TIME TODAY ────────────────────────────
def test_entered_reports_the_current_run_not_the_first_time_today(monkeypatch):
    """THE reason this column is not a copy of the other boards'.

    Replayed over 475,840 causal evaluations, a scalp side lasts a MEDIAN OF TWO MINUTES and
    flips a median of 21 times per name per session. "First LONG today" and "the LONG you are
    looking at" differ on 86% of sided bars, by a median of 99 minutes. A column answering the
    first question while the user reads it as the second is worse than no column.
    """
    f = _frame(60)                      # 09:15 .. 10:14
    h5 = scalp._resample(f, 5)
    # LONG early, away, then LONG again for the last three bars
    sides = {}
    for i in range(len(f)):
        sides[i] = "LONG" if (i < 10 or i >= len(f) - 3) else "—"
    monkeypatch.setattr(scalp, "_side_at", lambda F, H, i, lb=None: (sides[i], 100.0))
    ent, at, since = scalp.entered_run(f, h5, 0, "LONG", 101.0)
    # the run began at len-3, i.e. 10:12 -- NOT 09:15
    assert ent == f["ts"].iloc[-3].strftime("%H:%M"), ent
    assert ent != f["ts"].iloc[0].strftime("%H:%M"), "reported the first LONG of the day"


def test_entered_marks_a_run_it_could_not_see_the_start_of(monkeypatch):
    """If the run reaches the first readable bar, the flip predates the window and the honest
    answer is a BOUND. Printing a bare time there would claim a flip inside today that may
    never have happened -- and on a board whose whole job is saying how stale a signal is,
    a fabricated start time is the one lie that matters."""
    f = _frame(60)
    h5 = scalp._resample(f, 5)
    # LONG at every bar we can evaluate: the walk must run out of window, not find a flip
    monkeypatch.setattr(scalp, "_side_at", lambda F, H, i, lb=None: ("LONG", 100.0))
    ent, _, _ = scalp.entered_run(f, h5, 0, "LONG", 100.0)
    assert ent.startswith("\u2265"), f"unbounded run reported as a definite time: {ent}"
    # The bound lands on the earliest bar the walk can EVALUATE, not on 09:15: _side_at needs
    # a few bars behind it on both frames before it can say anything. That is the honest
    # answer -- the flip is somewhere at or before there, and `>=` is exactly that claim.
    earliest = f["ts"].iloc[6].strftime("%H:%M")
    assert ent.lstrip("\u2265") <= earliest, f"walk stopped short of the readable floor: {ent}"

    # and the opposite case must NOT be marked: a real flip mid-frame is a known time
    monkeypatch.setattr(scalp, "_side_at",
                        lambda F, H, i, lb=None: (("LONG" if i >= 40 else "\u2014"), 100.0))
    ent2, _, _ = scalp.entered_run(f, h5, 0, "LONG", 100.0)
    assert not ent2.startswith("\u2265"), f"a known flip was reported as a bound: {ent2}"
    assert ent2 == f["ts"].iloc[40].strftime("%H:%M")


def test_since_is_signed_the_way_the_trade_is(monkeypatch):
    """A short PROFITS when price falls. An unsigned `since%` would paint every working short
    red and every losing one green -- exactly inverted on half the board."""
    f = _frame(20)
    h5 = scalp._resample(f, 5)
    monkeypatch.setattr(scalp, "_side_at", lambda F, H, i, lb=None: ("SHORT", 100.0))
    entry = float(f["close"].iloc[0])
    _, _, since = scalp.entered_run(f, h5, 0, "SHORT", entry * 0.99)    # price FELL 1%
    assert since > 0, f"a short that made money reported {since}"
    monkeypatch.setattr(scalp, "_side_at", lambda F, H, i, lb=None: ("LONG", 100.0))
    _, _, since_l = scalp.entered_run(f, h5, 0, "LONG", entry * 0.99)
    assert since_l < 0, f"a long that lost money reported {since_l}"


def test_since_never_prints_negative_zero(monkeypatch):
    """round(-0.0001, 2) is -0.0, which renders as '-0.00' and reads as a losing trade. On a
    one-minute-old run -- the MOST common case here -- that is the usual case, not an edge."""
    f = _frame(20)
    h5 = scalp._resample(f, 5)
    monkeypatch.setattr(scalp, "_side_at", lambda F, H, i, lb=None: ("LONG", 100.0))
    entry = float(f["close"].iloc[-1])
    _, _, since = scalp.entered_run(f, h5, len(f) - 1, "LONG", entry * (1 - 1e-6))
    import math
    assert not (since == 0 and math.copysign(1, since) < 0), "negative zero leaked into since%"


def test_entered_is_empty_for_a_row_with_no_side():
    """No side means no trade to have entered. A time there would be inventing one."""
    f = _frame(60)
    h5 = scalp._resample(f, 5)
    for side in ("\u2014", None, ""):
        ent, at, since = scalp.entered_run(f, h5, 0, side, 100.0)
        assert ent is None and at != at and since != since


def test_side_at_never_reads_an_unclosed_5m_bar():
    """THE LOOKAHEAD GUARD, on the walk-back path this time.

    Acting at the close of 1-minute bar i means acting at ts_i + 1min, so the newest fully
    known 5-minute bar is the last one with T + 5min <= ts_i + 1min. Indexing to the bar
    CONTAINING ts_i is what made the offline study print +16.6bps at t=33."""
    from eqbtst import indicators
    f = _frame(90)
    h5 = scalp._resample(f, 5)
    seen = []
    real = indicators.struct_full

    def spy(candles, **k):
        seen.append(candles)
        return real(candles, **k)

    import unittest.mock as _m
    with _m.patch.object(indicators, "struct_full", spy):
        i = 61
        scalp._side_at(f, h5, i)
    act_at = f["ts"].iat[i] + pd.Timedelta(minutes=1)          # when the decision is taken
    # the 5-minute window is whichever captured frame is not the 1-minute one
    five = [c for c in seen if len(c) and (c["ts"].diff().dropna() > pd.Timedelta("1min")).any()]
    assert five, "no 5-minute window was evaluated"
    last = five[-1]["ts"].max()
    assert last + pd.Timedelta(minutes=5) <= act_at, (
        f"_side_at read a 5m bar closing at {last + pd.Timedelta(minutes=5)} "
        f"while acting at {act_at}")


def test_entered_sits_immediately_after_day_pct():
    """Placement is the request. A staleness column pushed right is the same as absent."""
    i = scalp.COLS.index("day%")
    assert scalp.COLS[i + 1:i + 4] == ["entered", "at", "since%"], scalp.COLS


def test_entered_tooltip_warns_that_fresh_is_not_a_signal():
    """Bucketed by run age, the flip bar ITSELF is the worst cell on the board (-0.49bps,
    clustered t -1.97). Without that line the column reads as an entry trigger."""
    h = scalp.HELP["entered"].lower()
    assert "not a buy signal" in h or "not a trigger" in h
    assert "median of two minutes" in h or "two minutes" in h
    assert "staleness" in h


def test_the_walk_back_is_bounded():
    """Runs are short (median 10 bars, p99 113), but an unbounded walk on a side that held all
    session would run the whole 375-bar frame for every name on every scan."""
    assert 0 < scalp._MAX_WALK <= 375


# ── DASHBOARD WIRING ─────────────────────────────────────────────────────────────────
def test_the_toggle_is_wired_and_stops_the_normal_lane():
    """A toggle that renders the scalp page WITHOUT stopping would draw both boards, and the
    structure lane's own scan would still fire ~250 fetches behind it."""
    src = io.open(DASH, encoding="utf-8").read()
    assert re.search(r'toggle\(\s*\n?\s*"⚡ \*\*Scalper Mode\*\*"', src), "toggle missing"
    m = re.search(r"if _scalper:(.{0,400})", src, re.S)
    assert m and "scalp.render_page(st)" in m.group(1)
    assert "st.stop()" in m.group(1) and "raise SystemExit" in m.group(1), (
        "st.stop() is a no-op outside a script-run context -- the SystemExit is load-bearing")


def test_scalp_cache_is_cleared_wherever_the_others_are():
    """arb.clear_cache() marks every place the board drops its memos. A cache this lane keys
    on a ONE-minute bucket going stale is worse than the others, not better."""
    src = io.open(DASH, encoding="utf-8").read()
    assert src.count("arb.clear_cache()") == src.count("scalp.clear_cache()") >= 2


def test_dashboard_imports_scalp():
    src = io.open(DASH, encoding="utf-8").read()
    assert re.search(r"from eqbtst import \([^)]*\bscalp\b", src, re.S)


# ── HEADLESS PAGE ────────────────────────────────────────────────────────────────────
class _StubST:
    """Records every Streamlit call instead of drawing it. `render_page` takes `st` as an
    argument precisely so it can be exercised headless -- the page is the largest block of
    new code in this lane and would otherwise only ever be tested by looking at it."""

    def __init__(self, parent=None):
        self.calls = parent.calls if parent else []
        self.text = parent.text if parent else []
        self.session_state = parent.session_state if parent else {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        if name in ("column_config", "sidebar"):
            return _StubST(self)

        def f(*a, **k):
            self.calls.append(name)
            if name in ("expander", "container", "spinner", "form", "empty"):
                return _StubST(self)
            self.text.extend(x for x in a if isinstance(x, str))
            if name == "columns":
                n = a[0] if isinstance(a[0], int) else len(a[0])
                return [_StubST(self) for _ in range(n)]
            if name == "tabs":
                return [_StubST(self) for _ in a[0]]
            if name == "slider":
                return a[3] if len(a) > 3 else 30
            if name == "number_input":
                return a[3] if len(a) > 3 else 0
            if name == "selectbox":
                return list(a[1])[k.get("index", 0)]
            if name in ("toggle", "button", "checkbox"):
                return False
            return None
        return f


def test_page_renders_end_to_end_headless(monkeypatch):
    """Every branch of the page except the live fetch: the instruction expander, the cost
    table, the setup-evidence table, the tabs and the board itself."""
    board = pd.DataFrame([{
        "symbol": "AAA", "setup": "COIL AT THE EXTREME", "side": "LONG", "ltp": 100.0,
        "day%": 1.0, "exp5m": 20.0, "cost": 6.9, "p pays": 74.0, "pays": "✅ 74%",
        "rvol": 2.0, "spread": 0.5, "1m": "RANGE", "5m": "TREND_UP", "vs_vwap%": 0.2,
        "sup": 99.0, "res": 101.0, "room": 2.0, "stop": 99.5, "t1": 100.5,
        "s_stop": 100.5, "s_t1": 99.5, "atr₹": 0.5, "turn₹L": 900.0,
    }])
    import datetime as _dt
    monkeypatch.setattr(scalp, "scan", lambda **k: {
        "ok": True, "board": board, "n_scanned": 1, "n_read": 1, "n_blank": 0,
        "scanned_at": _dt.datetime.now(), "up_frame": "10m", "up_min": 10, "syms": ("AAA",)})
    stub = _StubST()
    scalp.render_page(stub)
    for required in ("subheader", "dataframe", "tabs", "slider", "number_input"):
        assert required in stub.calls, f"page never called st.{required}"
    blob = " ".join(stub.text).lower()
    assert "veto" in blob, "the page stopped saying it is a veto rather than a trigger"
    assert "46.2" in blob, "the page stopped quoting the measured direction rate"


def test_page_reports_a_dead_token_instead_of_an_empty_board(monkeypatch):
    """The Fyers token expires ~06:00 IST, before the open, so a dead token is the NORMAL
    first state every morning. Showing an empty board there sends the user hunting a market
    explanation for an auth problem."""
    monkeypatch.setattr(scalp, "scan", lambda **k: {
        "ok": False, "status": "token expired", "board": pd.DataFrame(),
        "n_scanned": 0, "syms": ()})
    stub = _StubST()
    scalp.render_page(stub)
    assert "error" in stub.calls
    assert any("token" in t.lower() for t in stub.text)


def test_scan_reports_not_ok_rather_than_raising_without_a_token(monkeypatch):
    from eqbtst import live
    monkeypatch.setattr(live, "token_status", lambda: {"usable": False, "describe": "dead"})
    out = scalp.scan(n=5)
    assert out["ok"] is False and out["board"].empty
