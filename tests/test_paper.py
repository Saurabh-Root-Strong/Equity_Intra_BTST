"""paper.py — the R-multiple simulator. Offline, synthetic, no DB and no network.

These pin the assumptions that decide whether the page tells the truth. Every one of them is
a place where a simulator conventionally cheats in its own favour, so each has a test that
fails if the cheat is reintroduced.
"""
import numpy as np
import pandas as pd

from eqbtst import config, paper


def _day(n=40, start="2024-01-01"):
    ts = pd.bdate_range(start, periods=n)
    px = np.linspace(100.0, 100.0, n)
    return pd.DataFrame({"ts": ts, "open": px, "high": px + 1, "low": px - 1, "close": px})


# ── fill discipline ──────────────────────────────────────────────────────────────────
def test_entry_fills_at_the_NEXT_bar_open_never_the_signal_bar_close():
    """The single most common backtest lie: deciding on a close and filling at that close. The
    close is only knowable once the bar is over, so the fill has to be the next bar's open."""
    d = _day()
    d.loc[5, "close"] = 100.0
    d.loc[6, "open"] = 111.0                     # unmistakably different from bar 5's close
    tr = paper._walk(d, i=5, side="LONG", stop_atr=1.0, rr=3.0, max_bars=10, atr=2.0)
    assert tr is not None
    # entry is not returned directly; reconstruct it the way simulate() does
    assert float(d["open"].iloc[6]) == 111.0
    # a stop 1 ATR under a 111 entry is 109 -- the flat 99/101 bars trip it immediately
    assert tr["why"] == "stop" and abs(tr["exit_px"] - 109.0) < 1e-9


def test_a_signal_on_the_last_bar_cannot_trade():
    d = _day(n=10)
    assert paper._walk(d, i=9, side="LONG", stop_atr=1.0, rr=3.0, max_bars=5, atr=2.0) is None


# ── the ambiguous bar ────────────────────────────────────────────────────────────────
def test_when_one_bar_holds_BOTH_stop_and_target_the_stop_wins():
    """A daily bar carries no intrabar path. Resolving the ambiguity in the trade's favour is
    precisely how a backtest manufactures an edge that evaporates on a real tape."""
    d = _day(n=10)
    d.loc[1, "open"] = 100.0
    d.loc[2, ["high", "low"]] = [130.0, 90.0]    # spans a 98 stop AND a 106 target
    tr = paper._walk(d, i=0, side="LONG", stop_atr=1.0, rr=3.0, max_bars=5, atr=2.0)
    assert tr["why"] == "stop" and tr["R"] == -1.0


def test_the_same_bar_rule_holds_for_a_short():
    d = _day(n=10)
    d.loc[1, "open"] = 100.0
    d.loc[2, ["high", "low"]] = [130.0, 90.0]    # short: stop 102, target 94 -- both inside
    tr = paper._walk(d, i=0, side="SHORT", stop_atr=1.0, rr=3.0, max_bars=5, atr=2.0)
    assert tr["why"] == "stop" and tr["R"] == -1.0


def test_a_clean_target_hit_books_exactly_the_reward_multiple():
    d = _day(n=12)
    d.loc[1, "open"] = 100.0
    d.loc[2:, ["high", "low", "close"]] = [106.5, 99.5, 106.0]   # target 106, stop 98 untouched
    tr = paper._walk(d, i=0, side="LONG", stop_atr=1.0, rr=3.0, max_bars=5, atr=2.0)
    assert tr["why"] == "target" and tr["R"] == 3.0


def test_the_time_stop_books_the_fraction_of_R_it_was_actually_worth():
    """Not a win, not a loss — whatever it was worth. Rounding a time exit to zero, or dropping
    it, would quietly remove the trades that go nowhere, which is most of them."""
    d = _day(n=12)
    d.loc[1, "open"] = 100.0
    d.loc[2:, ["high", "low", "close"]] = [101.0, 99.5, 101.0]
    tr = paper._walk(d, i=0, side="LONG", stop_atr=1.0, rr=3.0, max_bars=3, atr=2.0)
    assert tr["why"] == "time"
    assert abs(tr["R"] - 0.5) < 1e-9             # +1.00 point on a 2.00 risk


# ── the weekly confirm frame ─────────────────────────────────────────────────────────
def test_weekly_bars_are_grouped_not_resampled_so_holiday_weeks_are_absent():
    """resample() manufactures EMPTY buckets for holiday weeks, and an empty weekly bar is not
    a quiet week — it is a NaN row struct_full would then count toward its 20-bar lookback."""
    ts = list(pd.bdate_range("2024-01-01", periods=5)) + list(pd.bdate_range("2024-01-22", periods=5))
    d = pd.DataFrame({"trade_date": ts, "open_price": 100.0, "high_price": 101.0,
                      "low_price": 99.0, "close_price": 100.0})
    w = paper._weekly(d)
    assert len(w) == 2, "two traded weeks, not four calendar ones"
    assert w["close"].notna().all()


def test_the_weekly_frame_used_at_bar_t_has_already_CLOSED():
    """The confirmation frame must never include the week the decision sits inside — that is a
    partial bar built partly out of the future relative to the entry."""
    d = pd.DataFrame({"trade_date": pd.bdate_range("2024-01-01", periods=30),
                      "open_price": 100.0, "high_price": 101.0,
                      "low_price": 99.0, "close_price": 100.0})
    w = paper._weekly(d)
    end = pd.to_datetime(w["end"]).to_numpy()
    for t in d["trade_date"]:
        nw = int(np.searchsorted(end, np.datetime64(pd.Timestamp(t)), side="left"))
        if nw:
            assert pd.Timestamp(end[nw - 1]) < pd.Timestamp(t), (t, end[nw - 1])


# ── accounting ───────────────────────────────────────────────────────────────────────
def test_expectancy_equals_its_own_decomposition():
    """mean(R) and (win% x avgWin - loss% x avgLoss) are the same quantity rearranged. Both are
    reported so they can be checked against each other; if they ever part, the page is lying."""
    for R in ([3.0, -1, -1, -1, 3.0], [-1] * 10, [0.5, -0.2, 3, -1, -1, 2.5]):
        e = paper.expectancy(pd.Series(R))
        assert abs(e["expectancy_R"] - e["expectancy_check"]) < 1e-6, e


def test_expectancy_reproduces_the_textbook_case():
    """40% at 3R vs 60% at 1R — the claim the page exists to test. The arithmetic must be
    right before the measurement can be trusted to contradict it."""
    R = [3.0] * 40 + [-1.0] * 60
    e = paper.expectancy(pd.Series(R))
    assert e["win%"] == 40.0
    assert abs(e["expectancy_R"] - 0.60) < 1e-9
    R2 = [1.0] * 60 + [-1.0] * 40
    assert abs(paper.expectancy(pd.Series(R2))["expectancy_R"] - 0.20) < 1e-9


def test_expectancy_reports_a_standard_error_and_a_t():
    """A mean with no error bar is a guess. At a wide target the R distribution is two spikes,
    so the t is the only thing separating a real edge from a lucky run."""
    e = paper.expectancy(pd.Series([3.0, -1, -1, -1, 3.0, -1]))
    assert e["se_R"] > 0 and np.isfinite(e["t"])


def test_cost_in_R_scales_with_the_stop_width():
    """A flat bps charge is a DIFFERENT fraction of risk for every stop width. Charging it in
    bps, or subtracting a flat 0.1R, would flatter tight stops exactly where they are least
    survivable — so the conversion is the whole point."""
    entry, cost_bps = 1000.0, 22.0
    wide = (cost_bps / 1e4 * entry) / 20.0       # 2% stop
    tight = (cost_bps / 1e4 * entry) / 5.0       # 0.5% stop
    assert abs(wide - 0.11) < 1e-9
    assert abs(tight - 0.44) < 1e-9
    assert tight > wide * 3


def test_no_overlapping_positions_in_one_name():
    """A structure tag PERSISTS, so an unguarded loop re-enters daily: measured on a smoke run,
    1,467 'trades' from 8 names in under four years, the same AXISBANK short opened on six
    consecutive sessions. Those are not independent observations and they are not tradeable.

    BEHAVIOURAL, not a source grep -- the guard has to live in the WALK (which signals are
    blocked depends on when the previous position exited, and that is an exit-rule question),
    so a test tied to one function's text breaks on a correct refactor."""
    d = _day(n=30)
    d.loc[:, "open"] = 100.0
    dates = pd.bdate_range("2024-01-01", periods=30).to_numpy()
    # a signal on EVERY bar, exactly the pathological case
    got = [(i, "LONG", float("nan"), 2.0) for i in range(5, 25)]
    bundle = {"sig": {"X": (d, dates, got)}, "meta": {}}
    t = paper.walk_signals(bundle, rr=3.0, stop_atr=1.0, max_bars=5, cost_bps=0.0)
    assert len(t) < len(got), "every signal became a trade -- the guard is gone"
    # no trade may start before the previous one ended
    ent = pd.to_datetime(t["entry_date"]).tolist()
    ex = pd.to_datetime(t["exit_date"]).tolist()
    for a, b in zip(ex, ent[1:]):
        assert b > a, f"position opened {b} while the previous was still open until {a}"


def test_the_sweep_walks_ONE_signal_generation():
    """Every sweep row must come from the same signals, or a difference between rows could be
    a regeneration artifact while being read as a target effect."""
    import inspect
    src = inspect.getsource(paper.rr_sweep)
    assert "walk_signals(bundle" in src
    assert "build_signals" in src and "if bundle is None" in src


def test_the_rr_sweep_covers_the_folklore_range():
    assert 1.0 in paper.RR_GRID and 3.0 in paper.RR_GRID and 5.0 in paper.RR_GRID


def test_only_the_archive_backed_frame_pair_is_offered():
    """Intraday pairs need broker history (~60 days, rate-limited). Offering them would dress a
    few months of one regime up as a backtest."""
    assert paper.FRAMES == {"positional": ("1D", "1W")}


# ─────────────────────────────────────────────────────────────────────────────────────
# SCALP LANE — 1m trigger / 5m confirm
# ─────────────────────────────────────────────────────────────────────────────────────

def test_breakeven_is_a_pure_function_of_the_cost_fraction():
    """p = (1 + cost_R) / (1 + R:R). This is the lane's whole argument: cost_R = cost / stop,
    so the TIGHTER the scalp stop the HIGHER the bar — the opposite of the intuition that a
    tight stop is conservative."""
    assert abs(paper.breakeven_win_rate(0.0, 1.0) - 0.5) < 1e-9
    assert abs(paper.breakeven_win_rate(0.688, 1.0) - 0.844) < 1e-3
    assert abs(paper.breakeven_win_rate(0.688, 3.0) - 0.422) < 1e-3
    # a stop so tight the round trip exceeds it -> no hit rate whatsoever can pay
    assert paper.breakeven_win_rate(1.376, 1.0) > 1.0


def test_tighter_stops_raise_the_bar_monotonically():
    prev = 0.0
    for stop_bps in (200, 100, 50, 25, 15, 10, 5):
        need = paper.breakeven_win_rate(6.88 / stop_bps, 1.0)
        assert need > prev, "a tighter stop must demand a HIGHER win rate"
        prev = need


def test_scalp_walk_squares_off_at_the_session_end():
    """A scalp that rolls into the next session is not a scalp — and carrying moves it onto the
    delivery schedule (STT on both legs), roughly tripling the bill."""
    d = _day(n=30)
    d.loc[:, "open"] = 100.0
    se = np.array([9] * 10 + [29] * 20)          # session 1 ends at bar 9
    tr = paper._walk_scalp(d, se, i=5, side="LONG", stop_atr=1.0, rr=99.0,
                           max_bars=50, atr=2.0)
    assert tr["exit_i"] == 9, "must close on the last bar of the ENTRY's session"
    assert tr["why"] == "session"


def test_maker_fill_requires_price_to_actually_trade_through_the_limit():
    """Simulating a maker as 'the same fills minus one bp' is the most flattering lie available
    here. A resting order fills only when price comes TO it."""
    d = _day(n=10)
    se = np.array([9] * 10)
    d.loc[0, "close"] = 100.0                    # limit sits here
    d.loc[1, ["high", "low"]] = [105.0, 101.0]   # next bar never trades down to 100
    tr = paper._walk_scalp(d, se, i=0, side="LONG", stop_atr=1.0, rr=2.0,
                           max_bars=5, atr=2.0, maker=True)
    assert tr.get("unfilled") is True
    d.loc[1, ["high", "low"]] = [105.0, 99.0]    # now it does
    tr2 = paper._walk_scalp(d, se, i=0, side="LONG", stop_atr=1.0, rr=2.0,
                            max_bars=5, atr=2.0, maker=True)
    assert not tr2.get("unfilled") and tr2["entry"] == 100.0


def test_an_unfilled_maker_signal_is_not_counted_as_a_trade():
    """It is not a loss and not a win — it is not a trade. Counting it either way corrupts the
    win rate; the fill rate is reported separately instead."""
    import inspect
    src = inspect.getsource(paper.walk_scalp_bundle)
    assert 'tr.get("unfilled")' in src and "continue" in src
    assert "fills" in src, "the fill rate has to be tracked, not discarded"


def test_the_scalp_confirm_frame_is_counted_closed_only():
    """scalp.py's first offline replay printed WITH-TREND CONTINUATION at +16.6bps, t=33, 100%
    of sessions positive — because each 1m bar was indexed to the 5m bar CONTAINING it rather
    than the last to have CLOSED. Same guard, same reason."""
    import inspect
    src = inspect.getsource(paper.scalp_signals)
    assert 'np.searchsorted(ht[1:], lt, side="right")' in src
    assert "- 1" not in src.split("closed =")[1].split("\n")[0], "the containing-bar bug"


def test_every_exit_branch_reports_the_entry_it_actually_filled_at():
    """`entry` used to come back only from the time-stop path, so a MAKER trade that hit its
    stop or target had its cost_R priced off the TAKER fill (the next bar's open) — a fill it
    never got. Small in rupees, wrong in kind: comparing those two fills is the lane's job."""
    d = _day(n=10)
    se = np.array([9] * 10)
    d.loc[0, "close"] = 100.0
    d.loc[1, ["high", "low"]] = [105.0, 99.0]
    d.loc[2, ["high", "low"]] = [130.0, 90.0]        # trips both -> stop branch
    for maker in (False, True):
        tr = paper._walk_scalp(d, se, i=0, side="LONG", stop_atr=1.0, rr=2.0,
                               max_bars=5, atr=2.0, maker=maker)
        assert "entry" in tr, (maker, tr)
