"""paper_hz.py — the four-horizon Paper-trade lane. Offline, synthetic, no DB and no network.

Each test pins one place where a multi-timeframe simulator conventionally flatters itself:
fills through gaps, the higher frame seen before it closed, the end of the tape counted as an
exit, a universe chosen with hindsight, correlated trades counted as independent.
"""
import numpy as np
import pandas as pd

from eqbtst import indicators, mtf, paper_hz as P


def _flat(n=30, px=100.0):
    o = np.full(n, px)
    return o.copy(), o + 1, o - 1, o.copy()


# ── fills ────────────────────────────────────────────────────────────────────────────
def test_a_gap_through_the_stop_fills_at_the_open_not_the_stop():
    """The old walk filled at the stop price whatever the open, capping every overnight gap
    loss at exactly -1R — the most flattering assumption available on volatile names."""
    o, h, l, c = _flat()
    o[3], h[3], l[3], c[3] = 90.0, 91.0, 89.0, 90.0          # opens 5 ATR under a 98 stop
    tr = P.walk(o, h, l, c, i=1, side="LONG", stop_atr=1.0, rr=2.0, max_bars=10, atr=2.0)
    assert tr["why"] == "gap-stop" and tr["exit_px"] == 90.0
    assert abs(tr["R"] - (-5.0)) < 1e-9


def test_a_short_gapped_through_its_stop_also_fills_at_the_open():
    o, h, l, c = _flat()
    o[3], h[3], l[3], c[3] = 110.0, 111.0, 109.0, 110.0
    tr = P.walk(o, h, l, c, i=1, side="SHORT", stop_atr=1.0, rr=2.0, max_bars=10, atr=2.0)
    assert tr["why"] == "gap-stop" and tr["R"] == -5.0


def test_a_gap_through_the_target_fills_at_the_open():
    o, h, l, c = _flat()
    o[3], h[3], l[3], c[3] = 110.0, 111.0, 109.0, 110.0
    tr = P.walk(o, h, l, c, i=1, side="LONG", stop_atr=1.0, rr=2.0, max_bars=10, atr=2.0)
    assert tr["why"] == "gap-target" and tr["R"] == 5.0


def test_the_entry_bar_is_never_treated_as_a_gap():
    """The entry IS the open of bar i+1; comparing that open to the stop would be circular."""
    o, h, l, c = _flat()
    tr = P.walk(o, h, l, c, i=1, side="LONG", stop_atr=1.0, rr=2.0, max_bars=3, atr=2.0)
    assert tr["why"] == "time"


def test_both_in_one_bar_is_still_a_stop():
    o, h, l, c = _flat()
    h[3], l[3] = 130.0, 70.0
    tr = P.walk(o, h, l, c, i=1, side="LONG", stop_atr=1.0, rr=2.0, max_bars=10, atr=2.0)
    assert tr["why"] == "stop" and tr["R"] == -1.0


def test_running_off_the_end_of_the_data_is_OPEN_not_a_time_exit():
    """A trade still running when the tape ends has no result yet. Booking it as a time-stop
    exit mixes a mark-to-market into the statistics."""
    o, h, l, c = _flat(n=10)
    tr = P.walk(o, h, l, c, i=5, side="LONG", stop_atr=1.0, rr=2.0, max_bars=20, atr=2.0)
    assert tr["why"] == "open" and tr["exit_i"] == 9


def test_intraday_is_squared_off_at_the_session_last_bar():
    o, h, l, c = _flat(n=30)
    tr = P.walk(o, h, l, c, i=2, side="LONG", stop_atr=1.0, rr=9.0, max_bars=20, atr=2.0,
                last_i=6)
    assert tr["why"] == "session" and tr["exit_i"] == 6


def test_the_old_daily_walk_now_uses_the_gap_rule_too():
    from eqbtst import paper
    o, h, l, c = _flat()
    o[3], h[3], l[3], c[3] = 90.0, 91.0, 89.0, 90.0
    d = pd.DataFrame({"open": o, "high": h, "low": l, "close": c})
    tr = paper._walk(d, i=1, side="LONG", stop_atr=1.0, rr=2.0, max_bars=10, atr=2.0)
    assert tr["why"] == "gap-stop" and tr["R"] < -1.0


# ── the clock ────────────────────────────────────────────────────────────────────────
def _session_15m(days=3):
    ts = []
    for d in pd.bdate_range("2025-01-06", periods=days):
        ts += list(pd.date_range(d + pd.Timedelta("9h15min"), d + pd.Timedelta("15h15min"),
                                 freq="15min"))
    n = len(ts)
    px = 100 + np.arange(n) * 0.01
    return pd.DataFrame({"ts": ts, "open": px, "high": px + 0.5, "low": px - 0.5,
                         "close": px, "volume": 1.0})


def test_a_4h_bar_is_invisible_to_the_1h_bars_inside_it():
    """The containing-bar lookahead: the 09:15 4h bar ends 13:15, so the 10:15 1h bar must
    not see it. Visibility is end <= trigger end, never 'the bar that contains me'."""
    f = _session_15m(1)
    fr = P.build_frames("btst", pd.DataFrame({"ts": pd.to_datetime(["2025-01-03"]),
                                              "open": [1.0], "high": [1.0], "low": [1.0],
                                              "close": [1.0]}), f)
    h1, h4 = fr["1h"], fr["4h"]
    assert list(h4["end"].dt.strftime("%H:%M")) == ["13:15", "15:30"]
    assert h1["end"].iloc[-1].strftime("%H:%M") == "15:30", "stub-merged last 1h bar"
    seen = np.searchsorted(h4["end"].to_numpy(), h1["end"].to_numpy(), side="right")
    for e, k in zip(h1["end"], seen):
        assert k == (0 if e < h4["end"].iloc[0] else (1 if e < h4["end"].iloc[1] else 2)), e


def test_weekly_and_monthly_bars_close_at_the_PERIOD_end_not_the_last_row():
    """Stamping a period with its last row's date would make the RUNNING week/month count as
    closed on the archive's final day."""
    d = pd.DataFrame({"ts": pd.bdate_range("2025-01-01", "2025-02-12")})
    d["open"] = d["high"] = d["low"] = d["close"] = 100.0
    d["high"] += 1
    w = P._period_frame(d, "W-FRI")
    m = P._period_frame(d, "M")
    assert w["end"].iloc[-1] == pd.Timestamp("2025-02-14 15:30"), "a Wednesday is mid-week"
    assert m["end"].iloc[-1] == pd.Timestamp("2025-02-28 15:30")
    assert all(e.dayofweek == 4 for e in w["end"])


def test_the_upper_frame_ladder_is_the_boards():
    assert {k: v["up"] for k, v in P.HORIZONS.items()} == {"sml_intraday": "1h",
        "intraday": "4h", "btst": "1D", "swing": "1W", "positional": "1M"}
    assert mtf.one_frame_up("2h") == "4h"


# ── the real indicator, just faster ──────────────────────────────────────────────────
def test_the_array_view_gives_struct_full_identical_output():
    rng = np.random.default_rng(7)
    c = 100 + np.cumsum(rng.normal(0, 1, 400))
    df = pd.DataFrame({"open": c + rng.normal(0, .2, 400), "high": c + 1.2, "low": c - 1.2,
                       "close": c})
    A = P._arrays(df)
    for i in range(40, 400, 7):
        assert indicators.struct_full(df.iloc[i - 40:i]) == \
            indicators.struct_full(P._View(A, i - 40, i)), i


def test_a_tail_window_reads_the_same_setup_as_the_whole_prefix():
    """struct_full reads only the last STRUCT_LOOKBACK bars (+14 for ATR), so slicing the tail
    must not change the synthesized tag."""
    rng = np.random.default_rng(3)
    c = 100 + np.cumsum(rng.normal(0, 1, 300))
    df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c})
    hs = indicators.struct_full(df.iloc[:120])
    for i in range(60, 300, 11):
        full = mtf.synthesize(hs, indicators.struct_full(df.iloc[:i]), float(c[i - 1]))
        tail = mtf.synthesize(hs, indicators.struct_full(df.iloc[i - P._STRUCT_TAIL:i]),
                              float(c[i - 1]))
        assert full["tag"] == tail["tag"], i


# ── universe ─────────────────────────────────────────────────────────────────────────
def _daily(vol, n=200, start="2024-01-01"):
    ts = pd.bdate_range(start, periods=n)
    c = np.full(n, 100.0)
    return pd.DataFrame({"ts": ts, "open": c, "high": c * (1 + vol), "low": c * (1 - vol),
                         "close": c})


def test_the_volatility_rank_uses_only_data_before_as_of():
    """A name that turns volatile AFTER the ranking date must not be ranked on it."""
    calm, wild = _daily(0.01), _daily(0.01)
    wild.loc[150:, "high"] = 130.0
    wild.loc[150:, "low"] = 70.0
    cut = wild["ts"].iloc[150]
    r = P.vol_rank({"CALM": calm, "WILD": wild}, as_of=cut, n=2, window=100)
    assert abs(r.set_index("symbol").loc["WILD", "atr%"] -
               r.set_index("symbol").loc["CALM", "atr%"]) < 1e-9
    r2 = P.vol_rank({"CALM": calm, "WILD": wild}, as_of=None, n=1, window=100)
    assert r2["symbol"].iloc[0] == "WILD"


def test_a_fresh_listing_is_not_ranked_on_its_first_weeks():
    r = P.vol_rank({"IPO": _daily(0.2, n=30), "OLD": _daily(0.01)}, n=2, window=120)
    assert list(r["symbol"]) == ["OLD"]


def test_room_buckets_match_the_boards_thresholds():
    assert P.room_bucket(float("inf")) == "✅ Has room"
    assert P.room_bucket(1.0) == "✅ Has room"
    assert P.room_bucket(0.99) == "⚠ Tight" and P.room_bucket(0.5) == "⚠ Tight"
    assert P.room_bucket(0.49) == "🧱 Capped"
    assert P.room_bucket(float("nan")) == "n/a"


# ── the walk over a bundle ───────────────────────────────────────────────────────────
def _bundle(hz="btst", sigs=None, n=20, members=None, per_day=6):
    o, h, l, c = _flat(n)
    ts = []
    for d in pd.bdate_range("2025-01-06", periods=n // per_day + 1):
        ts += [d + pd.Timedelta(hours=9 + k) for k in range(per_day)]
    ts = np.array(ts[:n], dtype="datetime64[ns]")
    s = [{"i": i, "ts": ts[i], "side": "LONG", "tag": "T", "level": np.nan, "atr": 2.0,
          "spot": 100.0, "up_wall": "", "up_gap": np.inf, "room": "✅ Has room"}
         for i in (sigs or [])]
    return {"sig": {"X": {"o": o, "h": h, "l": l, "c": c, "ts": ts,
                          "sess_last": P._session_last(ts), "sig": s}},
            "members": members, "meta": {"hz": hz}}


def test_a_signal_on_the_last_bar_is_PENDING_and_never_counted():
    t = P.walk_bundle(_bundle(sigs=[19]), rr=2.0, max_bars=3, cost_bps=0.0)
    assert list(t["status"]) == ["PENDING"]
    assert P.summarize(t) == {}


def test_an_intraday_signal_on_the_sessions_last_bar_is_skipped():
    t = P.walk_bundle(_bundle(hz="intraday", sigs=[5]), rr=2.0, max_bars=3, cost_bps=0.0)
    assert t.empty


def test_a_name_outside_that_months_top20_does_not_trade():
    m = {pd.Period("2025-01", "M"): {"OTHER"}}
    assert P.walk_bundle(_bundle(sigs=[2], members=m), max_bars=3, cost_bps=0.0).empty
    m = {pd.Period("2025-01", "M"): {"X"}}
    assert len(P.walk_bundle(_bundle(sigs=[2], members=m), max_bars=3, cost_bps=0.0)) == 1


def test_the_room_filter_decides_which_signals_are_TAKEN():
    b = _bundle(sigs=[2, 8])
    b["sig"]["X"]["sig"][0]["room"] = "🧱 Capped"
    t = P.walk_bundle(b, max_bars=3, cost_bps=0.0, room="✅ Has room")
    assert list(t["signal_ts"]) == [pd.Timestamp(b["sig"]["X"]["ts"][8])]


def test_session_clustering_deflates_t_when_trades_move_together():
    """Twenty names stopping out on one gap-down morning are one event, not twenty."""
    rows = []
    for k, d in enumerate(pd.bdate_range("2025-01-01", periods=30)):
        r = 1.0 if k % 2 else -0.6
        rows += [{"entry_ts": d, "R": r}] * 10              # 10 perfectly correlated trades/day
    t = pd.DataFrame(rows)
    from eqbtst import paper
    assert abs(P.clustered_t(t)) < abs(paper.expectancy(t["R"])["t"]) / 2


def test_the_fast_trigger_levels_equal_sr_levels_levels_kind():
    rng = np.random.default_rng(11)
    c = 100 + np.cumsum(rng.normal(0, 1, 300))
    df = pd.DataFrame({"open": c, "high": c + rng.uniform(.3, 1.5, 300),
                       "low": c - rng.uniform(.3, 1.5, 300), "close": c})
    from eqbtst import config
    for i in range(80, 300, 9):
        w = df.iloc[i - 60:i]
        a = indicators.atr(w, 14)
        fast = indicators.walls_kind(w["high"].to_numpy(), w["low"].to_numpy(),
                                     config.SR_TOL_ATR * a)
        slow = indicators.sr_levels(w, lookback=60).get("levels_kind", [])
        assert fast == slow, i


def test_pivot_flags_reproduce_pivots_inside_any_window():
    """The S/R lens slices pivots computed once instead of re-running pivots() per bar."""
    rng = np.random.default_rng(1)
    for trial in range(200):
        n = int(rng.integers(25, 90))
        h = 100 + np.cumsum(rng.normal(0, 1, n))
        l = h - rng.uniform(0, 2, n)
        if trial % 5 == 0:
            h[rng.integers(0, n)] = np.nan
        if trial % 7 == 0:
            h[10:20] = h[10]
            l[10:20] = h[10]
        if trial % 3 == 0:
            h, l = np.round(h), np.round(l)
        fh, fl = indicators.pivot_flags(h, l)
        k = np.arange(n)
        for lo, hi in ((0, n), (3, n - 1), (5, n)):
            his, los = indicators.pivots(h[lo:hi], l[lo:hi])
            m = (k >= lo + 2) & (k < hi - 2)
            assert list(h[fh & m]) == his and list(l[fl & m]) == los, (trial, lo, hi)


# ── the trade log's plain-language columns ───────────────────────────────────────────
def _closed_rows():
    return pd.DataFrame({
        "side": ["LONG", "LONG", "SHORT", "LONG"],
        "entry": [100.0, 100.0, 100.0, 100.0], "exit": [102.0, 98.0, 97.0, 100.05],
        "R": [0.9, -1.1, 1.4, -0.05], "why": ["target", "stop", "gap-target", "time"],
        "entry_ts": pd.to_datetime(["2026-09-24 10:00", "2026-09-24 11:00",
                                    "2026-09-25 10:00", "2026-09-25 11:00"])})


def test_win_means_money_made_AFTER_cost_not_target_hit():
    """A time exit 5 paise up is a gross gain and a net LOSS; the log must say LOSS."""
    v = P.trade_view(_closed_rows(), cost_bps=22.0)
    assert list(v["result"]) == ["✅ WIN", "❌ LOSS", "✅ WIN", "❌ LOSS"]
    assert v["pnl_rs"].iloc[3] > 0 and v["pnl_pct"].iloc[3] < 0


def test_pnl_pct_is_signed_by_side_and_net_of_cost():
    v = P.trade_view(_closed_rows(), cost_bps=22.0)
    assert abs(v["pnl_pct"].iloc[0] - (2.0 - 0.22)) < 1e-9
    assert abs(v["pnl_pct"].iloc[2] - (3.0 - 0.22)) < 1e-9, "a short gains when price falls"
    assert v["exit_how"].iloc[2] == "🚀 Gapped past target"


def test_day_scoreboard_counts_wins_and_losses_per_session():
    d = P.day_scoreboard(P.trade_view(_closed_rows(), cost_bps=22.0)).set_index("day")
    assert d.loc[pd.Timestamp("2026-09-24").date(), "wins"] == 1
    assert d.loc[pd.Timestamp("2026-09-25").date(), "losses"] == 1
    assert d.loc[pd.Timestamp("2026-09-25").date(), "day_result"] == "✅ Profit day"


# ── Sml-Intraday: 5m trigger / 15m confirm / 1h upper ────────────────────────────────
def _session_5m(days=2):
    ts = []
    for d in pd.bdate_range("2025-01-06", periods=days):
        ts += list(pd.date_range(d + pd.Timedelta("9h15min"), d + pd.Timedelta("15h25min"),
                                 freq="5min"))
    n = len(ts)
    px = 100 + np.sin(np.arange(n) / 7.0)
    return pd.DataFrame({"ts": ts, "open": px, "high": px + 0.3, "low": px - 0.3,
                         "close": px, "volume": 1.0})


def test_sml_intraday_is_5m_15m_with_a_1h_upper_frame():
    h = P.HORIZONS["sml_intraday"]
    assert (h["ltf"], h["htf"], h["up"]) == ("5m", "15m", "1h")
    assert P.SOURCE["sml_intraday"] == "5m" and P.SQUARE_OFF["sml_intraday"]
    assert P.HORIZON_ORDER[0] == "sml_intraday"
    assert "sml_intraday" not in mtf.PRESETS, "the live board has no 5m frame"


def test_sml_frames_are_resampled_from_the_5m_source_and_close_on_time():
    f = _session_5m(2)
    fr = P.build_frames("sml_intraday", None, f)
    assert len(fr["5m"]) == 150 and len(fr["15m"]) == 50
    # 15m built from 5m has the same OHLC a native 15m bar would
    first = f.iloc[:3]
    b = fr["15m"].iloc[0]
    assert b["open"] == first["open"].iloc[0] and b["close"] == first["close"].iloc[-1]
    assert b["high"] == first["high"].max() and b["low"] == first["low"].min()
    # the day's last 1h bar is the stub-merged 14:15 bar and closes at 15:30
    last_1h = fr["1h"][fr["1h"]["ts"].dt.date == fr["1h"]["ts"].dt.date.iloc[0]].iloc[-1]
    assert last_1h["ts"].strftime("%H:%M") == "14:15"
    assert last_1h["end"].strftime("%H:%M") == "15:30"
    # a 15m bar is invisible to the 5m bars inside it
    seen = np.searchsorted(fr["15m"]["end"].to_numpy(), fr["5m"]["end"].to_numpy(), side="right")
    assert list(seen[:6]) == [0, 0, 1, 1, 1, 2]
