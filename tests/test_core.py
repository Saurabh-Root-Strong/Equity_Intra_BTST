"""Offline unit tests — no DB, no network. Guard the LOCKED logic + no-lookahead."""
import io
import re
import datetime as dt

import numpy as np
import pandas as pd

from eqbtst import features, config


def _bars(n=40, sym="X"):
    d = pd.date_range("2025-01-01", periods=n, freq="B")
    base = pd.DataFrame({
        "trade_date": d, "symbol": sym,
        "prev_close": 100.0, "open_price": 100.0, "high_price": 101.0,
        "low_price": 99.0, "close_price": 100.0, "avg_price": 100.0,
        "ttl_trd_qnty": 1000, "deliv_qty": 400, "deliv_per": 40.0,
        "turnover_lacs": 1.0, "no_of_trades": 10,
    })
    return base


def _flat_nifty(df):
    """A flat index so rs_idx == stock ret (index return 0)."""
    return pd.DataFrame({"trade_date": df["trade_date"].unique(), "close_val": 20000.0})


def test_clr_and_body():
    df = _bars()
    df.loc[df.index[-1], ["low_price", "high_price", "close_price", "open_price"]] = [90, 100, 99, 91]
    df = features.add_features(df)
    r = df.iloc[-1]
    assert abs(r["clr"] - 0.9) < 1e-9      # close near high
    assert r["body"] > 0                    # green candle


def test_rolling_median_is_causal():
    """vol_ratio for day t must not use day t's own volume in its baseline."""
    df = _bars(n=30)
    df.loc[df.index[-1], "ttl_trd_qnty"] = 10_000_000    # huge spike on last day
    df = features.add_features(df)
    # baseline median (denominator) is built from shift(1); the spike day's ratio is huge,
    # but the PRIOR day's ratio is unaffected by the future spike.
    assert df.iloc[-1]["vol_ratio"] > 100
    assert abs(df.iloc[-2]["vol_ratio"] - 1.0) < 1e-6


def test_signal_mask_requires_full_stack():
    df = _bars(n=30)
    i = df.index[-1]
    # set a perfect footprint on the last day
    df.loc[i, ["low_price", "high_price", "close_price", "open_price", "prev_close"]] = \
        [100, 105, 104.5, 100.5, 100]
    df.loc[i, "avg_price"] = 102.0          # close 104.5 is ~2.5% above VWAP -> path-persistent
    df.loc[i, "ttl_trd_qnty"] = 5000        # 5x the 1000 baseline
    df.loc[i, "turnover_lacs"] = 3000.0     # >= Rs20cr liquidity floor
    df.loc[i, "deliv_per"] = 75.0           # today's (post-hoc) delivery
    # prior days: high delivery (so trailing avg >= 60, the leak-free leg) + up-days for RS
    for k in range(2, 12):
        df.loc[df.index[-k], "close_price"] = 100.3
        df.loc[df.index[-k], "deliv_per"] = 70.0
    df = features.add_features(df)
    df = features.add_relative_strength(df, _flat_nifty(df))
    assert bool(features.signal_mask(df).iloc[-1]) is True
    # break ONE condition (weak close) -> no signal
    df2 = df.copy(); df2.loc[i, "clr"] = 0.3
    assert bool(features.signal_mask(df2).iloc[-1]) is False
    # break the PATH filter (closed at/below VWAP = spike-and-fade) -> no signal
    df3 = df.copy(); df3.loc[i, "close_vs_vwap"] = 0.0
    assert bool(features.signal_mask(df3).iloc[-1]) is False
    # break RELATIVE STRENGTH (persistent laggard) -> no signal
    df4 = df.copy(); df4.loc[i, "rs_idx_cum"] = -0.05
    assert bool(features.signal_mask(df4).iloc[-1]) is False
    # break TRAILING DELIVERY (no sustained accumulation) -> no signal
    df5 = df.copy(); df5.loc[i, "deliv_trail"] = 40.0
    assert bool(features.signal_mask(df5).iloc[-1]) is False


def test_signal_is_leak_free():
    """No signal leg may use a value unknowable at the 15:15 close. Delivery% lands
    ~6pm, so the delivery leg must be TRAILING (deliv_trail, through t-1), never
    today's deliv_per/deliv_spike."""
    import inspect
    src = inspect.getsource(features.signal_mask)
    assert "deliv_trail" in src
    assert "deliv_per" not in src and "deliv_spike" not in src   # today's delivery = look-ahead


def test_sector_cap():
    """N candidates in one sector must be capped; weights sum to 1."""
    from eqbtst import portfolio, data
    orig = data.load_sectors
    data.load_sectors = lambda: {
        "A": "Banking", "B": "Banking", "C": "Banking", "D": "IT", "E": "Pharma"}
    try:
        cand = pd.DataFrame({"symbol": ["A", "B", "C", "D", "E"],
                             "score": [5, 4, 3, 2, 1]})   # already sorted best-first
        book = portfolio.select(cand)
    finally:
        data.load_sectors = orig
    assert (book["sector"] == "Banking").sum() <= config.MAX_PER_SECTOR
    assert "C" not in set(book["symbol"])            # 3rd Banking name dropped
    assert abs(book["weight"].sum() - 1.0) < 1e-6


def test_size_multiplier_scales_book():
    """The self-calibrated multiplier governs GROSS exposure: weights sum to it,
    0 = stand aside (empty book), never levers above 1.0. Closes the improving loop."""
    from eqbtst import portfolio, data
    orig = data.load_sectors
    data.load_sectors = lambda: {"A": "IT", "B": "Pharma", "C": "Auto"}
    try:
        cand = pd.DataFrame({"symbol": ["A", "B", "C"], "score": [3, 2, 1]})
        full = portfolio.select(cand, size_mult=1.0)
        half = portfolio.select(cand, size_mult=0.5)
        aside = portfolio.select(cand, size_mult=0.0)
        over = portfolio.select(cand, size_mult=1.8)
    finally:
        data.load_sectors = orig
    assert abs(full["weight"].sum() - 1.0) < 1e-3           # ~equal-weight (4dp rounding)
    assert abs(half["weight"].sum() - 0.5) < 1e-3           # gross scales with the multiplier
    assert aside.empty and aside.attrs["gross_exposure"] == 0.0   # 0 -> stand aside
    assert abs(over["weight"].sum() - 1.0) < 1e-3           # never levers above full backtest


def test_volume_pace_is_time_normalised():
    """The raw cum-volume/median-daily ratio is TIME-BIASED (only ~4% of a day's volume
    has traded by 09:15), so a genuine 2x day could never clear the 2.0 gate before the
    close. volume_pace divides by the elapsed-volume fraction so 2.0 means the same at
    any hour — AND at/after 15:25 it must EQUAL the raw ratio, or the 8yr backtest (which
    used the full-day ratio) would be invalidated."""
    from eqbtst import indicators as I

    # profile is monotonic and terminal at 1.0
    assert I.day_fraction("09:15") < I.day_fraction("11:00") < I.day_fraction("14:00") < 1.0
    assert I.day_fraction("15:25") == 1.0 and I.day_fraction("15:30") == 1.0

    med = 1000.0                                  # 20d median DAILY volume
    for t, frac in [("09:30", I.day_fraction("09:30")), ("12:00", I.day_fraction("12:00"))]:
        cum = 2.0 * frac * med                    # a name exactly ON PACE for a 2x day
        c = pd.DataFrame({"open": [10], "high": [11], "low": [9], "close": [10],
                          "volume": [cum], "ts": [pd.Timestamp(f"2026-01-01 {t}")]})
        assert I.volume_surge(c, med) < 2.0                       # raw would FAIL the gate
        assert abs(I.volume_pace(c, med, t) - 2.0) < 1e-6         # pace reads 2.0 ✓

    # CLOSE-DECISION INTEGRITY: at 15:25 pace == raw (backtest untouched)
    c = pd.DataFrame({"open": [10], "high": [11], "low": [9], "close": [10],
                      "volume": [2500.0], "ts": [pd.Timestamp("2026-01-01 15:25")]})
    assert I.volume_pace(c, med, "15:25") == I.volume_surge(c, med) == 2.5


def test_vol_pace_clock_is_explicit_not_bar_ts():
    """REGRESSION GUARD. 'now' for the volume-pace normalisation must be the real clock,
    NEVER inferred from the last bar's timestamp — a bar's ts is its OPEN, so a 4h frame
    at a 15:15 cut would report 13:15 and inflate the pace ~1.5x (a fabricated volume
    surge, and a FALSE BTST-CARRY). Pace must be identical across timeframes at the same
    clock time."""
    from eqbtst import indicators as I

    coarse = pd.DataFrame({                      # a 4h frame: last bar OPENS 13:15
        "open": [100, 101], "high": [102, 103], "low": [99, 100], "close": [101, 102],
        "volume": [500, 500],
        "ts": [pd.Timestamp("2026-01-01 09:15"), pd.Timestamp("2026-01-01 13:15")]})
    fine = pd.DataFrame({                        # a 15m frame at the same 15:15 cut
        "open": [100, 101], "high": [102, 103], "low": [99, 100], "close": [101, 102],
        "volume": [500, 500],
        "ts": [pd.Timestamp("2026-01-01 15:00"), pd.Timestamp("2026-01-01 15:15")]})

    a = I.live_state(coarse, 100, ref_avg_day_vol=1000, now_hhmm="15:15")["vol_pace"]
    b = I.live_state(fine, 100, ref_avg_day_vol=1000, now_hhmm="15:15")["vol_pace"]
    assert a == b                                    # tf-invariant at the same clock
    assert abs(a - 1.0 / I.day_fraction("15:15")) < 0.01     # divides by the CUT's fraction
    # and the buggy behaviour (dividing by the 13:15 bar-open) must NOT recur
    assert a < 1.5 * (1.0 / I.day_fraction("15:15"))


def test_archive_staleness_guard():
    """If the nightly EOD sync is stale, the archive's ref_close is the WRONG session's
    close — and prev_close / vol_med20 / ATR / deliv_trail are all wrong with it, silently
    corrupting every signal. Cross-checking against the broker's live prev_close catches
    this with no trading calendar (holiday-proof). A handful of corporate actions must NOT
    trip it; a bulk disagreement must."""
    from eqbtst import live
    ref = pd.DataFrame({"ref_close": [100.0] * 30}, index=[f"S{i}" for i in range(30)])

    fresh = {f"NSE:S{i}-EQ": {"prev_close_price": 100.0} for i in range(30)}
    assert live.archive_health(ref, fresh)["stale"] is False

    stale = {f"NSE:S{i}-EQ": {"prev_close_price": 103.0} for i in range(30)}
    assert live.archive_health(ref, stale)["stale"] is True          # whole board disagrees

    corp = {f"NSE:S{i}-EQ": {"prev_close_price": (50.0 if i >= 28 else 100.0)}
            for i in range(30)}                                       # 2 splits/bonuses
    assert live.archive_health(ref, corp)["stale"] is False          # must not false-trip

    assert live.archive_health(ref, {})["stale"] is False            # no quotes -> no claim


def test_live_signal_equals_backtested_signal():
    """THE most important invariant in the project. The LIVE board's BTST-CARRY must fire
    on exactly the footprint features.signal_mask() was validated on for 8 years — or the
    board suggests names the backtest never blessed, and the edge you trade is not the edge
    you measured.

    This caught a real divergence: the live path was missing the PATH-SIGNATURE leg
    (close_vs_vwap, worth +26->+30bps and flips 2025 positive) and was using TODAY's RS
    burst instead of the RS_LOOKBACK-day PERSISTENT cumulative (+30 vs +19bps).
    """
    from eqbtst import live

    # every leg combination around each threshold — no live-only or backtest-only fires
    rows = []
    for clr in (0.69, 0.70, 0.85):
        for ret in (0.009, 0.010, 0.03):
            for vr in (1.9, 2.0, 3.0):
                for rsc in (-0.01, 0.0, 0.05):
                    for cv in (0.004, 0.005, 0.02):
                        rows.append((clr, ret, vr, rsc, cv))
    df = pd.DataFrame(rows, columns=["clr", "ret", "vol_ratio", "rs_idx_cum", "close_vs_vwap"])
    df["deliv_trail"] = 70.0                 # delivery + liquidity held constant & passing
    df["turnover_lacs"] = 5000.0

    truth = features.signal_mask(df).to_numpy()
    ready = np.array([live.btst_readiness({"clr": c}, 100 * r, rc, vr, cv)
                      for c, r, vr, rc, cv in zip(df.clr, df.ret, df.vol_ratio,
                                                  df.rs_idx_cum, df.close_vs_vwap)])
    live_sig = (ready >= live.BTST_LEGS) & (df.deliv_trail >= config.DELIV_TRAIL_TH).to_numpy() \
        & (df.turnover_lacs >= config.LIQ_MIN_LACS).to_numpy()

    assert (truth == live_sig).all(), "LIVE signal has diverged from the BACKTESTED signal"
    assert truth.sum() > 0 and live_sig.sum() > 0          # the test actually exercises fires

    # a missing session-VWAP must NOT be a free pass on the path-signature leg
    assert live.btst_readiness({"clr": 0.9}, 3.0, 0.05, 3.0, cvwap=None) < live.BTST_LEGS


def test_live_regime_gate_matches_backtest():
    """The regime gate is mandatory and load-bearing, so the LIVE gate must be the same
    rule the backtest validated: the SIGNAL DAY's own close vs its own 50-day MA. The
    archive only runs through yesterday, so an archive lookup gates today's trade on
    YESTERDAY's regime — wrong on ~9% of sessions, and wrong precisely at the turns."""
    from eqbtst import regime

    # a synthetic index: flat, then a jump that crosses the MA today
    nf = pd.DataFrame({"trade_date": pd.date_range("2025-01-01", periods=60, freq="B"),
                       "close_val": [100.0] * 60})
    truth = regime.nifty_regime(nf).dropna()
    assert not truth["up"].iloc[-1]                     # flat -> not above its own MA

    # TODAY closes well above the 50-day MA -> live gate must say risk-ON, even though
    # every archived close (yesterday and before) was flat.
    assert regime.is_risk_on_live(120.0, nf) is True
    # TODAY closes below -> risk-OFF
    assert regime.is_risk_on_live(80.0, nf) is False
    # no index price -> stand aside, never a free pass
    assert regime.is_risk_on_live(None, nf) is False

    # and it reproduces the backtest exactly when fed each day's own close
    ma50 = nf["close_val"].rolling(config.REGIME_MA).mean().iloc[-1]
    assert regime.is_risk_on_live(float(ma50) + 1, nf) is True
    assert regime.is_risk_on_live(float(ma50) - 1, nf) is False


def test_clr_bounded_when_broker_range_lags_ltp():
    """Seen LIVE: the broker's high/low momentarily lag the last trade, printing lp ABOVE
    high. clr = (c-l)/(h-l) then exceeds 1.0 and SPURIOUSLY passes the >=CLR_TH strong-close
    leg — a footprint fabricated from a stale tick. The LTP is by definition inside the day's
    range, so the range must be reconciled to it: clr stays in [0,1]."""
    from eqbtst import indicators as I

    # raw (inconsistent) quote: lp above high -> clr > 1 and the leg would wrongly pass
    raw = I.price_action(100.0, 102.0, 99.0, 103.0)["clr"]
    assert raw > 1.0 and raw >= config.CLR_TH          # the bug, reproduced

    # reconciled the way quotes_board now does it
    o, h, l, c = 100.0, 102.0, 99.0, 103.0
    h, l = max(h, c), min(l, c)
    assert 0.0 <= I.price_action(o, h, l, c)["clr"] <= 1.0

    # and the mirror case (lp below low)
    o, h, l, c = 100.0, 102.0, 99.0, 98.0
    h, l = max(h, c), min(l, c)
    assert 0.0 <= I.price_action(o, h, l, c)["clr"] <= 1.0


def test_long_only_locked():
    assert config.LONG_ONLY is True                  # short side proven dead (win 20%)


def test_indicators():
    from eqbtst import indicators as I
    assert I.rsi(list(range(1, 40))) > 90            # monotonic up -> RSI high
    assert I.rsi(list(range(40, 1, -1))) < 10        # monotonic down -> RSI low
    assert I.price_action(100, 105, 99.5, 104.8)["character"] == "marubozu_bull"
    assert I.price_action(100, 101, 90, 100.2)["character"] == "hammer"
    assert I.price_action(100, 110, 99, 101)["character"] == "shooting_star"
    assert abs(I.price_action(100, 102, 98, 101)["clr"] - 0.75) < 1e-6
    c = pd.DataFrame({"open": [100, 101, 102], "high": [101, 102, 103],
                      "low": [99, 100, 101], "close": [101, 102, 102.5],
                      "volume": [10, 20, 30]})
    ls = I.live_state(c, prev_close=100, ref_avg_day_vol=40, index_ret=0.005)
    assert ls["above_vwap"] and "rsi7" in ls and "rs_vs_index" in ls


def test_structure():
    from eqbtst import indicators as I
    up = pd.DataFrame({"open": range(1, 25), "high": [x + 0.5 for x in range(1, 25)],
                       "low": [x - 0.5 for x in range(1, 25)], "close": range(1, 25),
                       "volume": [10] * 24})
    assert I.structure(up) in ("BREAKOUT_UP", "TREND_UP")     # efficient up move
    rng = pd.DataFrame({"open": [10, 11] * 6, "high": [11.5] * 12, "low": [9.5] * 12,
                        "close": [10, 11] * 6, "volume": [10] * 12})
    assert I.structure(rng) == "RANGE"                        # choppy, no direction
    assert I.structure(pd.DataFrame({"open": [1], "high": [1], "low": [1],
                                     "close": [1], "volume": [1]})) == "n/a"


def test_atr_and_levels():
    from eqbtst import indicators as I
    c = pd.DataFrame({"open": [100, 101, 102, 101, 103], "high": [102, 103, 104, 103, 105],
                      "low": [99, 100, 101, 100, 102], "close": [101, 102, 103, 102, 104],
                      "volume": [10, 20, 30, 15, 25]})
    a = I.atr(c, 3)
    assert a > 0
    lv = I.levels(104, a, day_low=102)
    assert lv["entry"] == 104 and lv["stop"] < 104 < lv["t1"] < lv["t2"]
    assert lv["stop"] >= 102 - 1e-9              # structural stop (day-low) respected
    assert I.levels(100, 0) == {}               # no ATR -> no levels (no fabrication)


def test_earnings_guard():
    """upcoming() excludes names reporting within the hold window; empty calendar
    degrades gracefully to an empty set (no crash, no silent pass claim)."""
    import datetime as dt
    import pandas as pd
    from eqbtst import events
    ev = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"],
                       "date": [dt.date(2026, 7, 12), dt.date(2026, 7, 20), dt.date(2026, 7, 10)],
                       "purpose": ["Financial Results"] * 3})
    up = events.upcoming(dt.date(2026, 7, 10), horizon_days=3, events=ev)
    assert up == {"AAA"}                    # 12th is in (10, 13]; 20th too far; 10th not > asof
    assert events.upcoming(dt.date(2026, 7, 10), events=pd.DataFrame(
        columns=["symbol", "date", "purpose"])) == set()


def test_locked_thresholds():
    # tripwire: these are LOCKED by the 8yr validation. A change is a decision, not a typo.
    assert (config.CLR_TH, config.DELIV_TRAIL_TH, config.VOL_TH, config.RET_TH) == (0.70, 60.0, 2.0, 0.01)
    assert config.DELIV_TRAIL_WIN == 3            # trailing delivery window (leak-free)
    assert config.CVWAP_TH == 0.005           # path-signature refiner (flips 2025 positive)
    assert (config.RS_LOOKBACK, config.RS_MIN) == (10, 0.0)   # persistent-RS refiner
    assert config.LIQ_MIN_LACS == 2000.0                       # realism/liquidity floor
    assert config.MAX_HOLD == "overnight"     # never day-2


def test_calibrate_shrinks_and_only_cuts():
    """Self-calibration learns the KNOB: posterior shrinks toward the backtest prior,
    the size multiplier only CUTS below 1.0, a confidently-negative record -> 0, and
    a noisy sub-window never falsely zeroes a healthy edge."""
    import numpy as np
    import pandas as pd
    from eqbtst import calibrate, ledger

    mu0, tau = 20.0, 20.0
    # posterior is a proper convex blend of prior and data mean
    post, sd = calibrate._posterior(mu0, tau, m=40.0, s=115.0, n=50)
    assert mu0 < post < 40.0                     # strictly between prior and data
    assert sd < tau                              # evidence tightened the estimate
    assert calibrate._posterior(mu0, tau, 999.0, 115.0, 0)[0] == mu0   # n=0 -> prior unchanged

    def _mult(df):
        st = {"open": df.iloc[:0], "closed": df, "summary": {}}
        orig = ledger.state
        ledger.state = lambda path=None, _s=st: _s
        try:
            return calibrate.calibrate(persist=False)
        finally:
            ledger.state = orig

    def _synth(n, mean, seed):
        p = np.random.default_rng(seed).normal(mean, 115.0, n)
        return pd.DataFrame({"date": pd.date_range("2026-01-01", periods=n, freq="D"),
                             "symbol": "X", "entry_px": 100.0, "exit_px": 100.0,
                             "net_bps": p, "status": "CLOSED"})

    assert _mult(_synth(80, -30.0, 1))["size_multiplier"] == 0.0     # clearly negative -> stand aside
    assert _mult(_synth(300, 26.0, 2))["size_multiplier"] > 0.0      # healthy large sample not falsely killed
    assert _mult(_synth(3, 20.0, 3))["size_multiplier"] == calibrate._SIZE_FLOOR   # tiny n -> floor


# ── multi-timeframe HTF x LTF synthesis (eqbtst/mtf.py) ────────────────────────
def test_struct_full_matches_structure_label():
    """struct_full must be a pure superset of structure() — the label cannot drift when the
    box/ER context is added, or every existing filter silently changes meaning."""
    from eqbtst import indicators
    rng = np.random.default_rng(11)
    for _ in range(300):
        c = 100 + np.cumsum(rng.normal(0, 1, 60))
        d = pd.DataFrame({"open": c, "high": c + rng.random(60),
                          "low": c - rng.random(60), "close": c, "volume": 1})
        assert indicators.struct_full(d)["struct"] == indicators.structure(d)
    assert indicators.struct_full(pd.DataFrame({"close": [1.0, 2.0]}))["struct"] == "n/a"


def test_location_decides_whether_a_break_is_real():
    """The same LTF breakout is a RESOLUTION at the HTF box edge and a TRAP mid-box. This is
    the entire point of using two timeframes — if it ever stops holding, the tag is noise."""
    from eqbtst import mtf
    htf = {"struct": "RANGE", "hi": 110.0, "lo": 100.0, "n": 20}
    up = {"struct": "BREAKOUT_UP", "n": 20}
    dn = {"struct": "BREAKOUT_DOWN", "n": 20}
    assert mtf.synthesize(htf, up, 109.5)["tag"] == "RANGE-TOP BREAK"     # loc 0.95
    assert mtf.synthesize(htf, up, 105.0)["tag"] == "FALSE-BREAK TRAP"    # loc 0.50
    assert mtf.synthesize(htf, dn, 100.5)["tag"] == "RANGE-FLOOR BREAK"   # loc 0.05
    assert mtf.synthesize(htf, dn, 105.0)["tag"] == "FALSE-BREAK TRAP"


def test_sideways_is_not_a_squeeze():
    """REGRESSION: lumping plain RANGE in with CONSOLIDATION stamped 'NESTED SQUEEZE' on 85 of
    140 names on a live board. A squeeze needs a real volatility CONTRACTION on some frame."""
    from eqbtst import mtf
    box = {"hi": 110.0, "lo": 100.0, "n": 20}
    rng_ = {"struct": "RANGE", **box}
    coil = {"struct": "CONSOLIDATION", **box}
    assert mtf.synthesize(rng_, rng_, 105.0)["tag"] == "RANGE-BOUND (no setup)"
    assert mtf.synthesize(coil, rng_, 105.0)["tag"] == "NESTED SQUEEZE"
    assert mtf.synthesize(rng_, coil, 105.0)["tag"] == "NESTED SQUEEZE"
    assert mtf.synthesize(coil, coil, 105.0)["tag"] == "NESTED SQUEEZE"


def test_with_trend_continuation_and_warming_guards():
    from eqbtst import mtf
    trend = {"struct": "TREND_UP", "hi": 110.0, "lo": 100.0, "n": 20}
    coil20 = {"struct": "CONSOLIDATION", "n": 20}
    # A PULLBACK IS A MOVE AGAINST THE TREND, AND THE TAG NOW REQUIRES ONE TO HAVE HAPPENED.
    # 105 in a 100-110 box = loc 0.50: price genuinely retraced into the box -> textbook.
    assert mtf.synthesize(trend, coil20, 105.0)["tag"] == "WITH-TREND CONTINUATION"
    # 108 = loc 0.80: coiling at the CEILING. Nothing retraced, so it is not a pullback.
    assert mtf.synthesize(trend, coil20, 108.0)["tag"] == "COIL AT THE EXTREME"
    # The short mirror is the one that mattered: a downtrend coiling ON ITS FLOOR is a BASE,
    # and it measured -0.54% as a short (n=19,679, t=-7.47) -- 79% of every short the old
    # loc-blind tag served. 101 in a 100-110 box = loc 0.10.
    dn = {"struct": "TREND_DOWN", "hi": 110.0, "lo": 100.0, "n": 20}
    s = mtf.synthesize(dn, coil20, 101.0)
    assert s["tag"] == "COIL AT THE EXTREME" and s["dir"] == "DOWN"
    # ...while a real rally back into the box IS the textbook continuation short.
    assert mtf.synthesize(dn, coil20, 105.0)["tag"] == "WITH-TREND CONTINUATION"
    assert mtf.synthesize(trend, {"struct": "TREND_UP", "n": 20},
                          108.0)["tag"] == "EXTENDED (aligned)"
    assert mtf.synthesize(trend, {"struct": "TREND_DOWN", "n": 20},
                          108.0)["tag"] == "PULLBACK vs HTF"
    # too few closed bars must NEVER produce a tradeable-looking tag
    assert mtf.synthesize({"struct": "TREND_UP", "n": 3}, trend, 105.0)["tag"] == "HTF warming"
    assert mtf.synthesize(trend, {"struct": "n/a", "n": 0}, 105.0)["tag"] == "LTF warming"


def test_every_preset_is_nested_and_ranked():
    """A preset whose 'lower' frame is not below its 'higher' frame inverts the whole read."""
    from eqbtst import mtf
    order = {"15m": 15, "1h": 60, "2h": 120, "4h": 240, "1D": 1440, "1W": 10080}
    for k in mtf.PRESET_ORDER:
        p = mtf.PRESETS[k]
        assert order[p["ltf"]] < order[p["htf"]], k
    for tag in mtf.TAG_ICON:                      # every tag must be rankable and drawable
        assert tag in mtf.TAG_RANK, tag
    assert mtf.TAG_RANK["WITH-TREND CONTINUATION"] < mtf.TAG_RANK["FALSE-BREAK TRAP"]


def test_failed_archive_read_is_not_cached():
    """REGRESSION: _daily_hist cached {} on failure, pinning every name's 1D/1W structure to
    'n/a' for the WHOLE DAY after one transient DuckDB lock (DCM mid-sync). A lock must cost
    one scan, not a session."""
    from eqbtst import live
    live._DAILY_HIST.clear()
    orig = live.data.load_eod
    try:
        live.data.load_eod = lambda **k: (_ for _ in ()).throw(RuntimeError("db locked"))
        assert live._daily_hist() == {}
        assert not live._DAILY_HIST          # the failure must NOT be memoised
    finally:
        live.data.load_eod = orig
    live._DAILY_HIST.clear()


def test_daily_timeframe_is_not_silently_hourly():
    """REGRESSION: _RES had no '1D', so .get(tf,'60') fell through to SIXTY-MINUTE candles and
    labelled them daily — the positional entry, ATR stop and targets were built on hourly bars
    while the UI said 1D."""
    from eqbtst import live
    assert live._RES["1D"] == "D"
    assert live._LOOKBACK["1D"] <= 366       # Fyers rejects a longer daily range outright
    for tf in ("15m", "1h", "2h", "4h", "1D"):
        assert tf in live._RES, tf


# ── touch-counted dynamic S/R + corporate actions ─────────────────────────────
def test_corporate_action_back_adjustment():
    """REGRESSION: the EOD archive is UNADJUSTED. 26 of 268 names carry a split/bonus cliff,
    and 4 of the 5 with one inside the structure window were labelled a FAKE TREND_DOWN
    (TATAMOTORS read TREND_DOWN on a 40% demerger; post-event bars alone read RANGE)."""
    from eqbtst import indicators
    pre, post = 1000.0, 200.0                       # a 1:5 split, INSIDE the 20-bar window
    rng = np.random.default_rng(9)
    c = np.r_[pre + rng.normal(0, 5, 12), post + rng.normal(0, 1, 10)]
    df = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1})
    adj = indicators.adjust_corporate_actions(df)["close"].to_numpy(float)
    assert adj.max() / adj.min() < 1.10             # cliff gone
    assert abs(adj[-1] - c[-1]) < 1e-6              # scaled ONTO today's price, not away
    # unadjusted, the split masquerades as directional structure; adjusted, it cannot
    assert indicators.structure(df) in ("TREND_DOWN", "BREAKOUT_DOWN")
    adj_df = pd.DataFrame({"open": adj, "high": adj * 1.01, "low": adj * 0.99,
                           "close": adj, "volume": 1})
    assert indicators.structure(adj_df) not in ("TREND_DOWN", "BREAKOUT_DOWN")
    # a NORMAL move must never be "adjusted" away
    n = 100 + np.cumsum(np.random.default_rng(2).normal(0, 1, 40))
    nd = pd.DataFrame({"open": n, "high": n + 1, "low": n - 1, "close": n, "volume": 1})
    assert np.allclose(indicators.adjust_corporate_actions(nd)["close"], n)


def test_walls_count_touches_and_ignore_the_forming_bar():
    from eqbtst import indicators
    rng = np.random.default_rng(3)
    c = 100 + np.cumsum(rng.normal(0, 1, 60))
    h, l = c + rng.random(60), c - rng.random(60)
    h[-1] = h.max() + 8                              # a still-forming bar at a new extreme
    l[-1] = l.min() - 8
    w = indicators.walls(h, l, 0.6)
    assert not any(abs(x - h[-1]) < 0.6 for x, _ in w)   # cannot invent a level from it
    assert not any(abs(x - l[-1]) < 0.6 for x, _ in w)
    assert all(t >= 1 for _, t in w)
    # a level touched repeatedly must accumulate touches
    osc = np.tile([100.0, 110.0], 20)
    w2 = indicators.walls(osc + 1, osc - 1, 0.5)
    assert max(t for _, t in w2) >= 5


def test_sr_levels_degenerate_inputs_return_empty_not_garbage():
    from eqbtst import indicators
    def mk(c):
        c = np.asarray(c, float)
        return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1})
    assert indicators.sr_levels(mk(np.arange(5.0))) == {}          # too few bars
    assert indicators.sr_levels(mk(np.full(40, 100.0))) == {}      # flat: no ATR
    assert indicators.sr_levels(mk(np.zeros(40))) == {}            # zero price
    assert indicators.sr_levels(mk(np.arange(40.0) + 100)) == {}   # pure ramp: no pivots
    assert indicators.sr_levels(None) == {}


def test_every_preset_frame_can_fill_the_sr_window():
    """REGRESSION: the coarse frames are resampled from ONE 15m fetch, so its lookback decides
    how many bars they get. At 20 calendar days the 4h frame held 30 bars — BELOW the 40-bar
    S/R window — and 4h is the BTST preset's HTF and Swing's LTF, so two of four horizons were
    finding levels on a starved frame."""
    from eqbtst import live, mtf
    bars_per_session = {"15m": 25, "1h": 6.25, "2h": 3.1, "4h": 1.5}
    sessions = live._MTF_FETCH_DAYS * 5 / 7          # calendar days -> trading sessions
    for k in mtf.PRESET_ORDER:
        p = mtf.PRESETS[k]
        for frame in (p["ltf"], p["htf"]):
            if frame in bars_per_session:            # 1D/1W come from the archive, not this fetch
                assert bars_per_session[frame] * sessions >= 40, f"{k}/{frame} starves S/R"


def test_levels_track_live_price_and_flip_polarity():
    """REGRESSION: sup/res/headroom were computed at SCAN time and frozen while price ticked
    every 5s. As price approached a wall headroom stayed wide, and once price traded THROUGH
    a level the board still listed it as resistance overhead — on a 4h frame the scan can be
    hours old. The walls are past structure and rightly freeze; WHICH is nearest and HOW FAR
    are functions of live price and must follow the tick."""
    from eqbtst import live
    w = [(100.0, 3, "4h"), (110.0, 2, "4h"), (95.0, 1, "1h")]
    b = live._live_levels(pd.DataFrame(
        [{"symbol": "T", "ltp": px, "_wall_pair": w, "_sr_atr": 2.0}
         for px in (97.0, 99.8, 104.0, 112.0)]))
    assert b.loc[0, "res"] == 100.0 and b.loc[0, "res_t"] == 3      # wall is overhead
    assert b.loc[1, "at_wall"].startswith("RES 100.00")             # testing it RIGHT NOW
    # POLARITY FLIP: once price is above it, the same level is support — with its touches
    assert b.loc[2, "sup"] == 100.0 and b.loc[2, "sup_t"] == 3
    assert b.loc[3, "sup"] == 110.0 and b.loc[3, "headroom"] == np.inf   # clear road above
    # a live test must NOT inflate the touch count
    assert b.loc[1, "res_t"] != 4


def test_live_levels_survive_missing_inputs():
    from eqbtst import live
    b = live._live_levels(pd.DataFrame([
        {"symbol": "A", "ltp": None, "_wall_pair": [(10.0, 2, "1h")], "_sr_atr": 1.0},
        {"symbol": "B", "ltp": 100.0, "_wall_pair": [], "_sr_atr": 1.0},
        {"symbol": "C", "ltp": 100.0, "_wall_pair": [(90.0, 2, "1h")], "_sr_atr": np.nan},
    ]))
    assert b["headroom"].tolist() == [np.inf, np.inf, np.inf]
    assert (b["at_wall"] == "").all()
    assert live._live_levels(pd.DataFrame()).empty


def test_setup_side_is_direction_not_tag():
    """REGRESSION: setup tags are DIRECTION-BLIND. 'WITH-TREND CONTINUATION' is the textbook
    setup in either direction — a downtrend coiling for continuation is a SHORT and carries
    the identical tag. The long-side filter split on the tag, so on a live board it returned
    15 names of which 8 were actually short setups (53% wrong): bearish continuations served
    as buys, and pullbacks inside downtrends served as dips to buy."""
    from eqbtst import mtf
    up = {"struct": "TREND_UP", "hi": 110.0, "lo": 100.0, "n": 20}
    dn = {"struct": "TREND_DOWN", "hi": 110.0, "lo": 100.0, "n": 20}
    coil = {"struct": "CONSOLIDATION", "n": 20}
    su, sd = mtf.synthesize(up, coil, 105.0), mtf.synthesize(dn, coil, 105.0)
    assert su["tag"] == sd["tag"] == "WITH-TREND CONTINUATION"      # tag cannot tell them apart
    assert su["dir"] == "UP" and sd["dir"] == "DOWN"                # direction can
    assert mtf.side_of(su["tag"], su["dir"]) == "LONG"
    assert mtf.side_of(sd["tag"], sd["dir"]) == "SHORT"
    # a pullback inside a DOWNtrend is a short entry, never a dip-buy
    pb = mtf.synthesize(dn, {"struct": "TREND_UP", "n": 20}, 105.0)
    assert pb["tag"] == "PULLBACK vs HTF" and mtf.side_of(pb["tag"], pb["dir"]) == "SHORT"
    # traps and squeezes take no side at all
    for tag in mtf.AVOID_TAGS | mtf.WAIT_TAGS:
        assert mtf.side_of(tag, "UP") == "—", tag
    # every directional outcome must resolve to a side
    rng = {"struct": "RANGE", "hi": 110.0, "lo": 100.0, "n": 20}
    for h, l, spot in [(rng, {"struct": "BREAKOUT_UP", "n": 20}, 109.5),
                       (rng, {"struct": "BREAKOUT_DOWN", "n": 20}, 100.5),
                       (up, {"struct": "TREND_UP", "n": 20}, 105.0)]:
        s = mtf.synthesize(h, l, spot)
        assert mtf.side_of(s["tag"], s["dir"]) in ("LONG", "SHORT"), s["tag"]


def test_cash_short_horizons_are_flagged():
    """Indian cash equity has no overnight short — it must be squared off same day. Only the
    intraday horizon is reachable in cash; the rest need the futures leg."""
    from eqbtst import mtf
    assert mtf.SHORTABLE_IN_CASH["intraday"] is True
    assert not any(mtf.SHORTABLE_IN_CASH[k] for k in ("btst", "swing", "positional"))
    assert set(mtf.SHORTABLE_IN_CASH) == set(mtf.PRESET_ORDER)


def test_short_edge_decays_with_hold_length():
    """Measured on this universe (495,607 daily obs, 2018-2026; 43,042 down-structure days):
    the downtrend is an INTRADAY move (-4.4bps in-session) while the overnight gap runs
    +10.8bps AGAINST a short. So short P&L decays monotonically with hold length — the exact
    opposite of the long side. Guard the ordering so a future edit cannot quietly imply that
    holding a short longer is better."""
    from eqbtst import mtf
    e = mtf.SHORT_EDGE_BPS
    assert set(e) == set(mtf.PRESET_ORDER)
    assert e["intraday"] > e["btst"] > e["swing"]          # strictly worse the longer you hold
    assert e["positional"] <= e["swing"]
    assert e["intraday"] < 22.0        # even the best case is under the round-trip cost floor
    assert all(v < 22.0 for v in e.values())
    # only the horizon that can actually be squared off same day is cash-shortable
    assert mtf.SHORTABLE_IN_CASH["intraday"] and not mtf.SHORTABLE_IN_CASH["swing"]


def test_weekly_frame_completeness_rule():
    """A part-formed weekly bar spans fewer sessions, so its range is mechanically narrower —
    and the coil test compares the latest 3-bar span to the typical one. On a Monday (a 1-day
    'week') weekly CONSOLIDATION rose 15.3% -> 21.3% of the universe purely from missing days.

    REGRESSION on the FIX itself: 'last daily bar is before the Friday label' is not the same
    question as 'this week is unfinished'. W-FRI labels a group with its Friday whether or not
    that Friday traded, so every FRIDAY HOLIDAY discarded a COMPLETE Mon-Thu week — and it
    stayed discarded all of the following week."""
    from eqbtst import live

    def daily(a, b, skip=()):
        ds = [d for d in pd.date_range(a, b) if d.weekday() < 5 and d.date() not in skip]
        return pd.DataFrame({"ts": ds, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5})

    # mid-week -> the current week is genuinely unfinished, drop it
    d = daily("2026-05-04", "2026-07-15")                       # ends Wednesday
    w = live.weekly_frame(d, now=pd.Timestamp("2026-07-15"))
    assert w["ts"].iloc[-1].date() == dt.date(2026, 7, 10), "unfinished week must be dropped"

    # Friday traded -> that week is complete
    d = daily("2026-05-04", "2026-07-17")
    w = live.weekly_frame(d, now=pd.Timestamp("2026-07-17"))
    assert w["ts"].iloc[-1].date() == dt.date(2026, 7, 17)

    # FRIDAY HOLIDAY: last bar is Thursday, but by Monday the week is plainly over
    hol = dt.date(2026, 7, 17)
    d = daily("2026-05-04", "2026-07-17", skip=(hol,))          # last bar Thu 16 Jul
    w = live.weekly_frame(d, now=pd.Timestamp("2026-07-20"))    # the following Monday
    assert w["ts"].iloc[-1].date() == hol, "a complete Mon-Thu week must NOT be discarded"



def test_merged_walls_do_not_hide_a_stronger_level():
    """REGRESSION: the two frames are resampled from the SAME series, so the merged wall list
    is full of near-duplicates. Picking 'nearest by price' reported the DUPLICATE — a 3-touch
    wall at 100.00 sitting 0.05 from a 1-touch at 100.05 displayed as x1."""
    from eqbtst import live
    w = [(100.00, 3, "4h"), (100.05, 1, "1h"), (110.0, 2, "4h")]
    r = live._live_levels(pd.DataFrame(
        [{"symbol": "T", "ltp": 105.0, "_wall_pair": w, "_sr_atr": 2.0}]))
    assert r.loc[0, "sup"] == 100.00 and r.loc[0, "sup_t"] == 3
    # touches are MAXed, never summed — one swing seen twice is not two rejections
    assert r.loc[0, "sup_t"] != 4


def test_structure_label_is_scale_invariant():
    """The 8-year short study back-adjusted prices using a ratio only known LATER. That is
    leak-free only because a uniform rescale cannot change a label — ER, ATR and the breakout
    comparison all scale together. Guard it, or the study's validity silently changes."""
    from eqbtst import indicators
    rng = np.random.default_rng(1)
    for _ in range(100):
        c = 100 + np.cumsum(rng.normal(0, 1, 60))
        d = pd.DataFrame({"open": c, "high": c + rng.random(60),
                          "low": c - rng.random(60), "close": c, "volume": 1})
        d2 = d.copy()
        d2[["open", "high", "low", "close"]] *= rng.uniform(0.1, 10)
        assert indicators.structure(d) == indicators.structure(d2)


def test_short_side_anti_predictive_tags_are_recorded():
    """Measured by reconstructing the REAL pipeline (structure -> synthesize -> side_of)
    causally over 468,661 daily observations, 2018-2026: the SHORT side OUTPERFORMED the
    universe by +0.57% over 20 days. The side is inverted, not merely weak — in a rising
    market an extended or coiling downtrend is a bottoming pattern, so shorting it sells the
    low. Only RANGE-FLOOR BREAK underperformed (-1.09%)."""
    from eqbtst import mtf
    # every anti-predictive short tag must carry POSITIVE excess (positive = the short lost)
    for tag in mtf.SHORT_ANTI_PREDICTIVE:
        assert mtf.SIDE_EXCESS_20D[("SHORT", tag)] > 0, tag
    for tag in mtf.SHORT_VALIDATED:
        assert mtf.SIDE_EXCESS_20D[("SHORT", tag)] < 0, tag
    # the two sets must partition the directional short tags — no tag may be silently unrated
    short_tags = {t for (s, t) in mtf.SIDE_EXCESS_20D if s == "SHORT"}
    assert short_tags == mtf.SHORT_ANTI_PREDICTIVE | mtf.SHORT_VALIDATED
    assert not (mtf.SHORT_ANTI_PREDICTIVE & mtf.SHORT_VALIDATED)
    # long side measured positive on every tag (average only — it is sign-unstable by year)
    assert all(v > 0 for (s, _), v in mtf.SIDE_EXCESS_20D.items() if s == "LONG")


def test_price_band_default_is_user_owned_and_wired():
    """The price band is a position-SIZING preference, so the NUMBER belongs to the user and
    lives in config. What must not regress is the wiring and the visibility: the band cuts
    BEFORE the structure logic, so it decides what the horizon dropdown may even consider.
    Measured on a 243-name board, a Rs900 cap removed 137 names (56%) including RELIANCE /
    ICICIBANK / INFY, taking LONG candidates from 9 to 2 — fine when intended, a trap when
    forgotten. Hence: user owns the number, the board always states the cut."""
    import re
    from eqbtst import config as _c
    # the DEFAULT is the user's to choose — the test guards the WIRING, not the number
    assert hasattr(_c, "PRICE_MAX_DEFAULT") and hasattr(_c, "PRICE_MIN_DEFAULT")
    src = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    m = re.search(r'key="price_max"', src)
    assert m, "price_max widget missing"
    blk = src[max(0, m.start() - 300):m.end()]
    assert "config.PRICE_MAX_DEFAULT" in blk, "widget must read the user's config default"
    # 0 must always mean 'no cap', whatever number the user picks
    import pandas as _pd
    assert float(_c.PRICE_MAX_DEFAULT) >= 0


def test_universe_scan_has_no_ttl():
    """REGRESSION: _uni_scan carried ttl=1800, which contradicted its own design comment. Once
    the entry expired, the NEXT interaction of any kind paid a 30-60s cold re-scan — so typing
    in the price box kicked off a full universe fetch while the previous frame stayed on
    screen, making the filter look like it had done nothing (Max 900 -> 10000 should widen the
    board from 106 to 229 names; it appeared unchanged). The nonce must be the ONLY trigger."""
    import re
    src = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    m = re.search(r"@st\.cache_data\(([^)]*)\)\s*\ndef _uni_scan", src)
    assert m, "_uni_scan decorator not found"
    assert "ttl" not in m.group(1), "a TTL re-introduces surprise cold re-scans on filter edits"


def test_history_fetch_is_rate_paced():
    """REGRESSION: the universe scan issued ~250 /history calls at ~450 req/min and Fyers
    replied HTTP 429 'request limit reached' for a slice of them. fetch_intraday turns a 429
    into an EMPTY frame, which mtf_structure turns into 'n/a' — so 48 of 243 names (RELIANCE,
    TCS — not thin names) had NO intraday structure and vanished from every HTF/LTF filter
    while the board looked complete. Measured, 120 names, clean window each time:
        6 workers unpaced -> 450 req/min -> 7 x 429
        3 workers, 0.35s  -> 170 req/min -> 0 failures
        2 workers, 0.25s  -> 174 req/min -> 0 failures
    The budget is a ROLLING window a burst POISONS: paced run 65s after a burst failed 63/120,
    the same run from a clean window failed 0 — which is why the 1s retry sweep never helped.
    With the pacer the full scan returns 0 blanks (86s vs ~40s)."""
    import inspect
    from eqbtst import live as _l
    assert hasattr(_l, "_hist_pace"), "the /history rate pacer is gone"
    assert 0.3 <= _l._HIST_GAP <= 0.6, "gap must keep the scan under ~200 req/min"
    src = inspect.getsource(_l.fetch_intraday)
    assert "_hist_pace()" in src, "fetch_intraday must pace before every /history call"
    # the retry sweep must wait for the rolling window to actually roll, not 1 second
    scan = inspect.getsource(_l.universe_mtf_scan)
    m = re.search(r"time\.sleep\(([\d.]+)\)", scan)
    assert m and float(m.group(1)) >= 10, "retry sweep must outlast the poisoned rate window"


def test_headroom_is_measured_in_the_trigger_frames_atr():
    """headroom exists to answer 'does my 1xATR target sit beyond a defended level?'. That
    target and its stop are built from the LOWER (trigger) frame's ATR, so normalising the
    distance by the HIGHER frame's ATR quoted it in a unit 1.6x-2.4x too large and made every
    distance read that much too small. Measured on the live board, the '< 0.5 = buying INTO a
    wall' warning fired on 100/195 (intraday), 93/188 (BTST), 64/173 (swing), 100/183
    (positional) — about half the universe; in the trigger frame's own ATR, ~a quarter."""
    import inspect
    from eqbtst import live as _l
    src = inspect.getsource(_l.add_setup)
    m = re.search(r'b\["_sr_atr"\]\s*=\s*b\[f"sr_atr\{(\w+)\}"\]', src)
    assert m, "_sr_atr assignment not found"
    assert m.group(1) == "ltf", "headroom/at_wall must use the TRIGGER frame's ATR"


def test_wall_merge_does_not_chain_away_the_stronger_level():
    """REGRESSION (DALBHARAT, Swing, live): walls 1794.6 x2, 1799.6 x2, 1804.0 x1 with a 5.6
    tolerance. Comparing each candidate against the cluster ANCHOR absorbed the middle wall,
    after which 1804.0 measured 9.4 from the anchor, survived as its own 'level', and was
    displayed as support x1 — 4.4 away from a wall that had just been declared the same level.
    Absorbing a member must not shrink the cluster's reach."""
    import pandas as pd
    from eqbtst import live as _l
    b = pd.DataFrame([{"ltp": 1823.2, "_sr_atr": 27.94,
                       "_wall_pair": [(1794.6, 2, "1D"), (1799.6, 2, "1D"), (1804.0, 1, "4h")]}])
    out = _l._live_levels(b)
    assert out["sup"].iloc[0] == 1794.6, "the anchor must be the cluster's strongest member"
    assert out["sup_t"].iloc[0] == 2, "touches are MAXed across the cluster, never understated"


def test_session_stub_is_folded_into_the_previous_bar():
    """NSE trades 375 minutes, which 60 and 120 do not divide, so binning from 09:15 ends every
    day with a bar built from ONE 15-minute candle (15:15-15:30) that the pipeline then treats
    as a full 1h/2h bar. The structure classifier compares bar RANGES and normalises by ATR, so
    a quarter-length bar is a different random variable wearing the same label. Measured over
    70 names x 60 days: 1h 24/70 names changed structure label once folded (ATR 6% low); 2h
    26/70 changed, ATR 26% low, and 14 of those flips were CONSOLIDATION -> RANGE — the stub
    was MANUFACTURING coils. 4h is untouched: its 13:15 bar holds 135 of 240 minutes."""
    from eqbtst import live
    day = pd.Timestamp("2026-07-17 09:15")
    ts = [day + pd.Timedelta(minutes=15 * i) for i in range(25)]      # a full 09:15-15:30 session
    f = pd.DataFrame({"ts": ts, "open": 100.0, "high": 101.0, "low": 99.0,
                      "close": 100.0, "volume": 10})
    f.loc[24, ["high", "low", "close"]] = [110.0, 98.0, 109.0]        # the 15:15 candle
    h1 = live._resample_ohlcv(f, "60min")
    assert len(h1) == 6, "1h must be 6 bars/day, not 7 with a 15-minute impostor"
    assert h1["ts"].iloc[-1].strftime("%H:%M") == "14:15"
    assert h1["high"].iloc[-1] == 110.0 and h1["close"].iloc[-1] == 109.0, "stub data must survive"
    assert h1["volume"].sum() == f["volume"].sum(), "no volume may be lost in the fold"
    assert len(live._resample_ohlcv(f, "120min")) == 3
    assert len(live._resample_ohlcv(f, "240min")) == 2, "4h's 13:15 bar is real — leave it alone"
    # 15m is exact (375/15) and must pass through untouched
    assert len(live.merge_session_stubs(f, 15)) == len(f)


def test_broker_bars_get_the_same_stub_treatment_as_our_resample():
    """The broker's own 60-minute series has the identical trailing stub. If only one path is
    corrected, the board's structure label and the trade card's ATR stop are computed on two
    different definitions of '1h' — measured on RELIANCE: broker TREND_UP / ATR 7.84 vs our
    resample BREAKOUT_UP / ATR 8.02, same name, same minute."""
    import inspect
    from eqbtst import live
    src = inspect.getsource(live.fetch_intraday)
    assert "merge_session_stubs" in src, "broker bars must be folded too, or the paths diverge"


def test_structure_and_levels_read_the_same_adjusted_series():
    """sr_levels back-adjusts corporate actions internally; struct_full and band_pct do not. A
    split inside the 60-day intraday window would therefore put the LEVELS on the adjusted
    series and the STRUCTURE LABEL on the raw one. The daily archive is adjusted upstream,
    which is why 1D/1W were never exposed and the intraday frames were. Adjust once at the
    fetch instead — and double-adjustment must be a verified no-op, or fixing this breaks 1D."""
    import inspect
    from eqbtst import live, indicators
    assert "adjust_corporate_actions" in inspect.getsource(live.mtf_structure)
    raw = pd.DataFrame({"ts": pd.date_range("2026-01-01", periods=30, freq="D"),
                        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000})
    raw.loc[20:, ["open", "high", "low", "close"]] /= 2.0
    a1 = indicators.adjust_corporate_actions(raw)
    a2 = indicators.adjust_corporate_actions(a1)
    assert np.allclose(a1["close"], a2["close"]), "adjusting twice must not re-scale"


def test_failed_quote_chunk_is_counted_not_swallowed():
    """A /quotes batch is FIFTY names. The old `except: continue` dropped the whole chunk
    silently, and because PHASE 1 builds rows only from what the quote returned, those names
    were never rows at all — n_scanned already excluded them, so the board looked complete at
    a smaller size. Same class as the /history 429 hole, wider blast radius (it is the FIRST
    fetch, so everything downstream inherits the gap)."""
    import inspect
    from eqbtst import live as _l
    src = inspect.getsource(_l._fetch_quotes)
    assert "_QUOTE_GAP" in src, "a dropped chunk must be counted"
    assert "for attempt in" in src, "a dropped chunk must be retried once"
    scan = inspect.getsource(_l.universe_mtf_scan)
    assert "n_quote_gap" in scan, "the scan must report the gap to the UI"
    dash = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    assert "n_quote_gap" in dash, "the UI must SAY when names are missing entirely"


def test_repaint_is_declared_per_preset():
    """The trigger bar is still printing, so the tag can relabel until it closes — and the
    board never said so. Measured by replaying whole sessions on 49 names, rebuilding BOTH
    frames from candles truncated to each 15-minute checkpoint: Intraday passes through 3-5
    tags per session, its midday tag differs from its closing tag 57% of the time, and it
    only settles after 92% of the session has elapsed. BTST 2-3 / 53% / 79%. The daily-trigger
    presets cannot repaint intraday at all. That ordering is the point — the faster the
    trigger frame, the more provisional the read — so it must not silently invert."""
    from eqbtst import mtf as _m
    assert set(_m.REPAINT) == set(_m.PRESETS), "every preset must declare its repaint risk"
    order = [_m.REPAINT[p]["midday_differs"] for p in _m.PRESET_ORDER]
    assert order == sorted(order, reverse=True), "repaint risk must fall as the hold lengthens"
    assert _m.REPAINT["swing"]["midday_differs"] == 0, "a closed daily bar cannot repaint"
    dash = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    assert "PROVISIONAL" in dash, "the UI must warn when the tag can still change"


def test_census_describes_the_market_not_your_price_band():
    """The census exists to show what the OTHER names are doing, so a rare setup is
    distinguishable from an empty tape. It was already fixed once (it used to be taken after
    the setup filter). But `light` is cut by the PRICE BAND the moment the board loads, so
    with a Rs900 cap it read '6 setup types across all 106 scanned names' while 243 were
    scanned — describing 44% of the market and calling it 'all'. A price cap is a
    position-SIZING choice; it must not decide what the tape is doing."""
    src = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    m = re.search(r"_census = live\.add_setup\((\w+(?:\[[^\]]*\])?)", src)
    assert m, "census must be built with add_setup over an unfiltered board"
    assert "sc[" in m.group(1), "census must come from the SCAN, not the price-filtered board"


def test_clear_road_is_rendered_not_blank():
    """live._set stores +inf when there is NO multi-touch wall overhead, precisely so 'nothing
    above you' stays distinguishable from 'not computed'. A NumberColumn printed inf as an
    empty cell — identical to missing — so the column's own tooltip promised a symbol the
    table could never show. 51 of 243 names (21%) on the BTST preset were silently ambiguous."""
    src = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    assert "∞ clear" in src, "infinite headroom must render as an explicit answer"
    m = re.search(r'"headroom": st\.column_config\.(\w+)', src)
    assert m and m.group(1) == "TextColumn", "headroom must be text so ∞ survives to the screen"


def test_at_wall_and_the_sr_columns_cannot_contradict():
    """at_wall deliberately has NO min-distance filter — its job is 'price is ON a level right
    now'. sup/res applied 0.25*ATR to EVERY wall. So the instant at_wall fired, the level it
    named was BY CONSTRUCTION excluded from sup/res and the table showed the next level back.
    Measured: they disagreed on 100% of firings on every preset (28/28, 29/29, 27/27, 31/31),
    which on screen reads as two columns contradicting each other while the hidden one is the
    level that matters most. A >=2-touch wall is never a micro-swing, however close it sits."""
    import pandas as pd
    from eqbtst import live as _l
    b = pd.DataFrame([{"ltp": 100.0, "_sr_atr": 4.0,
                       "_wall_pair": [(99.9, 3, "1h"), (95.0, 2, "4h"), (104.0, 2, "4h")]}])
    out = _l._live_levels(b)
    assert out["sup"].iloc[0] == 99.9, "a 3-touch wall 0.025 ATR away is the support, not noise"
    assert out["sup_t"].iloc[0] == 3
    assert out["at_wall"].iloc[0].startswith("SUP 99.9"), "at_wall must name that same level"


def test_short_side_is_ranked_by_measurement_not_by_textbook():
    """TAG_RANK is a CHARTIST ranking and on the short side it is inverted. WITH-TREND
    CONTINUATION is TAG_RANK 0 — the best-looking setup — and measured +0.47% excess against
    a short (n=20,220): it LOST. RANGE-FLOOR BREAK, the only tag that worked short (-1.09%),
    sorted BELOW it. On the live board 88/95/100/50% of SHORT names were anti-predictive tags
    and the top ten rows were 10-of-10 WITH-TREND CONTINUATION on three presets, with the one
    validated name buried underneath — while the warning box above said 'never short these'.
    People act on order, not on prose."""
    from eqbtst import mtf as _m
    # THE EVIDENCE IS HORIZON-SPECIFIC AND IT INVERTS. Re-measured at the hold each preset
    # actually trades (entry at the next open, 499,387 rows): RANGE-FLOOR BREAK is the BEST
    # multi-day short (+33.0 relative) and the WORST next-day one (-40.6 excess, 1 of 9 years
    # positive, -162.8bps in 2020) -- a fresh breakdown BOUNCES. WITH-TREND CONTINUATION is
    # the exact mirror: best next-day (+15.6), worst multi-day (+7.8). Ranking every preset
    # off the 20-day study put the single worst name on the page at the top of the two
    # fastest boards.
    # CORRECTED after auditing my own numbers: the first table was measured with a PROXY for
    # the board's tags (one frame's label + loc vs a prior-20-day range) rather than the real
    # synthesize(). The real RANGE-TOP BREAK also requires the HIGHER frame to be a range and
    # turned out to be only 43% of the proxy's rows. Re-measured faithfully, TWO OF THREE
    # SHORT NUMBERS CHANGED SIGN and the board was calling WITH-TREND CONTINUATION the best
    # short geometry (+15.6) when it is -8.7. Guard the corrected facts.
    #
    # On the session AFTER the signal, NO structural short tag beats shorting at random:
    for t, v in _m.SHORT_EVIDENCE["intraday"].items():
        assert v < 0, f"{t} must not read as a tradeable intraday short"
        _v = _m.short_verdict(t, "intraday")
        assert "Worse than shorting" in _v or "BOUNCES" in _v
    # a fresh breakdown BOUNCES on the next session -- the one robust short-side result,
    # and it is a warning, not a trade (-41.3 excess, and it survived the proxy at -40.6)
    assert _m.SHORT_EVIDENCE["intraday"]["RANGE-FLOOR BREAK"] <= -35
    assert "BOUNCES" in _m.short_verdict("RANGE-FLOOR BREAK", "intraday")
    # ...yet it is the LEAST-BAD short at every longer hold, which is the horizon inversion
    for p in ("btst", "swing", "positional"):
        assert _m.short_rank("RANGE-FLOOR BREAK", p) < _m.short_rank("COIL AT THE EXTREME", p)
    # no multi-day short may be described as working: ABSOLUTE 20-day P&L is negative for
    # every tag, because the baseline itself is -122bps. "Excess" over that is not profit.
    for p in ("swing", "positional"):
        for t, v in _m.SHORT_EVIDENCE[_m._LONG_HOLD[p]].items():
            assert "LOSES" in _m.short_verdict(t, p) or v < 0
    for hold, preset in (("intraday", "intraday"), ("overnight", "btst"),
                         ("5d", "swing"), ("20d", "positional")):
        tags = sorted(_m.SHORT_EVIDENCE[hold], key=lambda t: _m.short_rank(t, preset))
        ex = [_m.SHORT_EVIDENCE[hold][t] for t in tags]
        assert ex == sorted(ex, reverse=True), f"{hold} rank must descend with measured excess"
    dash = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    assert "short_rank(t, preset)" in dash, "the SHORT tab must rank at the SELECTED horizon"


def test_long_side_is_ranked_at_the_selected_hold():
    """The long side inverts with the horizon the same way the short side does, in the
    opposite direction. Measured, entry at the next open, 493,987 rows, as excess over
    buying a random name:

        tag                      overnight  intraday    5d      20d    yrs+ (overnight)
        RANGE-TOP BREAK            +32.1     -13.1    -24.7    +28.0    9 of 9
        WITH-TREND CONT. (dip)      +4.4      -4.6    +36.3    +69.8    5 of 9
        COIL AT THE EXTREME         +2.3      -1.3     +9.8    +51.5    3 of 9

    RANGE-TOP BREAK is the best OVERNIGHT long (+49.4bps absolute, positive in ALL NINE
    YEARS) and the WORST 5-day one (-7.3bps absolute — it loses money if held). TAG_RANK
    ranks continuation first, which is right for Swing/Positional and backwards for BTST."""
    from eqbtst import mtf as _m
    assert _m.long_rank("RANGE-TOP BREAK", "btst") < _m.long_rank("WITH-TREND CONTINUATION", "btst")
    for p in ("swing", "positional"):
        assert _m.long_rank("WITH-TREND CONTINUATION", p) < _m.long_rank("RANGE-TOP BREAK", p)
    # The session AFTER the gap gives it back. Guard the ABSOLUTE number, not the excess:
    # re-measured faithfully, WITH-TREND CONTINUATION is +10.5 EXCESS on that window, which
    # sounds like an edge until you add the -10.2bps baseline and get +0.3 absolute. Excess
    # against a negative baseline is exactly the trap that made the first short table wrong.
    for t, v in _m.LONG_EVIDENCE["intraday"].items():
        assert v + _m._INTRADAY_BASE <= 1.0, f"{t} must not read as a tradeable intraday long"
        _v = _m.long_verdict(t, "intraday")
        assert "gives it back" in _v, "must say the gap is already gone"
        assert "on the day" in _v, "must quote the ABSOLUTE day return, not the excess"
    # holding past the first night must never be sold as free: the 5d baseline IS the
    # overnight baseline (+17.4 vs +17.3), so RANGE-TOP BREAK goes outright negative there
    assert _m.LONG_EVIDENCE["5d"]["RANGE-TOP BREAK"] < 0
    assert "LOSES money" in _m.long_verdict("RANGE-TOP BREAK", "swing")
    for hold, preset in (("overnight", "btst"), ("5d", "swing"), ("20d", "positional")):
        tags = sorted(_m.LONG_EVIDENCE[hold], key=lambda t: _m.long_rank(t, preset))
        ex = [_m.LONG_EVIDENCE[hold][t] for t in tags]
        assert ex == sorted(ex, reverse=True), f"{hold} long rank must follow the measurement"
    dash = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    assert "long_rank(t, preset)" in dash, "the LONG tab must rank at the SELECTED horizon"


def test_a_dead_token_is_diagnosed_not_blamed_on_the_clock():
    """The scan-failure branch used to choose its message from market_open() alone: open ->
    "transient, just hit re-scan"; closed -> "your token may be stale". That is inverted for
    the case that happens EVERY TRADING MORNING. The Fyers token expires ~06:00 IST, three
    hours before the 09:15 open, so the normal first failure of the day is a DEAD TOKEN WHILE
    THE MARKET IS OPEN — and the board told the user it was transient and to keep pressing a
    button that can never fix it. The cause must be diagnosed from token_status(), not the
    clock, and the auto-retry must not fire on a dead token."""
    src = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    # anchor on the SCAN's failure branch specifically — the file has three `except _e` blocks
    i = src.find("DIAGNOSE THE CAUSE, DO NOT INFER IT FROM THE CLOCK")
    assert i > 0, "scan-failure branch not found"
    blk = src[i:i + 2600]
    tok = blk.find("live.token_status()")
    retry = blk.find('st.session_state["_uni_retried"] = True')
    assert tok > 0, "the failure branch must ask the token why it failed"
    assert retry > tok, "the token must be checked BEFORE the auto-retry burns a rerun"
    assert "re-authenticate" in blk, "a dead token must say re-authenticate"
    assert "will not fix it" in blk, "must say that re-scanning cannot fix an expired token"


def test_a_forming_bar_cannot_manufacture_a_coil():
    """A coil is a COMPLETED observation. "The range is contracting" cannot be said from a bar
    five minutes old — and the coil test is exactly a span comparison, so a part-printed newest
    bar makes the latest 3-bar span mechanically narrow. Measured by replaying a session at
    wall-clock checkpoints (4h frame, 60 names): coil fired on 7% of the board at 12:20, then
    30% at 13:20 the moment the 13:15 bar opened holding ONE candle, decaying 30->28->23->20%
    as it filled. Excluding the forming bar it is a flat 7%. A 4x spike out of nothing, and the
    fourth instance of one defect family here after the original coil detector, the partial
    weekly bar and the trailing session stub."""
    import pandas as pd
    from eqbtst import indicators
    # The mechanism is DISPLACEMENT: the window keeps the last 20 bars, so a brand-new sliver
    # pushes a full-width bar OUT of the trailing 3-bar span. Build exactly that — a wide
    # history, two recent narrower bars, then a bar that has barely printed.
    def bar(i, hi, lo):
        return {"ts": pd.Timestamp("2026-07-20 09:15") + pd.Timedelta(hours=i),
                "open": 100.0, "high": hi, "low": lo, "close": 100.0, "volume": 1000}
    rows = [bar(i, 104.0, 96.0) for i in range(18)]          # typical span 8
    rows += [bar(18, 101.0, 99.0), bar(19, 101.0, 99.0)]     # two narrower bars, span 2
    rows.append(bar(20, 100.1, 99.9) | {"volume": 10})       # the forming bar: barely printed
    c = pd.DataFrame(rows)
    live_read = indicators.struct_full(c, forming=True)
    naive = indicators.struct_full(c, forming=False)
    assert naive["struct"] == "CONSOLIDATION", "the sliver bar should fool the naive test"
    assert live_read["struct"] != "CONSOLIDATION", "a forming bar must not manufacture a coil"
    # the BREAKOUT/TREND tests must still see the live bar — that is the point of a live board
    up = c.copy()
    up.loc[up.index[-1], ["high", "low", "close"]] = [130.0, 129.0, 129.5]
    assert indicators.struct_full(up, forming=True)["struct"] == "BREAKOUT_UP"


def test_only_live_intraday_frames_are_marked_forming():
    """1D/1W come from the EOD archive and are complete by construction, so they must never be
    truncated; and off-hours even the intraday frames are closed."""
    import inspect
    from eqbtst import live as _l
    src = inspect.getsource(_l.mtf_structure)
    assert "_live_bar = market_open()" in src, "the flag must be gated on the session"
    assert 'forming=True' in src
    i1d = src.find('_set("1D"')
    assert i1d > 0 and "forming" not in src[i1d:i1d + 120], "the daily frame is never forming"


def test_the_structure_explainer_matches_the_code():
    """The "how structure is computed" panel is the page's own documentation, and it had drifted
    from the code it describes: it still said the fetch was 20 calendar days (it is 60), that 1h
    gives ~6.5 bars/day and 2h ~3.5 (the session-stub merge makes them exactly 6 and 3), and it
    defined the coil as "3-bar range < 60% of the prior range" — which is the ORIGINAL BUG, a
    short window compared against a long one, fixed long ago. It also omitted 15m entirely, the
    trigger frame of the Intraday preset. Documentation that contradicts the code is worse than
    none: it is believed."""
    from eqbtst import live as _l
    src = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    i = src.find("How structure is computed")
    assert i > 0
    blk = src[i:i + 6000]
    assert f"{_l._MTF_FETCH_DAYS} calendar days" in blk, "fetch window must match _MTF_FETCH_DAYS"
    assert "| **15m** |" in blk, "the Intraday preset's trigger frame must appear in the table"
    # NSE 09:15-15:30 is 375 min; after folding the trailing stub these are exact, not approx
    for frame, per_day in (("1h", 6), ("2h", 3), ("4h", 2), ("15m", 25)):
        assert f"| **{frame}** | {per_day} " in blk, f"{frame} must say {per_day} bars/day"
    assert "median" in blk, "the coil must be defined against the MEDIAN span, not 'the prior range'"
    assert "ignores the bar still forming" in blk, "the live coil rule must be documented"


def test_the_structure_tabs_actually_tick():
    """The LONG/SHORT/No-side tabs are what the structure board OPENS on, and they had no
    fragment at all — the ltp column sat frozen at scan time while its own tooltip promised
    "LIVE, refreshed every 5s on every tab". Measured after wiring: 196 of 243 prices moved
    within 6 seconds, `loc` re-derived on 79 names and the nearest support/resistance on 17,
    while the setup tag changed on 0 — which is the point. Price and everything that is a
    FUNCTION of price ticks; the 20-bar boxes, wall lists and ATRs stay pinned until a bar
    closes, because that is the only thing that can change them."""
    import inspect
    from eqbtst import live as _l
    assert hasattr(_l, "refresh_light_prices"), "the structure lane needs a price refresh"
    src = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    i = src.find("def _side_tabs")
    assert i > 0, "the side tabs must live in a fragment"
    assert 'run_every="5s"' in src[max(0, i - 400):i], "the tabs fragment must tick at 5s"
    body = src[i:i + 900]
    assert "refresh_light_prices" in body and "add_setup" in body, \
        "the tick must re-price AND re-derive loc/side/levels from the new price"
    # one batch quote, not a re-scan: the expensive half must stay pinned
    assert "universe_mtf_scan" not in body, "a 5s tick must never trigger a full re-scan"
    # and the quote fetch must be concurrent, since it now runs on a 5-second loop
    assert "ThreadPoolExecutor" in inspect.getsource(_l._fetch_quotes)


def test_sr_tolerance_lets_touch_counts_accumulate():
    """The touch count IS the value of the S/R feature — "how many times price rejected here".
    The old tol=0.2*ATR was tighter than a single day's range, so two rejections a week apart
    at what a chartist calls ONE level almost never landed within tolerance: measured 67% of
    all levels came out 1-touch, i.e. the count never accumulated. A human reads a level as a
    zone ~2-3% wide (≈0.6-1.0 ATR on a daily frame), not 0.2. Reproduces the MOTILALOFS case
    the user drew by hand: three swing lows ~1% apart that the eye calls one support."""
    import numpy as np, pandas as pd
    from eqbtst import indicators, config
    assert 0.4 <= config.SR_TOL_ATR <= 1.0, "the zone must be a chartist's width, not a hair"
    assert config.SR_LOOKBACK >= 50, "levels persist longer than the 20-bar regime window"
    # A FLAT base ~845 (so the ONLY swing lows are the three we plant), rejecting three times
    # at 812 / 816 / 820 — half an ATR apart, exactly what the eye calls one support zone.
    rows = [{"ts": pd.Timestamp("2026-05-01") + pd.Timedelta(days=i),
             "open": 845, "high": 852, "low": 838, "close": 845, "volume": 1000}
            for i in range(60)]
    for k, px in ((12, 812.0), (30, 816.0), (48, 820.0)):
        rows[k] = {**rows[k], "low": px, "high": px + 6, "close": px + 3}
    c = pd.DataFrame(rows)
    tight = indicators.sr_levels(c, spot=880.0, tol_atr=0.2)
    wide = indicators.sr_levels(c, spot=880.0)             # config default (0.6)
    def touches_near(sr, target):
        return max((t for x, t in sr.get("levels", []) if abs(x - target) <= 12), default=0)
    assert touches_near(tight, 816) == 1, "at 0.2 ATR the three touches stay split (the bug)"
    assert touches_near(wide, 816) == 3, "at the chartist width they are one 3-touch wall"


def test_both_clustering_sites_share_one_tolerance():
    """sr_levels clusters pivots into walls; live._live_levels re-clusters the merged HTF+LTF
    list against live price. If they used different tolerances the second would re-split what
    the first merged, and the displayed touch count would silently disagree with the wall list
    it was computed from. Both must read config.SR_TOL_ATR."""
    import inspect
    from eqbtst import live as _l, indicators
    assert "config.SR_TOL_ATR" in inspect.getsource(_l._live_levels)
    src = inspect.getsource(indicators.sr_levels)
    assert "config.SR_TOL_ATR" in src and "config.SR_LOOKBACK" in src
    assert "tol_atr: float = 0.2" not in src, "the old hardcoded 0.2 default must be gone"


def test_archive_staleness_uses_an_independent_calendar():
    """data.last_trading_date() returns the archive's OWN max date, so it cannot detect that
    the archive is stale — a mirror is not a calendar. Fyers' daily bars ARE the trading
    calendar, so archive_staleness compares the two and flags the gap. Measured live
    2026-07-26: archive 07-23, Fyers had the 07-24 session (MOTILALOFS -7.3%), so the board
    was showing a bullish tag on pre-crash data with price already through its 'support'. It
    must exclude today's still-forming bar (that is not a stale session) and must never block
    the board when Fyers is unreachable."""
    import inspect
    from eqbtst import live as _l
    assert hasattr(_l, "archive_staleness")
    src = inspect.getsource(_l.archive_staleness)
    assert "fetch_intraday" in src, "must use Fyers as the independent calendar, not the archive"
    assert "< today" in src or "< today)" in src, "must exclude today's forming bar"
    # the failure path must be graceful: ok=False, no exception bubbles
    s = _l.archive_staleness()
    assert set(s) >= {"stale_days", "archive_date", "market_date", "ok"}
    assert isinstance(s["stale_days"], int) and s["stale_days"] >= 0
    dash = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    assert "archive_staleness()" in dash and "behind the market" in dash, \
        "the UI must warn when the archive is behind"


def test_setup_tag_shows_trend_direction():
    """WITH-TREND CONTINUATION (and COIL/EXTENDED/PULLBACK) are direction-agnostic — the same
    tag is a LONG in an uptrend and a SHORT in a downtrend. On the SHORT tab, and in the census
    which has no side column at all, the bare tag hid that. The display now appends the HTF
    trend arrow (↑ up / ↓ down) from `dir`, and the census groups by (tag, direction) so a
    bull-flag group and a bear-base group under the same tag are never merged with one read."""
    import pandas as pd
    from eqbtst import dashboard as D
    df = pd.DataFrame([
        {"setup": "WITH-TREND CONTINUATION", "dir": "DOWN"},
        {"setup": "WITH-TREND CONTINUATION", "dir": "UP"},
        {"setup": "NESTED SQUEEZE", "dir": "NONE"},
    ])
    out = D._fmt(df)["setup"].tolist()
    assert out[0].endswith("↓"), "downtrend continuation must show ↓"
    assert out[1].endswith("↑"), "uptrend continuation must show ↑"
    assert not out[2].endswith(("↑", "↓")), "a NONE-direction tag gets no arrow"
    # the census must group by (tag, dir): the two continuations are DIFFERENT rows
    src = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    assert '_census = live.add_setup(sc["board"]' in src and '"dir"]].copy()' in src, \
        "census must carry dir"
    assert '_census["setup"] + _census["dir"]' in src, "census must key on (setup, dir)"


def test_sr_cluster_never_spans_the_price():
    """A cluster must stay on ONE side of price — a wall below is a floor, above is a ceiling.
    MPHASIS live (BTST): a run 2315(x4) 2336(x4) 2361(x1) 2388(x4) 2412(x2), each pair within
    tolerance, single-linkage chained into ONE cluster anchored at 2315.28 (BELOW price 2332.90),
    so a x4 RESISTANCE at 2388 was absorbed into a SUPPORT at 2315 and `res` showed the weaker
    far 2412(x2) with a stronger x4 hidden below it. Clustering each side of price separately
    makes that impossible: sup = nearest-and-strongest BELOW, res = nearest ABOVE."""
    import pandas as pd
    from eqbtst import live as _l
    b = pd.DataFrame([{"ltp": 2332.90, "_sr_atr": 45.37,
                       "_wall_pair": [(2315.28, 4, "4h"), (2336.25, 4, "1h"), (2361.80, 1, "1h"),
                                      (2388.28, 4, "4h"), (2412.65, 2, "1h")]}])
    o = _l._live_levels(b).iloc[0]
    assert o["sup"] < 2332.90 < o["res"], "sup must be below price, res above — never spanning it"
    assert o["res"] == 2336.25, "res must be the nearest ABOVE wall, not a far weak one across a chain"
    assert o["sup"] == 2315.28


def test_sr_wall_exactly_at_price_is_the_resistance_at_wall_names():
    """HAVELLS live: ltp 1238.00 with a 2-touch wall at exactly 1238.00. at_wall classifies a
    wall at price as RES (x >= px), but sup/res used strict x < px / x > px, so a wall exactly
    at price fell through both and at_wall named a level absent from sup/res. The boundary now
    matches: x >= px goes to the ceiling, so res == the level at_wall reports."""
    import pandas as pd
    from eqbtst import live as _l
    b = pd.DataFrame([{"ltp": 1238.00, "_sr_atr": 6.07,
                       "_wall_pair": [(1231.43, 3, "1h"), (1238.0, 2, "1h"), (1246.9, 1, "4h")]}])
    o = _l._live_levels(b).iloc[0]
    assert o["res"] == 1238.0 and o["res_t"] == 2, "wall at price must be the resistance"
    assert o["at_wall"].startswith("RES 1238"), "at_wall must name it, consistent with res"
    assert o["sup"] == 1231.43, "the floor below stays the support"


def test_big_wall_surfaces_the_higher_frame_level_the_pair_is_blind_to():
    """sup/res/headroom read only the two frames of the horizon. A 1h/4h long can sit right
    under a DAILY resistance the pair never looked at, buy into it, and reverse — the classic
    'traded the small frames, the big level capped it' loss (SIEMENS live: pair headroom '∞
    clear' with a 4-touch 1D wall 0.29 ATR overhead). `big_wall` surfaces the nearest defended
    (≥2-touch) 1D/1W wall in the trade's direction; `big_gap` is the distance in trigger ATR."""
    import numpy as np, pandas as pd
    from eqbtst import live as _l
    # a BTST (1h/4h) LONG: 4h uptrend, 1h coiling mid-box -> WITH-TREND CONTINUATION UP
    row = {"symbol": "TST", "ltp": 100.0,
           "s1h": "CONSOLIDATION", "box_h1h": 102, "box_l1h": 98, "box_n1h": 20,
           "s4h": "TREND_UP", "box_h4h": 110, "box_l4h": 90, "box_n4h": 20,
           "sr_wall1h": [(99.0, 2)], "sr_wall4h": [(101.0, 1)], "sr_atr1h": 2.0,
           "sr_wall1D": [(105.0, 3), (95.0, 2)], "sr_wall1W": [(120.0, 4)]}
    b = _l.add_setup(pd.DataFrame([row]), ltf="1h", htf="4h").iloc[0]
    assert b["side"] == "LONG", "setup should be a long here"
    assert b["big_wall"] == "1D 105.00 ×3", "must surface the nearest >=2-touch 1D wall ABOVE a long"
    assert abs(b["big_gap"] - 2.5) < 1e-6, "gap = (105-100)/trigger ATR 2.0 = 2.5"

    # ONE FRAME ONLY: BTST (HTF 4h) checks 1D and IGNORES 1W, even when 1W sits NEARER. The
    # relevant bigger frame scales with the hold -- an overnight BTST cares about the daily,
    # not the weekly. Here 1W has a wall at 103 (nearer) but the daily one at 108 is what shows.
    one = dict(row, sr_wall1D=[(108.0, 3)], sr_wall1W=[(103.0, 4)])
    bo = _l.add_setup(pd.DataFrame([one]), ltf="1h", htf="4h").iloc[0]
    assert bo["big_wall"] == "1D 108.00 ×3", "BTST must use ONLY its one frame (1D), not the nearer 1W"

    # a SHORT looks DOWN: nearest defended 1D/1W wall BELOW price
    rs = dict(row, s4h="TREND_DOWN", box_h4h=110, box_l4h=90, ltp=100.0)
    bs = _l.add_setup(pd.DataFrame([rs]), ltf="1h", htf="4h").iloc[0]
    assert bs["side"] == "SHORT"
    assert bs["big_wall"] == "1D 95.00 ×2", "a short must look at the FLOOR below, not the ceiling"

    # POSITIONAL (htf=1W): the one frame above the weekly is the MONTH — so it checks 1M,
    # not "none". All four horizons get a higher frame: Intraday->4h, BTST->1D, Swing->1W,
    # Positional->1M. A row carrying a monthly wall above a long must surface it.
    # HTF 1W trending up + LTF 1D coiling mid-box -> WITH-TREND CONTINUATION UP -> LONG
    prow = dict(row, s1W="TREND_UP", box_h1W=110, box_l1W=90, box_n1W=20,
                s1D="CONSOLIDATION", box_h1D=104, box_l1D=96, box_n1D=20,
                sr_atr1D=2.0, sr_wall1M=[(106.0, 4)])
    bp = _l.add_setup(pd.DataFrame([prow]), ltf="1D", htf="1W").iloc[0]
    assert bp["side"] == "LONG", f"expected LONG, got {bp['setup']}/{bp['side']}"
    assert bp["big_wall"] == "1M 106.00 ×4", "Positional must check the MONTHLY frame above 1W"
    # graceful when no monthly wall exists
    bn = _l.add_setup(pd.DataFrame([dict(prow, sr_wall1M=[])]), ltf="1D", htf="1W").iloc[0]
    assert bn["big_wall"] == "" and np.isinf(bn["big_gap"])


def test_upper_tf_room_filter_wired():
    """The big-wall was display-only; the user wanted a SELECTION on it — show only setups the
    one-frame-up S/R confirms. The 'Upper-TF room' filter keeps 'Room above' (big_gap ∞ or ≥1
    ATR — the higher frame is not capping the trade) or 'Capped' (big_gap <0.5 — a defended
    higher-frame wall right in the trade's direction). Composes with Setup quality: Long-side +
    Room above = confirmed longs with a clear higher frame. Filters on the NUMERIC big_gap
    (inf=clear), before _fmt stringifies it."""
    import numpy as np, pandas as pd
    src = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    assert 'key="mtf_roomf"' in src, "the Upper-TF room selectbox must exist"
    assert '"✅ Has room"' in src and '"🧱 Capped"' in src
    # the filter must run on the numeric big_gap (inf for clear), not the rendered string
    assert "np.isinf(_bg) | (_bg >= 1.0)" in src, "Room = clear or >=1 ATR"
    assert "_bg < 0.5" in src, "Capped = <0.5 ATR"
    # behavioural: the thresholds partition as intended
    bg = pd.to_numeric(pd.Series([np.inf, 0.3, 0.8, 1.5, 0.1]), errors="coerce")
    room = [x for x in bg[np.isinf(bg) | (bg >= 1.0)]]
    capped = [x for x in bg[bg < 0.5]]
    assert np.inf in room and 1.5 in room and 0.8 not in room, "Room = inf or >=1.0"
    assert 0.3 in capped and 0.1 in capped and 0.8 not in capped, "Capped = <0.5; marginal excluded"


def test_enriched_view_carries_and_splits_by_setup_side():
    """The MTF-setup layer (side, big_wall, big_gap) was added to add_setup/the light board and
    the pre-filter tabs, but enrich_mtf never carried it — so the POST-filter enriched view lost
    those columns and split its LONG/SHORT tabs by the intraday footprint `action` instead of
    the setup `side`, with a dump-all fallback. Off-hours (no footprint) that fallback fired
    every time, showing NESTED SQUEEZE / RANGE-BOUND (no-side) rows under the LONG tab. enrich
    must carry side/big_wall/big_gap and the tabs must split on side, consistent with the filter
    the user selected on."""
    import inspect
    from eqbtst import live as _l
    esrc = inspect.getsource(_l.enrich_mtf)
    for col in ('"side"', '"big_wall"', '"big_gap"'):
        assert col in esrc, f"enrich_mtf must carry {col} so the enriched view keeps it"
    dash = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    i = dash.find("def _struct_panel")
    body = dash[i:i + 2500]
    assert 'bb["side"] == "LONG"' in body, "enriched LONG tab must split on the setup side"
    assert 'bb["side"] == "SHORT"' in body, "enriched SHORT tab must split on the setup side"
    assert "else bb" not in body.split("bb[\"side\"] == \"LONG\"")[0][-200:], \
        "the dump-all fallback (show every match under LONG) must be gone"


def test_big_gap_ticks_live_like_headroom():
    """big_gap (distance to the one-frame-up wall) is a function of the LIVE price, exactly like
    headroom, which the 5s tick already re-derives. In the enriched view refresh_prices ticked
    headroom but left big_gap FROZEN — a stale number beside a live price. The wall PRICE is a
    1D/1W/1M level that cannot repaint intraday, so it stays; only the gap to it moves. add_setup
    now carries _big_wall_px and refresh_prices re-derives the gap from it."""
    import inspect
    from eqbtst import live as _l
    assert "_big_wall_px" in inspect.getsource(_l.add_setup), "add_setup must carry the wall price"
    rp = inspect.getsource(_l.refresh_prices)
    assert '"big_gap"' in rp and "_big_wall_px" in rp, "refresh_prices must re-derive big_gap on the tick"
    # numeric check: gap shrinks as price nears the wall
    import numpy as np
    bpx, atr = 108.0, 2.0
    assert round(abs(bpx - 100.0) / atr, 2) == 4.0 and round(abs(bpx - 104.0) / atr, 2) == 2.0


def test_auto_refresh_cadence_is_trigger_frame_aware():
    """Auto-refresh must re-scan on the TRIGGER (Lower TF) frame's bar close, not blindly every
    15 min. Intraday(15m) fires 4x/hr; BTST(1h) once/hr at :15; Swing(4h) at 13:15 & the close;
    Positional(1D) never intraday. Replicates the _bar_bucket math in dashboard._auto_rescan and
    guards that the source still resolves the trigger from persisted session_state (the widgets
    are defined later in the script, so referencing them directly would NameError)."""
    import datetime as dt, inspect
    from eqbtst import dashboard as _d
    src = inspect.getsource(_d)
    assert 'st.session_state.get("mtf_preset"' in src, "trigger must come from persisted preset key"
    assert 'st.session_state.get("mtf_ltf"' in src, "custom trigger must come from persisted ltf key"

    _LTF_MIN = {"15m": 15, "1h": 60, "2h": 120, "4h": 240, "1D": 1440}

    def bucket(tf, t):
        m = _LTF_MIN[tf]
        if m >= 1440:
            return f"{t:%Y-%m-%d}"
        e = (t.hour * 60 + t.minute) - (9 * 60 + 15)
        return f"{t:%Y-%m-%d}:{max(0, e) // m}"

    def rescans(tf):
        seen, n = set(), 0
        for hm in ("09:16", "10:15", "11:15", "13:15", "14:15", "15:15"):
            t = dt.datetime.strptime("2026-07-27 " + hm, "%Y-%m-%d %H:%M")
            b = bucket(tf, t)
            if b not in seen:
                n += 1
                seen.add(b)
        return n

    # over those 6 checkpoints: 15m rolls every time (6), 1h every hour (6 here all on :15),
    # 4h only at 09:16 & 13:15 (2), 1D once for the whole day (1)
    assert rescans("15m") == 6
    assert rescans("1h") == 6
    assert rescans("4h") == 2
    assert rescans("1D") == 1
