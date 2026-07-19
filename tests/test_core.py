"""Offline unit tests — no DB, no network. Guard the LOCKED logic + no-lookahead."""
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
    assert mtf.synthesize(trend, {"struct": "CONSOLIDATION", "n": 20},
                          108.0)["tag"] == "WITH-TREND CONTINUATION"
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


def test_weekly_frame_drops_an_incomplete_week():
    """REGRESSION: a part-formed weekly bar spans fewer sessions, so its range is mechanically
    narrower — and the coil test compares the latest 3-bar span to the typical one. On a
    Monday (a 1-day 'week') weekly CONSOLIDATION rose 15.3% -> 21.3% of the universe purely
    from missing days. Same class as the original coil bug."""
    import inspect
    from eqbtst import live
    src = inspect.getsource(live.mtf_structure)
    assert 'w = w.iloc[:-1]' in src and 'd["ts"].max() < w["ts"].iloc[-1]' in src
    # the guard must only fire when the week is genuinely unfinished
    d = pd.DataFrame({"ts": pd.to_datetime(["2026-07-13", "2026-07-14"])})
    w = pd.DataFrame({"ts": pd.to_datetime(["2026-07-17"])})          # week ENDS Friday
    assert d["ts"].max() < w["ts"].iloc[-1]                            # mid-week -> drop
    d2 = pd.DataFrame({"ts": pd.to_datetime(["2026-07-17"])})
    assert not (d2["ts"].max() < w["ts"].iloc[-1])                     # Friday -> keep


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
