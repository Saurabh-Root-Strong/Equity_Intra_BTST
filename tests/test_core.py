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
    df.loc[i, "deliv_per"] = 75.0           # high, and >> 40 baseline
    # give it a few prior up-days so 10d cumulative RS vs a flat index is > 0
    for k in range(2, 12):
        df.loc[df.index[-k], "close_price"] = 100.3
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
    assert (config.CLR_TH, config.DELIV_TH, config.VOL_TH, config.RET_TH) == (0.70, 60.0, 2.0, 0.01)
    assert config.CVWAP_TH == 0.005           # path-signature refiner (flips 2025 positive)
    assert (config.RS_LOOKBACK, config.RS_MIN) == (10, 0.0)   # persistent-RS refiner
    assert config.LIQ_MIN_LACS == 2000.0                       # realism/liquidity floor
    assert config.MAX_HOLD == "overnight"     # never day-2
