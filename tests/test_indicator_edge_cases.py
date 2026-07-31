"""Degenerate-input hardening for the S/R + structure layer. Offline, no DB.

Every case here was found by feeding the indicators inputs a real tape can produce but a
normal backtest never does: a halted/frozen name, a missing candle, a perfectly flat stretch.
The common failure mode is the dangerous one — none of them raised. They returned a confident,
wrong answer (RSI 100 on a dead tape, a level "defended 72 times", a breakout silently
undetectable), which on a decision board is worse than an exception.
"""
import numpy as np
import pandas as pd

from eqbtst import config, indicators as I


def _bars(h, l, c=None):
    n = len(h)
    c = c if c is not None else h
    return pd.DataFrame({"ts": pd.date_range("2025-01-01", periods=n, freq="D"),
                         "open": c, "high": h, "low": l, "close": c,
                         "volume": np.full(n, 1000.0)})


# ── RSI on a tape that never moved ────────────────────────────────────────────────
def test_rsi_of_a_frozen_tape_is_neutral_not_overbought():
    # zero average loss AND zero average gain is 0/0, not "maximally overbought". Returning
    # 100 made rsi_state read tone='strong' for a name that had not ticked in 14 bars.
    assert I.rsi(np.full(40, 100.0), 14) == 50.0
    assert I.rsi_state(np.full(40, 100.0))["tone"] == "neutral"


def test_rsi_extremes_still_hold_for_genuinely_one_sided_series():
    assert I.rsi(np.arange(30.0, 60.0), 14) == 100.0     # never fell → really is 100
    assert I.rsi(np.arange(60.0, 30.0, -1), 14) == 0.0   # never rose → really is 0


# ── a swing requires the price to have moved ──────────────────────────────────────
def test_a_flat_window_manufactures_no_pivots_and_no_wall():
    flat = np.full(40, 100.0)
    his, los = I.pivots(flat, flat, w=2)
    assert (his, los) == ([], []), "a frozen tape has no swing highs or lows"
    assert I.walls(flat, flat, tol=0.6) == []


def test_flat_window_cannot_manufacture_touch_strength():
    # before the guard: every bar passed BOTH the >= and <= tests, so a 40-bar flat window
    # produced ONE level with ~72 touches — the strongest wall the board can render, off a
    # name that never traded. Touch count is the feature's whole value; inactivity must not
    # be able to fabricate it.
    flat = np.full(40, 250.0)
    assert max([t for _, t in I.walls(flat, flat, tol=1.0)], default=0) == 0


def test_pivot_guard_does_not_touch_a_normal_oscillating_series():
    h = np.array([10, 11, 12, 11, 10, 11, 12, 11, 10, 11, 12, 11, 10, 11, 12, 11], float)
    his, los = I.pivots(h, h - 1.0, w=2)
    assert len(his) >= 3 and len(los) >= 3
    assert {round(x, 2) for x in his} == {12.0}


# ── one missing candle must not disable the breakout test ─────────────────────────
def test_a_nan_high_does_not_silently_suppress_a_breakout():
    # np.sort puts NaN LAST, so the old range-top read NaN; every `close > prior_hi + margin`
    # comparison against NaN is False, so the name read RANGE forever with no error and no
    # 'n/a' — a genuine +20% break went undetected because of one bad candle.
    up = np.concatenate([np.full(19, 100.0), [120.0]])
    holed = up.copy()
    holed[5] = np.nan
    assert np.isfinite(I._range_bound(holed[:-1], 2.0, upper=True))
    assert I.structure(_bars(holed, up - 1.0, up)) == "BREAKOUT_UP"
    assert I.structure(_bars(up, up - 1.0, up)) == "BREAKOUT_UP"      # unchanged when clean


def test_range_bound_still_drops_a_lone_spike():
    # the spike-robustness the function exists for must survive the NaN hardening
    vals = np.concatenate([np.full(18, 100.0), [140.0]])
    assert I._range_bound(vals, atr_val=2.0, upper=True) == 100.0
    lows = np.concatenate([np.full(18, 100.0), [60.0]])
    assert I._range_bound(lows, atr_val=2.0, upper=False) == 100.0


# ── unit-independence: a label must not depend on the price of the share ──────────
def test_structure_and_band_are_scale_invariant():
    rng = np.random.default_rng(3)
    c = 100 + np.cumsum(rng.normal(0, 1, 60))
    one = _bars(c + 0.5, c - 0.5, c)
    ten = _bars((c + 0.5) * 10, (c - 0.5) * 10, c * 10)
    assert I.structure(one) == I.structure(ten)
    assert I.band_pct(one) == I.band_pct(ten)
    a, b = I.sr_levels(one), I.sr_levels(ten)
    assert a["res_touches"] == b["res_touches"] and a["head_up"] == b["head_up"]


# ── zone activity: what the eye counts, beside what the pivot rule counts ────────
def test_zone_visits_counts_approaches_not_bars():
    # one long stay in the zone is ONE visit, not twenty. The eye counts approaches.
    idx = pd.date_range("2025-01-01", periods=20, freq="D")
    c = np.array([100.0] * 10 + [130.0] * 10)
    d = pd.DataFrame({"ts": idx, "open": c, "high": c + 1, "low": c - 1,
                      "close": c, "volume": 1.0})
    z = I.zone_visits(d, 100.0, atr_val=2.0)
    assert z["visits"] == 1 and z["bars"] == 10 and z["closes"] == 10


def test_zone_visits_separates_repeat_approaches():
    idx = pd.date_range("2025-01-01", periods=20, freq="D")
    c = np.array([100.0, 100.0, 130.0, 130.0, 100.0, 100.0, 130.0, 130.0, 100.0, 130.0] * 2)
    d = pd.DataFrame({"ts": idx, "open": c, "high": c + 1, "low": c - 1,
                      "close": c, "volume": 1.0})
    assert I.zone_visits(d, 100.0, atr_val=2.0)["visits"] == 6


def test_zone_visits_can_exceed_the_pivot_count():
    # THE POINT OF THE COLUMN: a shelf revisited inside a choppy range never forms a 5-bar
    # fractal extreme, so the pivot rule under-reports a level the chart clearly uses.
    rng = np.random.default_rng(5)
    base = np.concatenate([np.full(12, 100.0) + rng.normal(0, 0.3, 12),
                           np.full(12, 118.0) + rng.normal(0, 0.3, 12)] * 4)
    d = pd.DataFrame({"ts": pd.date_range("2025-01-01", periods=len(base), freq="D"),
                      "open": base, "high": base + 1, "low": base - 1,
                      "close": base, "volume": 1.0})
    z = I.zone_visits(d, 100.0, atr_val=3.0)
    assert z["visits"] >= 4 and z["time_pct"] > 25


def test_zone_visits_is_degenerate_safe():
    for bad in (None, pd.DataFrame()):
        assert I.zone_visits(bad, 100.0, 2.0)["visits"] == 0
    d = _bars(np.full(20, 100.0), np.full(20, 100.0))
    assert I.zone_visits(d, 100.0, atr_val=0.0)["visits"] == 0     # no ATR -> no zone


def test_sr_levels_reports_activity_beside_touch_count():
    rng = np.random.default_rng(11)
    c = 500 + np.cumsum(rng.normal(0, 4, 150))
    sr = I.sr_levels(_bars(c + 2, c - 2, c), lookback=config.SR_LOOKBACK)
    for k in ("sup_visits", "sup_time_pct", "res_visits", "res_time_pct"):
        assert k in sr
    assert sr["sup_visits"] >= 0 and 0 <= sr["sup_time_pct"] <= 100


def test_sr_levels_orders_support_below_price_and_resistance_above():
    rng = np.random.default_rng(11)
    c = 500 + np.cumsum(rng.normal(0, 4, 120))
    sr = I.sr_levels(_bars(c + 2, c - 2, c), lookback=config.SR_LOOKBACK)
    px = float(c[-1])
    if sr.get("support") is not None:
        assert sr["support"] < px
    if sr.get("resistance") is not None:
        assert sr["resistance"] > px
    for k in ("head_up", "head_dn"):
        if sr.get(k) is not None:
            assert sr[k] >= 0, "headroom is a distance and can never be negative"
