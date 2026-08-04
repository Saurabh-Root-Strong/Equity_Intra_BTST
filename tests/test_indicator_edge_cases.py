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


# ── broker feed artifacts: a repeated bar is not a session ────────────────────────
def test_phantom_bar_is_dropped_and_a_real_session_is_kept():
    """Caught live on Sunday 2026-08-02: the Fyers 1D feed carried a 01-Aug (SATURDAY) bar
    byte-identical to 31-Jul, so the staleness detector reported the EOD archive "1 trading day
    behind the market" when it was perfectly current — telling the user to distrust every 1D/1W
    verdict on the page and re-run a sync with nothing to do."""
    from eqbtst import live
    idx = pd.to_datetime(["2026-07-30", "2026-07-31", "2026-08-01"])
    dup = pd.DataFrame({"ts": idx,
                        "open": [1279.8, 1295.0, 1295.0], "high": [1297.0, 1309.7, 1309.7],
                        "low": [1275.3, 1293.6, 1293.6], "close": [1292.9, 1307.8, 1307.8],
                        "volume": [12158451, 8624996, 8624996]})
    out = live.drop_phantom_bars(dup)
    assert len(out) == 2 and out["ts"].max() == pd.Timestamp("2026-07-31")

    # a GENUINE Saturday session (NSE runs occasional drills) carries its own prices and MUST
    # survive — keying the filter on the weekday instead of on duplicate data would bin it
    real = dup.copy()
    real.loc[2, ["open", "high", "low", "close", "volume"]] = [1300.0, 1312.0, 1298.0, 1310.0, 51234]
    assert len(live.drop_phantom_bars(real)) == 3


def test_phantom_filter_is_a_no_op_on_ordinary_bars():
    from eqbtst import live
    rng = np.random.default_rng(4)
    c = 100 + np.cumsum(rng.normal(0, 1, 30))
    df = pd.DataFrame({"ts": pd.date_range("2026-06-01", periods=30, freq="B"),
                       "open": c, "high": c + 1, "low": c - 1, "close": c,
                       "volume": rng.integers(1e5, 1e6, 30)})
    assert len(live.drop_phantom_bars(df)) == 30
    assert live.drop_phantom_bars(pd.DataFrame()).empty


# ── big-wall correctness: the three fixes of 2026-08-04 ────────────────────────────
# Each guards a defect that made the one-frame-up read say "clear" while a level sat
# plainly on the chart. Measured before/after: 62-73% of names misreported -> 2-6%.

def test_walls_ext_edges_bracket_the_mean_and_match_walls():
    """walls_ext must be walls() plus the cluster's extreme members — not a different merge."""
    import numpy as np
    from eqbtst import indicators as I
    h = np.array([10, 12, 10, 12.4, 10, 20, 10, 12.2, 10, 11, 10, 30, 10], dtype=float)
    l = np.array([9,  8,  9,  8,    9,  8,  9,  8,    9,  8,  9,  8,  9], dtype=float)
    ext = I.walls_ext(h, l, tol=0.5)
    assert ext, "no clusters built"
    assert [(x, t) for x, t, _, _ in ext] == I.walls(h, l, tol=0.5)
    for x, t, lo, hi in ext:
        assert lo <= x <= hi, f"mean {x} outside its own members [{lo}, {hi}]"
        if t == 1:
            assert lo == hi == x


def test_blind_zone_reports_the_bars_pivots_cannot_see():
    """pivots() needs +/-2 neighbours, so the last two bars can never BE a level.

    That is exactly where "price is testing the high right now" lives — two WEEKS on a
    weekly frame. sr_levels must hand that extreme back separately.
    """
    import numpy as np
    import pandas as pd
    from eqbtst import indicators as I
    n = 40
    hi = np.full(n, 100.0)
    lo = np.full(n, 90.0)
    hi[10] = 120.0                      # a real pivot, mid-window
    hi[-1] = 150.0                      # the blind zone: last bar, highest of all
    df = pd.DataFrame({"open": lo, "high": hi, "low": lo, "close": lo})
    sr = I.sr_levels(df, lookback=n)
    assert sr, "no levels"
    assert max(x for x, _ in sr["levels"]) < 150.0, "a last-bar high must not become a pivot"
    assert sr["blind"][1] == 150.0, "blind-zone high not reported"
    # ...and with a FORMING last bar it must be excluded, or the 'level' moves with price
    sr_f = I.sr_levels(df, lookback=n, forming=True)
    assert sr_f["blind"][1] != 150.0


def test_single_touch_level_is_kept_by_the_big_wall_rule():
    """One violent rejection IS a level. The gate used to require two touches, which is the
    same bug already fixed for headroom — it made a lone spike read 'clear'."""
    import numpy as np
    import pandas as pd
    from eqbtst import indicators as I
    n = 40
    hi = np.full(n, 100.0)
    lo = np.full(n, 90.0)
    hi[20] = 130.0                      # ONE touch, well above price, mid-window
    df = pd.DataFrame({"open": lo, "high": hi, "low": lo, "close": lo})
    sr = I.sr_levels(df, lookback=n)
    one_touch = [x for x, t in sr["levels"] if t == 1 and x > 110]
    assert one_touch, "the single rejection vanished from the level list"
