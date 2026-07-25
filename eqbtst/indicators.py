"""
indicators.py — intraday indicators + price-action character. PURE functions
(no IO), computed from an intraday OHLCV candle frame. Offline-testable.

These are the LIVE-computable half of the footprint. Fyers streams price + volume,
so VWAP / RSI / clr-so-far / body-wick / volume-surge / relative-strength are all
available intraday. Delivery% is NOT (NSE publishes it post-close) — so the live
board shows a name FORMING the footprint; delivery confirms it after the bell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vwap(candles: pd.DataFrame) -> float:
    """Session VWAP from intraday candles (typical price × volume, cumulative)."""
    tp = (candles["high"] + candles["low"] + candles["close"]) / 3.0
    v = candles["volume"].to_numpy(float)
    if v.sum() <= 0:
        return float(candles["close"].iloc[-1])
    return float((tp.to_numpy(float) * v).sum() / v.sum())


def rsi(closes, period: int = 14) -> float:
    """Wilder RSI of a close series. Returns 50 on insufficient data."""
    c = np.asarray(closes, float)
    if len(c) <= period:
        return 50.0
    d = np.diff(c)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ag, al = up[:period].mean(), dn[:period].mean()
    for i in range(period, len(d)):                       # Wilder smoothing
        ag = (ag * (period - 1) + up[i]) / period
        al = (al * (period - 1) + dn[i]) / period
    if al == 0:
        return 100.0
    return float(100.0 - 100.0 / (1.0 + ag / al))


def rsi_state(closes) -> dict:
    """'Proactive' RSI read: fast (7) + standard (14) + slope of the fast line.
    Fast RSI turns before the slow one; the slope flags momentum rolling over
    EARLY — the point of a proactive read vs a lagging 14-only cross."""
    c = np.asarray(closes, float)
    fast, slow = rsi(c, 7), rsi(c, 14)
    slope = fast - rsi(c[:-1], 7) if len(c) > 8 else 0.0
    if fast >= 60 and slope >= 0:
        tone = "strong"
    elif fast <= 40 and slope <= 0:
        tone = "weak"
    elif slope < -3:
        tone = "rolling-over"
    else:
        tone = "neutral"
    return {"rsi7": round(fast, 1), "rsi14": round(slow, 1),
            "slope": round(slope, 1), "tone": tone}


def price_action(o: float, h: float, l: float, c: float) -> dict:
    """Candle anatomy + a small, honest character label (not a 60-pattern zoo)."""
    rng = h - l
    if rng <= 0:
        return {"clr": 0.5, "body": 0.0, "upper_wick": 0.0, "lower_wick": 0.0,
                "character": "flat"}
    body = (c - o) / rng
    upper = (h - max(o, c)) / rng
    lower = (min(o, c) - l) / rng
    clr = (c - l) / rng
    bull = c >= o
    if abs(body) >= 0.7:
        ch = "marubozu_bull" if bull else "marubozu_bear"
    elif lower >= 0.5 and upper <= 0.2:
        ch = "hammer"                 # long lower wick, tiny upper -> demand rejected lows
    elif upper >= 0.5 and lower <= 0.2:
        ch = "shooting_star"          # long upper wick, tiny lower -> supply capped highs
    elif abs(body) <= 0.15:
        ch = "doji"
    elif clr >= 0.66:
        ch = "strong_close"
    elif clr <= 0.33:
        ch = "weak_close"
    else:
        ch = "mid"
    return {"clr": round(clr, 3), "body": round(body, 3),
            "upper_wick": round(upper, 3), "lower_wick": round(lower, 3),
            "character": ch}


def volume_surge(candles: pd.DataFrame, ref_avg_day_vol: float | None = None) -> float:
    """RAW ratio of cumulative session volume so far vs a reference average daily
    volume (from the EOD archive). NOTE: this is TIME-BIASED intraday — at 10am only
    ~22% of a day's volume has traded, so a genuine 2x-volume day reads ~0.44 here.
    Correct at the CLOSE (what the 8yr backtest used); use volume_pace() intraday."""
    vol = float(candles["volume"].sum())
    if not ref_avg_day_vol or ref_avg_day_vol <= 0:
        return float("nan")
    return vol / ref_avg_day_vol


def day_fraction(now_hhmm: str | None = None) -> float:
    """Median fraction of a day's volume completed by `now` (config.VOL_PROFILE).
    Before the open -> the first bucket; after 15:25 -> 1.0. Interpolates by taking
    the latest bucket at or before `now`."""
    from . import config
    import datetime as _dt
    prof = config.VOL_PROFILE
    if now_hhmm is None:
        now_hhmm = _dt.datetime.now().strftime("%H:%M")
    keys = sorted(prof)
    if now_hhmm < keys[0]:
        return prof[keys[0]]
    if now_hhmm >= keys[-1]:
        return 1.0
    lo = keys[0]
    for k in keys:                      # latest bucket at or before now
        if k <= now_hhmm:
            lo = k
        else:
            break
    return prof[lo]


def volume_pace(candles: pd.DataFrame, ref_avg_day_vol: float | None = None,
                now_hhmm: str | None = None) -> float:
    """TIME-NORMALISED volume surge — 'is this name ON PACE for an N× volume day?'

    raw = cum_session_vol / median_daily_vol is mechanically small in the morning
    (only ~4% of the day has traded at 09:15), so the same 2.0 gate means a 2x day at
    the close but a ~7x-pace outlier at 10am. Dividing by the elapsed volume fraction
    makes the number mean the SAME THING at any hour: a true 2x-volume day reads 2.0
    all day. At/after 15:25 the fraction is 1.0, so this EQUALS the raw ratio — the
    validated close decision (and the 8yr backtest) is mathematically untouched."""
    raw = volume_surge(candles, ref_avg_day_vol)
    if raw != raw:                                   # NaN -> unknown, stays unknown
        return float("nan")
    frac = day_fraction(now_hhmm)
    return raw / frac if frac > 0 else float("nan")


def atr(candles: pd.DataFrame, period: int = 14) -> float:
    """Average True Range over the candle frame (any timeframe). Wilder mean of
    TR = max(h-l, |h-prev_close|, |l-prev_close|). Returns high-low mean if short."""
    h = candles["high"].to_numpy(float)
    l = candles["low"].to_numpy(float)
    c = candles["close"].to_numpy(float)
    if len(c) < 2:
        return float(h[-1] - l[-1]) if len(c) else 0.0
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    n = min(period, len(tr))
    return float(pd.Series(tr).rolling(n).mean().iloc[-1])


def levels(ltp: float, atr_val: float, day_low: float | None = None,
           day_high: float | None = None, direction: str = "long") -> dict:
    """RISK GEOMETRY for a long — NOT a price forecast. Entry at market (LTP) with a
    better-fill limit near the day's structure; stop = 1 ATR (or below day low, the
    tighter/structural one); targets at 1 and 2 ATR. R:R quoted vs the stop. The
    validated alpha is the overnight drift; these levels are for trade management,
    so a thin edge is not given back through sloppy exits."""
    if atr_val <= 0 or ltp <= 0:
        return {}
    stop_atr = ltp - 1.0 * atr_val
    # structural stop if tighter, but only if the day-low is genuinely below entry
    # (a name that closed AT its low would give stop == entry = zero risk)
    stop = max(stop_atr, day_low) if (day_low and day_low < ltp) else stop_atr
    risk = ltp - stop
    t1, t2 = ltp + 1.0 * atr_val, ltp + 2.0 * atr_val
    return {
        "entry": round(ltp, 2),
        "limit_buy": round(day_low, 2) if day_low else round(ltp - 0.3 * atr_val, 2),
        "stop": round(stop, 2),
        "t1": round(t1, 2), "t2": round(t2, 2),
        "risk%": round(100 * risk / ltp, 2),
        "rr1": round((t1 - ltp) / risk, 2) if risk > 0 else None,
        "atr%": round(100 * atr_val / ltp, 2),
    }


_CORP_ACTION_GAP = 0.25        # |overnight move| above which a gap is a corporate action


def adjust_corporate_actions(candles: pd.DataFrame,
                             thresh: float = _CORP_ACTION_GAP) -> pd.DataFrame:
    """BACK-ADJUST prices across splits / bonuses / demergers. Cash equity only — indices
    never do this, which is why the logic this was modelled on had no need for it.

    The EOD archive is UNADJUSTED: KOTAKBANK prints 2132.6 then 421.0 across its 1:5, and
    BAJFINANCE 9331 -> 938 across its 1:10. Measured: 26 of 268 names in the universe carry
    such a discontinuity, and 4 of the 5 with one inside the structure window were being
    labelled a FAKE TREND_DOWN — TATAMOTORS read TREND_DOWN on a 1D frame whose entire
    'decline' was a 40% demerger (post-event bars alone read RANGE). Any support/resistance
    built on those bars inherits the same fiction, with walls at pre-split prices.

    Back-adjustment (multiply every PRIOR bar by the close ratio) rather than truncation, so
    the 20-bar window keeps its history instead of going 'n/a' for months after an event.

    WHY 25% IS SAFE HERE: this universe is NSE F&O cash, where daily price bands cap real
    moves well below that, so an overnight gap past 25% is a corporate action rather than a
    trade. A genuine crash beyond the band cannot happen in one session. The threshold is
    deliberately conservative — mis-adjusting a real move would erase it, so the bar is set
    where real moves cannot reach."""
    if candles is None or len(candles) < 3 or "close" not in candles.columns:
        return candles
    c = candles["close"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = c[1:] / c[:-1]
    idx = np.where(np.isfinite(ratio) & (np.abs(ratio - 1.0) > thresh))[0]
    if not len(idx):
        return candles
    out = candles.copy()
    cols = [x for x in ("open", "high", "low", "close") if x in out.columns]
    arr = {x: out[x].to_numpy(float).copy() for x in cols}
    for i in idx:                       # i = last bar BEFORE the event; ratio applies onward
        f = float(ratio[i])
        if not np.isfinite(f) or f <= 0:
            continue
        for x in cols:
            arr[x][:i + 1] *= f         # scale all prior bars onto the post-event scale
    for x in cols:
        out[x] = arr[x]
    return out


def structure(candles: pd.DataFrame, lookback: int | None = None) -> str:
    """Structure LABEL only — thin wrapper over struct_full (see it for the rules)."""
    return struct_full(candles, lookback)["struct"]


def struct_full(candles: pd.DataFrame, lookback: int | None = None,
                forming: bool = False) -> dict:
    """Structure label PLUS the context a multi-timeframe synthesis needs: the window's
    range hi/lo (the BOX price lives in), Kaufman ER, coil ratio, bar count, last close.

    The box is the point. A label alone cannot tell you whether a lower-TF breakout is real
    — that depends on WHERE inside the higher-TF box price sits (edge = possible resolution,
    middle = statistically a false break). See mtf.synthesize.

    Market structure on this timeframe (CONTEXT, not a signal — intraday structure has
    no validated edge). MAGNITUDE-gated, ATR-normalised: the label is defined by the SIZE
    of the move, not just topology, so a marginal new high is NOT called a breakout.

    `forming=True` says the LAST bar has not closed yet (a live intraday frame). Only the
    COIL test changes: it then reads closed bars only, because a span comparison against a
    part-printed bar manufactures squeezes. Breakout/trend still see the live bar.

      BREAKOUT_UP/DOWN  last close clears the prior 19-bar range by >= STRUCT_BREAKOUT_ATR x
                        ATR (a real break, not a poke). On daily ATR that is ~1-1.5% beyond
                        the range; the ATR unit auto-scales the same rule to 15m..1W.
      TREND_UP/DOWN     efficient travel (Kaufman ER >= STRUCT_TREND_ER) AND the window's net
                        move covers >= STRUCT_TREND_ATR x ATR (a REAL directional move, not
                        micro-drift that happens to be efficient).
      CONSOLIDATION     range TIGHTENING — recent 3-bar range < STRUCT_COIL x the prior range.
      RANGE             oscillating sideways: no efficient direction, no break, not tightening.
    """
    from . import config
    lb = lookback if lookback is not None else config.STRUCT_LOOKBACK
    if candles is None or "close" not in getattr(candles, "columns", []):
        return {"struct": "n/a", "n": 0}
    c = candles["close"].to_numpy(float)
    if len(c) < 5:
        return {"struct": "n/a", "n": int(len(c))}
    seg = c[-lb:]
    net = seg[-1] - seg[0]
    denom = np.abs(np.diff(seg)).sum()
    er = abs(net) / denom if denom > 0 else 0.0            # Kaufman efficiency ratio
    hi = candles["high"].to_numpy(float)[-lb:]
    lo = candles["low"].to_numpy(float)[-lb:]
    last = seg[-1]
    # ATR = the frame's OWN volatility unit, so the % gates auto-scale across timeframes.
    a = atr(candles, min(14, len(candles)))
    if not a or a <= 0:                                    # flat/degenerate → fall back to span
        a = (hi.max() - lo.min()) / lb if lb else 0.0
    margin = config.STRUCT_BREAKOUT_ATR * a
    prior_hi = hi[:-1].max() if len(hi) > 1 else last
    prior_lo = lo[:-1].min() if len(lo) > 1 else last
    coil = None
    # A COIL IS A COMPLETED OBSERVATION. "The range is contracting" cannot be said from a bar
    # that is five minutes old, and the coil test is precisely a span comparison — so a
    # part-printed newest bar makes the latest 3-bar span mechanically narrow and manufactures
    # a squeeze. Measured by replaying a session at wall-clock checkpoints, 4h frame, 60 names:
    #
    #     clock    coil fires INCL forming bar    EXCL forming bar
    #     12:20            7%                          17%
    #     13:20           30%   <- the 13:15 bar         7%
    #     13:35           28%      opens with ONE        7%
    #     15:20           20%      candle in it          7%
    #
    # A 4x spike the moment a coarse bar opens, decaying as it fills. Fourth instance of one
    # defect family in this codebase — after the original coil detector, the partial weekly
    # bar and the trailing session stub — all of them a window comparison where one side holds
    # fewer bars. The BREAKOUT and TREND tests deliberately keep the forming bar: detecting a
    # break as it happens is the point of a live board, and those tests compare a CLOSE to a
    # prior range rather than one span to another, so a young bar does not bias them.
    _h, _l = (hi[:-1], lo[:-1]) if (forming and len(hi) > 8) else (hi, lo)
    if len(_h) >= 8:
        _sp = np.array([_h[i:i + 3].max() - _l[i:i + 3].min() for i in range(len(_h) - 2)])
        _typ = float(np.median(_sp[:-1]))
        coil = round(float(_sp[-1]) / _typ, 2) if _typ > 0 else None
    ctx = {"hi": float(hi.max()), "lo": float(lo.min()), "er": round(float(er), 2),
           "coil": coil, "n": int(len(c)), "last": float(last), "atr": float(a)}

    def _r(s):
        return {"struct": s, **ctx}

    # BREAKOUT — only if the close clears the range by a MEANINGFUL margin (not a 0.1% poke)
    if last > prior_hi + margin:
        return _r("BREAKOUT_UP")
    if last < prior_lo - margin:
        return _r("BREAKOUT_DOWN")
    # TREND — efficient AND a real distance covered
    if er >= config.STRUCT_TREND_ER and abs(net) >= config.STRUCT_TREND_ATR * a:
        return _r("TREND_UP" if net > 0 else "TREND_DOWN")
    # COIL — the recent range is TIGHT vs its OWN typical range. Apples-to-apples: the latest
    # 3-bar span against the MEDIAN 3-bar span (same window length). The old test compared 3
    # bars to the other 17 — mechanically biased, since fewer bars always span less, so it fired
    # on ~70% of random walks (labelled normal chop as "coiling"). This measures a REAL
    # volatility contraction: the recent squeeze is < STRUCT_COIL x what this name usually does.
    if coil is not None and coil < config.STRUCT_COIL:
        return _r("CONSOLIDATION")
    return _r("RANGE")


def pivots(h, l, w: int = 2):
    """Swing highs / lows — a bar whose high (low) is the extreme of its +/-w neighbours.

    The +/-w requirement is also what makes this SAFE ON A FORMING BAR: the last w bars can
    never qualify, so a level cannot be invented from a candle that is still printing."""
    h = np.asarray(h, float)
    l = np.asarray(l, float)
    his, los = [], []
    for i in range(w, len(h) - w):
        if h[i] >= h[i - w:i + w + 1].max() - 1e-9:
            his.append(float(h[i]))
        if l[i] <= l[i - w:i + w + 1].min() + 1e-9:
            los.append(float(l[i]))
    return his, los


def walls(h, l, tol: float, w: int = 2) -> list[tuple[float, int]]:
    """TOUCH-COUNTED dynamic support/resistance: cluster swing pivots that sit within `tol`
    of each other, and carry HOW MANY times price turned there.

    A 3-touch cluster is a level the market rejected three separate times; a 1-touch cluster
    is just a pivot. That count is the whole point — it is the difference between a line
    drawn on a chart and a level someone is actually defending.

    `tol` MUST be volatility-scaled by the caller (a fraction of ATR), never a fixed rupee
    amount: 2 rupees is a wide zone on a 90-rupee name and noise on a 9,000-rupee one.

    Returns [(level, touches)] sorted by price. Levels are the touch-weighted mean of their
    cluster, so a level drifts toward where price actually turned most often."""
    his, los = pivots(h, l, w=w)
    lv: list[list] = []
    for x in sorted(his + los):
        for c in lv:
            if abs(x - c[0]) <= tol:
                c[0] = (c[0] * c[1] + x) / (c[1] + 1)     # running touch-weighted mean
                c[1] += 1
                break
        else:
            lv.append([x, 1])
    return [(round(x, 2), int(t)) for x, t in lv]


def sr_levels(candles: pd.DataFrame, spot: float | None = None,
              lookback: int | None = None, tol_atr: float | None = None,
              min_dist_atr: float = 0.25) -> dict:
    """Nearest touch-counted support and resistance around `spot`, plus headroom.

    Deliberately reports the nearest wall on EACH side separately from the nearest
    MULTI-TOUCH wall: a 1-touch pivot 0.1 ATR overhead is noise, a 3-touch wall in the same
    place is the reason your target will not fill.

      support / resistance      nearest cluster each side, beyond min_dist_atr (micro-swings
                                hugging price are not levels)
      sup_touches / res_touches rejection count = strength
      head_up / head_dn         distance to the nearest >=2-touch wall each way, in ATR

    Returns {} when there are too few bars to have structure at all."""
    from . import config
    if lookback is None:
        lookback = config.SR_LOOKBACK
    if tol_atr is None:
        tol_atr = config.SR_TOL_ATR
    if candles is None or len(candles) < 12:
        return {}
    cd = adjust_corporate_actions(candles)               # phantom walls at pre-split prices
    h = cd["high"].to_numpy(float)
    l = cd["low"].to_numpy(float)
    c = cd["close"].to_numpy(float)
    px = float(spot) if spot else float(c[-1])
    if px <= 0:
        return {}
    a = atr(cd, min(14, len(cd)))
    if not a or a <= 0:
        a = float(np.mean(h[-14:] - l[-14:])) if len(h) >= 14 else float(np.mean(h - l))
    if not a or a <= 0:
        return {}
    hh, ll = h[-lookback:], l[-lookback:]
    lv = walls(hh, ll, tol_atr * a)
    if not lv:
        return {}
    md = min_dist_atr * a
    above = [(x, t) for x, t in lv if x > px + md]
    below = [(x, t) for x, t in lv if x < px - md]
    res, res_t = min(above, key=lambda z: z[0]) if above else (None, 0)
    sup, sup_t = max(below, key=lambda z: z[0]) if below else (None, 0)
    # HEADROOM uses ALL >=2-touch walls with NO min-distance filter — that filter exists to
    # keep the DISPLAY clean, and applying it here would hide the closest, most dangerous
    # wall from the very warning meant to flag it.
    up2 = [x for x, t in lv if t >= 2 and x > px]
    dn2 = [x for x, t in lv if t >= 2 and x < px]
    return {
        "support": sup, "sup_touches": sup_t,
        "resistance": res, "res_touches": res_t,
        "head_up": round((min(up2) - px) / a, 2) if up2 else None,
        "head_dn": round((px - max(dn2)) / a, 2) if dn2 else None,
        "atr": a, "levels": lv,
    }


def band_pct(candles: pd.DataFrame, label: str | None = None,
             lookback: int | None = None) -> float:
    """The oscillation band half-width as ±% of price — the LIVE, per-name, per-frame
    'how wide is this range' number (volatility-driven; auto-scales with timeframe because it
    is measured from THAT frame's own bars). For a COIL it reports the CONTRACTED box (recent
    3 bars — the tight squeeze); otherwise the full lookback range. NaN if too few bars."""
    from . import config
    lb = lookback if lookback is not None else config.STRUCT_LOOKBACK
    if candles is None or len(candles) < 3:
        return float("nan")
    hi = candles["high"].to_numpy(float)
    lo = candles["low"].to_numpy(float)
    c = float(candles["close"].to_numpy(float)[-1])
    if c <= 0:
        return float("nan")
    if label == "CONSOLIDATION" and len(hi) >= 3:
        span = hi[-3:].max() - lo[-3:].min()          # the tight coil box (current squeeze)
    else:
        # the oscillation channel — ROBUST to a single spike/wick: a lone outlier bar would
        # blow out a raw max-min and overstate how wide price actually swings. Trim the extreme
        # ~1 bar each side (95th-pct high vs 5th-pct low) so the band is the TYPICAL channel.
        h, l = hi[-lb:], lo[-lb:]
        span = (np.percentile(h, 95) - np.percentile(l, 5)) if len(h) >= 8 else (h.max() - l.min())
    return round(span / 2.0 / c * 100, 2)             # ± half-width, % of price


def band(price: float, atr_val: float) -> dict:
    """Calibrated expected-move BAND — where price is LIKELY to be, not a target.
    Uses the ATR-multiples calibrated on the F&O universe. Returns the ~68% next-day
    close band, the ~74% full-range band, and the expected-move %. Empty if no ATR."""
    from . import config
    if atr_val <= 0 or price <= 0:
        return {}
    b68 = config.BAND_CLOSE_68 * atr_val
    brg = config.BAND_RANGE * atr_val
    return {
        "band_lo": round(price - b68, 2), "band_hi": round(price + b68, 2),   # ~68% next close
        "range_lo": round(price - brg, 2), "range_hi": round(price + brg, 2),  # ~74% next H/L
        "exp_move%": round(100 * b68 / price, 2),                              # ±% at ~68%
    }


def live_state(candles: pd.DataFrame, prev_close: float,
               ref_avg_day_vol: float | None = None,
               index_ret: float | None = None,
               now_hhmm: str | None = None) -> dict:
    """Full live indicator + price-action state for one stock from its intraday
    candles. index_ret (Nifty % today) -> live relative strength.

    `now_hhmm` is the CLOCK time the state is computed at — it drives the volume-pace
    normalisation. It MUST be passed explicitly and must NOT be inferred from the last
    bar's timestamp: a bar's ts is its OPEN, so a 4h frame at a 15:15 cut would report
    13:15 and inflate the pace by ~1.5x (a false volume surge). Replay passes its cut;
    live defaults to the wall clock."""
    o = float(candles["open"].iloc[0])
    h = float(candles["high"].max())
    l = float(candles["low"].min())
    c = float(candles["close"].iloc[-1])
    vw = vwap(candles)
    pa = price_action(o, h, l, c)
    rs = rsi_state(candles["close"].to_numpy(float))
    day_ret = c / prev_close - 1 if prev_close else 0.0
    # 'now' for the volume-pace normalisation. Explicit only — NEVER the last bar's ts
    # (that is the bar's OPEN; on a 4h frame it lags the real clock by hours).
    if now_hhmm is None:
        import datetime as _dt
        now_hhmm = _dt.datetime.now().strftime("%H:%M")
    st = {
        "ltp": round(c, 2), "day_ret": round(100 * day_ret, 2),
        "vwap": round(vw, 2), "vs_vwap": round(100 * (c / vw - 1), 2),
        "above_vwap": c >= vw, "structure": structure(candles),
        "vol_surge": round(volume_surge(candles, ref_avg_day_vol), 2),      # raw (time-biased)
        "vol_pace": round(volume_pace(candles, ref_avg_day_vol, now_hhmm), 2),  # time-normalised
        **pa, **rs,
    }
    if index_ret is not None:
        st["rs_vs_index"] = round(100 * (day_ret - index_ret), 2)   # live RS vs Nifty
    return st
