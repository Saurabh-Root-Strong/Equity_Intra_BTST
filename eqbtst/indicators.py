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
    """Ratio of cumulative session volume so far vs a reference average daily
    volume (from the EOD archive). >1 = participating above normal. None ref ->
    NaN (unknown until the EOD baseline is joined)."""
    vol = float(candles["volume"].sum())
    if not ref_avg_day_vol or ref_avg_day_vol <= 0:
        return float("nan")
    return vol / ref_avg_day_vol


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
    stop = max(stop_atr, day_low) if day_low else stop_atr    # structural stop if tighter
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


def structure(candles: pd.DataFrame, lookback: int = 20) -> str:
    """Market structure on this timeframe (CONTEXT, not a signal — intraday structure
    has no validated edge). Kaufman efficiency ratio + range logic:
      BREAKOUT_UP/DOWN  last close beyond the prior range extreme
      TREND_UP/DOWN     efficient directional travel (ER >= 0.4)
      CONSOLIDATION     range contracting (recent range < 0.6x prior)
      RANGE             choppy, no direction
    """
    c = candles["close"].to_numpy(float)
    if len(c) < 5:
        return "n/a"
    seg = c[-lookback:]
    net = seg[-1] - seg[0]
    denom = np.abs(np.diff(seg)).sum()
    er = abs(net) / denom if denom > 0 else 0.0            # Kaufman efficiency ratio
    hi = candles["high"].to_numpy(float)[-lookback:]
    lo = candles["low"].to_numpy(float)[-lookback:]
    last = c[-1]
    if len(hi) > 1 and last > hi[:-1].max():
        return "BREAKOUT_UP"
    if len(lo) > 1 and last < lo[:-1].min():
        return "BREAKOUT_DOWN"
    if er >= 0.4:
        return "TREND_UP" if net > 0 else "TREND_DOWN"
    if len(hi) >= 6:
        recent = hi[-3:].max() - lo[-3:].min()
        prior = hi[:-3].max() - lo[:-3].min()
        if prior > 0 and recent < 0.6 * prior:
            return "CONSOLIDATION"
    return "RANGE"


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
               index_ret: float | None = None) -> dict:
    """Full live indicator + price-action state for one stock from its intraday
    candles. index_ret (Nifty % today) -> live relative strength."""
    o = float(candles["open"].iloc[0])
    h = float(candles["high"].max())
    l = float(candles["low"].min())
    c = float(candles["close"].iloc[-1])
    vw = vwap(candles)
    pa = price_action(o, h, l, c)
    rs = rsi_state(candles["close"].to_numpy(float))
    day_ret = c / prev_close - 1 if prev_close else 0.0
    st = {
        "ltp": round(c, 2), "day_ret": round(100 * day_ret, 2),
        "vwap": round(vw, 2), "vs_vwap": round(100 * (c / vw - 1), 2),
        "above_vwap": c >= vw, "structure": structure(candles),
        "vol_surge": round(volume_surge(candles, ref_avg_day_vol), 2),
        **pa, **rs,
    }
    if index_ret is not None:
        st["rs_vs_index"] = round(100 * (day_ret - index_ret), 2)   # live RS vs Nifty
    return st
