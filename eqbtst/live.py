"""
live.py — live Fyers layer for the intraday board. Reuses Tradebot's token/auth
(no re-plumbing). Degrades gracefully: if the token is not usable (e.g. the
~06:00 IST daily expiry) it returns a status telling you to re-auth, instead of
throwing — the dashboard renders the message.

Two fetch tiers keep API load bounded:
  * quotes_board()  — ONE batch /quotes call for the whole liquid universe -> live
    price-action state (LTP, day OHLC, clr-so-far, body/wick, day%, vol, RS vs
    Nifty). The fast scan, refreshable every ~20-30s.
  * deep_state(sym) — per-symbol intraday candles (/history) -> adds VWAP + RSI for
    a shortlisted name. Called only for names worth a closer look (bounded calls).

Honest actions (long-only; short overnight is proven dead):
  FORMING  price-action footprint building intraday (confirm delivery at close)
  BUY      near the close, footprint holding + regime-on + liquid
  AVOID    below VWAP / weak close / RSI rolling over / illiquid / regime-off
  HOLD/SELL come from the paper ledger (exit an open long next morning)
"""
from __future__ import annotations

import datetime as dt
import sys

import pandas as pd
import requests

from . import config, data, indicators, regime


# ── Tradebot token reuse ──────────────────────────────────────────────────────
def _token():
    p = str(config.TRADEBOT_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    from tradebot.adapters.broker import token          # type: ignore
    return token


def token_status() -> dict:
    try:
        t = _token()
        return {"usable": t.is_usable(), "describe": t.describe()}
    except Exception as e:
        return {"usable": False, "describe": f"token module unavailable: {e}"}


def _auth_header() -> str:
    return _token().auth_header()


def fy_symbol(sym: str) -> str:
    return f"NSE:{sym}-EQ"


def market_open(now: dt.datetime | None = None) -> bool:
    """NSE cash session check: weekday, 09:15–15:30 IST. (Holidays not special-cased
    — a holiday just reads as a flat/stale board, which the staleness badge covers.)"""
    try:
        from zoneinfo import ZoneInfo
        now = now or dt.datetime.now(ZoneInfo("Asia/Kolkata"))
    except Exception:
        now = now or dt.datetime.now()
    if now.weekday() >= 5:                       # Sat/Sun
        return False
    t = now.time()
    return dt.time(9, 15) <= t <= dt.time(15, 30)


# ── liquid universe (from the EOD archive's last close) ────────────────────────
def liquid_universe(date: pd.Timestamp | None = None) -> pd.DataFrame:
    """The tradeable set + the EOD reference fields the live board needs
    (prev_close proxy, avg daily volume baseline). One DuckDB read."""
    date = pd.Timestamp(date) if date is not None else data.last_trading_date()
    start = (date - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    df = data.load_eod(start=start, end=date.strftime("%Y-%m-%d")).sort_values(
        ["symbol", "trade_date"])
    g = df.groupby("symbol", group_keys=False)
    df["vol_med20"] = g["ttl_trd_qnty"].transform(lambda s: s.rolling(20).median())
    # daily ATR14 (Wilder TR) for risk-geometry levels — from EOD, so it costs no live call
    pc = g["close_price"].shift(1)
    tr = pd.concat([df["high_price"] - df["low_price"],
                    (df["high_price"] - pc).abs(),
                    (df["low_price"] - pc).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.groupby(df["symbol"]).transform(lambda s: s.rolling(14).mean())
    last = df[df["trade_date"] == date].copy()
    last = last[last["turnover_lacs"] >= config.LIQ_MIN_LACS]
    sectors = data.load_sectors()
    last["sector"] = last["symbol"].map(lambda s: sectors.get(s, f"_{s}"))
    return last[["symbol", "sector", "close_price", "vol_med20", "atr14",
                 "high_price", "low_price"]].rename(
        columns={"close_price": "ref_close", "high_price": "pdh", "low_price": "pdl"})


# ── tier 1: batch quotes scan ──────────────────────────────────────────────────
def _fetch_quotes(fy_syms: list[str]) -> dict:
    out: dict = {}
    for i in range(0, len(fy_syms), 50):                # /quotes caps the batch size
        chunk = fy_syms[i:i + 50]
        try:
            r = requests.get(config.FYERS_QUOTES_URL,
                             headers={"Authorization": _auth_header(), "version": "3"},
                             params={"symbols": ",".join(chunk)}, timeout=8)
            for it in (r.json().get("d") or []):
                v = it.get("v") or {}
                if it.get("n"):
                    out[it["n"]] = v
        except Exception:
            continue
    return out


def quotes_board(date: pd.Timestamp | None = None) -> dict:
    """Live price-action scan of the liquid universe. Returns status + a DataFrame
    with a live action per name. No VWAP/RSI here (that is deep_state)."""
    ts = token_status()
    uni = liquid_universe(date)
    risk_on = regime.is_risk_on(pd.Timestamp(date) if date is not None
                                else data.last_trading_date())
    if not ts["usable"]:
        return {"ok": False, "status": ts["describe"], "risk_on": risk_on,
                "board": pd.DataFrame()}

    fy = {fy_symbol(s): s for s in uni["symbol"]}
    q = _fetch_quotes(list(fy))
    # live Nifty return for relative strength
    nifty = _fetch_quotes([config.NIFTY_FYERS]).get(config.NIFTY_FYERS, {})
    idx_ret = _chp(nifty)
    # earnings guard: names reporting during the overnight hold can't be a BTST carry
    from . import events
    d0 = (pd.Timestamp(date) if date is not None else data.last_trading_date()).date()
    earn = events.upcoming(d0, horizon_days=3)

    rows = []
    ref = uni.set_index("symbol")
    for fys, v in q.items():
        sym = fy.get(fys)
        if not sym or sym not in ref.index:
            continue
        o, h, l, c = v.get("open_price"), v.get("high_price"), v.get("low_price"), v.get("lp")
        pc = v.get("prev_close_price") or float(ref.loc[sym, "ref_close"])
        if None in (o, h, l, c) or not pc:
            continue
        pa = indicators.price_action(float(o), float(h), float(l), float(c))
        day_ret = 100 * (float(c) / float(pc) - 1)
        rs = day_ret - idx_ret if idx_ret is not None else None
        vol = v.get("volume") or 0
        med = ref.loc[sym, "vol_med20"]
        vsurge = (vol / float(med)) if (vol and med and med == med) else float("nan")
        atr14 = float(ref.loc[sym, "atr14"]) if ref.loc[sym, "atr14"] == ref.loc[sym, "atr14"] else 0.0
        lv = indicators.levels(float(c), atr14, day_low=float(l))
        ready = btst_readiness(pa, day_ret, rs, vsurge)
        sready = short_readiness(pa, day_ret, rs, vsurge)
        # short-side levels: stop ABOVE, targets BELOW (mirror of the long geometry)
        s_stop = round(float(c) + atr14, 2) if atr14 > 0 else None
        s_t1 = round(float(c) - atr14, 2) if atr14 > 0 else None
        s_t2 = round(float(c) - 2 * atr14, 2) if atr14 > 0 else None
        rows.append({
            "symbol": sym, "sector": ref.loc[sym, "sector"], "ltp": float(c),
            "day%": round(day_ret, 2), "clr": pa["clr"], "character": pa["character"],
            "body": pa["body"], "vol×": round(vsurge, 2) if vsurge == vsurge else None,
            "RS%": round(rs, 2) if rs is not None else None,
            "btst": f"{ready}/4", "exp_ON": "+0.3–0.4%" if ready >= 4 else "",
            "short": f"{sready}/4",
            "entry": lv.get("entry"), "stop": lv.get("stop"),
            "t1": lv.get("t1"), "t2": lv.get("t2"),
            "s_stop": s_stop, "s_t1": s_t1, "s_t2": s_t2,
            "risk%": lv.get("risk%"), "atr%": lv.get("atr%"),
            "earnings": "⚠" if sym in earn else "",
            "action": ("EARNINGS" if sym in earn
                       else _live_action(pa, day_ret, rs, vsurge, risk_on)),
            "sell": _sell_action(pa, day_ret, rs, vsurge),
        })
    board = pd.DataFrame(rows)
    if not board.empty:
        board = board.sort_values(["action", "clr"], ascending=[True, False])
    return {"ok": True, "status": ts["describe"], "risk_on": risk_on,
            "idx_ret": idx_ret, "board": board, "n": len(board),
            "market_open": market_open()}


def _chp(v: dict):
    return float(v["chp"]) if v and v.get("chp") is not None else None


def btst_readiness(pa: dict, day_ret: float, rs, vsurge) -> int:
    """How many of the LIVE overnight-footprint legs are met (0-4): strong close,
    up ≥1%, volume surge ≥2×, relative-strength leader. The higher this is near the
    close, the more the day's action looks like the accumulation that historically
    drifts UP overnight — i.e. the more a next-day move is expectable. (VWAP-hold and
    delivery% are the 2 further confirmations: deep-chart view + the EOD close.)"""
    n = 0
    n += pa["clr"] >= config.CLR_TH
    n += day_ret >= 100 * config.RET_TH
    n += (vsurge != vsurge) or (vsurge >= config.VOL_TH)     # NaN volume = unknown, don't penalise
    n += (rs is None) or (rs > 0)
    return int(n)


def short_readiness(pa: dict, day_ret: float, rs, vsurge) -> int:
    """Mirror of btst_readiness for the DISTRIBUTION footprint (0-4): weak close,
    down ≥1%, volume surge, relative-strength laggard. High = the day looks like
    supply/distribution. Intraday only — overnight short is proven -EV."""
    n = 0
    n += pa["clr"] <= (1 - config.CLR_TH)           # closed in bottom 30% of range
    n += day_ret <= -100 * config.RET_TH
    n += (vsurge != vsurge) or (vsurge >= config.VOL_TH)
    n += (rs is None) or (rs < 0)
    return int(n)


def _sell_action(pa: dict, day_ret: float, rs, vsurge) -> str:
    """Intraday-short label (NOT overnight — that is proven dead). SHORT = full
    distribution footprint now; WEAK = building. Square off before the close."""
    ready = short_readiness(pa, day_ret, rs, vsurge)
    bearish = pa["character"] in ("marubozu_bear", "shooting_star", "weak_close")
    if ready >= 4 and bearish:
        return "SHORT"
    if ready >= 3:
        return "WEAK"
    return "—"


def _live_action(pa: dict, day_ret: float, rs, vsurge, risk_on: bool) -> str:
    """Honest live label with the intraday→BTST handoff.

    BTST-CARRY = near the close, the full footprint is holding → SHIFT this intraday
    name to an overnight BTST hold, because the data says a next-day move is expectable
    (exit next-morning strength; delivery% confirms at the close). FORMING = footprint
    building earlier in the day. NEUTRAL/AVOID otherwise. Long-only.
    """
    if not risk_on:
        return "AVOID"
    ready = btst_readiness(pa, day_ret, rs, vsurge)
    near_close = dt.datetime.now().time() >= dt.time(15, 10)
    if ready >= 4 and near_close:
        return "BTST-CARRY"          # shift intraday -> overnight
    if ready >= 4:
        return "FORMING"             # building; may become a carry into the close
    if pa["clr"] <= 0.33 or day_ret < 0:
        return "AVOID"
    return "NEUTRAL"


# ── tier 2: per-symbol deep state (VWAP + RSI) ─────────────────────────────────
_RES = {"1h": "60", "2h": "120", "15m": "15", "5m": "5"}


def fetch_intraday(sym: str, tf: str = "1h", lookback_days: int = 10) -> pd.DataFrame:
    """Candles for one symbol at timeframe tf ('1h','2h','15m','5m') via /history.
    Pulls a few days so ATR14/RSI14 have enough bars on the hourly frames."""
    res = _RES.get(tf, "60")
    d_to = dt.date.today()
    d_from = d_to - dt.timedelta(days=lookback_days)
    r = requests.get(config.FYERS_HISTORY_URL,
                     headers={"Authorization": _auth_header(), "version": "3"},
                     params={"symbol": fy_symbol(sym), "resolution": res,
                             "date_format": "1", "range_from": d_from.isoformat(),
                             "range_to": d_to.isoformat(), "cont_flag": "1"}, timeout=15)
    j = r.json()
    if j.get("s") != "ok" or not j.get("candles"):
        return pd.DataFrame()
    df = pd.DataFrame(j["candles"], columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return df


def tf_scan(tf: str = "1h", max_names: int = 25, date=None) -> dict:
    """Timeframe-driven stock list: scan the liquid universe on the chosen bar
    timeframe (1h/2h/15m). Bounded API: ONE batch-quote call pre-filters to the
    strongest up-names today, then TF candles are pulled only for that shortlist.
    Ranks by the price-action footprint ON THAT TIMEFRAME (last-bar clr, above VWAP,
    RSI tone, volume, RS) + ATR levels. Long-only."""
    ts = token_status()
    risk_on = regime.is_risk_on(pd.Timestamp(date) if date is not None
                                else data.last_trading_date())
    if not ts["usable"]:
        return {"ok": False, "status": ts["describe"], "risk_on": risk_on, "board": pd.DataFrame()}

    uni = liquid_universe(date).set_index("symbol")
    q = _fetch_quotes([fy_symbol(s) for s in uni.index])
    idx_ret = _chp(_fetch_quotes([config.NIFTY_FYERS]).get(config.NIFTY_FYERS, {}))
    # pre-filter: up on the day + closing the daily bar in the upper half
    pre = []
    for fys, v in q.items():
        sym = fys.replace("NSE:", "").replace("-EQ", "")
        if sym not in uni.index:
            continue
        c, pc = v.get("lp"), v.get("prev_close_price") or uni.loc[sym, "ref_close"]
        h, l = v.get("high_price"), v.get("low_price")
        if None in (c, pc, h, l) or h == l:
            continue
        day = 100 * (float(c) / float(pc) - 1)
        clr = (float(c) - float(l)) / (float(h) - float(l))
        if day >= 100 * config.RET_TH and clr >= 0.5:
            pre.append((sym, day, clr))
    pre.sort(key=lambda x: (x[2], x[1]), reverse=True)
    pre = pre[:max_names]

    rows = []
    for sym, day, _ in pre:
        ds = deep_state(sym, tf=tf, ref_close=float(uni.loc[sym, "ref_close"]),
                        ref_avg_vol=None, idx_ret=idx_ret)
        if not ds:
            continue
        s, lv = ds["state"], ds["levels"]
        rows.append({
            "symbol": sym, "sector": uni.loc[sym, "sector"], "ltp": s["ltp"],
            "day%": s["day_ret"], "bar_clr": s["clr"], "character": s["character"],
            "vs_vwap%": s["vs_vwap"], "above_vwap": s["above_vwap"],
            "rsi7": s["rsi7"], "rsi14": s["rsi14"], "tone": s["tone"],
            "RS%": s.get("rs_vs_index"),
            "entry": lv.get("entry"), "stop": lv.get("stop"),
            "t1": lv.get("t1"), "t2": lv.get("t2"), "atr%": lv.get("atr%"),
            "action": _tf_action(s, risk_on),
        })
    board = pd.DataFrame(rows)
    if not board.empty:
        board = board.sort_values(["action", "bar_clr"], ascending=[True, False])
    return {"ok": True, "status": ts["describe"], "tf": tf, "risk_on": risk_on,
            "idx_ret": idx_ret, "board": board, "n_scanned": len(pre)}


def _tf_action(s: dict, risk_on: bool) -> str:
    """Action on a bar timeframe: strong bar + above VWAP + RSI not weak + RS>0."""
    if not risk_on:
        return "AVOID"
    strong = (s["clr"] >= config.CLR_TH and s["above_vwap"]
              and s["tone"] in ("strong", "neutral")
              and (s.get("rs_vs_index") is None or s["rs_vs_index"] > 0))
    if strong:
        return "LONG"
    if s["clr"] <= 0.33 or not s["above_vwap"]:
        return "AVOID"
    return "NEUTRAL"


def deep_state(sym: str, tf: str = "1h", ref_close: float | None = None,
               ref_avg_vol: float | None = None, idx_ret: float | None = None) -> dict:
    """1h/2h chart read for one name: VWAP + proactive RSI + price-action character
    + ATR-based risk levels (entry/stop/t1/t2). Returns the candle frame too, so the
    dashboard can chart it. Empty dict if no candles (pre-open / token / no trades)."""
    if not token_status()["usable"]:
        return {}
    c = fetch_intraday(sym, tf=tf)
    if c.empty:
        return {}
    today = c[c["ts"].dt.date == c["ts"].dt.date.max()]     # today's session for VWAP/clr
    session = today if len(today) >= 2 else c
    pc = ref_close if ref_close else float(c["close"].iloc[0])
    state = indicators.live_state(session, pc, ref_avg_vol,
                                  (idx_ret / 100.0) if idx_ret is not None else None)
    atr_tf = indicators.atr(c, 14)                          # ATR on the chosen timeframe
    lv = indicators.levels(state["ltp"], atr_tf,
                           day_low=float(session["low"].min()),
                           day_high=float(session["high"].max()))
    return {"tf": tf, "state": state, "levels": lv, "candles": c}
