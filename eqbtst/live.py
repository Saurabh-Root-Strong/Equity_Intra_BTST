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
import threading
import time
from pathlib import Path

import numpy as np
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
_UNI_CACHE: dict = {}          # date_key -> (frame, built_at)
_UNI_TTL = 120.0               # seconds — the EOD universe changes once a DAY, not every 5s


def clear_universe_cache() -> None:
    """Drop EVERY cache derived from the EOD archive, so a fresh nightly sync is picked up at
    once rather than after a TTL.

    A NEW trading date already refreshes itself: each of these is keyed by date, so tomorrow's
    ingest simply misses and recomputes. The case this exists for is the archive being rewritten
    for the SAME date — a partial sync that later completes, or a corrected bhavcopy. The key
    does not change then, so without an explicit clear the board would serve the first,
    incomplete read for the life of the process while cheerfully reporting the right date.

    The delivery tables were missing from this list even though the docstring promised "a fresh
    nightly sync is picked up immediately" — the ↻ button rebuilt the universe around stale
    delivery numbers.
    """
    for c in (_UNI_CACHE, _DELIV_MOM, _DELIV_WK, _DAILY_HIST, _MONTHLY_HIST,
              _MTF_CACHE, _UNISCAN_CACHE, _STALE_CACHE):
        c.clear()


def liquid_universe(date: pd.Timestamp | None = None) -> pd.DataFrame:
    """The tradeable set + the EOD reference fields the live board needs
    (prev_close, volume baseline, ATR, trailing delivery, cumulative RS). One DuckDB read.

    CACHED: this is derived from the EOD archive and changes once a DAY, but the live board
    refreshes every 5s — uncached it re-opened DuckDB and re-scanned 40 days x 270 symbols
    on every tick (~0.5-0.8s, i.e. ~10% of the refresh budget and ~720 archive reads/hour).
    Returns a copy so a caller can never mutate the cached frame.
    """
    import time as _time
    _key = str(pd.Timestamp(date).date()) if date is not None else "_live"
    _hit = _UNI_CACHE.get(_key)
    if _hit is not None and (_time.time() - _hit[1]) < _UNI_TTL:
        return _hit[0].copy()
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
    df["prev_c"] = pc                              # close of the PRIOR trading day
    # trailing delivery THROUGH the last completed day (all published) — for a NEXT-
    # session entry this is leak-free and includes the last EOD row (no shift).
    df["deliv_trail"] = df.groupby("symbol")["deliv_per"].transform(
        lambda s: s.rolling(config.DELIV_TRAIL_WIN).mean())
    # PERSISTENT relative strength — the validated leg is the RS_LOOKBACK-day CUMULATIVE
    # RS vs the index (a sustained leader), NOT a single day's burst. The live board can
    # only see today's RS, so carry the completed part here: the sum of daily (stock −
    # index) returns over the last RS_LOOKBACK-1 sessions THROUGH this archive row. The
    # caller adds today's live RS to it to reconstruct the full 10-day cumulative.
    nf = data.load_nifty(start=start, end=date.strftime("%Y-%m-%d")).sort_values("trade_date")
    nf["idx_ret"] = nf["close_val"].pct_change()
    df = df.merge(nf[["trade_date", "idx_ret"]], on="trade_date", how="left")
    df = df.sort_values(["symbol", "trade_date"])
    df["rs_d"] = (df["close_price"] / df.groupby("symbol")["close_price"].shift(1) - 1) - df["idx_ret"]
    _w = config.RS_LOOKBACK - 1
    g2 = df.groupby("symbol", group_keys=False)["rs_d"]
    df["rs_cum9"] = g2.transform(lambda s: s.rolling(_w).sum())          # through this row
    df["rs_cum9_prior"] = g2.transform(lambda s: s.shift(1).rolling(_w).sum())  # excl. this row
    # delivery through the PRIOR row too — replay's "today" IS this row, and today's
    # delivery is not published until ~6pm, so replay must not read it.
    df["deliv_trail_prior"] = df.groupby("symbol")["deliv_per"].transform(
        lambda s: s.shift(1).rolling(config.DELIV_TRAIL_WIN).mean())
    last = df[df["trade_date"] == date].copy()
    # LIQUIDITY: the backtest gates on the SIGNAL DAY's turnover. When `date` is given
    # (replay / EOD) this row IS the signal day, so filter here. But LIVE, this row is
    # YESTERDAY — and a footprint day carries a 2x volume surge, so yesterday's turnover
    # systematically excludes the very names that just exploded into liquidity. Measured:
    # gating on t-1 drops 170 of 747 validated signals (23%), and those missed names average
    # +29.3bps — the BEST ones. So live keeps the universe broad and gates on TODAY's
    # turnover (volume x price), computed from the live quote in quotes_board.
    if date is not None:
        last = last[last["turnover_lacs"] >= config.LIQ_MIN_LACS]
    sectors = data.load_sectors()
    last["sector"] = last["symbol"].map(lambda s: sectors.get(s, f"_{s}"))
    out = last[["symbol", "sector", "close_price", "prev_c", "vol_med20", "atr14",
                "deliv_trail", "deliv_trail_prior", "rs_cum9", "rs_cum9_prior",
                "high_price", "low_price"]].rename(
        columns={"close_price": "ref_close", "prev_c": "prev_close",
                 "high_price": "pdh", "low_price": "pdl"})
    _UNI_CACHE[_key] = (out, _time.time())
    return out.copy()


# ── tier 1: batch quotes scan ──────────────────────────────────────────────────
_QUOTE_GAP: list[int] = [0]        # symbols lost to a failed /quotes chunk on the last call


def _fetch_quotes(fy_syms: list[str]) -> dict:
    """Batch quotes, 50 symbols per request.

    A DROPPED CHUNK IS A DROPPED FIFTY NAMES. The old `except: continue` swallowed a whole
    batch on any timeout or rate-limit, and those names never reached the board at all --
    not as 'n/a' rows, but as rows that were never built, so `n_scanned` already excluded
    them and nothing could flag the loss. Same failure class as the /history 429 hole that
    cost 20% of the universe: a failure recorded as an answer. It has a wider blast radius
    here because this is the FIRST fetch -- everything downstream inherits the gap.
    So: retry once, then COUNT what is still missing so the caller can say so out loud."""
    chunks = [fy_syms[i:i + 50] for i in range(0, len(fy_syms), 50)]

    def _one(chunk: list[str]) -> tuple[dict, int]:
        for attempt in (0, 1):
            try:
                r = requests.get(config.FYERS_QUOTES_URL,
                                 headers={"Authorization": _auth_header(), "version": "3"},
                                 params={"symbols": ",".join(chunk)}, timeout=8)
                d = r.json().get("d") or []
                if not d:
                    raise ValueError("empty quote batch")
                return ({it["n"]: (it.get("v") or {}) for it in d if it.get("n")}, 0)
            except Exception:
                if attempt == 0:
                    time.sleep(0.5)                     # one transient blip, not a policy
        return ({}, len(chunk))

    # CONCURRENT, because this now runs on a 5-SECOND LOOP. The chunks are independent GETs
    # and each costs ~0.65s, so serially the whole universe took 3.3s -- two thirds of the
    # refresh interval spent waiting, for a call that is only 5 requests wide. Bounded to the
    # chunk count (at most 6 for ~270 names), which is nowhere near the /quotes budget.
    out: dict = {}
    missing = 0
    if len(chunks) == 1:
        out, missing = _one(chunks[0])
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(6, len(chunks))) as ex:
            for got, miss in ex.map(_one, chunks):
                out.update(got)
                missing += miss
    _QUOTE_GAP[0] = missing
    return out


_LIVE_STATE = Path(__file__).resolve().parent.parent / "data" / "live_state"


def mark_first_seen(day, symbols) -> dict:
    """First-seen tracker for the LIVE snapshot (which has no intraday bars to compute a
    causal 'entered'). Stamps the wall-clock HH:MM each symbol FIRST appeared in a
    tradeable tab, persisted per trading day so it survives dashboard reloads. Reads
    'when OUR scanner first saw it qualify' — accurate for a board running from the open,
    later if you start the dashboard mid-session. Returns {symbol: 'HH:MM'}."""
    import json
    f = _LIVE_STATE / f"seen_{pd.Timestamp(day).date()}.json"
    seen = {}
    if f.exists():
        try:
            seen = json.loads(f.read_text())
        except Exception:
            seen = {}
    now = dt.datetime.now().strftime("%H:%M")
    changed = False
    for s in symbols:
        if s not in seen:
            seen[s] = now
            changed = True
    if changed:
        _LIVE_STATE.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(seen))
    return seen


def archive_health(ref: pd.DataFrame, quotes: dict, tol: float = 0.005) -> dict:
    """Is the EOD archive actually the LAST session? Cross-checks the archive's ref_close
    against the BROKER's live prev_close — the same number by definition (both are the
    previous session's close). If they disagree in bulk, the nightly DCM sync is STALE or
    misaligned, which means ref_close / vol_med20 / atr14 / deliv_trail are ALL from the
    wrong session and every signal below them is silently corrupted.

    Holiday-proof by construction: no trading calendar needed — it tests the thing that
    actually matters (does our baseline equal the broker's baseline?).

    SESSION-PHASE AWARE (this matters): the broker's prev_close only ROLLS FORWARD when the
    NEXT session opens. So once the nightly sync has run (every evening, and all weekend) the
    archive's latest row is the just-completed session while the broker still reports the one
    BEFORE it — a legitimate one-session offset, not corruption. Comparing only against the
    archive's latest therefore fired a false "STALE — DO NOT TRADE" every evening/weekend.
    So accept a match against EITHER the archive's latest close OR its prior close; alarm only
    when the broker's baseline matches NEITHER (that is a genuinely mis-synced archive)."""
    n = mism = 0
    has_prev = "prev_close" in getattr(ref, "columns", [])
    for fys, v in quotes.items():
        sym = fys.replace("NSE:", "").replace("-EQ", "")
        if sym not in ref.index:
            continue
        bpc = v.get("prev_close_price")
        apc = ref.loc[sym, "ref_close"]
        if not bpc or apc != apc or float(apc) <= 0:
            continue
        n += 1
        ok = abs(float(bpc) / float(apc) - 1.0) <= tol            # broker == archive LATEST
        if not ok and has_prev:                                   # …or archive PRIOR session
            ppc = ref.loc[sym, "prev_close"]
            if ppc == ppc and float(ppc) > 0:
                ok = abs(float(bpc) / float(ppc) - 1.0) <= tol
        if not ok:
            mism += 1
    pct = (mism / n) if n else 0.0
    return {"checked": n, "mismatch": mism, "pct": round(100 * pct, 1),
            # a few names can differ (corporate actions); a BULK disagreement means the
            # archive is a different session entirely.
            "stale": bool(n >= 20 and pct > 0.20)}


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
    # REGIME GATE on TODAY's index, not yesterday's. The archive runs only through the
    # last close, so an archive lookup would gate today's trade on yesterday's regime —
    # not the rule the backtest validated (it gates on the signal day's OWN close). The
    # regime flips on ~6.7% of sessions and ~7.5% of validated signals land on a flip,
    # i.e. exactly at the turns where the gate earns its keep.
    if date is None and nifty.get("lp"):
        risk_on = regime.is_risk_on_live(nifty.get("lp"), max_stale_days=config.REGIME_MAX_STALE_DAYS)
    # earnings guard: names reporting during the overnight hold can't be a BTST carry
    from . import events
    d0 = (pd.Timestamp(date) if date is not None else data.last_trading_date()).date()
    earn = events.upcoming(d0, horizon_days=3)

    rows = []
    _live_pc: dict = {}                    # symbol -> broker prev_close, for the trigger fetch
    _pending: dict = {}                    # symbol -> legs, for the session-VWAP (cvwap) pass
    _vols: dict = {}                       # symbol -> today's cumulative volume
    _hilo: dict = {}                       # symbol -> (day high, day low)
    ref = uni.set_index("symbol")
    for fys, v in q.items():
        sym = fy.get(fys)
        if not sym or sym not in ref.index:
            continue
        o, h, l, c = v.get("open_price"), v.get("high_price"), v.get("low_price"), v.get("lp")
        pc = v.get("prev_close_price") or float(ref.loc[sym, "ref_close"])
        if None in (o, h, l, c) or not pc:
            continue
        o, h, l, c = float(o), float(h), float(l), float(c)
        # The broker's high/low can momentarily LAG the last trade: we have seen lp print
        # ABOVE high intraday, which makes clr = (c-l)/(h-l) exceed 1.0 and SPURIOUSLY pass
        # the >=CLR_TH strong-close leg — a fabricated footprint from a stale tick. The LTP
        # is by definition inside the day's range, so reconcile the range to it.
        h, l = max(h, c), min(l, c)
        pa = indicators.price_action(o, h, l, c)
        _live_pc[sym] = float(pc)          # broker prev_close (authoritative, stale-proof)
        day_ret = 100 * (float(c) / float(pc) - 1)
        rs = day_ret - idx_ret if idx_ret is not None else None
        vol = v.get("volume") or 0
        med = ref.loc[sym, "vol_med20"]
        # TIME-NORMALISED volume PACE. The raw cum_vol/median_daily is mechanically tiny
        # in the morning (~4% of the day has traded at 09:15), so a genuine 2x-volume day
        # would read 0.27 at 09:30 and fail the 2.0 gate. Dividing by the elapsed volume
        # fraction makes 2.0 mean "on pace for a 2x day" at ANY hour. At/after 15:25 the
        # fraction is 1.0 -> identical to the raw ratio, so the validated close decision
        # (and the 8yr backtest, which used the full-day ratio) is untouched.
        _raw = (vol / float(med)) if (vol and med and med == med) else float("nan")
        _frac = indicators.day_fraction()
        vsurge = (_raw / _frac) if (_raw == _raw and _frac > 0) else float("nan")
        atr14 = float(ref.loc[sym, "atr14"]) if ref.loc[sym, "atr14"] == ref.loc[sym, "atr14"] else 0.0
        lv = indicators.levels(float(c), atr14, day_low=float(l))
        # PERSISTENT RS = the completed RS_LOOKBACK-1 sessions (archive) + today's live RS.
        # The validated leg is the sustained leader, not a one-day burst.
        _c9 = ref.loc[sym, "rs_cum9"]
        rs_cum = (float(_c9) + (rs / 100.0)) if (_c9 == _c9 and rs is not None) else None
        # TODAY's turnover in lacs. The archive's turnover_lacs is exactly
        # volume x avg_price (the day's VWAP), so the honest universe-wide proxy from a
        # quote is the TYPICAL price (h+l+c)/3 — median error 0.15% vs 0.39% for close.
        # For the SHORTLIST the true session VWAP is fetched below, making the gate exact
        # where the decision is actually made.
        turn_lacs = (vol * (h + l + float(c)) / 3.0) / 1e5 if vol else 0.0
        liquid_ok = turn_lacs >= config.LIQ_MIN_LACS
        # cvwap (path signature) needs today's session VWAP, which a 5s quote lacks. It is
        # fetched below for the shortlist only; None here => that leg is NOT met.
        ready = btst_readiness(pa, day_ret, rs_cum, vsurge, cvwap=None)
        sready = short_readiness(pa, day_ret, rs, vsurge)
        dtrail = float(ref.loc[sym, "deliv_trail"]) if ref.loc[sym, "deliv_trail"] == ref.loc[sym, "deliv_trail"] else 0.0
        # short-side levels: stop ABOVE, targets BELOW (mirror of the long geometry)
        s_stop = round(float(c) + atr14, 2) if atr14 > 0 else None
        s_t1 = round(float(c) - atr14, 2) if atr14 > 0 else None
        s_t2 = round(float(c) - 2 * atr14, 2) if atr14 > 0 else None
        _vols[sym] = vol
        _hilo[sym] = (h, l)
        _pending[sym] = (pa, day_ret, rs_cum, vsurge, dtrail, float(c), liquid_ok)
        rows.append({
            "symbol": sym, "time": dt.datetime.now().strftime("%H:%M:%S"),
            "sector": ref.loc[sym, "sector"], "ltp": float(c),
            "day%": round(day_ret, 2), "clr": pa["clr"], "character": pa["character"],
            "body": pa["body"], "vol×": round(vsurge, 2) if vsurge == vsurge else None,
            "RS%": round(rs, 2) if rs is not None else None,
            "rsCum%": round(100 * rs_cum, 2) if rs_cum is not None else None,
            "cvwap%": None,                      # filled for the shortlist below
            "btst": f"{ready}/{BTST_LEGS}", "delivTr": round(dtrail, 1),
            "exp_ON": "",
            "short": f"{sready}/4",
            "entry": lv.get("entry"), "stop": lv.get("stop"),
            "t1": lv.get("t1"), "t2": lv.get("t2"),
            "s_stop": s_stop, "s_t1": s_t1, "s_t2": s_t2,
            "risk%": lv.get("risk%"), "atr%": lv.get("atr%"),
            "band_lo": indicators.band(float(c), atr14).get("band_lo"),
            "band_hi": indicators.band(float(c), atr14).get("band_hi"),
            "earnings": "⚠" if sym in earn else "",
            "turn₹L": round(turn_lacs, 0),
            "action": ("EARNINGS" if sym in earn
                       else _live_action(pa, day_ret, rs_cum, vsurge, risk_on,
                                         deliv_trail=dtrail, cvwap=None, liquid=liquid_ok)),
            "sell": _sell_action(pa, day_ret, rs, vsurge),
        })
    board = pd.DataFrame(rows)
    if not board.empty:
        # ── SHORTLIST PASS: session VWAP -> the PATH-SIGNATURE leg (cvwap) ──────────
        # A 5s quote carries no VWAP, but cvwap is part of the VALIDATED signal_mask —
        # without it the live board is not the backtested edge. Only names already
        # holding the other legs can possibly become BTST-CARRY, so fetch 5-min bars for
        # that handful (cached per 5-min bucket) and finalise their action. This also
        # gives their EXACT trigger time: first-seen would read "when this dashboard
        # started", so opening the laptop at 13:00 would mislabel a 10:40 trigger.
        shortlist = board[board["action"].isin(["BTST-CARRY", "FORMING"])]["symbol"].tolist()
        cvwaps, trigs, trigpx, since, turns, estcl = {}, {}, {}, {}, {}, {}
        for s_ in shortlist:
            pa_, day_, rsc_, vs_, dt_, ltp_, liq_ = _pending[s_]
            ex = _session_extras(s_, _live_pc[s_], ref.loc[s_, "vol_med20"],
                                 rs_cum9=ref.loc[s_, "rs_cum9"])
            vw_ = ex.get("vwap")
            cv_ = ((ltp_ - vw_) / vw_) if (vw_ and vw_ > 0) else None
            cvwaps[s_] = cv_
            # EXACT liquidity for the names that can actually become CARRY: the archive's
            # turnover_lacs IS volume x the day's VWAP, and we now hold that VWAP.
            _v = _vols.get(s_)
            if vw_ and vw_ > 0 and _v:
                _t = (_v * vw_) / 1e5
                turns[s_] = round(_t, 0)
                liq_ = _t >= config.LIQ_MIN_LACS
            if ex.get("trigger"):
                trigs[s_] = ex["trigger"]
            # move SINCE the footprint fired — has the name held/extended, or faded?
            # NOT a P&L: the BTST entry is at the CLOSE, not at the trigger.
            _px = ex.get("trig_px")
            if _px and _px > 0:
                trigpx[s_] = round(_px, 2)
                since[s_] = round(100 * (ltp_ / _px - 1), 2)
            # ── JUDGE THE CLOSE-DECISION ON THE ESTIMATED OFFICIAL CLOSE ──────────────
            # NSE's official close is the VWAP of 15:00-15:30, NOT the last trade — and it
            # is the official close the EOD archive stores and the 8yr backtest gates on.
            # Measured (38,099 stock-days): the last-30-min VWAP tracks it to 1.6 bps, the
            # 15:30 LTP to 14.8 bps. Judging the legs on the LTP therefore evaluates a
            # DIFFERENT price than the one that was validated. Before 15:00 no estimate
            # exists, so the LTP stands and the board is a forecast (as it should be).
            _c30 = ex.get("close30")
            if _c30 and _c30 > 0:
                _h_, _l_ = _hilo[s_]
                _h_, _l_ = max(_h_, _c30), min(_l_, _c30)
                if _h_ > _l_:
                    pa_ = dict(pa_, clr=round((_c30 - _l_) / (_h_ - _l_), 3))
                day_ = 100 * (_c30 / _live_pc[s_] - 1)
                if vw_ and vw_ > 0:
                    cv_ = (_c30 - vw_) / vw_
                    cvwaps[s_] = cv_
                estcl[s_] = round(_c30, 2)
            rdy_ = btst_readiness(pa_, day_, rsc_, vs_, cv_)
            act_ = ("EARNINGS" if s_ in earn else
                    _live_action(pa_, day_, rsc_, vs_, risk_on, deliv_trail=dt_,
                                 cvwap=cv_, liquid=liq_))
            m_ = board["symbol"] == s_
            board.loc[m_, "cvwap%"] = round(100 * cv_, 2) if cv_ is not None else None
            if s_ in estcl:
                board.loc[m_, "est_close"] = estcl[s_]
                board.loc[m_, "clr"] = pa_["clr"]
                board.loc[m_, "day%"] = round(day_, 2)
            if s_ in turns:
                board.loc[m_, "turn₹L"] = turns[s_]
            board.loc[m_, "btst"] = f"{rdy_}/{BTST_LEGS}"
            board.loc[m_, "action"] = act_
            board.loc[m_, "exp_ON"] = ("+0.3–0.4%" if act_ == "BTST-CARRY" else "")
        # 'entered' — exact trigger where we fetched it, first-seen as the fallback
        qual = board[board["action"].isin(["BTST-CARRY", "FORMING"])
                     | board["sell"].isin(["SHORT", "WEAK"])]["symbol"].tolist()
        seen = mark_first_seen(d0, qual)
        board["entered"] = board["symbol"].map(lambda s_: trigs.get(s_) or seen.get(s_))
        board["at"] = board["symbol"].map(trigpx)          # price when it fired
        board["since%"] = board["symbol"].map(since)       # move since it fired

        # ── RISK LAYER on the ACTIONABLE surface ────────────────────────────────
        # This board is what you act on at 15:10-15:30. Until now it applied NONE of the
        # risk controls — no sector cap, no top-N, no calibrated size — so it could show
        # eight CARRY names, five of them one sector (a single macro bet), at full size.
        # Those controls lived only in the EOD board, which you would not see until the
        # NEXT day. Construct the book here, where the decision is actually made.
        board["book"], board["wt%"] = None, None
        carry = board[board["action"] == "BTST-CARRY"]
        if not carry.empty:
            from . import portfolio
            try:                                    # cheap JSON read; never run a backtest here
                from . import calibrate as _cal
                _st = _cal.load_state()
                _mult = _st["size_multiplier"] if _st else 1.0
            except Exception:
                _mult = 1.0
            cand = carry.copy()
            cand["score"] = (                        # mirrors features.conviction_score
                cand["clr"].rank(pct=True)
                + cand["delivTr"].rank(pct=True)
                + cand["vol×"].clip(upper=6).rank(pct=True)
                + cand["day%"].clip(lower=0).rank(pct=True)
                + cand["rsCum%"].rank(pct=True))
            cand = cand.sort_values("score", ascending=False)
            bk = portfolio.select(cand, size_mult=_mult)
            wts = dict(zip(bk["symbol"], bk["weight"]))
            for s_ in cand["symbol"]:
                m_ = board["symbol"] == s_
                if s_ in wts:
                    board.loc[m_, "book"] = "✓ TAKE"
                    board.loc[m_, "wt%"] = round(100 * wts[s_], 1)
                elif _mult <= 0:
                    # NOT a concentration decision — the self-calibrator has vetoed the
                    # whole book (edge decayed / posterior negative). Say so, or the label
                    # would blame the sector cap for a stand-aside.
                    board.loc[m_, "book"] = "✗ STAND ASIDE (calib)"
                    board.loc[m_, "wt%"] = 0.0
                else:                                # dropped by the sector cap or top-N
                    board.loc[m_, "book"] = "✗ capped"
                    board.loc[m_, "wt%"] = 0.0
        board = board.sort_values(["action", "clr"], ascending=[True, False])
    health = archive_health(ref, q)          # is our EOD baseline the broker's baseline?
    return {"ok": True, "status": ts["describe"], "risk_on": risk_on, "archive": health,
            "idx_ret": idx_ret, "board": board, "n": len(board),
            "market_open": market_open()}


def _chp(v: dict):
    return float(v["chp"]) if v and v.get("chp") is not None else None


BTST_LEGS = 5          # the live footprint must have EXACTLY the backtested legs


def btst_readiness(pa: dict, day_ret: float, rs_cum, vsurge, cvwap=None) -> int:
    """How many of the LIVE overnight-footprint legs are met (0-5). These MUST be the
    same legs features.signal_mask() was validated on for 8 years, or the board would
    suggest names the backtest never blessed:

        clr ≥ CLR_TH            strong close (buyers held into the bell)
        day_ret ≥ RET_TH        up on the day (demand in control)
        vol pace ≥ VOL_TH       real participation (time-normalised)
        rs_cum > RS_MIN         PERSISTENT leader — the RS_LOOKBACK-day CUMULATIVE RS,
                                NOT a one-day burst (burst-only decays +30 -> +19bps)
        cvwap ≥ CVWAP_TH        PATH SIGNATURE — closed well above session VWAP, i.e.
                                trended-and-held, not spiked-and-faded (+26 -> +30bps,
                                and it flips 2025 from -11.6 to +6.2)

    Delivery (deliv_trail) and the regime gate are applied by the caller. cvwap=None
    means the session VWAP has not been fetched yet -> that leg is NOT met (no free pass).
    """
    n = 0
    n += pa["clr"] >= config.CLR_TH
    n += day_ret >= 100 * config.RET_TH
    n += vsurge == vsurge and vsurge >= config.VOL_TH        # NaN volume = NOT met (no free pass)
    n += (rs_cum is not None) and (rs_cum > config.RS_MIN)   # persistent, not today's burst
    n += (cvwap is not None) and (cvwap == cvwap) and (cvwap >= config.CVWAP_TH)
    return int(n)


def short_readiness(pa: dict, day_ret: float, rs, vsurge) -> int:
    """Mirror of btst_readiness for the DISTRIBUTION footprint (0-4): weak close,
    down ≥1%, volume surge, relative-strength laggard. High = the day looks like
    supply/distribution. Intraday only — overnight short is proven -EV."""
    n = 0
    n += pa["clr"] <= (1 - config.CLR_TH)           # closed in bottom 30% of range
    n += day_ret <= -100 * config.RET_TH
    n += vsurge == vsurge and vsurge >= config.VOL_TH
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


def _live_action(pa: dict, day_ret: float, rs_cum, vsurge, risk_on: bool,
                 now_time: "dt.time | None" = None, deliv_trail: float = 0.0,
                 cvwap: float | None = None, liquid: bool = True) -> str:
    """Honest live label with the intraday→BTST handoff.

    BTST-CARRY = near the close, the FULL LEAK-FREE footprint holds — the price legs
    (clr/up/vol/RS) AND trailing delivery ≥ threshold (sustained accumulation, known at
    the close). Shift this name to an overnight hold. FORMING = the price footprint is
    building (delivery leg may or may not be there yet). NEUTRAL/AVOID otherwise.

    now_time lets REPLAY pass the point-in-time clock (else wall-clock).
    """
    if not risk_on:
        return "AVOID"
    ready = btst_readiness(pa, day_ret, rs_cum, vsurge, cvwap)
    deliv_ok = deliv_trail >= config.DELIV_TRAIL_TH
    near_close = (now_time or dt.datetime.now().time()) >= dt.time(15, 10)
    # BTST-CARRY must be the EXACT validated footprint: all BTST_LEGS price legs (incl.
    # path-signature + persistent RS) AND trailing delivery AND the regime gate AND the
    # close window. Anything looser would suggest names the 8yr backtest never blessed.
    if ready >= BTST_LEGS and deliv_ok and near_close and liquid:
        return "BTST-CARRY"
    if ready >= BTST_LEGS - 1:
        return "FORMING"             # one leg short (often delivery or VWAP-path) — watch
    if pa["clr"] <= 0.33 or day_ret < 0:
        return "AVOID"
    return "NEUTRAL"


# ── tier 2: per-symbol deep state (VWAP + RSI) ─────────────────────────────────
_RES = {"1D": "D", "4h": "240", "2h": "120", "1h": "60", "15m": "15", "5m": "5",
        "10m": "10", "1m": "1"}
# "1m" and "10m" are the SCALPER lane's frames (eqbtst/scalp.py). Both are native Fyers
# resolutions -- verified against /history at 375 and 38 candles per session -- so they
# need no resampling here. NOTE 375 is not divisible by 10, so a 10-minute series ends
# every session on a FIVE-minute stub; the ">15" guard below deliberately does not fold
# it, and scalp._resample owns that rule for the frames that lane builds.
# lookback days per tf — coarse bars = fewer/day, need more days for ATR14/RSI14/structure(20).
# 4h ≈ 1.5 bars/day → 30d ≈ 45 bars; 2h ≈ 3/day → fine at 15d; intraday minutes plenty at 10d.
# 1D ≈ 0.68 bars/calendar-day → 300d ≈ 200 bars. NOT more: Fyers rejects a daily range beyond
# ~1 year with "Invalid input" (verified), and the request fails CLOSED to an empty frame.
# The "1D": "D" entry is load-bearing. Without it _RES.get(tf, "60") fell through to SIXTY-MINUTE
# bars and labelled them daily — so a positional entry, its ATR stop and its targets were all
# built on hourly candles while the UI said 1D. Silent, plausible-looking, and ~6x too tight.
_LOOKBACK = {"1D": 300, "4h": 30, "2h": 15}


# ── /history RATE PACER ──────────────────────────────────────────────────────────────
# MEASURED, not assumed. 120 names, identical symbols, three configurations, each after a
# 70s cooldown, capturing the broker's own reply rather than an empty frame:
#
#     6 workers, unpaced   -> 450 req/min ->   7 x HTTP 429 "request limit reached"
#     3 workers, 0.35s gap -> 170 req/min ->   0 failures
#     2 workers, 0.25s gap -> 174 req/min ->   0 failures
#
# So the ceiling is a RATE, and ~180 req/min clears it completely. Worse, the budget is a
# ROLLING window that a burst POISONS: a paced run started 65s after an unpaced burst failed
# 63 of 120, while the same paced run from a clean window failed 0. That is exactly the
# production bug -- the scan bursts, 43-48 names 429, and the retry sweep one second later is
# still inside the window the burst poisoned, so it fails too and the blanks persist all day.
#
# A global token bucket in front of every /history call fixes both: no burst to poison the
# window, and the retry has a clean one to run in. Cost is ~81s for 243 names instead of ~40s.
# That is the right trade: a scan that takes twice as long is visibly slower, whereas a scan
# missing 20% of the universe looks complete and silently drops those names out of EVERY
# structure filter. Concurrency is kept (the pool still hides the ~0.7s round-trip); only the
# ISSUE RATE is capped.
_HIST_GAP = 0.33                       # seconds between /history calls -> ~180 req/min
_HIST_LOCK = threading.Lock()
_HIST_LAST = [0.0]


def _hist_pace() -> None:
    """Block until this thread may issue the next /history call. Global across the pool."""
    with _HIST_LOCK:
        wait = _HIST_LAST[0] + _HIST_GAP - time.time()
        if wait > 0:
            time.sleep(wait)
        _HIST_LAST[0] = time.time()


def fetch_intraday(sym: str, tf: str = "1h", lookback_days: int | None = None) -> pd.DataFrame:
    """Candles for one symbol at timeframe tf ('4h','2h','1h','15m','5m') via /history.
    Pulls a few days so ATR14/RSI14 have enough bars on the coarse (hourly+) frames.

    Rate-paced: see _hist_pace. An unpaced fan-out returns 429s that this function would
    otherwise convert into an empty frame, i.e. into a silent 'n/a' structure label."""
    _hist_pace()
    res = _RES.get(tf, "60")
    lb = lookback_days if lookback_days is not None else _LOOKBACK.get(tf, 10)
    d_to = dt.date.today()
    d_from = d_to - dt.timedelta(days=lb)
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
    # The broker's coarse intraday series carries the same trailing session stub our own
    # resample does (NSE's 375-minute session is not divisible by 60 or 120), so it gets the
    # same treatment -- otherwise the board's structure label and the trade card's ATR stop are
    # computed on two different definitions of '1h'. See merge_session_stubs.
    if res.isdigit() and int(res) > 15:
        df = merge_session_stubs(df, int(res))
    return df


_SESS_CACHE: dict = {}          # (symbol, date) -> ({trigger, vwap}, checked_5min_bucket)


def _bucket5(now: dt.datetime | None = None) -> str:
    """The current 5-minute bar bucket — no new intraday information can arrive within
    one, so a miss need not be re-fetched until the bucket turns."""
    n = now or dt.datetime.now()
    return f"{n.hour:02d}:{(n.minute // 5) * 5:02d}"


def _nifty_prev_close() -> float | None:
    """Nifty's previous close (archive) — the baseline for the index's intraday return."""
    key = ("_NIFTYPC", dt.date.today(), "d")
    if key in _SESS_CACHE:
        return _SESS_CACHE[key]
    try:
        nf = data.load_nifty().sort_values("trade_date")
        v = float(nf["close_val"].iloc[-1])
    except Exception:
        v = None
    _SESS_CACHE[key] = v
    return v


def _rs_context(ts_index, rs_cum9) -> tuple | None:
    """(rs_cum9, index-return-so-far at each of the stock's bars) — the running RS leg.
    None if the index series or the archive baseline is unavailable (leg then not applied)."""
    if rs_cum9 is None or rs_cum9 != rs_cum9:
        return None
    nif, npc = _index_intraday_5m(), _nifty_prev_close()
    if nif.empty or not npc:
        return None
    aligned = nif.reindex(pd.Index(ts_index).union(nif.index)).sort_index().ffill()
    aligned = aligned.reindex(pd.Index(ts_index))
    idx_ret = (aligned.to_numpy(float) / npc) - 1.0
    idx_ret = np.nan_to_num(idx_ret, nan=0.0)
    return (float(rs_cum9), idx_ret)


def _session_extras(sym: str, ref_close: float, ref_avg_vol: float | None = None,
                    rs_cum9: float | None = None) -> dict:
    """Today's session facts that a 5-second QUOTE cannot give us, from the 5-min bars:

      trigger : WHEN the footprint first fired (5-min resolution, timeframe-independent —
                a 4h frame only has 09:15/13:15 bars and could never say 12:30).
      vwap    : today's session VWAP — needed for the PATH-SIGNATURE leg (close well
                above VWAP = trended-and-held, vs spiked-and-faded). This leg is in the
                validated signal_mask; without it the live board is not the backtest.

    Cached per 5-min bucket (no new information arrives inside a bar). A fired trigger is
    immutable and is carried forward even as VWAP keeps moving."""
    today = dt.date.today()
    key = (sym, today)
    bucket = _bucket5()
    hit = _SESS_CACHE.get(key)
    if hit is not None and hit[1] == bucket:
        return hit[0]
    prev = hit[0] if hit else {}
    _empty = {"trigger": None, "trig_px": None, "vwap": None}
    try:
        fine = fetch_intraday(sym, tf="5m", lookback_days=2)
        if fine.empty:
            return prev or _empty
        fine = fine[fine["ts"].dt.date == fine["ts"].dt.date.max()]     # today's 5m bars
        _ctx = _rs_context(fine["ts"], rs_cum9)
        trig, trig_px = _formed_at(fine, ref_close, ref_avg_vol, _ctx)
        if prev.get("trigger"):             # once fired today, the trigger is immutable
            trig, trig_px = prev["trigger"], prev.get("trig_px")
        # ESTIMATE OF THE OFFICIAL CLOSE. NSE does not close at the last traded price — the
        # official close is the VWAP of all trades in the final 30 minutes (15:00-15:30),
        # and THAT is the number the EOD archive stores and the 8yr backtest gates on.
        # Measured on 38,099 stock-days: the last-30-min VWAP predicts the official close to
        # 1.6 bps, while the 15:30 LTP is off by 14.8 bps. Gating the live close-decision on
        # the LTP therefore judges a DIFFERENT price than the one that was validated.
        _w = fine[fine["ts"].dt.time >= dt.time(15, 0)]
        close30 = indicators.vwap(_w) if (len(_w) and float(_w["volume"].sum()) > 0) else None
        out = {"trigger": trig, "trig_px": trig_px,
               "vwap": indicators.vwap(fine), "close30": close30}
    except Exception:
        return prev or _empty
    _SESS_CACHE[key] = (out, bucket)
    return out


def _trigger_time(sym: str, ref_close: float, ref_avg_vol: float | None = None) -> str | None:
    """Just the trigger minute (see _session_extras)."""
    return _session_extras(sym, ref_close, ref_avg_vol).get("trigger")


# The canonical daily structure enums are TREND_UP/TREND_DOWN/RANGE/BREAKOUT_UP/
# BREAKOUT_DOWN/CONSOLIDATION. An earlier version of this set said "BREAKDOWN", which
# is not a value indicators.struct_full ever returns -- it silently matched nothing and
# dropped every fresh breakdown from the count. struct_full owns the vocabulary;
# spelling one of its labels from memory is how that happened.
_BEARISH_D = {"TREND_DOWN", "BREAKOUT_DOWN"}


def add_downtrend_age(df: pd.DataFrame, symbols=None, max_back: int = 180) -> pd.DataFrame:
    """How many SESSIONS the name's DAILY structure has been bearish. Zero broker cost.

    WHY A DURATION AND NOT A CLOCK TIME
        A short on this board has no moment in the session. `side` comes from
        synthesize(htf, ltf, ltp) -> side_of(), and SHORT is decided by the HIGHER frame's
        structure ENUM (TREND_DOWN / BREAKDOWN / COIL). Swept densely across and beyond the
        box, not one SHORT-yielding combination changes side with price -- `loc` only bites
        when the higher frame is a RANGE, which never yields SHORT. So the side flips when a
        BAR CLOSES, not when price ticks, and a price-path replay would have stamped 09:15 on
        every short every day. (test_columns.py pins that finding.)

        The honest answer to "when did this short happen" is therefore an AGE, and the
        cleanest frame to measure it on is the DAILY one -- it is what "this name has been in
        a downtrend for three weeks" actually means.

    WHY THE DAILY FRAME AND NOT THE TAB'S OWN 1h/4h
        Cost and reach. _daily_hist() is ONE archive read per day, already warmed for the
        1D/1W structure reads and already back-adjusted for splits and bonuses, so this adds
        no network call at all and works with a dead broker token. The tab's own frames would
        need per-name broker history and the ~60-day intraday limit truncates exactly the
        names that have been bearish longest -- the ones the column is most useful for.
        The trade-off is stated in the column's tooltip: it is a DAILY read beside an
        intraday tab.

    Returns `dn_age` (int sessions, NaN when not bearish or not computable). Never raises:
    a missing name simply gets NaN.
    """
    if df is None or df.empty or "symbol" not in df.columns:
        return df
    out = df.copy()
    if "dn_age" not in out.columns:
        out["dn_age"] = np.nan
    want = [x for x in (symbols if symbols is not None else out["symbol"]) if isinstance(x, str)]
    want = [x for x in dict.fromkeys(want) if x in set(out["symbol"])]
    if not want:
        return out
    try:
        hist = _daily_hist()
    except Exception:                                    # noqa: BLE001
        return out
    if not hist:
        return out

    lb = config.LOOKBACK

    def _age(sym: str):
        g = hist.get(sym)
        if g is None or len(g) < lb + 2:
            return np.nan
        # Walk back one session at a time, relabelling the trailing window each step, and
        # count consecutive bearish closes. Capped so a name bearish for years costs a
        # bounded amount of work -- the cap is reported as ">N" by the caller, not as a
        # smaller number, so it can never UNDERSTATE the age silently.
        n = 0
        for k in range(min(max_back, len(g) - lb)):
            w = g.iloc[len(g) - lb - k:len(g) - k]
            try:
                st_ = indicators.struct_full(w)["struct"]
            except Exception:                            # noqa: BLE001
                break
            if st_ not in _BEARISH_D:
                break
            n += 1
        return float(n) if n else np.nan

    ages = {s_: _age(s_) for s_ in want}
    got = out["symbol"].map(ages)
    out["dn_age"] = pd.to_numeric(out["dn_age"].where(got.isna(), got), errors="coerce")
    return out


def add_short_entry_times(df: pd.DataFrame, htf: str, ltf: str, symbols=None) -> pd.DataFrame:
    """`entered` for the SHORT tab: the bar TODAY at which the name became a SHORT.

    WHY IT CANNOT REUSE add_trigger_times
        That one marks the ACCUMULATION footprint (day_ret >= RET_TH, close-in-range,
        close above VWAP, RS, volume on pace). Every leg is long-directional, so it is None
        on a short by construction -- which is why the column was blank there.

    WHY IT IS NOT A PRICE REPLAY EITHER
        A first attempt replayed today's 5m closes through synthesize() with the boxes held
        fixed. That is wrong for most names: swept over the six real structure enums, the
        large majority of SHORT-yielding combinations are SHORT at EVERY price, because the
        side is decided by the higher frame's structure ENUM. Only a RANGE/CONSOLIDATION
        higher frame under a BREAKOUT_DOWN lower frame actually moves with price. Holding the
        boxes fixed would therefore have stamped the first bar of the session on nearly every
        short -- a fake 09:15.

    WHAT IT ACTUALLY DOES
        Recomputes the STRUCTURE ITSELF at each of today's lower-frame bar closes -- resample
        the 15m series to htf/ltf, run struct_full over the trailing window ENDING AT THAT
        BAR, synthesize, side_of -- and returns the first bar of today whose verdict is
        SHORT. That is the same pipeline the live board runs, evaluated at past bars.

        Resolution is one LOWER-FRAME bar (typically 1h), not 5 minutes, because the side
        cannot change between bar closes for the structure-driven majority.

        None when the name was ALREADY short at today's first bar -- the flip predates today,
        and `dn age` carries that duration instead. None is the honest answer there; a time
        would be inventing one.

    COST: one 15m fetch per name (60 days, the same call mtf_structure makes) and ~7 pure
    struct_full evaluations. NEVER RAISES.
    """
    if df is None or df.empty or "symbol" not in df.columns:
        return df
    from . import mtf as _mtf_mod
    out = df.copy()
    if "entered" not in out.columns:
        out["entered"] = None
    for c in ("at", "since%"):
        if c not in out.columns:
            out[c] = np.nan
    want = [x for x in (symbols if symbols is not None else out["symbol"]) if isinstance(x, str)]
    want = [x for x in dict.fromkeys(want) if x in set(out["symbol"])]
    if not want:
        return out
    _FREQ = {"1h": "60min", "2h": "120min", "4h": "240min"}
    lb = config.LOOKBACK

    def _frame(f, tf):
        return f if tf == "15m" else _resample_ohlcv(f, _FREQ.get(tf, "60min"))

    def _one(sym: str):
        try:
            f = fetch_intraday(sym, tf="15m", lookback_days=_MTF_FETCH_DAYS)
            if f is None or f.empty:
                return sym, (None, None)
            f = indicators.adjust_corporate_actions(f)
            hf, lf = _frame(f, htf), _frame(f, ltf)
            if hf is None or lf is None or len(lf) < lb + 1 or len(hf) < lb + 1:
                return sym, (None, None)
            def _side_at(ts):
                lw = lf[lf["ts"] <= ts].tail(lb)
                hw = hf[hf["ts"] <= ts].tail(lb)
                if len(lw) < 5 or len(hw) < 5:
                    return None, None
                sh, sl = indicators.struct_full(hw), indicators.struct_full(lw)
                px = float(lw["close"].iloc[-1])
                syn = _mtf_mod.synthesize(
                    {"struct": sh["struct"], "hi": sh.get("hi"), "lo": sh.get("lo"),
                     "n": int(sh.get("n", 0))},
                    {"struct": sl["struct"], "hi": sl.get("hi"), "lo": sl.get("lo"),
                     "n": int(sl.get("n", 0))}, px)
                return _mtf_mod.side_of(syn["tag"], syn.get("dir", "NONE")), px

            today = lf["ts"].dt.date.max()
            todays = lf[lf["ts"].dt.date == today]
            prior = lf[lf["ts"].dt.date < today]
            # THE PREVIOUS SESSION'S CLOSE IS THE BASELINE, not "was it short at 09:15".
            # Those are different questions and conflating them loses the most interesting
            # case: a name that broke down ON today's opening bar is short at the first bar,
            # and reporting None for it hides a flip that genuinely happened today. Only a
            # name that was ALREADY short when yesterday closed has a flip predating today.
            was_short = False
            if len(prior):
                _s, _ = _side_at(prior["ts"].iloc[-1])
                was_short = _s == "SHORT"
            if was_short:
                return sym, (None, None)
            for ts in todays["ts"]:
                side, px = _side_at(ts)
                if side == "SHORT":
                    return sym, (pd.Timestamp(ts).strftime("%H:%M"), round(px, 2))
            return sym, (None, None)
        except Exception:                                # noqa: BLE001
            return sym, (None, None)

    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(_SCAN_WORKERS, len(want))) as ex:
            res = dict(ex.map(_one, want))
    except Exception:                                    # noqa: BLE001
        return out

    m = out["symbol"].map(lambda s_: res.get(s_) or (None, None))
    trg, tpx = m.map(lambda t: t[0]), m.map(lambda t: t[1])
    out["entered"] = out["entered"].where(trg.isna(), trg)
    out["at"] = pd.to_numeric(out["at"].where(tpx.isna(), tpx), errors="coerce")
    if "ltp" in out.columns:
        _since = [round(100 * (l / p - 1), 2) if (p and l and p > 0) else np.nan
                  for l, p in zip(out["ltp"], tpx)]
        out["since%"] = pd.to_numeric(
            [n if n == n else o for n, o in zip(_since, out["since%"])], errors="coerce")
    return out


def add_trigger_times(df: pd.DataFrame, symbols=None) -> pd.DataFrame:
    """Attach the CAUSAL footprint trigger (`entered` / `at` / `since%`) to a scan frame.

    WHY THIS IS SEPARATE FROM THE SCAN
        universe_mtf_scan is STRUCTURE-first and deliberately cheap: one 15m fetch per name,
        resampled locally. It never evaluates the BTST footprint, so it has no trigger time
        to report -- which is why the 3-tab universe table shipped with no `entered` column
        at all while every other board had one.

        The trigger cannot be derived from what that scan already holds. It is a 5-MINUTE
        resolution wall-clock read (a 4h bar only exists at 09:15/13:15 and could never
        report a 12:30 trigger), so it needs today's 5m bars -- one more fetch per name.

    WHY IT TAKES A `symbols` SUBSET
        That fetch is the whole cost. Applied to every scanned name it roughly doubles the
        scan and pushes into the Fyers rate limit that already produces the "unreadable, not
        neutral" rows. Applied only to the names that took a SIDE it is ~20-30 fetches.
        Names with no side have no trade whose start could be timed, so the column would be
        an em-dash for them anyway.

    COST AND CACHING
        _session_extras memoises per symbol per 5-minute bucket, so a caller inside a 5s
        fragment pays this once per bucket, not every tick. Fan-out is bounded by the same
        _SCAN_WORKERS pool the scan uses.

    NEVER RAISES. A trigger is context on a structure table; a rate-limited fetch must cost
    you the column, never the board. Missing names simply keep None -> "—" at render.
    """
    if df is None or df.empty or "symbol" not in df.columns:
        return df
    out = df.copy()
    want = [s for s in (symbols if symbols is not None else out["symbol"]) if isinstance(s, str)]
    want = [s for s in dict.fromkeys(want) if s in set(out["symbol"])]
    # `at` / `since%` seed as float NaN, NOT Python None. A column of None is object dtype,
    # and any caller that renders without going through the dashboard's _fmt() coercion
    # prints the literal string "None" in every cell -- which reads as a bug rather than as
    # "nothing fired". NaN renders blank everywhere. `entered` stays object: _fmt turns it
    # into the em-dash that says the same thing in a text column.
    if "entered" not in out.columns:
        out["entered"] = None
    for c in ("at", "since%"):
        if c not in out.columns:
            out[c] = np.nan
    if not want:
        return out

    idx = out.set_index("symbol")

    def _one(sym: str):
        try:
            # `_pc` / `_vol_med20` / `_rs_cum9` are carried by universe_mtf_scan precisely
            # so the footprint can be re-evaluated later without re-deriving them. Falling
            # back to None keeps this usable on frames that lack them.
            r = idx.loc[sym]
            if isinstance(r, pd.DataFrame):        # duplicate symbol -- take the first row
                r = r.iloc[0]
            return sym, _session_extras(sym, r.get("_pc"), r.get("_vol_med20"),
                                        rs_cum9=r.get("_rs_cum9"))
        except Exception:                          # noqa: BLE001
            return sym, {}

    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(_SCAN_WORKERS, len(want))) as ex:
            res = dict(ex.map(_one, want))
    except Exception:                              # noqa: BLE001
        return out

    m = out["symbol"].map(lambda s_: res.get(s_) or {})
    trg = m.map(lambda d: d.get("trigger"))
    tpx = m.map(lambda d: d.get("trig_px"))
    out["entered"] = out["entered"].where(trg.isna(), trg)
    out["at"] = pd.to_numeric(
        out["at"].where(tpx.isna(), tpx.map(lambda v: round(v, 2) if v else np.nan)),
        errors="coerce")
    if "ltp" in out.columns:
        _since = [round(100 * (l / p - 1), 2) if (p and l and p > 0) else np.nan
                  for l, p in zip(out["ltp"], tpx)]
        out["since%"] = pd.to_numeric(
            [n if n == n else o for n, o in zip(_since, out["since%"])], errors="coerce")
    return out


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
    _nf = _fetch_quotes([config.NIFTY_FYERS]).get(config.NIFTY_FYERS, {})
    idx_ret = _chp(_nf)
    if date is None and _nf.get("lp"):          # gate on TODAY's index, not yesterday's
        risk_on = regime.is_risk_on_live(_nf.get("lp"), max_stale_days=config.REGIME_MAX_STALE_DAYS)
    # pre-filter: up on the day + closing the daily bar in the upper half
    pre_up, pre_dn = [], []
    for fys, v in q.items():
        sym = fys.replace("NSE:", "").replace("-EQ", "")
        if sym not in uni.index:
            continue
        c, pc = v.get("lp"), v.get("prev_close_price") or uni.loc[sym, "ref_close"]
        h, l = v.get("high_price"), v.get("low_price")
        if None in (c, pc, h, l) or h == l:
            continue
        h, l = max(float(h), float(c)), min(float(l), float(c))   # broker range can lag the LTP
        # LIQUIDITY FLOOR — today's turnover (volume x typical price). liquid_universe no
        # longer pre-filters on the archive's STALE t-1 turnover (that dropped 23% of the
        # validated signals), and quotes_board gained a today-turnover gate — but tf_scan
        # was left with NO floor at all, so it surfaced unfillable names complete with
        # entry/stop/targets. A level you cannot get filled at is worse than no level.
        _vol = v.get("volume") or 0
        if ((_vol * (h + l + float(c)) / 3.0) / 1e5) < config.LIQ_MIN_LACS:
            continue
        day = 100 * (float(c) / float(pc) - 1)
        clr = (float(c) - float(l)) / (float(h) - float(l)) if h > l else 0.5
        # TWO pre-filters, one per side. The long screen admits only UP-and-strong names —
        # and feeding the SHORT tab from that same list made it STRUCTURALLY BLIND: a name
        # down 3% closing at its low could never appear, because the shortlist only ever
        # contained names up >=1%. The short side gets the mirror screen (down, closing
        # weak) so the weakness tab can actually show weakness.
        if day >= 100 * config.RET_TH and clr >= 0.5:
            pre_up.append((sym, day, clr, float(pc)))
        elif day <= -100 * config.RET_TH and clr <= 0.5:
            pre_dn.append((sym, day, clr, float(pc)))
    pre_up.sort(key=lambda x: (x[2], x[1]), reverse=True)          # strongest closes first
    pre_dn.sort(key=lambda x: (x[2], x[1]))                        # weakest closes first
    pre = pre_up[:max_names] + pre_dn[:max(4, max_names // 2)]

    rows = []
    for sym, day, _, live_pc in pre:
        # prev_close MUST be the broker's live prev_close (authoritative), not the EOD
        # archive's ref_close: if the nightly archive sync is stale, ref_close is the
        # WRONG session's close and every day%/trigger below it is silently corrupted.
        ds = deep_state(sym, tf=tf, ref_close=live_pc,
                        ref_avg_vol=None, idx_ret=idx_ret)
        if not ds:
            continue
        s, lv = ds["state"], ds["levels"]
        atr_tf = ds.get("atr_tf", 0.0)
        ltp = s["ltp"]
        cndl = ds.get("candles")
        _tfmin = {"1D": 1440, "4h": 240, "2h": 120, "1h": 60, "15m": 15, "5m": 5}.get(tf, 60)
        bar_time = None
        if cndl is not None and len(cndl):
            o = cndl["ts"].iloc[-1]
            close_t = min(o + pd.Timedelta(minutes=_tfmin), o.normalize() + pd.Timedelta("15h30min"))
            bar_time = f"{o.strftime('%H:%M')}-{close_t.strftime('%H:%M')}"   # candle span (open->close)
        # TRIGGER TIME — the wall-clock minute the footprint first fired, at 5-MIN
        # resolution, INDEPENDENT of the selected timeframe. Must not come from the tf
        # bars: a 4h bar only exists at 09:15/13:15, so it could never report a 12:30
        # trigger. The timeframe governs the structure/RSI/levels read, NOT the clock.
        _ex = _session_extras(sym, live_pc, uni.loc[sym, "vol_med20"],
                              rs_cum9=uni.loc[sym, "rs_cum9"])
        entered = _ex.get("trigger")
        _tpx = _ex.get("trig_px")
        at_px = round(_tpx, 2) if _tpx else None
        since_pct = round(100 * (ltp / _tpx - 1), 2) if (_tpx and _tpx > 0) else None
        # is the current tf bar still FORMING? (a mid-candle trigger can repaint)
        forming = None
        if cndl is not None and len(cndl):
            _o = cndl["ts"].iloc[-1]
            _close_t = min(_o + pd.Timedelta(minutes=_tfmin),
                           _o.normalize() + pd.Timedelta("15h30min"))
            forming = dt.datetime.now() < _close_t.to_pydatetime()
        # short-side levels on this timeframe (stop ABOVE, targets BELOW)
        s_stop = round(ltp + atr_tf, 2) if atr_tf > 0 else None
        s_t1 = round(ltp - atr_tf, 2) if atr_tf > 0 else None
        s_t2 = round(ltp - 2 * atr_tf, 2) if atr_tf > 0 else None
        _mtf = mtf_structure(sym)               # 6-TF structure, one cached 15m fetch
        rows.append({
            "symbol": sym, "entered": entered, "at": at_px, "since%": since_pct,
            "time": bar_time,
            "bar": ("⏳ forming" if forming else "✓ closed") if forming is not None else None,
            "s15m": _mtf["15m"], "s1h": _mtf["1h"], "s2h": _mtf["2h"],
            "s4h": _mtf["4h"], "s1D": _mtf["1D"], "s1W": _mtf["1W"],
            "sector": uni.loc[sym, "sector"], "ltp": ltp,
            "day%": s["day_ret"], "structure": s["structure"], "bar_clr": s["bar_clr"],
            "character": s["character"], "vs_vwap%": s["vs_vwap"],
            "above_vwap": s["above_vwap"], "rsi7": s["rsi7"], "rsi14": s["rsi14"],
            "tone": s["tone"], "RS%": s.get("rs_vs_index"),
            "entry": lv.get("entry"), "stop": lv.get("stop"),
            "t1": lv.get("t1"), "t2": lv.get("t2"),
            "s_stop": s_stop, "s_t1": s_t1, "s_t2": s_t2, "atr%": lv.get("atr%"),
            "action": _tf_action(s, risk_on), "sell": _tf_sell_action(s, risk_on),
            # Carried for the CHEAP tier-1 refresh: everything the LEVELS and the VERDICT
            # are built from that does NOT change tick-to-tick. With these on the row, one
            # batch quote can rebuild bar_clr / vs_vwap / RS / action — so the table never
            # shows a live price beside a stale LONG.
            "_atr_tf": atr_tf, "_pc": live_pc,
            "_daylow": float(cndl["low"].min()) if (cndl is not None and len(cndl)) else None,
            "_bar_h": float(cndl["high"].iloc[-1]) if (cndl is not None and len(cndl)) else None,
            "_bar_l": float(cndl["low"].iloc[-1]) if (cndl is not None and len(cndl)) else None,
            "_vwap": s.get("vwap"),
        })
    board = pd.DataFrame(rows)
    if not board.empty:
        board = board.sort_values(["action", "bar_clr"], ascending=[True, False])
    # WHEN this snapshot was taken. The tf scan is heavy (~70 /history calls) so it CANNOT
    # tick like the 5s quote board — and the dashboard's tf branch never reaches the
    # auto-refresh fragment, so the table is frozen until the user re-scans. The ltp and
    # every level derived from it (entry/stop/T1/T2) age silently. Surface the age.
    return {"ok": True, "status": ts["describe"], "tf": tf, "risk_on": risk_on,
            "idx_ret": idx_ret, "board": board, "n_scanned": len(pre),
            "scanned_at": dt.datetime.now()}


_REPLAY_CACHE = Path(__file__).resolve().parent.parent / "data" / "replay"


_DUPE_NOTE = """The broker returns BYTE-IDENTICAL duplicate candles when range_from == range_to.

Verified on RELIANCE for 2026-08-25: the 15m series came back 50 rows / 25 unique with 24 rows
exact duplicates of another row; 1m came back 750 / 375; 5m 150 / 75. A multi-day range does
NOT do it -- only the degenerate single-date form, which is exactly what _fetch_day_candles and
fetch_intraday_range use, and neither of them de-duplicated.

Nothing about a doubled series looks wrong. The range BOX is unchanged (min and max do not care
how many times a bar appears), so every structure label still renders and every chart still
draws -- while every VOLUME figure doubles, and a 20-bar structure window silently covers only
TEN real bars. Replay has been reading half the history it claimed to."""


def _dedupe_candles(f: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate timestamps from a broker candle frame. See _DUPE_NOTE."""
    if f is None or getattr(f, "empty", True) or "ts" not in getattr(f, "columns", []):
        return f
    return f.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)


def _fetch_day_candles(date, resolution: str = "15") -> pd.DataFrame:
    """All liquid-universe intraday candles for a past trading DATE (long format:
    symbol, ts, ohlcv). Cached to parquet per (date, res) — fetched once, then any
    replay time slices it instantly. Heavy first call (~250 /history requests)."""
    d = pd.Timestamp(date).date()
    cache = _REPLAY_CACHE / f"{d}_{resolution}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    uni = liquid_universe(date)
    frames = []
    for sym in uni["symbol"]:
        try:
            r = requests.get(config.FYERS_HISTORY_URL,
                             headers={"Authorization": _auth_header(), "version": "3"},
                             params={"symbol": fy_symbol(sym), "resolution": resolution,
                                     "date_format": "1", "range_from": d.isoformat(),
                                     "range_to": d.isoformat(), "cont_flag": "1"}, timeout=15)
            j = r.json()
            if j.get("s") != "ok" or not j.get("candles"):
                continue
            f = pd.DataFrame(j["candles"], columns=["ts", "open", "high", "low", "close", "volume"])
            f["ts"] = pd.to_datetime(f["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            f = _dedupe_candles(f)      # range_from == range_to doubles every bar -- _DUPE_NOTE
            f["symbol"] = sym
            frames.append(f)
        except Exception:
            continue
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        _REPLAY_CACHE.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache, index=False)
    return out


def _fetch_tf_history(date, tf: str, lookback_days: int = 16) -> pd.DataFrame:
    """Multi-day tf candles ENDING at `date` (long format symbol,ts,ohlcv), for the
    coarse-tf structure/RSI read in replay — a single day has too few 2h/4h bars.
    Cached per (date, tf). Heavy first call (~250 /history requests), like the day fetch."""
    d = pd.Timestamp(date).date()
    res = _RES.get(tf, "60")
    cache = _REPLAY_CACHE / f"{d}_tfhist_{tf}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    d_from = (d - dt.timedelta(days=lookback_days)).isoformat()
    uni = liquid_universe(date)
    frames = []
    for sym in uni["symbol"]:
        try:
            r = requests.get(config.FYERS_HISTORY_URL,
                             headers={"Authorization": _auth_header(), "version": "3"},
                             params={"symbol": fy_symbol(sym), "resolution": res,
                                     "date_format": "1", "range_from": d_from,
                                     "range_to": d.isoformat(), "cont_flag": "1"}, timeout=15)
            j = r.json()
            if j.get("s") != "ok" or not j.get("candles"):
                continue
            f = pd.DataFrame(j["candles"], columns=["ts", "open", "high", "low", "close", "volume"])
            f["ts"] = pd.to_datetime(f["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            f = _dedupe_candles(f)      # range_from == range_to doubles every bar -- _DUPE_NOTE
            f["symbol"] = sym
            frames.append(f)
        except Exception:
            continue
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        _REPLAY_CACHE.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache, index=False)
    return out


# ── multi-timeframe structure (one cheap fetch, five resamples, archive for D/W) ──
_MTF_FETCH_DAYS = 60           # calendar days of 15m history behind every resampled frame
_MTF_CACHE: dict = {}          # (sym, date, bucket5) -> {tf: structure}
_DAILY_HIST: dict = {}         # date -> {sym: daily OHLC frame}  (one archive read per day)


def _daily_hist() -> dict:
    """Per-symbol DAILY bars for the whole universe, ~14 months back — ONE archive read per
    day, then free. Feeds the 1D and 1W structure reads (leak-free: through the last close;
    no forming daily bar, so unlike the intraday TFs these never repaint intraday)."""
    key = dt.date.today()
    if key in _DAILY_HIST:
        return _DAILY_HIST[key]
    try:
        start = (pd.Timestamp(key) - pd.Timedelta(days=430)).strftime("%Y-%m-%d")
        df = data.load_eod(start=start)[["symbol", "trade_date", "open_price",
                                         "high_price", "low_price", "close_price"]]
        df = df.rename(columns={"trade_date": "ts", "open_price": "open",
                                "high_price": "high", "low_price": "low",
                                "close_price": "close"})
        # BACK-ADJUST at the source. The archive is unadjusted, so a split/bonus/demerger is
        # a raw price cliff — measured 26 of 268 names — and it manufactures fake TREND_DOWN
        # labels plus phantom S/R at pre-split prices. Fixing it here means every downstream
        # reader (1D structure, 1W structure, S/R walls) is clean by construction.
        out = {s: indicators.adjust_corporate_actions(
                   g.sort_values("ts").reset_index(drop=True))
               for s, g in df.groupby("symbol")}
    except Exception:
        out = {}
    if not out:
        # DO NOT CACHE A FAILED READ. The archive is DuckDB: one writer excludes readers, so a
        # scan that lands while DCM is syncing raises here — and caching {} pinned EVERY name's
        # 1D and 1W structure to 'n/a' for the WHOLE DAY, silently, with no retry and no signal.
        # A transient lock must cost one scan, not a session. Returning uncached means the next
        # call retries; the day-keyed cache still holds once a read genuinely succeeds.
        return {}
    _DAILY_HIST.clear()                     # never hold two days of frames in memory
    _DAILY_HIST[key] = out
    return out


_MONTHLY_HIST: dict = {}       # date -> {sym: monthly OHLC frame}  (deep read, once per day)


def _monthly_hist() -> dict:
    """Per-symbol MONTHLY bars for the whole universe, ~6 years back — the source for the
    Positional horizon's one-higher-frame (1M) big-wall. SEPARATE from the 14-month _daily_hist
    because a monthly S/R level is a multi-year object: measured, 14 months of monthly bars
    put a >=2-touch wall on 36% of names, ~60 months on 99%. Corporate-action adjusted at the
    source like _daily_hist. One archive read per day, then free. Failed reads are NOT cached
    (a DuckDB lock costs one scan, not a session).

    CA RISK IS AMPLIFIED HERE, ACCEPTED. adjust_corporate_actions back-adjusts any >25% single-
    day step; over this 6-year window 110 names have one (mostly real splits/demergers, correctly
    adjusted). A genuine NEWS crash among them (PAYTM/IDEA-class) is wrongly adjusted, and unlike
    the 60-DAY daily window -- where it scrolls out in ~2 months -- here it can sit in the window
    for years, leaving phantom OLD monthly levels at shifted prices. Impact is bounded: the
    big-wall shows the NEAREST monthly level in the trade direction, and a phantom old level is
    usually far from current price, so it rarely becomes the one shown. Not fixed (no price-only
    rule separates a demerger from a crash -- see adjust_corporate_actions), documented."""
    key = dt.date.today()
    if key in _MONTHLY_HIST:
        return _MONTHLY_HIST[key]
    try:
        start = (pd.Timestamp(key) - pd.Timedelta(days=2400)).strftime("%Y-%m-%d")   # ~6.5yr
        df = data.load_eod(start=start)[["symbol", "trade_date", "open_price",
                                         "high_price", "low_price", "close_price"]].rename(
            columns={"trade_date": "ts", "open_price": "open", "high_price": "high",
                     "low_price": "low", "close_price": "close"})
        out = {s: monthly_frame(indicators.adjust_corporate_actions(
                   g.sort_values("ts").reset_index(drop=True)))
               for s, g in df.groupby("symbol")}
    except Exception:
        out = {}
    if not out:
        return {}
    _MONTHLY_HIST.clear()
    _MONTHLY_HIST[key] = out
    return out


_DELIV_MOM: dict = {}          # date -> DataFrame(symbol, wtd_deliv7, avg_deliv100, deliv_vs_100d)


def deliv_momentum(date=None) -> pd.DataFrame:
    """Per-stock turnover-weighted DELIVERY metrics — ported verbatim from the DCM
    sector-rotation view (Daily_Cash_Market/src/analytics/sector_rotation.py):

      • wtd_deliv7    = 7-CALENDAR-day turnover-weighted delivery %
                        SUM(deliv% x turnover) / SUM(turnover)   — 'smart-money' conviction now
      • avg_deliv100  = 100-TRADING-day turnover-weighted delivery % baseline (strictly BEFORE
                        as_of), same units so the comparison is apples-to-apples
      • deliv_vs_100d = (wtd_deliv7 / avg_deliv100 - 1) x 100    — recent vs OWN historical norm
                        (+15 = delivery 15% above its own 100D norm; -10 = fading interest)

    ONE archive read per day (cached in _DELIV_MOM), off the 5s path. Leak-free for a next-
    session entry: as_of = the last COMPLETED close, whose delivery is already published."""
    key = (pd.Timestamp(date).date() if date is not None else data.last_trading_date().date())
    hit = _DELIV_MOM.get(key)
    if hit is not None:
        return hit
    as_of = pd.Timestamp(key)
    start = (as_of - pd.Timedelta(days=170)).strftime("%Y-%m-%d")   # ~100 trading days + buffer
    df = data.load_eod(start=start, end=as_of.strftime("%Y-%m-%d"))
    df = df[(df["turnover_lacs"] > 0) & df["deliv_per"].notna()]
    recent_cut = as_of - pd.Timedelta(days=7)                        # DCM uses 7 CALENDAR days

    def _wtd(x):
        t = x["turnover_lacs"].sum()
        return float((x["deliv_per"] * x["turnover_lacs"]).sum() / t) if t > 0 else np.nan

    rows = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("trade_date")
        rec = g[g["trade_date"] > recent_cut]                       # last 7 cal days (incl as_of)
        hist = g[g["trade_date"] < as_of].tail(100)                 # 100 trading days BEFORE as_of
        w7 = _wtd(rec) if len(rec) else np.nan
        w100 = _wtd(hist) if len(hist) else np.nan
        vs = (w7 / w100 - 1) * 100 if (w100 and w100 > 0 and not np.isnan(w7)) else np.nan
        rows.append({"symbol": sym, "wtd_deliv7": w7, "avg_deliv100": w100, "deliv_vs_100d": vs})
    res = pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame(
        columns=["wtd_deliv7", "avg_deliv100", "deliv_vs_100d"])
    _DELIV_MOM.clear()                                              # never hold two days of frames
    _DELIV_MOM[key] = res
    return res


_DELIV_WK: dict = {}          # (as_of, n_weeks, base_days) -> DataFrame(symbol -> weekly trend)
_DELIV_WK_NORM = 30           # DEFAULT baseline in trading days (see DELIV_BASE_BY_HORIZON)
_DELIV_WK_MINHIST = 15        # ... and at least this many days, or no base is published

# BASELINE PER TRADE HORIZON. The user's instinct — that a 100-day baseline is too long for a
# short hold — is MEASURED CORRECT; the specific ladder they proposed is not, and the reason is
# sampling noise. Two independent tests, 2018-2026:
#
#   (a) universe-wide IC of the deviation vs forward return, t-stat clustered by date:
#         horizon        base5  base15  base30  base45  base60  base100
#         Intraday(1d)    3.03    4.80    5.21    5.53    5.22    5.08
#         BTST(overnight) 1.27    2.18    2.83    2.27    2.43    1.64
#         Swing(5d)       3.67    7.22    6.67    7.00    5.81    5.34
#         Positional(20d) 1.70    2.51    3.57    3.24    1.45    1.42
#   (b) as a refiner ON this engine's own footprint triggers (n=692, the BTST use case),
#       HIGH-minus-low third of the deviation:
#         5d +14.1 (t1.31) · 15d +6.3 (t0.54) · 30d +23.1 (t2.07, 7/9yr) · 60d +19.4 · 100d +21.3
#
# base=5 is the WORST row in (a) everywhere — a 4-day reading against a 5-day baseline is two
# tiny samples disagreeing, which is why the Intraday setting is 15 and not the 5 first proposed.
# Everything from 15 to 60 sits on a broad plateau, so the ladder below is a DESIGN choice made
# inside a measured-safe band rather than an optimum: it rises monotonically with the hold, which
# is easy to remember and to explain, and it costs nothing detectable (Swing 45d IC t=7.00 vs
# 6.67 at 30d — better on IC, marginally worse on decile spread, both inside noise).
# THE ONE WEAK CELL IS POSITIONAL 60d (t=1.45, against 3.57 at 30d). It is kept because the
# ladder is a preference and Positional is not a horizon this engine trades; if that ever
# changes, that is the cell to revisit first.
DELIV_BASE_BY_HORIZON = {
    "intraday":   15,   # raised from the proposed 5 — base5 measured worst on every horizon
    "btst":       30,   # the best cell for the one horizon this engine actually trades
    "swing":      45,
    "positional": 60,   # weakest cell in the grid; kept for the clean ladder, see above
}

# BUCKET GRANULARITY also follows the hold. For a one-night carry, five WEEKLY buckets smear
# away the detail that matters — what the last five SESSIONS did. For a multi-week hold, daily
# buckets are noise and weeks are the readable unit.
DELIV_BUCKET_BY_HORIZON = {
    "intraday": "day", "btst": "day", "swing": "week", "positional": "week",
}


def deliv_weeks(date=None, n_weeks: int = 5, base_days: int | None = None,
                bucket: str = "week") -> pd.DataFrame:
    """Turnover-weighted delivery % for each of the last `n_weeks` WEEKS, per symbol,
    plus that stock's OWN long-run norm — the 'are the big players building or leaving' read.

    WHY THE WEEKLY SEQUENCE AND NOT ONE NUMBER. Measured on the archive: a single week moves a
    median 5.0pp week-over-week while the 5-week span is 12.8pp. One week is mostly noise; the
    DIRECTION across five is the readable part. Hence a series, not a point.

    WHY RAW % *AND* A NORM, WHICH IS THE WHOLE DESIGN QUESTION.
      Raw delivery % is NOT comparable across names. The universe's own 100-day norms run from
      15% to 66% (p5 27.9, median 47.5, p95 59.3), so an identical reading means opposite
      things on two rows of the same table. Measured example, both printing ~50% last week:
          VMM         51, 49, 60, 38, 51   norm 64%  ->  -21% vs its own norm  (distribution)
          HINDCOPPER  37, 33, 36, 43, 48   norm 22%  -> +117% vs its own norm  (accumulation)
      But converting the SERIES to ratios destroys the magnitude a trader reads ("this name
      delivers 60%"), and the trend is already self-normalising because it is one stock against
      itself. So: keep the series in raw percentage points, and carry the norm in the same cell
      as the anchor that makes the level comparable. Both questions answered, one column.

    THE BASELINE DELIBERATELY EXCLUDES THE WEEKS ON DISPLAY. It is the 100 trading days ending
    BEFORE the first shown week. Comparing five weeks against a norm that contains those same
    five weeks is self-referential and drags the baseline toward whatever just happened, muting
    exactly the move you are looking for. NOTE this is therefore NOT the same baseline as the
    `deliv_vs_100d` column, whose avg_deliv100 is the 100 days before as_of and so overlaps the
    recent weeks by ~25 days. Two similar-sounding numbers, two different questions.

    LEAK-FREE, and this is the leg that has bitten this project before: NSE publishes delivery
    ~6pm, so TODAY's figure does not exist at a 15:15 decision. Everything here is read from the
    archive through `date` (the last COMPLETED, published session), and the newest week is
    flagged `partial` whenever its Friday has not passed — so a Wednesday reading is honestly a
    three-day week, never dressed up as a full one.

    Cached per as-of DATE (never on today()), so Replay gets its own answer.
    """
    base_days = int(base_days or _DELIV_WK_NORM)
    bucket = "day" if str(bucket).startswith("d") else "week"
    key = (pd.Timestamp(date).date() if date is not None else data.last_trading_date().date(),
           int(n_weeks), base_days, bucket)
    hit = _DELIV_WK.get(key)
    if hit is not None:
        return hit
    as_of = pd.Timestamp(key[0])
    start = (as_of - pd.Timedelta(days=330)).strftime("%Y-%m-%d")   # 5wk + 100td baseline + slack
    df = data.load_eod(start=start, end=as_of.strftime("%Y-%m-%d"))
    df = df[(df["turnover_lacs"] > 0) & df["deliv_per"].notna()]
    if df.empty:
        res = pd.DataFrame(columns=["norm", "partial", "cell"])
        _DELIV_WK.clear(); _DELIV_WK[key] = res
        return res

    df = df.copy()
    if bucket == "day":
        # ONE BUCKET PER SESSION. A weekly grid smears exactly the detail a one-night carry
        # needs. Within a single day the turnover weighting is a no-op, so a bucket is just
        # that day's delivery %. The date axis is GLOBAL so every row lines up; a name that did
        # not trade a given session shows a dash rather than silently shifting its history left.
        slots = sorted(df["trade_date"].unique())[-n_weeks:]
        grid = (df[df["trade_date"].isin(slots)]
                .pivot_table("deliv_per", "symbol", "trade_date").reindex(columns=slots))
        cut = pd.Timestamp(slots[0])
    else:
        df["wk"] = df["trade_date"].dt.to_period("W-FRI")
        wk = (df.assign(_num=df["deliv_per"] * df["turnover_lacs"])
                .groupby(["symbol", "wk"], observed=True)[["_num", "turnover_lacs"]].sum())
        wk = (wk["_num"] / wk["turnover_lacs"].replace(0, np.nan)).rename("d").reset_index()
        slots = sorted(wk["wk"].unique())[-n_weeks:]
        if slots:
            grid = wk[wk["wk"].isin(slots)].pivot(index="symbol", columns="wk", values="d")
            grid = grid.reindex(columns=slots)
            cut = pd.Timestamp(slots[-1].start_time)
    if not len(slots):
        res = pd.DataFrame(columns=["norm", "partial", "cell"])
        _DELIV_WK.clear(); _DELIV_WK[key] = res
        return res
    weeks = slots

    # BASELINE = the `base_days` trading days ending immediately BEFORE the CURRENT week.
    # It therefore overlaps the older weeks on display, and that is deliberate: the earlier
    # design gapped it out (ended it before ALL five weeks) to stop a surge dragging its own
    # yardstick, but gapping measured WORSE at both window lengths tested — 30d: +23.1 adjacent
    # vs +12.7 gapped; 100d: +21.3 vs +16.6. Adjacent is also the plainer statement: "this week
    # against this stock's recent normal", where recent means the past month or so.
    hist = df[df["trade_date"] < cut].sort_values("trade_date")
    def _norm(g):
        g = g.tail(base_days)
        t = g["turnover_lacs"].sum()
        return float((g["deliv_per"] * g["turnover_lacs"]).sum() / t) if (
            t > 0 and len(g) >= min(_DELIV_WK_MINHIST, base_days)) else np.nan
    norm = (hist.groupby("symbol").apply(_norm, include_groups=False)
            if not hist.empty else pd.Series(dtype=float))
    out = grid.copy()
    out.columns = [f"w{i+1}" for i in range(len(weeks))]
    out["norm"] = norm.reindex(out.index)
    if bucket == "day":
        out["partial"] = False          # a session is complete once its delivery is published
        out["cur_days"] = 0
    else:
        # a week is finished once the CALENDAR has passed its Friday (holiday-proof, same rule
        # as weekly_frame) — otherwise the newest column is a part-week and must say so.
        out["partial"] = bool(as_of < pd.Timestamp(weeks[-1].end_time).normalize())
        # trading days actually printed in the newest week, PER SYMBOL — a "week" one session
        # old is the noisiest number in the cell and must not look like a full one.
        ndays = df[df["wk"] == weeks[-1]].groupby("symbol").size()
        out["cur_days"] = ndays.reindex(out.index).fillna(0).astype(int)

    wcols = [c for c in out.columns if c.startswith("w") and c[1:].isdigit()]

    # WHAT THE % COMPARES. In WEEK mode it is the newest week — the leading number in the cell.
    # In DAY mode it is the last 5 SESSIONS TOGETHER, not the newest single day: measured on the
    # archive, a one-day reading against the base swings a median 14.9pp per session versus
    # 3.6pp for a five-day one — 4.1x jumpier. A number that repaints that hard every morning
    # is not a read, so the daily view keeps the five buckets for the eye and aggregates them
    # for the %.
    if bucket == "day":
        recent = (df[df["trade_date"].isin(weeks)]
                  .assign(_n=lambda x: x.deliv_per * x.turnover_lacs)
                  .groupby("symbol")[["_n", "turnover_lacs"]].sum())
        recent = (recent["_n"] / recent["turnover_lacs"].replace(0, np.nan)).reindex(out.index)
    else:
        recent = pd.Series(
            [next((v for v in reversed([r[c] for c in wcols]) if pd.notna(v)), np.nan)
             for _i, r in out.iterrows()], index=out.index)
    out["recent"] = recent

    # ONE definition of the deviation, used by BOTH the column and the rendered string. They
    # were computed separately at first — the same duplicate-formula pattern that caused the
    # drift bugs in the sector-tilt port. Derive once, format from it.
    out["dev_pct"] = np.where(
        out["norm"].notna() & (out["norm"] > 0) & out["recent"].notna(),
        (out["recent"] / out["norm"].replace(0, np.nan) - 1.0) * 100.0, np.nan)

    def _cell(r):
        vals = [r[c] for c in wcols]                      # stored oldest -> newest
        if not any(pd.notna(v) for v in vals):
            return "—"
        # DISPLAY IS NEWEST FIRST. The eye lands on the leftmost number, and the current week is
        # the decision-relevant one; the four behind it are the context. Note this reverses the
        # chart convention, so a RISING delivery trend reads as DESCENDING numbers -- the tooltip
        # says so explicitly, and the current week is marked, so there is no ambiguity about
        # which end is now.
        seq = ["–" if pd.isna(v) else f"{v:.0f}" for v in vals][::-1]
        # only mark a week that HAS a reading — a name that did not trade this week shows a
        # dash, and calling that dash "partial" would dress absence up as an in-progress figure
        if r["partial"] and pd.notna(vals[-1]):
            seq[0] += f"*{int(r['cur_days'])}d" if r["cur_days"] else "*"
        body = ", ".join(seq)
        # THE TAIL MUST NAME ITSELF. Shipped first as a bare "+1%" hanging off five percentages,
        # which reads like a week-over-week change, a slope, or a return — three things it is
        # not. It is the CURRENT week measured against this stock's own long-run BASE, so the
        # cell now says "base 40 (+1%)": the absolute anchor restored (you can see what normal
        # looks like for this name) with the comparable relative form beside it.
        # RELATIVE, not percentage points: +19pp on a 29% base (+66%) and +19pp on a 60% base
        # (+32%) are different events, and only the relative form makes two rows comparable.
        if pd.isna(r["dev_pct"]):
            return f"{body}  Base -> (n/a)"
        return f"{body}  Base -> ({r['dev_pct']:+.0f}%)"

    out["cell"] = out.apply(_cell, axis=1)
    # Keep one entry PER HORIZON for the current as-of date, and drop every other date. The
    # earlier clear()-everything held a single entry, so flipping the horizon dropdown recomputed
    # each time (~0.25s); stale DATES must still go, or a replay session would pin yesterday.
    for k in [k for k in _DELIV_WK if k[0] != key[0]]:
        _DELIV_WK.pop(k, None)
    if len(_DELIV_WK) > 8:
        _DELIV_WK.pop(next(iter(_DELIV_WK)), None)
    _DELIV_WK[key] = out
    return out


def mtf_structure(sym: str) -> dict:
    """Kaufman structure on SIX timeframes for one name — {15m,1h,2h,4h,1D,1W} — from ONE
    15-minute fetch (~20 days, resampled locally to 1h/2h/4h) plus the daily archive
    (resampled W-FRI for weekly). NOT six fetches. Cached per 5-min bucket.

    HONESTY: the intraday TFs include today's forming bar, so they can REPAINT until that
    bar closes; 1D/1W are through the LAST CLOSE (they cannot repaint intraday, but also do
    not see today). The WEEKLY frame drops an incomplete final week: a part-formed weekly
    bar spans fewer sessions, so it reads narrower and faked coils (measured: Monday
    CONSOLIDATION 15.3% -> 21.3% of the universe). Weekly structure is therefore as-of the
    last COMPLETE week -- which is also the correct read: a weekly breakout is not one
    until the week closes. Cross-TF alignment (e.g. BREAKOUT on 4h while CONSOLIDATION on 1h) is
    classic MTF tape-reading — the related validated evidence in this stack is Daily×Weekly
    breakout-from-tight-base (sister project); INTRADAY MTF alignment is unvalidated context,
    same as everything else in this lane."""
    key = (sym, dt.date.today(), _bucket5())
    hit = _MTF_CACHE.get(key)
    if hit is not None:
        return hit
    # out carries BOTH the label (s-key, used for filtering) and the live band ±% (b-key, used
    # for display). Additive b-keys keep every existing caller working (they read only labels).
    _TFS = ("15m", "1h", "2h", "4h", "1D", "1W")
    out = {t: "n/a" for t in _TFS}
    out.update({f"b{t}": float("nan") for t in _TFS})
    # h/l = that frame's RANGE BOX, n = closed bars behind the label. Carried so the HTFxLTF
    # synthesis can compute WHERE price sits in the higher-TF box (the variable that separates
    # a real range resolution from a false break) WITHOUT re-fetching a single candle.
    out.update({f"h{t}": float("nan") for t in _TFS})
    out.update({f"l{t}": float("nan") for t in _TFS})
    out.update({f"n{t}": 0 for t in _TFS})
    out.update({f"{p}{t}": float("nan") for t in _TFS for p in ("sup", "res")})
    out.update({f"{p}{t}": 0 for t in _TFS for p in ("supt", "rest")})
    out.update({f"{p}{t}": float("inf") for t in _TFS for p in ("hup", "hdn")})
    out.update({f"wall{t}": [] for t in _TFS})
    out.update({f"wallx{t}": [] for t in _TFS})
    out.update({f"blind{t}": (float("nan"), float("nan")) for t in _TFS})
    out.update({f"atr{t}": float("nan") for t in _TFS})

    # Which frames have a bar that has NOT closed yet? Only the intraday ones, and only while
    # the session is running. 1D/1W come from the EOD archive, which is complete by
    # construction. The flag reaches only the COIL test (see indicators.struct_full).
    _live_bar = market_open()

    def _set(tf, frame, forming=False, sr_lb=None):
        sf = (indicators.struct_full(frame, forming=forming and _live_bar)
              if (frame is not None and len(frame) >= 5) else {"struct": "n/a", "n": 0})
        lab = sf["struct"]
        out[tf] = lab
        out[f"b{tf}"] = indicators.band_pct(frame, lab) if lab != "n/a" else float("nan")
        out[f"h{tf}"] = float(sf.get("hi", float("nan")) or float("nan"))
        out[f"l{tf}"] = float(sf.get("lo", float("nan")) or float("nan"))
        out[f"n{tf}"] = int(sf.get("n", 0))
        # TOUCH-COUNTED S/R on this frame — computed here because this is the one place the
        # candles exist. Carrying the result means switching horizon later costs no fetch.
        # sr_lb overrides the S/R lookback for THIS frame: the DAILY level window is longer than
        # the intraday one (a daily support persists 6-18 months; the shared 60-bar window is
        # only ~3 months, so it undercounted major bases -- NATIONALUM's ×5 year-base read ×2,
        # and universe-wide the nearest-wall touch count ran ~half). The structure LABEL and box
        # are untouched (they read the last STRUCT_LOOKBACK bars regardless of frame length).
        sr = (indicators.sr_levels(frame, lookback=sr_lb, forming=forming and _live_bar)
              if lab != "n/a" else {})
        out[f"sup{tf}"] = sr.get("support", float("nan")) or float("nan")
        out[f"supt{tf}"] = int(sr.get("sup_touches", 0) or 0)
        out[f"res{tf}"] = sr.get("resistance", float("nan")) or float("nan")
        out[f"rest{tf}"] = int(sr.get("res_touches", 0) or 0)
        # head_* is None when there is NO multi-touch wall that way — that means CLEAR ROAD,
        # which is the opposite of "unknown". inf preserves the distinction through pandas.
        _hu, _hd = sr.get("head_up"), sr.get("head_dn")
        out[f"hup{tf}"] = float("inf") if _hu is None else float(_hu)
        out[f"hdn{tf}"] = float("inf") if _hd is None else float(_hd)
        out[f"wall{tf}"] = sr.get("levels", [])
        # (level, touches, lo, hi) + the pivot blind zone's extreme — both feed ONLY the
        # one-frame-up big-wall check, which is the read that has to name the level a chartist
        # can see. The pair's own sup/res/headroom are unchanged.
        out[f"wallx{tf}"] = sr.get("levels_ext", [])
        out[f"blind{tf}"] = sr.get("blind", (float("nan"), float("nan")))
        out[f"atr{tf}"] = sr.get("atr", float("nan"))

    try:
        # 60 CALENDAR DAYS, not 20. The coarse frames are resampled from this ONE fetch, so the
        # lookback is what feeds them: at 20d the 4h frame held just 30 bars — BELOW the 40-bar
        # S/R window — and 4h is the HTF of the BTST preset and the LTF of Swing, so two of the
        # four horizons were finding levels on a starved frame. 60d gives 4h ~82 bars.
        # Verified strictly additive: structure labels were IDENTICAL on 24/24 name-frames
        # (the 20-bar window reads the same recent bars either way), while 4h wall counts rose
        # (TRENT 5->9, KOTAKBANK 6->8). Same single API call, larger payload.
        f = fetch_intraday(sym, tf="15m", lookback_days=_MTF_FETCH_DAYS)
        if not f.empty:
            # ADJUST ONCE, HERE, SO EVERY READER SEES THE SAME SERIES. sr_levels back-adjusts
            # internally but struct_full and band_pct do not, so a split inside the 60-day
            # window would have put the LEVELS on the adjusted series and the STRUCTURE LABEL
            # on the raw one -- a name whose chart is half at pre-split prices. The daily
            # archive is already adjusted upstream (_daily_hist), which is why 1D/1W were
            # never exposed; the intraday frames come straight from the broker and were.
            # Currently latent: 0 of 70 sampled names had a >25% step inside 60 days. Latent
            # is not fixed -- a 1:5 split lands whenever it lands. Adjusting at the source
            # costs one pass per name and makes sr_levels' own pass a verified no-op.
            f = indicators.adjust_corporate_actions(f)
            _set("15m", f, forming=True)
            for lab, freq in (("1h", "60min"), ("2h", "120min"), ("4h", "240min")):
                _set(lab, _resample_ohlcv(f, freq), forming=True)   # already stub-merged
    except Exception:
        pass
    try:
        d = _daily_hist().get(sym)
        if d is not None and len(d) >= 5:
            _set("1D", d.tail(config.SR_DAILY_LOOKBACK), sr_lb=config.SR_DAILY_LOOKBACK)
            _set("1W", weekly_frame(d))
            # MONTHLY S/R walls only — the ONE-higher-frame the POSITIONAL horizon (1D/1W)
            # checks for its big-wall (1W is the top of the intraday ladder, so the frame above
            # it is the month). Not a full frame (no struct/box/band): the big-wall check needs
            # only the touch-counted levels. Built from a SEPARATE ~6-year daily read
            # (_monthly_hist), NOT the shallow 14-month _daily_hist: measured, 14 months of
            # monthly bars gave a ≥2-touch level on only 36% of names (a monthly level barely
            # gets touched twice in a year), while ~60 months gives one on 99%. A monthly level
            # is a multi-year object; anything less is starved.
            _mo = _monthly_hist().get(sym)
            _ok = _mo is not None and len(_mo) >= 12
            _sm = indicators.sr_levels(_mo) if _ok else {}
            out["wall1M"] = _sm.get("levels", [])
            out["wallx1M"] = _sm.get("levels_ext", [])
            # monthly_frame() has already dropped the incomplete current month, so every bar
            # here is closed -> forming=False (the default) is correct.
            out["blind1M"] = _sm.get("blind", (float("nan"), float("nan")))
            out["atr1M"] = indicators.atr(_mo, min(14, len(_mo))) if (_mo is not None and len(_mo) >= 5) else float("nan")
    except Exception:
        pass
    _MTF_CACHE[key] = out
    return out


def monthly_frame(d: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Daily bars -> monthly (calendar month), dropping the INCOMPLETE current month — the same
    rule as weekly_frame, one rung up. A part-formed month reads narrower and would fake a coil;
    a monthly level is not one until the month closes. Used only for the Positional big-wall."""
    m = (d.set_index("ts").groupby(pd.Grouper(freq="ME"))
         .agg(open=("open", "first"), high=("high", "max"),
              low=("low", "min"), close=("close", "last")).dropna().reset_index())
    if len(m):
        mo_end = m["ts"].iloc[-1]
        today = (now or pd.Timestamp.now()).normalize()
        if d["ts"].max() < mo_end and today <= mo_end:
            m = m.iloc[:-1]
    return m


def weekly_frame(d: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Daily bars -> weekly (W-FRI), truncated to the last COMPLETE week.

    DROP AN INCOMPLETE FINAL WEEK. A part-formed weekly bar spans fewer sessions, so its range
    is mechanically narrower -- and the coil test compares the latest 3-bar span against the
    typical one. Measured: on a Monday (a 1-day "week") weekly CONSOLIDATION jumped 15.3% ->
    21.3% of the universe, purely from the missing days. Same defect class as the original coil
    bug and as the intraday session stub: a window comparison where one side holds fewer bars.
    It is also the correct chartist call -- a weekly breakout is not one until the week closes.
    The live price still reaches the read through the daily frame.

    BUT "the last daily bar is before the Friday label" is NOT the question "is this week
    unfinished". W-FRI labels every group with its Friday date whether or not that Friday
    traded, so on any FRIDAY HOLIDAY the old test fired on a week that was already over: last
    daily bar Thursday < Friday label -> a COMPLETE Mon-Thu week silently discarded, and it
    stayed discarded for the whole following week. Verified on a synthetic June-2026 calendar
    with Friday the 26th removed. A week is finished once the CALENDAR has passed its Friday,
    regardless of whether the exchange opened that day; only then does the daily-bar test
    apply. `now` is injectable so the boundary is testable without waiting for a holiday."""
    w = (d.set_index("ts").groupby(pd.Grouper(freq="W-FRI"))
         .agg(open=("open", "first"), high=("high", "max"),
              low=("low", "min"), close=("close", "last")).dropna().reset_index())
    if len(w):
        wk_end = w["ts"].iloc[-1]
        today = (now or pd.Timestamp.now()).normalize()
        if d["ts"].max() < wk_end and today <= wk_end:
            w = w.iloc[:-1]
    return w


_UNISCAN_CACHE: dict = {}          # bucket5 -> {"board":..., "risk_on":..., "idx_ret":..., "scanned_at":...}
_SCAN_WORKERS = 6                  # concurrent /history fetches — the scan is I/O-bound. 6 keeps
                                   # us near Fyers' ~10 req/s history budget (a burst 429 just
                                   # yields 'n/a' for that name, non-fatal; a re-scan fills it).


def universe_mtf_scan(date=None) -> dict:
    """STRUCTURE-FIRST scan: the WHOLE liquid universe with its 6-timeframe structure —
    NO day-move / close-strength pre-screen. This is the source list the HTF/LTF structure
    filter then narrows (e.g. keep only BREAKOUT_UP on 4h + CONSOLIDATION on 1h).

    Cost is the point to understand: to FILTER by structure you must first COMPUTE structure
    for every name — one 15-min /history fetch each (~270 calls, resampled locally to
    1h/2h/4h; 1D/1W from the EOD archive at zero API cost). No bulk history endpoint exists, so
    the per-name fetch is unavoidable — BUT it is embarrassingly parallel I/O, so the fetches
    run CONCURRENTLY (bounded pool), turning ~3 min of serial round-trips into ~30s. Then FREE
    within the same 5-minute bucket (memoised in _UNISCAN_CACHE). Manual snapshot, not a 5s
    auto-loop. Returns a LIGHT board (structure + ltp + day% + turnover + delivery only);
    levels/RSI/verdict are added later by enrich_mtf on the FILTERED survivors."""
    ts = token_status()
    risk_on = regime.is_risk_on(pd.Timestamp(date) if date is not None
                                else data.last_trading_date())
    if not ts["usable"]:
        return {"ok": False, "status": ts["describe"], "risk_on": risk_on, "board": pd.DataFrame()}

    key = _bucket5()
    hit = _UNISCAN_CACHE.get(key)
    if hit is not None and date is None:
        return {"ok": True, **hit}

    uni = liquid_universe(date).set_index("symbol")
    dm = deliv_momentum(date)                    # DCM-ported delivery conviction (one read/day)
    # NOTE the delivery TREND column is deliberately NOT built here. Its buckets and baseline
    # both follow the trade horizon, which is chosen after this cached scan runs, so the column
    # is attached at RENDER time (dashboard._dw). Pre-seeding a default here was worse than
    # useless: on any failure of that render-time call the stale default survived while the
    # header still advertised the horizon's settings — a silently mismatched column. Absent
    # beats wrong, and _cols() drops a missing column cleanly.
    q = _fetch_quotes([fy_symbol(s) for s in uni.index])
    _nf = _fetch_quotes([config.NIFTY_FYERS]).get(config.NIFTY_FYERS, {})
    idx_ret = _chp(_nf)
    if date is None and _nf.get("lp"):
        risk_on = regime.is_risk_on_live(_nf.get("lp"), max_stale_days=config.REGIME_MAX_STALE_DAYS)

    # ── PHASE 1: cheap pass over the ONE batch quote — no per-name network here ──
    cand = []
    for fys, v in q.items():
        sym = fys.replace("NSE:", "").replace("-EQ", "")
        if sym not in uni.index:
            continue
        c, pc = v.get("lp"), v.get("prev_close_price") or uni.loc[sym, "ref_close"]
        h, l = v.get("high_price"), v.get("low_price")
        if None in (c, pc, h, l) or h == l:
            continue
        h, l = max(float(h), float(c)), min(float(l), float(c))   # broker range can lag the LTP
        # NO liquidity floor (user override): the WHOLE F&O universe appears; turn₹L is carried
        # so you can SEE how thin a name is and judge fill risk yourself.
        _vol = v.get("volume") or 0
        turn_l = round((_vol * (h + l + float(c)) / 3.0) / 1e5, 1)   # today's turnover, ₹lacs
        cand.append((sym, float(c), float(pc), 100 * (float(c) / float(pc) - 1), turn_l))

    # ── PHASE 2: the 15-min structure fetch is the ONLY per-name network cost — and it is
    # embarrassingly parallel I/O. Run the fetches CONCURRENTLY (bounded pool) instead of one
    # blocking round-trip at a time: ~30s vs ~3min for the full universe. Pre-warm the shared
    # daily-archive read ONCE before the fan-out, else N worker threads each trigger a DuckDB
    # load (mtf_structure's _MTF_CACHE / _daily_hist are plain dicts — GIL makes get/set atomic,
    # but the first heavy LOAD must be serialised).
    from concurrent.futures import ThreadPoolExecutor
    _daily_hist()                                   # warm the 1D/1W source once
    _monthly_hist()                                 # warm the 1M (Positional big-wall) source once
    syms = [x[0] for x in cand]
    with ThreadPoolExecutor(max_workers=_SCAN_WORKERS) as ex:
        mtfs = dict(zip(syms, ex.map(mtf_structure, syms)))

    # THROTTLE SWEEP — a burst of ~250 /history calls trips Fyers' rate limit for a slice of
    # the universe, and mtf_structure swallows the error into 'n/a'. Measured: 48 of 243 names
    # (TCS, RELIANCE — not thin names) came back blank on EVERY intraday frame, and a blank
    # row silently VANISHES from any HTF/LTF filter. That is the worst failure mode there is:
    # a filter that quietly answers from 80% of the universe while looking complete. Retry the
    # blanks once, slower and narrower, and report what still failed.
    _blank = [s for s in syms if (mtfs.get(s) or {}).get("15m", "n/a") == "n/a"]
    if _blank:
        # 15s, not 1s. The old 1s was a guess and it was wrong: the rate window is ROLLING, so
        # a retry issued one second after the burst that caused the 429s runs inside the very
        # window those calls poisoned and 429s again. That is why the sweep never cleared the
        # blanks. With fetch_intraday now paced there should be nothing left to sweep; this
        # stays as the belt-and-braces path for a genuine transient (network blip, one slow
        # symbol) and it now waits long enough for the window to actually roll.
        time.sleep(15.0)
        for s in _blank:
            _MTF_CACHE.pop((s, dt.date.today(), _bucket5()), None)   # drop the cached blank
        with ThreadPoolExecutor(max_workers=2) as ex:    # gentler than the main fan-out
            for s, v in zip(_blank, ex.map(mtf_structure, _blank)):
                mtfs[s] = v
    _still = sum(1 for s in syms if (mtfs.get(s) or {}).get("15m", "n/a") == "n/a")

    _TFS = ("15m", "1h", "2h", "4h", "1D", "1W")
    _NA = {t: "n/a" for t in _TFS}
    _NA.update({f"{p}{t}": (0 if p == "n" else float("nan")) for t in _TFS for p in "bhln"})
    rows = []
    for sym, c, pc, day, turn_l in cand:
        _mtf = mtfs.get(sym) or _NA
        _wd = float(dm.loc[sym, "wtd_deliv7"]) if sym in dm.index else np.nan
        _dv = float(dm.loc[sym, "deliv_vs_100d"]) if sym in dm.index else np.nan
        rows.append({
            "symbol": sym, "sector": uni.loc[sym, "sector"], "ltp": c, "day%": round(day, 2),
            "turn₹L": turn_l,                                          # today's turnover (₹lacs)
            "wtd_deliv7": round(_wd, 1) if _wd == _wd else np.nan,     # NaN-safe (x==x)
            "deliv_vs_100d": round(_dv, 1) if _dv == _dv else np.nan,
            "s15m": _mtf["15m"], "s1h": _mtf["1h"], "s2h": _mtf["2h"],
            "s4h": _mtf["4h"], "s1D": _mtf["1D"], "s1W": _mtf["1W"],
            "bnds15m": _mtf["b15m"], "bnds1h": _mtf["b1h"], "bnds2h": _mtf["b2h"],
            "bnds4h": _mtf["b4h"], "bnds1D": _mtf["b1D"], "bnds1W": _mtf["b1W"],
            # the per-frame range BOX + bar count — feeds the HTFxLTF synthesis (mtf.py)
            **{f"box_{p}{t}": _mtf.get(f"{p}{t}", float("nan"))
               for t in _TFS for p in ("h", "l", "n")},
            # touch-counted S/R per frame — feeds the level read (add_setup picks the pair)
            **{f"sr_{p}{t}": _mtf.get(f"{p}{t}")
               for t in _TFS for p in ("sup", "supt", "res", "rest", "hup", "hdn", "wall", "atr",
                                       "wallx", "blind")},
            # MONTHLY walls (Positional big-wall only) — carried explicitly, not a full frame
            "sr_wall1M": _mtf.get("wall1M"), "sr_atr1M": _mtf.get("atr1M"),
            "sr_wallx1M": _mtf.get("wallx1M"), "sr_blind1M": _mtf.get("blind1M"),
            # carried for enrich_mtf (needs the authoritative live prev_close + EOD baselines):
            "_pc": pc, "_vol_med20": float(uni.loc[sym, "vol_med20"] or 0),
            "_rs_cum9": float(uni.loc[sym, "rs_cum9"] or 0),
        })
    board = pd.DataFrame(rows)
    if not board.empty:
        board = board.sort_values("day%", ascending=False)
    out = {"status": ts["describe"], "risk_on": risk_on, "idx_ret": idx_ret,
           "board": board, "n_scanned": len(rows), "scanned_at": dt.datetime.now(),
           # names whose intraday frames are STILL blank after the retry — they cannot match
           # any intraday structure filter, so the count must be visible, not swallowed.
           "n_blank_intraday": int(_still),
           # names that never even reached PHASE 1 because their /quotes chunk failed. These
           # are invisible everywhere else -- no row is built, so n_scanned already excludes
           # them and the board would look complete at a smaller size.
           "n_quote_gap": int(_QUOTE_GAP[0])}
    if date is None:
        _UNISCAN_CACHE.clear()                  # never hold two buckets of full-universe boards
        _UNISCAN_CACHE[key] = out
    return {"ok": True, **out}


_AT_WALL_ATR = 0.15        # within this fraction of ATR = price is ON the level right now


def _live_levels(b: pd.DataFrame) -> pd.DataFrame:
    """Recompute WHICH wall is nearest each side, how far, and whether price is testing one
    RIGHT NOW — against whatever `ltp` currently holds.

    The walls themselves are past structure and only change when a bar closes, so they are
    computed once per scan. But `nearest support`, `nearest resistance` and `headroom` are
    all functions of the CURRENT price, and freezing them was actively misleading: as price
    ticks toward a level headroom stayed wide, and once price traded THROUGH a level the
    board still listed it as resistance overhead. On a 4h frame the scan can be hours old.

    `at_wall` is the live-tape answer to 'is price being rejected here': price sitting within
    0.15 ATR of a level that has already turned it >=2 times before. The touch COUNT is
    deliberately not incremented — a touch is only a rejection once price actually turns, and
    counting the test in progress would let a level inflate its own strength while breaking."""
    if b is None or b.empty or "_wall_pair" not in b.columns:
        return b
    sup, sup_t, res, res_t, head, at_w = [], [], [], [], [], []
    for _, r in b.iterrows():
        wl = r.get("_wall_pair")
        px = r.get("ltp")
        a = r.get("_sr_atr")
        try:
            px = float(px)
        except (TypeError, ValueError):
            px = 0.0
        a = float(a) if (a is not None and a == a and float(a) > 0) else 0.0
        if not isinstance(wl, list) or not wl or px <= 0:
            sup.append(np.nan); sup_t.append(0); res.append(np.nan); res_t.append(0)
            head.append(np.inf); at_w.append("")
            continue
        # RE-CLUSTER THE MERGED LIST. The two frames are resampled from the SAME series, so
        # the same physical swing shows up in both — the raw merge is full of near-duplicates.
        # Picking "nearest by price" then reported the DUPLICATE: a 3-touch wall at 100.00
        # sitting 0.05 from a 1-touch at 100.05 was displayed as x1, understating the level
        # that actually matters. Touches are MAXed, never summed: the two frames are seeing
        # one swing twice, so adding them would manufacture strength that never happened.
        if a > 0:
            # CLUSTER EACH SIDE OF PRICE SEPARATELY. A cluster must never span the current
            # price: a wall below price is a FLOOR and a wall above is a CEILING -- functionally
            # opposite -- so single-linkage chaining across price is nonsense. Measured live
            # (MPHASIS, BTST): a run 2315(x4) 2336(x4) 2361(x1) 2388(x4) 2412(x2), each pair
            # within tolerance, chained into ONE cluster anchored at 2315.28 -- BELOW price --
            # so a x4 RESISTANCE at 2388 was absorbed into a SUPPORT at 2315, and `res` then
            # showed the weaker 2412(x2) with a stronger x4 hiding 24 points below it. Splitting
            # by side first makes that impossible. A wall exactly AT price goes to the ceiling
            # (x >= px), matching at_wall's own classifier ("RES if x >= px").
            #
            # Within a side: SINGLE-LINKAGE anchored on the strongest member (the DALBHARAT
            # fix). Tolerance on the RUNNING EDGE (not the anchor) so absorbing a member never
            # shrinks the cluster's reach; a width cap stops the chain drifting across the
            # chart. Touches MAXed, never summed -- the two frames resample ONE series.
            _tol = config.SR_TOL_ATR
            _cap = 3.0 * config.SR_TOL_ATR

            def _cluster(pts):
                merged: list[list] = []                  # [anchor_px, touches, edge_px, min]
                for x, t in sorted(pts, key=lambda z: z[0]):
                    if merged and abs(x - merged[-1][2]) <= _tol * a and \
                            (x - merged[-1][3]) <= _cap * a:
                        m = merged[-1]
                        if t > m[1]:
                            m[0], m[1] = x, t             # strongest takes the anchor
                        m[2] = x                          # running edge
                    else:
                        merged.append([x, t, x, x])
                return [(m[0], m[1]) for m in merged]

            below_c = _cluster([(x, t) for x, t, _ in wl if x < px])
            above_c = _cluster([(x, t) for x, t, _ in wl if x >= px])   # x==px -> ceiling
            wl = [(x, t, "") for x, t in below_c + above_c]
        # THE MIN-DISTANCE FILTER MUST NOT SWALLOW A DEFENDED LEVEL. It stops a 1-touch
        # micro-swing hugging price from being called "the level" -- but a >=2-touch wall is
        # never noise, however close it sits, so the filter applies only to 1-touch pivots.
        # (at_wall has NO min-dist filter -- its job is "price is ON a level now" -- so without
        # this exception the level at_wall names would vanish from sup/res, and the two columns
        # would contradict on every firing, as they did before this rule: 28/28, 29/29 ...).
        # `above` uses x >= px so a wall exactly at price is the resistance at_wall calls it.
        md = 0.25 * a
        below = [(x, t) for x, t, _ in wl if x < px and (t >= 2 or x < px - md)]
        above = [(x, t) for x, t, _ in wl if x >= px and (t >= 2 or x > px + md)]
        s_ = max(below, key=lambda z: z[0]) if below else (np.nan, 0)
        r_ = min(above, key=lambda z: z[0]) if above else (np.nan, 0)
        sup.append(round(s_[0], 2) if s_[0] == s_[0] else np.nan)
        sup_t.append(int(s_[1]))
        res.append(round(r_[0], 2) if r_[0] == r_[0] else np.nan)
        res_t.append(int(r_[1]))
        # HEADROOM = DISTANCE TO THE LEVEL THE res/sup COLUMN ALREADY SHOWS, in the trade's
        # direction. Two fixes live here. (1) Direction: a long's room-to-run is UP to the
        # nearest ceiling, a SHORT's is DOWN to the nearest floor (it was hardcoded up, so the
        # SHORT tab showed the wrong side). (2) Touch gate DROPPED: headroom used to count only
        # >=2-touch walls, so a single VIOLENT rejection -- PPLPHARMA's 07-May spike-and-crash to
        # 196, one dramatic touch -- read "inf clear" while the res column plainly showed 196.44
        # overhead. Those two columns contradicting each other is the bug. MEASURED before
        # trusting the >=2 gate had a reason (7,061 daily approaches, placebo-controlled): a real
        # swing high is respected 69.2% on the next approach vs 70.5% for a RANDOM displaced line
        # (real-random -1.3pp, t=-1.45), and >=2-touch (69.2%) == 1-touch-violent (69.1%) ==
        # 1-touch-all (68.6%). The touch count carries NO forward edge, and neither does the level
        # itself -- so gating headroom on it only made the board lie about clear road. Tie headroom
        # to the SAME nearest level res/sup display (r_/s_ above): consistent by construction, and
        # honest -- it maps where the visible level is, it does not claim the level predicts.
        if str(r.get("side")) == "SHORT":
            head.append(round((px - s_[0]) / a, 2) if (s_[0] == s_[0] and a > 0) else np.inf)
        else:
            head.append(round((r_[0] - px) / a, 2) if (r_[0] == r_[0] and a > 0) else np.inf)
        # LIVE TEST IN PROGRESS — price is sitting on a previously-defended level right now
        tol = _AT_WALL_ATR * a
        on = [(x, t) for x, t, _ in wl if t >= 2 and abs(x - px) <= tol] if a > 0 else []
        if on:
            x, t = min(on, key=lambda z: abs(z[0] - px))
            at_w.append(f"{'RES' if x >= px else 'SUP'} {x:.2f} x{t}")
        else:
            at_w.append("")
    b["sup"], b["sup_t"] = sup, sup_t
    b["res"], b["res_t"] = res, res_t
    b["headroom"], b["at_wall"] = head, at_w
    return b


_PHANTOM_COLS = ("open", "high", "low", "close", "volume")


def drop_phantom_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Remove bars that REPEAT the previous bar exactly — a broker artifact, not a session.

    Over a weekend the Fyers daily feed can carry an extra bar stamped on the NEXT calendar
    slot whose OHLCV is byte-identical to the last real session. Caught live on Sunday
    2026-08-02: the 1D feed held a 01-Aug (SATURDAY) bar identical to 31-Jul, and
    archive_staleness counted it as a session the archive had not ingested — raising a loud
    "the EOD archive is 1 trading day behind the market" on an archive that was perfectly
    current. A false staleness alarm is expensive: it tells you to distrust every 1D/1W
    verdict on the page and to go re-run a sync that has nothing to do.

    Deliberately keyed on DUPLICATE DATA rather than on the weekday. NSE does hold occasional
    Saturday sessions (disaster-recovery drills), and those carry their own prices — a
    weekday-only rule would silently discard a real session, which is the worse error. A bar
    that repeats its predecessor tick-for-tick carries no information either way.

    FALSE-POSITIVE RISK MEASURED, not assumed: across 167,084 archive rows (2024-2026) ZERO
    consecutive daily bars matched on all of open/high/low/close/volume. Volume is what makes
    that safe — two real sessions can share an OHLC by chance, they do not share a share count.
    """
    if df is None or df.empty:
        return df
    cols = [c for c in _PHANTOM_COLS if c in df.columns]
    if not cols:
        return df
    d = df.sort_values("ts").reset_index(drop=True)
    dup = (d[cols] == d[cols].shift(1)).all(axis=1)
    dup.iloc[0] = False                      # the first bar has nothing to repeat
    return d[~dup].reset_index(drop=True)


_STALE_CACHE: dict = {}


def archive_staleness() -> dict:
    """Is the EOD archive behind the market? data.last_trading_date() only returns the
    archive's OWN max date, so it cannot detect its own staleness — it is a mirror, not a
    calendar. Fyers IS an independent calendar: its daily bars are the real trading days.

    Compare the archive's latest date to Fyers' latest COMPLETED daily bar (a bar dated
    strictly before today; today's bar is still forming intraday and is excluded). If Fyers
    has a finished session the archive does not, the archive is stale — the DCM nightly sync
    has not ingested it, and every 1D/1W structure, level and verdict is that many sessions
    behind. Measured live 2026-07-26: archive latest 07-23, Fyers had 07-24 (MOTILALOFS closed
    it -7.3%), so the board showed a bullish tag on pre-crash data with the price already
    through its 'support'. This detector makes that visible instead of silent.

    Returns {stale_days, archive_date, market_date, ok}. ok=False if Fyers can't be reached
    (then we simply do not warn — never block the board on the probe)."""
    from . import data as _data
    key = dt.date.today()
    if key in _STALE_CACHE:
        return _STALE_CACHE[key]
    out = {"stale_days": 0, "archive_date": None, "market_date": None, "ok": False}
    try:
        arch = pd.Timestamp(_data.last_trading_date()).normalize()
        out["archive_date"] = arch
        # RELIANCE trades every NSE session, so its daily bars ARE the trading calendar.
        # KNOWN SINGLE POINT OF FAILURE, accepted: if RELIANCE alone were halted for a session
        # the feed would lack that bar and this would UNDER-report staleness — a silent miss
        # rather than a false alarm, which is the safer direction for a warning, and the reason
        # a quorum of symbols (three API calls instead of one) is not worth the cost here.
        f = drop_phantom_bars(fetch_intraday("RELIANCE", tf="1D", lookback_days=12))
        if f.empty:
            _STALE_CACHE[key] = out
            return out
        today = pd.Timestamp(dt.date.today())
        completed = f[f["ts"].dt.normalize() < today]      # exclude today's forming bar
        if completed.empty:
            _STALE_CACHE[key] = out
            return out
        mkt = completed["ts"].dt.normalize().max()
        out["market_date"] = mkt
        out["ok"] = True
        # count how many of Fyers' completed sessions the archive is missing
        out["stale_days"] = int((f["ts"].dt.normalize() > arch).sum()
                                - (f["ts"].dt.normalize() >= today).sum())
        out["stale_days"] = max(0, out["stale_days"])
    except Exception:
        pass
    _STALE_CACHE[key] = out
    return out


def refresh_light_prices(board: pd.DataFrame) -> pd.DataFrame:
    """Re-price a STRUCTURE-SCAN board from one batch quote. Structure stays pinned.

    The structure lane's own docstrings describe a two-tier refresh -- the expensive half
    (bars, structure) moves only when a bar closes, the cheap half (price and everything
    derived from it) moves every tick. That was wired for the ENRICHED table and for the live
    snapshot, but NOT for the LONG/SHORT tabs the board opens on, which had no fragment at
    all. Those tabs were therefore a still photograph while the ltp column's own tooltip
    promised "LIVE, refreshed every 5s on every tab".

    Cost is one batch quote for the whole board -- 243 names is 5 requests on /quotes, not
    243 on /history -- so this is cheap enough to run every 5 seconds. Everything the scan
    carried (the 20-bar boxes, the wall lists, the ATRs) is untouched and stays pinned; the
    caller re-runs add_setup afterwards so `loc`, the setup tag, the side and the live levels
    all re-derive from the new price, which is exactly what they are functions of."""
    if board is None or board.empty or "symbol" not in board.columns:
        return board
    q = _fetch_quotes([fy_symbol(s) for s in board["symbol"]])
    if not q:
        return board                                  # a missed quote must not blank the board
    b = board.copy()
    ltp, day = [], []
    for _, r in b.iterrows():
        v = q.get(fy_symbol(r["symbol"])) or {}
        px = v.get("lp")
        if not px:
            ltp.append(r.get("ltp")); day.append(r.get("day%")); continue
        pc = v.get("prev_close_price") or r.get("_pc")
        ltp.append(round(float(px), 2))
        day.append(round(100 * (float(px) / float(pc) - 1), 2) if pc else r.get("day%"))
    b["ltp"], b["day%"] = ltp, day
    return b


def add_setup(board: pd.DataFrame, ltf: str, htf: str) -> pd.DataFrame:
    """Attach the HTFxLTF chartist synthesis to every row: setup tag, quality rank, the
    plain-English read, and `loc` (where the LTP sits inside the higher-TF range box).

    ZERO network cost — the scan already carried each frame's box (box_h/box_l/box_n), so
    this is pure arithmetic over the existing board and re-runs instantly when you switch
    preset. See mtf.py for what the tags mean and why none of them is a validated signal."""
    from . import mtf as _mtf_mod
    if board is None or board.empty:
        return board
    b = board.copy()
    tags, reads, ranks, locs = [], [], [], []
    dirs, sides = [], []
    for _, r in b.iterrows():
        def _side(tf):
            return {"struct": r.get(f"s{tf}", "n/a"),
                    "hi": r.get(f"box_h{tf}"), "lo": r.get(f"box_l{tf}"),
                    "n": int(r.get(f"box_n{tf}") or 0)}
        s = _mtf_mod.synthesize(_side(htf), _side(ltf), r.get("ltp"))
        tags.append(s["tag"])
        reads.append(s["read"])
        dirs.append(s.get("dir", "NONE"))
        sides.append(_mtf_mod.side_of(s["tag"], s.get("dir", "NONE")))
        ranks.append(_mtf_mod.TAG_RANK.get(s["tag"], 9))
        locs.append(round(s["loc"], 2) if s.get("loc") is not None else np.nan)
    b["setup"] = tags
    b["setup_rank"] = ranks
    b["loc"] = locs                      # 0 = at HTF box low, 1 = at HTF box high
    b["setup_read"] = reads
    b["dir"] = dirs
    b["side"] = sides

    # ── TOUCH-COUNTED S/R for the chosen pair ────────────────────────────────────────
    # The HIGHER frame supplies the levels that matter (a 4h wall outranks a 1h one), but
    # the LOWER frame is checked too: a coarse bar SWALLOWS several fine swings, so a level
    # price rejected three times intraday can be invisible to the higher frame entirely.
    def _g(col, default=np.nan):
        return b[col] if col in b.columns else pd.Series(default, index=b.index)

    # sup / res / headroom / at_wall are NOT set from the scan snapshot — they are derived
    # from the merged wall list against LIVE price, below and again on every 5s tick.
    # Carry the MERGED wall list (both frames, tagged) so the 5-second tick can recompute
    # which level is nearest and how far away it is WITHOUT refetching a candle. The walls
    # themselves only move when a bar closes; what is nearest is a function of live price.
    pair = []
    for _, r in b.iterrows():
        wl, wh = r.get(f"sr_wall{ltf}"), r.get(f"sr_wall{htf}")
        m = ([(float(x), int(t), htf) for x, t in wh] if isinstance(wh, list) else []) + \
            ([(float(x), int(t), ltf) for x, t in wl] if isinstance(wl, list) else [])
        pair.append(m)
    b["_wall_pair"] = pair
    # THE ATR UNIT MUST BE THE TRIGGER FRAME'S, NOT THE CONFIRMATION FRAME'S.
    # `headroom` exists to answer one question: does my 1xATR target sit on the far side of a
    # defended level? That target, and the stop beside it, are built from the LOWER timeframe's
    # ATR (enrich_mtf calls deep_state with tf=ltf). Normalising headroom by the HIGHER frame's
    # ATR therefore quoted the distance in a unit ~2x larger than the target it is compared
    # against, so every distance read ~2x too small. Measured across the live board, the
    # "< 0.5 = you are buying INTO a wall" warning fired on 100/195 names on Intraday, 93/188
    # BTST, 64/173 Swing, 100/183 Positional -- roughly half the universe. In the trigger
    # frame's own ATR it fires on 41/50/45/43, i.e. about a quarter. The old numbers were not
    # a stricter setting, they were the wrong unit: HTF ATR / LTF ATR runs 1.6x-2.4x by preset.
    # The same `a` also sets the wall-merge tolerance and the at_wall test, and the trigger
    # frame is the right resolution for both -- "price is ON this level" should mean on it at
    # the resolution you are timing the entry, not within a weekly bar's noise.
    # The WALL LIST still merges BOTH frames; only the yardstick changes.
    b["_sr_atr"] = b[f"sr_atr{ltf}"] if f"sr_atr{ltf}" in b.columns else np.nan
    b = _live_levels(b)                       # nearest/headroom against the price we have now

    # ── THE ONE HIGHER-FRAME WALL THE PAIR IS BLIND TO ───────────────────────────────────
    # sup/res/headroom above see ONLY the two frames of the preset. But a long can be sitting
    # right under a resistance ONE FRAME ABOVE the pair -- one it never looked at -- walk
    # straight into it, and reverse: the classic "traded the small frames, the big level capped
    # it" loss. Confirmed live: SIEMENS read pair-headroom "∞ clear" with a 4-touch 1D wall 0.29
    # ATR overhead; ATUL sat 0.02 ATR under a 5-touch daily wall.
    #
    # Surface the nearest DEFENDED (>=2-touch) wall from exactly ONE frame -- the next standard
    # CONFIRMATION frame above the pair's HTF, walking the 4x ladder (1h -> 4h -> 1D -> 1W -> 1M)
    # the whole system is built on. ALL FOUR horizons get their one frame: Intraday (HTF 1h) ->
    # 4h; BTST (4h) -> 1D; Swing (1D) -> 1W; Positional (1W) -> 1M (the MONTH -- the frame above
    # the weekly, so Positional is not left blind). Just ONE frame, as a chartist checks the
    # next chart up -- not the whole stack (that buried the signal and double-showed the pair's
    # own levels via the 2h/4h resamples of the same series). The 2h "tweener" is skipped -- no
    # preset uses it as a confirmation frame. The relevant frame scales with the HOLD: an
    # overnight BTST can blow through a daily level but not a weekly; a weeks-long Positional can
    # reach the monthly. Direction follows the trade: a ceiling above a long, a floor below a
    # short. Distance in the TRIGGER frame's ATR. CONTEXT, not a veto -- a break of the big
    # level is often the move -- but you must SEE it before you buy in.
    _TF_ORD = ("15m", "1h", "2h", "4h", "1D", "1W", "1M")
    _LADDER = ("1h", "4h", "1D", "1W", "1M")      # the 4x confirmation-frame ladder, incl. month
    # Same 0.25-ATR min-distance the pair path uses: a 1-touch pivot hugging price is inside the
    # noise, a >=2-touch wall is never noise however close it sits.
    _BIG_MIN_DIST_ATR = 0.25
    _hi = _TF_ORD.index(htf) if htf in _TF_ORD else -1
    _one = next((f for f in _LADDER if _TF_ORD.index(f) > _hi), None)
    _ctx = [_one] if _one else []
    big_w, big_g, big_px = [], [], []
    for _, r in b.iterrows():
        px, a = r.get("ltp"), r.get("_sr_atr")
        try:
            px = float(px)
        except (TypeError, ValueError):
            px = 0.0
        a = float(a) if (a is not None and a == a and float(a) > 0) else 0.0
        if px <= 0 or a <= 0 or not _ctx:
            big_w.append(""); big_g.append(np.inf); big_px.append(np.nan); continue
        look_up = r.get("side") != "SHORT"          # long / no-side watch the ceiling; short the floor
        md = _BIG_MIN_DIST_ATR * a
        walls = []
        for f in _ctx:
            wl = r.get(f"sr_wallx{f}") or r.get(f"sr_wall{f}")
            for _lv in (wl if isinstance(wl, list) else []):
                # tolerate both shapes: (level, touches) and (level, touches, lo, hi)
                x, t = float(_lv[0]), int(_lv[1])
                lo, hi = (float(_lv[2]), float(_lv[3])) if len(_lv) >= 4 else (x, x)
                # (3) CLUSTER EDGE. A cluster is quoted at its touch-weighted MEAN. When price
                # sits INSIDE the cluster that mean falls on the wrong side, so the ceiling over
                # a long renders as a floor under it. Quote the edge facing the trade instead.
                if look_up and x <= px < hi:
                    x = hi
                elif (not look_up) and x >= px > lo:
                    x = lo
                # (1) KEEP 1-TOUCH LEVELS. Identical rule to the pair path in _live_levels: a
                # single violent rejection IS a level, and only the min-distance filter should
                # drop it. This gate used to be a bare `t >= 2`, which is the same bug already
                # found and fixed for headroom -- it made a lone spike-and-crash read "clear"
                # while the level sat plainly on the chart.
                if t >= 2 or (x > px + md if look_up else x < px - md):
                    walls.append((x, t, f, ""))
            # (2) THE PIVOT BLIND ZONE. pivots() needs +/-2 neighbours, so the last two bars of
            # the frame can never BE a level -- two WEEKS on 1W, two MONTHS on 1M. That is where
            # "price is testing the high right now" lives, and it was the single biggest source
            # of a false "clear" (50.2% of names on 1W, 57.0% on 1M). Closed bars only, so it
            # cannot repaint. Marked "~" because it is an untested extreme, not a defended level.
            bz = r.get(f"sr_blind{f}")
            if isinstance(bz, (list, tuple)) and len(bz) == 2:
                e = bz[1] if look_up else bz[0]
                try:
                    e = float(e)
                except (TypeError, ValueError):
                    e = float("nan")
                if e == e and ((e > px) if look_up else (e < px)):
                    walls.append((e, 1, f, "~"))
        side_walls = [w for w in walls if (w[0] > px) == look_up]
        if side_walls:
            x, t, f, k = (min if look_up else max)(side_walls, key=lambda z: z[0])
            big_w.append(f"{f} {x:.2f} ×{t}{k}")
            big_g.append(round(abs(x - px) / a, 2))
            big_px.append(x)                        # the wall PRICE, so the 5s tick re-derives the gap
        else:
            big_w.append(""); big_g.append(np.inf); big_px.append(np.nan)
    b["big_wall"], b["big_gap"], b["_big_wall_px"] = big_w, big_g, big_px

    # NO CONFLUENCE FLAG — DELIBERATELY. The sister project's one POSITIVE level result was
    # pivot-meets-CALL-WALL confluence (+9.8pp, overhead only), and it is tempting to mirror
    # it here as "a 1h wall sitting on a 4h wall". That mirror is broken, for a reason worth
    # writing down: those two wall sets are derived from THE SAME PRICE SERIES (4h is a
    # resample of the same candles as 1h), so a 4h swing high usually IS a 1h swing high.
    # Agreement between them is close to tautological, not confirmatory.
    # MEASURED, rather than argued: a 1h/4h "confluence" flag fired on 196 of 197 names at a
    # 25bps tolerance, and a NULL version — one frame's walls randomly displaced 1-5% —
    # still fired on 148. Tightening to 10bps and requiring proximity to price gave 70.6%
    # real vs 31.0% null. A flag that fires on most names, and whose shuffled control also
    # fires constantly, carries no information; it would have looked like confirmation on
    # every chart you opened. The validated version worked because OPEN INTEREST is an
    # INDEPENDENT source from price. Cash equity names here have no chain, so that second
    # source does not exist, and the honest move is to ship no flag rather than a decorative
    # one. Touch counts and headroom stay: those are descriptions, not claims.
    return b


def enrich_mtf(board: pd.DataFrame, ltf: str = "1h", risk_on: bool = True,
               idx_ret: float = 0.0) -> pd.DataFrame:
    """Add levels / RSI / structure / verdict to the FILTERED survivors of universe_mtf_scan.
    The LOWER timeframe (ltf) drives the risk geometry and the read — HTF is the context gate,
    LTF is where you actually time and size the entry. Bounded to the handful the MTF filter
    let through, so the heavy per-name deep_state fetch only runs on names you care about."""
    if board.empty:
        return board
    _tfmin = {"1D": 1440, "4h": 240, "2h": 120, "1h": 60, "15m": 15, "5m": 5}.get(ltf, 60)

    def _one(r):
        sym, live_pc = r["symbol"], r["_pc"]
        ds = deep_state(sym, tf=ltf, ref_close=live_pc, ref_avg_vol=None, idx_ret=idx_ret)
        if not ds:
            return None
        s, lv = ds["state"], ds["levels"]
        atr_tf = ds.get("atr_tf", 0.0)
        ltp = s["ltp"]
        cndl = ds.get("candles")
        bar_time = None
        if cndl is not None and len(cndl):
            o = cndl["ts"].iloc[-1]
            close_t = min(o + pd.Timedelta(minutes=_tfmin), o.normalize() + pd.Timedelta("15h30min"))
            bar_time = f"{o.strftime('%H:%M')}-{close_t.strftime('%H:%M')}"
        _ex = _session_extras(sym, live_pc, r["_vol_med20"], rs_cum9=r["_rs_cum9"])
        entered = _ex.get("trigger")
        _tpx = _ex.get("trig_px")
        at_px = round(_tpx, 2) if _tpx else None
        since_pct = round(100 * (ltp / _tpx - 1), 2) if (_tpx and _tpx > 0) else None
        forming = None
        if cndl is not None and len(cndl):
            _o = cndl["ts"].iloc[-1]
            _close_t = min(_o + pd.Timedelta(minutes=_tfmin),
                           _o.normalize() + pd.Timedelta("15h30min"))
            forming = dt.datetime.now() < _close_t.to_pydatetime()
        s_stop = round(ltp + atr_tf, 2) if atr_tf > 0 else None
        s_t1 = round(ltp - atr_tf, 2) if atr_tf > 0 else None
        s_t2 = round(ltp - 2 * atr_tf, 2) if atr_tf > 0 else None
        return {
            "symbol": sym, "entered": entered, "at": at_px, "since%": since_pct,
            "time": bar_time,
            "bar": ("⏳ forming" if forming else "✓ closed") if forming is not None else None,
            "s15m": r["s15m"], "s1h": r["s1h"], "s2h": r["s2h"],
            "s4h": r["s4h"], "s1D": r["s1D"], "s1W": r["s1W"],
            "bnds15m": r.get("bnds15m"), "bnds1h": r.get("bnds1h"), "bnds2h": r.get("bnds2h"),
            "bnds4h": r.get("bnds4h"), "bnds1D": r.get("bnds1D"), "bnds1W": r.get("bnds1W"),
            "sector": r["sector"], "ltp": ltp, "turn₹L": r.get("turn₹L"),
            # carried through so the enriched table keeps the HTFxLTF read it was selected on
            "setup": r.get("setup"), "loc": r.get("loc"), "dir": r.get("dir"),
            # SIDE + the one-frame-up big wall MUST travel through enrich too, or the enriched
            # view loses the very columns the setup filter and the room filter select on -- and
            # its LONG/SHORT tabs then have nothing to split on but the footprint `action`.
            "side": r.get("side"), "big_wall": r.get("big_wall"), "big_gap": r.get("big_gap"),
            "_big_wall_px": r.get("_big_wall_px"),   # so refresh_prices ticks big_gap live
            "setup_rank": r.get("setup_rank"), "setup_read": r.get("setup_read"),
            "sup": r.get("sup"), "sup_t": r.get("sup_t"), "res": r.get("res"),
            "res_t": r.get("res_t"), "headroom": r.get("headroom"),
            "at_wall": r.get("at_wall"),
            # the wall list + its ATR travel with the row so refresh_prices can re-derive
            # nearest/headroom/at_wall on every tick without a fetch
            "_wall_pair": r.get("_wall_pair"), "_sr_atr": r.get("_sr_atr"),
            "wtd_deliv7": r.get("wtd_deliv7"), "deliv_vs_100d": r.get("deliv_vs_100d"),
            # `deliv trend` is attached at RENDER time (horizon-dependent), not carried
            "day%": s["day_ret"], "structure": s["structure"], "bar_clr": s["bar_clr"],
            "character": s["character"], "vs_vwap%": s["vs_vwap"],
            "above_vwap": s["above_vwap"], "rsi7": s["rsi7"], "rsi14": s["rsi14"],
            "tone": s["tone"], "RS%": s.get("rs_vs_index"),
            "entry": lv.get("entry"), "stop": lv.get("stop"),
            "t1": lv.get("t1"), "t2": lv.get("t2"),
            "s_stop": s_stop, "s_t1": s_t1, "s_t2": s_t2, "atr%": lv.get("atr%"),
            "action": _tf_action(s, risk_on), "sell": _tf_sell_action(s, risk_on),
            "_atr_tf": atr_tf, "_pc": live_pc,
            "_daylow": float(cndl["low"].min()) if (cndl is not None and len(cndl)) else None,
            "_bar_h": float(cndl["high"].iloc[-1]) if (cndl is not None and len(cndl)) else None,
            "_bar_l": float(cndl["low"].iloc[-1]) if (cndl is not None and len(cndl)) else None,
            "_vwap": s.get("vwap"),
        }

    # Same parallel-I/O pattern as the universe scan: each survivor's deep_state is one blocking
    # /history fetch, independent of the rest. Pre-warm the shared index series once (RS leg
    # source), then fan out. Bounded pool — survivors are capped upstream, so this is small.
    def _one_safe(r):
        # deep_state -> fetch_intraday does a bare requests.get: a network blip / bad candle on
        # ONE name would raise, and ex.map surfaces it on iteration — killing the WHOLE table.
        # Isolate each name: a failure drops that row, never the batch.
        try:
            return _one(r)
        except Exception:
            return None

    from concurrent.futures import ThreadPoolExecutor
    _index_intraday_5m(); _nifty_prev_close()       # warm the RS-leg source before the fan-out
    recs = [r for _, r in board.iterrows()]
    with ThreadPoolExecutor(max_workers=_SCAN_WORKERS) as ex:
        out = [o for o in ex.map(_one_safe, recs) if o is not None]
    eb = pd.DataFrame(out)
    if not eb.empty:
        eb = eb.sort_values(["action", "bar_clr"], ascending=[True, False])
    return eb


def refresh_prices(board: pd.DataFrame, risk_on: bool = True) -> pd.DataFrame:
    """TIER-1 refresh of a tf_scan board from ONE batch quote: live ltp / day% / RS, the
    ATR levels rebuilt on that price, AND the VERDICT recomputed.

    Why this exists: in the tf table the EXPENSIVE half (structure, RSI, tone — ~70
    /history calls) only changes when a BAR CLOSES (on 4h, twice a day), while the CHEAP
    half changes every tick. Freezing the whole table froze the wrong half.

    Why the verdict is recomputed too: a live price beside a stale LONG is worse than a
    stale table. A name flagged LONG at 10:02 can slide under VWAP by 10:45 — and it would
    still have read LONG, now next to a live (lower) price, which is exactly what an
    actionable signal looks like. bar_clr, vs_vwap and RS all rebuild from the quote (the
    bar's high/low and the session VWAP are carried on the row), so the gate is honest.
    RSI/tone/structure still need candles and stay as-of the scan — they are slow-moving
    and the table stamps their age."""
    if board is None or board.empty or "_atr_tf" not in board.columns:
        return board
    b = board.copy()
    q = _fetch_quotes([fy_symbol(s) for s in b["symbol"]])
    idx_ret = _chp(_fetch_quotes([config.NIFTY_FYERS]).get(config.NIFTY_FYERS, {}))
    for i, r in b.iterrows():
        v = q.get(fy_symbol(r["symbol"]))
        if not v or not v.get("lp"):
            continue
        ltp = float(v["lp"])
        pc = v.get("prev_close_price") or r.get("_pc")
        atr = float(r.get("_atr_tf") or 0)
        b.at[i, "ltp"] = round(ltp, 2)
        day = 100 * (ltp / float(pc) - 1) if pc else r.get("day%")
        b.at[i, "day%"] = round(day, 2) if day is not None else None
        # since% is DERIVED from ltp — if the price ticks and this does not, the column
        # silently contradicts the two numbers sitting beside it (at, ltp).
        _at = r.get("at")
        if _at and float(_at) > 0:
            b.at[i, "since%"] = round(100 * (ltp / float(_at) - 1), 2)
        if atr > 0:
            lv = indicators.levels(ltp, atr, day_low=r.get("_daylow"))
            for k in ("entry", "stop", "t1", "t2", "atr%"):
                b.at[i, k] = lv.get(k)
            b.at[i, "s_stop"] = round(ltp + atr, 2)      # mirror geometry for the short side
            b.at[i, "s_t1"] = round(ltp - atr, 2)
            b.at[i, "s_t2"] = round(ltp - 2 * atr, 2)

        # ── rebuild the VERDICT's cheap inputs, then the verdict itself ──────────────
        bh, bl, vw = r.get("_bar_h"), r.get("_bar_l"), r.get("_vwap")
        st_ = {"tone": r.get("tone"), "clr": r.get("bar_clr")}
        if bh and bl:                       # the forming bar's range must include the LTP
            bh, bl = max(float(bh), ltp), min(float(bl), ltp)
            bclr = ((ltp - bl) / (bh - bl)) if bh > bl else 0.5
            st_["bar_clr"] = round(bclr, 3)
            b.at[i, "bar_clr"] = st_["bar_clr"]
        else:
            st_["bar_clr"] = r.get("bar_clr")
        if vw and float(vw) > 0:
            vs = 100 * (ltp / float(vw) - 1)
            b.at[i, "vs_vwap%"] = round(vs, 2)
            st_["above_vwap"] = ltp >= float(vw)
        else:
            st_["above_vwap"] = bool(r.get("above_vwap"))
        rs = (day - idx_ret) if (day is not None and idx_ret is not None) else None
        if rs is not None:
            b.at[i, "RS%"] = round(rs, 2)
        st_["rs_vs_index"] = rs
        b.at[i, "action"] = _tf_action(st_, risk_on)
        b.at[i, "sell"] = _tf_sell_action(st_, risk_on)
    # BIG-GAP FOLLOWS THE TICK TOO. The big-wall PRICE is a 1D/1W/1M level that cannot repaint
    # intraday (so the wall itself stays), but the GAP to it is a function of the LIVE price --
    # exactly like headroom, which _live_levels already ticks. Leaving big_gap frozen while
    # headroom ticked was the same "stale number beside a live price" inconsistency. Re-derive
    # the gap to the SAME pinned wall; the string (frame + price + touches) is unchanged.
    if "_big_wall_px" in b.columns:
        _bpx = pd.to_numeric(b["_big_wall_px"], errors="coerce")
        _bltp = pd.to_numeric(b["ltp"], errors="coerce")
        _batr = pd.to_numeric(b["_sr_atr"], errors="coerce")
        _ok = _bpx.notna() & (_batr > 0)
        b.loc[_ok, "big_gap"] = ((_bpx[_ok] - _bltp[_ok]).abs() / _batr[_ok]).round(2)
    return _live_levels(b)


_REPLAY_TF = {"15m": None, "1h": "60min", "2h": "120min", "4h": "240min"}


def _resample_ohlcv(df: pd.DataFrame, freq: str | None) -> pd.DataFrame:
    """Resample one symbol's intraday OHLCV (ts sorted) to a coarser bar `freq`,
    aligned to the 09:15 session open. None -> unchanged (native fine bars).

    THE TRAILING STUB. NSE trades 09:15-15:30 = 375 minutes, which 60 and 120 do not divide.
    Binning from 09:15 therefore ends every single day with a bar built from ONE 15-minute
    candle (15:15-15:30) that the rest of the pipeline treats as a full 1h or 2h bar. That is
    not a cosmetic mismatch -- the structure classifier compares bar RANGES against each other
    and normalises by ATR, so a bar with a quarter (or an eighth) of the usual span is a
    different random variable wearing the same label. Measured over 70 names, 60 days:

        1h : 24/70 names (34%) carry a DIFFERENT structure label once the stub is folded in;
             ATR runs 6% low
        2h : 26/70 (37%) change; ATR runs 26% low, and the dominant flip is
             CONSOLIDATION -> RANGE (14 of 26) -- the stub was MANUFACTURING coils
        4h : 0/70 change (its second bar holds 9 of 16 candles = a real 2h15m bar, which is
             also what every charting package shows, so it is left alone)

    Same defect family as the partial weekly bar and the original coil detector: a window
    comparison where one side holds fewer bars. 1h is the confirmation frame of the Intraday
    preset and the TRIGGER frame of BTST, so a third of those two boards was reading a label
    produced by the last fifteen minutes of the day.

    Fix: fold a bin holding less than half its nominal candles into the PREVIOUS bar (so the
    day's last 1h bar spans 14:15-15:30). Nothing is discarded -- the close, the high/low and
    the volume of those fifteen minutes all survive into the bar before it, which is also the
    honest chartist reading: that stub is the tail of the 14:15 bar, not a bar of its own."""
    if not freq or df.empty:
        return df
    r = (df.set_index("ts").groupby(pd.Grouper(freq=freq, origin="start_day", offset="9h15min"))
         .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
              close=("close", "last"), volume=("volume", "sum")).dropna().reset_index())
    return merge_session_stubs(r, int(pd.Timedelta(freq).total_seconds() // 60))


_SESSION_END = pd.Timedelta("15h30min")     # NSE cash close
_STUB_FRAC = 0.5                            # less than half a bar's worth of session = not a bar


def merge_session_stubs(df: pd.DataFrame, freq_min: int) -> pd.DataFrame:
    """Fold a bar that has less than half its nominal span of SESSION left into the bar before
    it. Applies to any coarse intraday frame, however the bars were produced.

    Whose bars these are matters less than what they are: the broker's own 60-minute series
    has the identical defect, so this must run on both paths or the board and the trade card
    disagree about what '1h' means. Measured on RELIANCE: broker 1h -> TREND_UP / ATR 7.84,
    our resample -> BREAKOUT_UP / ATR 8.02, same name, same minute. Only 4h escapes, because
    its second bar (13:15-15:30) holds 135 of 240 minutes and is a real bar -- which is also
    what every charting package draws.

    The rule is SPAN, not candle count, deliberately: a bar with missing candles but a full
    span is a thin bar, and thinness is a liquidity fact worth seeing, not a reason to merge."""
    if df is None or df.empty or freq_min <= 15:
        return df
    rows: list[dict] = []
    for _, row in df.iterrows():
        d = row.to_dict()
        ts = d["ts"]
        avail = (min(ts + pd.Timedelta(minutes=freq_min), ts.normalize() + _SESSION_END) - ts)
        if rows and avail < pd.Timedelta(minutes=_STUB_FRAC * freq_min):
            p = rows[-1]                                   # fold the stub into the bar before it
            p["high"] = max(p["high"], d["high"])
            p["low"] = min(p["low"], d["low"])
            p["close"] = d["close"]
            if "volume" in p and "volume" in d:
                p["volume"] = p["volume"] + d["volume"]
        else:
            rows.append(d)
    return pd.DataFrame(rows)


def _index_intraday_5m() -> pd.Series:
    """Nifty's 5-min closes for TODAY, indexed by ts — for the running RS leg of the
    trigger. Cached per 5-min bucket (one call, shared by every symbol)."""
    key = ("_NIFTY5", dt.date.today(), _bucket5())
    if key in _SESS_CACHE:
        return _SESS_CACHE[key]
    try:
        d = fetch_intraday(config.NIFTY_FYERS.replace("NSE:", "").replace("-INDEX", ""),
                           tf="5m", lookback_days=2)
        if d.empty:
            d = pd.DataFrame(columns=["ts", "close"])
    except Exception:
        d = pd.DataFrame(columns=["ts", "close"])
    if not d.empty:
        d = d[d["ts"].dt.date == d["ts"].dt.date.max()]
    s = d.set_index("ts")["close"] if not d.empty else pd.Series(dtype=float)
    _SESS_CACHE[key] = s
    return s


def _first_formed(gf: pd.DataFrame, prev_close: float,
                  ref_avg_vol: float | None = None, _with_price: bool = False,
                  _rs_ctx: tuple | None = None):
    """The HH:MM the accumulation footprint FIRST formed, causally, from the fine
    intraday bars (≤ cut). Legs, all live-computable and evaluated at EVERY bar:
      up ≥ RET_TH on the day · running close-in-range ≥ CLR_TH · above running session
      VWAP · volume ON PACE for ≥ VOL_TH× a normal day (time-normalised).

    The volume leg needs `ref_avg_vol` (20d median daily volume). It matters: without
    it, a name 'forms' at 09:20 on one 5-min bar of tape. With it, the name must ALSO be
    genuinely participating — which is what the validated footprint means. Time-normalised
    so 2.0 means 'on pace for a 2x day' at any hour, not '2x already traded' (impossible
    before the close). None if it never formed by the cut."""
    if gf is None or len(gf) == 0 or not prev_close:
        return (None, None) if _with_price else None
    d = gf.sort_values("ts")
    h, l, c = d["high"].to_numpy(float), d["low"].to_numpy(float), d["close"].to_numpy(float)
    v = d["volume"].to_numpy(float)
    tp = (h + l + c) / 3.0
    cv = np.cumsum(v)
    vwap = np.where(cv > 0, np.cumsum(tp * v) / np.where(cv > 0, cv, 1), c)   # running session VWAP
    run_hi = np.maximum.accumulate(h)
    run_lo = np.minimum.accumulate(l)
    rng = run_hi - run_lo
    clr = np.where(rng > 0, (c - run_lo) / np.where(rng > 0, rng, 1), 0.5)     # running close-in-range
    day_ret = c / prev_close - 1.0
    # PATH SIGNATURE at the exact validated threshold: the signal wants the close a clear
    # CVWAP_TH above session VWAP (trended-and-held), not merely >= VWAP. Using ">= vwap"
    # here made `entered` fire on a LOOSER condition than the trade it claims to mark.
    cvwap = np.where(vwap > 0, (c - vwap) / np.where(vwap > 0, vwap, 1), 0.0)
    formed = ((day_ret >= config.RET_TH) & (clr >= config.CLR_TH)
              & (cvwap >= config.CVWAP_TH))
    if _rs_ctx is not None:
        # PERSISTENT RS leg: the completed 9 sessions (from the archive) PLUS the running
        # intraday RS at this bar (stock so far − index so far). Without it, `entered` was
        # ignoring a leg the validated signal requires.
        _cum9, _idx_ret_at = _rs_ctx
        formed = formed & ((_cum9 + (day_ret - _idx_ret_at)) > config.RS_MIN)
    if ref_avg_vol and ref_avg_vol > 0:                 # volume-on-pace leg (time-normalised)
        # cumulative volume through bar i covers up to that bar's CLOSE, so the elapsed-
        # volume fraction must be read at the bar's close, not its open (ts). Using the
        # open overstates pace on the first bars (~1.6x at 09:15) = false early triggers.
        ts = d["ts"]
        dur = (ts.iloc[1] - ts.iloc[0]) if len(ts) > 1 else pd.Timedelta(minutes=5)
        frac = np.array([indicators.day_fraction((t + dur).strftime("%H:%M")) for t in ts])
        pace = np.where(frac > 0, (cv / float(ref_avg_vol)) / np.where(frac > 0, frac, 1), 0.0)
        formed = formed & (pace >= config.VOL_TH)
    idx = np.argmax(formed) if formed.any() else -1
    if idx < 0:
        return (None, None) if _with_price else None
    t = d["ts"].iloc[idx].strftime("%H:%M")
    return (t, float(c[idx])) if _with_price else t


def _formed_at(gf: pd.DataFrame, prev_close: float,
               ref_avg_vol: float | None = None, rs_ctx: tuple | None = None) -> tuple:
    """(HH:MM, price) of the bar where the footprint first formed — the trigger PRICE, so
    the board can show how far the name has run (or faded) SINCE it fired. (None, None) if
    it never formed."""
    return _first_formed(gf, prev_close, ref_avg_vol, _with_price=True, _rs_ctx=rs_ctx)


def replay_board(date, time_str: str = "13:00", tf: str = "15m",
                 resolution: str = "15") -> dict:
    """Reconstruct the board AS OF `date` `time_str` (HH:MM), causally — only bars at
    or before that minute are used (no lookahead). This is the practice/backtest lens:
    at 1pm you see FORMING; scrub to 15:15 to see which became BTST-CARRY. `tf` sets the
    candle timeframe (15m native, or 1h/2h/4h resampled) that VWAP/RSI/structure read."""
    ts = token_status()
    d = pd.Timestamp(date)
    risk_on = regime.is_risk_on(d)
    if not ts["usable"]:
        return {"ok": False, "status": ts["describe"], "risk_on": risk_on, "board": pd.DataFrame()}
    cut = dt.datetime.strptime(time_str, "%H:%M").time()
    freq = _REPLAY_TF.get(tf)                       # None for 15m (native fine bars)
    allc = _fetch_day_candles(date, resolution)
    if allc.empty:
        return {"ok": True, "status": ts["describe"], "risk_on": risk_on,
                "board": pd.DataFrame(), "time": time_str, "date": str(d.date())}
    uni = liquid_universe(date).set_index("symbol")
    # index return as of cut (Nifty), causal
    idx_ret = None
    try:
        niff = fetch_intraday_range(config.NIFTY_FYERS, date, resolution)
        niff = niff[niff["ts"].dt.time <= cut]
        if len(niff) >= 2:
            idx_ret = 100 * (niff["close"].iloc[-1] / niff["open"].iloc[0] - 1)
    except Exception:
        pass
    earn = _events_upcoming(d.date())
    # coarse tf: multi-day causal frame for a stable structure/RSI read (a single day
    # has too few 2h/4h bars). Grouped by symbol for O(1) lookup in the loop.
    cut_dt = dt.datetime.combine(d.date(), cut)
    tfh = None
    if freq is not None:
        _h = _fetch_tf_history(date, tf)
        if not _h.empty:
            _h = _h[_h["ts"] <= cut_dt]                # causal: nothing after the cut minute
            tfh = {s: gg for s, gg in _h.groupby("symbol")}
    rows = []
    for sym, g in allc[allc["ts"].dt.time <= cut].groupby("symbol"):
        if sym not in uni.index or len(g) < 1:
            continue
        gf = g                                       # fine bars — for the causal "entered" time
        g = _resample_ohlcv(g, freq)                 # coarsen to chosen tf (15m = native)
        if g.empty:
            continue
        pc = uni.loc[sym, "prev_close"]              # prior-day close = correct day% baseline
        pc = float(pc) if pc == pc else float(g["open"].iloc[0])
        st_ = indicators.live_state(g, pc, uni.loc[sym, "vol_med20"],
                                    (idx_ret / 100.0) if idx_ret is not None else None,
                                    now_hhmm=time_str)   # CAUSAL clock = the replay cut
        # coarse tf: recompute structure + RSI on the multi-day causal frame (single-day
        # 2h/4h has too few bars). VWAP/clr/character stay session-based (from g).
        if tfh is not None and sym in tfh and len(tfh[sym]) >= 5:
            h = tfh[sym]
            st_["structure"] = indicators.structure(h)
            rst = indicators.rsi_state(h["close"].to_numpy(float))
            st_["rsi7"], st_["rsi14"], st_["tone"] = rst["rsi7"], rst["rsi14"], rst["tone"]
        pa = {k: st_[k] for k in ("clr", "body", "upper_wick", "lower_wick", "character")}
        day_ret, rs = st_["day_ret"], st_.get("rs_vs_index")
        vsurge = st_["vol_pace"]        # time-normalised (causal: 'now' = last bar ≤ cut)
        atr14 = float(uni.loc[sym, "atr14"]) if uni.loc[sym, "atr14"] == uni.loc[sym, "atr14"] else 0.0
        lv = indicators.levels(st_["ltp"], atr14, day_low=float(g["low"].min()))
        # replay's "today" IS this archive row, so the delivery and RS baselines must come
        # from the PRIOR row — today's delivery is not published until ~6pm, and today's RS
        # is added live below. (Using the through-row values would be lookahead.)
        _dtr = uni.loc[sym, "deliv_trail_prior"]
        dtrail = float(_dtr) if _dtr == _dtr else 0.0
        _c9 = uni.loc[sym, "rs_cum9_prior"]
        rs_cum = (float(_c9) + (rs / 100.0)) if (_c9 == _c9 and rs is not None) else None
        cvwap = st_["vs_vwap"] / 100.0          # PATH SIGNATURE — close vs session VWAP
        ready = btst_readiness(pa, day_ret, rs_cum, vsurge, cvwap)
        _trg, _trgpx = _formed_at(gf, pc, uni.loc[sym, "vol_med20"],
                                  _rs_context(gf["ts"], uni.loc[sym, "rs_cum9_prior"]))
        rows.append({
            "symbol": sym, "time": g["ts"].iloc[-1].strftime("%H:%M"),
            "entered": _trg,
            "at": round(_trgpx, 2) if _trgpx else None,
            "since%": round(100 * (st_["ltp"] / _trgpx - 1), 2) if _trgpx else None,
            "sector": uni.loc[sym, "sector"], "ltp": st_["ltp"],
            "day%": day_ret, "structure": st_["structure"], "clr": pa["clr"],
            "character": pa["character"], "vwap": st_["vwap"], "vs_vwap%": st_["vs_vwap"],
            "cvwap%": round(100 * cvwap, 2),
            "rsi7": st_["rsi7"], "rsi14": st_["rsi14"], "tone": st_["tone"],
            "vol×": round(vsurge, 2) if vsurge == vsurge else None,
            "RS%": round(rs, 2) if rs is not None else None,
            "rsCum%": round(100 * rs_cum, 2) if rs_cum is not None else None,
            "btst": f"{ready}/{BTST_LEGS}",
            "delivTr": round(dtrail, 1) if dtrail else None,
            "entry": lv.get("entry"), "stop": lv.get("stop"),
            "t1": lv.get("t1"), "t2": lv.get("t2"),
            "s_stop": round(st_["ltp"] + atr14, 2) if atr14 > 0 else None,
            "s_t1": round(st_["ltp"] - atr14, 2) if atr14 > 0 else None,
            "s_t2": round(st_["ltp"] - 2 * atr14, 2) if atr14 > 0 else None,
            "atr%": lv.get("atr%"),
            "action": ("EARNINGS" if sym in earn else _live_action(
                pa, day_ret, rs_cum, vsurge, risk_on, now_time=cut,
                deliv_trail=dtrail, cvwap=cvwap)),
            "sell": _sell_action(pa, day_ret, rs, vsurge),
        })
    board = pd.DataFrame(rows)
    if not board.empty:
        board = board.sort_values(["action", "btst"], ascending=[True, False])
    return {"ok": True, "status": ts["describe"], "risk_on": risk_on, "idx_ret": idx_ret,
            "board": board, "time": time_str, "tf": tf, "date": str(d.date()), "n": len(board)}


def fetch_intraday_range(sym: str, date, resolution: str = "15") -> pd.DataFrame:
    """One symbol's candles for a single past date."""
    d = pd.Timestamp(date).date()
    r = requests.get(config.FYERS_HISTORY_URL,
                     headers={"Authorization": _auth_header(), "version": "3"},
                     params={"symbol": sym, "resolution": resolution, "date_format": "1",
                             "range_from": d.isoformat(), "range_to": d.isoformat(),
                             "cont_flag": "1"}, timeout=15)
    j = r.json()
    if j.get("s") != "ok" or not j.get("candles"):
        return pd.DataFrame()
    f = pd.DataFrame(j["candles"], columns=["ts", "open", "high", "low", "close", "volume"])
    f["ts"] = pd.to_datetime(f["ts"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return _dedupe_candles(f)       # range_from == range_to doubles every bar -- see _DUPE_NOTE


def _events_upcoming(d0) -> set:
    try:
        from . import events
        return events.upcoming(d0, horizon_days=3)
    except Exception:
        return set()


def _tf_action(s: dict, risk_on: bool) -> str:
    """Action on a bar timeframe: strong BAR (the last bar OF THIS TIMEFRAME — that is what
    'bar close-strength' means; s['clr'] is the whole session's and is a different metric)
    + above session VWAP + RSI not weak + RS>0. CONTEXT ONLY — proven -5bps at every tf."""
    if not risk_on:
        return "AVOID"
    bclr = s.get("bar_clr", s["clr"])
    strong = (bclr >= config.CLR_TH and s["above_vwap"]
              and s["tone"] in ("strong", "neutral")
              and (s.get("rs_vs_index") is None or s["rs_vs_index"] > 0))
    if strong:
        return "LONG"
    if bclr <= 0.33 or not s["above_vwap"]:
        return "AVOID"
    return "NEUTRAL"


def _tf_sell_action(s: dict, risk_on: bool) -> str:
    """SHORT side on a bar timeframe (INTRADAY ONLY, unvalidated): weak bar + below
    VWAP + RSI weak/rolling + RS-laggard. Square off same day."""
    # Read the SAME metric the long side does — the last bar OF THIS TIMEFRAME. Using the
    # session's clr here while _tf_action uses bar_clr put two different quantities in one
    # table under one timeframe column, so the SHORT verdict was not a timeframe read at all.
    bclr = s.get("bar_clr", s["clr"])
    weak = (bclr <= (1 - config.CLR_TH) and not s["above_vwap"]
            and s["tone"] in ("weak", "rolling-over")
            and (s.get("rs_vs_index") is None or s["rs_vs_index"] < 0))
    if weak:
        return "SHORT"
    if bclr <= 0.4 and not s["above_vwap"]:
        return "WEAK"
    return "—"


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
    # Use TODAY's bars whenever there is at least one. The old `len(today) >= 2 else c`
    # fallback silently swapped in the FULL MULTI-DAY frame — and on 4h today holds just
    # ONE bar until 13:15 (2h until 11:15), so every morning clr/VWAP were computed over
    # ~30 DAYS: "clr" became position in the 30-day range and "above_vwap" meant above the
    # 30-day VWAP. The list then selected "stocks near a 30-day high" while the UI claimed
    # "bar close-strength · above-VWAP". One bar is enough for both.
    session = today if len(today) >= 1 else c
    pc = ref_close if ref_close else float(c["close"].iloc[0])
    state = indicators.live_state(session, pc, ref_avg_vol,
                                  (idx_ret / 100.0) if idx_ret is not None else None)
    # STRUCTURE and RSI must read the MULTI-DAY tf frame, not one session. A single day
    # holds ~7 bars at 1h, 4 at 2h, 2 at 4h — fewer than RSI's period, so rsi() would fall
    # back to its 50.0 default, tone would read "neutral", and the "RSI not weak" leg of
    # _tf_action would silently ALWAYS PASS (a vacuous filter) on every coarse timeframe.
    # A 4h RSI means an RSI of 4h bars — which necessarily chains across days.
    state["structure"] = indicators.structure(c)
    _rs = indicators.rsi_state(c["close"].to_numpy(float))
    state["rsi7"], state["rsi14"] = _rs["rsi7"], _rs["rsi14"]
    state["slope"], state["tone"] = _rs["slope"], _rs["tone"]
    # TRUE bar close-strength: the LAST bar OF THE CHOSEN TIMEFRAME. state["clr"] is the
    # close-in-range of today's whole SESSION (the BTST metric) — a different thing. The
    # tf table's column and caption both promise "bar close-strength", so compute it.
    _b = c.iloc[-1]
    _rng = float(_b["high"]) - float(_b["low"])
    state["bar_clr"] = round((float(_b["close"]) - float(_b["low"])) / _rng, 3) if _rng > 0 else 0.5
    atr_tf = indicators.atr(c, 14)                          # ATR on the chosen timeframe
    lv = indicators.levels(state["ltp"], atr_tf,
                           day_low=float(session["low"].min()),
                           day_high=float(session["high"].max()))
    return {"tf": tf, "state": state, "levels": lv, "atr_tf": atr_tf, "candles": c}
