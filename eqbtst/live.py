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
    """Drop the cached EOD universe (the dashboard's ↻ refresh calls this, so a fresh
    nightly sync is picked up immediately rather than after the TTL)."""
    _UNI_CACHE.clear()


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
    actually matters (does our baseline equal the broker's baseline?)."""
    n = mism = 0
    for fys, v in quotes.items():
        sym = fys.replace("NSE:", "").replace("-EQ", "")
        if sym not in ref.index:
            continue
        bpc = v.get("prev_close_price")
        apc = ref.loc[sym, "ref_close"]
        if not bpc or apc != apc or float(apc) <= 0:
            continue
        n += 1
        if abs(float(bpc) / float(apc) - 1.0) > tol:
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
        risk_on = regime.is_risk_on_live(nifty.get("lp"))
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
            ex = _session_extras(s_, _live_pc[s_], ref.loc[s_, "vol_med20"])
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
_RES = {"4h": "240", "2h": "120", "1h": "60", "15m": "15", "5m": "5"}
# lookback days per tf — coarse bars = fewer/day, need more days for ATR14/RSI14/structure(20).
# 4h ≈ 1.5 bars/day → 30d ≈ 45 bars; 2h ≈ 3/day → fine at 15d; intraday minutes plenty at 10d.
_LOOKBACK = {"4h": 30, "2h": 15}


def fetch_intraday(sym: str, tf: str = "1h", lookback_days: int | None = None) -> pd.DataFrame:
    """Candles for one symbol at timeframe tf ('4h','2h','1h','15m','5m') via /history.
    Pulls a few days so ATR14/RSI14 have enough bars on the coarse (hourly+) frames."""
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
    return df


_SESS_CACHE: dict = {}          # (symbol, date) -> ({trigger, vwap}, checked_5min_bucket)


def _bucket5(now: dt.datetime | None = None) -> str:
    """The current 5-minute bar bucket — no new intraday information can arrive within
    one, so a miss need not be re-fetched until the bucket turns."""
    n = now or dt.datetime.now()
    return f"{n.hour:02d}:{(n.minute // 5) * 5:02d}"


def _session_extras(sym: str, ref_close: float, ref_avg_vol: float | None = None) -> dict:
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
        trig, trig_px = _formed_at(fine, ref_close, ref_avg_vol)
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
        risk_on = regime.is_risk_on_live(_nf.get("lp"))
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
        _tfmin = {"4h": 240, "2h": 120, "1h": 60, "15m": 15, "5m": 5}.get(tf, 60)
        bar_time = None
        if cndl is not None and len(cndl):
            o = cndl["ts"].iloc[-1]
            close_t = min(o + pd.Timedelta(minutes=_tfmin), o.normalize() + pd.Timedelta("15h30min"))
            bar_time = f"{o.strftime('%H:%M')}-{close_t.strftime('%H:%M')}"   # candle span (open->close)
        # TRIGGER TIME — the wall-clock minute the footprint first fired, at 5-MIN
        # resolution, INDEPENDENT of the selected timeframe. Must not come from the tf
        # bars: a 4h bar only exists at 09:15/13:15, so it could never report a 12:30
        # trigger. The timeframe governs the structure/RSI/levels read, NOT the clock.
        _ex = _session_extras(sym, live_pc, uni.loc[sym, "vol_med20"])
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
        rows.append({
            "symbol": sym, "entered": entered, "at": at_px, "since%": since_pct,
            "time": bar_time,
            "bar": ("⏳ forming" if forming else "✓ closed") if forming is not None else None,
            "sector": uni.loc[sym, "sector"], "ltp": ltp,
            "day%": s["day_ret"], "structure": s["structure"], "bar_clr": s["bar_clr"],
            "character": s["character"], "vs_vwap%": s["vs_vwap"],
            "above_vwap": s["above_vwap"], "rsi7": s["rsi7"], "rsi14": s["rsi14"],
            "tone": s["tone"], "RS%": s.get("rs_vs_index"),
            "entry": lv.get("entry"), "stop": lv.get("stop"),
            "t1": lv.get("t1"), "t2": lv.get("t2"),
            "s_stop": s_stop, "s_t1": s_t1, "s_t2": s_t2, "atr%": lv.get("atr%"),
            "action": _tf_action(s, risk_on), "sell": _tf_sell_action(s, risk_on),
        })
    board = pd.DataFrame(rows)
    if not board.empty:
        board = board.sort_values(["action", "bar_clr"], ascending=[True, False])
    return {"ok": True, "status": ts["describe"], "tf": tf, "risk_on": risk_on,
            "idx_ret": idx_ret, "board": board, "n_scanned": len(pre)}


_REPLAY_CACHE = Path(__file__).resolve().parent.parent / "data" / "replay"


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
            f["symbol"] = sym
            frames.append(f)
        except Exception:
            continue
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        _REPLAY_CACHE.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache, index=False)
    return out


_REPLAY_TF = {"15m": None, "1h": "60min", "2h": "120min", "4h": "240min"}


def _resample_ohlcv(df: pd.DataFrame, freq: str | None) -> pd.DataFrame:
    """Resample one symbol's intraday OHLCV (ts sorted) to a coarser bar `freq`,
    aligned to the 09:15 session open. None -> unchanged (native fine bars)."""
    if not freq or df.empty:
        return df
    r = (df.set_index("ts").groupby(pd.Grouper(freq=freq, origin="start_day", offset="9h15min"))
         .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
              close=("close", "last"), volume=("volume", "sum")).dropna().reset_index())
    return r


def _first_formed(gf: pd.DataFrame, prev_close: float,
                  ref_avg_vol: float | None = None, _with_price: bool = False):
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
    formed = (day_ret >= config.RET_TH) & (clr >= config.CLR_TH) & (c >= vwap)
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
               ref_avg_vol: float | None = None) -> tuple:
    """(HH:MM, price) of the bar where the footprint first formed — the trigger PRICE, so
    the board can show how far the name has run (or faded) SINCE it fired. (None, None) if
    it never formed."""
    return _first_formed(gf, prev_close, ref_avg_vol, _with_price=True)


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
        _trg, _trgpx = _formed_at(gf, pc, uni.loc[sym, "vol_med20"])       # causal form time+price
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
    return f


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
    weak = (s["clr"] <= (1 - config.CLR_TH) and not s["above_vwap"]
            and s["tone"] in ("weak", "rolling-over")
            and (s.get("rs_vs_index") is None or s["rs_vs_index"] < 0))
    if weak:
        return "SHORT"
    if s["clr"] <= 0.4 and not s["above_vwap"]:
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
