"""
index_fno.py — Index F&O Management: the index option chain strike by strike, live.

Index Management (index_mgmt.py) answers "which weight bucket is moving the index". This lane
answers the options-desk question for the INDEX itself: on the 17 strikes around the money
(the ATM (At The Money) strike and 8 either side), what are calls and puts doing — open
interest, volume, implied volatility, premium — and how did that build through the session.

SOURCE: NSE'S OWN WEBSITE FEED, ZERO FYERS CALLS (same reason as index_mgmt)
    Probed live 6 Nov 2026, 15:37 (after the close):
      /api/option-chain-contract-info?symbol=NIFTY      expiry list (18 for Nifty) + strikes
      /api/option-chain-v3?type=Indices&symbol=NIFTY&expiry=09-Nov-2026
                                                       one expiry, EVERY listed strike (147 for
                                                       Nifty), both sides: lastPrice, change,
                                                       openInterest, changeinOpenInterest,
                                                       totalTradedVolume, impliedVolatility,
                                                       best bid / ask, + whole-chain totals
      …GetQuoteApi?functionName=getSymbolDerivativesData&symbol=NIFTY   index futures
    The old /api/option-chain-indices answers HTTP 404 — it is gone.

UNITS — CHECKED AGAINST THE BHAVCOPY, NOT ASSUMED
    openInterest and changeinOpenInterest are CONTRACTS. NIFTY 25000 CE 09-Nov-2026:
    live OI 176,995, change -7,011 -> yesterday 184,006 contracts x 65 = 11,960,390 shares =
    the 5 Nov bhavcopy exactly; the put at the same strike matched exactly too, and
    lastPrice - change = the bhavcopy close (35.90). Unlike the stock feed (which lists only
    contracts traded today) the index chain lists every strike, so  OI - change  is a
    complete yesterday. The lot is PROVEN per expiry from that ratio (_lot) and only falls
    back to a stated constant when no bhavcopy row exists (a brand-new expiry).

WHAT IS LOGGED
    data/index_fno/chain_{CODE}_{YYYYMMDD}.jsonl — one row per NSE refresh per expiry, the
    ATM strike and KEEP strikes either side (41 strikes), 8 numbers per side. The page shows
    8 either side; the extra 12 exist so a strike that comes INTO the window when the index
    moves already has its history from the open. fut_{CODE}_{YYYYMMDD}.jsonl — index futures.
    Logged only 09:15-15:30 and only when NSE's own timestamp moved (no duplicate rows).

WHAT THIS PAGE DOES NOT CLAIM — READ BEFORE ACTING ON A WORD
    1. No forecast. Tradebot's backtests on index option OI are null or negative: walls
       FOLLOW price, max pain does not pin (49% vs 50%), an OI surge precedes SMALLER moves,
       CE/PE crossovers are mildly contrarian at best. This is a map of positioning.
    2. "Long build / writing / short covering / long unwind" from PREMIUM change is what most
       sites print. On the stock archive that read was a price echo (same sign as the stock
       99% of the time) — a call's premium rises on any up-move whoever traded. The IV read
       (OI change vs NSE's own IV change since the first reading) is offered beside it; a
       rally lowers index IV across the chain (spot-vol), so on a strong up-day it leans
       "writers". Neither has been validated on index options.
    3. Live index-FUTURES open interest is not yet checked against the evening bhavcopy (the
       live STOCK-futures figure was found inflated). Option OI matched the bhavcopy exactly.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from . import index_mgmt as im

CODES = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY")
DISPLAY = im.DISPLAY
BAND = 8                 # strikes shown either side of ATM -> 17 strikes
KEEP = 20                # strikes LOGGED either side, so a strike entering the window has history
FIELDS = ("ltp", "chg", "oi", "doi", "vol", "iv", "bid", "ask")
NF = len(FIELDS)

CHAIN_EVERY = 60.0       # nearest expiry, and whatever expiry is on screen (NSE refreshes ~60 s)
OTHER_EVERY = 180.0      # the other captured expiries
FUT_EVERY = 120.0
EXPIRIES_EVERY = 3600.0
IDLE_EVERY = 900.0       # outside market hours, only for an index someone is looking at
VIEW_TTL = 600.0
STALE_AFTER = 240.0
HEAL_AFTER = 300.0
MIN_STRIKES = 2 * BAND + 1
# STRIKE GAP. NSE lists Nifty / Fin Nifty every 50 points and Midcap Select every 25 near the
# money, but positions sit on the round hundreds. Measured on 130 sessions (May-Nov 2026,
# nearest expiry, strikes within 4% of the future, bhavcopy): 100-point strikes held 70% of
# Nifty OI on half the strikes (median OI per strike 3.71M vs 1.23M on the 50s in between),
# 81% on Fin Nifty (5,745 vs 1,380), 80% on Midcap (226k vs 16k — the 50s hold 90%). Bank
# Nifty is listed every 100 already. So the page shows a 100-point grid by default; the
# listed grid is one click away.
GRID_DEFAULT = {"NIFTY": 100, "BANKNIFTY": 100, "FINNIFTY": 100, "MIDCPNIFTY": 100}
LOG_GRID, LOG_GRID_N = 100, 20   # also log every 100-pt strike within 20 steps of ATM
# Used ONLY when no bhavcopy row can prove the lot (a brand-new expiry with no other expiry
# proven today); the page then says "assumed". Nov 2026 values.
LOT_FALLBACK = {"NIFTY": 65, "BANKNIFTY": 30, "FINNIFTY": 60, "MIDCPNIFTY": 120}
CAPTURE_END = dt.time(15, 50)   # the writer keeps polling past 15:30 to catch NSE's corrected close
CLOSE_FROM = dt.time(15, 35)    # a chain stamped after this is kept as the day's CLOSE snapshot
HOLIDAY_EVERY = 300.0           # NSE still serving an older day's chain after 09:20 = shut today
IV_BAND = 0.5            # vol points: an IV move smaller than this is "unclear"
OI_FLOOR = 0.02          # share of the strike's OI: a smaller change since first reading = "little change"

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "index_fno"
_CHAIN = "/api/option-chain-v3?type=Indices&symbol={sym}&expiry={exp}"
_INFO = "/api/option-chain-contract-info?symbol={sym}"

_now = im._now
_num = im._num
_ts = im._ts
in_session = im.in_session
market_window = im.market_window


# ── parsing ───────────────────────────────────────────────────────────────────────────
def _exp_str(e: dt.date) -> str:
    return e.strftime("%d-%b-%Y")


def _side(d) -> tuple | None:
    if not isinstance(d, dict):
        return None
    v = [_num(d.get(k)) for k in ("lastPrice", "change", "openInterest",
                                  "changeinOpenInterest", "totalTradedVolume",
                                  "impliedVolatility", "buyPrice1", "sellPrice1")]
    # zeros that mean "nothing": no trade -> ltp 0 and IV 0; empty book -> bid/ask 0
    for i in (0, 5, 6, 7):
        if v[i] is not None and v[i] <= 0:
            v[i] = None
    if v[2] is None:                 # a side with no OI figure is not a contract we can read
        return None
    return tuple(v)


def parse_expiries(j) -> list:
    out = []
    for s in (j or {}).get("expiryDates") or []:
        t = _ts(str(s) + " 00:00")
        if t:
            out.append(t.date())
    return sorted(set(out))


def max_pain(ce_oi: dict, pe_oi: dict) -> float | None:
    """The settlement strike at which option BUYERS would collect least (writers pay least),
    over every listed strike of the expiry. Ties -> the lower strike."""
    ks = sorted(set(ce_oi) | set(pe_oi))
    if not ks:
        return None
    K = np.array(ks, dtype=float)
    c = np.array([ce_oi.get(k) or 0.0 for k in ks])
    p = np.array([pe_oi.get(k) or 0.0 for k in ks])
    if c.sum() + p.sum() <= 0:
        return None
    S = K[:, None]
    pay = (np.clip(S - K[None, :], 0, None) * c[None, :]
           + np.clip(K[None, :] - S, 0, None) * p[None, :]).sum(axis=1)
    return float(K[int(np.argmin(pay))])


def parse_chain(j, expiry: dt.date) -> dict:
    """One expiry of NSE's v3 chain -> dict. Raises im.NseError when unusable."""
    rec = (j or {}).get("records") or {}
    rows = rec.get("data") or []
    if not rows:
        raise im.NseError(f"empty chain for {expiry}")
    ce, pe = {}, {}
    for r in rows:
        K = _num(r.get("strikePrice"))
        if not K or K <= 0:
            continue
        # v3 answers one expiry, but be strict: a row from another expiry would mix series
        e = _ts(str(r.get("expiryDates") or r.get("expiryDate") or "") + " 00:00")
        if e and e.date() != expiry:
            continue
        a, b = _side(r.get("CE")), _side(r.get("PE"))
        if a:
            ce[K] = a
        if b:
            pe[K] = b
    spot = _num(rec.get("underlyingValue"))
    if not spot or spot <= 0:
        for r in rows:
            spot = _num((r.get("CE") or r.get("PE") or {}).get("underlyingValue"))
            if spot:
                break
    both = sorted(set(ce) & set(pe))
    if len(both) < MIN_STRIKES or not spot:
        raise im.NseError(f"chain {expiry}: {len(both)} two-sided strikes, spot {spot}")
    filt = (j or {}).get("filtered") or {}
    fc, fp = filt.get("CE") or {}, filt.get("PE") or {}
    tc = [_num(fc.get("totOI")), _num(fc.get("totVol")), _num(fp.get("totOI")),
          _num(fp.get("totVol"))]
    if tc[0] is None:                # totals missing: sum ourselves (same thing)
        tc = [sum(v[2] or 0 for v in ce.values()), sum(v[4] or 0 for v in ce.values()),
              sum(v[2] or 0 for v in pe.values()), sum(v[4] or 0 for v in pe.values())]
    return dict(ts=_ts(rec.get("timestamp")), spot=spot, expiry=expiry, ce=ce, pe=pe,
                strikes=both, tot=tc,
                mp=max_pain({k: v[2] for k, v in ce.items()}, {k: v[2] for k, v in pe.items()}),
                fetched=_now())


def pick_atm(strikes, spot: float) -> float | None:
    """Listed two-sided strike nearest to spot; an exact half-way tie goes to the lower."""
    if not strikes or spot is None:
        return None
    return float(min(strikes, key=lambda k: (abs(k - spot), k)))


def window(strikes, atm: float, n: int = BAND) -> list:
    """`atm` and n listed strikes either side (fewer at the edge of the listed range).
    Uses the LISTED strikes, so Nifty's 50-pt grid near the money and 100-pt grid further
    out are both respected; nothing assumes a step."""
    ks = sorted(strikes)
    if atm not in ks:
        atm = pick_atm(ks, atm)
    if atm is None:
        return []
    i = ks.index(atm)
    return ks[max(0, i - n): i + n + 1]


def on_grid(k: float, step: float | None) -> bool:
    return not step or abs(k / step - round(k / step)) < 1e-9


def grid(strikes, step: float | None) -> list:
    """The listed strikes on a `step`-point grid (all of them when step is None, or when the
    grid would leave fewer than 3 — never an empty ladder)."""
    ks = sorted(strikes)
    g = [k for k in ks if on_grid(k, step)]
    return g if step and len(g) >= 3 else ks


def listed_step(strikes, atm) -> float | None:
    """NSE's strike spacing at the money (Nifty 50, Bank Nifty 100, Midcap 25)."""
    ks = sorted(strikes)
    if atm not in ks or len(ks) < 2:
        return None
    i = ks.index(atm)
    gaps = [b - a for a, b in zip(ks[max(0, i - 3):i + 3], ks[max(0, i - 3) + 1:i + 4])]
    return float(min(gaps)) if gaps else None


def compact(chain: dict, keep: int = KEEP, anchor: float | None = None) -> dict:
    """A chain as one log row, both sides inline: ATM and `keep` LISTED strikes either side,
    plus every 100-point strike within LOG_GRID_N steps of the ATM AND of the day's opening
    level `anchor` — the page's ladder is fixed at the 09:15 open, so its far strikes must
    stay in the log however far the index runs from there (Bank Nifty on a 200 gap reaches
    1,600 points from the open; a 1% move used to push that edge out of the log)."""
    atm = pick_atm(chain["strikes"], chain["spot"])
    reach = (LOG_GRID_N + 1) * LOG_GRID
    centres = [atm] + ([anchor] if anchor else [])
    ks = sorted(set(window(chain["strikes"], atm, keep))
                | {k for k in chain["strikes"] if on_grid(k, LOG_GRID)
                   and any(abs(k - c) <= reach for c in centres)})
    none = [None] * NF
    return {"t": im._iso(chain["ts"]), "f": im._iso(chain["fetched"]),
            "e": chain["expiry"].isoformat(), "u": chain["spot"], "a": atm,
            "mp": chain["mp"], "tc": chain["tot"],
            "k": [[k, *(chain["ce"].get(k) or none), *(chain["pe"].get(k) or none)]
                  for k in ks]}


PARITY_NEAR = 3          # strikes either side of ATM whose call - put sets the forward
MID_MAX_SPREAD = 0.05    # use the bid/ask mid when the spread is within 5% of it (or 2 ticks)
ITM_IV_GAP = 5.0         # vol points: an ITM-side IV further than this from the OTM side is stale


def _px(d: dict | None):
    """Best price for one side: the quote mid when the book is tight, else the last trade."""
    if not d:
        return None
    b, a, l = d.get("bid"), d.get("ask"), d.get("ltp")
    if b and a and a >= b and (a - b) <= max(0.10, MID_MAX_SPREAD * (a + b) / 2):
        return (a + b) / 2
    # A last price only counts once the contract has TRADED TODAY. 5 Nov 2026 (a +4% gap),
    # Fin Nifty 26100 call at 09:15: volume 0, 'last' Rs 1,345 with the index at 26,140 —
    # a price from an earlier session; its IV read 67% until the first trade at 09:28 (14%).
    if (d.get("vol") or 0) <= 0:
        return None
    return l


def model_iv(row: dict, expiry: dt.date, t: dt.datetime | None) -> dict:
    """Our own IV for every logged strike, both sides — {"F": forward, "v": [[K, ivc, ivp]]}.

    WHY NOT NSE'S impliedVolatility. NSE prices each option against the SPOT index. On
    6 Nov 2026 at the close, NSE's call IV sat 5-9 points BELOW its put IV at the same
    strike on every Nifty and Bank Nifty expiry (ATM Nifty 09-Nov: call 16.0, put 24.2) —
    impossible for one surface: put-call parity ties them. Two causes: the closing index is
    NSE's 15:00-15:30 average while premiums are last trades, and spot ignores the carry to
    expiry. Here the forward comes FROM the options (median of K + (C - P)e^rT over the ATM
    and 3 strikes either side), prices are quote mids when the book is tight, and time is
    the market's clock (sessions left + what is left of today). Measured on the same 6 Nov
    snapshots: call vs put IV at one strike within 0.3-0.8 points (Midcap Nov 2.1).
    Empty when there is no time left (an expiry after 15:30 on its day) or no parity."""
    if t is None:
        return {}
    T = im._trading_years(t, expiry)
    if T <= 0:
        return {}
    ce, pe = row_sides(row)
    ks = sorted(set(ce) & set(pe))
    a = row.get("a")
    if a not in ks:
        return {}
    i = ks.index(a)
    r = im.OPT_RATE
    fs = []
    for K in ks[max(0, i - PARITY_NEAR): i + PARITY_NEAR + 1]:
        c, p = _px(ce[K]), _px(pe[K])
        if c and p:
            fs.append(K + (c - p) * np.exp(r * T))
    u = row.get("u")
    if len(fs) < 3 or not u:
        return {}
    F = float(np.median(fs))
    if abs(F / u - 1) > 0.03:            # a forward 3% from spot is a broken quote, not carry
        return {}
    allk = sorted(set(ce) | set(pe))
    K = np.array(allk, float)
    C = np.array([_px(ce.get(k)) or np.nan for k in allk], float)
    P = np.array([_px(pe.get(k)) or np.nan for k in allk], float)
    n = len(K)
    ivc = im.implied_vol(C, np.full(n, F), K, np.full(n, T), np.ones(n, bool)) * 100
    ivp = im.implied_vol(P, np.full(n, F), K, np.full(n, T), np.zeros(n, bool)) * 100
    # The IN-THE-MONEY side trades rarely and its last price goes stale (5 Nov 2026: Fin Nifty
    # 27100 put traded once all day, IV 26.8 vs ~14 across the chain; Bank Nifty 57600 put
    # read 30.9 at the open while in the money). Parity says call and put IV at one strike
    # are equal, so the ITM side is kept only when the OTM side exists and agrees within
    # ITM_IV_GAP; the OTM side (the liquid one, and the smile's) is always kept.
    itm_call = K < F
    otm = np.where(itm_call, ivp, ivc)
    itm = np.where(itm_call, ivc, ivp)
    ok = np.isfinite(otm) & np.isfinite(itm) & (np.abs(itm - otm) <= ITM_IV_GAP)
    ivc = np.where(itm_call & ~ok, np.nan, ivc)
    ivp = np.where(~itm_call & ~ok, np.nan, ivp)
    rd = lambda x: None if not np.isfinite(x) else round(float(x), 2)  # noqa: E731
    return {"F": round(F, 2), "v": [[k, rd(c), rd(p)] for k, c, p in zip(allk, ivc, ivp)]}


def _fp(row: dict) -> int:
    """Fingerprint of the chain DATA (not the stamp): a refresh that changes nothing is not
    a new reading, and a feed whose numbers stop moving while the stamp ticks is frozen."""
    return hash(json.dumps(row.get("k"), separators=(",", ":")))


def parse_fut(j) -> dict:
    """Index futures from getSymbolDerivativesData: {expiry iso: [ltp, prev, oi, doi, vol]}."""
    rows = (j or {}).get("data") or []
    F, spot = {}, None
    for r in rows:
        if not str(r.get("instrumentType") or "").startswith("FUT"):
            continue
        e = _ts(str(r.get("expiryDate")) + " 00:00")
        if not e:
            continue
        F[e.date().isoformat()] = [_num(r.get("lastPrice")), _num(r.get("prevClose")),
                                   _num(r.get("openInterest")),
                                   _num(r.get("changeinOpenInterest")),
                                   _num(r.get("totalTradedVolume"))]
        spot = spot or _num(r.get("underlyingValue"))
    if not F:
        raise im.NseError("no index futures in the payload")
    return {"t": im._iso(_ts(j.get("timestamp"))), "f": im._iso(_now()), "s": spot,
            "F": dict(sorted(F.items()))}


# ── capture: one writer per machine (its own lock; index_mgmt's lock is separate) ─────────
_LOCK = threading.RLock()
_S: dict = {}
_CLIENT: list = [None]
_MINE = {"owner": False}
_GEN_ATTR = "_eqbtst_index_fno_gen"
_LOCK_STALE = 60.0
_HEAL = {"at": 0.0, "refresh_seen": 0.0}
_BEAT = [0.0]


def _client() -> im.NseClient:
    if _CLIENT[0] is None:
        _CLIENT[0] = im.NseClient(min_gap=0.4)
    return _CLIENT[0]


def _st(code: str) -> dict:
    with _LOCK:
        return _S.setdefault(code, dict(expiries=[], exp_at=0.0, exp_day=None, chain={},
                                        chain_at={}, last_t={}, fp={}, log_fp={},
                                        changed_at={}, nearest=None, fut=None, fut_at=0.0,
                                        fut_last_t=None, viewed=0.0, view_exp=None,
                                        err=None))


_APPEND = threading.RLock()


def capture_window(now: dt.datetime | None = None) -> bool:
    """Mon-Fri 09:00-15:50: full-speed polling. Past 15:30 only to catch NSE's corrected
    close (it appeared at 15:40 on 6 Nov 2026); nothing after 15:30 enters the session log."""
    now = now or _now()
    return now.weekday() < 5 and dt.time(9, 0) <= now.time() <= CAPTURE_END


def _path(kind: str, code: str, day: dt.date) -> Path:
    return DATA_DIR / f"{kind}_{code}_{day:%Y%m%d}.jsonl"


def _claim() -> bool:
    f = DATA_DIR / "poller.lock"
    me = os.getpid()
    try:
        j = json.loads(f.read_text(encoding="utf-8"))
        if j.get("pid") != me and time.time() - float(j.get("beat", 0)) < _LOCK_STALE:
            _MINE["owner"] = False
            return False
    except (OSError, ValueError):
        pass
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"pid": me, "beat": time.time()}), encoding="utf-8")
        _MINE["owner"] = True
    except OSError:
        _MINE["owner"] = False
    return _MINE["owner"]


def _release() -> None:
    f = DATA_DIR / "poller.lock"
    try:
        if json.loads(f.read_text(encoding="utf-8")).get("pid") == os.getpid():
            f.unlink()
    except (OSError, ValueError):
        pass
    _MINE["owner"] = False


def _log_error(family: str, msg: str) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with (DATA_DIR / "errors.log").open("a", encoding="utf-8") as f:
            f.write(f"{_now():%Y-%m-%d %H:%M:%S} {family:7s} {msg}\n")
    except OSError:
        pass


def _last_logged(path: Path, expiry: str | None = None) -> dict | None:
    """The last row already in today's log (for one expiry), so a restarted writer neither
    logs the same refresh twice nor the same numbers under a new stamp."""
    last = None
    for r in im._load_jsonl(path):
        if (expiry is None or r.get("e") == expiry) and r.get("t"):
            last = r
    return last


DCM_INDEX_NAME = {"NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank",
                  "FINNIFTY": "Nifty Financial Services", "MIDCPNIFTY": "Nifty Midcap Select"}
_DCM_OPEN: dict = {}


def _dcm_open(code: str, day: dt.date) -> float | None:
    """DCM's end-of-day official index open (index_data), for finished days."""
    key = (code, day)
    if key in _DCM_OPEN:
        return _DCM_OPEN[key]
    v = None
    try:
        from . import data
        with data._connect() as c:
            r = c.execute("select open_val from index_data where trade_date = ? and index_name = ?",
                          [day, DCM_INDEX_NAME[code]]).fetchone()
        v = float(r[0]) if r and r[0] else None
    except Exception:                                        # noqa: BLE001
        return None                                          # locked archive: ask again later
    _DCM_OPEN[key] = v
    return v


_PREV: dict = {}


def prev_close(code: str, day: dt.date) -> float | None:
    """The index's previous session close — the base NSE and every broker use for '+/− today'.
    Index Management's same-day NSE read first (NSE's own previousClose, 11 Nov 2026: 25,044.65),
    then DCM's end-of-day row for `day`, then DCM's last close before `day` (a holiday in
    between is skipped by construction: 10 Nov 2026 was shut)."""
    try:
        j = json.loads((im.DATA_DIR / f"latest_{code}.json").read_text(encoding="utf-8"))
        c = j.get("cash") or {}
        t = dt.datetime.fromisoformat(c["ts"]) if c.get("ts") else None
        pv = (c.get("index") or {}).get("prev")
        if t and t.date() == day and pv and pv > 0:
            return float(pv)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    key = (code, day)
    if key in _PREV:
        return _PREV[key]
    v = None
    try:
        from . import data
        with data._connect() as c:
            r = c.execute("select prev_close from index_data where trade_date = ? and "
                          "index_name = ?", [day, DCM_INDEX_NAME[code]]).fetchone()
            if not (r and r[0]):
                r = c.execute("select close_val from index_data where trade_date < ? and "
                              "index_name = ? order by trade_date desc limit 1",
                              [day, DCM_INDEX_NAME[code]]).fetchone()
        v = float(r[0]) if r and r[0] else None
    except Exception:                                        # noqa: BLE001
        return None                                          # locked archive: ask again later
    _PREV[key] = v
    return v


def day_open(code: str, day: dt.date, rows: list) -> tuple:
    """(opening level, source text) — the 09:15 point the page fixes the ladder and buckets
    on. NSE's official open first (stored in the log rows by the writer, or DCM's end-of-day
    file for a finished day, or Index Management's same-day read); the first logged reading
    only when it is from the open itself (by 09:20); otherwise unknown — never a late
    reading passed off as the open (capture has started at 09:56 and 11:00 on past days)."""
    for r in reversed(rows or []):
        if r.get("o") and r.get("os") in ("nse", "nse-eod"):
            return float(r["o"]), "NSE official open"
    v = _dcm_open(code, day)
    if v:
        return v, "NSE official open (end-of-day file)"
    v = None
    try:
        j = json.loads((im.DATA_DIR / f"latest_{code}.json").read_text(encoding="utf-8"))
        c = j.get("cash") or {}
        t = dt.datetime.fromisoformat(c["ts"]) if c.get("ts") else None
        o = (c.get("index") or {}).get("open")
        if t and t.date() == day and t.time() >= OPEN_FROM and o and o > 0:
            v = float(o)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if v:
        return v, "NSE official open"
    if rows and rows[0].get("u") and rows[0]["t"][11:16] <= "09:20":
        return float(rows[0]["u"]), f"first reading {rows[0]['t'][11:19]} (official open not read)"
    return None, ("capture started " + rows[0]["t"][11:16] + " — opening level unknown"
                  if rows else "no reading from the open")


def captured_expiries(expiries: list, today: dt.date) -> list:
    """Nearest, next, and the nearest MONTHLY expiry (for Nifty the weekly is nearest, so
    that is three; for Bank / Fin / Midcap, monthly-only since Nov 2024, it is two)."""
    live = [e for e in expiries if e >= today]
    out = live[:2]
    monthly = [e for e in live if is_monthly(e, expiries)]
    if monthly and monthly[0] not in out:
        out.append(monthly[0])
    return out


def is_monthly(e: dt.date, expiries: list) -> bool:
    """Last listed expiry of its month. From NSE's own list, never from a weekday rule:
    holidays move expiries (09-Nov-2026 is a Monday)."""
    return not any(x > e and (x.year, x.month) == (e.year, e.month) for x in expiries)


def poll_expiries(code: str, client=None) -> list:
    s = _st(code)
    try:
        ex = parse_expiries((client or _client()).get(_INFO.format(sym=code)))
        if ex:
            with _LOCK:
                s["expiries"], s["exp_at"], s["exp_day"] = ex, time.time(), _now().date()
    except Exception as exc:                                 # noqa: BLE001
        s["exp_at"] = time.time() - EXPIRIES_EVERY + 120     # retry in 2 minutes
        s["err"] = f"expiries: {exc}"
    return s["expiries"]


OPEN_FROM = dt.time(9, 16)    # NSE's index 'open' is settled by then (the 09:15 print)
OPEN_RETRY = 60.0


def nse_open(code: str, day: dt.date, client=None) -> float | None:
    """NSE's OFFICIAL index open for `day` (the 09:15 print from the pre-open auction).
    Zero calls when Index Management's writer already holds a same-day read; otherwise one
    call to NSE's index feed. None before 09:16 or when the read is not today's."""
    try:
        j = json.loads((im.DATA_DIR / f"latest_{code}.json").read_text(encoding="utf-8"))
        c = j.get("cash") or {}
        t = dt.datetime.fromisoformat(c["ts"]) if c.get("ts") else None
        o = (c.get("index") or {}).get("open")
        if t and t.date() == day and t.time() >= OPEN_FROM and o and o > 0:
            return float(o)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        c = im.fetch_cash(code, client or _client())
        t, o = c.get("ts"), (c.get("index") or {}).get("open")
        if t and t.date() == day and t.time() >= OPEN_FROM and o and o > 0:
            return float(o)
    except Exception as exc:                                 # noqa: BLE001
        _log_error("open", f"{code}: {exc}")
    return None


def _day_open_writer(code: str, now: dt.datetime) -> float | None:
    s = _st(code)
    o = s.get("open") or {}
    if o.get("day") == now.date():
        return o.get("o")
    if not in_session(now) or now.time() < OPEN_FROM:
        return None
    if time.time() - s.get("open_tried", 0.0) < OPEN_RETRY:
        return None
    s["open_tried"] = time.time()
    v = nse_open(code, now.date())
    if v:
        s["open"] = {"day": now.date(), "o": v}
    return v


def poll_chain(code: str, expiry: dt.date, client=None) -> bool:
    s = _st(code)
    key = expiry.isoformat()
    s["chain_at"][key] = time.time()
    try:
        j = (client or _client()).get(_CHAIN.format(sym=code, exp=_exp_str(expiry)))
        ch = parse_chain(j, expiry)
    except Exception as exc:                                 # noqa: BLE001
        s["err"] = f"chain {key}: {exc}"
        # once per expiry per hour: a far expiry NSE serves nearly empty (Fin Nifty Dec on
        # 9 Nov 2026: one two-sided strike) failed every 3 minutes and buried real errors
        last = s.setdefault("err_logged", {}).get(key, 0.0)
        if time.time() - last >= 3600:
            s["err_logged"][key] = time.time()
            _log_error("chain", f"{code} {key}: {exc}")
        return False
    ts = ch["ts"] or ch["fetched"]
    day_open = _day_open_writer(code, _now()) if ts.date() == _now().date() else None
    row = compact(ch, anchor=day_open)
    if day_open:
        row["o"], row["os"] = day_open, "nse"
    try:
        # {"v": []} when the model found no forward: the row still counts as MODEL-sourced,
        # so its IV is blank rather than silently NSE's (stale at a gap open)
        row.update(model_iv(row, expiry, ts) or {"v": []})
    except Exception as exc:                                 # noqa: BLE001
        _log_error("iv", f"{code} {key}: {type(exc).__name__}: {exc}")
    fp = _fp(row)
    now_t = time.time()
    with _APPEND:                    # page thread and writer thread can both poll
        if s["fp"].get(key) != fp:
            s["fp"][key], s["changed_at"][key] = fp, now_t
        row["c"] = dt.datetime.fromtimestamp(s["changed_at"].get(key, now_t)).isoformat(
            timespec="seconds")
        with _LOCK:
            s["chain"][key] = row
            s["err"] = None
        today = _now().date()
        if in_session(ts) and ts.date() == today:
            p = _path("chain", code, ts.date())
            if key not in s["last_t"]:
                last = _last_logged(p, key)
                s["last_t"][key] = last.get("t") if last else None
                s["log_fp"][key] = _fp(last) if last else None
            # a new reading = NSE's numbers moved (never just its clock), not older than the last
            if row["t"] and s["log_fp"].get(key) != fp and (
                    s["last_t"][key] is None or row["t"] >= s["last_t"][key]):
                im._append(p, row)
                s["last_t"][key], s["log_fp"][key] = row["t"], fp
        elif ts.date() == today and CLOSE_FROM <= ts.time() <= dt.time(16, 30):
            _save_close(code, key, row)
    _save_latest(code)
    return True


def _close_path(code: str, day: dt.date) -> Path:
    return DATA_DIR / f"close_{code}_{day:%Y%m%d}.json"


def _save_close(code: str, key: str, row: dict) -> None:
    """NSE CORRECTS open interest after the bell. 6 Nov 2026, Nifty 09-Nov chain: whole-chain
    call OI 2,296,445 at 15:37 -> 2,235,879 at 15:40 (-2.6%), puts -2.2%; the 25000 call
    176,995 -> 170,238. The 15:40 figures matched the bhavcopy EXACTLY (every strike, OI and
    volume). Index FUTURES were worse: Nifty Nov +8,927 contracts on the day at 15:37,
    -8,509 official — the opposite sign. So the session log (09:15-15:30) never holds the
    true close; this file does, and reconcile() scores the log against it every night."""
    p = _close_path(code, dt.date.fromisoformat(row["t"][:10]))
    try:
        cur = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cur = {}
    old = (cur.get("chain") or {}).get(key) if key != "fut" else cur.get("fut")
    if old and old.get("t", "") > row.get("t", ""):
        return
    if key == "fut":
        cur["fut"] = row
    else:
        cur.setdefault("chain", {})[key] = row
    im._write_json(p, cur)


def load_close(code: str, day: dt.date) -> dict:
    try:
        return json.loads(_close_path(code, day).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


RECONCILE_FILE = "reconcile.jsonl"
RECONCILE_MIN_END = dt.time(15, 20)     # a log ending earlier is not scored (laptop slept)


def reconcile_day(code: str, day: dt.date) -> list:
    """Last in-session reading vs NSE's corrected close, per captured expiry (+ futures).

    Per expiry, over the ATM and 8 strikes either side at the close, both sides:
      sign  = share of contracts whose change-on-the-day had the SAME SIGN live and official
              (both non-zero) — what every 'build / unwind' word rests on
      err   = median |live change - official change| as % of official OI
      bias  = whole-band live OI / official OI - 1, calls and puts separately
    Futures: near-month live change vs official change, both in contracts."""
    close = load_close(code, day)
    out = []
    for e, crow in (close.get("chain") or {}).items():
        rows = chain_log(code, day, e)
        if not rows or rows[-1]["t"][11:16] < RECONCILE_MIN_END.strftime("%H:%M"):
            continue
        live = rows[-1]
        lce, lpe = row_sides(live)
        cce, cpe = row_sides(crow)
        ks = window(sorted(set(cce) & set(cpe)), crow.get("a"), BAND)
        agree, errs, rec = [], [], {"day": day.isoformat(), "code": code, "e": e,
                                    "live_t": live["t"], "close_t": crow["t"]}
        for side, L, C in (("CE", lce, cce), ("PE", lpe, cpe)):
            lo = co = 0.0
            for K in ks:
                l, c = L.get(K), C.get(K)
                if not l or not c or l.get("doi") is None or c.get("doi") is None:
                    continue
                lo += l["oi"]
                co += c["oi"]
                if l["doi"] and c["doi"]:
                    agree.append(np.sign(l["doi"]) == np.sign(c["doi"]))
                if c["oi"]:
                    errs.append(abs(l["doi"] - c["doi"]) / c["oi"] * 100)
            rec[f"bias_{side}"] = round((lo / co - 1) * 100, 2) if co else None
        rec["sign"] = round(float(np.mean(agree)) * 100, 1) if agree else None
        rec["err"] = round(float(np.median(errs)), 2) if errs else None
        rec["n"] = len(agree)
        out.append(rec)
    cf = close.get("fut")
    frows = fut_log(code, day)
    if cf and frows and frows[-1]["t"][11:16] >= RECONCILE_MIN_END.strftime("%H:%M"):
        lf = fut_series(frows[-1:])
        cfs = fut_series([cf])
        if not lf.empty and not cfs.empty and lf["expiry"].iloc[0] == cfs["expiry"].iloc[0]:
            out.append({"day": day.isoformat(), "code": code, "e": "fut",
                        "live_doi": lf["doi"].iloc[0], "close_doi": cfs["doi"].iloc[0],
                        "live_t": frows[-1]["t"], "close_t": cf["t"]})
    return out


def reconcile(days_back: int = 10, today: dt.date | None = None) -> int:
    """Score every finished, not yet scored day that has both a log and a close snapshot."""
    import re
    today = today or _now().date()
    p = DATA_DIR / RECONCILE_FILE
    done = {(r.get("day"), r.get("code")) for r in im._load_jsonl(p)}
    n = 0
    for f in DATA_DIR.glob("close_*_????????.json"):
        m = re.match(r"close_([A-Z]+)_(\d{8})\.json$", f.name)
        if not m:
            continue
        code, day = m.group(1), dt.datetime.strptime(m.group(2), "%Y%m%d").date()
        if day >= today and _now().time() < dt.time(16, 0):
            continue
        if (today - day).days > days_back or (day.isoformat(), code) in done:
            continue
        recs = reconcile_day(code, day)
        for r in recs or [{"day": day.isoformat(), "code": code, "e": None}]:
            im._append(p, r)
        n += 1
    return n


def reconcile_summary(code: str, n: int = 10) -> dict | None:
    """Mean scores over the last n scored days, nearest expiry of each day."""
    rows = [r for r in im._load_jsonl(DATA_DIR / RECONCILE_FILE)
            if r.get("code") == code and r.get("e")]
    if not rows:
        return None
    days = sorted({r["day"] for r in rows})[-n:]
    opt = []
    for d in days:
        ex = sorted(r["e"] for r in rows if r["day"] == d and r["e"] != "fut")
        if ex:
            opt.append(next(r for r in rows if r["day"] == d and r["e"] == ex[0]))
    fut = [r for r in rows if r["day"] in days and r["e"] == "fut"]
    mean = lambda xs: round(float(np.mean(xs)), 1) if xs else None  # noqa: E731
    return dict(days=len(days), first=days[0], last=days[-1],
                sign=mean([r["sign"] for r in opt if r.get("sign") is not None]),
                err=mean([r["err"] for r in opt if r.get("err") is not None]),
                bias_ce=mean([r["bias_CE"] for r in opt if r.get("bias_CE") is not None]),
                bias_pe=mean([r["bias_PE"] for r in opt if r.get("bias_PE") is not None]),
                fut_sign=mean([float(np.sign(r["live_doi"]) == np.sign(r["close_doi"])) * 100
                               for r in fut if r.get("live_doi") and r.get("close_doi")]),
                fut_n=len(fut))


def poll_fut(code: str, client=None) -> bool:
    s = _st(code)
    s["fut_at"] = time.time()
    try:
        row = parse_fut((client or _client()).get(im._DERIV + quote(code, safe="")))
    except Exception as exc:                                 # noqa: BLE001
        _log_error("fut", f"{code}: {exc}")
        return False
    with _LOCK:
        s["fut"] = row
    t = dt.datetime.fromisoformat(row["t"]) if row["t"] else None
    with _APPEND:
        if t and in_session(t) and t.date() == _now().date():
            p = _path("fut", code, t.date())
            if s["fut_last_t"] is None:
                last = _last_logged(p)
                s["fut_last_t"] = last.get("t") if last else None
            if s["fut_last_t"] is None or row["t"] > s["fut_last_t"]:
                im._append(p, row)
                s["fut_last_t"] = row["t"]
        elif t and t.date() == _now().date() and CLOSE_FROM <= t.time() <= dt.time(16, 30):
            _save_close(code, "fut", row)
    _save_latest(code)
    return True


def _save_latest(code: str) -> None:
    s = _st(code)
    with _LOCK:
        out = dict(expiries=[e.isoformat() for e in s["expiries"]], chain=dict(s["chain"]),
                   fut=s["fut"], err=s["err"], written=_now().isoformat(timespec="seconds"))
    im._write_json(DATA_DIR / f"latest_{code}.json", out)


def _heartbeat() -> None:
    if time.time() - _BEAT[0] < 20:
        return
    _BEAT[0] = time.time()
    out = {"written": _now().isoformat(timespec="seconds"), "writer_pid": os.getpid(),
           "indices": {}}
    with _LOCK:
        for code, s in _S.items():
            out["indices"][code] = dict(
                expiries=[e.isoformat() for e in captured_expiries(s["expiries"],
                                                                   _now().date())],
                chain_stamps={k: v.get("t") for k, v in s["chain"].items()},
                fut_stamp=(s["fut"] or {}).get("t"), error=s["err"])
    im._write_json(DATA_DIR / "status.json", out)


def request_refresh() -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "refresh.request").write_text(str(time.time()), encoding="utf-8")
    except OSError:
        pass
    _READ.clear()


def _reset_client(why: str) -> None:
    _CLIENT[0] = im.NseClient(min_gap=0.4)
    with _LOCK:
        for s in _S.values():
            s["chain_at"].clear()
            s["fut_at"] = 0.0
    _log_error("reset", why)


def _maybe_heal() -> None:
    try:
        m = (DATA_DIR / "refresh.request").stat().st_mtime
    except OSError:
        m = 0.0
    if m > _HEAL["refresh_seen"]:
        _HEAL["refresh_seen"] = m
        if m > time.time() - 600:
            _reset_client("refresh button")
            return
    now = _now()
    if time.time() - _HEAL["at"] < 120 or not in_session(now):
        return
    # Judged on each index's NEAREST expiry only — the one polled every minute. (The first
    # version looked at every chain held in memory: an expiry viewed once and then left,
    # or yesterday's expired contract, goes 'stale' by design and reset the NSE session
    # every 2 minutes all day.) Stale = NSE's numbers have not changed for HEAL_AFTER, or
    # the last successful read is that old — a frozen feed with a ticking clock counts.
    stale = []
    with _LOCK:
        for code, s in _S.items():
            k = s.get("nearest")
            row = s["chain"].get(k) if k else None
            if not row:
                continue
            f = row.get("f")
            changed = s["changed_at"].get(k)
            read_age = (now - dt.datetime.fromisoformat(f)).total_seconds() if f else 0.0
            same_for = time.time() - changed if changed else 0.0
            if max(read_age, same_for) > HEAL_AFTER:
                stale.append(f"{code} {k}")
    if stale:
        _HEAL["at"] = time.time()
        _reset_client("chain stale or frozen: " + ", ".join(stale[:4]))


def _due(at: float, every: float) -> bool:
    return time.time() - at >= every


def _cycle() -> None:
    """One pass of the writer: expiries hourly, chains and futures on their cadence."""
    now = _now()
    live = capture_window(now)
    today = now.date()
    for code in CODES:
        s = _st(code)
        viewing, view_exp = _viewing(code)
        if not live and not viewing:
            continue
        if _due(s["exp_at"], EXPIRIES_EVERY) or s["exp_day"] != today or not s["expiries"]:
            poll_expiries(code)
        targets = captured_expiries(s["expiries"], today)
        keep = {e.isoformat() for e in s["expiries"] if e >= today}
        with _LOCK:                      # expired / no-longer-listed chains leave memory
            for k in [k for k in s["chain"] if k not in keep]:
                s["chain"].pop(k, None)
        s["nearest"] = targets[0].isoformat() if targets else None
        # A weekday holiday: NSE keeps serving the last session's chain all day. After
        # 09:20, a nearest-expiry stamp from an earlier date means shut -> every 5 minutes.
        near = s["chain"].get(s["nearest"]) if s["nearest"] else None
        shut = bool(live and near and near.get("t") and near["t"][:10] < today.isoformat()
                    and now.time() >= dt.time(9, 20))
        if (viewing and view_exp and view_exp not in targets
                and view_exp in s["expiries"] and view_exp >= today):
            targets.append(view_exp)
        for i, e in enumerate(targets):
            fast = i == 0 or (viewing and e == view_exp)
            every = ((HOLIDAY_EVERY if shut else CHAIN_EVERY if fast else OTHER_EVERY)
                     if live else IDLE_EVERY)
            if _due(s["chain_at"].get(e.isoformat(), 0.0), every):
                poll_chain(code, e)
        if _due(s["fut_at"], (HOLIDAY_EVERY if shut else FUT_EVERY) if live else IDLE_EVERY):
            poll_fut(code)


def _loop(gen: float) -> None:
    while getattr(sys, _GEN_ATTR, None) == gen:
        try:
            if _claim():
                _heartbeat()
                _maybe_heal()
                _cycle()
        except Exception as exc:                             # noqa: BLE001
            _log_error("loop", f"{type(exc).__name__}: {exc}")
        time.sleep(3.0 + random.random())


def ensure_poller() -> None:
    """Start the single background capture thread (idempotent; a hot-reload replaces it)."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return                           # never poll NSE or take the live lock from a test
    with _LOCK:
        t = getattr(sys, "_eqbtst_index_fno_thread", None)
        if t is not None and t.is_alive() and getattr(t, "_owner", None) is _S:
            return
        try:
            _HEAL["refresh_seen"] = (DATA_DIR / "refresh.request").stat().st_mtime
        except OSError:
            pass
        gen = time.time()
        setattr(sys, _GEN_ATTR, gen)
        t = threading.Thread(target=_loop, args=(gen,), daemon=True, name="eqbtst-index-fno")
        t._owner = _S
        sys._eqbtst_index_fno_thread = t
        t.start()


def stop_poller() -> None:
    setattr(sys, _GEN_ATTR, None)
    time.sleep(0.1)
    _release()


VIEW_WRITE_EVERY = 20.0


def touch(code: str, expiry: dt.date | None = None) -> None:
    """Mark an index (and expiry) as on screen. Also written to view_{code}.json: the writer
    may be ANOTHER process (the scheduled capture, or the dashboard while a second one
    reads), and only a file tells it what someone is looking at -- without it an off-hours
    page, or an expiry outside the captured three, was never fetched."""
    s = _st(code)
    s["viewed"] = time.time()
    if expiry is not None:
        s["view_exp"] = expiry
    if (time.time() - s.get("view_written", 0.0) >= VIEW_WRITE_EVERY
            or s.get("view_written_exp") != s["view_exp"]):
        s["view_written"], s["view_written_exp"] = time.time(), s["view_exp"]
        im._write_json(DATA_DIR / f"view_{code}.json",
                       {"at": time.time(), "exp": im._iso(s["view_exp"])})


def _viewing(code: str) -> tuple:
    """(on screen in ANY process within VIEW_TTL, the expiry on screen or None)."""
    s = _st(code)
    at, exp = s["viewed"], s["view_exp"]
    try:
        j = json.loads((DATA_DIR / f"view_{code}.json").read_text(encoding="utf-8"))
        if float(j.get("at", 0)) > at:
            at = float(j["at"])
            exp = dt.date.fromisoformat(j["exp"]) if j.get("exp") else exp
    except (OSError, ValueError, TypeError):
        pass
    return time.time() - at < VIEW_TTL, exp


# ── reading (any process; the page reads files, so writer and reader share one path) ──────
_READ: dict = {}      # path -> (mtime, size, rows)


def _rows(path: Path) -> list:
    """jsonl rows with an mtime/size cache: the page redraws every 15 s and today's file
    only grows, so an unchanged file is not parsed again."""
    try:
        st_ = path.stat() if path.exists() else path.with_suffix(".jsonl.gz").stat()
        sig = (st_.st_mtime, st_.st_size)
    except OSError:
        return []
    hit = _READ.get(path)
    if hit and hit[0] == sig:
        return hit[1]
    rows = im._load_jsonl(path)
    _READ[path] = (sig, rows)
    return rows


def load_latest(code: str) -> dict:
    try:
        return json.loads((DATA_DIR / f"latest_{code}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def chain_log(code: str, day: dt.date, expiry: str) -> list:
    """One expiry's readings, oldest first, one row per NSE stamp (the last one wins). Two
    writers can overlap for a few seconds when a stale lock is taken over; this keeps a
    doubled row from doubling a block or a chart step."""
    by_t = {}
    for r in _rows(_path("chain", code, day)):
        if r.get("e") == expiry and r.get("t"):
            by_t[r["t"]] = r
    return [by_t[t] for t in sorted(by_t)]


def fut_log(code: str, day: dt.date) -> list:
    return _rows(_path("fut", code, day))


def logged_days(code: str) -> list:
    import re
    out = set()
    for f in DATA_DIR.glob(f"chain_{code}_*.jsonl*"):
        m = re.search(r"_(\d{8})\.jsonl", f.name)
        if m:
            out.add(dt.datetime.strptime(m.group(1), "%Y%m%d").date())
    return sorted(out, reverse=True)


def logged_expiries(code: str, day: dt.date) -> list:
    return sorted({r.get("e") for r in _rows(_path("chain", code, day)) if r.get("e")})


def compress_old_logs(today: dt.date | None = None) -> int:
    """gzip finished days (verified read-back before the plain file goes)."""
    import gzip
    import re
    today = today or _now().date()
    n = 0
    for f in DATA_DIR.glob("*_????????.jsonl"):
        m = re.search(r"_(\d{8})\.jsonl$", f.name)
        if not m or dt.datetime.strptime(m.group(1), "%Y%m%d").date() >= today:
            continue
        gz = f.with_suffix(".jsonl.gz")
        try:
            raw = f.read_bytes()
            with gzip.open(gz, "wb") as out:
                out.write(raw)
            with gzip.open(gz, "rb") as back:
                if back.read() != raw:
                    gz.unlink()
                    continue
            f.unlink()
            n += 1
        except OSError:
            continue
    return n


# ── analytics ─────────────────────────────────────────────────────────────────────────
def frame(rows: list) -> pd.DataFrame:
    """Log rows -> long frame: t, u (spot), a (row ATM), K, side, *FIELDS."""
    recs = []
    for r in rows:
        try:
            t = pd.Timestamp(r["t"])
        except Exception:                                    # noqa: BLE001
            continue
        u, a = r.get("u"), r.get("a")
        mv = {float(x[0]): (x[1], x[2]) for x in r.get("v") or [] if len(x) >= 3}
        has = "v" in r
        for k in r.get("k") or []:
            if len(k) < 1 + 2 * NF:
                continue
            K = float(k[0])
            for j, (side, off) in enumerate((("CE", 1), ("PE", 1 + NF))):
                v = k[off:off + NF]
                if v[2] is None:
                    continue
                ivm = mv.get(K, (None, None))[j] if has else v[5]
                recs.append((t, u, a, K, side, *v, ivm))
    df = pd.DataFrame(recs, columns=["t", "u", "a", "K", "side", *FIELDS, "IV"])
    for c in (*FIELDS, "IV"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def row_sides(row: dict) -> tuple:
    """A compact log row -> ({K: dict(field=value)} for CE, same for PE). Each side also gets
    'miv' (our parity-forward IV) when the row carries it, and 'IV' = the IV every reader
    should use: ours if this row has the model at all, else NSE's (rows logged before the
    model existed). Never a mix within one row, so a change in IV compares like with like
    when both rows share the source (see iv_source)."""
    ce, pe = {}, {}
    mv = {float(x[0]): (x[1], x[2]) for x in row.get("v") or [] if len(x) >= 3}
    has = "v" in row
    for k in row.get("k") or []:
        if len(k) < 1 + 2 * NF:
            continue
        a, b = k[1:1 + NF], k[1 + NF:1 + 2 * NF]
        K = float(k[0])
        m = mv.get(K, (None, None))
        if a[2] is not None:
            ce[K] = dict(zip(FIELDS, a), miv=m[0])
            ce[K]["IV"] = m[0] if has else ce[K]["iv"]
        if b[2] is not None:
            pe[K] = dict(zip(FIELDS, b), miv=m[1])
            pe[K]["IV"] = m[1] if has else pe[K]["iv"]
    return ce, pe


def iv_source(row: dict | None) -> str:
    return "model" if row and "v" in row else "nse"


BASE_SCAN = 150          # readings scanned for each strike's first valid IV (~75-150 min)


def baseline_iv(rows: list, strikes=None, step: float | None = None) -> dict:
    """{"CE": {K: iv}, "PE": {K: iv}, "atm": iv} — each strike's FIRST reading of the day that
    has an IV, not simply the day's first reading. At a gap open the first readings carry no
    IV on illiquid strikes (no trade yet: 5 Nov 2026, Fin Nifty ATM untraded until 09:28), so
    'change since the first reading' would be blank all day. Model and NSE-sourced rows are
    never mixed: only rows of the same source as the LAST row count."""
    out = {"CE": {}, "PE": {}, "atm": None}
    if not rows:
        return out
    src = iv_source(rows[-1])
    want = set(strikes) if strikes is not None else None
    for r in rows[:BASE_SCAN]:
        if iv_source(r) != src:
            continue
        ce, pe = row_sides(r)
        for side, d in (("CE", ce), ("PE", pe)):
            for K, x in d.items():
                if x.get("IV") and K not in out[side] and (want is None or K in want):
                    out[side][K] = x["IV"]
        if out["atm"] is None:
            a = grid_atm(r, step)
            ivs = [x for x in ((ce.get(a) or {}).get("IV"), (pe.get(a) or {}).get("IV")) if x]
            if ivs:
                out["atm"] = float(np.mean(ivs))
        if want is not None and out["atm"] is not None and all(
                K in out["CE"] and K in out["PE"] for K in want):
            break
    return out


def premium_read(side: str, chg, doi) -> str | None:
    """What most sites print: premium change x OI change (vs yesterday's close)."""
    if chg is None or doi is None or doi == 0 or chg == 0:
        return None
    if doi > 0:
        q = "long build" if chg > 0 else "writing"
    else:
        q = "short covering" if chg > 0 else "long unwind"
    return q


_BULL = {"CE": ("long build", "short covering", "buyers adding", "writers covering"),
         "PE": ("writing", "long unwind", "writers adding", "buyers exiting")}


def iv_read(d_oi, d_iv, oi_now, band: float = IV_BAND, floor: float = OI_FLOOR) -> str | None:
    """OI change vs NSE's IV change, both since the first reading today. More contracts AND
    richer volatility = someone paying up (buyers); more contracts at cheaper volatility =
    someone supplying (writers). A move inside the band is not called."""
    if d_oi is None or d_iv is None or not oi_now or np.isnan(d_oi) or np.isnan(d_iv):
        return None
    if abs(d_oi) < floor * max(oi_now, 1.0):
        return "little change"
    if abs(d_iv) < band:
        return "unclear"
    if d_oi > 0:
        return "buyers adding" if d_iv > 0 else "writers adding"
    return "writers covering" if d_iv > 0 else "buyers exiting"


def lean(side: str, word: str | None) -> str:
    if not word or word in ("little change", "unclear"):
        return "⚪"
    if side == "CE":
        return "🟢" if word in _BULL["CE"] else "🔴"
    return "🟢" if word in _BULL["PE"] else "🔴"


def ladder(latest: dict, first: dict | None, strikes: list, mode: str = "iv",
           atm=None, base: dict | None = None) -> pd.DataFrame:
    """The 17-strike table: calls left, strike, puts right. Numeric columns are float (a
    missing value is NaN, never None — None printed as the word 'None' on the page).
    `base` = baseline_iv(): each strike's first valid IV; without it, the first row's."""
    ce, pe = row_sides(latest)
    ce0, pe0 = row_sides(first) if first else ({}, {})
    same_src = first is not None and iv_source(first) == iv_source(latest)
    ivb = base
    if ivb is None and same_src:
        ivb = {"CE": {k: v["IV"] for k, v in ce0.items() if v.get("IV")},
               "PE": {k: v["IV"] for k, v in pe0.items() if v.get("IV")}}
    ivb = ivb or {"CE": {}, "PE": {}}
    atm = latest.get("a") if atm is None else atm
    out = []
    for K in strikes:
        rec = {"Strike": K, "atm": K == atm}
        for side, cur, base in (("CE", ce, ce0), ("PE", pe, pe0)):
            d = cur.get(K) or {}
            b = base.get(K) or {}
            ltp, chg = d.get("ltp"), d.get("chg")
            prev = (ltp - chg) if ltp is not None and chg is not None else None
            b_iv = ivb[side].get(K)
            div = (d["IV"] - b_iv) if d.get("IV") and b_iv else None
            d_oi = (d["oi"] - b["oi"]) if d.get("oi") is not None and b.get("oi") is not None \
                else None
            if mode == "iv":
                w = iv_read(d_oi, div, d.get("oi"))
            else:                    # no trade today = the premium change is not today's
                w = premium_read(side, chg, d.get("doi")) if (d.get("vol") or 0) > 0 else None
            rec.update({
                f"{side} read": (f"{lean(side, w)} {w}" if w else "—"),
                f"{side} ΔOI": d.get("doi"), f"{side} OI": d.get("oi"),
                f"{side} ΔOI since first": d_oi,
                f"{side} Volume": d.get("vol"), f"{side} IV": d.get("IV"),
                f"{side} ΔIV": div, f"{side} LTP": ltp,
                f"{side} Chg %": (chg / prev * 100) if prev and prev > 0 else None})
        out.append(rec)
    df = pd.DataFrame(out)
    for c in df.columns:
        if c not in ("Strike", "atm") and not c.endswith("read"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


def walls(latest: dict, strikes: list | None = None) -> dict:
    ce, pe = row_sides(latest)
    if strikes is not None:
        ce = {k: v for k, v in ce.items() if k in strikes}
        pe = {k: v for k, v in pe.items() if k in strikes}
    top = lambda d, f: max(d, key=lambda k: d[k].get(f) or 0) if d else None  # noqa: E731
    return dict(ce_oi=top(ce, "oi"), pe_oi=top(pe, "oi"),
                ce_add=top(ce, "doi"), pe_add=top(pe, "doi"))


def grid_atm(row: dict | None, step: float | None):
    """ATM on the chosen strike grid: the grid strike nearest the index, from strikes that
    have both a call and a put in this row. step None = NSE's listed ATM (row['a'])."""
    if not row:
        return None
    if not step:
        return row.get("a")
    ce, pe = row_sides(row)
    return pick_atm(grid(set(ce) & set(pe), step), row.get("u"))


def headline(latest: dict, first: dict | None, strikes: list, step: float | None = None) -> dict:
    """Header numbers for the selected expiry (ATM on the chosen strike grid)."""
    ce, pe = row_sides(latest)
    atm, u = grid_atm(latest, step), latest.get("u")
    c, p = ce.get(atm) or {}, pe.get(atm) or {}
    cp, pp = _px(c), _px(p)
    straddle = (cp + pp) if cp and pp else None
    # the model's forward (median over 7 strikes) when the row has it, else this strike's
    fwd = latest.get("F") or ((atm + cp - pp) if straddle else None)
    ivs = [x for x in (c.get("IV"), p.get("IV")) if x]
    atm_iv = float(np.mean(ivs)) if ivs else None
    atm_iv0 = None
    if first and iv_source(first) == iv_source(latest):
        c0, p0 = row_sides(first)
        a0 = grid_atm(first, step)
        iv0 = [x for x in ((c0.get(a0) or {}).get("IV"), (p0.get(a0) or {}).get("IV")) if x]
        atm_iv0 = float(np.mean(iv0)) if iv0 else None
    band_ce = sum((ce.get(k) or {}).get("oi") or 0 for k in strikes)
    band_pe = sum((pe.get(k) or {}).get("oi") or 0 for k in strikes)
    tc = latest.get("tc") or [None] * 4
    ks = sorted(strikes)
    skew = None
    if len(ks) >= 9 and atm in ks:
        i = ks.index(atm)
        lo, hi = ks[max(0, i - 4)], ks[min(len(ks) - 1, i + 4)]
        pv, cv = (pe.get(lo) or {}).get("IV"), (ce.get(hi) or {}).get("IV")
        skew = (pv - cv) if pv and cv else None
    return dict(
        spot=u, atm=atm, straddle=straddle, move_pct=(straddle / u * 100) if straddle and u else None,
        fwd=fwd, atm_iv=atm_iv, atm_iv0=atm_iv0, skew=skew,
        pcr_band=(band_pe / band_ce) if band_ce else None,
        pcr_all=(tc[2] / tc[0]) if tc[0] and tc[2] is not None else None,
        mp=latest.get("mp"), **walls(latest, strikes))


SERIES_QUORUM = 0.9      # share of the window's strikes a reading needs to be plotted


def session_series(df: pd.DataFrame, rows: list, strikes: list, lot: int | None,
                   base: str = "yday", step: float | None = None) -> pd.DataFrame:
    """Per reading, over the SAME strikes all session (the window on screen), so a line moves
    only when positions move — not when the ATM drifts and different strikes enter the sum.

    base 'yday': OI change since yesterday's close (NSE's changeinOpenInterest).
    base 'first': OI change since today's first reading of that strike.
    ₹ Cr = contracts x lot x spot at that reading / 1e7 (exposure, not premium)."""
    if df.empty:
        return pd.DataFrame()
    w = df[df["K"].isin(strikes)].copy()
    if base == "first":
        first_oi = w.sort_values("t").groupby(["side", "K"])["oi"].transform("first")
        w["chg_"] = w["oi"] - first_oi
    else:
        w["chg_"] = w["doi"]
    need = SERIES_QUORUM * len(strikes)
    g = w.groupby(["t", "side"])
    n = g["K"].nunique().unstack()
    s = g["chg_"].sum().unstack()
    oi = g["oi"].sum().unstack()
    for c in ("CE", "PE"):
        for x in (n, s, oi):
            if c not in x:
                x[c] = np.nan
    ok = (n["CE"] >= need) & (n["PE"] >= need)
    spot = df.groupby("t")["u"].first()
    # ONE index level for the whole session (the first reading's): with the live level a
    # 1% index move alone moved every ₹ line 1% with no position changing
    s0 = spot.dropna().iloc[0] if spot.notna().any() else None
    mult = (lot * s0 / 1e7) if lot and s0 else 1.0
    out = pd.DataFrame({
        "spot": spot,
        "calls": (s["CE"] * mult).where(ok), "puts": (s["PE"] * mult).where(ok),
        "pcr_band": (oi["PE"] / oi["CE"]).where(ok)})
    out["calls − puts"] = out["calls"] - out["puts"]
    meta = {pd.Timestamp(r["t"]): r for r in rows if r.get("t")}
    tc = pd.Series({t: (r.get("tc") or [None] * 4) for t, r in meta.items()}, dtype=object)
    out["pcr_all"] = [((tc[t][2] / tc[t][0]) if t in tc and tc[t][0] and tc[t][2] is not None
                       else np.nan) for t in out.index]
    out["max pain"] = [meta[t].get("mp") if t in meta else np.nan for t in out.index]
    # ATM straddle and ATM IV at EACH reading's own ATM (what a desk calls "the ATM"), on the
    # chosen strike grid
    if step:
        G = np.array(sorted({k for k in df["K"].unique() if on_grid(k, step)}), float)
        if len(G):
            uu = df["u"].to_numpy(float)
            idx = np.abs(G[None, :] - uu[:, None]).argmin(axis=1)
            ga = G[idx]
            a = df[df["K"].to_numpy(float) == ga]
        else:
            a = df[df["K"] == df["a"]]
    else:
        a = df[df["K"] == df["a"]]
    a = a.assign(px=[(bb + aa) / 2 if (bb and aa and aa >= bb and
                                        (aa - bb) <= max(0.10, MID_MAX_SPREAD * (aa + bb) / 2))
                     else l for bb, aa, l in zip(a["bid"].fillna(0), a["ask"].fillna(0),
                                                  a["ltp"])])
    piv = a.pivot_table(index="t", columns="side", values=["px", "IV"], aggfunc="first")
    if ("px", "CE") in piv and ("px", "PE") in piv:
        out["straddle"] = piv[("px", "CE")] + piv[("px", "PE")]
    else:
        out["straddle"] = np.nan
    ivc = [piv[("IV", c)] for c in ("CE", "PE") if ("IV", c) in piv]
    out["atm iv"] = pd.concat(ivc, axis=1).mean(axis=1) if ivc else np.nan
    return out.sort_index()


# ── strike buckets: Best / Better / Good (user, 7 Nov 2026) ──────────────────────────────
# The Index Management idea (Top 10 / Next 10 / Rest 30) applied to strikes, by DISTANCE FROM
# THE MONEY on the ladder on screen: of the 16 strikes around ATM, the nearest 3 either side
# are Best (with ATM itself), the next 2 either side Better, the outer 3 either side Good.
# Steps are counted on the chosen strike gap (100 pts by default), so Best = ATM +-300 on
# Nifty. Wider ladders put everything past step 5 in Good.
BUCKETS = (("Best", 0, 3), ("Better", 4, 5), ("Good", 6, 10 ** 6))
BUCKET_NAMES = tuple(b[0] for b in BUCKETS)
BUCKET_COLOR = {"Best": "#f5b041", "Better": "#3a8dde", "Good": "#9aa4b2"}
BUCKET_LEAN_BAND = 0.10  # calls vs puts added within 10% of each other = "balanced"


AUTO_GAPS = (100, 200)   # round grids the 'auto' gap chooses from (200 x 8 stays inside the log)
AUTO_SCAN = 30           # readings scanned from the open for a computable straddle


def straddle_move(row: dict | None) -> float | None:
    """ATM straddle on the 100-pt grid as % of the index, or None when either side has not
    traded today (14% of Fin Nifty mornings at 09:20 in the archive)."""
    if not row or not row.get("u"):
        return None
    ce, pe = row_sides(row)
    a = pick_atm(grid(set(ce) & set(pe), 100), row["u"])
    c, p = _px(ce.get(a)), _px(pe.get(a))
    return (c + p) / row["u"] * 100 if c and p else None


def auto_gap(rows: list, latest: dict | None = None) -> tuple:
    """(gap, expected move %, why). 100 points — where Nifty / Fin / Midcap OI sits — unless 8
    strikes of 100 do not reach the move the ATM straddle prices; then 200.

    Measured on 87 sessions (Tradebot archive, 09:20): with 100 points the whole 17-strike
    ladder sat INSIDE the priced move on Bank Nifty from 3 sessions to expiry (Good reached
    0.5-1.0x the straddle's move, Best 0.2-0.4x), so its 'Good' bucket was not the wings. The
    rule keeps 100 for Nifty, Fin and Midcap at every expiry and moves Bank Nifty to 200 only
    there. Decided from the session's FIRST computable straddle, so the buckets never change
    mid-session; 100 when none is computable."""
    for r in (rows or [])[:AUTO_SCAN] or ([latest] if latest else []):
        em = straddle_move(r)
        if em:
            u = r["u"]
            for g in AUTO_GAPS:
                if BAND * g / u * 100 >= em:
                    return g, em, f"8 × {g} pts reaches the ±{em:.2f}% the straddle prices"
            g = AUTO_GAPS[-1]
            return g, em, f"widest allowed; the straddle prices ±{em:.2f}%"
    return AUTO_GAPS[0], None, "no straddle computable yet (ATM untraded) — 100 by default"


def bucket_map(strikes: list, center) -> dict:
    """{K: bucket} by steps from `center` (the ladder's middle strike). Membership is fixed for
    the session by the ladder, like the strike lines: a wall moving from Good to Best as the
    index drifts would otherwise move a bucket's line with no position changing."""
    ks = sorted(strikes)
    if not ks:
        return {}
    c = center if center in ks else pick_atm(ks, center if center is not None else ks[len(ks) // 2])
    i = ks.index(c)
    out = {}
    for j, K in enumerate(ks):
        d = abs(j - i)
        out[K] = next(name for name, lo, hi in BUCKETS if lo <= d <= hi)
    return out


def _lean(calls_add, puts_add) -> str:
    """Which side added more contracts today. Words, not arrows: index option writers
    dominate, so 'put side heavier' is usually read as support, but Tradebot's tests found
    no edge in any of it (walls follow price; crossovers mildly contrarian)."""
    if calls_add is None or puts_add is None or (not calls_add and not puts_add):
        return "—"
    big = max(abs(calls_add), abs(puts_add))
    if big and abs(calls_add - puts_add) <= BUCKET_LEAN_BAND * big:
        return "⚪ balanced"
    return "🟢 put side heavier" if puts_add > calls_add else "🔴 call side heavier"


def _ranges(ks: list, center) -> str:
    """'24,000–24,100 · 24,900–25,000 (4)': a bucket away from the money is two runs of
    strikes, one below and one above it; one 'low–high' would claim the strikes between."""
    lo = [k for k in ks if center is not None and k < center]
    hi = [k for k in ks if center is None or k >= center]
    part = lambda xs: (f"{xs[0]:,.0f}" if len(xs) == 1 else f"{xs[0]:,.0f}–{xs[-1]:,.0f}")  # noqa: E731
    runs = [part(x) for x in (lo, hi) if x]
    if len(runs) == 2 and lo[-1] < center <= hi[0] and (center in ks):
        runs = [part(sorted(ks))]                  # one unbroken run through the money
    return " · ".join(runs) + f" ({len(ks)})"


def spot_vs_buckets(spot, bmap: dict) -> dict:
    """{bucket: 'where the index is from this bucket'} for the table's index column.

    The buckets are fixed at the 09:15 open; the index is not. Exactly one bucket is 'here' —
    the one holding the ladder strike NEAREST the index (so the half-way point between two
    buckets is never claimed by both). The others say how far the index must travel to reach
    their nearest strike. Past the outermost strike by more than half a step, no bucket holds
    it and Good says so: the morning ladder no longer covers the money."""
    ks = sorted(bmap)
    if not ks or spot is None or not np.isfinite(spot):
        return {b: "—" for b in set(bmap.values())}
    steps = np.diff(ks)
    half = (float(steps[steps > 0].min()) if (steps > 0).any() else 0.0) / 2
    near = min(ks, key=lambda k: (abs(k - spot), k))
    off = spot < ks[0] - half or spot > ks[-1] + half
    here = None if off else bmap[near]
    out = {}
    for name in BUCKET_NAMES:
        mine = [k for k in ks if bmap[k] == name]
        if not mine:
            continue
        if name == here:
            out[name] = "◀ index here"
            continue
        k = min(mine, key=lambda x: (abs(x - spot), x))
        d = k - spot
        txt = f"{abs(d):,.0f} pts {'up' if d > 0 else 'down'} to {k:,.0f} ({abs(d) / spot * 100:.1f}%)"
        if off and name == "Good":
            txt = (f"⚠ index past the ladder — {abs(d):,.0f} pts "
                   f"{'above' if d < 0 else 'below'} {k:,.0f}")
        out[name] = txt
    return out


def bucket_table(latest: dict, first: dict | None, bmap: dict, lot: int | None) -> pd.DataFrame:
    """One row per bucket: OI, change today and since the first reading, PCR, volume, ₹."""
    ce, pe = row_sides(latest)
    best = sorted(k for k, b in bmap.items() if b == "Best")
    center_k = best[len(best) // 2] if best else None
    ce0, pe0 = row_sides(first) if first else ({}, {})
    # the FIRST reading's index level, as bucket_series / session_series use: with the live
    # level the table's ₹ and the line's last point disagreed by the day's index move
    # (5 Nov Nifty Best -78,435 vs -77,744 Cr)
    u = (first or {}).get("u") or latest.get("u")
    where = spot_vs_buckets(latest.get("u"), bmap)
    now_u = latest.get("u")
    rows = []
    for name in BUCKET_NAMES:
        ks = sorted(k for k, b in bmap.items() if b == name)
        if not ks:
            continue
        def tot(d, f, keys=ks):
            vals = [(d.get(k) or {}).get(f) for k in keys]
            vals = [v for v in vals if v is not None]
            return float(sum(vals)) if vals else None
        coi, poi = tot(ce, "oi"), tot(pe, "oi")
        cd, pd_ = tot(ce, "doi"), tot(pe, "doi")
        c0, p0 = tot(ce0, "oi"), tot(pe0, "oi")
        net_cr = ((cd - pd_) * lot * u / 1e7) if lot and u and cd is not None and pd_ is not None \
            else None
        rows.append({
            "Bucket": name, "Strikes": _ranges(ks, center_k),
            "Index now": (f"{now_u:,.2f} · {where.get(name, '—')}" if now_u else "—"),
            "Call OI": coi, "Put OI": poi,
            "PCR": (poi / coi) if coi else None,
            "PCR yday": ((poi - pd_) / (coi - cd)) if coi and cd is not None and pd_ is not None
            and coi - cd else None,
            "Call ΔOI": cd, "Put ΔOI": pd_,
            "Call ΔOI since first": (coi - c0) if first and coi is not None and c0 is not None else None,
            "Put ΔOI since first": (poi - p0) if first and poi is not None and p0 is not None else None,
            "Call volume": tot(ce, "vol"), "Put volume": tot(pe, "vol"),
            "Calls − puts ₹ Cr": net_cr,
            "Today": _lean(cd, pd_)})
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c not in ("Bucket", "Strikes", "Index now", "Today"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


# ── bucket drill-down: what each strike's calls and puts are doing ─────────────────────
_BULL_STATES = {"CE": ("Long build-up", "Short covering"),
                "PE": ("Short build-up", "Long unwinding")}
STATE_NAME = {"long build": "Long build-up", "writing": "Short build-up",
              "short covering": "Short covering", "long unwind": "Long unwinding"}
_IV_WORD = {"buyers adding": "long build", "writers adding": "writing",
            "writers covering": "short covering", "buyers exiting": "long unwind"}
SHORT_WORD = {"Buyers in": "B+", "Writers in": "W+", "Writers out": "W−", "Buyers out": "B−",
              "·": "·", "?": "?"}
PLAIN_STATE = {"Long build-up": "Buyers in", "Short build-up": "Writers in",
               "Short covering": "Writers out", "Long unwinding": "Buyers out",
               "Quiet": "·", "Unclear": "?"}
STATE_CODE = {"Long build-up": "LB", "Short build-up": "SB", "Short covering": "SC",
              "Long unwinding": "LU", "Quiet": "·", "Unclear": "?"}
BLOCK_FLOOR = 0.005      # in a block, an OI change under 0.5% of the strike's OI = quiet
BLOCK_IV_BAND = 0.25     # in a block, a skew-adjusted IV move under this = unclear (vol points)
DRILL_BY = ("premium", "IV")


def iv_pressure(df: pd.DataFrame) -> pd.Series:
    """Per log row: the strike's IV change since its previous reading, with the part the index
    move explains taken out (vol points). Index -> {(t, K, side): value}.

    A fixed strike's IV moves when the index moves even if nobody trades it: the strike slides
    along the smile (index down -> the strike is now relatively higher -> lower IV for a
    downward-sloping smile). Sticky-moneyness: dIV_expected = -slope * dS, slope = dIV/dK of
    the same reading's smile (mean of call and put IV per strike, central differences across
    every logged strike). What is left is pressure: buyers paying up (+) or writers selling
    vol (-). MEASURED (Tradebot archive Jun-Nov 2026, non-expiry days, Best bucket, 15-min
    blocks): it REVERSES next block (rank -0.09 Nifty .. -0.27 Midcap, t -5.7 .. -25), so a
    single block's value is mostly last-trade bounce, not flow. The 9 Nov 'persists on Nifty'
    reading was an expiry-day artefact (whole surface drifting up) and is retracted."""
    if df.empty or "IV" not in df:
        return pd.Series(dtype=float)
    sm = df.groupby(["t", "K"])["IV"].mean().dropna().reset_index().sort_values(["t", "K"])
    parts = []
    for t, g in sm.groupby("t"):
        if len(g) >= 3:
            parts.append(pd.DataFrame({"t": t, "K": g["K"].values,
                                       "slope": np.gradient(g["IV"].values, g["K"].values)}))
    w = df[["t", "K", "side", "IV", "u"]].copy()
    w = w.merge(pd.concat(parts) if parts else pd.DataFrame(columns=["t", "K", "slope"]),
                on=["t", "K"], how="left").sort_values(["side", "K", "t"])
    grp = w.groupby(["side", "K"])
    w["p"] = grp["IV"].diff() + grp["slope"].shift() * grp["u"].diff()
    return w.set_index(["t", "K", "side"])["p"]


def build_state(side: str, d_oi, d_x, oi_now, by: str = "premium",
                floor: float = OI_FLOOR, band: float = IV_BAND) -> tuple:
    """(state, lean) for one strike and side over one stretch of time.

    by='premium' - the sites' reading: OI change x premium change.
        OI up + premium up = Long build-up      OI up + premium down = Short build-up
        OI down + premium up = Short covering   OI down + premium down = Long unwinding
    by='IV' - the same four, with the option's IV change in place of its premium. A
        premium mostly moves with the index (delta) and decays with time (theta, steepest on
        expiry day), so the premium reading largely restates where the index went; IV is what
        is left once both are taken out — paid up = buyers, cheapened = writers.
    lean: +1 bullish for the index, -1 bearish, 0 none. Calls: long build-up / short
    covering = bullish. Puts: the mirror (put writing = bullish, put buying = bearish)."""
    vals = (d_oi, d_x, oi_now)
    if any(v is None for v in vals) or any(isinstance(v, float) and np.isnan(v) for v in vals):
        return None, 0
    if not oi_now or abs(d_oi) < floor * max(float(oi_now), 1.0):
        return "Quiet", 0
    if (by == "IV" and abs(d_x) < band) or d_x == 0:
        return "Unclear", 0
    key = (("long build" if d_x > 0 else "writing") if d_oi > 0
           else ("short covering" if d_x > 0 else "long unwind"))
    return STATE_NAME[key], (1 if key in _BULL[side] else -1)


def strike_states(df: pd.DataFrame, strikes: list, minutes: int = 15,
                  by: str = "premium") -> pd.DataFrame:
    """Per strike, side and block: OI change, premium change, IV change and the state.

    Each block runs from the previous block's last reading to its own last reading. The
    first block runs from YESTERDAY's close for OI (OI - change) and premium (last - change),
    so the opening rush is in it; IV has no close of yesterday, so its first block starts at
    the first reading. Blocks of OI change sum exactly to the day's change."""
    cols = ["block", "t", "K", "side", "oi", "d_oi", "d_px", "d_iv", "vol", "fresh",
            "state", "lean"]
    w = df[df["K"].isin(strikes)].dropna(subset=["oi"]).sort_values("t")
    if w.empty:
        return pd.DataFrame(columns=cols)
    # the smile fit runs per reading: only when IV is the judge (premium mode stays cheap)
    press = iv_pressure(df) if by == "IV" else pd.Series(dtype=float)
    first_blk = None
    out = []
    for (side, K), g in w.groupby(["side", "K"]):
        g = g.reset_index(drop=True)
        g["block"] = block_start(g["t"], minutes).values
        g["ivp"] = press.reindex(list(zip(g["t"], g["K"], g["side"]))).values \
            if len(press) else np.nan
        dv = g["vol"].diff()
        dv[dv < 0] = np.nan
        if first_blk is None:
            first_blk = g["t"].iloc[0].normalize() + pd.Timedelta(
                hours=SESSION_OPEN.hour, minutes=SESSION_OPEN.minute)
        # NSE volume is cumulative from 09:15: the first reading's count belongs to the
        # first block only when that reading is inside the session's first block
        dv.iloc[0] = g["vol"].iloc[0] if g["block"].iloc[0] == first_blk else np.nan
        g["dv"] = dv
        bvol = g.groupby("block")["dv"].sum(min_count=1)
        bivp = g.groupby("block")["ivp"].sum(min_count=1)
        last = g.groupby("block").tail(1).set_index("block")
        oi0 = g["oi"].iloc[0] - (g["doi"].iloc[0] if pd.notna(g["doi"].iloc[0]) else 0.0)
        c0 = g["chg"].iloc[0]
        px0 = g["ltp"].iloc[0] - c0 if pd.notna(c0) and pd.notna(g["ltp"].iloc[0]) else np.nan
        d_oi = last["oi"].diff()
        d_oi.iloc[0] = last["oi"].iloc[0] - oi0
        d_px = last["ltp"].diff()
        d_px.iloc[0] = last["ltp"].iloc[0] - px0
        for b in last.index:
            div = bivp.get(b, np.nan)                      # skew-adjusted IV change
            dx = div if by == "IV" else d_px[b]
            stt, ln = build_state(side, d_oi[b], dx, last["oi"][b], by, BLOCK_FLOOR,
                                  BLOCK_IV_BAND)
            v = bvol.get(b, np.nan)
            fr = (d_oi[b] / v) if pd.notna(v) and v > 0 and pd.notna(d_oi[b]) else np.nan
            out.append((b, last["t"][b], K, side, last["oi"][b], d_oi[b], d_px[b], div,
                        v, fr, stt, ln))
    return pd.DataFrame(out, columns=cols)


def bull_bear(states: pd.DataFrame) -> pd.DataFrame:
    """Per block: contracts on the bullish side minus contracts on the bearish side, each
    strike-side weighted by the size of its OI change. A Quiet / Unclear cell counts zero."""
    if states.empty:
        return pd.DataFrame(columns=["block", "bull", "bear", "net"])
    s = states.assign(w=states["d_oi"].abs().fillna(0.0))
    g = s.groupby("block").apply(
        lambda x: pd.Series({"bull": float(x.loc[x["lean"] > 0, "w"].sum()),
                             "bear": float(x.loc[x["lean"] < 0, "w"].sum())}),
        include_groups=False).reset_index()
    g["net"] = g["bull"] - g["bear"]
    return g


WHO_COLORS = {"Buyers in": "#2e86de", "Buyers out": "#9cc8f0",
              "Writers in": "#e67e22", "Writers out": "#f5c89a"}


def day_flow(states: pd.DataFrame) -> pd.DataFrame:
    """Per strike, side and who: contracts (absolute OI change) added up over the blocks.
    'signed' = + for contracts that help the index up, − for those that push it down.
    Quiet / unclear blocks are left out (nobody can be named for them)."""
    s = states[states["state"].isin(STATE_NAME.values())].copy()
    if s.empty:
        return pd.DataFrame(columns=["K", "side", "word", "v", "signed", "effect", "rank"])
    s["word"] = s["state"].map(PLAIN_STATE)
    s["v"] = s["d_oi"].abs()
    s["signed"] = s["v"] * s["lean"]
    g = s.groupby(["K", "side", "word"]).agg(v=("v", "sum"), signed=("signed", "sum")).reset_index()
    g["effect"] = np.where(g["signed"] > 0, "helps UP", "pushes DOWN")
    g["rank"] = g["word"].map({w: i for i, w in enumerate(WHO_COLORS)})
    return g


CHECK_BAND = 0.5          # vol pts: a strike's IV move vs the chain's under this = can't tell
CHECK_NOISE_X = 3.0       # ... and never under 3x the strike's own reading-to-reading wobble
CHECK_MIN_VOL = 10_000    # contracts today: thinner strikes' IV is quote noise (Fin Nifty 11 Nov
                          # 2026: 46-1,667 contracts, IV vs chain zig-zagging +0.7/+1.3/-0.75/+1.5)
CHURN_FRESH = 0.02        # |OI change| under 2% of the day's volume = churn, label is weak
CHAIN_MONEYNESS = 0.05    # strikes within ±5% of the index set the chain-wide IV move


_PREV_SESSION: dict = {}


def prev_session(code: str, day: dt.date) -> dt.date | None:
    """The last trading session before `day` per DCM's index archive (None if unknown)."""
    if (code, day) in _PREV_SESSION:
        return _PREV_SESSION[(code, day)]
    try:
        from . import data
        with data._connect() as c:
            r = c.execute("select max(trade_date) from index_data where trade_date < ? and "
                          "index_name = ?", [day, DCM_INDEX_NAME[code]]).fetchone()
        v = r[0] if r and r[0] else None
    except Exception:                                        # noqa: BLE001
        return None                                          # locked archive: ask again later
    _PREV_SESSION[(code, day)] = v
    return v


def close_base(code: str, day: dt.date, expiry: str) -> dict | None:
    """The PREVIOUS SESSION's close snapshot of this expiry (with our model IV), else None.

    Only that one session: the premium label counts from yesterday's close, so a base two
    sessions old (laptop off at yesterday's close, or yesterday's snapshot without IV) would
    compare different stretches of time. Weekends and holidays are skipped by looking back
    up to 7 days for the newest snapshot (10 Nov 2026 was shut); when DCM knows the previous
    session and the snapshot is not from it, there is no base."""
    for back in range(1, 8):
        d = day - dt.timedelta(days=back)
        c = load_close(code, d)
        if not c:
            continue
        ps = prev_session(code, day)
        if ps is not None and ps != d:
            return None
        r = (c.get("chain") or {}).get(expiry)
        return r if r and r.get("v") else None
    return None


def _smile_slope(row: dict) -> dict:
    """{K: dIV/dK} from one reading's smile (mean of call and put IV per strike)."""
    ce, pe = row_sides(row)
    iv = {}
    for K in set(ce) | set(pe):
        vals = [x for x in ((ce.get(K) or {}).get("IV"), (pe.get(K) or {}).get("IV")) if x]
        if vals:
            iv[K] = float(np.mean(vals))
    ks = sorted(iv)
    if len(ks) < 3:
        return {}
    return dict(zip(ks, np.gradient([iv[k] for k in ks], ks)))


def day_check(df: pd.DataFrame, latest: dict, base_row: dict | None) -> dict:
    """Per strike and side: the day's IV move with the index's effect taken out, and the
    same measured against the rest of the chain.

      own = IV now - IV at the base + slope x (index now - index at the base)
            (the slope term removes the strike's slide along the smile: sticky-moneyness)
      rel = own - median(own) over strikes within ±5% of the index
            (the chain-wide move — index IV usually falls in a rally and rises in a fall —
            is the market's, not this strike's: without it a +1% day stamps 'writers' on
            every strike)
    Base = yesterday's close snapshot of this expiry when there is one, else each strike's
    first reading today with an IV. Returns {"by": {(side, K): (own, rel)}, "common": x,
    "base": "close" | "first"}."""
    out = {"by": {}, "common": None, "base": None}
    u1 = latest.get("u")
    if not u1:
        return out
    # the index move that slides a strike along the smile is the move of the FORWARD the IVs
    # were solved on. Yesterday's close row carries NSE's closing index (a 15:00-15:30
    # average) next to last-trade premiums: spot said +236 since the 9 Nov close, the
    # options' forward +279 (11 Nov 2026, Nifty; Bank +1,156 vs +1,238)
    use_f = bool(base_row and base_row.get("F") and latest.get("F"))
    x1 = latest["F"] if use_f else u1
    ce1, pe1 = row_sides(latest)
    sl1 = _smile_slope(latest)
    base = {}
    if base_row and base_row.get("u"):
        ce0, pe0 = row_sides(base_row)
        sl0 = _smile_slope(base_row)
        for side, d in (("CE", ce0), ("PE", pe0)):
            for K, x in d.items():
                if x.get("IV"):
                    base[(side, K)] = (x["IV"], base_row["F"] if use_f else base_row["u"],
                                       sl0.get(K))
        out["base"] = "close"
    elif not df.empty:
        f0 = df.dropna(subset=["IV", "u"]).sort_values("t").groupby(["side", "K"]).first()
        for (side, K), r in f0.iterrows():
            base[(side, K)] = (r["IV"], r["u"], None)
        out["base"] = "first"
    own = {}
    for side, d in (("CE", ce1), ("PE", pe1)):
        for K, x in d.items():
            b = base.get((side, K))
            if not b or not x.get("IV"):
                continue
            s1, s0 = sl1.get(K), b[2]
            sl = np.nanmean([v for v in (s1, s0) if v is not None]) if (s1 is not None or
                                                                          s0 is not None) else 0.0
            own[(side, K)] = x["IV"] - b[0] + sl * (x1 - b[1])
    near = [v for (side, K), v in own.items() if abs(K / u1 - 1) <= CHAIN_MONEYNESS]
    if near:
        out["common"] = float(np.median(near))
        out["by"] = {k: (v, v - out["common"]) for k, v in own.items()}
    # each strike's own noise: how much its IV wobbles from one reading to the next (robust
    # spread of the per-reading pressure). A thin strike (Fin Nifty wings: day moves of 2-26
    # pts that were noise, 5-11 Nov 2026) must move 3x its wobble before anything is called.
    out["band"] = {}
    if not df.empty:
        pr = iv_pressure(df)
        if len(pr):
            pr = pr.dropna().reset_index()
            for (side, K), g in pr.groupby(["side", "K"]):
                x = g["p"].to_numpy()
                if len(x) >= 10:
                    mad = 1.4826 * float(np.median(np.abs(x - np.median(x))))
                    out["band"][(side, K)] = max(CHECK_BAND, CHECK_NOISE_X * mad)
    return out


def check_label(state: str | None, doi, vol, rel, expiry_day: bool,
                band: float = CHECK_BAND) -> tuple:
    """(verdict text, state the IV evidence supports or None) for one strike-side's
    premium-based label.

    OI up: buyers paying up lift the strike's IV against the chain, writers selling press it.
    OI down: shorts buying back lift it (covering), longs selling out press it (unwinding)."""
    if state not in STATE_NAME.values():
        return "—", None
    if expiry_day:
        return "— not checked (expiry day)", None
    if rel is None or (isinstance(rel, float) and np.isnan(rel)):
        return "— no IV to check", None
    if not vol or vol < CHECK_MIN_VOL:
        return f"— too thin to check ({(vol or 0):,.0f} traded)", None
    if doi is not None and abs(doi) < CHURN_FRESH * vol:
        return f"≈ churn ({abs(doi) / vol:.1%} of volume)", None
    opening = state in ("Long build-up", "Short build-up")
    if abs(rel) < band:
        return ("≈ index move only" if band <= CHECK_BAND + 1e-9
                else f"≈ inside this strike's noise (±{band:.1f})"), None
    iv_state = (("Long build-up" if rel > 0 else "Short build-up") if opening
                else ("Short covering" if rel > 0 else "Long unwinding"))
    if iv_state == state:
        return f"✅ confirmed (IV {rel:+.1f} vs chain)", iv_state
    return f"⚠️ IV says {PLAIN_STATE[iv_state]} ({rel:+.1f})", iv_state


def net_driver(states: pd.DataFrame) -> pd.DataFrame:
    """Per strike and block: calls − puts OI change since yesterday's close at the block's end
    (`level`), the block's step (`step` = calls' change − puts' change), and WHO drove the
    step: the side that moved more contracts, with its state and what that does to the index.

    A rising calls − puts line is NOT bullish by itself: it rises on call buying (bullish),
    call writing (bearish), put long unwinding (bullish) or put short covering (bearish) — so
    the driver, not the slope, carries the meaning. `lean` = sign of both sides' bullish minus
    bearish contracts; `code` names the bigger mover. A side with no reading in a block counts
    zero; a tie goes to the calls."""
    cols = ["K", "block", "level", "step", "dC", "dP", "side", "state", "lean", "code",
            "c_state", "p_state"]
    if states.empty:
        return pd.DataFrame(columns=cols)
    w = states.pivot_table(index=["K", "block"], columns="side",
                           values="d_oi", aggfunc="sum").reindex(columns=["CE", "PE"])
    st_ = states.set_index(["K", "block", "side"])
    out = []
    for K, g in w.groupby(level=0):
        g = g.droplevel(0).sort_index()
        dc, dp = g["CE"].fillna(0.0), g["PE"].fillna(0.0)
        level = (dc - dp).cumsum()
        for b in g.index:
            def look(sd, col, default=None):
                try:
                    v = st_.loc[(K, b, sd), col]
                    return v.iloc[0] if isinstance(v, pd.Series) else v
                except KeyError:
                    return default
            cs, ps = look("CE", "state"), look("PE", "state")
            side = "CE" if abs(dc[b]) >= abs(dp[b]) else "PE"
            state = cs if side == "CE" else ps
            # colour = BOTH sides together (bullish minus bearish contracts at this strike in
            # this block): the bigger mover alone would hide a big opposite move on the other
            # side. The label still names the bigger mover.
            lc, lp = (look("CE", "lean", 0) or 0), (look("PE", "lean", 0) or 0)
            lean = int(np.sign(lc * abs(dc[b]) + lp * abs(dp[b])))
            word = PLAIN_STATE.get(state, "") if state else ""
            code = (("C" if side == "CE" else "P") + SHORT_WORD.get(word, "")
                    if state in STATE_NAME.values() else "")
            out.append((K, b, level[b], dc[b] - dp[b], dc[b], dp[b], side,
                        state, lean, code, cs, ps))
    return pd.DataFrame(out, columns=cols)


def _strike_says(lc: int, lp: int, sc: str | None = None, sp: str | None = None) -> str:
    """One verdict per strike from its call and put states. Same state on BOTH sides is its
    own read, not 'mixed': writers on both sides = a straddle being sold (a range expected;
    9 Nov 2026 expiry day: every Best strike), buyers on both sides = a move expected."""
    if sc and sc == sp:
        both = {"Short build-up": "⚪ writers on both sides (range expected)",
                "Long build-up": "⚪ buyers on both sides (move expected)",
                "Short covering": "⚪ writers exiting both sides",
                "Long unwinding": "⚪ buyers exiting both sides"}
        if sc in both:
            return both[sc]
    t = lc + lp
    if t >= 2:
        return "🟢 bullish (both sides)"
    if t <= -2:
        return "🔴 bearish (both sides)"
    if t == 1:
        return "🟢 leaning bullish"
    if t == -1:
        return "🔴 leaning bearish"
    return "⚪ mixed" if lc or lp else "⚪ quiet"


def drill_table(latest: dict, df: pd.DataFrame, ks: list, center, by: str = "premium",
                spot=None, check: dict | None = None, expiry_day: bool = False) -> pd.DataFrame:
    """One row per strike of the bucket (high to low): today's call and put state, judged
    since yesterday's close (premium) or since each strike's first IV of the day (IV)."""
    ce, pe = row_sides(latest)
    cum_iv = {}
    if by == "IV" and not df.empty:
        pr = iv_pressure(df)
        if len(pr):
            pr = pr.reset_index()
            pr = pr[pr["K"].isin(ks)]
            cum_iv = pr.groupby(["side", "K"])["p"].sum(min_count=1).to_dict()
    near = min(ks, key=lambda k: (abs(k - spot), k)) if spot and ks else None
    rows = []
    for K in sorted(ks, reverse=True):
        rec = {"Strike": f"{K:,.0f}" + (" ★" if K == center else "")
               + (" ◀ index" if K == near else "")}
        leans, states, cleans, cstates = [], [], [], []
        for side, d, nm in (("CE", ce, "Call"), ("PE", pe, "Put")):
            x = d.get(K) or {}
            oi, doi = x.get("oi"), x.get("doi")
            if by == "IV":
                dx = cum_iv.get((side, K))
                dx = None if dx is None or (isinstance(dx, float) and np.isnan(dx)) else dx
            else:
                dx = x.get("chg")
            stt, ln = build_state(side, doi, dx, oi, by)
            leans.append(ln)
            states.append(stt)
            rec[f"{nm} OI"] = oi
            rec[f"{nm} ΔOI"] = doi
            rec[f"{nm} {'Δ premium' if by != 'IV' else 'Δ IV'}"] = dx
            vol = x.get("vol")
            rec[f"{nm} volume"] = vol
            rec[f"{nm} fresh %"] = (100.0 * doi / vol) if vol and doi is not None else None
            word = PLAIN_STATE[stt] if stt in STATE_NAME.values() else stt
            rec[f"{nm} state"] = ("—" if stt is None else
                                  ("🟢 " if ln > 0 else "🔴 " if ln < 0 else "⚪ ") + word)
            if check is not None:
                rel = (check.get("by", {}).get((side, K)) or (None, None))[1]
                verdict, ivs = check_label(stt, doi, vol, rel, expiry_day,
                                           check.get("band", {}).get((side, K), CHECK_BAND))
                rec[f"{nm} IV vs chain"] = rel
                rec[f"{nm} check"] = verdict
                cst = stt if verdict.startswith("✅") else ivs if verdict.startswith("⚠") else None
                cstates.append(cst)
                cleans.append(0 if cst is None else (1 if cst in _BULL_STATES[side] else -1))
        c, p_ = rec["Call ΔOI"], rec["Put ΔOI"]
        rec["Calls − puts ΔOI"] = (c - p_) if c is not None and p_ is not None else None
        rec["Strike says"] = _strike_says(*leans, *states)
        if check is not None:
            v = _strike_says(*cleans, *cstates)
            rec["Strike says (checked)"] = "⚪ nothing confirmed" if v == "⚪ quiet" else v
        rows.append(rec)
    out = pd.DataFrame(rows)
    for c in out.columns:
        if c != "Strike" and not c.endswith(("state", "says", "check", "(checked)")):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


def bucket_series(df: pd.DataFrame, bmap: dict, lot: int | None,
                  base: str = "yday") -> pd.DataFrame:
    """Long frame per reading x bucket: calls / puts OI change (₹ Cr at the first reading's
    index level, or contracts without a lot), calls - puts, PCR, and call / put OI. A reading
    counts for a bucket only when 90% of its strikes are in it (as session_series)."""
    if df.empty or not bmap:
        return pd.DataFrame()
    w = df[df["K"].isin(list(bmap))].copy()
    if w.empty:
        return pd.DataFrame()
    w["bucket"] = w["K"].map(bmap)
    if base == "first":
        w = w.sort_values("t")
        w["chg_"] = w["oi"] - w.groupby(["side", "K"])["oi"].transform("first")
    else:
        w["chg_"] = w["doi"]
    spot = df.groupby("t")["u"].first().dropna()
    mult = (lot * spot.iloc[0] / 1e7) if lot and len(spot) else 1.0
    size = pd.Series(bmap).value_counts()
    g = w.groupby(["t", "bucket", "side"]).agg(n=("K", "nunique"), chg=("chg_", "sum"),
                                                oi=("oi", "sum")).unstack("side")
    out = pd.DataFrame(index=g.index)
    for side, nm in (("CE", "calls"), ("PE", "puts")):
        if ("chg", side) not in g:
            return pd.DataFrame()
        need = SERIES_QUORUM * pd.Series([size[b] for b in g.index.get_level_values("bucket")],
                                         index=g.index)
        ok = g[("n", side)] >= need
        out[nm] = (g[("chg", side)] * mult).where(ok)
        out[f"{nm} OI"] = g[("oi", side)].where(ok)
    out["calls − puts"] = out["calls"] - out["puts"]
    out["PCR"] = out["puts OI"] / out["calls OI"]
    return out.reset_index()


METRICS = {
    "OI change since yesterday": ("doi", "contracts", "+,.0f"),
    "Open interest": ("oi", "contracts", ",.0f"),
    "Volume per block": ("vol5", "contracts", ",.0f"),
    "Implied volatility": ("IV", "IV %", ".2f"),
    "Premium (last price)": ("ltp", "₹", ",.2f"),
}


BLOCKS = (5, 15, 30, 60)          # minutes; every block counted from 09:15 (NSE's hourly candles)
SESSION_OPEN = dt.time(9, 15)


def block_start(t, minutes: int) -> pd.Series:
    """Start of the session block each time falls in, counted from 09:15: an hour block is
    09:15-10:15, 10:15-11:15 ... 14:15-15:15, 15:15-15:30 (as NSE / chart sites draw hourly
    candles). Clock-hour blocks (09:00-10:00) would give the first hour only 45 minutes."""
    t = pd.Series(pd.to_datetime(t)).reset_index(drop=True)
    anchor = t.dt.normalize() + pd.Timedelta(hours=SESSION_OPEN.hour, minutes=SESSION_OPEN.minute)
    m = pd.Timedelta(minutes=minutes)
    return anchor + ((t - anchor) // m) * m


def strike_panels(df: pd.DataFrame, strikes: list, metric: str,
                  minutes: int | None = None) -> pd.DataFrame:
    """Long frame for the small-multiples grid: t, K, side, v.

    minutes=None: every reading. Otherwise one point per block:
      levels (OI, OI change, IV, premium) - the LAST reading in the block, plotted at that
        reading's own time (the block still running ends at 'now', not in the future);
      volume - contracts traded IN the block (bars at the block's start), default 5 min."""
    col = METRICS[metric][0]
    w = df[df["K"].isin(strikes)]
    if w.empty:
        return pd.DataFrame(columns=["t", "K", "side", "v"])
    if col == "vol5":
        m = minutes or 5
        w = w.sort_values("t")
        parts = []
        for (side, K), g in w.groupby(["side", "K"]):
            v = g.set_index("t")["vol"]
            step = v.diff()
            step[step < 0] = np.nan          # NSE resets / a bad print: not negative volume
            if len(v):
                # NSE's volume is cumulative from 09:15, so the first reading's own count is
                # trading between 09:15 and that reading: it belongs to one block only when the
                # reading is inside the session's FIRST block. A late start (laptop woke at
                # 10:40) would otherwise pile the whole morning into one bar.
                b0 = block_start([v.index[0]], m).iloc[0]
                first_block = b0 == b0.normalize() + pd.Timedelta(
                    hours=SESSION_OPEN.hour, minutes=SESSION_OPEN.minute)
                step.iloc[0] = v.iloc[0] if first_block and pd.notna(v.iloc[0]) else np.nan
            b = step.groupby(block_start(step.index, m).values).sum(min_count=1)
            parts.append(pd.DataFrame({"t": b.index, "K": K, "side": side, "v": b.values}))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["t", "K", "side", "v"])
    out = w[["t", "K", "side", col]].rename(columns={col: "v"})
    if minutes:
        out = out.dropna(subset=["v"]).sort_values("t").reset_index(drop=True)
        out = out.groupby(["side", "K", block_start(out["t"], minutes)]).tail(1)
    return out.reset_index(drop=True)


def oi_blocks(df: pd.DataFrame, strikes: list, minutes: int = 15) -> pd.DataFrame:
    """Per strike, side and block: contracts opened (+) or closed (-) in that block. The first
    block is measured from YESTERDAY's close (OI - change at the first reading), so the opening
    rush is in it; after that, from the previous block's last reading. Blocks sum exactly to
    the day's change at the last reading."""
    w = df[df["K"].isin(strikes)].sort_values("t")
    out = []
    for (side, K), g in w.groupby(["side", "K"]):
        g = g.dropna(subset=["oi"])
        if g.empty:
            continue
        prev0 = g["oi"].iloc[0] - (g["doi"].iloc[0] or 0.0)
        last = g.groupby(block_start(g["t"], minutes).values)["oi"].last().dropna()
        chg = last.diff()
        chg.iloc[0] = last.iloc[0] - prev0
        out.append(pd.DataFrame({"block": chg.index, "K": K, "side": side, "chg": chg.values}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["block", "K", "side", "chg"])


def fut_series(rows: list) -> pd.DataFrame:
    """Near-month index future through the session (next month on its expiry day)."""
    recs = []
    for r in rows:
        try:
            t = pd.Timestamp(r["t"])
        except Exception:                                    # noqa: BLE001
            continue
        F = r.get("F") or {}
        live = [k for k in sorted(F) if dt.date.fromisoformat(k) > t.date()] or sorted(F)
        if not live:
            continue
        ltp, prev, oi, doi, vol = (F[live[0]] + [None] * 5)[:5]
        s = r.get("s")
        recs.append(dict(t=t, expiry=live[0], ltp=ltp, prev=prev, oi=oi, doi=doi, vol=vol,
                         spot=s))
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df = df.set_index("t").sort_index()
    df["fut %"] = (df["ltp"] / df["prev"] - 1) * 100
    df["premium bps"] = (df["ltp"] / df["spot"] - 1) * 1e4
    df["oi %"] = df["doi"] / (df["oi"] - df["doi"]) * 100
    return df


def fut_word(chg, doi) -> str | None:
    if chg is None or doi is None or chg == 0 or doi == 0:
        return None
    return ("long build" if chg > 0 else "short build") if doi > 0 else \
        ("short covering" if chg > 0 else "long unwind")


# ── lot size: proven from the bhavcopy, never guessed silently ─────────────────────────────
_LOTS: dict = {}          # key -> (lot, how, checked_at); a proof is kept all day
LOT_RETRY_S = 600.0       # an unproven lot is asked again after this long


def _lot(code: str, expiry: str, latest: dict, day: dt.date) -> tuple:
    """(lot, how). Proof: for every strike in the row with yesterday >= 100 contracts,
    bhavcopy OI (shares) / (live OI - change) must be the SAME whole number.

    Unproven results (archive locked by a DCM write, a pre-open chain whose 'change' is
    against the day before, a brand-new expiry) are retried every 10 minutes, never kept
    for the day."""
    key = (code, expiry, day)
    hit = _LOTS.get(key)
    if hit and (hit[1].startswith("proven") or time.time() - hit[2] < LOT_RETRY_S):
        return hit[0], hit[1]
    lot, how = None, None
    try:
        from . import data
        ce, pe = row_sides(latest)
        with data._connect() as c:
            prev = c.execute("select max(trade_date) from fno_bhavcopy where trade_date < ? "
                             "and symbol = ? and instrument = 'OPTIDX'", [day, code]).fetchone()[0]
            df = c.execute("""select strike_price k, option_type s, open_interest oi
                              from fno_bhavcopy where trade_date = ? and symbol = ?
                              and instrument = 'OPTIDX' and expiry_date = ?""",
                           [prev, code, dt.date.fromisoformat(expiry)]).df() if prev else None
        if df is not None and len(df):
            ratios = []
            for r in df.itertuples():
                d = (ce if r.s == "CE" else pe).get(float(r.k))
                if not d or d.get("doi") is None:
                    continue
                p = d["oi"] - d["doi"]
                if p >= 100:
                    ratios.append(r.oi / p)
            if len(ratios) >= 5:
                q = np.round(np.array(ratios))
                if np.all(np.abs(np.array(ratios) - q) < 1e-6) and len(set(q)) == 1 and q[0] > 0:
                    lot, how = int(q[0]), f"proven on {len(ratios)} strikes vs {prev:%d %b}"
    except Exception:                                        # noqa: BLE001
        pass
    if lot is None:
        other = [v for (c_, e_, d_), v in _LOTS.items()
                 if c_ == code and d_ == day and v[1].startswith("proven")]
        if other:
            lot, how = other[0][0], "same as this index's other expiry (proven)"
        elif code in LOT_FALLBACK:
            lot, how = LOT_FALLBACK[code], "assumed — no bhavcopy row could prove it"
        else:
            how = "unknown"
    _LOTS[key] = (lot, how, time.time())
    return lot, how


def sessions_left(day: dt.date, expiry: dt.date) -> int | None:
    im._dcm()                                    # puts DCM on the path for its calendar
    n = im._sessions_left(day, expiry)
    if n is None:
        n = int(np.busday_count(day + dt.timedelta(days=1), expiry + dt.timedelta(days=1)))
    return n


# ── page ──────────────────────────────────────────────────────────────────────────────
CE_COLOR, PE_COLOR = "#ef6c57", "#2bb673"

HELP = """
**What this page is.** NSE's live option chain for one index and one expiry, cut to the
**ATM (At The Money)** strike — the listed strike nearest the index — and **8 strikes either
side (17 strikes)**. Every NSE refresh (about once a minute) is logged from 09:15, so each
strike has its own history for the session: calls and puts, **OI (Open Interest)**, volume,
**IV (Implied Volatility)** and premium.

**Units.** OI and volume are **contracts**, as NSE prints them (checked: live OI × lot =
the bhavcopy exactly). Session totals are also shown in **₹ crore of exposure** = contracts ×
lot × index level at the day's first reading — the size of the position, not the premium.

**Live OI is provisional.** NSE corrects open interest after the bell (6 Nov 2026: whole Nifty
weekly chain −2.6% calls / −2.2% puts at 15:40; Nifty futures' change on the day flipped from
+8,927 to −8,509). The 15:40 figures equal the bhavcopy exactly. The page keeps that corrected
close and scores each day's live log against it — the score sits under the strike table.

**IV is computed here**, not taken from NSE: Black-76 against the forward implied by the
options' own put-call parity, from quote mids, on the market's clock. NSE's figure prices
against spot and disagreed between call and put at the same strike by 3-9 points on 6 Nov.

**The two reads in the table.**
- *IV read (default)*: OI change and IV change, both **since this morning's first reading**.
  More contracts at a richer IV = buyers paying up; more contracts at a cheaper IV = writers
  supplying. A rally lowers index IV across the chain, so on a strong up-day this leans
  "writers" — read it strike against strike, not as an absolute.
- *Premium read (as most sites)*: premium change × OI change vs yesterday. On the stock
  archive this was a price echo — a call's premium rises on any up-move.

**Colours.** 🟢 = the bullish reading of that side (call buying / call shorts covering; put
writing / put buyers leaving), 🔴 = the bearish one, ⚪ = not called.

**Nothing here forecasts.** Tradebot's tests on index option OI: walls follow price, max pain
does not pin (49% vs 50%), an OI surge precedes *smaller* moves, call/put crossovers are at
best mildly contrarian. Use it to see who is positioned where, not which way the index goes.
"""

TIP = {
    "Strike": "Strike price. ★ marks the ATM (At The Money) strike — the strike on the chosen "
              "strike gap nearest the index right now. Example: Nifty 24,938 → ATM 24,900 on "
              "the 100-point gap, 24,950 on NSE's listed 50-point gap.",
    "read": "One-word reading of this strike (see 'Read' switch above the table). Example: "
            "a put with 'writers adding' = more puts sold at a cheaper IV since the morning "
            "— someone is willing to be long the index below that strike. 🟢 bullish reading "
            "of that side, 🔴 bearish, ⚪ not called.",
    "ΔOI": "Change in open interest since YESTERDAY's close, in contracts (NSE's own figure). "
           "Example: +64,000 on the 25,000 put = 64,000 put contracts opened today.",
    "OI": "Open interest now, contracts. Example: 1,77,000 calls open at 25,000.",
    "ΔOI since first": "Change in open interest since TODAY's first logged reading (09:15 if "
                       "the log started then). This is what the IV read uses.",
    "Volume": "Contracts traded today so far. Example: 71,73,879 on the ATM call the day "
              "before expiry — volume many times OI is normal that close to expiry.",
    "IV": "Implied volatility %, computed here (Black-76) against the forward implied by the "
          "options themselves (put-call parity), from the bid/ask mid when the quote is tight, "
          "on the market's clock. NOT NSE's figure: NSE prices against spot and on 6 Nov put "
          "its call IV 5-9 points below its put IV at the same strike. Blank = no price gives "
          "an IV (stale print at or below intrinsic value). Example: 18.1 = priced for ~18% "
          "annualised moves.",
    "ΔIV": "IV now minus IV at today's first reading, vol points. Example: −1.5 = the option "
           "got 1.5 points cheaper in volatility terms since the morning.",
    "LTP": "Last traded premium, ₹. It is the LAST TRADE — on a strike that has not traded "
           "for a while it is old.",
    "Chg %": "Premium change vs yesterday's 'close' — NSE's close for an option is its LAST "
             "TRADE, which can be stale (5 Nov 2026: Nifty 25000 call 'close' ₹35.90 while "
             "NSE's own settlement value was ₹5.55). Example: +364% on an ATM call after a "
             "+400 pt day the session before expiry.",
}


def _fmt_n(v, f="{:,.0f}"):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def _ladder_view(st, tab: pd.DataFrame, latest: dict, mode: str) -> None:
    if tab.empty:
        st.info("No strikes to show.")
        return
    w = walls(latest, list(tab["Strike"]))
    cols_ce = ["CE read", "CE ΔOI", "CE OI", "CE ΔOI since first", "CE Volume", "CE ΔIV",
               "CE IV", "CE Chg %", "CE LTP"]
    cols_pe = [c.replace("CE", "PE") for c in reversed(cols_ce)]
    view = tab.copy()
    view["Strike"] = [f"★ {k:,.0f}" if a else f"{k:,.0f}" for k, a in zip(view["Strike"], view["atm"])]
    view = view[cols_ce + ["Strike"] + (["Bucket"] if "Bucket" in view else []) + cols_pe]

    def _row_style(r):
        atm = str(r["Strike"]).startswith("★")
        return ["background-color: rgba(255,200,0,0.13)" if atm else "" for _ in r]

    sty = view.style.apply(_row_style, axis=1)
    for side, key in (("CE", "ce_oi"), ("PE", "pe_oi")):
        k = w.get(key)
        if k is not None:
            mask = tab["Strike"] == k
            sty = sty.apply(lambda c, m=mask: ["font-weight: 700; color: #f5b041" if x else ""
                                                for x in m], subset=[f"{side} OI"], axis=0)
    for c in ("CE ΔOI", "PE ΔOI", "CE ΔOI since first", "PE ΔOI since first", "CE ΔIV",
              "PE ΔIV", "CE Chg %", "PE Chg %"):
        sty = sty.map(lambda v: ("color: #2bb673" if isinstance(v, (int, float)) and v > 0
                                 else "color: #ef6c57" if isinstance(v, (int, float)) and v < 0
                                 else ""), subset=[c])
    fmt = {}
    for side in ("CE", "PE"):
        fmt.update({f"{side} ΔOI": "{:+,.0f}", f"{side} OI": "{:,.0f}",
                    f"{side} ΔOI since first": "{:+,.0f}", f"{side} Volume": "{:,.0f}",
                    f"{side} ΔIV": "{:+.2f}", f"{side} IV": "{:.2f}",
                    f"{side} Chg %": "{:+.1f}", f"{side} LTP": "{:,.2f}"})
    sty = sty.format(fmt, na_rep="—")
    cc = st.column_config
    cfg = {"Strike": cc.TextColumn("Strike", help=TIP["Strike"]),
           "Bucket": cc.TextColumn("Bucket", help=TIP_BUCKET["Bucket"])}
    for side in ("CE", "PE"):
        nm = "Call" if side == "CE" else "Put"
        for c, tip in (("read", "read"), ("ΔOI", "ΔOI"), ("OI", "OI"),
                       ("ΔOI since first", "ΔOI since first"), ("Volume", "Volume"),
                       ("IV", "IV"), ("ΔIV", "ΔIV"), ("LTP", "LTP"), ("Chg %", "Chg %")):
            cfg[f"{side} {c}"] = cc.Column(f"{side} {c}", help=f"{nm}s — " + TIP[tip])
    st.dataframe(sty, column_config=cfg, hide_index=True, width="stretch",
                 height=38 + 35 * len(view))
    st.caption(f"Read: **{'IV read' if mode == 'iv' else 'premium read'}**. Bold amber OI = "
               f"largest open interest in these {len(tab)} strikes on that side (the 'wall' — walls "
               "follow price in Tradebot's tests; context, not a level to trade).")


TIP_BUCKET = {
    "Bucket": "Strikes grouped by distance from the money on the ladder above: Best = ATM "
              "(At The Money) and 3 strikes either side, Better = the next 2 either side, "
              "Good = the outer 3 either side. Example on Nifty with ATM 25,000 and a 100-point "
              "gap: Best 24,700-25,300, Better 24,500-24,600 and 25,400-25,500, Good the rest.",
    "Strikes": "Lowest–highest strike in the bucket (how many strikes).",
    "Call OI": "Call open interest summed over the bucket's strikes, contracts.",
    "Put OI": "Put open interest summed over the bucket's strikes, contracts.",
    "PCR": "PCR (Put-Call Ratio) = put OI ÷ call OI in this bucket now. Example: 1.40 = 40% "
           "more puts than calls open here.",
    "PCR yday": "The same ratio at yesterday's close (OI − today's change), same strikes. On a "
                "big-move day these strikes sat somewhere else relative to the index yesterday: "
                "5 Nov 2026, Nifty +2.3%, today's Best strikes (24,200-24,800) were far above "
                "the market at the 4 Nov close, few puts open there — PCR 0.03 then, 0.85 now.",
    "Call ΔOI": "Calls opened (+) or closed (−) in this bucket today vs yesterday's close, "
                "contracts.",
    "Put ΔOI": "Puts opened (+) or closed (−) in this bucket today, contracts.",
    "Call ΔOI since first": "Calls opened or closed since today's first logged reading.",
    "Put ΔOI since first": "Puts opened or closed since today's first logged reading.",
    "Index now": "The index level at the latest reading and where it sits against this "
                 "bucket. The buckets are fixed at the 09:15 open, the index moves: '◀ index "
                 "here' marks the bucket holding the strike nearest the index; the others show "
                 "how far the index must travel to reach their nearest strike.",
    "Call volume": "Call contracts traded today in this bucket.",
    "Put volume": "Put contracts traded today in this bucket.",
    "Calls − puts ₹ Cr": "Calls added minus puts added today, as ₹ crore of exposure "
                         "(contracts × lot × index level). Positive = call side added more.",
    "Today": "Which side added more contracts today (within 10% = balanced). Put side "
             "heavier is usually read as support because index writers dominate — but it is a "
             "MIRROR of the move, not a forecast: on 87 sessions the bucket PCR rose with the "
             "index in the same 15-30 minutes and said nothing about the next.",
}


DRILL_COLORS = {"bullish · new positions": "#1f9d55", "bullish · exits": "#8fd8a8",
                "bearish · new positions": "#d64545", "bearish · exits": "#f2a7a7",
                "quiet / unclear": "#3a3f47"}


def _drill_cat(state, lean) -> str:
    if not lean:
        return "quiet / unclear"
    new = state in ("Long build-up", "Short build-up")
    return f"{'bullish' if lean > 0 else 'bearish'} · {'new positions' if new else 'exits'}"


def _bucket_drill(st, code: str, pick: str, latest: dict, df: pd.DataFrame, bmap: dict,
                  center, clicked: bool) -> None:
    """The strikes of one bucket, one by one: today's state per call and put, the state block
    by block through the session, each strike's call vs put OI change, and the bucket's
    bullish-minus-bearish contracts per block."""
    import altair as alt
    ks = sorted(k for k, b in bmap.items() if b == pick)
    if not ks:
        return
    with st.container(border=True):
        st.markdown(f"#### 🔎 {pick} — {len(ks)} strikes, one by one"
                    + ("" if clicked else " · :gray[click a row in the table to open Better or Good]"))
        c1, c2 = st.columns([1, 1])
        by = c1.segmented_control(
            "Judge by", list(DRILL_BY), default="premium", key=f"if_dby_{code}",
            format_func=lambda x: ("premium (as sites do)" if x == "premium"
                                   else "IV pressure (experimental)"),
            help="Premium: OI change x premium change - the usual long build-up / short "
                 "build-up / short covering / long unwinding. It mostly restates the index "
                 "move (a call's premium rises when the index rises) and time decay. IV "
                 "pressure: the strike's IV change with the index-move slide along the smile "
                 "taken out — buyers paying up lift it, writers selling vol press it. OI up + "
                 "IV pressure up = buying; OI up + IV pressure down = writing.") or "premium"
        mins = c2.segmented_control("Time block", list(BLOCKS), default=15,
                                    key=f"if_dblk_{code}",
                                    format_func=lambda x: "1 hr" if x == 60 else f"{x} min") or 15
        if by == "IV" and str(latest.get("e", ""))[:10] == str(latest.get("t", ""))[:10]:
            # 9 Nov 2026 (Nifty weekly expiry): with hours left IV jumps on any tick — the
            # 25,400 call showed +200 vol points of 'pressure' by 15:29. Not a read.
            st.warning("**Expiry day of this contract** — with hours to go its IV jumps on "
                       "every tick (9 Nov: +200 vol points on one strike), so IV pressure is "
                       "not read today. Showing the premium reading instead.")
            by = "premium"
        expiry_day = str(latest.get("e", ""))[:10] == str(latest.get("t", ""))[:10]
        chk = None
        if by == "premium":
            try:
                dday = dt.date.fromisoformat(str(latest.get("t", ""))[:10])
                chk = day_check(df, latest, close_base(code, dday, str(latest.get("e", ""))[:10]))
            except ValueError:
                chk = None
        tab = drill_table(latest, df, ks, center, by, latest.get("u"), chk, expiry_day)
        dx = "Δ premium" if by != "IV" else "Δ IV"
        cc = st.column_config
        cfg = {}
        for nm in ("Call", "Put"):
            cfg[f"{nm} OI"] = cc.NumberColumn(f"{nm} OI", format="%,.0f")
            cfg[f"{nm} ΔOI"] = cc.NumberColumn(f"{nm} ΔOI", format="%+,.0f",
                                               help="Contracts added (+) or cut (−) since "
                                                    "yesterday's close.")
            cfg[f"{nm} {dx}"] = cc.NumberColumn(
                f"{nm} {dx}", format="%+,.2f",
                help=("Premium change since yesterday's close (₹)." if by != "IV" else
                      "IV pressure since the first reading: the strike's IV change minus "
                      "what the index move alone explains (slide along the smile). Vol "
                      f"points; under {IV_BAND} = unclear."))
            cfg[f"{nm} volume"] = cc.NumberColumn(f"{nm} volume", format="%,.0f",
                                                  help="Contracts traded today.")
            cfg[f"{nm} fresh %"] = cc.NumberColumn(
                f"{nm} fresh %", format="%+.1f%%",
                help="OI change ÷ volume. Near ±100% = nearly every trade opened (+) or "
                     "closed (−) a position: positional money. Near 0% = heavy trading that "
                     "left positions unchanged: intraday churn, the state means little.")
        cfg["Calls − puts ΔOI"] = cc.NumberColumn("Calls − puts ΔOI", format="%+,.0f")
        for nm in ("Call", "Put"):
            cfg[f"{nm} IV vs chain"] = cc.NumberColumn(
                f"{nm} IV vs chain", format="%+.1f",
                help="This strike's IV move today minus the whole chain's (vol points), with "
                     "the index move's effect taken out. + = paid up here more than elsewhere "
                     "(buyers), − = sold here more than elsewhere (writers).")
            cfg[f"{nm} check"] = cc.TextColumn(
                f"{nm} check",
                help="Does the IV agree with the premium label? ✅ confirmed · ⚠️ IV points the "
                     f"other way · ≈ IV moved under {CHECK_BAND} pts vs the chain (the premium "
                     "move is just the index) · churn = OI change under "
                     f"{CHURN_FRESH:.0%} of the volume.")
        cfg["Strike says (checked)"] = cc.TextColumn(
            "Strike says (checked)",
            help="The verdict again, counting only sides whose label the IV confirmed (or "
                 "using the IV's reading where it disagreed). Index-move-only sides count zero.")
        if chk is not None and not tab.empty:
            # most decision-useful first: strike, verdicts, then the numbers
            lead = [c for c in ("Strike", "Call state", "Call check", "Put state", "Put check",
                                "Strike says", "Strike says (checked)") if c in tab.columns]
            tab = tab[lead + [c for c in tab.columns if c not in lead]]
        st.dataframe(tab, column_config=cfg, hide_index=True, width="stretch")
        if chk is not None:
            if expiry_day:
                st.caption("Expiry day of this contract: IV is not used for the check today.")
            elif chk.get("common") is not None:
                cm = chk["common"]
                st.caption(
                    f"**The check** — IV since {'yesterday’s close' if chk['base'] == 'close' else 'the first reading today'}"
                    f", index effect removed. Whole chain (strikes within ±5%): **{cm:+.1f} vol "
                    "pts** — "
                    + ("options got cheaper everywhere (sellers in charge market-wide)" if cm < -CHECK_BAND
                       else "options got dearer everywhere (buyers in charge market-wide)" if cm > CHECK_BAND
                       else "about flat")
                    + ". Each strike is judged AGAINST that: ✅ when its own IV move agrees "
                    "with the premium label, ⚠️ when it points the other way, ≈ when the premium "
                    "label is just the index move. **One IV per strike:** put-call parity ties a "
                    "strike's call and put to the same volatility, so IV says vol was bought or "
                    "sold AT that strike, not through which side — it is pinned on the side whose "
                    "OI grew (11 Nov Bank Nifty 60,600: call +1.0, put +1.1). A consistency check "
                    "on the day's numbers, not a sight of the trades (NSE's chain does not show "
                    "who hit whom).")
        st.caption("**Calls:** long build-up / short covering = 🟢 bullish, short build-up "
                   "(call writing) / long unwinding = 🔴 bearish. **Puts:** the mirror — put "
                   "writing (short build-up) and put long unwinding are 🟢, put buying (long "
                   "build-up) and put short covering are 🔴. An OI change under "
                   f"{OI_FLOOR:.0%} of the strike's OI = quiet. ★ = the morning's centre strike, "
                   "◀ index = strike nearest the index now.")

        stt = strike_states(df, ks, mins, by)
        if stt.empty:
            st.caption("The block picture appears once a few readings are logged.")
            return
        stt["word"] = stt["state"].map(PLAIN_STATE).fillna("")
        stt["strike"] = [f"{k:,.0f}" + (" ★" if k == center else "") for k in stt["K"]]
        order = [f"{k:,.0f}" + (" ★" if k == center else "") for k in sorted(ks, reverse=True)]
        stt["panel"] = stt["side"].map({"CE": "CALLS", "PE": "PUTS"})
        span = "1 hr" if mins == 60 else f"{mins} min"

        # 1) BLOCK BY BLOCK — the main view: index move on top, then calls, then puts, each
        # the full page width so the cells are wide enough for words
        stt["effect"] = np.where(stt["lean"] > 0, "helps the index UP",
                                 np.where(stt["lean"] < 0, "pushes the index DOWN",
                                          "no clear effect"))
        stt["blk"] = pd.to_datetime(stt["block"]).dt.strftime("%H:%M")
        # EVERY block between the first and the last, so a capture gap (laptop asleep) shows
        # as empty columns — an ordinal axis otherwise closes it up silently, and the block
        # after it, which absorbs the whole gap's change, looks like one ordinary block
        b_ts = pd.to_datetime(stt["block"])
        blocks = [x.strftime("%H:%M") for x in
                  pd.date_range(b_ts.min(), b_ts.max(), freq=f"{mins}min")]
        n = len(blocks)
        # what fits in a cell on a laptop-wide page: full words up to ~14 columns (30-min /
        # 1-hr blocks; at 15-min, 25 columns, 'Writers out' ran into its neighbours — 11 Nov
        # 2026), short codes up to ~80 (5-min), colour only beyond
        stt["cell"] = (stt["word"] if n <= 14 else
                       stt["word"].map(SHORT_WORD).fillna("") if n <= 80 else "")
        st.markdown(f"**Who did what, block by block** — {pick}, {span} blocks from 09:15")
        x_enc = alt.X("blk:O", title=None, sort=blocks, scale=alt.Scale(domain=blocks),
                      axis=alt.Axis(labelAngle=0, labelFontSize=9, labelOverlap=True))
        Y_W = 70                                       # same label gutter on all three charts
        u = df.groupby("t")["u"].first().dropna()
        ub = u.groupby(block_start(pd.Series(u.index), mins).values).last()
        mv = ub.diff()
        if len(ub):
            # the cells' first block counts OI and premium from yesterday's close, so the
            # gap is in it: the index bar must count from there too, or a gap-up morning
            # shows green cells over a flat bar
            try:
                pc0 = prev_close(code, dt.date.fromisoformat(str(latest.get("t", ""))[:10]))
            except ValueError:
                pc0 = None
            mv.iloc[0] = ub.iloc[0] - (pc0 or latest.get("o") or u.iloc[0])
        idx = pd.DataFrame({"blk": pd.to_datetime(ub.index).strftime("%H:%M"), "pts": mv.values})
        idx = idx[idx["blk"].isin(blocks)]
        idx["dir"] = np.where(idx["pts"] >= 0, "index up", "index down")
        # titles and legend live OUTSIDE the charts: Streamlit fits a chart's title + legend
        # inside its height, which crushed the index bars to nothing and the put rows to half
        # height (11 Nov 2026)
        st.markdown("**INDEX** — points moved in each block")
        st.altair_chart(
            alt.Chart(idx).mark_bar().encode(
                x=x_enc.axis(None),
                y=alt.Y("pts:Q", title="index pts",
                        axis=alt.Axis(format="+,.0f", minExtent=Y_W, maxExtent=Y_W,
                                      tickCount=3)),
                color=alt.Color("dir:N", legend=None,
                                scale=alt.Scale(domain=["index up", "index down"],
                                                range=["#1f9d55", "#d64545"])),
                tooltip=[alt.Tooltip("blk:N", title="block from"),
                         alt.Tooltip("pts:Q", title="index move", format="+,.1f")])
            .properties(height=120), width="stretch")
        for side, title in (("CE", "CALLS"), ("PE", "PUTS")):
            d = stt[stt["side"] == side]
            if side == "PE" and latest.get("u"):
                # the index level between the two grids: calls above it, puts below
                o, now_u = latest.get("o"), latest["u"]
                try:
                    pc = prev_close(code, dt.date.fromisoformat(str(latest.get("t", ""))[:10]))
                except ValueError:
                    pc = None
                st.metric(DISPLAY.get(code, code), f"{now_u:,.2f}",
                          (f"{now_u - pc:+,.2f} ({(now_u / pc - 1) * 100:+.2f}%)" if pc else None),
                          help=f"NSE's index level at {str(latest.get('t', ''))[11:19]}"
                               + (f". Change vs the previous close {pc:,.2f} (as NSE and "
                                  "brokers show it)" if pc else "")
                               + (f" · {now_u - o:+,.2f} ({(now_u / o - 1) * 100:+.2f}%) from "
                                  f"the 09:15 open {o:,.2f}" if o else ""))
            st.markdown(f"**{title}**")
            base = alt.Chart(d).encode(
                x=x_enc, y=alt.Y("strike:N", sort=order, title=None,
                                 axis=alt.Axis(labelFontSize=12, labelFontWeight="bold",
                                               labelOverlap=False,
                                               minExtent=Y_W, maxExtent=Y_W)))
            rect = base.mark_rect(stroke="#0e1117", strokeWidth=1).encode(
                color=alt.Color("effect:N", title=None,
                                scale=alt.Scale(domain=["helps the index UP",
                                                        "pushes the index DOWN",
                                                        "no clear effect"],
                                                range=["#1f9d55", "#d64545", "#3a3f47"]),
                                legend=None),
                tooltip=[alt.Tooltip("blk:N", title="block from"),
                         alt.Tooltip("strike:N"), alt.Tooltip("word:N", title="who"),
                         alt.Tooltip("d_oi:Q", title="OI change (contracts)", format="+,.0f"),
                         alt.Tooltip("d_px:Q", title="premium change ₹", format="+,.2f"),
                         alt.Tooltip("d_iv:Q", title="IV pressure (pts)", format="+,.2f"),
                         alt.Tooltip("vol:Q", title="volume", format=",.0f"),
                         alt.Tooltip("fresh:Q", title="fresh (OI change ÷ volume)",
                                     format="+.1%")])
            txt = base.mark_text(fontSize=12 if n <= 14 else 11, fontWeight="bold",
                                 color="#ffffff").encode(text="cell:N")
            st.altair_chart((rect + txt).properties(height=40 * len(ks)), width="stretch")
        st.markdown(":green-background[ helps the index UP ] &nbsp; "
                    ":red-background[ pushes the index DOWN ] &nbsp; "
                    ":gray-background[ no clear effect ]")
        st.caption(
            "**Buyers in** = new longs (they paid) · **Writers in** = new shorts (they sold the "
            "option) · **Writers out** = shorts buying back · **Buyers out** = longs selling."
            + (" Codes: **B+** buyers in · **W+** writers in · **W−** writers out · "
               "**B−** buyers out." if 14 < n <= 80 else "")
            + " **Green = helps the index up** (calls: buyers in, writers out; puts: writers "
            "in, buyers out). '·' = quiet, '?' = unclear. Hover a cell for the numbers. A "
            "column that turns the index bar's colour on both sides is mostly the candle "
            "showing through.")

        # 2) the day added up, per strike
        with st.expander(f"Day total per strike — who did what, added up ({pick})"):
            # 1) THE DAY IN ONE PICTURE: per strike and side, today's contracts split by who did it
            flow = day_flow(stt)
            if not flow.empty:
                flow["strike"] = [f"{k:,.0f}" + (" ★" if k == center else "") for k in flow["K"]]
                flow["panel"] = flow["side"].map({"CE": "CALLS", "PE": "PUTS"})
                net = (flow.groupby(["panel", "strike"])
                       .agg(total=("v", "sum"), net=("signed", "sum")).reset_index())
                net["lab"] = [("🟢 " if n > 0 else "🔴 " if n < 0 else "⚪ ") + f"net {n:+,.0f}"
                              for n in net["net"]]
                # one data frame for every layer (Altair facets need it): the label rides on the
                # first row of each strike-side
                flow = flow.merge(net, on=["panel", "strike"], how="left")
                flow["first"] = ~flow.duplicated(["panel", "strike"])
                y_enc = alt.Y("strike:N", sort=order, title=None,
                              axis=alt.Axis(labelFontSize=11, labelFontWeight="bold"))
                bars = (alt.Chart().mark_bar().encode(
                    y=y_enc, x=alt.X("v:Q", title="contracts", stack="zero",
                                     axis=alt.Axis(format="~s")),
                    color=alt.Color("word:N", title=None,
                                    scale=alt.Scale(domain=list(WHO_COLORS),
                                                    range=list(WHO_COLORS.values())),
                                    legend=alt.Legend(orient="bottom")),
                    order=alt.Order("rank:Q"),
                    tooltip=[alt.Tooltip("panel:N", title="side"), alt.Tooltip("strike:N"),
                             alt.Tooltip("word:N", title="who"),
                             alt.Tooltip("v:Q", title="contracts", format=",.0f"),
                             alt.Tooltip("effect:N", title="for the index")]))
                labels = (alt.Chart().mark_text(align="left", dx=4, fontSize=10, color="#d0d4da")
                          .transform_filter("datum.first")
                          .encode(y=y_enc, x="total:Q", text="lab:N"))
                ch = (alt.layer(bars, labels, data=flow).properties(height=34 * len(ks), width=430)
                      .facet(column=alt.Column("panel:N", title=None, sort=["CALLS", "PUTS"],
                                               header=alt.Header(labelFontSize=13,
                                                                 labelFontWeight="bold"))))
                st.altair_chart(ch)
                st.caption(
                    "Each bar = the contracts that changed hands into or out of positions at that "
                    "strike today, split by who: **Buyers in** (new longs) · **Writers in** (new "
                    "shorts — they sold the option) · **Writers out** (shorts buying back) · "
                    "**Buyers out** (longs selling). **Net** = contracts that help the index UP minus "
                    "those that push it DOWN — calls: buyers in + writers out are up; puts: the "
                    "mirror, writers in + buyers out are up.")


        bb = bull_bear(stt)
        if not bb.empty:
            bb["blk"] = pd.to_datetime(bb["block"]).dt.strftime("%H:%M")
            long = pd.concat([
                bb.assign(kind="bullish contracts", v=bb["bull"]),
                bb.assign(kind="bearish contracts", v=-bb["bear"])])
            last = bb.iloc[-1]
            st.markdown(f"**Tug of war in {pick}: bullish vs bearish contracts per block**")
            bars = (alt.Chart(long).mark_bar(opacity=0.85).encode(
                x=x_enc, y=alt.Y("v:Q", title="contracts", axis=alt.Axis(format="~s")),
                color=alt.Color("kind:N", title=None,
                                scale=alt.Scale(domain=["bullish contracts", "bearish contracts"],
                                                range=["#1f9d55", "#d64545"]),
                                legend=alt.Legend(orient="bottom")),
                tooltip=[alt.Tooltip("blk:N", title="block from"),
                         alt.Tooltip("bull:Q", title="bullish contracts", format=",.0f"),
                         alt.Tooltip("bear:Q", title="bearish contracts", format=",.0f"),
                         alt.Tooltip("net:Q", title="net (bullish − bearish)", format="+,.0f")]))
            net = (alt.Chart(bb).mark_tick(color="#ffffff", thickness=3, size=28)
                   .encode(x=x_enc, y="net:Q",
                           tooltip=[alt.Tooltip("net:Q", title="net", format="+,.0f")]))
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#888").encode(y="y:Q")
            st.altair_chart((bars + zero + net).properties(height=200), width="stretch")
            st.caption(
                "Every strike's OI change in the block, sorted by what it does to the index: "
                "**green up** = call buyers in, call writers out, put writers in, put buyers out; "
                "**red down** = call writers in, call buyers out, put buyers in, put writers "
                f"out. White tick = the difference. Last block ({last['blk']}): "
                f"{last['bull']:,.0f} bullish vs {last['bear']:,.0f} bearish contracts → "
                + ("**bulls ahead**." if last["net"] > 0 else "**bears ahead**."
                   if last["net"] < 0 else "even."))

        _driver_panels(st, net_driver(stt), ks, center, mins, blocks, span)
        net = strike_panels(df, ks, "OI change since yesterday", mins)
        if not net.empty:
            net = (net.pivot_table(index=["t", "K"], columns="side", values="v")
                   .reindex(columns=["CE", "PE"]))
            net = (net["CE"] - net["PE"]).rename("v").reset_index().dropna()
            net["side"] = "CE"                                    # one line per panel
            st.markdown("**Each strike — calls − puts OI change** (above 0 = more calls added)")
            _net_panels(st, net, ks, center, mins)
        st.caption(
            "⚠️ **Buying or writing — what this data can and cannot tell.** **Premium:** on 9 "
            "Nov 2026 the colour matched the SAME block's index move 91% (Nifty) / 97% (Bank) "
            "— it re-describes the candle. **IV pressure** was tested on Tradebot's archive "
            "(Jun–Nov 2026, 55-86 non-expiry days per index, Best bucket, 15-min blocks): a "
            "block's IV pressure is followed by the OPPOSITE in the next block (rank −0.09 "
            "Nifty, −0.15 Bank, −0.25 Fin, −0.27 Midcap; t −5.7 to −25) — real buying or writing "
            "would carry on, so most of a block's IV move is measurement noise (last-trade "
            "bounce). Read only large, repeated moves. **Trade side** (last price at the ask vs "
            "the bid) was also tested and dropped: no persistence, no agreement with IV. Who "
            "really bought or wrote needs trade-by-trade data, which NSE's chain does not "
            "give. The bucket's bullish − bearish net did NOT forecast the next block on any "
            "index (t −1.4 to +1.5). **Fresh %** (OI change ÷ volume) is the solid number: "
            "only 7-24% of a block's volume becomes positions — the rest is intraday churn. "
            "Description of positioning, not a signal.")


def _net_panels(st, long: pd.DataFrame, strikes: list, center, minutes) -> None:
    import altair as alt
    long = long.copy()
    long["strike"] = [f"{k:,.0f}" + (" ★" if k == center else "") for k in long["K"]]
    order = [f"{k:,.0f}" + (" ★" if k == center else "") for k in sorted(strikes)]
    long["sign"] = np.where(long["v"] >= 0, "calls heavier", "puts heavier")
    area = (alt.Chart(long).mark_bar(opacity=0.85)
            .encode(x=alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M", labelFontSize=9,
                                                             tickCount=4)),
                    y=alt.Y("v:Q", title=None, axis=alt.Axis(labelFontSize=9, format="~s")),
                    color=alt.condition("datum.v >= 0", alt.value(CE_COLOR), alt.value(PE_COLOR)),
                    tooltip=[alt.Tooltip("t:T", title="time", format="%H:%M"),
                             alt.Tooltip("K:Q", title="strike", format=",.0f"),
                             alt.Tooltip("v:Q", title="calls − puts", format="+,.0f")]))
    zero = alt.Chart().mark_rule(color="#888").encode(y=alt.datum(0))
    ch = (alt.layer(area, zero, data=long).properties(width=230, height=110)
          .facet(facet=alt.Facet("strike:N", sort=order, title=None,
                                 header=alt.Header(labelFontSize=11, labelFontWeight="bold")),
                 columns=4)
          .resolve_scale(y="independent"))
    st.altair_chart(ch)



def _driver_panels(st, nd: pd.DataFrame, ks: list, center, mins: int, blocks: list,
                   span: str) -> None:
    """Calls − puts OI change per strike. Each block is one SEGMENT of the line, coloured by
    what that block's change did to the index (both sides counted); only the three biggest
    moves per strike are labelled with who drove them."""
    import altair as alt
    st.markdown(f"**Each strike — calls − puts OI change, and who moved it** ({span} blocks)")
    if nd.empty:
        st.caption("Not enough logged readings yet.")
        return
    nd = nd.sort_values(["K", "block"]).copy()
    nd["t"] = pd.to_datetime(nd["block"])
    nd["strike"] = [f"{k:,.0f}" + (" ★" if k == center else "") for k in nd["K"]]
    # lowest strike first, reading left-to-right / top-to-bottom like a price axis
    order = [f"{k:,.0f}" + (" ★" if k == center else "") for k in sorted(ks)]
    g = nd.groupby("K")
    # each segment runs from the previous block's level to this block's; the first starts
    # at 0 one block earlier (the count is since yesterday's close)
    nd["t0"] = g["t"].shift().fillna(nd["t"] - pd.Timedelta(minutes=mins))
    nd["y0"] = g["level"].shift().fillna(0.0)
    gap = (nd["t"] - nd["t0"]) > pd.Timedelta(minutes=mins)
    nd["absstep"] = nd["step"].abs()
    # colour only the moves that matter (≥10% of the strike's biggest block): a colour flip
    # on every few-hundred-contract wiggle striped the whole line (11 Nov 2026)
    sig = nd["absstep"] >= 0.10 * nd.groupby("K")["absstep"].transform("max")
    nd["effect"] = np.where(gap, "gap in the log",
                            np.where(~sig, "small move",
                                     np.where(nd["lean"] > 0, "helps the index UP",
                                              np.where(nd["lean"] < 0, "pushes the index DOWN",
                                                       "no clear effect"))))
    nd["blk"] = nd["t"].dt.strftime("%H:%M")
    nd["c_who"] = [PLAIN_STATE.get(x, "—") if isinstance(x, str) else "—" for x in nd["c_state"]]
    nd["p_who"] = [PLAIN_STATE.get(x, "—") if isinstance(x, str) else "—" for x in nd["p_state"]]

    def k_(v):
        a = abs(v)
        return (f"{v / 1e5:+.1f}L" if a >= 1e5 else f"{v / 1e3:+.0f}k" if a >= 1e3
                else f"{v:+.0f}")
    # label up to 3 of the biggest moves per strike, at least 2 blocks apart (labels crowded
    # each other at the open), each saying what the DRIVER did in its own contracts:
    # 'PW+ 29k' = put writers added 29,000 (not the line's −29k step)
    nd["drv"] = np.where(nd["side"] == "CE", nd["dC"], nd["dP"]).astype(float)
    big = nd["absstep"] >= 0.25 * nd.groupby("K")["absstep"].transform("max")
    nd["lab"] = ""
    for K, gk in nd[big & (nd["code"] != "") & ~gap].groupby("K"):
        picked = []
        for ix, r in gk.sort_values("absstep", ascending=False).iterrows():
            if len(picked) == 3:
                break
            if all(abs((r["t"] - q).total_seconds()) > 2 * mins * 60 for q in picked):
                picked.append(r["t"])
                nd.at[ix, "lab"] = r["code"] + " " + k_(abs(r["drv"])).lstrip("+")
    nd["up"] = nd["step"] >= 0
    x = alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M", labelFontSize=9, tickCount=7))
    tip = [alt.Tooltip("blk:N", title="block from"), alt.Tooltip("strike:N"),
           alt.Tooltip("level:Q", title="calls − puts since yesterday", format="+,.0f"),
           alt.Tooltip("step:Q", title="change in this block", format="+,.0f"),
           alt.Tooltip("dC:Q", title="calls' OI change", format="+,.0f"),
           alt.Tooltip("c_who:N", title="calls: who"),
           alt.Tooltip("dP:Q", title="puts' OI change", format="+,.0f"),
           alt.Tooltip("p_who:N", title="puts: who"), alt.Tooltip("effect:N")]
    ysc = alt.Scale(zero=False, nice=True, padding=22)      # headroom for the labels
    y = alt.Y("level:Q", title=None, scale=ysc,
              axis=alt.Axis(format="~s", labelFontSize=10))
    seg = alt.Chart().mark_rule(strokeWidth=3, strokeCap="round").encode(
        x="t0:T", x2="t:T", y=alt.Y("y0:Q", title=None, scale=ysc), y2="level:Q",
        tooltip=tip,
        color=alt.Color("effect:N", title=None,
                        scale=alt.Scale(domain=["helps the index UP", "pushes the index DOWN",
                                                "small move", "no clear effect",
                                                "gap in the log"],
                                        range=["#22b864", "#e04848", "#8a93a3", "#8a93a3",
                                               "#4a505a"]),
                        legend=alt.Legend(orient="bottom")),
        strokeDash=alt.condition("datum.effect == 'gap in the log'", alt.value([4, 4]),
                                 alt.value([1, 0])))
    end = (alt.Chart().transform_filter("datum.lab != ''")
           .mark_circle(size=70, color="#f2f4f7", stroke="#0e1117", strokeWidth=1.5)
           .encode(x=x, y=y, tooltip=tip))
    lab_up = (alt.Chart().transform_filter("datum.lab != '' && datum.up")
              .mark_text(dy=-15, fontSize=11, fontWeight="bold", color="#f2f4f7")
              .encode(x=x, y=y, text="lab:N"))
    lab_dn = (alt.Chart().transform_filter("datum.lab != '' && !datum.up")
              .mark_text(dy=17, fontSize=11, fontWeight="bold", color="#f2f4f7")
              .encode(x=x, y=y, text="lab:N"))
    zero = alt.Chart().mark_rule(color="#555", strokeDash=[3, 3]).encode(y=alt.datum(0))
    ch = (alt.layer(zero, seg, end, lab_up, lab_dn, data=nd).properties(width=560, height=210)
          .facet(facet=alt.Facet("strike:N", sort=order, title=None,
                                 header=alt.Header(labelFontSize=12, labelFontWeight="bold")),
                 columns=2)
          .resolve_scale(y="independent"))
    st.altair_chart(ch)
    st.caption(
        "The line = calls' OI change minus puts' since yesterday's close (above the dashed 0 = "
        "more calls added than puts). **Each piece of the line is one block, coloured by what "
        "that block did to the index**, both sides counted: 🟢 more contracts helping it up "
        "(call buyers in, call writers out, put writers in, put buyers out), 🔴 more pushing "
        "it down; grey = a small move (under 10% of the strike's biggest). A rise is not "
        "bullish by itself — a red rise is call writing. Up to three big moves per strike "
        "are labelled with who drove them and how many contracts THEY added or cut: C/P = "
        "calls/puts, then B+ buyers in · W+ writers in · W− writers out · B− buyers out "
        "(PW+ 29k = put writers added 29,000, which pulls the line down). Hover any piece "
        "for both sides. Own scale per panel.")


def _bucket_section(st, code: str, latest: dict, first: dict | None, df: pd.DataFrame,
                    strikes: list, center, lot: int | None, move_pct=None) -> dict:
    """Best / Better / Good: table + PCR through the session + calls vs puts per bucket."""
    import altair as alt
    bmap = bucket_map(strikes, center)
    st.subheader("Strike buckets — Best / Better / Good")
    sizes = {n: sum(1 for b in bmap.values() if b == n) for n in BUCKET_NAMES}
    st.caption(f"Grouped by distance from the money on the ladder above (centre {center:,.0f}): "
               f"**Best** = centre ± 3 strikes ({sizes['Best']}), **Better** = the next 2 either "
               f"side ({sizes['Better']}), **Good** = the outer strikes ({sizes['Good']}). The "
               "groups stay fixed for the session so a line moves only when positions move.")
    # What a bucket MEANS differs by index and expiry: +-3 strikes of 100 is +-0.5% on Bank
    # Nifty but +-2.1% on Midcap Select (5 Nov 2026). Said against the move the ATM straddle
    # prices to expiry, so 'Good' is never mistaken for 'the tails'.
    spot = latest.get("u")
    if spot and center:
        cov = []
        for n in BUCKET_NAMES:
            ks_n = [k for k, b in bmap.items() if b == n]
            if ks_n:
                reach = max(abs(k - center) for k in ks_n) / spot * 100
                cov.append(f"{n} ±{reach:.1f}%"
                           + (f" ({reach / move_pct:.1f}× the straddle's move)" if move_pct else ""))
        if cov:
            st.caption("How far each bucket reaches from the centre: " + " · ".join(cov)
                       + (f". The ATM straddle prices about ±{move_pct:.2f}% to expiry — a "
                          "bucket under 1× sits inside the move the market expects."
                          if move_pct else "."))
    bt = bucket_table(latest, first, bmap, lot)
    if bt.empty:
        st.info("No strikes to group.")
        return bmap
    cc = st.column_config
    cfg = {c: cc.Column(c, help=TIP_BUCKET[c]) for c in bt.columns if c in TIP_BUCKET}
    # name the index and the time: 'Nifty 50 @ 11:42' (a past day shows its last reading)
    cfg["Index now"] = cc.TextColumn(f"{DISPLAY.get(code, code)} @ {latest.get('t', '')[11:16]}",
                                     help=TIP_BUCKET["Index now"])
    for c in ("Call OI", "Put OI", "Call volume", "Put volume"):
        cfg[c] = cc.NumberColumn(c, format="%,.0f", help=TIP_BUCKET[c])
    for c in ("Call ΔOI", "Put ΔOI", "Call ΔOI since first", "Put ΔOI since first"):
        cfg[c] = cc.NumberColumn(c, format="%+,.0f", help=TIP_BUCKET[c])
    cfg["PCR"] = cc.NumberColumn("PCR", format="%.2f", help=TIP_BUCKET["PCR"])
    cfg["PCR yday"] = cc.NumberColumn("PCR yday", format="%.2f", help=TIP_BUCKET["PCR yday"])
    cfg["Calls − puts ₹ Cr"] = cc.NumberColumn("Calls − puts ₹ Cr", format="%+,.0f",
                                               help=TIP_BUCKET["Calls − puts ₹ Cr"])
    ev = st.dataframe(bt, column_config=cfg, hide_index=True, width="stretch",
                      on_select="rerun", selection_mode="single-row", key=f"if_btab_{code}")
    try:
        sel = list(ev.selection.rows)
    except Exception:                                        # noqa: BLE001 - no selection API
        sel = []
    pick = str(bt["Bucket"].iloc[sel[0]]) if sel and sel[0] < len(bt) else "Best"
    _bucket_drill(st, code, pick, latest, df, bmap, center, bool(sel))

    b1, b2 = st.columns([2, 1])
    bbase = b1.segmented_control(
        "Count from", ["since yesterday's close", "since this morning's first reading"],
        default="since yesterday's close", key=f"if_bbase_{code}") or "since yesterday's close"
    bs = bucket_series(df, bmap, lot, "first" if bbase.startswith("since this") else "yday")
    if bs.empty:
        st.caption("Bucket lines appear once a few readings are logged.")
        return bmap
    unit = "₹ Cr" if lot else "contracts"
    st.caption("**PCR (Put-Call Ratio) through the session, by bucket** — put OI ÷ call OI. Up = "
               "puts building faster than calls in that bucket. **It follows the index; it does "
               "not lead it.** Tested on 87 sessions of Tradebot's chain archive (22 Jun – 5 Nov "
               "2026, 4 indices, buckets fixed from 09:20): in the SAME 15/30-minute block the "
               "Best bucket's PCR rises with the index (rank correlation Nifty +0.36/+0.59, Bank "
               "Nifty +0.40/+0.57); in the NEXT block nothing (pooled t −0.7 to +0.9; 'puts "
               "built' was followed by a rise 45-54% of the time against a 48-51% base). A "
               "rising PCR is what a rally looks like, not a reason to expect one.")
    _panel_lines(st, bs.pivot(index="t", columns="bucket", values="PCR")
                 .reindex(columns=[b for b in BUCKET_NAMES if b in set(bs["bucket"])]),
                 "put ÷ call OI", BUCKET_COLOR, height=200, fmt=".2f")
    metric = b2.segmented_control("Show", ["OI change", "Open interest", "Volume per 5 min"],
                                  default="OI change", key=f"if_bmetric_{code}") or "OI change"
    if metric == "Volume per 5 min":
        v = strike_panels(df, strikes, "Volume per block", 5)
        v["bucket"] = v["K"].map(bmap)
        long = (v.dropna(subset=["v"]).groupby(["t", "bucket", "side"])["v"].sum()
                .reset_index())
        ylab, fmt = "contracts / 5 min", ",.0f"
    else:
        cols = ("calls", "puts") if metric == "OI change" else ("calls OI", "puts OI")
        long = bs.melt(id_vars=["t", "bucket"], value_vars=list(cols), var_name="side",
                       value_name="v")
        long["side"] = long["side"].map({cols[0]: "CE", cols[1]: "PE"})
        ylab = f"OI change, {unit}" if metric == "OI change" else "open interest, contracts"
        fmt = "+,.0f" if metric == "OI change" else ",.0f"
    long = long.dropna(subset=["v"])
    if long.empty:
        st.caption("Not enough logged readings yet.")
        return bmap
    long["g"] = long["bucket"] + "·" + long["side"]
    long = im._with_segments(long, "t", "g")
    base = alt.Chart(long)
    mark = base.mark_bar() if metric.startswith("Volume") else base.mark_line(strokeWidth=2)
    enc = dict(
        x=alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M", tickCount=6)),
        y=alt.Y("v:Q", title=ylab, scale=alt.Scale(zero=metric != "Open interest"),
                axis=alt.Axis(format="~s")),
        color=alt.Color("side:N", title=None,
                        scale=alt.Scale(domain=["CE", "PE"], range=[CE_COLOR, PE_COLOR]),
                        legend=alt.Legend(orient="bottom", labelExpr=(
                            "datum.label == 'CE' ? 'Calls (CE)' : 'Puts (PE)'"))),
        tooltip=[alt.Tooltip("t:T", title="time", format="%H:%M"), "bucket:N", "side:N",
                 alt.Tooltip("v:Q", title=ylab, format=fmt)])
    ch = mark.encode(**enc)
    ch = ch.encode(xOffset="side:N") if metric.startswith("Volume") else ch.encode(detail="line:N")
    ch = (ch.properties(width=330, height=200)
          .facet(column=alt.Column("bucket:N", sort=list(BUCKET_NAMES), title=None,
                                   header=alt.Header(labelFontSize=13, labelFontWeight="bold")))
          .resolve_scale(y="independent"))
    st.caption(f"**Calls vs puts by bucket** — {metric.lower()}"
               + (f", {bbase}" if metric == "OI change" else "") + ". Each panel has its own scale.")
    st.altair_chart(ch)
    if metric == "OI change":
        st.caption(f"**Calls − puts by bucket** ({unit}, ↑ call side added more)")
        _lines(st, bs.pivot(index="t", columns="bucket", values="calls − puts")
               .reindex(columns=[b for b in BUCKET_NAMES if b in set(bs["bucket"])]),
               f"calls − puts, {unit}", BUCKET_COLOR, height=220, zero=True)
    return bmap


def panel_titles(wide: pd.DataFrame, fmt: str = "{:.2f}", dfmt: str = "{:+.2f}") -> dict:
    """{series: 'Best · 0.48 now · -0.02 since 09:20'} from each line's first and last value."""
    out = {}
    for c in wide.columns:
        v = wide[c].dropna()
        if v.empty:
            out[c] = str(c)
            continue
        out[c] = (f"{c} · {fmt.format(v.iloc[-1])} now · "
                  f"{dfmt.format(v.iloc[-1] - v.iloc[0])} since {v.index[0]:%H:%M}")
    return out


def _panel_lines(st, wide: pd.DataFrame, ylab: str, colors: dict | None = None,
                 height: int = 170, fmt: str = ".2f", width: int = 330) -> None:
    """One small chart per line, each zoomed to its OWN range. On one shared axis the PCR
    lines sat at 0.5 / 0.6 / 1.3 and each bucket's real movement (a few hundredths) was
    flattened to nothing (9 Nov 2026). The header carries the level and the change, so the
    buckets still compare by number."""
    import altair as alt
    if wide is None or wide.empty or wide.dropna(how="all").empty:
        st.caption("Not enough logged readings yet.")
        return
    titles = panel_titles(wide, "{:" + fmt + "}", "{:+" + fmt + "}")
    order = [titles[c] for c in wide.columns]
    long = (wide.rename_axis("t").reset_index()
            .melt("t", var_name="series", value_name="v").dropna())
    long["panel"] = long["series"].map(titles)
    long = im._with_segments(long, "t", "series")
    color = alt.Color("series:N", legend=None)
    if colors:
        color = alt.Color("series:N", legend=None,
                          scale=alt.Scale(domain=list(colors), range=list(colors.values())))
    ch = (alt.Chart(long).mark_line(strokeWidth=2)
          .encode(x=alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M", tickCount=6)),
                  y=alt.Y("v:Q", title=ylab, scale=alt.Scale(zero=False, nice=True)),
                  color=color, detail="line:N",
                  tooltip=[alt.Tooltip("t:T", title="time", format="%H:%M"), "series:N",
                           alt.Tooltip("v:Q", title=ylab, format=fmt)])
          .properties(width=width, height=height)
          .facet(column=alt.Column("panel:N", sort=order, title=None,
                                   header=alt.Header(labelFontSize=12, labelFontWeight="bold")))
          .resolve_scale(y="independent"))
    st.altair_chart(ch)


def index_domain(spot: pd.Series, open_lvl=None, pad: float = 0.08) -> tuple | None:
    """y-range for the index line: the day's own low-high (and the open, if known) plus 8% each
    side. Altair's 'nice' rounding stretched a 160-point Nifty day to a 1,000-point axis
    (9 Nov 2026: the line sat flat on the 25,000 gridline)."""
    v = pd.to_numeric(spot, errors="coerce").dropna()
    if v.empty:
        return None
    lo, hi = float(v.min()), float(v.max())
    if open_lvl:
        lo, hi = min(lo, float(open_lvl)), max(hi, float(open_lvl))
    w = hi * 0.001                             # a flat stretch still gets 0.1% of height,
    if hi - lo < w:                            # centred on it
        extra = (w - (hi - lo)) / 2
        lo, hi = lo - extra, hi + extra
    span = hi - lo
    return lo - pad * span, hi + pad * span


def _index_line(st, spot: pd.Series, name: str, open_lvl=None, height: int = 220) -> None:
    """The index through the session on its OWN range, with the 09:15 open as a dashed line
    and the day's low / high / last in the header."""
    import altair as alt
    v = pd.to_numeric(spot, errors="coerce").dropna()
    dom = index_domain(v, open_lvl)
    if dom is None:
        st.caption("Not enough logged readings yet.")
        return
    long = v.rename("v").rename_axis("t").reset_index().assign(series=name)
    long = im._with_segments(long, "t", "series")
    last = float(v.iloc[-1])
    head = f"{name} {last:,.2f}"
    if open_lvl:
        head += f" ({last - open_lvl:+,.2f} · {(last / open_lvl - 1) * 100:+.2f}% from open {open_lvl:,.2f})"
    head += f" · low {v.min():,.2f} at {v.idxmin():%H:%M} · high {v.max():,.2f} at {v.idxmax():%H:%M}"
    line = (alt.Chart(long).mark_line(strokeWidth=2, color="#5aa9e6")
            .encode(x=alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M")),
                    y=alt.Y("v:Q", title=None,
                            scale=alt.Scale(domain=list(dom), nice=False, zero=False),
                            axis=alt.Axis(format=",.0f", tickCount=5)),
                    detail="line:N",
                    tooltip=[alt.Tooltip("t:T", title="time", format="%H:%M"),
                             alt.Tooltip("v:Q", title=name, format=",.2f")]))
    ch = line
    if open_lvl:
        ref = pd.DataFrame({"y": [float(open_lvl)], "lab": [f"open {open_lvl:,.2f}"]})
        ch = ch + alt.Chart(ref).mark_rule(color="#999", strokeDash=[4, 4]).encode(y="y:Q")             + alt.Chart(ref).mark_text(align="left", dx=4, dy=-6, color="#999", fontSize=10)             .encode(x=alt.value(0), y="y:Q", text="lab:N")
    st.markdown(f"**{head}**")
    st.altair_chart(ch.properties(height=height), width="stretch")


def _lines(st, wide: pd.DataFrame, ylab: str, colors: dict | None = None, height=240,
           zero=False, fmt="+,.0f"):
    import altair as alt
    if wide is None or wide.empty or wide.dropna(how="all").empty:
        st.caption("Not enough logged readings yet.")
        return
    long = (wide.rename_axis("t").reset_index()
            .melt("t", var_name="series", value_name="v").dropna())
    long = im._with_segments(long, "t", "series")
    color = alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom"))
    if colors:
        color = alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom"),
                          scale=alt.Scale(domain=list(colors), range=list(colors.values())))
    ch = (alt.Chart(long).mark_line(strokeWidth=2)
          .encode(x=alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M")),
                  y=alt.Y("v:Q", title=ylab, scale=alt.Scale(zero=False)),
                  color=color, detail="line:N",
                  tooltip=[alt.Tooltip("t:T", title="time", format="%H:%M"), "series:N",
                           alt.Tooltip("v:Q", title=ylab, format=fmt)])
          .properties(height=height))
    if zero:
        ch = ch + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#888").encode(y="y:Q")
    st.altair_chart(ch, width="stretch")


def _panels_chart(st, long: pd.DataFrame, strikes: list, atm, metric: str, shared: bool,
                  minutes: int | None = None):
    import altair as alt
    if long.empty or long["v"].dropna().empty:
        st.caption("Not enough logged readings yet.")
        return
    unit, fmt = METRICS[metric][1], METRICS[metric][2]
    long = long.dropna(subset=["v"]).copy()
    long["strike"] = [f"{k:,.0f}" + (" ★ ATM" if k == atm else "") for k in long["K"]]
    order = [f"{k:,.0f}" + (" ★ ATM" if k == atm else "") for k in sorted(strikes, reverse=True)]
    long["g"] = long["side"] + "·" + long["K"].astype(str)
    # a line breaks only where a whole block has no reading (laptop asleep), not between blocks
    long = im._with_segments(long, "t", "g", pd.Timedelta(minutes=(minutes or 0) + 6))
    mark = alt.Chart(long).mark_bar() if metric.startswith("Volume") else \
        alt.Chart(long).mark_line(strokeWidth=1.8)
    base = mark.encode(
        x=alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M", labelFontSize=9,
                                                 tickCount=4)),
        y=alt.Y("v:Q", title=None, scale=alt.Scale(zero=metric.startswith(("Volume", "OI c"))),
                axis=alt.Axis(labelFontSize=9, format="~s")),
        color=alt.Color("side:N", title=None,
                        scale=alt.Scale(domain=["CE", "PE"], range=[CE_COLOR, PE_COLOR]),
                        legend=alt.Legend(orient="bottom", labelExpr=(
                            "datum.label == 'CE' ? 'Calls (CE)' : 'Puts (PE)'"))),
        tooltip=[alt.Tooltip("t:T", title=("block from" if metric.startswith("Volume")
                                           else "time"), format="%H:%M"), "side:N",
                 alt.Tooltip("K:Q", title="strike", format=",.0f"),
                 alt.Tooltip("v:Q", title=unit, format=fmt)])
    if not metric.startswith("Volume"):
        base = base.encode(detail="line:N")
    else:
        base = base.encode(xOffset="side:N")
    ch = (base.properties(width=230, height=120)
          .facet(facet=alt.Facet("strike:N", sort=order, title=None,
                                 header=alt.Header(labelFontSize=11, labelFontWeight="bold")),
                 columns=4)
          .resolve_scale(y="shared" if shared else "independent"))
    st.altair_chart(ch)


def _heatmap(st, blk: pd.DataFrame, strikes: list, atm, minutes: int):
    import altair as alt
    if blk.empty:
        st.caption("Not enough logged readings yet.")
        return
    b = blk.copy()
    b["strike"] = [f"{k:,.0f}" + (" ★" if k == atm else "") for k in b["K"]]
    order = [f"{k:,.0f}" + (" ★" if k == atm else "") for k in sorted(strikes, reverse=True)]
    b["side"] = b["side"].map({"CE": "Calls (CE)", "PE": "Puts (PE)"})
    # colour range = 95th percentile of |change|, clamped: the opening block (counted from
    # yesterday's close) is nearly always the biggest and otherwise washes out every other cell
    m = float(np.nanquantile(np.abs(b["chg"].astype(float)), 0.95)) if len(b) else 1.0
    m = m if m > 0 else 1.0
    ch = (alt.Chart(b).mark_rect()
          .encode(x=alt.X("block:T", title=None, axis=alt.Axis(format="%H:%M")),
                  y=alt.Y("strike:N", sort=order, title=None),
                  color=alt.Color("chg:Q", title="contracts",
                                  scale=alt.Scale(scheme="redblue", domain=[-m, m],
                                                  clamp=True)),
                  tooltip=[alt.Tooltip("block:T", title=f"{minutes}-min block from",
                                       format="%H:%M"), "side:N",
                           alt.Tooltip("K:Q", title="strike", format=",.0f"),
                           alt.Tooltip("chg:Q", title="contracts opened (+) / closed (−)",
                                       format="+,.0f")])
          .properties(width=420, height=17 * len(strikes))
          .facet(column=alt.Column("side:N", title=None, sort=["Calls (CE)", "Puts (PE)"])))
    st.altair_chart(ch)


def _profile(st, latest: dict, strikes: list, atm=None):
    atm = latest.get("a") if atm is None else atm
    import altair as alt
    ce, pe = row_sides(latest)
    recs = []
    for K in strikes:
        for side, d in (("Calls (CE)", ce.get(K)), ("Puts (PE)", pe.get(K))):
            if not d:
                continue
            oi, doi = d.get("oi") or 0, d.get("doi") or 0
            recs += [dict(K=K, side=side, part="yesterday's OI still open", v=min(oi, oi - doi)),
                     dict(K=K, side=side, part="added today", v=max(doi, 0)),
                     dict(K=K, side=side, part="closed today", v=max(-doi, 0))]
    if not recs:
        return
    df = pd.DataFrame(recs)
    df["strike"] = [f"{k:,.0f}" + (" ★" if k == atm else "") for k in df["K"]]
    order = [f"{k:,.0f}" + (" ★" if k == atm else "") for k in
             sorted(strikes, reverse=True)]
    ch = (alt.Chart(df).mark_bar()
          .encode(y=alt.Y("strike:N", sort=order, title=None),
                  x=alt.X("v:Q", title="contracts", stack="zero"),
                  color=alt.Color("part:N", title=None, legend=alt.Legend(orient="bottom"),
                                  scale=alt.Scale(domain=["yesterday's OI still open",
                                                          "added today", "closed today"],
                                                  range=["#5b6577", "#f5b041", "#3a8dde"])),
                  tooltip=["side:N", alt.Tooltip("K:Q", title="strike", format=",.0f"),
                           "part:N", alt.Tooltip("v:Q", title="contracts", format=",.0f")])
          .properties(width=380, height=17 * len(strikes))
          .facet(column=alt.Column("side:N", title=None)))
    st.altair_chart(ch)


def smile_points(row: dict | None, strikes: list) -> dict:
    """{K: IV} from the OUT-OF-THE-MONEY side at each strike (puts below the forward, calls
    above) — the liquid side, and the desk convention for a smile. Falls back to the other
    side where the OTM side has no IV."""
    if not row:
        return {}
    ce, pe = row_sides(row)
    F = row.get("F") or row.get("u")
    out = {}
    for K in strikes:
        otm, itm = (pe, ce) if F and K < F else (ce, pe)
        v = (otm.get(K) or {}).get("IV") or (itm.get(K) or {}).get("IV")
        if v:
            out[K] = v
    return out


def _smile(st, latest: dict, first: dict | None, strikes: list, atm=None, base=None):
    recs = [dict(K=K, series="now", v=v) for K, v in smile_points(latest, strikes).items()]
    if base is not None:             # each strike's first valid IV today, OTM side
        F = latest.get("F") or latest.get("u")
        for K in strikes:
            otm, itm = ("PE", "CE") if F and K < F else ("CE", "PE")
            v = base[otm].get(K) or base[itm].get(K)
            if v:
                recs.append(dict(K=K, series="first reading", v=v))
    if not recs:
        st.caption("No IV for these strikes.")
        return
    import altair as alt
    df = pd.DataFrame(recs)
    dom = ["now", "first reading"]
    rng = ["#f5b041", "#8a8f98"]
    ch = (alt.Chart(df).mark_line(point=True)
          .encode(x=alt.X("K:Q", title="strike", scale=alt.Scale(zero=False),
                          axis=alt.Axis(format=",.0f")),
                  y=alt.Y("v:Q", title="IV %", scale=alt.Scale(zero=False)),
                  color=alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom"),
                                  scale=alt.Scale(domain=[d for d in dom if d in set(df.series)],
                                                  range=[r for d, r in zip(dom, rng)
                                                         if d in set(df.series)])),
                  strokeDash=alt.condition("indexof(datum.series, 'first') >= 0",
                                           alt.value([4, 3]), alt.value([1, 0])),
                  tooltip=["series:N", alt.Tooltip("K:Q", format=",.0f"),
                           alt.Tooltip("v:Q", title="IV %", format=".2f")])
          .properties(height=260))
    atm = latest.get("a") if atm is None else atm
    if atm:
        ch = ch + alt.Chart(pd.DataFrame({"x": [atm]})).mark_rule(
            color="#f5b041", strokeDash=[2, 2]).encode(x="x:Q")
    st.altair_chart(ch, width="stretch")


def _expiry_label(e: dt.date, expiries: list, today: dt.date) -> str:
    kind = "monthly" if is_monthly(e, expiries) else "weekly"
    n = sessions_left(today, e) if e >= today else None
    left = ("expires today" if e == today else
            f"{n} session{'s' if n != 1 else ''} left" if n is not None else "")
    return f"{e:%d %b %Y} · {kind}" + (f" · {left}" if left else "")


def freshness(latest: dict, fut: dict | None, rows: list, live: bool,
              now: dt.datetime | None = None) -> dict:
    """The page's clocks as ages in seconds — the same four Index Management shows.

      chain stamp  - the time NSE printed on the option chain (moves about once a minute)
      chain read   - when the writer last ASKED NSE for it (can be seconds ago while the stamp
                     is a minute old: NSE has nothing newer, we are not late)
      changed      - when the chain's NUMBERS last changed (a frozen feed keeps a ticking stamp)
      futures      - NSE's stamp on the index futures read, and when it was read
    Levels follow Index Management: green up to 3 minutes, orange to 6, red beyond — only
    while the market is open; grey otherwise."""
    now = now or _now()

    def ts(x):
        try:
            return dt.datetime.fromisoformat(x) if x else None
        except (TypeError, ValueError):
            return None

    def age(x):
        return None if x is None else max(0.0, (now - x).total_seconds())

    open_now = live and in_session(now) and capture_window(now)

    def level(a):
        if a is None:
            return "none"
        if not open_now:
            return "closed"
        return "ok" if a <= im.FRESH_OK else "warn" if a <= im.FRESH_WARN else "stale"

    c_stamp, c_read, c_chg = ts(latest.get("t")), ts(latest.get("f")), ts(latest.get("c"))
    f_stamp, f_read = ts((fut or {}).get("t")), ts((fut or {}).get("f"))
    out = dict(chain_stamp=c_stamp, chain_age=age(c_stamp), chain_read_age=age(c_read),
               changed_age=age(c_chg) if live else None,
               fut_stamp=f_stamp, fut_age=age(f_stamp), fut_read_age=age(f_read),
               drawn=now, open=open_now,
               next_chain=None if c_read is None or not open_now
               else max(0.0, CHAIN_EVERY - age(c_read)),
               next_fut=None if f_read is None or not open_now
               else max(0.0, FUT_EVERY - age(f_read)),
               logged=len(rows or []),
               log_from=rows[0]["t"][11:16] if rows else None)
    # The DOT is judged on how old NSE's data was WHEN READ plus how overdue the next read is,
    # not on its age now: futures are asked every 120 s and NSE's stamp is already 40-66 s old
    # when read (9 Nov 2026), so 'age now' crossed the 3-minute orange line between every two
    # reads with nothing wrong. A frozen feed (old when read) and a dead writer (reads
    # overdue) both still turn it orange / red.
    def effective(a, read_a, every):
        if a is None or read_a is None:
            return a
        return max(0.0, a - read_a) + max(0.0, read_a - every)
    out["chain_level"] = level(effective(out["chain_age"], out["chain_read_age"], CHAIN_EVERY))
    out["fut_level"] = level(effective(out["fut_age"], out["fut_read_age"], FUT_EVERY))
    return out


def _freshness(st, latest: dict, fut: dict | None, rows: list, live: bool):
    """The time strip above the page, refreshed with it every 15s (as Index Management's)."""
    f = freshness(latest, fut, rows, live)
    ago, lv = im._ago, im._LEVEL
    with st.container(border=True):
        c = st.columns(4)
        if f["chain_stamp"]:
            c[0].markdown(
                f"{lv[f['chain_level']]} **Option chain** · NSE time "
                f"**{f['chain_stamp']:%H:%M:%S}**  \n:gray[{ago(f['chain_age'])}"
                + (f" · numbers changed {ago(f['changed_age'])}" if f["changed_age"] is not None
                   else "") + "]")
        else:
            c[0].markdown(":gray[○] **Option chain** · not read yet")
        if f["fut_stamp"]:
            c[1].markdown(f"{lv[f['fut_level']]} **Index futures** · NSE time "
                          f"**{f['fut_stamp']:%H:%M:%S}**  \n:gray[{ago(f['fut_age'])}]")
        else:
            c[1].markdown(":gray[○] **Index futures** · not read yet")
        if f["open"]:
            c[2].markdown(f"**Next check** · chain in **{f['next_chain'] or 0:.0f}s**, futures in "
                          f"**{f['next_fut'] or 0:.0f}s**  \n"
                          f":gray[chain last asked {ago(f['chain_read_age'])}]")
        else:
            c[2].markdown("**Next check** · market closed  \n"
                          + (":gray[final numbers for the session]" if live
                             else ":gray[a past session, from the log]"))
        c[3].markdown(f"**Page drawn** · **{f['drawn']:%H:%M:%S}**  \n"
                      f":gray[redraws every 15s · {f['logged']} readings logged"
                      + (f" from {f['log_from']}" if f["log_from"] else "") + "]")
        if f["chain_level"] in ("warn", "stale") or f["fut_level"] in ("warn", "stale"):
            st.caption(":orange[A dot that is not green means NSE has not published newer data "
                       "for over 3 minutes (red: over 6). Read the page as of that time, not "
                       f"as now. The writer opens a fresh NSE session by itself after "
                       f"{HEAL_AFTER / 60:.0f} minutes of unchanged numbers; ↻ refresh in the "
                       "sidebar forces it now.]")
        else:
            st.caption("NSE time = the time printed on NSE's own data. The chain and the futures "
                       "each move about once a minute, however often this page asks — a stamp up "
                       "to two minutes old is normal. 'Numbers changed' catches a feed whose clock "
                       "ticks while its numbers have stopped.")

def _draw(st, code: str, expiry: str, day: dt.date, live: bool) -> None:
    rows = chain_log(code, day, expiry)
    lat = load_latest(code) if live else {}
    latest = (lat.get("chain") or {}).get(expiry) if live else None
    if latest is None and rows:
        latest = rows[-1]
    if latest is None:
        st.info("No chain read yet for this expiry — the capture asks NSE within a minute "
                "of the page opening (or at 09:15 on a trading day).")
        return
    first = rows[0] if rows else None
    fut_rows = fut_log(code, day)
    fut = (lat.get("fut") if live else None) or (fut_rows[-1] if fut_rows else None)
    _freshness(st, latest, fut, rows, live)
    if rows and rows[0].get("src") == "fyers":
        st.warning(
            "🧪 **Test data — Tradebot's Fyers capture, not NSE's feed.** Imported with "
            "`python -m eqbtst.index_fno_fyers` so the page can be tried on a real session "
            f"({rows[0]['t'][11:16]}–{rows[-1]['t'][11:16]}). Differences: 51 strikes per "
            "reading (max pain and whole-chain PCR cover those only), no bid/ask (IV from last "
            "prices), no index futures, gaps where the VM missed readings. Remove with "
            "`python -m eqbtst.index_fno_fyers remove`. Live capture is unaffected.")
    if not rows:
        st.caption("Nothing logged for this session yet (the log runs 09:15-15:30 on trading "
                   "days). The table shows NSE's latest chain; the 'since first reading' "
                   "columns and the session charts fill in once logging starts.")

    c1, c2, c3 = st.columns([1.3, 1.3, 2])
    anchor = c1.segmented_control(
        "Window", ["09:15 open (fixed)", "live ATM"], default="09:15 open (fixed)",
        key=f"if_anchor2_{code}", help="09:15 open (default): the strike nearest the index's "
        "official opening price, and 8 strikes either side, fixed for the day — the buckets "
        "(Best / Better / Good) are decided then and do not move. Example: Nifty opens at "
        "23,780 → 23,800 is the centre, 23,000-24,600 the ladder on a 100-point gap. Live ATM: "
        "the ladder re-centres on the index at every refresh (the day's lines are recomputed "
        "for those strikes).") or "09:15 open (fixed)"
    mode = c2.segmented_control(
        "Read", ["IV read", "premium read"], default="IV read", key=f"if_mode_{code}",
        help="IV read: OI change vs IV change since the first reading. Premium read: premium "
             "change vs OI change since yesterday — what most sites print.") or "IV read"
    mode = "iv" if mode == "IV read" else "premium"
    g1, g2 = st.columns([1.3, 2.3])
    lce, lpe = row_sides(latest)
    lstep = listed_step(set(lce) & set(lpe), latest.get("a"))
    gaps = ["auto", "listed"] + [s for s in (50, 100, 200) if not lstep or s > lstep]
    agap, aem, awhy = auto_gap(rows, latest)
    step = g1.segmented_control(
        "Strike gap", gaps, default="auto", key=f"if_step_{code}",
        format_func=lambda s: f"auto ({agap} pts)" if s == "auto" else
        (f"listed ({lstep:g})" if lstep else "listed") if s == "listed" else f"{s:g} pts",
        help="Which strikes make the ladder and its buckets. AUTO (default) = 100 points — "
             "positions sit on the round hundreds: on 130 sessions 100-point strikes held 70% "
             "of Nifty's OI on half the strikes (3.7M vs 1.2M per strike on the 50s between), "
             "81% on Fin Nifty, 80% on Midcap — unless 8 strikes of 100 do not reach the move "
             "the ATM straddle prices; then 200. On 87 archived sessions that switches only "
             "Bank Nifty from 3 sessions to expiry, where a 100-point ladder sat entirely "
             "inside the priced move (its 'Good' bucket reached 0.5-1.0x it). Chosen once from "
             f"the session's first straddle, so the buckets never change mid-session. Now: "
             f"{awhy}.")
    step = step if step in gaps else "auto"
    step = agap if step == "auto" else None if step == "listed" else step
    n_side = g2.select_slider(
        "Strikes each side of ATM", options=[4, 6, 8, 10, 12, 15, 20], value=BAND,
        key=f"if_nside_{code}",
        help=f"Default {BAND} (17 strikes). The log holds {KEEP} listed strikes and every "
             f"100-point strike within {LOG_GRID_N} steps either side, so a wide ladder has "
             "its history too.")
    ref = latest
    rce, rpe = row_sides(ref)
    all_ks = grid(set(rce) & set(rpe), step)
    atm_now = grid_atm(latest, step)
    o_lvl, o_src = day_open(code, day, rows)
    fixed = anchor.startswith("09:15") and o_lvl is not None
    centre = pick_atm(all_ks, o_lvl) if fixed else atm_now
    strikes = window(all_ks, centre, n_side)
    lot, lot_how = _lot(code, expiry, latest, day)
    with c3:
        st.caption(f"Lot **{lot or '—'}** ({lot_how or 'unknown'}). ₹ figures = contracts × "
                   "lot × index level (exposure, not premium).")

    h = headline(latest, first, strikes, step)
    # IV baselines: each strike's first VALID IV today (a gap open leaves the first readings
    # without one), only when the log and the latest read share an IV source
    ivbase = (baseline_iv(rows, strikes, step)
              if rows and iv_source(rows[-1]) == iv_source(latest) else None)
    if ivbase is not None:
        h["atm_iv0"] = ivbase["atm"]
    m = st.columns(7)
    m[0].metric(f"{DISPLAY[code]}", _fmt_n(h["spot"], "{:,.2f}"),
                help="Index level in the same NSE response as the chain.")
    fwd_txt = None
    if h["fwd"] and h["spot"]:
        fwd_txt = f"{(h['fwd'] / h['spot'] - 1) * 1e4:+.0f} bps vs spot"
    m[1].metric("ATM strike", _fmt_n(h["atm"]), fwd_txt, delta_color="off",
                help="ATM (At The Money) = listed strike nearest the index (the convention on "
                     "NSE's own chain). Below it: the forward implied by put-call parity on the "
                     "7 strikes around ATM vs the index, in basis points — the carry to expiry. "
                     "On a far monthly this is 2-3 strikes: Bank Nifty Dec forward ~59,350 vs "
                     "spot 59,086 on 6 Nov.")
    m[2].metric("ATM straddle", _fmt_n(h["straddle"], "₹{:,.1f}"),
                (f"±{h['move_pct']:.2f}% priced to expiry" if h["move_pct"] else None),
                delta_color="off",
                help="ATM call + ATM put premium (quote mids when tight). It is about the "
                     "AVERAGE absolute move the market is paying for between now and expiry "
                     "(≈0.8 of one standard deviation; one standard deviation ≈ 1.25 × "
                     "straddle). Example: ₹350 on Nifty 25,000 = ±1.4% average move.")
    m[3].metric("ATM IV", _fmt_n(h["atm_iv"], "{:.2f}%"),
                (f"{h['atm_iv'] - h['atm_iv0']:+.2f} pts since first reading"
                 if h["atm_iv"] and h["atm_iv0"] else None), delta_color="off",
                help="Average of the ATM call's and put's IV (Implied Volatility), computed "
                     "from the parity forward (see the IV column's note). On expiry day it "
                     "swings wildly in the last hour: very little time is left to divide by.")
    m[4].metric(f"PCR ({len(strikes)} strikes)", _fmt_n(h["pcr_band"], "{:.2f}"),
                (f"whole chain {h['pcr_all']:.2f}" if h["pcr_all"] else None), delta_color="off",
                help="PCR (Put-Call Ratio) = put OI ÷ call OI. Above 1 = more puts open than "
                     "calls. Measured on these 17 strikes and on every strike of the expiry.")
    m[5].metric("Max pain", _fmt_n(h["mp"]),
                (f"{h['mp'] - h['spot']:+,.0f} pts from index" if h["mp"] and h["spot"] else None),
                delta_color="off",
                help="Settlement strike at which option buyers collect least, over every "
                     "strike of this expiry. Tradebot's test: the index does NOT pin to it "
                     "(49% vs 50% by chance). Shown because people watch it.")
    m[6].metric("Put skew", _fmt_n(h["skew"], "{:+.2f} pts"),
                help="IV of the put 4 strikes below ATM minus IV of the call 4 strikes above. "
                     "Positive = downside protection costs more than upside calls (normal for "
                     "an index). A rising skew = demand for puts.")
    st.markdown(
        f"Largest OI in the window: calls **{_fmt_n(h['ce_oi'])}**, puts **{_fmt_n(h['pe_oi'])}**"
        f" · most added today: calls **{_fmt_n(h['ce_add'])}**, puts **{_fmt_n(h['pe_add'])}**")

    if latest.get("t", "")[11:16] > "15:30":
        st.caption("🕞 **After the bell.** The index value is NSE's official close (built from "
                   "15:00-15:30 trading) while premiums are last trades, so NSE's own IV is "
                   "distorted now (6 Nov: calls 5-9 points below puts at the same strike). "
                   "IV on this page is computed from the options' own parity forward and "
                   "does not have that problem. Open interest after 15:40 is NSE's corrected, "
                   "official figure.")
    if fixed:
        steps = (all_ks.index(atm_now) - all_ks.index(centre)
                 if atm_now in all_ks and centre in all_ks else None)
        where = bucket_map(strikes, centre).get(atm_now)
        msg = (f"📍 **Ladder fixed at the 09:15 open**: {DISPLAY[code]} opened at **{o_lvl:,.2f}** "
               f"({o_src}) → centre strike **{centre:,.0f}**, {len(strikes)} strikes, buckets fixed "
               f"for the day. Now {latest.get('u', 0):,.2f} → live ATM {atm_now:,.0f}"
               + (f", {abs(steps)} strike{'s' if abs(steps) != 1 else ''} "
                  f"{'above' if steps > 0 else 'below'} the open strike" if steps else
                  ", at the open strike" if steps == 0 else "")
               + (f" (in **{where}**)." if where else "."))
        if where is None:
            st.warning(msg + " The index has moved OFF the morning ladder — its live ATM is not "
                       "among these strikes. Switch Window to 'live ATM' to follow it.")
        else:
            st.caption(msg)
    elif anchor.startswith("09:15"):
        st.warning(f"⏱️ Opening level not available ({o_src}), so the ladder follows the live "
                   "ATM. Past days use NSE's end-of-day open once DCM has loaded it.")
    tab = ladder(latest, first, strikes, mode, atm_now, ivbase)
    center = centre
    tab["Bucket"] = tab["Strike"].map(bucket_map(strikes, center))
    # collapsed by default: the bucket drill-down below carries the per-strike story; the
    # full 17-strike table opens on demand
    with st.expander(f"📋 The {len(strikes)} strikes — calls left, puts right", expanded=False):
        _ladder_view(st, tab, latest, mode)
    rs = reconcile_summary(code)
    if rs and rs.get("sign") is not None:
        st.caption(
            f"📏 **How far to trust live OI** — last {rs['days']} logged day(s) "
            f"({rs['first']} → {rs['last']}), nearest expiry, live 15:2x reading vs NSE's "
            f"corrected close: change-on-the-day had the same sign on **{rs['sign']:.0f}%** of "
            f"contracts; median miss {rs['err']:.1f}% of OI; live OI high by "
            f"{rs['bias_ce']:+.1f}% (calls) / {rs['bias_pe']:+.1f}% (puts).")
    else:
        st.caption("📏 Live OI is **provisional**: NSE corrects it after the bell. 6 Nov 2026, "
                   "Nifty 09-Nov chain: whole-chain call OI −2.6% and put OI −2.2% between "
                   "15:37 and the 15:40 official figure. A nightly score of live vs official "
                   "appears here once a full day is logged.")

    df = frame(rows)
    _bucket_section(st, code, latest, first, df, strikes, center, lot, h.get("move_pct"))
    st.subheader(f"Calls vs puts in these {len(strikes)} strikes through the session")
    base = st.segmented_control(
        "Count from", ["since yesterday's close", "since this morning's first reading"],
        default="since yesterday's close", key=f"if_base_{code}") or "since yesterday's close"
    ss = session_series(df, rows, strikes, lot, "first" if base.startswith("since this") else "yday",
                        step)
    if not ss.empty:
        unit = "₹ Cr" if lot else "contracts"
        _index_line(st, ss["spot"], DISPLAY[code], o_lvl)
        _lines(st, ss[["calls", "puts"]].rename(columns={"calls": "calls opened",
                                                         "puts": "puts opened"}),
               f"OI change, {unit}", {"calls opened": CE_COLOR, "puts opened": PE_COLOR},
               zero=True)
        _lines(st, ss[["calls − puts"]], f"calls − puts, {unit} (↑ call side heavier)",
               {"calls − puts": "#9aa4b2"}, height=170, zero=True)
        st.caption(f"Same {len(strikes)} strikes at every reading (the window above), so a line moves when "
                   "positions move, not when the ATM drifts. A reading is plotted only when "
                   f"{SERIES_QUORUM:.0%} of those strikes were in it. Calls opened is NOT "
                   "bullish by itself — writers open contracts too; the reads in the table "
                   "and the heatmap say which side.")
    else:
        st.caption("Session lines appear once a few readings are logged.")

    st.subheader("Strike by strike — calls vs puts through the session")
    k1, k3, k2 = st.columns([3, 2, 1])
    metric = k1.segmented_control("Show", list(METRICS), default="OI change since yesterday",
                                  key=f"if_metric_{code}") or "OI change since yesterday"
    is_vol = metric.startswith("Volume")
    # volume has no 'each reading' (a 1-minute difference of a cumulative count is mostly
    # noise and NSE's stamp steps ~1 min), so it gets its own key and options
    pblk = k3.segmented_control(
        "Time block", list(BLOCKS) if is_vol else ["each reading", *BLOCKS], default=5,
        key=f"if_pblk_{code}" + ("_v" if is_vol else ""),
        format_func=lambda x: x if isinstance(x, str) else ("1 hr" if x == 60 else f"{x} min"),
        help="Blocks count from 09:15 (an hour = 09:15-10:15 ..., as NSE's hourly candles). "
             "Lines show the LAST reading of each block, so the block still running ends at "
             "now. Volume = contracts traded in the block. 'Each reading' = every ~1-min "
             "read, nothing smoothed.") or 5
    pmin = None if isinstance(pblk, str) else int(pblk)
    shared = k2.toggle("Same scale on every strike", value=True, key=f"if_shared_{code}",
                       help="On: panels compare with each other. Off: each panel zooms to its "
                            "own range (small strikes become readable, sizes no longer compare).")
    _panels_chart(st, strike_panels(df, strikes, metric, pmin), strikes, atm_now, metric,
                  shared, pmin)

    st.subheader("Where positions were opened and closed — by block")
    mins = st.segmented_control("Time block", list(BLOCKS), default=15, key=f"if_blk_{code}",
                                format_func=lambda x: "1 hr" if x == 60 else f"{x} min",
                                help="Blocks count from 09:15: an hour = 09:15-10:15, "
                                     "10:15-11:15 ... 15:15-15:30.") or 15
    _heatmap(st, oi_blocks(df, strikes, mins), strikes, atm_now, mins)
    st.caption("Each cell = contracts opened (blue) or closed (red) at that strike in that "
               "block. The first block counts from yesterday's close, so the opening rush is "
               "in it; each row sums to the strike's change at the last reading.")

    st.subheader("Open interest now")
    _profile(st, latest, strikes, atm_now)

    st.subheader("Volatility")
    v1, v2 = st.columns(2)
    with v1:
        st.caption("IV across strikes — now vs the first reading (dashed)")
        _smile(st, latest, first, strikes, atm_now, ivbase)
    with v2:
        if not ss.empty:
            st.caption("ATM IV and ATM straddle through the session")
            _lines(st, ss[["atm iv"]].rename(columns={"atm iv": "ATM IV %"}), "IV %",
                   height=120, fmt=".2f")
            _lines(st, ss[["straddle"]].rename(columns={"straddle": "ATM straddle ₹"}), "₹",
                   height=120, fmt=",.1f")
    if not ss.empty:
        p1, p2 = st.columns(2)
        with p1:
            st.caption("PCR (Put-Call Ratio) through the session")
            _panel_lines(st, ss[["pcr_band", "pcr_all"]].rename(
                columns={"pcr_band": f"{len(strikes)} strikes", "pcr_all": "whole chain"}),
                "put ÷ call OI", height=160, fmt=".2f", width=230)
        with p2:
            st.caption("Max pain vs the index")
            _lines(st, ss[["max pain", "spot"]].rename(columns={"spot": "index"}), "level",
                   height=170, fmt=",.0f")

    st.subheader("Index futures")
    fs = fut_series(fut_rows)
    official = (load_close(code, day).get("fut") or None)
    if official and (not fut or official.get("t", "") >= fut.get("t", "")):
        fut = official
    is_official = bool(fut) and fut.get("t", "")[11:16] >= CLOSE_FROM.strftime("%H:%M")
    if fut:
        st.markdown("**Official close (NSE's corrected 15:40 figures).**" if is_official else
                    "**Provisional — live open interest.** On 6 Nov 2026 the live Nifty "
                    "futures OI read +8,927 contracts on the day at 15:37; NSE's official "
                    "close was −8,509 (Bank Nifty +4,891 live vs −3,487). Price and premium "
                    "are fine; the OI change and its word can be the WRONG SIGN until 15:40.")
        F = fut.get("F") or {}
        if F:
            ft = pd.DataFrame([dict(expiry=k, last=v[0], prev=v[1], oi=v[2], doi=v[3], vol=v[4])
                               for k, v in F.items()])
            ft["chg %"] = (ft["last"] / ft["prev"] - 1) * 100
            ft["OI chg %"] = ft["doi"] / (ft["oi"] - ft["doi"]) * 100
            ft["premium bps"] = (ft["last"] / fut.get("s") - 1) * 1e4 if fut.get("s") else None
            ft["read"] = [(fut_word(c, d) or "—") + ("" if is_official or not fut_word(c, d)
                                                     else " (provisional)")
                          for c, d in zip(ft["last"] - ft["prev"], ft["doi"])]
            st.dataframe(ft[["expiry", "last", "chg %", "premium bps", "oi", "doi", "OI chg %",
                             "vol", "read"]], hide_index=True, width="stretch",
                         column_config={
                             "oi": st.column_config.NumberColumn("OI", format="%,d",
                                                                 help="Futures OI, contracts."),
                             "doi": st.column_config.NumberColumn("ΔOI", format="%+,d",
                                                                  help="Change vs yesterday."),
                             "vol": st.column_config.NumberColumn("Volume", format="%,d"),
                             "chg %": st.column_config.NumberColumn(format="%+.2f"),
                             "OI chg %": st.column_config.NumberColumn(format="%+.2f"),
                             "premium bps": st.column_config.NumberColumn(
                                 format="%+.0f", help="Futures over the index, basis points."),
                             "read": st.column_config.TextColumn(
                                 help="Price change × OI change vs yesterday: long build "
                                      "(up, OI up), short build (down, OI up), short covering "
                                      "(up, OI down), long unwind (down, OI down).")})
    if not fs.empty:
        f1, f2 = st.columns(2)
        with f1:
            _lines(st, fs[["premium bps"]], "futures premium, bps", height=170, fmt="+.0f")
        with f2:
            _lines(st, fs[["oi %"]].rename(columns={"oi %": "futures OI vs yesterday %"}),
                   "OI change %", height=170, zero=True, fmt="+.2f")
    rs = reconcile_summary(code)
    if rs and rs.get("fut_n"):
        st.caption(f"📏 Live futures OI change vs official, last {rs['fut_n']} logged day(s): "
                   f"same sign on {rs['fut_sign'] or 0:.0f}% of days.")
    st.caption("Checked 6 Nov 2026: NSE's 15:40 futures and option figures equal the evening "
               "bhavcopy exactly (OI and volume, all four indices). Before 15:40 the live OI is "
               "NSE's running estimate and is corrected after the bell.")


_RECON_DAY: list = [None]


def render_page(st) -> None:
    """The Index F&O Management lane. `st` is injected (this module never imports streamlit)."""
    st.title("📊 Index F&O Management — the option chain, strike by strike")
    st.markdown("🧭 **No arrow on this page.** It shows where calls and puts are being opened "
                "and closed on the 17 strikes around the money, live from NSE. It does not "
                "forecast — see 'How to read this page'.")
    with st.expander("How to read this page"):
        st.markdown(HELP)
    ensure_poller()
    if _RECON_DAY[0] != _now().date():   # cheap (a glob); scores any finished day once
        _RECON_DAY[0] = _now().date()
        try:
            reconcile()
        except Exception as exc:                             # noqa: BLE001
            _log_error("recon", f"{type(exc).__name__}: {exc}")
    code = st.segmented_control("Index", list(CODES), default="NIFTY",
                                format_func=lambda c: DISPLAY[c], key="if_index") or "NIFTY"
    today = _now().date()
    days = [d for d in logged_days(code) if d != today]
    opts = ["live"] + days
    pick = st.selectbox("Session", opts, index=0, key=f"if_day_{code}",
                        format_func=lambda d: "Today (live)" if d == "live"
                        else f"{d:%a %d %b %Y} (logged)")
    live = pick == "live"
    day = today if live else pick
    if live:
        touch(code)
        lat = load_latest(code)
        exps = [dt.date.fromisoformat(e) for e in lat.get("expiries") or []]
        if not exps and _claim():
            with st.spinner("Reading NSE…"):
                exps = poll_expiries(code)
        cap = captured_expiries(exps, today)
        if not cap:
            cap = [dt.date.fromisoformat(e) for e in logged_expiries(code, today)]
        if not cap:
            @st.fragment(run_every="5s")
            def _wait():                 # re-checks by itself; the writer may be elsewhere
                touch(code)
                if load_latest(code).get("expiries"):
                    st.rerun()
                st.info("Waiting for NSE's expiry list — the capture asks NSE within a few "
                        "seconds. This refreshes by itself.")
            _wait()
            return
        later = [e for e in exps if e >= today and e not in cap][:6]
        choices = cap + later
    else:
        choices = [dt.date.fromisoformat(e) for e in logged_expiries(code, day)]
        exps = choices
        if not choices:
            st.info("Nothing logged for that session.")
            return
    exp = st.selectbox("Expiry", choices, index=0, key=f"if_exp_{code}_{pick}",
                       format_func=lambda e: _expiry_label(e, exps or choices, day)
                       + ("" if not live or e in captured_expiries(exps, today)
                          else " · read only while open on screen"))
    if live:
        touch(code, exp)
        if exp.isoformat() not in (load_latest(code).get("chain") or {}) and _claim():
            with st.spinner("Reading the chain…"):
                poll_chain(code, exp)

    if live:
        @st.fragment(run_every="15s")
        def _live():
            try:
                touch(code, exp)
                _draw(st, code, exp.isoformat(), day, True)
            except Exception as exc:                         # noqa: BLE001
                st.error(f"Index F&O could not draw this refresh: {type(exc).__name__}: {exc}")
        _live()
    else:
        @st.fragment
        def _past():
            try:
                _draw(st, code, exp.isoformat(), day, False)
            except Exception as exc:                         # noqa: BLE001
                st.error(f"Could not rebuild {day:%d %b}: {type(exc).__name__}: {exc}")
        _past()


def capture(until: dt.time = dt.time(15, 45)) -> None:
    """Headless capture (python -m eqbtst.index_fno capture). Normally not needed: the
    dashboard starts this poller, and index_mgmt's scheduled capture starts it too."""
    print(f"[{_now():%d %b %H:%M:%S}] index F&O capture starting (pid {os.getpid()})", flush=True)
    try:
        compress_old_logs()
        reconcile()
    except Exception:                                        # noqa: BLE001
        pass
    ensure_poller()
    try:
        while _now().time() < until and _now().weekday() < 5:
            time.sleep(30)
    finally:
        stop_poller()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        capture()
