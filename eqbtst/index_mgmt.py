"""
index_mgmt.py — Index Management: who is moving the index RIGHT NOW (Top / Next / Rest).

The live twin of Daily_Cash_Market's "Index & Large Cap" tab. DCM answers "who carried the
index today" after the close, from the bhavcopy. This lane answers it during the session.

SOURCE: NSE'S OWN WEBSITE FEED, NOT FYERS — ZERO BROKER CALLS
    Tradebot already spends ~4,237 of the 5,000 daily Fyers data calls. This lane makes none.
    Probed live on 13 Oct 2026 (all four indices, market open):
      /api/equity-stock-indices?index=…          every constituent: ltp, prev close, traded value
      …indexTrackerApi?functionName=getConstituents   every constituent's INDEX WEIGHT
      …GetQuoteApi?functionName=getSymbolDerivativesData   one stock: every futures and option
                                                  contract with OI and its change vs yesterday
    The feed runs 1-2 minutes behind the tape (stock rows stamped 15:15:12 in a 15:16:27
    response). That is fine for a decomposition and is printed on the page.

WHAT IS EXACT HERE THAT DCM CANNOT HAVE
    DCM holds no weights, so its buckets are equal-weighted and "never index points". NSE
    publishes the weights live, and  weight x stock return x index previous close  rebuilds
    the index move to under half a point on every index (13 Oct 2026 15:16):
        Nifty 50   +338.78 rebuilt vs +338.70 actual     Bank Nifty  +548.09 vs +547.65
        Fin Nifty  +365.88 vs +365.70                    Midcap Sel.  +96.96 vs  +97.00
    So the weights behave as PREVIOUS-CLOSE weights (de-drifting them for today's move is
    worse: 333.68) and each bucket's contribution is shown in real index points. The gap
    between rebuilt and actual is printed every refresh — it is the self-check.
    Free-float cap share is NOT a substitute: on the capped indices it is off by up to
    12.8pp (Bank Nifty), because NSE applies capping factors. Only `weightage` is used.

WHAT IS REUSED FROM DCM, NOT COPIED
    Bucket membership, the OI-price quadrant, the bucket lean and its quorum, the options
    read and its expiry-cycle norm, the four-state partition and the regime sentence are all
    CALLED from DCM's index_largecap module (read-only, CLOUD_MODE). sector_tilt.py was a
    hand-ported copy and drifted from upstream twice; this lane cannot drift. If DCM is not
    importable the price decomposition still renders (buckets ranked by live weight) and the
    futures/options words are withheld rather than re-implemented.

WHAT IS PROVISIONAL UNTIL THE CLOSE — READ THIS BEFORE ACTING ON AN OI WORD
    1. Intraday open interest includes positions that will be squared off before 15:20. A
       "long build" at noon can be gone by the close; DCM's numbers are closing numbers.
    2. The options read subtracts what is NORMAL for this point in the expiry cycle. That
       norm was measured on full sessions. Here it is pro-rated by the share of a normal
       day's volume traded so far. Reasonable, NOT measured — no intraday stock OI history
       existed before this module started logging one (data/index_mgmt/).
    3. Options are switched OFF on a monthly expiry day, exactly as on DCM: expiring calls
       fall away faster than puts and the read comes out bullish 72-86% of the time.
    4. NOTHING HERE FORECASTS. DCM tested the end-of-day version of every column over 1,170+
       sessions and 200 candidates; nothing survives the index's own close-strength. An
       intraday version of a null is not an edge. This is a map of the move, not a signal.
"""
from __future__ import annotations

import contextlib
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
import requests

from . import config

try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
except Exception:                                            # noqa: BLE001
    _IST = None

# fno symbol (DCM's key) -> NSE's index SYMBOL. The long names 404 or return an empty list:
# "NIFTY FINANCIAL SERVICES" and "NIFTY MIDCAP SELECT" both answered {"data": []} with a 200,
# which reads as "no constituents" rather than "wrong name". These four are the ones that quote.
NSE_NAME = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "FINNIFTY": "NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NIFTY MID SELECT",
}
DISPLAY = {"NIFTY": "Nifty 50", "BANKNIFTY": "Bank Nifty",
           "FINNIFTY": "Fin Nifty", "MIDCPNIFTY": "Midcap Select"}
# Bucket sizes used ONLY when DCM's dated membership is unreachable; then names are ranked
# by live weight and the page says so.
_FALLBACK_CUTS = {"NIFTY": (10, 10), "BANKNIFTY": (2, 3),
                  "FINNIFTY": (2, 4), "MIDCPNIFTY": (2, 5)}

_BASE = "https://www.nseindia.com"
_PRIME = _BASE + "/market-data/live-equity-market"
_TRACKER = "/api/NextApi/apiClient/indexTrackerApi?functionName="
_DERIV = "/api/NextApi/apiClient/GetQuoteApi?functionName=getSymbolDerivativesData&symbol="
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": _PRIME,
}

CASH_EVERY = 30.0        # seconds between price polls of one index while NSE says Open
# Seconds between OI sweeps. Measured 13 Oct 2026, 15:24:48 -> 15:26:09 (81s apart): futures
# OI had changed on 48 of 50 stocks and options OI on 50 of 50, so NSE refreshes OI at least
# that often. 120s keeps a 50-stock sweep (~15s, 50 calls) to one call every ~2.4s on
# average. The true refresh interval is not yet measured — run
# `python -m eqbtst.index_mgmt cadence` in market hours and set this from what it prints.
DERIV_EVERY = 120.0
IDLE_EVERY = 900.0       # price poll when the market is shut (also catches special sessions)
WEIGHTS_EVERY = 1800.0   # weights are previous-close weights; they do not move intraday
VIEW_TTL = 600.0         # an index nobody has looked at for this long stops being swept for OI
STALE_AFTER = 240.0      # live feed older than this while NSE says Open = said out loud
_OI_CLIP = 50.0          # one thin contract can print +895%; same clip DCM uses

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "index_mgmt"


def _now() -> dt.datetime:
    return dt.datetime.now(_IST).replace(tzinfo=None) if _IST else dt.datetime.now()


def _num(v):
    """A finite float or None. NSE sends numbers, numeric strings, '-' and null."""
    try:
        f = float(str(v).replace(",", "")) if not isinstance(v, (int, float)) else float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _ts(s) -> dt.datetime | None:
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(str(s).strip(), fmt)
        except (TypeError, ValueError):
            continue
    return None


# ── NSE transport ─────────────────────────────────────────────────────────────────────
class NseError(RuntimeError):
    pass


_PRIME_URLS = (_PRIME, _BASE + "/", _BASE + "/option-chain")
BREAKER_MAX = 120.0      # longest pause after repeated failures, seconds


def _family(path: str) -> str:
    return ("deriv" if "GetQuoteApi" in path else "tracker" if "indexTracker" in path
            else "cash")


class NseClient:
    """One primed session, paced, with a circuit breaker PER ENDPOINT FAMILY.

    NSE sits behind Akamai: the API host answers only to a session that has loaded a page
    first, and the cookies expire. So a failure re-primes and retries once.

    15 Oct 2026, three flaws that together froze prices for 17 minutes while open interest
    kept updating — fixed here:
      1. A failed PRIME blocked every call. The re-prime loads an HTML page; when that page
         refused the connection, the session was never replaced, every later call tried to
         prime first, and every call failed — even endpoints that were answering. Now a
         failed prime tries other pages, then keeps the old session's cookies and lets the
         API call itself decide.
      2. One breaker for everything, escalating 1 -> 2 -> 4 -> 10 minutes. A short blip
         became a 10-minute blackout, renewed on the next failed attempt. Now each family
         (prices / weights / derivatives) has its own breaker, capped at BREAKER_MAX.
      3. Nothing could reset it: see request_refresh() and the poller's self-heal.
    """

    def __init__(self, session=None, min_gap: float = 0.22):
        self._s = session
        self._primed = session is not None
        self._gap = min_gap
        self._last = 0.0
        self._lock = threading.Lock()
        self.fails: dict = {}
        self.blocked: dict = {}
        self.last_error: str | None = None

    @property
    def blocked_until(self) -> float:
        return max(self.blocked.values(), default=0.0)

    def _prime(self) -> None:
        s = requests.Session()
        s.headers.update(_HEADERS)
        for url in _PRIME_URLS:
            try:
                s.get(url, timeout=15)
                if s.cookies:
                    self._s, self._primed = s, True
                    return
            except Exception:                                # noqa: BLE001
                continue
        # every priming page refused: keep the old session (its cookies may still be good)
        # and let the API answer for itself instead of failing without asking
        if self._s is None:
            self._s = s
        self._primed = True

    def _once(self, path: str):
        if not self._primed:
            self._prime()
        wait = self._gap - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        r = self._s.get(_BASE + path, timeout=20)
        if r.status_code != 200:
            raise NseError(f"HTTP {r.status_code}")
        if r.text.lstrip()[:1] not in ("{", "["):
            raise NseError("HTML instead of JSON (session not accepted)")
        return r.json()

    def get(self, path: str):
        fam = _family(path)
        with self._lock:
            if time.time() < self.blocked.get(fam, 0.0):
                raise NseError(f"paused {self.blocked[fam] - time.time():.0f}s more after "
                               f"repeated failures ({self.last_error})")
            try:
                try:
                    out = self._once(path)
                except Exception:                            # noqa: BLE001
                    self._primed = False                     # stale cookies: one fresh try
                    out = self._once(path)
            except Exception as exc:                         # noqa: BLE001
                n = self.fails.get(fam, 0) + 1
                self.fails[fam] = n
                self.last_error = f"{type(exc).__name__}: {str(exc)[:120]}"
                if n >= 3:
                    self.blocked[fam] = time.time() + min(30.0 * 2 ** (n - 3), BREAKER_MAX)
                _log_error(fam, self.last_error)
                raise NseError(self.last_error) from exc
            self.fails[fam], self.blocked[fam] = 0, 0.0
            self.last_error = None
            return out


def _log_error(family: str, msg: str) -> None:
    """One line per failed NSE call, so a stale spell can be explained afterwards."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with (DATA_DIR / "errors.log").open("a", encoding="utf-8") as f:
            f.write(f"{_now():%Y-%m-%d %H:%M:%S} {family:7s} {msg}\n")
    except OSError:
        pass


def fetch_cash(code: str, client) -> dict:
    """Index row + every constituent, from one call. Raises NseError on anything unusable."""
    name = NSE_NAME[code]
    j = client.get("/api/equity-stock-indices?index=" + quote(name))
    rows = j.get("data") if isinstance(j, dict) else None
    if not rows:
        raise NseError(f"no constituents returned for {name}")
    idx, stocks = None, {}
    for r in rows:
        sym = str(r.get("symbol") or "")
        ltp, prev = _num(r.get("lastPrice")), _num(r.get("previousClose"))
        if sym == name or r.get("priority") == 1:
            idx = dict(last=ltp, prev=prev, open=_num(r.get("open")),
                       high=_num(r.get("dayHigh")), low=_num(r.get("dayLow")))
            continue
        if not sym or not ltp or ltp <= 0 or not prev or prev <= 0:
            continue                                # a zero price is not a price
        stocks[sym] = dict(ltp=ltp, prev=prev, open=_num(r.get("open")),
                           pct=(ltp / prev - 1) * 100,
                           value=_num(r.get("totalTradedValue")),
                           upd=str(r.get("lastUpdateTime") or ""))
    if idx is None or not idx["last"] or not idx["prev"] or not stocks:
        raise NseError(f"{name}: index row or constituents unusable")
    return dict(code=code, ts=_ts(j.get("timestamp")),
                status=str((j.get("marketStatus") or {}).get("marketStatus") or ""),
                index=idx, stocks=stocks, fetched=_now())


def fetch_weights(code: str, client) -> dict:
    """{symbol: index weight %}. Raises unless the list sums to ~100."""
    j = client.get(_TRACKER + "getConstituents&&index=" + quote(NSE_NAME[code])
                   + "&&noofrecords=0")
    w = {str(r.get("cmSymbol")): _num(r.get("weightage"))
         for r in (j.get("data") or []) if r.get("cmSymbol")}
    w = {k: v for k, v in w.items() if v is not None and v > 0}
    if not w or abs(sum(w.values()) - 100.0) > 1.0:
        raise NseError(f"weights unusable (n={len(w)}, sum={sum(w.values()):.2f})")
    return w


def fetch_deriv(symbol: str, client) -> dict:
    """One stock's futures / call / put OI summed over EVERY live contract.

    Each contract carries its own change against yesterday, so a position rolling from this
    month to the next nets to zero in the sum — the same roll-immune total DCM uses (the
    front contract alone sheds 37.7% a day at two days to expiry). The contract settling
    today is included, which is DCM's settlement-day basis and is continuous through it.
    """
    j = client.get(_DERIV + quote(symbol, safe=""))
    rows = j.get("data") if isinstance(j, dict) else None
    if not rows:
        raise NseError(f"{symbol}: no derivative contracts")
    out = dict(fut_oi=0.0, fut_chg=0.0, ce_oi=0.0, ce_chg=0.0, pe_oi=0.0, pe_chg=0.0)
    exps, n_fut, fut_prev = set(), 0, {}
    # every option contract whose OI moved since yesterday: (side, strike, expiry, last,
    # previous close, OI change) — what option_flow() judges buyers and sellers on
    opts: list = []
    # per expiry, for the log: F = futures [OI, change, volume, last price];
    # C / P = calls / puts summed over strikes [OI, change, volume]
    by_exp: dict = {"F": {}, "C": {}, "P": {}}
    for r in rows:
        oi, chg = _num(r.get("openInterest")), _num(r.get("changeinOpenInterest"))
        if oi is None or chg is None:
            continue
        e = _ts(str(r.get("expiryDate")) + " 00:00")
        if e:
            exps.add(e.date())
        kind, side = str(r.get("instrumentType") or ""), str(r.get("optionType") or "")
        key = ("fut" if kind.startswith("FUT") else
               "ce" if side == "CE" else "pe" if side == "PE" else None)
        if key is None:
            continue
        n_fut += key == "fut"
        if key == "fut" and e:
            fut_prev[e.date()] = oi - chg
        if e:
            k = e.date().isoformat()
            vol = _num(r.get("totalTradedVolume")) or 0.0
            if key == "fut":
                # + previous close: NSE's prevClose == the archive's close_price exactly
                # (checked 16 Oct 2026 on RELIANCE / HDFCBANK / M&M, all three months)
                by_exp["F"][k] = [oi, chg, vol, _num(r.get("lastPrice")),
                                  _num(r.get("prevClose"))]
            else:
                cur = by_exp["C" if key == "ce" else "P"].setdefault(k, [0.0, 0.0, 0.0])
                cur[0] += oi
                cur[1] += chg
                cur[2] += vol
                if chg:
                    opts.append(("CE" if key == "ce" else "PE", _num(r.get("strikePrice")), k,
                                 _num(r.get("lastPrice")), _num(r.get("prevClose")), chg, vol))
        out[key + "_oi"] += oi
        out[key + "_chg"] += chg
    if not n_fut:
        raise NseError(f"{symbol}: no futures contract in the payload")
    out["expiries"] = sorted(exps)
    out["fut_prev"] = fut_prev
    # the cash price NSE quotes IN THE SAME PAYLOAD: the futures premium must compare two
    # prices taken at one moment, not a futures print against a cash row a minute older
    by_exp["S"] = _num(rows[0].get("underlyingValue"))
    out["by_exp"] = by_exp
    out["opts"] = [o for o in opts if o[1]]
    out["ts"] = _ts(j.get("timestamp"))
    return out


# ── DCM bridge (read-only; never a copy) ──────────────────────────────────────────────
_DCM: dict = {"mod": None, "tried": 0.0}


def _dcm():
    """DCM's index_largecap module, or None. CLOUD_MODE is set BEFORE the import: without
    it DCM opens DuckDB read-write and takes the exclusive lock on the archive (see fno.py)."""
    if _DCM["mod"] is not None or time.time() - _DCM["tried"] < 300:
        return _DCM["mod"]
    _DCM["tried"] = time.time()
    os.environ.setdefault("CLOUD_MODE", "true")
    root = config.DCM_DUCKDB.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from src.analytics import index_largecap as m
        _DCM["mod"] = m
    except Exception:                                        # noqa: BLE001
        _DCM["mod"] = None
    return _DCM["mod"]


def _sessions_left(day: dt.date, expiry: dt.date) -> int | None:
    """Trading sessions after `day` up to and including `expiry`, holiday-aware via DCM."""
    try:
        from src.analytics.fno_stocks import _sessions_until
        return int(_sessions_until(day, expiry))
    except Exception:                                        # noqa: BLE001
        return None


def resolve_buckets(code: str, weights: dict, live_syms, day: dt.date, dcm=None) -> dict:
    """Top / Next / Rest for `day`, reconciled against what NSE lists RIGHT NOW.

    NSE's live list is the authority on MEMBERSHIP; DCM's dated basket decides the TIER.
    A reconstitution therefore cannot silently break the page: a name NSE lists that DCM
    does not know lands in the tail bucket and is reported, and a DCM name NSE no longer
    lists is dropped and reported. `misranked` is information, not an error — DCM cuts
    Nifty on NSE's 6-month-average ranking precisely so a week's prices do not reshuffle it.
    """
    live = [s for s in live_syms]
    meta = None
    if dcm is not None:
        try:
            meta = dcm.index_meta(code, day)
        except Exception:                                    # noqa: BLE001
            meta = None
    if meta:
        buckets = {lbl: [s for s in mem if s in live] for lbl, mem in meta["buckets"].items()}
        known = {s for mem in meta["buckets"].values() for s in mem}
        added = [s for s in live if s not in known]
        dropped = sorted(known - set(live))
        tail = list(buckets)[-1]
        buckets[tail] = buckets[tail] + added
        # rename the labels if membership changed the counts, so "Rest 30" never holds 31
        out, sizes = {}, {}
        for lbl, mem in buckets.items():
            word = lbl.split()[0]
            new = f"{word} {len(mem)}" if (added or dropped) else lbl
            out[new], sizes[new] = mem, len(meta["buckets"][lbl])
        source = meta.get("source") or "DCM basket"
        band, scale = meta.get("carry_band", 0.15), meta.get("opt_cycle_scale", 1.0)
    else:
        a, b = _FALLBACK_CUTS[code]
        ranked = sorted(live, key=lambda s: -(weights.get(s) or 0.0))
        out = {f"Top {a}": ranked[:a], f"Next {b}": ranked[a:a + b],
               f"Rest {max(len(ranked) - a - b, 0)}": ranked[a + b:]}
        sizes = {k: len(v) for k, v in out.items()}
        added, dropped, band, scale = [], [], 0.15, 1.0
        source = "ranked by live NSE weight (DCM basket unreachable)"
    misranked = []
    if weights and meta:
        labels = list(out)
        for hi, lo in zip(labels[:-1], labels[1:]):
            floor = min((weights.get(s, 0.0) for s in out[hi]), default=0.0)
            for s in out[lo]:
                if weights.get(s, 0.0) > floor + 0.05:
                    misranked.append(f"{s} {weights[s]:.2f}% sits in {lo}, above the "
                                     f"lightest {hi} name ({floor:.2f}%)")
    return dict(buckets=out, sizes=sizes, source=source, added=added, dropped=dropped,
                misranked=misranked, carry_band=band, opt_scale=scale, from_dcm=bool(meta))


LOT_MIN_EXPIRIES = 2


def _proven_lot(pairs, known: int | None = None) -> int | None:
    """The lot size, from [(archive OI in shares, live previous OI in contracts)] per futures
    expiry — only when EVERY expiry lands on the SAME whole number, and there are at least two.

    The old rule took the MEDIAN ratio and accepted it within 0.02 of a whole number. That
    tolerance is 4% of the number line, so when the archive is a session behind, about one
    stock in 25 'proves' a lot by pure coincidence — measured 27 Oct 2026 against a stale
    23 Oct base: INFY 292 (true 400), HCLTECH 383 (400), BAJAJ-AUTO 72 (75), YESBANK 25,800;
    0 of 4 right. With the right session each expiry's ratio is exact, so requiring all of
    them to agree costs nothing on a good day and makes a stale-day coincidence (two or three
    independent 4% events landing on the SAME integer) practically impossible. Measured over
    5,153 logged stock-readings (7 days, 4 indices): right session 100% -> 99.0% proven,
    0 wrong either way; one session stale 7.4% 'proven' and ALL wrong -> 0%.

    The 1% lost is a stock showing ONE traded expiry (early morning; NSE lists only contracts
    traded today). `known` — the lot proven on an earlier day — rescues it: one ratio landing
    exactly on that one number is not a coincidence a stale base produces.

    A REVISION MAKES EXPIRIES DIFFER, LEGITIMATELY. NSE gives new contracts the new lot while
    old ones keep theirs: from the Jan-2027 series (28 Oct 2026) HDFCBANK is 650 / 650 / 750,
    INFY 400 / 400 / 500, CHOLAFIN 625 / 625 / 300. Every ratio is still whole, so that is the
    test, and the answer is the lot of the contract with the most positions. Differing lots
    pass only if a second expiry or the remembered lot backs that answer: a stale day could
    otherwise pair one unchanged far contract with one coincidence on the near one."""
    r = [(a / p, p) for a, p in pairs if a and p and p > 0]
    if len(r) == 1 and known and abs(r[0][0] - known) < 0.02:
        return int(known)
    if len(r) < LOT_MIN_EXPIRIES:
        return None
    if any(x < 1 or abs(x - round(x)) >= 0.02 for x, _ in r):
        return None
    lots = [round(x) for x, _ in r]
    lot = round(max(r, key=lambda t: t[1])[0])       # the most active contract's lot
    if len(set(lots)) > 1 and lots.count(lot) < 2 and lot != known:
        return None
    return lot


# ── options: who opened or closed, judged on IMPLIED VOLATILITY, not on premium ──────────
# A premium falls every day from time decay and moves with the stock, so "OI up, premium
# down = writing" mostly reads the calendar and the stock's move. Black-76 on the same-expiry
# FUTURE (in the same NSE payload) turns each premium into an implied volatility at
# yesterday's close and now; holding volatility fixed removes decay and the stock's move, so
# what is left — IV up or down — is whether buyers or sellers were pressing.
OPT_RATE = 0.065              # INR short rate; Black-76 discounting only, a small effect
OPT_IV_BAND = 0.005           # IV change (0.5 vol point) inside which the side is unclear
OPT_MIN_PREMIUM = 0.5         # Rs: below this a tick is too big a share of the price
OPT_QUADS = ("long build", "writing", "short covering", "long unwind")
OPT_BULL = {"CE": ("long build", "short covering"), "PE": ("writing", "long unwind")}
OPT_DETAIL_EVERY = 600.0      # seconds between per-contract snapshots in the OI log
OPT_READ_FROM = dt.date(2026, 11, 3)   # first session whose log can carry the options read
OPT_TICK = 0.05               # stock-option tick, Rs
OPT_MIN_TICKS = 2             # premium must beat its 'nothing changed' value by this many ticks
OPT_PARITY_GAP = 0.05         # call vs put IV at one strike further apart than this = stale
OPT_CHAIN_GAP = 0.15          # contract IV this far from its chain's median = stale / bad print


try:
    from scipy.special import ndtr as _ndtr
except Exception:                                            # noqa: BLE001
    _ndtr = None


def _ncdf(x):
    """Standard normal CDF, vectorised. Without SciPy: Abramowitz & Stegun 7.1.26 erf
    (max error 1.5e-7 — far below one premium tick), not a Python-level loop."""
    x = np.asarray(x, float)
    if _ndtr is not None:
        return _ndtr(x)
    z = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * y)


def b76(F, K, T, sig, call, r: float = OPT_RATE):
    """Black-76 price of options on a future: arrays in, array out."""
    F, K, T, sig = (np.asarray(a, float) for a in (F, K, T, sig))
    call = np.asarray(call, bool)
    sq = sig * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sq * sq) / sq
    d2 = d1 - sq
    df = np.exp(-r * T)
    c = df * (F * _ncdf(d1) - K * _ncdf(d2))
    p = df * (K * _ncdf(-d2) - F * _ncdf(-d1))
    return np.where(call, c, p), np.where(call, df * _ncdf(d1), df * (_ncdf(d1) - 1.0))


def implied_vol(price, F, K, T, call, r: float = OPT_RATE, lo=0.01, hi=3.0, n=60):
    """Vectorised bisection. NaN where no volatility in [lo, hi] gives the price — a premium
    at or below intrinsic value (deep in the money, stale print) or above the no-arbitrage
    bound. NaN is never guessed into a number."""
    price, F, K, T = (np.asarray(a, float) for a in (price, F, K, T))
    a, b = np.full(price.shape, lo), np.full(price.shape, hi)
    pa, _ = b76(F, K, T, a, call, r)
    pb, _ = b76(F, K, T, b, call, r)
    ok = (price > pa) & (price < pb) & (T > 0) & (F > 0) & (K > 0)
    for _ in range(n):
        m = 0.5 * (a + b)
        pm, _ = b76(F, K, T, m, call, r)
        hi_side = pm > price
        b = np.where(hi_side, m, b)
        a = np.where(hi_side, a, m)
    return np.where(ok, 0.5 * (a + b), np.nan)


def _consistent(iv, call, K, exp):
    """IVs with stale or bad prints set to NaN. A stock option's 'previous close' is its LAST
    TRADE, which can be hours older than the futures close it is priced against: on 3 Nov
    2026 INFY's 1154 call implied 19.7% and the 1154 put 43.5% from the same close — not
    possible together (put-call parity). Two checks: where a strike has both a call and a
    put, their IVs must be within OPT_PARITY_GAP; and every contract within OPT_CHAIN_GAP of
    its expiry's median IV."""
    iv = np.array(iv, float)
    exp = np.asarray(exp)
    for e in set(exp.tolist()):
        m = exp == e
        med = np.nanmedian(iv[m]) if np.isfinite(iv[m]).any() else np.nan
        if np.isfinite(med):
            iv[m & (np.abs(iv - med) > OPT_CHAIN_GAP)] = np.nan
        idx = np.where(m)[0]
        by_k: dict = {}
        for i in idx:
            by_k.setdefault(K[i], []).append(i)
        for ks in by_k.values():
            c = [i for i in ks if call[i] and np.isfinite(iv[i])]
            p = [i for i in ks if not call[i] and np.isfinite(iv[i])]
            if c and p and abs(iv[c[0]] - iv[p[0]]) > OPT_PARITY_GAP:
                iv[c[0]] = iv[p[0]] = np.nan
    return iv


_SESS_CACHE: dict = {}


def _sessions_after(day: dt.date, expiry: dt.date) -> int:
    """Trading sessions after `day` up to and including `expiry`: DCM's holiday-aware count,
    else weekdays. Cached — it is asked once per contract per sweep."""
    key = (day, expiry)
    if key not in _SESS_CACHE:
        n = _sessions_left(day, expiry)
        if n is None:
            n = int(np.busday_count(day + dt.timedelta(days=1), expiry + dt.timedelta(days=1)))
        _SESS_CACHE[key] = max(int(n), 0)
    return _SESS_CACHE[key]


def _trading_years(t0: dt.datetime, expiry: dt.date) -> float:
    """Time to expiry on the MARKET'S clock: sessions left (each 1/252 of a year) plus what is
    left of today's 09:15-15:30 session. Nights, weekends and holidays count as nothing.

    Calendar time charged a weekend's decay to Monday: measured on 2.0M archive changes, OI
    build-ups read as 'buying' on 68% of Mondays against 40% on ordinary days — a clock
    artefact. On trading time: 47% and 49%. (Days after a long holiday still lean to
    writing, 35%: the market prices part of a holiday's decay; 2% of the data.)"""
    open_, close = dt.datetime.combine(t0.date(), dt.time(9, 15)), dt.datetime.combine(
        t0.date(), dt.time(15, 30))
    left_today = min(max((close - max(t0, open_)).total_seconds(), 0.0), 22500.0) / 22500.0
    if t0.date() > expiry or (t0.date() == expiry and left_today <= 0):
        return 0.0
    return (_sessions_after(t0.date(), expiry) + left_today) / 252.0


def _prev_session(day: dt.date) -> dt.date:
    """The trading session before `day`, from the holiday calendar — NOT the archive's last
    date: when the F&O file is late (26 Oct 2026) that is two sessions back, and the option
    read would charge an extra day of decay and see buyers everywhere."""
    for back in range(1, 15):
        d = day - dt.timedelta(days=back)
        if d.weekday() < 5 and _sessions_after(d - dt.timedelta(days=1), d) == 1:
            return d
    return day - dt.timedelta(days=1)


def _years(t0: dt.datetime, expiry: dt.date) -> float:
    """Calendar time to 15:30 on the expiry day, in years (the Black-76 convention)."""
    return max((dt.datetime.combine(expiry, dt.time(15, 30)) - t0).total_seconds(), 0.0) / (
        365.0 * 86400)


def option_flow(opts: list, fut: dict, lots: dict, now: dt.datetime,
                prev_close_at: dt.datetime, mode: str = "iv", memory: dict | None = None) -> dict:
    """{"CE"|"PE": {quadrant: [contracts, Rs Cr], "unread": [contracts, Rs Cr]}} for ONE
    stock, every option changed since yesterday's close.

    opts: [(type 'CE'/'PE', strike, expiry iso, last, prev close, OI change)];
    fut:  {expiry iso: (future now, future at yesterday's close)};
    lots: {expiry iso: lot}.
    Rupees = contracts x lot x future x |delta|: exposure that actually moves with the stock,
    so a far out-of-the-money contract does not count like an at-the-money one. A contract
    whose premium cannot be priced both times (no same-expiry future, a premium under
    OPT_MIN_PREMIUM, or no volatility fits) is counted in 'unread', never forced to a side.

    mode="premium" judges on the premium itself, as most sites do — kept only to show the
    difference: measured on 2.0M stock-option changes (Aug 2024 - Oct 2026) its net reading
    pointed the same way as the stock's own move on 99% of stock-days (rank corr +0.87) and
    called 74% of OI build-ups 'writing' on days the stock did not move (time decay). The IV
    read (trading time, stale-print filters, 2-tick test): 36% / -0.28 and 52%, 24% of
    rupees unclear; Mondays 49% buyers vs 51% other days (calendar time: 68% vs 40%).
    That -0.28 is real and stable (both years): on up days the IV read leans to call writing
    — partly volatility falling as stocks rise, partly, maybe, calls sold into rallies. End-
    of-day data cannot separate the two. Same-moneyness IV (smile shifted) and IV relative to
    the stock's whole chain were tested too: -0.27 and -0.21..-0.25, no better, more unclear."""
    out = {leg: {q: [0.0, 0.0] for q in (*OPT_QUADS, "unread")} for leg in ("CE", "PE")}
    rows = [o for o in opts if o[5]]
    if not rows:
        return out
    typ = np.array([o[0] == "CE" for o in rows])
    K = np.array([o[1] for o in rows], float)
    exp = [o[2] for o in rows]
    last = np.array([o[3] or 0.0 for o in rows], float)
    prev = np.array([o[4] or 0.0 for o in rows], float)
    doi = np.array([o[5] for o in rows], float)
    F1 = np.array([(fut.get(e) or (0, 0))[0] or 0.0 for e in exp], float)
    F0 = np.array([(fut.get(e) or (0, 0))[1] or 0.0 for e in exp], float)
    T1 = np.array([_trading_years(now, dt.date.fromisoformat(e)) for e in exp])
    T0 = np.array([_trading_years(prev_close_at, dt.date.fromisoformat(e)) for e in exp])
    lot = np.array([lots.get(e) or 0 for e in exp], float)
    usable = (last >= OPT_MIN_PREMIUM) & (prev >= OPT_MIN_PREMIUM) & (F1 > 0) & (F0 > 0) & (lot > 0)
    iv1 = np.where(usable, implied_vol(np.where(usable, last, 1), np.where(F1 > 0, F1, 1), K,
                                       np.where(T1 > 0, T1, 1e-9), typ), np.nan)
    iv0 = np.where(usable, implied_vol(np.where(usable, prev, 1), np.where(F0 > 0, F0, 1), K,
                                       np.where(T0 > 0, T0, 1e-9), typ), np.nan)
    _, delta = b76(np.where(F1 > 0, F1, 1), K, np.where(T1 > 0, T1, 1e-9),
                   np.where(np.isfinite(iv1), iv1, 0.3), typ)
    if mode == "premium":
        div = np.where(usable & (last != prev), last - prev, np.nan)
        band = 0.0
    else:
        iv0 = _consistent(iv0, typ, K, exp)
        iv1 = _consistent(iv1, typ, K, exp)
        # the premium the contract WOULD have now if its volatility had not moved: today's
        # future and time, yesterday's volatility. What it beats that by is buyer or seller
        # pressure, in rupees — judged against the exchange tick, not an arbitrary IV band
        same, _ = b76(np.where(F1 > 0, F1, 1), K, np.where(T1 > 0, T1, 1e-9),
                      np.where(np.isfinite(iv0), iv0, 0.3), typ)
        div = np.where(np.isfinite(iv0) & np.isfinite(iv1), last - same, np.nan)
        band = OPT_TICK * OPT_MIN_TICKS - 1e-9
    # TRADE GATE (live, volatility read only). A contract's 'last price' is its last TRADE,
    # often minutes older than the future it is judged against, so an untraded contract
    # seemed to move against every tick of the stock: 14.6% of contract readings switched
    # buyer <-> writer between 10-minute snapshots on 4 Nov 2026 and the line jumped by
    # +-Rs 1,000 Cr. With `memory` a contract is re-judged only when its volume has risen
    # since it was last judged (proof its price is new); a premium back inside the band keeps
    # the side it had instead of dropping to 'unclear'.
    vols = [o[6] if len(o) > 6 else None for o in rows]
    gate = memory is not None and mode != "premium"
    for i in range(len(rows)):
        leg = "CE" if typ[i] else "PE"
        cr = abs(doi[i]) * lot[i] * F1[i] * abs(delta[i]) / 1e7 if lot[i] and F1[i] else 0.0
        side = None
        if np.isfinite(div[i]) and abs(div[i]) > band:
            side = 1 if div[i] > 0 else -1
        if gate:
            key = (leg, float(K[i]), exp[i])
            had = memory.get(key)
            traded = had is None or vols[i] is None or had[1] is None or vols[i] > had[1]
            if had is not None and (not traded or side is None):
                side = had[0]                     # stale price, or back inside the band
            if side is not None:
                memory[key] = (side, vols[i] if traded or had is None else had[1])
        if side is None:
            q = "unread"
        elif doi[i] > 0:
            q = "long build" if side > 0 else "writing"
        else:
            q = "short covering" if side > 0 else "long unwind"
        out[leg][q][0] += doi[i]
        out[leg][q][1] += cr
    return out


def _lots_by_exp(F: dict, base: dict | None, sym: str | None) -> dict:
    """{expiry iso: lot}. Each expiry's own whole-number ratio where it has one — after NSE's
    revision the January 2027 contracts carry new lots (HDFCBANK 750 vs 650) — else the
    stock's proven lot."""
    out, fallback = {}, _lot(F, base, sym)
    for e, v in F.items():
        a = (base or {}).get("fut_by_exp", {}).get(dt.date.fromisoformat(e))
        p = v[0] - v[1]
        x = a / p if a and p > 0 else None
        out[e] = round(x) if x and x >= 1 and abs(x - round(x)) < 0.02 else fallback
    return out


_OPT_MEM: dict = {}      # (day, stock) -> option_flow's per-contract trade-gate memory


def stock_option_flow(d: dict, base: dict | None, sym: str, now: dt.datetime,
                      prev_close_at: dt.datetime, remember: bool = True) -> dict | None:
    """{"iv"|"premium": {"CE"|"PE": [[contracts, Rs Cr] per OPT_QUADS + unread]}} for one
    stock's latest read, or None without per-contract rows or futures prices. `remember`
    keeps each contract's last judged side for the day (the trade gate in option_flow)."""
    F = (d.get("by_exp") or {}).get("F") or {}
    opts = d.get("opts") or []
    if not F or not opts:
        return None
    fut = {e: (v[3], v[4] if len(v) > 4 else None) for e, v in F.items()}
    lots = _lots_by_exp(F, base, sym)
    mem = None
    if remember:
        for k in [k for k in _OPT_MEM if k[0] != now.date()]:
            _OPT_MEM.pop(k, None)                     # a new session starts with no memory
        mem = _OPT_MEM.setdefault((now.date(), sym), {})
    out = {}
    for mode in ("iv", "premium"):
        r = option_flow(opts, fut, lots, now, prev_close_at, mode, mem if mode == "iv" else None)
        out[mode] = {leg: [[round(r[leg][q][0]), round(r[leg][q][1], 2)]
                           for q in (*OPT_QUADS, "unread")] for leg in ("CE", "PE")}
    return out


def oi_changes(d: dict, base: dict | None, sym: str | None = None) -> tuple:
    """(futures %, call %, put %, used_archive_base) for one stock's live OI read.

    THE NUMERATOR IS LIVE AND EXACT; THE DENOMINATOR IS NOT SAFE TO TAKE FROM THE SAME FEED.
    NSE's per-stock derivatives payload lists only contracts that have TRADED TODAY — audited
    13 Oct 2026: 3,786 contracts across the Nifty 50, zero of them with zero volume, while
    yesterday's bhavcopy shows e.g. ADANIENT puts with 35 of 62 contracts untraded. An
    untraded contract cannot have changed its open interest, so the summed CHANGE is complete.
    But yesterday's total rebuilt as (OI now - change) is missing every untraded strike:
    11% short on ADANIENT puts at the close, and far worse at 09:30 when only a handful of
    strikes have printed — a 2% build would read as 20%.
    So the base is yesterday's END-OF-DAY total from the archive, converted from shares to
    contracts. The lot size is not stored anywhere; it is recovered as archive OI / live
    previous OI on each futures contract, which lands on a whole number (309, 475, 125 ...)
    on 50 of 50 stocks. If it does not — archive a session behind, lot revised, corporate
    action — the stock falls back to the feed's own base and is counted as such on the page.
    """
    def live(k):
        b = d[k + "_oi"] - d[k + "_chg"]
        return (d[k + "_chg"] / b * 100) if b > 0 else None

    # NO remembered lot here: a proof failing usually means the base is the WRONG SESSION,
    # and then base / lot is the wrong denominator however right the lot is
    hit = _lot_cache().get(sym) if sym else None
    lot = _proven_lot([(base["fut_by_exp"].get(e), v) for e, v in d["fut_prev"].items()]
                      if base and d.get("fut_prev") else [], known=hit[0] if hit else None)
    if not lot:
        return live("fut"), live("ce"), live("pe"), False
    out = []
    for k in ("fut", "ce", "pe"):
        b = base[k] / lot
        out.append((d[k + "_chg"] / b * 100) if b > 0 else None)
    return (*out, True)


def session_fraction(ts: dt.datetime | None) -> float:
    """Share of a normal day's volume traded by `ts`, from 0 at 09:15 to 1 at 15:25 — no
    blank early window. Used to pro-rate the options cycle norm. pace_fraction() is None
    before 09:30 (right for Activity, where a ratio on 10% of a day is noise), and the old
    code read that None as 1.0 — so from 09:15 to 09:30 the FULL day's norm (about -0.7pp at
    9 sessions out) was subtracted from a gap that had barely started forming, tilting the
    first 15 minutes toward 'put side heavier'."""
    if ts is None:
        return 1.0
    hm = ts.hour * 60 + ts.minute
    if hm < 9 * 60 + 15:
        return 0.0
    if hm >= 15 * 60 + 25:
        return 1.0
    return config.VOL_PROFILE.get(f"{hm // 60:02d}:{hm % 60 // 5 * 5:02d}", 1.0)


def pace_fraction(ts: dt.datetime | None) -> float | None:
    """Share of a normal day's volume traded by `ts` (config.VOL_PROFILE). None before
    09:30, when ~10% has traded and any ratio against it is noise."""
    if ts is None:
        return None
    hm = ts.hour * 60 + ts.minute
    if hm < 9 * 60 + 30:
        return None
    if hm >= 15 * 60 + 25:
        return 1.0
    key = f"{hm // 60:02d}:{hm % 60 // 5 * 5:02d}"
    return config.VOL_PROFILE.get(key)


def decompose(cash: dict, weights: dict | None, deriv: dict | None, code: str, *,
              dcm=None, norms: dict | None = None, corp: dict | None = None,
              oi_base: dict | None = None, now: dt.datetime | None = None) -> dict:
    """Everything the page shows, from already-fetched payloads. Pure: no network, no clock
    except `now`. Never raises on a partial input — a missing leg comes back as None."""
    now = now or _now()
    weights, deriv, norms, corp = weights or {}, deriv or {}, norms or {}, corp or {}
    oi_base, n_arch = oi_base or {}, 0
    idx, stocks = cash["index"], cash["stocks"]
    ts = cash.get("ts") or cash.get("fetched") or now
    day = ts.date()
    prev = idx["prev"]
    ipts = idx["last"] - prev
    ipct = (idx["last"] / prev - 1) * 100
    notes: list[str] = []

    rb = resolve_buckets(code, weights, list(stocks), day, dcm)
    bucket_of = {s: lbl for lbl, mem in rb["buckets"].items() for s in mem}
    have_w = bool(weights) and all(s in weights for s in stocks)
    if weights and not have_w:
        notes.append("NSE's weight list and its constituent list disagree — index points "
                     "are withheld until they match again.")

    # monthly expiry: the nearest expiry across the swept stocks
    exps = sorted({e for d in deriv.values() for e in d.get("expiries", []) if e >= day})
    front = exps[0] if exps else None
    is_expiry = front == day
    left = _sessions_left(day, front) if front else None
    frac = pace_fraction(ts) if cash.get("status") == "Open" else 1.0
    ofrac = session_fraction(ts) if cash.get("status") == "Open" else 1.0

    recs = []
    for s, q in stocks.items():
        w = weights.get(s) if have_w else None
        d = deriv.get(s)
        ca = corp.get(s)
        fut = ce = pe = quad = None
        if d and not ca:
            fut, ce, pe, arch = oi_changes(d, oi_base.get(s), s)
            n_arch += arch
            if fut is not None and dcm is not None:
                quad = dcm._quadrant(q["pct"], fut)
        lag = None
        u = _ts(q.get("upd"))
        if u and cash.get("ts"):
            lag = (cash["ts"] - u).total_seconds()
        recs.append(dict(
            symbol=s, bucket=bucket_of.get(s), weight=w, pct=q["pct"], ltp=q["ltp"],
            pts=(w / 100 * q["pct"] / 100 * prev) if w is not None else None,
            value_cr=(q["value"] / 1e7) if q.get("value") else None,
            norm_cr=norms.get(s),
            fut_oi_pct=fut, quadrant=quad,
            ce_pct=None if ce is None else float(np.clip(ce, -_OI_CLIP, _OI_CLIP)),
            pe_pct=None if pe is None else float(np.clip(pe, -_OI_CLIP, _OI_CLIP)),
            lag_s=lag, corp=ca))
    df = pd.DataFrame(recs)

    for s, ca in corp.items():
        if s in stocks:
            notes.append(
                f"{s} is in the F&O ban today — no new derivative positions are allowed, so "
                "its open interest can only fall. Its futures and options votes are withheld."
                if ca == "F&O BAN" else
                f"{s}: {ca} ex-date today — its futures and options votes are withheld (open "
                "interest is re-based or re-listed, not traded).")
    big = df[df["pct"].abs() >= 20]
    for r in big.itertuples():
        if not r.corp:
            notes.append(f"{r.symbol} is {r.pct:+.1f}% — real moves this size happen "
                         "(POLICYBZR −36% on 24 Sep 2026), but check for a corporate action.")
    if cash.get("status") == "Open":
        halted = df[(df["lag_s"].notna()) & (df["lag_s"] > 600)]
        if len(halted) > len(df) / 2:
            # half the index cannot all stop trading at once: NSE's rows are stuck (4 Nov 2026,
            # all 50 Nifty rows frozen at 09:15:46 while the index print moved +126 points)
            notes.append(f"NSE's stock price rows for {len(halted)} of {len(df)} stocks are "
                         f"{halted['lag_s'].median() / 60:.0f}+ minutes behind its index print — "
                         "a feed fault, not a trading halt. Index points, returns and breadth "
                         "below are stale until NSE's rows move again (or the next derivatives "
                         "sweep supplies fresh prices).")
        elif len(halted):
            notes.append("No trade for 10+ minutes: " + ", ".join(halted["symbol"]) + ".")

    explained = float(df["pts"].sum()) if have_w else None
    rows, dcm_rows = [], []
    for lbl, mem in rb["buckets"].items():
        g = df[df["symbol"].isin(mem)]
        n_mem = rb["sizes"].get(lbl, len(mem))
        row = dict(label=lbl, n_members=max(n_mem, len(mem)), n_present=len(g),
                   weight=None, ret_ew=None, ret_cw=None, pts=None, share=None,
                   adv=0, dec=0, adv_pct=None, pace=None, value_share=None,
                   value_share_usual=None, fut_oi_pct=None, fut_valid=0,
                   fut_read=None, fut_votes="—", fut_lean=None, ce_pct=None, pe_pct=None,
                   opt_valid=0, opt_read=None, movers_up=[], movers_dn=[], top_pts=[])
        if len(g):
            row["ret_ew"] = float(g["pct"].mean())
            row["adv"], row["dec"] = int((g["pct"] > 0).sum()), int((g["pct"] < 0).sum())
            row["adv_pct"] = row["adv"] / len(g) * 100
            if have_w:
                row["weight"] = float(g["weight"].sum())
                row["pts"] = float(g["pts"].sum())
                row["ret_cw"] = float((g["weight"] * g["pct"]).sum() / g["weight"].sum())
                # a share of a ~zero move is a division by noise; points still stand
                if abs(ipct) >= 0.10 and explained:
                    row["share"] = row["pts"] / explained * 100
                k = g.reindex(g["pts"].abs().sort_values(ascending=False).index).head(3)
                row["top_pts"] = [(r.symbol, float(r.pts)) for r in k.itertuples()]
            tv = df["value_cr"].sum()
            if tv and g["value_cr"].notna().any():
                row["value_share"] = float(g["value_cr"].sum() / tv * 100)
            # The yardstick is the bucket's USUAL share of traded value, not its index
            # weight: big companies always trade a smaller slice of their size than small
            # ones. 15 Oct 2026: Nifty Top 10 = 52.8% of the index but 43% of the day's
            # value - against weight that reads "money leaving the heavyweights", against
            # its own normal it may be an ordinary day.
            tn = df["norm_cr"].sum()
            if tn and g["norm_cr"].notna().any():
                row["value_share_usual"] = float(g["norm_cr"].sum() / tn * 100)
            nv = g.dropna(subset=["value_cr", "norm_cr"])
            if frac and len(nv) >= max(1, len(g) * 0.6) and nv["norm_cr"].sum() > 0:
                row["pace"] = float(nv["value_cr"].sum() / (nv["norm_cr"].sum() * frac)
                                    * 100 - 100)
            s_ = g.sort_values("pct", ascending=False)
            row["movers_up"] = [(r.symbol, float(r.pct)) for r in s_.head(3).itertuples()
                                if r.pct > 0]
            row["movers_dn"] = [(r.symbol, float(r.pct)) for r in s_.tail(3).itertuples()
                                if r.pct < 0][::-1]
            f = g.dropna(subset=["fut_oi_pct"])
            row["fut_valid"] = len(f)
            if len(f):
                row["fut_oi_pct"] = float(np.clip(f["fut_oi_pct"], -_OI_CLIP, _OI_CLIP).mean())
            o = g.dropna(subset=["ce_pct", "pe_pct"])
            if len(o) and not is_expiry:
                row["opt_valid"] = len(o)
                row["ce_pct"], row["pe_pct"] = float(o["ce_pct"].mean()), float(o["pe_pct"].mean())
        if dcm is not None:
            # DCM's own row object does the judging: quorum, 1.5x majority, the name of
            # the dominant quadrant, and the options label net of the cycle norm. The norm
            # is linear in opt_scale, so scaling it by the session fraction pro-rates it.
            q = g["quadrant"].value_counts() if len(g) else {}
            b = dcm.BucketRow(
                label=lbl, n_members=row["n_members"], n_present=row["n_present"],
                ret_pct=row["ret_ew"], adv=row["adv"], dec=row["dec"],
                adv_pct=row["adv_pct"],
                fut_long=int(q.get("long build", 0)), fut_short=int(q.get("short build", 0)),
                fut_cover=int(q.get("short covering", 0)),
                fut_unwind=int(q.get("long unwind", 0)),
                fut_valid=row["fut_valid"], fut_oi_pct=row["fut_oi_pct"],
                ce_oi_pct=row["ce_pct"], pe_oi_pct=row["pe_pct"],
                opt_valid=row["opt_valid"], opt_dte=left,
                opt_scale=rb["opt_scale"] * ofrac)
            row.update(fut_read=b.fut_read, fut_votes=b.fut_votes, fut_lean=b.fut_lean,
                       opt_read=b.opt_read)
            dcm_rows.append(b)
        rows.append(row)

    adv, dec = int((df["pct"] > 0).sum()), int((df["pct"] < 0).sum())
    ew = float(df["pct"].mean())
    state = regime = None
    if dcm is not None and dcm_rows:
        try:
            whole = dcm.IndexLargeCap(
                trade_date=day, fno_symbol=code, display=DISPLAY[code], index_ret=ipct,
                equal_ret=ew, carry_spread=ipct - ew, adv_all=adv, dec_all=dec,
                n_present=len(df), rows=dcm_rows)
            state, regime = whole.state, whole.regime
        except Exception:                                    # noqa: BLE001
            pass

    age = (now - ts).total_seconds()
    live = cash.get("status") == "Open"
    if live and age > STALE_AFTER:
        notes.append(f"NSE's feed is {age / 60:.0f} minutes old while the market is open — "
                     "numbers below are the last good read, not now.")
    if rb["added"] or rb["dropped"]:
        notes.append("Basket drift against DCM's list — "
                     + (f"new on NSE: {', '.join(rb['added'])} (placed in the tail bucket). "
                        if rb["added"] else "")
                     + (f"No longer listed: {', '.join(rb['dropped'])}. " if rb["dropped"] else "")
                     + "Run DCM `scripts/refresh_index_buckets.py`.")
    return dict(
        code=code, display=DISPLAY[code], ts=ts, day=day, status=cash.get("status"),
        live=live, age_s=age,
        index=dict(last=idx["last"], prev=prev, pts=ipts, pct=ipct),
        ew=ew, spread=ipct - ew, carry_band=rb["carry_band"],
        adv=adv, dec=dec, unch=len(df) - adv - dec, n=len(df),
        have_weights=have_w, explained_pts=explained,
        residual_pts=(ipts - explained) if explained is not None else None,
        buckets=rows, stocks=df, state=state, regime=regime,
        bucket_source=rb["source"], misranked=rb["misranked"], from_dcm=rb["from_dcm"],
        front_expiry=front, sessions_left=left, is_expiry=is_expiry,
        session_frac=frac, n_deriv=len(deriv), n_archive_base=n_arch, notes=notes,
        opt_scale=rb["opt_scale"])


# ── intraday trend (from the tape this module logs) ───────────────────────────────────
def tape_frame(tape: list, buckets: dict, weights: dict) -> pd.DataFrame:
    """One row per logged poll: index points, each bucket's points, spread, breadth."""
    out = []
    for e in tape:
        p, prev = e["p"], e["prev"]
        if not p or not prev:
            continue
        row = {"t": pd.Timestamp(e["t"]), "Index": e["last"] - prev,
               "spread": (e["last"] / prev - 1) * 100 - float(np.mean(list(p.values()))),
               "adv_pct": sum(v > 0 for v in p.values()) / len(p) * 100}
        for lbl, mem in buckets.items():
            row[lbl] = sum((weights.get(s) or 0.0) / 100 * p[s] / 100 * prev
                           for s in mem if s in p)
        out.append(row)
    return pd.DataFrame(out).set_index("t") if out else pd.DataFrame()


def window_moves(frame: pd.DataFrame, labels: list, minutes=(15, 30, 60)) -> list:
    """Points added by the index and by each bucket over the last N minutes, plus since the
    first logged poll. A window the tape does not reach back to is left out, never guessed:
    the tape only exists while this dashboard has been running."""
    if frame.empty or len(frame) < 2:
        return []
    last, t1 = frame.iloc[-1], frame.index[-1]
    rows = []
    for m in minutes:
        past = frame[frame.index <= t1 - pd.Timedelta(minutes=m)]
        # the anchor must sit within 40% of the window, or a gap is being called a window
        if past.empty or (t1 - past.index[-1]) > pd.Timedelta(minutes=m * 1.4):
            continue
        a = past.iloc[-1]
        rows.append(dict(window=f"last {m} min", index=float(last["Index"] - a["Index"]),
                         **{lbl: float(last[lbl] - a[lbl]) for lbl in labels}))
    a = frame.iloc[0]
    rows.append(dict(window=f"since {frame.index[0]:%H:%M} (first log)",
                     index=float(last["Index"] - a["Index"]),
                     **{lbl: float(last[lbl] - a[lbl]) for lbl in labels}))
    return rows


# Dead bands for a WINDOW read. The day read uses DCM's 0.5% on OI and none on price; over
# 15-60 minutes both moves are smaller and noisier, so each axis gets its own floor. These two
# numbers are a judgement, NOT measured — there was no intraday stock OI history to measure
# them on before this module started logging one.
WIN_OI_BAND = 0.25       # % change in futures OI inside the window, below which no vote
WIN_PX_BAND = 0.10       # % price move inside the window, below which no vote


def recent_flow(oi_tape: list, tape: list, buckets: dict, dcm,
                minutes=(15, 30, 60)) -> list:
    """Futures build-up INSIDE the last N minutes, per bucket.

    The main table compares now with YESTERDAY'S CLOSE. That is the day's net position, and
    it goes stale as a description of the present: a stock up 2% since yesterday with OI up
    reads "long build" all afternoon even if for the last half hour it has been falling
    while OI keeps rising — which is a fresh SHORT build. This compares two of this
    module's own sweeps instead: price then vs now, OI then vs now.

    OI change is taken as the difference of NSE's "change vs yesterday" figures, never of
    the OI totals: the feed lists only contracts that have traded today, so a far-month
    contract printing its first trade makes the total jump without one new position.
    """
    if dcm is None or len(oi_tape) < 2 or len(tape) < 2:
        return []
    px = [(pd.Timestamp(e["t"]), e["p"]) for e in tape]

    def px_at(ts):
        prior = [x for x in px if x[0] <= ts + pd.Timedelta(seconds=90)]
        if not prior or ts - prior[-1][0] > pd.Timedelta(minutes=4):
            return None                          # no price logged near that sweep
        return prior[-1][1]

    now, t1 = oi_tape[-1], pd.Timestamp(oi_tape[-1]["t"])
    p1 = px_at(t1)
    if p1 is None:
        return []
    out = []
    for m in minutes:
        past = [e for e in oi_tape if pd.Timestamp(e["t"]) <= t1 - pd.Timedelta(minutes=m)]
        if not past or t1 - pd.Timestamp(past[-1]["t"]) > pd.Timedelta(minutes=m * 1.4):
            continue                             # the log does not reach back that far
        a = past[-1]
        p0 = px_at(pd.Timestamp(a["t"]))
        if p0 is None:
            continue
        for lbl, mem in buckets.items():
            c = dict(long=0, short=0, cover=0, unwind=0)
            dps, dois = [], []
            for s in mem:
                if not (s in a["d"] and s in now["d"] and s in p0 and s in p1):
                    continue
                base = a["d"][s][0]
                if base <= 0:
                    continue
                doi = (now["d"][s][1] - a["d"][s][1]) / base * 100
                dp = ((1 + p1[s] / 100) / (1 + p0[s] / 100) - 1) * 100
                dps.append(dp)
                dois.append(float(np.clip(doi, -_OI_CLIP, _OI_CLIP)))
                if abs(doi) <= WIN_OI_BAND or abs(dp) <= WIN_PX_BAND:
                    continue
                c["long" if dp > 0 and doi > 0 else "cover" if dp > 0 else
                  "short" if doi > 0 else "unwind"] += 1
            if not dps:
                continue
            # the same judge as the day read: quorum, 1.5x majority, named by what dominates
            b = dcm.BucketRow(label=lbl, n_members=len(mem), n_present=len(dps),
                              fut_long=c["long"], fut_short=c["short"],
                              fut_cover=c["cover"], fut_unwind=c["unwind"],
                              fut_valid=len(dps))
            out.append(dict(window=f"last {m} min", bucket=lbl,
                            since=f"{pd.Timestamp(a['t']):%H:%M}",
                            price=float(np.mean(dps)), oi=float(np.mean(dois)),
                            read=b.fut_read, votes=b.fut_votes,
                            names=f"{len(dps)}/{len(mem)}"))
    return out


# EFFORT vs RESULT — traded value (effort) against price move (result), per bucket, per
# window. The user's idea (15 Oct 2026): Top 10 up 1.5-2% on very heavy value = serious
# buying that should carry on; Top 10 value rising at 12:30 while price falls = someone
# selling, Nifty likely to follow down. MEASURED BEFORE SHIPPING (scratch script
# effort_result_bt.py): 393 sessions, Oct 2024 - Jun 2026, Tradebot 5-min candles, today's
# Nifty basket, weights propagated by price, 3,930 checkpoints every 30 min 10:15-14:45:
#   DIRECTION - nothing. Next 60 min after Top 10 UP >=0.2% in 30 min: heavy value (>=1.5x
#     normal for that time) +2.9bps, light/normal +3.0bps (permutation p 0.99). After Top 10
#     DOWN: heavy +4.0 vs +1.0 (p 0.49) - heavy selling was followed by a slight BOUNCE, not
#     more selling. Regression interaction (return x heavy) t -0.25. The heavy-vs-light gap
#     flips sign between the two halves of the sample (+7.4 then -6.6bps). Base: next 60 min
#     is 50.0% up. The user's two named scenarios had too few cases (<30) to grade at all.
#   SIZE - real. Heavy Top 10 value is followed by BIGGER Nifty swings in the next hour:
#     19.6bps average move vs 15.6 normal and 14.3 light; beside the market's own last-30-min
#     move it still adds +2.9bps (t +2.8), and it holds in both halves (21.4 vs 14.9, 18.0 vs
#     15.6). Same lesson as the cockpit's big-move-day flag: activity forecasts SIZE, not side.
# So the words below describe what is happening; the only forward claim is about swing size.
EFFORT_HEAVY, EFFORT_QUIET = 1.5, 0.8     # traded value vs normal for the same clock window
EFFORT_PX_30 = 0.20                      # % price move in 30 min that counts as a real move


def _effort_word(px: float | None, ratio: float | None, minutes: float) -> str | None:
    if px is None or ratio is None:
        return None
    band = EFFORT_PX_30 * (minutes / 30.0) ** 0.5 if minutes else 0.5
    move = "up" if px >= band else "down" if px <= -band else "flat"
    if ratio >= EFFORT_HEAVY:
        return {"up": "heavy buying", "down": "heavy selling",
                "flat": "heavy trade, no progress"}[move]
    if ratio <= EFFORT_QUIET:
        return {"up": "quiet rise", "down": "quiet fall", "flat": "quiet"}[move]
    return {"up": "rising, normal trade", "down": "falling, normal trade",
            "flat": "normal"}[move]


def effort_result(tape: list, buckets: dict, weights: dict, norms: dict,
                  minutes=(15, 30, 60)) -> list:
    """Per bucket and window: price move, traded value, value vs the normal for that clock
    window, the bucket's share of all value traded in the window, and the word.

    The normal for a window is each stock's 21-session median daily value times the share
    of a normal day's volume that trades between the window's two clock times
    (config.VOL_PROFILE) — so 09:15-09:45 is not compared with 12:30-13:00, which trade
    very different amounts on every ordinary day."""
    rows_ = [e for e in tape if e.get("v")]
    if len(rows_) < 1 or not weights:
        return []
    last, t1 = rows_[-1], pd.Timestamp(rows_[-1]["t"])
    f1 = session_fraction(t1.to_pydatetime())
    wins = []
    for m in minutes:
        past = [e for e in rows_ if pd.Timestamp(e["t"]) <= t1 - pd.Timedelta(minutes=m)]
        if past and t1 - pd.Timestamp(past[-1]["t"]) <= pd.Timedelta(minutes=m * 1.4):
            wins.append((f"last {m} min", m, past[-1]))
    wins.append(("since the open", 0, None))
    out = []
    for name, m, a in wins:
        f0 = session_fraction(pd.Timestamp(a["t"]).to_pydatetime()) if a else 0.0
        tot = 0.0
        recs = []
        for lbl, mem in buckets.items():
            px_num = px_den = val = nrm = 0.0
            for s in mem:
                if s not in last["p"] or s not in last["v"]:
                    continue
                if a is not None and (s not in a["p"] or s not in a.get("v", {})):
                    continue
                w = weights.get(s) or 0.0
                p0 = a["p"][s] if a else 0.0
                px_num += w * ((1 + last["p"][s] / 100) / (1 + p0 / 100) - 1) * 100
                px_den += w
                val += last["v"][s] - (a["v"][s] if a else 0.0)
                nrm += (norms.get(s) or 0.0) * max(f1 - f0, 0.0)
            px = px_num / px_den if px_den else None
            ratio = val / nrm if nrm > 0 else None
            tot += val
            recs.append(dict(window=name, since=(f"{pd.Timestamp(a['t']):%H:%M}" if a else "09:15"),
                             bucket=lbl, price=px, value_cr=val, ratio=ratio,
                             word=_effort_word(px, ratio, m)))
        for r in recs:
            r["value_share"] = r["value_cr"] / tot * 100 if tot else None
        out += recs
    return out


SERIES_QUORUM = 0.6     # share of a bucket's stocks a reading needs before it is plotted
# Pressure-line side hysteresis, % from the baseline price. Chosen on 75 bucket-days (7 days x
# 4 indices, Oct 2026): line chatter (total movement / net move) 4.75 -> 2.89, total rupee
# movement -28%, stock side flips 2.15 -> 1.03 a day, and the END reading unchanged on all 75
# (0.15 changed its sign on 5.3%, so 0.10 is the widest band that never moves the reading).
PRESSURE_BAND = 0.10


def session_series(oi_tape: list, tape: list, buckets: dict, weights: dict, oi_base: dict,
                   fut_close: dict, cash_prev: dict, day: dt.date) -> dict:
    """Per OI reading (one row per sweep), per bucket, six series for the charts:

      fut_oi / ce / pe   OI change vs yesterday's close, % — the SAME arithmetic as the main
                         table (oi_changes: today's change over yesterday's archive total), so
                         the last point of each line equals the table. Simple bucket average.
      pc                 put % minus call % (no cycle adjustment — the line shows the raw gap)
      fut_px             near-month futures move since yesterday's close, %, weighted like the
                         index (index weight, renormalised over the stocks present)
      basis              near-month futures premium over cash, basis points, same weighting

    NEAR MONTH, NOT "ALL MONTHS", FOR PRICE. OI is summed over every month (a roll nets to
    zero); a price cannot be summed, so one contract is followed all day: the nearest expiry,
    or the NEXT one on expiry day itself, when the front contract is pinned to cash and its
    premium collapses to zero by construction. The choice is made per day, so a line never
    jumps from one contract to another mid-session.

    A reading is plotted for a bucket only when at least SERIES_QUORUM of its stocks are in
    it; a sweep that caught three of ten names is not the bucket."""
    out = {k: [] for k in ("fut_oi", "ce", "pe", "pc", "fut_px", "basis")}
    if not oi_tape:
        return {k: pd.DataFrame() for k in out}
    import bisect
    px = [(pd.Timestamp(e["t"]), e["p"]) for e in tape]
    px_t = [x[0] for x in px]

    def cash_at(ts, s):
        # bisect, not a scan: the scan made this chart cost ~2s per 90 readings, run on
        # every 15-second redraw (a full session is ~300 readings)
        i = bisect.bisect_right(px_t, ts + pd.Timedelta(seconds=60)) - 1
        if i < 0 or s not in px[i][1] or s not in cash_prev:
            return None
        return cash_prev[s] * (1 + px[i][1][s] / 100)

    for e in oi_tape:
        ts = pd.Timestamp(e["t"])
        rows = {k: {"t": ts} for k in out}
        xs = e.get("x", {})
        for lbl, mem in buckets.items():
            oi_v = {"fut_oi": [], "ce": [], "pe": []}
            pw = {"fut_px": [], "basis": []}
            for s in mem:
                d0 = e["d"].get(s)
                if not d0:
                    continue
                x = xs.get(s, {})
                fexp = x.get("F", {})
                d = dict(fut_oi=d0[0], fut_chg=d0[1], ce_oi=d0[2], ce_chg=d0[3],
                         pe_oi=d0[4], pe_chg=d0[5],
                         fut_prev={dt.date.fromisoformat(k): v[0] - v[1]
                                   for k, v in fexp.items()})
                f, c, p_, _ = oi_changes(d, oi_base.get(s), s)
                for k, v in (("fut_oi", f), ("ce", c), ("pe", p_)):
                    if v is not None:
                        oi_v[k].append(float(np.clip(v, -_OI_CLIP, _OI_CLIP)))
                live = sorted(k for k in fexp if k >= day.isoformat())
                if not live:
                    continue
                k = live[1] if (live[0] == day.isoformat() and len(live) > 1) else live[0]
                f_ltp = fexp[k][3]
                f_prev = (fexp[k][4] if len(fexp[k]) > 4 and fexp[k][4]
                          else fut_close.get(s, {}).get(k))
                w = weights.get(s) or 0.0
                if f_ltp and f_prev and w:
                    pw["fut_px"].append((w, (f_ltp / f_prev - 1) * 100))
                spot = x.get("S") or cash_at(ts, s)
                if f_ltp and spot and w:
                    pw["basis"].append((w, (f_ltp / spot - 1) * 1e4))
            need = max(1, int(np.ceil(len(mem) * SERIES_QUORUM)))
            for k in ("fut_oi", "ce", "pe"):
                rows[k][lbl] = float(np.mean(oi_v[k])) if len(oi_v[k]) >= need else None
            rows["pc"][lbl] = (rows["pe"][lbl] - rows["ce"][lbl]
                               if rows["pe"][lbl] is not None and rows["ce"][lbl] is not None
                               else None)
            for k in ("fut_px", "basis"):
                v = pw[k]
                rows[k][lbl] = (float(sum(a * b for a, b in v) / sum(a for a, _ in v))
                                if len(v) >= need else None)
        for k in out:
            out[k].append(rows[k])
    return {k: pd.DataFrame(v).set_index("t").astype(float) for k, v in out.items()}


# PER-INTERVAL FLOW. The cumulative "OI vs yesterday" line rises on almost every day — same-
# day positions pile up from 09:15 — so a sharp unwind of hundreds of crores in five minutes
# barely bends it, and price sat on a different chart. This splits the session into blocks
# and, for each block and each stock, reads the CHANGE in futures OI and the CHANGE in its
# futures price inside the block, so every rupee of positions added or closed lands in one of
# the four quadrants. Money, not contract counts: contracts are not comparable across stocks
# (lot sizes differ), rupee notional (contracts x lot x price) is.
QUADRANTS = ("long build", "short build", "short covering", "long unwind", "price unchanged")
# "price unchanged": positions moved while the futures price printed the same between two
# readings. Dropping them lost 2-14% of the session's rupees on 16 Oct 2026 (Nifty Top 10:
# blocks summed to +3,949 Cr against +4,183 Cr read directly), so they are kept, in grey.
FLOW_SHARP_X = 3.0      # a block this many times the session's typical block is "unusual"
FLOW_SHARP_MIN = 6      # ... judged only once this many earlier blocks exist
# Options blocks are lumpier than futures (a few strikes can carry a block), so 3x flagged
# 11 option blocks in two hours on Fin Nifty (16 Oct 2026); 4x leaves 2-5 per index.
OPT_SHARP_X = 4.0


_LOTS_FILE = DATA_DIR / "lots.json"
_LOTS: dict = {"mem": None}
LOT_CACHE_DAYS = 45      # a remembered lot older than this is not trusted (NSE revises lots)
_REPLAY_TLS = threading.local()


@contextlib.contextmanager
def _replay():
    """While a PAST session is drawn on this thread, the lot memory neither learns nor lends.
    Proving an old day's lot would stamp it 'proven today' — after an NSE revision that would
    hand the old lot to the live page — and a lot learned later must not reach back in time.
    Thread-local: the poller keeps learning on its own thread meanwhile."""
    was = getattr(_REPLAY_TLS, "on", False)
    _REPLAY_TLS.on = True
    try:
        yield
    finally:
        _REPLAY_TLS.on = was


def _lot_cache() -> dict:
    """{symbol: [lot, iso date it was last proven]} — shared by every process via lots.json."""
    if _LOTS["mem"] is None:
        try:
            _LOTS["mem"] = json.loads(_LOTS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _LOTS["mem"] = {}
    return _LOTS["mem"]


def _remember_lot(sym: str, lot: int) -> None:
    if getattr(_REPLAY_TLS, "on", False):
        return
    mem = _lot_cache()
    today = dt.date.today().isoformat()
    if mem.get(sym) == [lot, today]:
        return
    mem[sym] = [lot, today]
    try:
        # pid-unique temp file: the dashboard and the capture task can both write
        tmp = _LOTS_FILE.with_name(f"{_LOTS_FILE.name}.{os.getpid()}.tmp")
        _LOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(mem), encoding="utf-8")
        os.replace(tmp, _LOTS_FILE)
    except OSError:
        pass


def _lot(fexp: dict, base: dict | None, sym: str | None = None) -> int | None:
    """Lot size from one reading's per-expiry futures detail: archive OI (shares) / live
    previous OI (contracts), only when it lands on a whole number (see oi_changes).

    That ratio needs the archive to hold the SAME session NSE calls "previous". When it is
    a session behind (26 Oct 2026: NSE's 19:30 F&O file came without the November contracts,
    DCM's sanity gate refused it, and 47 of 50 Nifty stocks had no lot), every rupee chart
    went blank and the pressure table crashed. So a lot proven against the archive is
    remembered per stock, and reused — with `sym` given — when today's proof fails."""
    hit = _lot_cache().get(sym) if sym else None
    lot = _proven_lot([(base["fut_by_exp"].get(dt.date.fromisoformat(e)), v[0] - v[1])
                       for e, v in fexp.items()] if base and fexp else [],
                      known=hit[0] if hit else None)
    if lot:
        if sym:
            _remember_lot(sym, lot)
        return lot
    if sym and not getattr(_REPLAY_TLS, "on", False):
        hit = _lot_cache().get(sym)
        if hit:
            age = (dt.date.today() - dt.date.fromisoformat(hit[1])).days
            if 0 <= age <= LOT_CACHE_DAYS:
                return int(hit[0])
    return None


def _near(fexp: dict, day: dt.date) -> str | None:
    live = sorted(k for k in fexp if k >= day.isoformat())
    if not live:
        return None
    return live[1] if (live[0] == day.isoformat() and len(live) > 1) else live[0]


def oi_flow_blocks(oi_tape: list, buckets: dict, weights: dict, oi_base: dict,
                   fut_close: dict, day: dt.date, minutes: int = 5) -> dict:
    """{"flow": long frame [end, start, bucket, quadrant, cr, stocks],
        "price": frame [end x bucket] futures % since yesterday's close at each block end,
        "sharp": list of unusual blocks}.

    Each block compares the LAST reading in it with the last reading of the previous block.
    OI change = difference of NSE's change-vs-yesterday figures (robust to a contract's first
    trade of the day); price change = the stock's near-month futures price between the same
    two readings. A block that follows a logging gap (more than 3 blocks long) is not drawn:
    an hour of activity squeezed into one bar would read as a 5-minute event. The first
    reading of the day is the baseline, so overnight + opening positioning is not a bar."""
    empty = dict(flow=pd.DataFrame(columns=["end", "start", "bucket", "quadrant", "cr",
                                            "stocks"]),
                 price=pd.DataFrame(), sharp=[],
                 options=pd.DataFrame(columns=["end", "start", "bucket", "leg", "move", "cr",
                                               "stocks"]), osharp=[])
    rows = [e for e in oi_tape if e.get("x")]
    if len(rows) < 2:
        return empty
    by_block: dict = {}
    for e in rows:
        ts = pd.Timestamp(e["t"])
        by_block[ts.floor(f"{minutes}min")] = (ts, e)        # last reading in each block
    blocks = sorted(by_block)
    flow, price, oflow = [], [], []
    lots: dict = {}
    seen: dict = {}                  # stock -> (block, time, its OI row, its futures detail)
    for i, b in enumerate(blocks):
        ts, e = by_block[b]
        prow = {"end": ts}
        for lbl, mem in buckets.items():
            num = den = 0.0
            for s in mem:
                f = e["x"].get(s, {}).get("F", {})
                k = _near(f, day)
                if not k:
                    continue
                prev = f[k][4] if len(f[k]) > 4 and f[k][4] else fut_close.get(s, {}).get(k)
                w = weights.get(s) or 0.0
                if f[k][3] and prev and w:
                    num += w * (f[k][3] / prev - 1) * 100
                    den += w
            prow[lbl] = num / den if den else None
        price.append(prow)
        if i == 0 and not seen:
            # the baseline is the day's FIRST reading, not the end of the first block — the
            # first block's own changes were being swallowed into the baseline
            e0 = rows[0]
            t0 = pd.Timestamp(e0["t"])
            for s, d0 in e0["d"].items():
                seen[s] = (t0.floor(f"{minutes}min"), t0, d0, e0["x"].get(s, {}).get("F", {}))
            if e0 is e:
                continue
        gap_block = i > 0 and (b - blocks[i - 1]) > pd.Timedelta(minutes=minutes * 3)
        for lbl, mem in buckets.items():
            acc = {q: [0.0, 0] for q in QUADRANTS}
            oacc = {(leg, mv): [0.0, 0] for leg in ("Calls", "Puts")
                    for mv in ("price up", "price down", "price unchanged")}
            starts = []
            for s in mem:
                c = e["d"].get(s)
                if c is None:
                    continue                                  # absent: keep its last reading
                fc = e["x"].get(s, {}).get("F", {})
                prev = seen.get(s)
                seen[s] = (b, ts, c, fc)
                # compared with THIS STOCK's own last reading: a stock NSE skipped in one
                # sweep used to lose that block's change for good (Nifty Rest 30, 16 Oct:
                # blocks summed 14% short of the session total)
                if prev is None or gap_block or (b - prev[0]) > pd.Timedelta(minutes=minutes * 3):
                    continue                                  # a gap, not a block
                _, pts, a, fa = prev
                k = _near(fc, day)
                if not k or k not in fa:
                    continue
                d_oi = c[1] - a[1]                            # contracts, all months
                p0, p1 = fa[k][3], fc[k][3]
                if not p0 or not p1:
                    continue
                if s not in lots:
                    lots[s] = _lot(fc, oi_base.get(s), s)
                lot = lots[s]
                if not lot:
                    continue
                # OPTIONS: calls / puts opened or closed in the block, every strike and
                # expiry, valued at the CASH price (option notional = contracts x lot x
                # underlying — exposure, NOT premium paid; deep out-of-the-money contracts
                # count the same as at-the-money ones). Same lot as the futures. Judged on
                # the same futures price move, so the two charts line up block for block.
                spot = e["x"].get(s, {}).get("S") or p1
                mv = ("price unchanged" if p1 == p0 else "price up" if p1 > p0 else
                      "price down")
                for leg, j in (("Calls", 3), ("Puts", 5)):
                    d_o = c[j] - a[j]
                    if d_o:
                        oacc[(leg, mv)][0] += d_o * lot * spot / 1e7
                        oacc[(leg, mv)][1] += 1
                if not d_oi:
                    starts.append(pts)
                    continue
                cr = d_oi * lot * p1 / 1e7
                up = p1 > p0
                q = ("price unchanged" if p1 == p0 else
                     "long build" if d_oi > 0 and up else "short build" if d_oi > 0 else
                     "short covering" if up else "long unwind")
                acc[q][0] += cr
                acc[q][1] += 1
                starts.append(pts)
            for q, (cr, n) in acc.items():
                if n:
                    flow.append(dict(end=ts, start=min(starts), bucket=lbl, quadrant=q,
                                     cr=float(cr), stocks=n))
            for (leg, mv), (cr, n) in oacc.items():
                if n:
                    oflow.append(dict(end=ts, start=min(starts), bucket=lbl, leg=leg,
                                      move=mv, cr=float(cr), stocks=n))
    fl = pd.DataFrame(flow, columns=["end", "start", "bucket", "quadrant", "cr", "stocks"])
    of = pd.DataFrame(oflow, columns=["end", "start", "bucket", "leg", "move", "cr", "stocks"])
    pr = pd.DataFrame(price).set_index("end").astype(float) if price else pd.DataFrame()
    sharp = []
    if not fl.empty:
        for lbl in buckets:
            g = fl[fl.bucket == lbl].groupby("end")
            net = g["cr"].sum()
            closed = fl[(fl.bucket == lbl) & (fl.cr < 0)].groupby("end")["cr"].sum()
            added = fl[(fl.bucket == lbl) & (fl.cr > 0)].groupby("end")["cr"].sum()
            for j, (end, v) in enumerate(net.items()):
                hist = net.iloc[:j].abs()
                if len(hist) < FLOW_SHARP_MIN:
                    continue
                typ = hist.median()
                if typ > 0 and abs(v) >= FLOW_SHARP_X * typ:
                    gq = fl[(fl.bucket == lbl) & (fl.end == end)]
                    top = gq.loc[gq.cr.abs().idxmax()]
                    sharp.append(dict(end=end, bucket=lbl, net=float(v),
                                      times=float(abs(v) / typ), main=top.quadrant,
                                      main_cr=float(top.cr),
                                      added=float(added.get(end, 0.0)),
                                      closed=float(closed.get(end, 0.0))))
    osharp = []
    if not of.empty:
        for lbl in buckets:
            for leg in ("Calls", "Puts"):
                net = of[(of.bucket == lbl) & (of.leg == leg)].groupby("end")["cr"].sum()
                for j, (end, v) in enumerate(net.items()):
                    hist = net.iloc[:j].abs()
                    if len(hist) < FLOW_SHARP_MIN or not hist.median() > 0:
                        continue
                    if abs(v) >= OPT_SHARP_X * hist.median():
                        g = of[(of.bucket == lbl) & (of.leg == leg) & (of.end == end)]
                        top = g.loc[g.cr.abs().idxmax()]
                        osharp.append(dict(end=end, bucket=lbl, leg=leg, net=float(v),
                                           times=float(abs(v) / hist.median()),
                                           move=top.move))
    return dict(flow=fl, price=pr, sharp=sharp, options=of, osharp=osharp)


BULL = ("long build", "short covering")       # price UP: buyers adding, or sellers leaving
BEAR = ("short build", "long unwind")         # price DOWN: sellers adding, or buyers leaving


def by_side(flow: pd.DataFrame) -> pd.DataFrame:
    """Re-sign the block flow by SIDE instead of by opened/closed.

    The raw bars put opened positions above zero and closed ones below, which drew short
    covering (bullish: price up, shorts leaving) pointing DOWN beside long unwind (bearish).
    Here every rupee of a bullish quadrant plots UP and every rupee of a bearish one DOWN, at
    its absolute size. 'price unchanged' has no side and is left out."""
    if flow.empty:
        return flow.assign(side_cr=pd.Series(dtype=float))
    f = flow[flow.quadrant.isin(BULL + BEAR)].copy()
    f["side_cr"] = np.where(f.quadrant.isin(BULL), f.cr.abs(), -f.cr.abs())
    return f


def pressure_series(flow: pd.DataFrame, labels: list) -> pd.DataFrame:
    """Running total through the session of (bullish rupees - bearish rupees), per bucket,
    one point per block. Rising = the bullish quadrants are out-trading the bearish ones."""
    f = by_side(flow)
    if f.empty:
        return pd.DataFrame()
    net = f.pivot_table(index="end", columns="bucket", values="side_cr", aggfunc="sum")
    return net.reindex(columns=[c for c in labels if c in net.columns]).fillna(0.0).cumsum()


def pressure_level(oi_tape: list, buckets: dict, oi_base: dict, fut_close: dict,
                   day: dt.date, since: str = "close", band: float = 0.0) -> dict:
    """Bullish minus bearish futures rupees per bucket at EVERY reading — block-free.

    WHY NOT ADD UP BLOCKS. Summing block-by-block labels was measured scale-dependent: on
    16 Oct 2026 Nifty Top 10 ended at -169 / +350 / -511 / +787 / +884 / +926 Cr for 1 / 2 /
    5 / 10 / 15 / 30-minute blocks, and the sign flipped by block size on every index and
    day checked. A block is labelled by its NET price move, so how the session is sliced
    decides which way the same positions are counted.

    WHAT THIS DOES INSTEAD. At each reading, for each stock: its futures OI change and its
    near-month futures price change, both since `since` ("close" = yesterday's close, "open"
    = the day's first reading). The rupee value of that OI change (contracts x lot x price)
    goes to the stock's quadrant — price up: long build (OI up) or short covering (OI down);
    price down: short build or long unwind. Bullish = long build + short covering, bearish
    = short build + long unwind. The value depends only on the two end points, so it cannot
    change with any block setting, and with since="close" its last point is the main
    table's "Futures now" rule in rupees (futures price instead of cash, no 0.5% floor —
    near-flat stocks can differ: 3 of 50 Nifty names on 16 Oct 2026).

    A stock missing from a reading keeps its last value; a bucket is drawn once
    SERIES_QUORUM of its stocks have been read; stocks whose price did not move since the
    baseline have no side and are counted in "unsided"."""
    rows = [e for e in oi_tape if e.get("x")]
    empty = dict(net=pd.DataFrame(), parts=pd.DataFrame(), price=pd.DataFrame())
    if not rows:
        return empty
    lots: dict = {}
    base0: dict = {}                     # since="open": stock -> (chg, ltp) at first sight
    cur: dict = {}                       # stock -> (quadrant, rupees, price %)
    side: dict = {}                      # stock -> +1 / -1, held inside the band
    net_rows, part_rows, px_rows = [], [], []
    for e in rows:
        ts = pd.Timestamp(e["t"])
        for s, d in e["d"].items():
            F = e["x"].get(s, {}).get("F", {})
            k = _near(F, day)
            if not k:
                continue
            ltp = F[k][3]
            prev = F[k][4] if len(F[k]) > 4 and F[k][4] else fut_close.get(s, {}).get(k)
            if s not in lots or lots[s] is None:
                lots[s] = _lot(F, oi_base.get(s), s)
            if not lots[s] or not ltp:
                continue
            if since == "open":
                if s not in base0:
                    base0[s] = (d[1], ltp)
                d_oi, p0 = d[1] - base0[s][0], base0[s][1]
            else:
                d_oi, p0 = d[1], prev
            if not p0:
                continue
            dp = ltp / p0 - 1
            cr = abs(d_oi) * lots[s] * ltp / 1e7
            # HYSTERESIS, not a dead band: a stock takes a side once its price is more than
            # `band` % from the baseline and KEEPS it until price is more than `band` % on the
            # other side. A plain dead band only split each flip into two half-jumps (bull ->
            # no side -> bear) and left the line exactly as jumpy (measured).
            if dp * 100 > band:
                side[s] = 1
            elif dp * 100 < -band:
                side[s] = -1
            sd = side.get(s)
            q = (None if sd is None or d_oi == 0 else
                 "long build" if sd > 0 and d_oi > 0 else "short covering" if sd > 0 else
                 "short build" if d_oi > 0 else "long unwind")
            cur[s] = (q, cr, dp * 100)
        nrow, prow = {"t": ts}, {"t": ts}
        for lbl, mem in buckets.items():
            seen = [s for s in mem if s in cur]
            if len(seen) < max(1, int(np.ceil(len(mem) * SERIES_QUORUM))):
                nrow[lbl] = None
                continue
            parts = {q: 0.0 for q in QUADRANTS[:4]}
            unsided = 0.0
            for s in seen:
                q, cr, _ = cur[s]
                if q is None:
                    unsided += cr
                else:
                    parts[q] += cr
            nrow[lbl] = (parts["long build"] + parts["short covering"]
                         - parts["short build"] - parts["long unwind"])
            part_rows.append(dict(t=ts, bucket=lbl, unsided=unsided, **parts))
        net_rows.append(nrow)
    return dict(net=pd.DataFrame(net_rows).set_index("t").astype(float),
                parts=pd.DataFrame(part_rows))


def net_blocks(net: pd.Series, minutes: int = 5) -> pd.DataFrame:
    """[end, cum, step] per block from a running bullish-minus-bearish series.

    cum  = the running total at the block's last reading (it does not depend on blocks)
    step = how much it moved during the block (cum minus the previous block's cum; the first
           block is measured from zero, the day's baseline)
    The steps therefore add up EXACTLY to the line, whatever block length is chosen — the
    bars and the line can never tell different stories. A block that follows a logging gap
    (more than 3 blocks) gets no bar: the move across the gap is not a 5-minute event."""
    s = net.dropna()
    if s.empty:
        return pd.DataFrame(columns=["end", "cum", "step", "gap"])
    blk = s.groupby(s.index.floor(f"{minutes}min")).last()
    ends = s.groupby(s.index.floor(f"{minutes}min")).apply(lambda x: x.index[-1])
    out = pd.DataFrame({"end": ends.values, "cum": blk.values}, index=blk.index)
    out["step"] = out["cum"].diff().fillna(out["cum"])
    gaps = out.index.to_series().diff() > pd.Timedelta(minutes=minutes * 3)
    out["gap"] = gaps.values
    out.loc[out["gap"], "step"] = np.nan
    return out.reset_index(drop=True)


def call_put_series(oi_tape: list, buckets: dict, oi_base: dict, day: dt.date) -> pd.DataFrame:
    """Per reading, per bucket: (calls - puts) opened since YESTERDAY'S CLOSE, Rs crore.

    For every stock: (today's call OI change - today's put OI change) x lot x share price,
    summed over the bucket. Above zero = more call exposure than put exposure added today
    (call side heavier); below zero = put side heavier. Every strike and expiry counts, valued
    at the share price — exposure, not premium paid.

    A stock missing from one reading keeps its last value instead of dropping out, so the
    bucket line does not jump down and back up when NSE skips one name in one sweep. A
    bucket is drawn only once SERIES_QUORUM of its stocks have been seen."""
    rows = [e for e in oi_tape if e.get("x")]
    if not rows:
        return pd.DataFrame()
    last: dict = {}                     # stock -> latest (calls - puts) Rs Cr
    lots: dict = {}
    out = []
    for e in rows:
        for s, d in e["d"].items():
            x = e["x"].get(s, {})
            F = x.get("F", {})
            if s not in lots or lots[s] is None:
                lots[s] = _lot(F, oi_base.get(s), s)
            if not lots[s]:
                continue
            k = _near(F, day)
            spot = x.get("S") or (F[k][3] if k else None)
            if not spot:
                continue
            last[s] = (d[3] - d[5]) * lots[s] * spot / 1e7
        row = {"t": pd.Timestamp(e["t"])}
        for lbl, mem in buckets.items():
            have = [last[s] for s in mem if s in last]
            row[lbl] = (float(sum(have))
                        if len(have) >= max(1, int(np.ceil(len(mem) * SERIES_QUORUM)))
                        else None)
        out.append(row)
    return pd.DataFrame(out).set_index("t").astype(float)


CP_RATE_CAP = 20          # sessions to expiry at or beyond this share one cycle-rate bucket
CP_RATE_FROM = "2024-07-24"   # first session of the stock-options archive


def cp_cycle_rate(symbols: tuple, day: dt.date) -> dict | None:
    """{sessions to expiry: median (calls - puts opened in a day) / (average call/put open
    interest notional)} for these stocks, from every archived session BEFORE `day`.

    WHY NOT DCM's _opt_cycle_norm. That norm is an equal-weighted average of stock % changes,
    capped at -3 points; in RUPEES it fits the last week of a cycle (Nifty, day before
    expiry: raw -1,424 Cr, norm -1,280) but under-corrects the first weeks ~2x, because new
    series fill from almost nothing. Measured out of sample (fit Jul-2024..Sep-2025, judged
    Sep-2025..Oct-2026), share of bucket-days reading 'call side heavier':
        Nifty      first week / day before expiry / all days
        raw          95%  /  2%  / 67%
        DCM norm     83%  / 45%  / 65%
        this rate    51%  / 67%  / 52%       (Bank Nifty 53% / 50% / 53%)
    One median per sessions-to-expiry, pooled over the stocks given (today's basket applied
    to history — approximate across reviews). Cached per day in memory and on disk."""
    key = ("cprate", day, symbols)
    if key in _CTX:
        return _CTX[key]
    if time.time() - _CTX.get(("cprate-miss",) + key, 0.0) < ARCHIVE_TTL_S:
        return None
    import hashlib
    # a STABLE name: hash() of a str is salted per process, so every process (dashboard,
    # capture task) would have written its own copy
    tag = hashlib.md5(",".join(symbols).encode()).hexdigest()[:10]
    path = DATA_DIR / f"cp_rate_{tag}_{day:%Y%m%d}.json"
    try:
        hit = json.loads(path.read_text(encoding="utf-8"))
        if hit.get("symbols") == list(symbols):
            _CTX[key] = {int(k): v for k, v in hit["rate"].items()}
            return _CTX[key]
    except (OSError, ValueError):
        pass
    out = None
    try:
        from . import data
        ph = ",".join("?" * len(symbols))
        with data._connect() as c:
            df = c.execute(f"""
                with dd as (select trade_date, lag(trade_date) over (order by trade_date) prev
                            from (select distinct trade_date from fno_bhavcopy
                                  where trade_date >= ? and trade_date < ?)),
                o as (select trade_date, symbol, option_type, expiry_date,
                             sum(open_interest) oi from fno_bhavcopy
                      where instrument = 'OPTSTK' and trade_date >= ? and trade_date < ?
                        and symbol in ({ph}) group by all),
                t as (select dd.trade_date, o.symbol,
                             sum(case when o.option_type = 'CE' then o.oi end) ce,
                             sum(case when o.option_type = 'PE' then o.oi end) pe
                      from dd join o on o.trade_date = dd.trade_date
                       and o.expiry_date >= dd.trade_date group by all),
                y as (select dd.trade_date, o.symbol,
                             sum(case when o.option_type = 'CE' then o.oi end) ce,
                             sum(case when o.option_type = 'PE' then o.oi end) pe
                      from dd join o on o.trade_date = dd.prev
                       and o.expiry_date >= dd.trade_date group by all),
                px as (select trade_date, symbol, close_price from daily_data
                       where series = 'EQ' and trade_date >= ? and trade_date < ?
                         and symbol in ({ph})),
                fr as (select trade_date, min(expiry_date) e from fno_bhavcopy
                       where instrument = 'FUTSTK' and trade_date >= ? and trade_date < ?
                       group by 1)
                select y.trade_date, fr.e front,
                       sum(((coalesce(t.ce, 0) - y.ce) - (coalesce(t.pe, 0) - y.pe))
                           * px.close_price) raw,
                       sum((y.ce + y.pe) / 2 * px.close_price) n
                from y left join t using (trade_date, symbol)
                join px using (trade_date, symbol) join fr using (trade_date)
                where y.ce > 0 and y.pe > 0 group by 1, 2 order by 1""",
                [CP_RATE_FROM, day, CP_RATE_FROM, day, *symbols, CP_RATE_FROM, day, *symbols,
                 CP_RATE_FROM, day]).df()
        if len(df) >= 60:
            days = list(pd.to_datetime(df["trade_date"]))
            pos = {d: i for i, d in enumerate(days)}
            fronts = pd.to_datetime(df["front"])
            s2e = [pos.get(f) - pos[d] if f in pos else None for d, f in zip(days, fronts)]
            df["s2e"] = s2e
            df = df.dropna(subset=["s2e"])
            df = df[df["s2e"] > 0]                        # expiry day: the chart is off
            df["rate"] = df["raw"] / df["n"]
            df["s2e"] = df["s2e"].astype(int).clip(upper=CP_RATE_CAP)
            med = df.groupby("s2e")["rate"].median()
            out = {int(k): float(v) for k, v in med.items()}
            # only the current day goes on disk: viewing a past session must not delete
            # today's file (the past day's rate is rebuilt in under a second if needed)
            if day >= _now().date():
                _write_json(path, dict(symbols=list(symbols), rate=out))
                for old in DATA_DIR.glob(f"cp_rate_{tag}_*.json"):
                    if old != path:
                        try:
                            old.unlink()
                        except OSError:
                            pass
    except Exception:                                        # noqa: BLE001
        out = None
    if out is None:
        # a miss (locked archive, too little history) is retried after ARCHIVE_TTL_S, not
        # remembered for the day — and not retried on every 15-second redraw either
        _CTX[("cprate-miss",) + key] = time.time()
    else:
        _CTX[key] = out
    return out


def call_put_usual(oi_tape: list, buckets: dict, oi_base: dict, day: dt.date,
                   rate: float | None) -> pd.DataFrame:
    """Per reading, per bucket: the calls - puts rupees the EXPIRY CYCLE alone would put on
    the line by that time of day — the part of call_put_series that is calendar, not view.

    `rate` is cp_cycle_rate()[sessions to expiry]: a typical full day's (calls - puts) as a
    share of the average call/put open interest notional. Per stock: rate x (yesterday's
    call + put OI in shares) / 2 x share price, pro-rated through the day by
    session_fraction() — the same unmeasured intraday assumption the table's 'Options now'
    label makes. A stock keeps its last value when NSE skips it, and the quorum matches
    call_put_series, so the two frames line up reading for reading."""
    rows = [e for e in oi_tape if e.get("x")]
    if not rows or rate is None:
        return pd.DataFrame()
    last: dict = {}
    out = []
    for e in rows:
        ts = pd.Timestamp(e["t"])
        frac = session_fraction(ts.to_pydatetime())
        for s in e["d"]:
            b = oi_base.get(s)
            x = e["x"].get(s, {})
            F = x.get("F", {})
            k = _near(F, day)
            spot = x.get("S") or (F[k][3] if k else None)
            # SAME MEMBERSHIP AS THE LINE: call_put_series drops a stock until its lot is
            # proven; counting it here anyway pushed the adjusted line toward 'put side' in
            # the first minutes (up to 11 of 25 Midcap names at 09:17 on 16 Oct, no memory)
            if not b or not spot or not _lot(F, b, s):
                continue
            last[s] = rate * frac * (b["ce"] + b["pe"]) / 2 * spot / 1e7
        row = {"t": ts}
        for lbl, mem in buckets.items():
            have = [last[s] for s in mem if s in last]
            row[lbl] = (float(sum(have))
                        if len(have) >= max(1, int(np.ceil(len(mem) * SERIES_QUORUM)))
                        else None)
        out.append(row)
    return pd.DataFrame(out).set_index("t").astype(float)


def option_positioning(oi_tape: list, buckets: dict, mode: str = "iv") -> dict:
    """{"last": {bucket: {"CE"|"PE": {quadrant|"unread": [contracts, Rs Cr]}}},
        "net": frame [t x bucket] bullish - bearish Rs Cr, "n": stocks read}.
    From the per-stock option reads logged with each sweep (option_flow). A stock NSE skipped
    keeps its last read; a bucket is drawn once SERIES_QUORUM of its stocks have one."""
    cur: dict = {}
    rows, rows_leg = [], {"CE": [], "PE": []}
    for e in oi_tape:
        for s, q in (e.get("q") or {}).items():
            if q and q.get(mode):
                cur[s] = q[mode]
        if not cur:
            continue
        t = pd.Timestamp(e["t"])
        row, legrow = {"t": t}, {"CE": {"t": t}, "PE": {"t": t}}
        for lbl, mem in buckets.items():
            have = [cur[s] for s in mem if s in cur]
            if len(have) < max(1, int(np.ceil(len(mem) * SERIES_QUORUM))):
                row[lbl] = legrow["CE"][lbl] = legrow["PE"][lbl] = None
                continue
            per = {"CE": 0.0, "PE": 0.0}
            for q in have:
                for leg in ("CE", "PE"):
                    for i, name in enumerate(OPT_QUADS):
                        per[leg] += q[leg][i][1] if name in OPT_BULL[leg] else -q[leg][i][1]
            row[lbl] = per["CE"] + per["PE"]
            legrow["CE"][lbl], legrow["PE"][lbl] = per["CE"], per["PE"]
        rows.append(row)
        for leg in ("CE", "PE"):
            rows_leg[leg].append(legrow[leg])
    last = {}
    for lbl, mem in buckets.items():
        agg = {leg: {q: [0.0, 0.0] for q in (*OPT_QUADS, "unread")} for leg in ("CE", "PE")}
        for s in mem:
            if s in cur:
                for leg in ("CE", "PE"):
                    for i, name in enumerate((*OPT_QUADS, "unread")):
                        agg[leg][name][0] += cur[s][leg][i][0]
                        agg[leg][name][1] += cur[s][leg][i][1]
        last[lbl] = agg
    def frame(rs):
        return pd.DataFrame(rs).set_index("t").astype(float) if rs else pd.DataFrame()
    # net = calls + puts together; net_leg = each side on its own (the page draws these two)
    return dict(last=last, net=frame(rows), n=len(cur),
                net_leg={leg: frame(rows_leg[leg]) for leg in ("CE", "PE")})


def observed_refresh(tape: list) -> float | None:
    """Median seconds between distinct NSE price stamps in the log. An UPPER bound on NSE's
    real interval — the log cannot see a refresh faster than this module polls."""
    ts = [pd.Timestamp(e["t"]) for e in tape[-40:]]
    gaps = [(b - a).total_seconds() for a, b in zip(ts[:-1], ts[1:])
            if (b - a).total_seconds() < 600]          # skip gaps where the board was shut
    return float(np.median(gaps)) if len(gaps) >= 5 else None


_CADENCE: dict = {"mtime": None, "info": None}


def tuned_intervals(result: dict) -> tuple:
    """(price poll seconds, OI sweep seconds) from a measure_cadence() result, or Nones.

    Polling faster than NSE publishes buys nothing, and slower throws data away, so each
    interval follows the measured median — inside hard limits. Prices: 15-60s. OI: 60-180s,
    because one sweep is 50 calls and NSE's tolerance for more than that is unknown. A feed
    that changed fewer than 5 times in the run is not timed at all: the default stands.
    """
    def med(k):
        r = result.get(k) or {}
        return r.get("median_s") if (r.get("changes") or 0) >= 5 else None
    px, oi = med("price stamp"), med("stock OI (futures)")
    return (None if px is None else float(min(max(px, 15.0), 60.0)),
            None if oi is None else float(min(max(oi, 60.0), 180.0)))


def apply_cadence() -> dict | None:
    """Adopt the intervals from data/index_mgmt/cadence.json if a measurement is there.
    Called from the poller loop, so a measurement taken while the board is running takes
    effect without a restart. Returns what was applied, for the page to show."""
    global CASH_EVERY, DERIV_EVERY
    f = DATA_DIR / "cadence.json"
    try:
        m = f.stat().st_mtime
    except OSError:
        return _CADENCE["info"]
    if m == _CADENCE["mtime"]:
        return _CADENCE["info"]
    _CADENCE["mtime"] = m
    try:
        j = json.loads(f.read_text(encoding="utf-8"))
        px, oi = tuned_intervals(j["result"])
    except Exception:                                        # noqa: BLE001
        return _CADENCE["info"]
    if px:
        CASH_EVERY = px
    if oi:
        DERIV_EVERY = oi
    _CADENCE["info"] = dict(measured=j.get("measured"), price_s=px, oi_s=oi,
                            result=j["result"])
    return _CADENCE["info"]


def measure_cadence(code: str = "NIFTY", symbol: str = "RELIANCE", minutes: float = 6.0,
                    every: float = 5.0, client=None) -> dict:
    """Poll NSE every `every` seconds and time how often each feed actually changes.

    Run it in market hours. It answers the question the 30s/120s poll settings rest on:
    how often does NSE put out a new price stamp, and how often does one stock's open
    interest change. ~145 calls for the default six minutes.
    """
    client = client or NseClient()
    seen = {"price stamp": [], "stock OI (futures)": [], "stock OI payload stamp": []}
    last = {}
    end = time.time() + minutes * 60
    while time.time() < end:
        t0 = time.time()
        try:
            c = fetch_cash(code, client)
            d = fetch_deriv(symbol, client)
            cur = {"price stamp": c["ts"], "stock OI (futures)": (d["fut_oi"], d["fut_chg"]),
                   "stock OI payload stamp": d["ts"]}
            for k, v in cur.items():
                if k in last and last[k] != v:
                    seen[k].append(time.time())
                last[k] = v
        except NseError as exc:
            print("  poll failed:", exc)
        time.sleep(max(0.0, every - (time.time() - t0)))
    out = {}
    for k, ts in seen.items():
        gaps = np.diff(ts)
        out[k] = dict(changes=len(ts),
                      median_s=float(np.median(gaps)) if len(gaps) else None,
                      min_s=float(gaps.min()) if len(gaps) else None,
                      max_s=float(gaps.max()) if len(gaps) else None)
    return out


# ── archive context (one read per day, never in the render path twice) ────────────────
_CTX: dict = {}
_CTX_T: dict = {}                     # key -> when it was cached (only for _ctx_fresh users)
ARCHIVE_TTL_S = 600


def _ctx_fresh(key) -> bool:
    """True when `key` is cached and younger than ARCHIVE_TTL_S. The previous-session base
    used to be cached for the whole day, so a bhavcopy that landed late (26 Oct 2026: DCM
    refused NSE's partial 19:30 file and the full one went in next morning) never reached a
    running dashboard until restart. Re-reading every 10 minutes picks it up mid-session."""
    if key not in _CTX:
        return False
    if key not in _CTX_T:
        _CTX_T[key] = time.time()             # first look after a store starts the clock
    if time.time() - _CTX_T[key] < ARCHIVE_TTL_S:
        return True
    del _CTX_T[key]                           # expired: the caller re-reads and re-stores
    return False


def _archive_norms(symbols: tuple, before: dt.date | None = None) -> dict:
    """{symbol: median daily traded value, Rs Cr, last 21 sessions}. {} if the archive is
    locked — activity is then simply not shown. `before`: only sessions before that day, so
    a past session is judged against what was normal THEN, not against later trading."""
    key = ("norm", before or dt.date.today(), symbols)
    if key in _CTX:
        return _CTX[key]
    out = {}
    try:
        from . import data
        ph = ",".join("?" * len(symbols))
        cut = "and trade_date < ?" if before else ""
        with data._connect() as c:
            df = c.execute(f"""
                select symbol, median(turnover_lacs) / 100 as cr from (
                    select symbol, turnover_lacs,
                           row_number() over (partition by symbol order by trade_date desc) rn
                    from daily_data where symbol in ({ph}) and series = 'EQ' {cut})
                where rn <= 21 group by 1""", [*symbols, *([before] if before else [])]).df()
        out = {r.symbol: float(r.cr) for r in df.itertuples() if r.cr and r.cr > 0}
        _CTX[key] = out
    except Exception:                                        # noqa: BLE001
        pass
    return out


def _archive_corp(symbols: tuple, day: dt.date) -> dict:
    """{symbol: 'SPLIT'|'BONUS'|'DEMERGER'|'SCHEME'} with an ex-date of `day`. From the
    official table only — a re-basing is never inferred from the tape."""
    key = ("corp", day, symbols)
    if key in _CTX:
        return _CTX[key]
    out = {}
    try:
        from . import data
        ph = ",".join("?" * len(symbols))
        with data._connect() as c:
            rows = c.execute(f"""
                select symbol, kind from corp_actions
                where ex_date = ? and symbol in ({ph})
                  and (kind in ('DEMERGER', 'SCHEME')
                       or (share_factor > 0 and share_factor <> 1))""",
                             [day, *symbols]).fetchall()
        out = {s: k for s, k in rows}
        _CTX[key] = out
    except Exception:                                        # noqa: BLE001
        pass
    return out


def _archive_oi_base(symbols: tuple, day: dt.date) -> dict:
    """{symbol: dict(fut, ce, pe, fut_by_exp, date)} — END-OF-DAY open interest (in shares)
    of the last archived session BEFORE `day`, over contracts still alive on `day`."""
    key = ("oibase", day, symbols)
    if _ctx_fresh(key):
        return _CTX[key]
    out = {}
    try:
        from . import data
        ph = ",".join("?" * len(symbols))
        with data._connect() as c:
            prev = c.execute("select max(trade_date) from fno_bhavcopy where trade_date < ?",
                             [day]).fetchone()[0]
            if prev is None:
                return out
            df = c.execute(f"""
                select symbol, instrument, option_type, expiry_date, sum(open_interest) oi
                from fno_bhavcopy
                where trade_date = ? and expiry_date >= ? and open_interest > 0
                  and instrument in ('FUTSTK', 'OPTSTK') and symbol in ({ph})
                group by 1, 2, 3, 4""", [prev, day, *symbols]).df()
        for sym, g in df.groupby("symbol"):
            f = g[g["instrument"] == "FUTSTK"]
            o = g[g["instrument"] == "OPTSTK"]
            out[sym] = dict(
                fut=float(f["oi"].sum()),
                ce=float(o.loc[o["option_type"] == "CE", "oi"].sum()),
                pe=float(o.loc[o["option_type"] == "PE", "oi"].sum()),
                fut_by_exp={pd.Timestamp(r.expiry_date).date(): float(r.oi)
                            for r in f.itertuples()},
                date=pd.Timestamp(prev).date())
        _CTX[key] = out
    except Exception:                                        # noqa: BLE001
        # the 10-minute re-read can meet a locked archive (a DCM write in progress): keep
        # the last good base rather than blanking every OI column for that redraw
        out = _CTX.get(key, out)
    return out


_BAN_URL = "https://nsearchives.nseindia.com/content/fo/fo_secban.csv"


def fetch_ban_list(day: dt.date, session=None) -> set | None:
    """Stocks in the F&O ban period on `day`, from NSE's daily file, or None if unreadable.

    In the ban period nobody may ADD derivative positions — open interest can only fall. So a
    banned stock's futures can only read 'short covering' or 'long unwind', and that is the
    rule, not a view. Its votes are withheld like a corporate action's. The file names its
    trade date in the first line ("Securities in Ban For Trade Date 14-OCT-2026:"); a file
    for any other day is ignored, never assumed current."""
    key = ("ban", day)
    if key in _CTX:
        return _CTX[key]
    # A MISS IS REMEMBERED TOO, for 10 minutes. It used to return None uncached, so on any
    # day the file did not match (every weekend, when NSE already lists the next session) the
    # page asked NSE again on every 15-second redraw: 0.6s of network inside the draw.
    miss = _CTX.get(("ban-miss", day))
    if miss and time.time() - miss < 600 and session is None:
        return None
    try:
        s = session or requests.Session()
        s.headers.update(_HEADERS)
        r = s.get(_BAN_URL, timeout=15)
        if r.status_code != 200:
            _CTX[("ban-miss", day)] = time.time()
            return None
        lines = [x.strip() for x in r.text.splitlines() if x.strip()]
        if not lines or f"{day:%d-%b-%Y}".upper() not in lines[0].upper():
            _CTX[("ban-miss", day)] = time.time()
            return None
        out = {x.split(",", 1)[1].strip() for x in lines[1:] if "," in x}
        _CTX[key] = out
        return out
    except Exception:                                        # noqa: BLE001
        _CTX[("ban-miss", day)] = time.time()
        return None


def _archive_fut_close(symbols: tuple, day: dt.date) -> dict:
    """{symbol: {expiry iso: previous session's futures close}} from the archive."""
    key = ("fclose", day, symbols)
    if _ctx_fresh(key):
        return _CTX[key]
    out: dict = {}
    try:
        from . import data
        ph = ",".join("?" * len(symbols))
        with data._connect() as c:
            prev = c.execute("select max(trade_date) from fno_bhavcopy where trade_date < ?",
                             [day]).fetchone()[0]
            rows = c.execute(f"""
                select symbol, expiry_date, close_price from fno_bhavcopy
                where trade_date = ? and instrument = 'FUTSTK' and expiry_date >= ?
                  and close_price > 0 and symbol in ({ph})""", [prev, day, *symbols]).fetchall()
        for s, e, cp in rows:
            out.setdefault(s, {})[pd.Timestamp(e).date().isoformat()] = float(cp)
        _CTX[key] = out
    except Exception:                                        # noqa: BLE001
        out = _CTX.get(key, out)              # locked archive: keep the last good read
    return out


def previous_close_read(code: str, before: dt.date | None = None):
    """DCM's own end-of-day read for the archive's last close, or (None, None). Shown beside
    the live read so "short build yesterday, covering now" is visible on one screen.
    `before`: the last close before that day, for a past session's 'yesterday'."""
    key = ("eod", before or dt.date.today(), code)
    # re-read on the archive TTL too: a late F&O file (26 Oct 2026) otherwise leaves this
    # box built without its futures/options words until the dashboard restarts
    if _ctx_fresh(key):
        return _CTX[key]
    out = (None, None)
    m = _dcm()
    if m is not None:
        try:
            from . import data
            if before:
                with data._connect() as c:
                    last = pd.Timestamp(c.execute(
                        "select max(trade_date) from daily_data where trade_date < ?",
                        [before]).fetchone()[0]).date()
            else:
                last = data.last_trading_date().date()
            d = m.get_index_largecap(last, code)
            if d.data_ok:
                out = (last, d)
                _CTX[key] = out
        except Exception:                                    # noqa: BLE001
            out = _CTX.get(key, out)          # locked archive: keep the last good read
    return out


def state_base_rates(code: str):
    key = ("base", dt.date.today(), code)
    if key in _CTX:
        return _CTX[key]
    m = _dcm()
    try:
        df = m.get_state_base_rates(code) if m is not None else None
        if df is not None and not df.empty:
            _CTX[key] = df
            return df
    except Exception:                                        # noqa: BLE001
        pass
    return None


# ── shared state + background poller ──────────────────────────────────────────────────
# Module-level on purpose: this is MARKET data, identical for every browser session, and
# the poller must outlive a single Streamlit rerun. Nothing per-user lives here.
_LOCK = threading.RLock()
_S: dict = {}                    # code -> dict(cash, weights, weights_at, tape, ...)
# symbol -> its latest OI read (+ "at"). Keyed by STOCK, not by index: HDFCBANK sits in three
# of the four indices, so switching from Nifty to Bank Nifty finds 5 of its 14 names already
# swept instead of asking NSE for them again.
_D: dict = {}
_CLIENT: list = [None]
_GEN_ATTR = "_eqbtst_index_mgmt_gen"


def _client() -> NseClient:
    if _CLIENT[0] is None:
        _CLIENT[0] = NseClient()
    return _CLIENT[0]


def _st(code: str) -> dict:
    with _LOCK:
        return _S.setdefault(code, dict(
            cash=None, cash_at=0.0, weights=None, weights_at=0.0, weights_day=None,
            deriv_at=0.0, deriv_ts=None, deriv_busy=False, err=None,
            viewed=0.0, tape=[], oi_tape=[], tape_day=None))


def _path(kind: str, code: str, day: dt.date) -> Path:
    return DATA_DIR / f"{kind}_{code}_{day:%Y%m%d}.jsonl"


def _append(path: Path, obj: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, separators=(",", ":")) + "\n")
    except OSError:
        pass                                 # the tape is a convenience; never break a poll


def compress_old_logs(today: dt.date | None = None) -> int:
    """gzip every finished day's .jsonl (about 10x smaller); today's stays plain because it
    is still being appended to. The .gz is verified to read back before the plain file goes."""
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


def _load_jsonl(path: Path) -> list:
    out = []
    if not path.exists() and path.with_suffix(".jsonl.gz").exists():
        import gzip
        try:
            with gzip.open(path.with_suffix(".jsonl.gz"), "rt", encoding="utf-8") as f:
                return [json.loads(x) for x in f if x.strip()]
        except (OSError, ValueError):
            return []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue                 # a half-written last line after a kill
    except OSError:
        pass
    return out


def _load_weights_file(code: str):
    try:
        j = json.loads((DATA_DIR / f"weights_{code}.json").read_text(encoding="utf-8"))
        day = dt.date.fromisoformat(j["day"])
        if (dt.date.today() - day).days <= 7:      # weights drift slowly; a week is the limit
            return j["w"], day
    except Exception:                                        # noqa: BLE001
        pass
    return None, None


# ── ONE WRITER, ANY NUMBER OF READERS ─────────────────────────────────────────────────
# Capture must start at 09:15 whether or not anyone has the page open — on 14 Oct 2026 the
# log began at 10:58, when the page was first opened, so the session trend and the
# last-15/30/60-minute read had nothing before that. So capture runs in whichever process
# gets there first: the dashboard (it starts the poller on load) or the scheduled
# `python -m eqbtst.index_mgmt capture` job. A lock file decides which ONE polls NSE and
# writes; any other process reads what it wrote. Two writers would double every log row
# and double the calls to NSE.
_LOCKFILE = "poller.lock"
_LOCK_STALE = 60.0           # a writer that has not refreshed the lock for this long is gone
_MINE = {"owner": False}


def _claim() -> bool:
    """True if this process is (or now becomes) the one writer."""
    f = DATA_DIR / _LOCKFILE
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
    f = DATA_DIR / _LOCKFILE
    try:
        if json.loads(f.read_text(encoding="utf-8")).get("pid") == os.getpid():
            f.unlink()
    except (OSError, ValueError):
        pass
    _MINE["owner"] = False


def in_session(ts: dt.datetime | None) -> bool:
    """09:15-15:30 by the clock. NSE kept answering 'Open' for minutes after 15:30 on
    14 Oct 2026 while OI kept moving (+0.11% between 15:17 and 15:32 with prices frozen) —
    post-close reconciliation, not trading. Rows outside the session are not logged, so a
    'last 15 minutes' read can never be built from them."""
    return ts is not None and dt.time(9, 15) <= ts.time() <= dt.time(15, 30)


def market_window(now: dt.datetime | None = None) -> bool:
    """Mon-Fri 09:00-15:40 IST — when capture polls at full speed without being asked.
    Holidays are not special-cased: NSE then answers 'Closed' and nothing is logged."""
    now = now or _now()
    return now.weekday() < 5 and dt.time(9, 0) <= now.time() <= dt.time(15, 40)


def _iso(x):
    return x.isoformat() if isinstance(x, (dt.date, dt.datetime)) else x


def _save_latest(code: str) -> None:
    """The writer's current state for one index, for reader processes."""
    s = _st(code)
    with _LOCK:
        c = s["cash"]
        if not c:
            return
        out = dict(cash=dict(c, ts=_iso(c["ts"]), fetched=_iso(c["fetched"])),
                   weights=s["weights"], weights_day=_iso(s["weights_day"]),
                   deriv_ts=_iso(s["deriv_ts"]), err=s["err"])
    _write_json(DATA_DIR / f"latest_{code}.json", out)


def _save_deriv() -> None:
    with _LOCK:
        # per-contract option rows stay out: readers draw options from the logged summary,
        # and ~90 rows x 100 stocks rewritten every sweep would only slow every reader
        out = {k: dict({a: b for a, b in v.items() if a != "opts"}, ts=_iso(v.get("ts")),
                       day=_iso(v["day"]),
                       expiries=[_iso(e) for e in v.get("expiries", [])],
                       fut_prev={_iso(e): x for e, x in v.get("fut_prev", {}).items()})
               for k, v in _D.items()}
    _write_json(DATA_DIR / "deriv_latest.json", out)


def _write_json(path: Path, obj) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj), encoding="utf-8")
        os.replace(tmp, path)                # a reader never sees half a file
    except OSError:
        pass


_SEEN: dict = {}                             # path -> mtime last loaded by this reader


def _changed(path: Path) -> bool:
    try:
        m = path.stat().st_mtime
    except OSError:
        return False
    if _SEEN.get(path) == m:
        return False
    _SEEN[path] = m
    return True


def _sync_from_disk(code: str) -> None:
    """Reader side: pick up whatever the writer process has logged since the last look."""
    s = _st(code)
    f = DATA_DIR / f"latest_{code}.json"
    if _changed(f):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
            c = j["cash"]
            c["ts"] = dt.datetime.fromisoformat(c["ts"]) if c["ts"] else None
            c["fetched"] = dt.datetime.fromisoformat(c["fetched"])
            with _LOCK:
                s.update(cash=c, cash_at=time.time(), weights=j["weights"], err=j["err"],
                         weights_day=(dt.date.fromisoformat(j["weights_day"])
                                      if j["weights_day"] else None),
                         deriv_ts=(dt.datetime.fromisoformat(j["deriv_ts"])
                                   if j["deriv_ts"] else None))
        except (OSError, ValueError, KeyError, TypeError):
            pass
    g = DATA_DIR / "deriv_latest.json"
    if _changed(g):
        try:
            j = json.loads(g.read_text(encoding="utf-8"))
            d = {}
            for k, v in j.items():
                v["day"] = dt.date.fromisoformat(v["day"])
                v["ts"] = dt.datetime.fromisoformat(v["ts"]) if v.get("ts") else None
                v["expiries"] = [dt.date.fromisoformat(e) for e in v.get("expiries", [])]
                v["fut_prev"] = {dt.date.fromisoformat(e): x
                                 for e, x in v.get("fut_prev", {}).items()}
                d[k] = v
            with _LOCK:
                _D.update(d)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    with _LOCK:
        c = s["cash"]
    if c:
        day = (c["ts"] or c["fetched"]).date()
        for kind, key in (("tape", "tape"), ("oi", "oi_tape")):
            path = _path(kind, code, day)
            if _changed(path) or s["tape_day"] != day:
                rows = _load_jsonl(path)
                with _LOCK:
                    s[key] = rows
        with _LOCK:
            s["tape_day"] = day


STUCK_ROW_S = 120.0     # a stock row this far behind the index stamp is not current


def _fresh_prices(cash: dict, deriv: dict) -> dict:
    """The cash read with each STUCK stock price replaced by the one in that stock's own
    derivatives quote, when that quote is newer.

    4 Nov 2026, 09:15-09:35+: NSE's index-constituents feed served every stock row frozen at
    its 09:15:46 opening print (two NSE servers alternated: one stuck at 09:15, one at 09:19)
    while the index row moved on. The 50 frozen prices rebuilt -47 points against the
    index's +126; the derivatives quotes' underlying prices, stamped a minute old, rebuilt
    +108. A replaced stock's traded value is dropped (it is as frozen as its price), so
    Activity and value share are not computed from a 09:15 number. Only while the market is
    open: after the close every row is final anyway. Returns a NEW dict; marks it with
    "stuck" = (stocks replaced, the stuck rows' newest update time)."""
    ts = cash.get("ts")
    if cash.get("status") != "Open" or ts is None or not deriv:
        return cash
    out, swapped, stuck_at = dict(cash, stocks=dict(cash["stocks"])), [], None
    for sym, q in cash["stocks"].items():
        u = _ts(q.get("upd"))
        if u is not None and (ts - u).total_seconds() <= STUCK_ROW_S:
            continue
        d = deriv.get(sym) or {}
        S, dts = (d.get("by_exp") or {}).get("S"), d.get("ts")
        if not S or dts is None or (u is not None and dts <= u) or not q.get("prev"):
            continue
        stuck_at = max(stuck_at, u) if (stuck_at and u) else (u or stuck_at)
        out["stocks"][sym] = dict(q, ltp=S, pct=(S / q["prev"] - 1) * 100, value=None,
                                  upd=f"{dts:%Y-%m-%d %H:%M:%S}", src="deriv")
        swapped.append(sym)
    if swapped:
        # plain types only: this dict is written to latest_{code}.json for reader processes
        out["stuck"] = [swapped, f"{stuck_at:%H:%M:%S}" if stuck_at else None]
    return out


def poll_cash(code: str, client=None) -> bool:
    """One price poll (+ weights when due). Updates state and the tape. Never raises."""
    client = client or _client()
    s = _st(code)
    try:
        cash = fetch_cash(code, client)
        # A PARTIAL LIST IS NOT THE INDEX. At 09:15:55 on 19 Oct 2026 NSE answered Midcap
        # Select with 18 of its 25 stocks (marked "Closed"); taken as-is, the page would have
        # built buckets and breadth from a third of the index missing. Rejected, the last
        # good read stays on screen and the next poll tries again.
        known = s["weights"] or _load_weights_file(code)[0]
        if known and len(cash["stocks"]) < 0.9 * len(known):
            raise NseError(f"{NSE_NAME[code]}: partial list, {len(cash['stocks'])} of "
                           f"{len(known)} stocks")
    except Exception as exc:                                 # noqa: BLE001
        with _LOCK:
            s["err"], s["cash_at"] = str(exc), time.time()
        if not isinstance(exc, NseError) or "partial list" in str(exc) or "unusable" in str(exc):
            _log_error("cash", str(exc)[:160])   # transport errors are logged by the client
        return False
    w, wday = s["weights"], s["weights_day"]
    if w is None or time.time() - s["weights_at"] > WEIGHTS_EVERY:
        try:
            w, wday = fetch_weights(code, client), _now().date()
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            (DATA_DIR / f"weights_{code}.json").write_text(
                json.dumps({"day": wday.isoformat(), "w": w}), encoding="utf-8")
            # a dated copy too: the plain file is overwritten every day, and the weights
            # are the one input a later study cannot rebuild from anything else
            (DATA_DIR / f"weights_{code}_{wday:%Y%m%d}.json").write_text(
                json.dumps({"day": wday.isoformat(), "w": w}), encoding="utf-8")
        except Exception:                                    # noqa: BLE001
            if w is None:
                w, wday = _load_weights_file(code)
    with _LOCK:
        day = (cash["ts"] or cash["fetched"]).date()
        if s["tape_day"] != day:                 # new session, or first poll of this process
            s["tape"] = _load_jsonl(_path("tape", code, day))
            s["oi_tape"] = _load_jsonl(_path("oi", code, day))
            s["tape_day"] = day
            if s["deriv_ts"] and s["deriv_ts"].date() != day:
                s["deriv_ts"] = None
        # stuck NSE stock rows -> each stock's own derivatives-quote price, BEFORE logging,
        # so the tape and every chart built on it carry current prices too
        cash = _fresh_prices(cash, _fresh_deriv(list(cash["stocks"]), day))
        s.update(cash=cash, cash_at=time.time(), err=None, weights=w, weights_day=wday)
        if w is not None:
            s["weights_at"] = time.time()
        ns = (cash["ts"] or cash["fetched"]).isoformat()
        p_now = {k: round(v["pct"], 4) for k, v in cash["stocks"].items()}
        last = s["tape"][-1] if s["tape"] else None
        # A row is logged when NSE's stamp moves OR any price moves. Logging on the stamp
        # alone kept one row every ~106s on 15 Oct 2026 although prices changed about every
        # 60s: the stamp on this feed advances less often than the numbers under it. A row
        # whose stamp did not move is timed by when it was READ (and says so in `ns`).
        changed = last is None or last.get("ns", last["t"]) != ns or last["p"] != p_now
        t = ns
        if last is not None and last.get("ns", last["t"]) == ns:
            t = cash["fetched"].isoformat(timespec="seconds")
        if last is not None and t <= last["t"]:
            changed = False                       # never log time going backwards
        if (cash["status"] == "Open" and in_session(cash["ts"] or cash["fetched"])
                and changed):
            e = dict(t=t, ns=ns, last=cash["index"]["last"], prev=cash["index"]["prev"],
                     hi=cash["index"].get("high"), lo=cash["index"].get("low"),
                     p=p_now,
                     # cumulative traded value so far today, Rs crore — so a window's
                     # value is a difference of two rows (rows before 15 Oct lack it)
                     v={k: round(v["value"] / 1e7, 3) for k, v in cash["stocks"].items()
                        if v.get("value")})
            s["tape"].append(e)
            _append(_path("tape", code, day), e)
    _save_meta(code)
    _save_latest(code)
    return True


def _save_meta(code: str) -> None:
    """Once per index per session: everything needed to rebuild the day's page from the log
    alone — previous closes, weights, the bucket membership used, the F&O ban list and
    corporate actions, and which archive date the OI base came from. Rewritten only when
    one of those changes, so it costs nothing on an ordinary poll."""
    s = _st(code)
    with _LOCK:
        c, w = s["cash"], s["weights"]
    if not c:
        return
    day = (c["ts"] or c["fetched"]).date()
    syms = tuple(sorted(c["stocks"]))
    try:
        rb = resolve_buckets(code, w or {}, list(c["stocks"]), day, _dcm())
        base = _archive_oi_base(syms, day)
        meta = dict(day=day.isoformat(), index=NSE_NAME[code],
                    index_prev=c["index"]["prev"],
                    prev={k: v["prev"] for k, v in c["stocks"].items()},
                    weights=w, weights_day=_iso(s["weights_day"]),
                    buckets=rb["buckets"], bucket_source=rb["source"],
                    ban=sorted(fetch_ban_list(day) or []),
                    corp=_archive_corp(syms, day),
                    oi_base_date=_iso(next((b["date"] for b in base.values()), None)))
    except Exception:                                        # noqa: BLE001
        return
    f = DATA_DIR / f"meta_{code}_{day:%Y%m%d}.json"
    try:
        # A FINISHED DAY'S META IS FROZEN. Before the next open NSE still stamps its feed with
        # the previous session, so the first poll of the morning rewrote that day's file with
        # the NEW day's weights and an empty ban list (the ban file is date-checked, so it
        # missed): 15, 16, 22 and 27 Oct 2026 were overwritten that way.
        if f.exists() and _now().date() > day:
            return
        if f.exists() and json.loads(f.read_text(encoding="utf-8")) == meta:
            return
    except (OSError, ValueError):
        pass
    _write_json(f, meta)


def _fresh_deriv(symbols, day: dt.date, max_age: float | None = None) -> dict:
    """The cached OI reads for `symbols` taken on session `day` (never yesterday's)."""
    out = {}
    for sym in symbols:
        d = _D.get(sym)
        if not d or d["day"] != day:
            continue
        if max_age is not None and time.time() - d["at"] > max_age:
            continue
        out[sym] = d
    return out


def poll_deriv(code: str, client=None) -> int:
    """One OI sweep of the index's constituents. Returns how many stocks were fetched."""
    client = client or _client()
    s = _st(code)
    with _LOCK:
        if s["deriv_busy"] or not s["cash"]:
            return 0
        s["deriv_busy"] = True
        day = (s["cash"]["ts"] or s["cash"]["fetched"]).date()
        is_open = s["cash"]["status"] == "Open"
        syms = list(s["cash"]["stocks"])
        # skip what another index's sweep fetched moments ago
        have = _fresh_deriv(syms, day, DERIV_EVERY * 0.8 if is_open else None)
        todo = [x for x in syms if x not in have]
    n, new = 0, {}
    # yesterday's close for the option read: its TIME from the calendar (not the archive,
    # which can be a session late), the archive only for lot sizes; None skips the read
    try:
        _ob = _archive_oi_base(tuple(sorted(syms)), day)
        _pc = dt.datetime.combine(_prev_session(day), dt.time(15, 30))
    except Exception:                                        # noqa: BLE001
        _ob, _pc = {}, None
    try:
        for sym in todo:
            try:
                d = fetch_deriv(sym, client)
            except NseError:
                # only the DERIVATIVES breaker stops a sweep; a price-feed outage must not
                blocked = getattr(client, "blocked", None)
                until = (blocked.get("deriv", 0.0) if isinstance(blocked, dict)
                         else getattr(client, "blocked_until", 0.0))
                if time.time() < until:
                    break                        # breaker open: stop, keep what we have
                continue                         # this stock keeps its last read
            d.update(day=day, at=time.time())
            if _pc is not None:
                try:
                    d["q"] = stock_option_flow(d, _ob.get(sym), sym, d.get("ts") or _now(), _pc)
                except Exception as exc:                     # noqa: BLE001
                    # the option read is extra: it must never cost the stock its OI row
                    _log_error("optflow", f"{sym}: {type(exc).__name__}: {exc}")
            new[sym] = d
            n += 1
    finally:
        with _LOCK:
            # COMMITTED IN ONE STEP. Written stock by stock, the page drew mid-sweep and
            # showed "7/10 · long build" for a bucket whose other three names had simply
            # not been asked yet (seen 13 Oct 2026) — a label built on a partial vote.
            _D.update(new)
            s["deriv_busy"], s["deriv_at"] = False, time.time()
            got = _fresh_deriv(syms, day, DERIV_EVERY * 1.5 if is_open else None)
            if got and (n or s["deriv_ts"] is None):
                s["deriv_ts"] = _now()
                if is_open and n and in_session(s["deriv_ts"]):
                    e = dict(t=s["deriv_ts"].isoformat(timespec="seconds"),
                             d={k: [v["fut_oi"], v["fut_chg"], v["ce_oi"], v["ce_chg"],
                                    v["pe_oi"], v["pe_chg"]] for k, v in got.items()},
                             # per-expiry detail and NSE's own stamp per stock, so the log
                             # can answer questions the sums cannot (near vs next month,
                             # futures volume, futures premium over cash)
                             x={k: v["by_exp"] for k, v in got.items() if v.get("by_exp")},
                             ns={k: _iso(v["ts"]) for k, v in got.items() if v.get("ts")},
                             # options judged per contract (option_flow), since 3 Nov 2026
                             q={k: v["q"] for k, v in got.items() if v.get("q")})
                    # the contracts themselves every OPT_DETAIL_EVERY: enough to re-derive
                    # or audit the read later, without 90 rows x 50 stocks every 2 minutes
                    if time.time() - s.get("opt_detail_at", 0.0) >= OPT_DETAIL_EVERY:
                        e["o"] = {k: [[o[0][0], o[1], o[2], o[3], o[4], o[5],
                                       o[6] if len(o) > 6 else None] for o in v["opts"]]
                                  for k, v in got.items() if v.get("opts")}
                        s["opt_detail_at"] = time.time()
                    s["oi_tape"].append(e)
                    _append(_path("oi", code, day), e)
    _save_deriv()
    _save_latest(code)
    return n


_BEAT = [0.0]


def _heartbeat() -> None:
    """Write what the poller holds to data/index_mgmt/status.json every ~20s. The page reads
    its state from memory; this file is for checking the SERVER from outside it — a board
    that silently stopped polling looks identical to a quiet market otherwise."""
    if time.time() - _BEAT[0] < 20:
        return
    _BEAT[0] = time.time()
    out = {"written": _now().isoformat(timespec="seconds"), "writer_pid": os.getpid(),
           "indices": {}}
    with _LOCK:
        for code, s in _S.items():
            c = s["cash"]
            out["indices"][code] = dict(
                nse_price_stamp=c["ts"].isoformat() if c and c["ts"] else None,
                market=c["status"] if c else None,
                index_last=c["index"]["last"] if c else None,
                last_price_poll_s_ago=round(time.time() - s["cash_at"]) if s["cash_at"] else None,
                last_oi_sweep=s["deriv_ts"].isoformat(timespec="seconds") if s["deriv_ts"] else None,
                oi_stocks=len(_fresh_deriv(c["stocks"], (c["ts"] or c["fetched"]).date())) if c else 0,
                viewed_s_ago=round(time.time() - s["viewed"]) if s["viewed"] else None,
                tape_rows=len(s["tape"]), oi_sweeps_logged=len(s["oi_tape"]), error=s["err"])
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "status.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    except OSError:
        pass


_SWEEPER: list = [None]
_HEAL = {"at": 0.0, "refresh_seen": 0.0}
HEAL_AFTER = 180.0       # prices this stale while NSE says Open -> throw the session away


def request_refresh() -> None:
    """Called by the board's refresh button. Whichever process is capturing picks it up
    within ~3 seconds: a brand-new NSE session, breakers cleared, every index re-read now.
    A file, because the writer may be the separate scheduled capture process."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "refresh.request").write_text(str(time.time()), encoding="utf-8")
    except OSError:
        pass
    for s in list(_S.values()):          # a reader re-reads the writer's files at once
        s["cash_at"] = 0.0
    _SEEN.clear()


def _reset_client(why: str) -> None:
    _CLIENT[0] = NseClient()
    with _LOCK:
        for s in _S.values():
            s["cash_at"] = 0.0
            if s["cash"] and s["cash"]["status"] == "Open":
                s["deriv_at"] = 0.0
    _log_error("reset", why)


def _maybe_heal() -> None:
    """Writer side: honour a refresh request, and replace a session that has gone bad."""
    f = DATA_DIR / "refresh.request"
    try:
        m = f.stat().st_mtime
    except OSError:
        m = 0.0
    if m > _HEAL["refresh_seen"]:
        _HEAL["refresh_seen"] = m
        if m > time.time() - 600:
            _reset_client("refresh button")
            return
    if time.time() - _HEAL["at"] < 120:
        return
    with _LOCK:
        stale = [c for c, s in _S.items()
                 if s["cash"] and s["cash"]["status"] == "Open" and market_window()
                 and s["cash"].get("fetched")
                 and (_now() - s["cash"]["fetched"]).total_seconds() > HEAL_AFTER]
    if stale:
        _HEAL["at"] = time.time()
        _reset_client(f"prices stale over {HEAL_AFTER:.0f}s: {', '.join(stale)}")


def _sweep_async(code: str) -> None:
    """Start one OI sweep in the background unless one is already running. One at a time
    across all indices: the sweeps share the stock cache, and two at once would only
    double-book the same NSE calls."""
    th = _SWEEPER[0]
    if th is not None and th.is_alive():
        return
    th = threading.Thread(target=poll_deriv, args=(code,), daemon=True,
                          name=f"eqbtst-index-mgmt-oi-{code}")
    _SWEEPER[0] = th
    th.start()


def _loop(gen: float) -> None:
    while getattr(sys, _GEN_ATTR, None) == gen:
        try:
            if not _claim():
                # another process is capturing; this one only reads what it writes
                with _LOCK:
                    codes = list(_S)
                for code in codes:
                    _sync_from_disk(code)
                time.sleep(5.0)
                continue
            _heartbeat()
            apply_cadence()
            _maybe_heal()
            live_hours = market_window()
            for code in NSE_NAME if live_hours else list(_S):
                s = _st(code)
                viewing = time.time() - s["viewed"] < VIEW_TTL
                is_open = bool(s["cash"]) and s["cash"]["status"] == "Open"
                # In market hours every index is captured whether or not anyone is looking,
                # so the session log starts at the open. Outside them only a page someone
                # has open is refreshed, slowly.
                # inside the window but NSE says shut (a holiday, or 15:30-15:40): look every
                # 5 minutes, not every 30s — a full-speed holiday is ~7,000 pointless calls
                waiting = (s["cash"] is None or is_open
                           or _now().time() < dt.time(9, 20))
                every = (CASH_EVERY if waiting else 300.0) if live_hours else IDLE_EVERY
                if time.time() - s["cash_at"] >= every and (live_hours or viewing):
                    poll_cash(code)
                if not s["cash"] or s["deriv_busy"]:
                    continue
                # OI sweeps run on their OWN thread. In this loop they blocked price reads
                # for 15-25s each: on 15 Oct 2026 Nifty logged a price row every 64s but the
                # other three indices only every 104-128s, against NSE's ~60s refresh. The
                # NSE client still serialises every call, so this adds no request rate.
                if is_open:
                    # Nifty and whatever is on screen at the measured OI pace; the other
                    # indices at half of it (their stocks are mostly Nifty's anyway)
                    gap = DERIV_EVERY if (viewing or code == "NIFTY") else DERIV_EVERY * 2
                    if time.time() - s["deriv_at"] >= gap:
                        _sweep_async(code)
                elif viewing and (s["deriv_ts"] is None
                                  or time.time() - s["deriv_at"] >= IDLE_EVERY * 4):
                    _sweep_async(code)
        except Exception:                                    # noqa: BLE001
            pass                                 # a poller that dies silently is worse
        time.sleep(2.0 + random.random())


def _init_heal() -> None:
    try:
        _HEAL["refresh_seen"] = (DATA_DIR / "refresh.request").stat().st_mtime
    except OSError:
        pass


def ensure_poller() -> None:
    """Start the single background poller. A Streamlit hot-reload re-imports this module;
    the generation stamp on `sys` makes the previous thread exit instead of doubling up."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return                           # never poll NSE or take the live lock from a test
    with _LOCK:
        t = getattr(sys, "_eqbtst_index_mgmt_thread", None)
        if t is not None and t.is_alive() and getattr(t, "_owner", None) is _S:
            return
        gen = time.time()
        _init_heal()
        setattr(sys, _GEN_ATTR, gen)
        t = threading.Thread(target=_loop, args=(gen,), daemon=True,
                             name="eqbtst-index-mgmt")
        t._owner = _S
        sys._eqbtst_index_mgmt_thread = t
        t.start()


def touch(code: str) -> None:
    _st(code)["viewed"] = time.time()


def snapshot(code: str) -> dict | None:
    """The current decomposition from whatever the poller holds. No network."""
    s = _st(code)
    if not _MINE["owner"]:
        _sync_from_disk(code)
    with _LOCK:
        cash, w = s["cash"], s["weights"]
        deriv = (_fresh_deriv(cash["stocks"], (cash["ts"] or cash["fetched"]).date())
                 if cash else {})
        meta = dict(err=s["err"], deriv_ts=s["deriv_ts"], deriv_busy=s["deriv_busy"],
                    weights_day=s["weights_day"], tape=list(s["tape"]),
                    oi_tape=list(s["oi_tape"]))
    if not cash:
        return dict(empty=True, **meta)
    syms = tuple(sorted(cash["stocks"]))
    day = (cash["ts"] or cash["fetched"]).date()
    # again at draw time: an OI sweep may have landed since the last price poll
    cash = _fresh_prices(cash, deriv)
    base = _archive_oi_base(syms, day)
    corp = dict(_archive_corp(syms, day))
    ban = fetch_ban_list(day)
    for sym in (ban or ()):
        if sym in cash["stocks"]:
            corp.setdefault(sym, "F&O BAN")
    snap = decompose(cash, w, deriv, code, dcm=_dcm(), norms=_archive_norms(syms),
                     corp=corp, oi_base=base)
    snap["withheld"] = set(corp)
    snap["stuck"] = cash.get("stuck")
    if cash.get("stuck"):
        names, at = cash["stuck"]
        snap["notes"].insert(0, (
            f"NSE's live stock prices are stuck{f' at {at}' if at else ''} for {len(names)} of "
            f"{len(cash['stocks'])} stocks while its index figure keeps moving — an NSE feed "
            "fault, not a trading halt. For those stocks this page uses the price in each "
            "stock's own derivatives quote instead (refreshed every couple of minutes), so "
            "index points, returns and breadth stay current; their traded value is left out, "
            "so Activity and value share cover only the stocks NSE is updating."))
    stamps = [d["ts"] for d in deriv.values() if d.get("ts")]
    lags = snap["stocks"]["lag_s"].dropna() if "lag_s" in snap["stocks"] else []
    snap.update(meta, empty=False, weights=w or {},
                price_read_at=cash.get("fetched"),
                stock_lag_s=float(np.median(lags)) if len(lags) else None,
                oi_nse_from=min(stamps) if stamps else None,
                oi_nse_to=max(stamps) if stamps else None,
                oi_base_date=next((b["date"] for b in base.values()), None))
    return snap


# ── past sessions: rebuilt from the log, through the same decompose() as the live page ─
COVER_GAP_MIN = 6       # readings further apart than this are a hole in the log (OI is read
                        # every 2 minutes, prices every 30 s; NSE publishes about once a minute)
COVER_COMPLETE = 95.0   # % of the session both logs must cover to call a day complete
COVER_THIN = 10.0
_REPLAY_CACHE: dict = {}
REPLAY_CACHE_N = 6      # sessions kept in memory (a day's two logs are a few MB)
_COV_CACHE: dict = {}


def _coverage_of(rows: list, day: dt.date, now: dt.datetime | None = None) -> dict:
    """How much of 09:15-15:30 (so far, if `day` is today) one log covers."""
    t0 = dt.datetime.combine(day, dt.time(9, 15))
    t1 = dt.datetime.combine(day, dt.time(15, 30))
    end = min(now, t1) if now is not None and now.date() == day else t1
    span = (end - t0).total_seconds() / 60
    out = dict(n=0, first=None, last=None, pct=0.0, gaps=[], span=max(span, 0.0))
    if span <= 0:
        return out
    ts = []
    for r in rows:
        x = _ts(str(r.get("t", "")).replace("T", " "))
        if x is not None and t0 <= x <= t1:
            ts.append(x)
    ts.sort()
    if not ts:
        out["gaps"] = [(t0, end)]
        return out
    covered = 0.0
    for a, b in zip([t0] + ts, ts + [max(end, ts[-1])]):
        g = (b - a).total_seconds() / 60
        if g <= COVER_GAP_MIN:
            covered += g
        else:
            out["gaps"].append((a, b))
    out.update(n=len(ts), first=ts[0], last=ts[-1], pct=min(covered / span * 100, 100.0))
    return out


def session_coverage(tape: list, oi_tape: list, day: dt.date,
                     now: dt.datetime | None = None) -> dict:
    """{prices, oi, grade}: grade is complete / partial / thin / none. A day is COMPLETE only
    when BOTH logs cover it: a full price log with no open interest is not a full day."""
    p, o = _coverage_of(tape, day, now), _coverage_of(oi_tape, day, now)
    if not p["n"] and not o["n"]:
        grade = "none"
    elif min(p["pct"], o["pct"]) >= COVER_COMPLETE:
        grade = "complete"
    elif max(p["pct"], o["pct"]) < COVER_THIN:
        grade = "thin"
    else:
        grade = "partial"
    return dict(prices=p, oi=o, grade=grade, day=day)


_COV_WORD = {"complete": "🟢 complete", "partial": "🟡 partial", "thin": "🟠 very little",
             "none": "⚫ no log"}


def _span_txt(c: dict) -> str:
    return f"{c['first']:%H:%M}–{c['last']:%H:%M} ({c['pct']:.0f}%)" if c["n"] else "none"


def _holes_txt(cov: dict) -> str:
    """The stretches of the session with no prices or no open interest, merged."""
    spans = sorted(cov["prices"]["gaps"] + cov["oi"]["gaps"])
    merged: list = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return ", ".join(f"{a:%H:%M}–{b:%H:%M}" for a, b in merged)


def coverage_text(cov: dict) -> str:
    """One line: what this session's log holds and what it does not."""
    if cov["grade"] == "none":
        return "⚫ **No minute-by-minute log for this session.**"
    holes = _holes_txt(cov)
    return (f"{_COV_WORD[cov['grade']]} — prices {_span_txt(cov['prices'])}, open interest "
            f"{_span_txt(cov['oi'])}" + (f"; **not logged: {holes}**" if holes else ""))


def _log_size(kind: str, code: str, day: dt.date) -> int:
    p = _path(kind, code, day)
    for f in (p, p.with_suffix(".jsonl.gz")):
        try:
            return f.stat().st_size
        except OSError:
            continue
    return 0


def _day_coverage(code: str, day: dt.date) -> dict:
    """Coverage of a FINISHED day. Re-read only if a log file changes size."""
    key = (code, day, _log_size("tape", code, day), _log_size("oi", code, day))
    if key not in _COV_CACHE:
        _COV_CACHE[key] = session_coverage(_load_jsonl(_path("tape", code, day)),
                                           _load_jsonl(_path("oi", code, day)), day)
    return _COV_CACHE[key]


def _logged_days(code: str) -> set:
    import re
    out = set()
    for kind in ("tape", "oi"):
        for f in DATA_DIR.glob(f"{kind}_{code}_*.jsonl*"):
            m = re.search(r"_(\d{8})\.jsonl", f.name)
            if m:
                out.add(dt.datetime.strptime(m.group(1), "%Y%m%d").date())
    return out


def list_sessions(code: str, today: dt.date | None = None) -> list:
    """Finished sessions newest first: every logged day, plus every day the archive says
    traded since logging began — a session with no log must be SEEN as missing, not skipped."""
    today = today or _now().date()
    logged = {d for d in _logged_days(code) if d < today}
    if not logged:
        return []
    key = ("sessions", today, min(logged))
    if _ctx_fresh(key):
        traded = _CTX[key]
    else:
        traded = _CTX.get(key, set())
        try:
            from . import data
            with data._connect() as c:
                rows = c.execute("select distinct trade_date from daily_data "
                                 "where trade_date >= ? and trade_date < ?",
                                 [min(logged), today]).fetchall()
            traded = {pd.Timestamp(r[0]).date() for r in rows}
            _CTX[key] = traded
        except Exception:                                    # noqa: BLE001
            pass
    return sorted(logged | traded, reverse=True)


def _weights_for(code: str, day: dt.date, meta: dict | None) -> tuple:
    """(weights, the day they are from) as known ON `day`: the dated copy, else the newest
    within a week BEFORE it. Never a later list. The meta file's own weights come last: until
    29 Oct 2026 a pre-open poll could overwrite a finished day's meta with the NEXT day's."""
    for back in range(0, 8):
        d = day - dt.timedelta(days=back)
        f = DATA_DIR / f"weights_{code}_{d:%Y%m%d}.json"
        try:
            w = json.loads(f.read_text(encoding="utf-8")).get("w")
        except (OSError, ValueError):
            continue
        if w:
            return w, d
    wd = dt.date.fromisoformat(meta["weights_day"]) if meta and meta.get("weights_day") else None
    if meta and meta.get("weights") and wd and wd <= day:
        return meta["weights"], wd
    return None, None


def _deriv_from_log(oi_tape: list) -> dict:
    """{stock: its LAST logged reading}, in the shape fetch_deriv() returns. A stock NSE
    skipped in the final sweep keeps its previous one, exactly as on the live page."""
    out: dict = {}
    for e in reversed(oi_tape):
        ts = _ts(str(e.get("t", "")).replace("T", " "))
        for s, d in (e.get("d") or {}).items():
            if s in out or len(d) < 6:
                continue
            x = (e.get("x") or {}).get(s) or {}
            fx = x.get("F") or {}
            exps, fut_prev = set(), {}
            for leg in ("F", "C", "P"):
                exps |= {dt.date.fromisoformat(k) for k in (x.get(leg) or {})}
            for k, v in fx.items():
                fut_prev[dt.date.fromisoformat(k)] = v[0] - v[1]
            ns = (e.get("ns") or {}).get(s)
            out[s] = dict(fut_oi=d[0], fut_chg=d[1], ce_oi=d[2], ce_chg=d[3], pe_oi=d[4],
                          pe_chg=d[5], expiries=sorted(exps), fut_prev=fut_prev, by_exp=x,
                          ts=_ts(str(ns).replace("T", " ")) if ns else ts)
    return out


def session_snapshot(code: str, day: dt.date) -> dict | None:
    """A finished session as the page saw it at its LAST logged reading — not the close,
    unless the log reached it. No network, nothing written, nothing later than `day` used:
    the normal-value yardstick stops before `day`, weights never come from after it, and no
    lot size is remembered. None when the day has no price log."""
    hit = _REPLAY_CACHE.get((code, day))
    sizes = (_log_size("tape", code, day), _log_size("oi", code, day))
    if hit and hit[1] == sizes and time.time() - hit[0] < ARCHIVE_TTL_S:
        return hit[2]
    tape = [r for r in _load_jsonl(_path("tape", code, day))
            if r.get("p") and r.get("last") and r.get("prev")]
    if not tape:
        return None
    oi_tape = _load_jsonl(_path("oi", code, day))
    try:
        meta = json.loads((DATA_DIR / f"meta_{code}_{day:%Y%m%d}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = None
    row = tape[-1]
    as_of = dt.datetime.fromisoformat(row["t"])
    prevs = (meta or {}).get("prev") or {}
    stocks = {}
    for s, pct in row["p"].items():
        pv = prevs.get(s)
        val = (row.get("v") or {}).get(s)
        stocks[s] = dict(ltp=pv * (1 + pct / 100) if pv else None, prev=pv, open=None, pct=pct,
                         value=val * 1e7 if val else None, upd="")
    # status 'Open' so the pace and options-cycle fractions are those of the last reading's
    # time of day (a log that stopped at 11:26 must not be judged against a full day's norm)
    cash = dict(code=code, ts=as_of, status="Open", fetched=as_of, stocks=stocks,
                index=dict(last=row["last"], prev=row["prev"], open=None,
                           high=row.get("hi"), low=row.get("lo")))
    weights, wday = _weights_for(code, day, meta)
    syms = tuple(sorted(stocks))
    base = _archive_oi_base(syms, day)
    corp = dict((meta or {}).get("corp") or _archive_corp(syms, day))
    for s in (meta or {}).get("ban") or ():
        if s in stocks:
            corp.setdefault(s, "F&O BAN")
    deriv = _deriv_from_log(oi_tape)
    with _replay():
        snap = decompose(cash, weights, deriv, code, dcm=_dcm(),
                         norms=_archive_norms(syms, before=day), corp=corp, oi_base=base,
                         now=as_of)
    stamps = [d["ts"] for d in deriv.values() if d.get("ts")]
    reads = [_ts(str(e.get("t", "")).replace("T", " ")) for e in oi_tape if e.get("t")]
    meta_day = (meta or {}).get("weights_day")
    snap.update(empty=False, err=None, deriv_busy=False, weights_day=wday, tape=tape,
                oi_tape=oi_tape, weights=weights or {}, price_read_at=as_of, stock_lag_s=None,
                deriv_ts=reads[-1] if reads and deriv else None,
                oi_nse_from=min(stamps) if stamps else None,
                oi_nse_to=max(stamps) if stamps else None,
                oi_base_date=next((b["date"] for b in base.values()), None),
                withheld=set(corp), live=False, replay=True, as_of=as_of,
                coverage=session_coverage(tape, oi_tape, day),
                # rewritten after its session (see _save_meta): that day's ban list is lost
                ban_unknown=meta is None or bool(meta_day and meta_day > day.isoformat()))
    for k in [k for k in _REPLAY_CACHE if k != (code, day)][:-(REPLAY_CACHE_N - 1)]:
        _REPLAY_CACHE.pop(k, None)
    _REPLAY_CACHE[(code, day)] = (time.time(), sizes, snap)
    return snap


def official_close_read(code: str, day: dt.date):
    """DCM's end-of-day read of `day` itself, or None. What a partial log never reached."""
    key = ("official", day, code)
    if _ctx_fresh(key):
        return _CTX[key]
    m = _dcm()
    if m is None:
        return None
    try:
        d = m.get_index_largecap(day, code)
        if d.data_ok:
            _CTX[key] = d
            return d
    except Exception:                                        # noqa: BLE001
        return _CTX.get(key)
    return None


# ── page ──────────────────────────────────────────────────────────────────────────────
_FUT = {"long build": "🟢 long build", "short covering": "🟢 short covering",
        "long build + covering": "🟢 long build + covering", "mixed": "⚪ mixed",
        "short build": "🔴 short build", "long unwind": "🔴 long unwind",
        "short build + unwind": "🔴 short build + unwind", None: "—"}
_OPT = {"put writing": "🟢 put writing", "put side heavier": "🟢 put side heavier",
        "balanced": "⚪ balanced", "call side heavier": "🔴 call side heavier",
        "call writing": "🔴 call writing", None: "—"}
_QUAD = {"long build": "🟢 long build", "short covering": "🟢 short covering",
         "short build": "🔴 short build", "long unwind": "🔴 long unwind",
         "flat": "⚪ flat", None: "—"}

HELP = """
**What this page is.** The index is cap-weighted, so its members do not move it equally.
This splits it into weight buckets — Top, Next, Rest — and shows, live, which bucket is
actually moving it, in **real index points**.

**Where the numbers come from.** NSE's own website feed — not Fyers, so it costs the broker
quota nothing. It runs 1–2 minutes behind the tape; the exact age is printed above.

**How to read it.**
- **Index pts** — points this bucket added to or took from the index today. The three add up
  to the index's move; the caption under the table shows by how much they miss (normally
  under one point).
- **Return %** — plain average of the bucket's stocks. Compare with *Index pts*: a bucket can
  be up on average and still pull the index down if its heaviest name fell.
- **Carry spread** — index minus the equal-weight basket. Positive = heavyweights carrying.
- **Futures / Options** — the same words as the Daily Cash Market tab, built by the same code,
  but on *intraday* open interest. They are **provisional until the close**: day traders'
  positions are inside these numbers and leave before 15:20.

**What it is not.** A forecast. The end-of-day version of every column here was tested over
1,170+ sessions; nothing predicts the index once the day's own close-strength is known.
Use it to see *what is happening*, not *what happens next*.
"""


# ── column tooltips ───────────────────────────────────────────────────────────────────
# PLAIN ENGLISH, ONE PATTERN: what it is -> an example with numbers -> how to read it.
# Written for someone who has never heard "open interest" or "basis points". Every column
# of every table on this page has one (a test pins that), because a number with no
# explanation gets read as whatever the reader hopes it means.
# {column: (number format or None for text, tooltip)}
_T_FUT_WORDS = (
    "- 🟢 **Long build** — price UP and open contracts UP. New buyers are coming in.\n"
    "- 🟢 **Short covering** — price UP and open contracts DOWN. Sellers are closing "
    "their bets.\n"
    "- 🔴 **Short build** — price DOWN and open contracts UP. New sellers are coming in.\n"
    "- 🔴 **Long unwind** — price DOWN and open contracts DOWN. Buyers are leaving.\n")

TIP_MAIN = {
    "Bucket": (None,
        "A group of index stocks, sorted by how heavy they are in the index.\n\n"
        "- **Top** = the heaviest stocks.\n- **Next** = the stocks just after them.\n"
        "- **Rest** = all the others.\n\n"
        "The number is how many stocks are in the group. 'Top 10' = the 10 heaviest."),
    "Weight %": ("%.1f",
        "How much of the index this group makes up.\n\n"
        "**Example:** 52.8 means these stocks are 52.8% of the index. If all of them rise "
        "1%, the index rises about 0.53%.\n\nThe three rows add up to 100."),
    "Index pts": ("%+.1f",
        "How many index points this group has added or taken away today.\n\n"
        "**Example:** +158.4 means the index is 158 points higher today because of these "
        "stocks. A minus number means they pulled the index down.\n\n"
        "Add the three rows and you get today's total index move."),
    "Share of move %": ("%.0f",
        "Out of today's total index move, how much came from this group.\n\n"
        "**Example:** 44 means 44% of today's move came from this group.\n\n"
        "It can go above 100 or below 0. That happens when groups pull in opposite "
        "directions — one pushing the index up while another pulls it down.\n\n"
        "Blank when the index has hardly moved (less than 0.10%), because the number "
        "would mean nothing."),
    "Return %": ("%+.2f",
        "The simple average move of the stocks in this group today. Every stock counts "
        "the same, big or small.\n\n"
        "**Example:** +1.36 means the stocks in this group are up 1.36% on average.\n\n"
        "This is NOT how much the group moved the index — for that, look at **Index "
        "pts**. A group can be up on average and still pull the index down, if its "
        "heaviest stock fell."),
    "Adv": (None,
        "How many stocks in this group are up today.\n\n"
        "**Example:** 9/10 means 9 of the 10 stocks are up, and 1 is down or unchanged.\n\n"
        "A high count means the whole group is rising together. A low count with a plus "
        "in **Index pts** means one or two heavy stocks are doing all the work."),
    "Activity %": ("%+.0f",
        "Is trading busier or quieter than normal for this time of day?\n\n"
        "It compares the money traded so far today with what these stocks normally trade "
        "by this time (normal = their last 21 trading days).\n\n"
        "**Example:** +109 means about double the usual trading. −30 means 30% quieter "
        "than usual. 0 means normal.\n\nBlank before 09:30 — too early to compare."),
    "Value share %": ("%.0f",
        "Out of all the money traded in this index's stocks today, how much went through "
        "this group.\n\n"
        "**Compare it with Usual share %** (the column to its right), NOT with Weight %. "
        "Big companies always trade a smaller slice of their size than small ones, so "
        "Top 10 normally gets less of the money than its weight.\n\n"
        "**Example:** Value share 50 against a usual 43 = more money than normal is going "
        "through this group today. 36 against a usual 43 = money is going elsewhere.\n\n"
        "It says WHERE the money is going, not whether it is buying or selling — every "
        "trade has a buyer and a seller. Price tells the direction."),
    "Usual share %": ("%.0f",
        "The share of the index's traded money this group USUALLY gets, from its stocks' "
        "last 21 trading days.\n\n"
        "Put it next to Value share %: today above usual = money crowding into this group; "
        "below usual = money going to the other groups."),
    "Futures OI %": ("%+.2f",
        "**Open interest (OI)** = the number of futures contracts that are still open "
        "(not yet closed).\n\n"
        "This column shows how much that number has changed today, compared with "
        "yesterday's close.\n\n"
        "**Example:** +1.79 means 1.79% more open contracts than yesterday — new "
        "positions were added. A minus number means positions were closed.\n\n"
        "All expiry months are added together, so moving a position from this month to "
        "next month does not look like an exit.\n\n"
        "During the day this also counts same-day positions, and many of those are "
        "closed before 15:20.\n\n"
        "**Provisional.** NSE's live futures number runs HIGH and keeps climbing all day: "
        "on 19 Oct 2026 it read about +2% at 15:29 while the official close was flat, and "
        "in expiry week it read +8% to +13%. The note under the table shows how often it "
        "has matched the official close lately."),
    "Futures now": (None,
        "What futures traders are doing in this group today, in one phrase. It joins two "
        "things: did the price go up or down, and did open contracts go up or down.\n\n"
        + _T_FUT_WORDS +
        "- ⚪ **Mixed** — the stocks in the group do not agree.\n\n"
        "Each stock gives one vote. The group gets a colour only when one side clearly "
        "has more votes (more than 1.5 times the other side).\n\n"
        "Early in the day you will often see 'mixed' or a dash. That is normal: a stock "
        "votes only after its open contracts have changed by more than 0.5%.\n\n"
        "This compares NOW with YESTERDAY'S CLOSE. For what is happening in the last few "
        "minutes, use the **Fresh positioning** table lower down.\n\n"
        "Keep in mind: this is the market's usual way of reading it. Every contract has "
        "both a buyer and a seller, so it is a reading, not a proven fact."),
    "Stock votes": (None,
        "The count behind the **Futures now** phrase, so you can check it yourself.\n\n"
        "**Example:** '▲ 6 long · 0 cover | ▼ 1 short · 0 unwind | 3 flat' means 6 stocks "
        "show long build, 0 show short covering, 1 shows short build, 0 show long unwind, "
        "and 3 stocks did not change enough to vote.\n\n"
        "▲ = votes for the up side. ▼ = votes for the down side."),
    "Call OI %": ("%+.2f",
        "How much the number of open CALL option contracts has changed today, compared "
        "with yesterday's close.\n\n"
        "**Example:** −2.13 means 2.13% fewer open calls than yesterday.\n\n"
        "All strike prices and all expiry months are added together.\n\n"
        "Usual reading: calls going up = traders selling calls, because they expect the "
        "price not to rise much more. But this number cannot show whether the calls were "
        "bought or sold."),
    "Put OI %": ("%+.2f",
        "How much the number of open PUT option contracts has changed today, compared "
        "with yesterday's close.\n\n"
        "**Example:** +2.11 means 2.11% more open puts than yesterday.\n\n"
        "All strike prices and all expiry months are added together.\n\n"
        "Usual reading: puts going up = traders selling puts, because they expect the "
        "price not to fall much. But this number cannot show whether the puts were bought "
        "or sold."),
    "Options now": (None,
        "Which side of the options market is growing faster today — calls or puts.\n\n"
        "- 🟢 **Put writing** — puts are up and calls are down.\n"
        "- 🟢 **Put side heavier** — puts are growing faster than calls.\n"
        "- ⚪ **Balanced** — both about the same.\n"
        "- 🔴 **Call side heavier** — calls are growing faster than puts.\n"
        "- 🔴 **Call writing** — calls are up and puts are down.\n\n"
        "Green is the usual bullish reading: traders who SELL puts expect the price to "
        "hold. That is true only if the new puts were sold and not bought — this data "
        "cannot tell which.\n\n"
        "Before choosing the label, the page takes out the normal monthly pattern (calls "
        "usually grow faster early in the month, puts near expiry), so the label shows "
        "what is unusual today. During the day that adjustment is an estimate.\n\n"
        "Switched off on the monthly expiry day, because the numbers are distorted then."),
    "F&O names": (None,
        "How many stocks in this group have futures and options data in the latest "
        "reading.\n\n"
        "**Example:** 10/10 means all ten — the futures and options columns are "
        "complete.\n\n"
        "0/10 with empty cells beside it means the first reading has not finished yet "
        "(about 15–25 seconds). If it stays below full, those columns are based on fewer "
        "stocks."),
}

TIP_SOURCE = {
    "Numbers": (None, "Which numbers on this page this row is talking about."),
    "Source": (None,
        "Where those numbers come from.\n\n- **NSE live feed** = the NSE website, read "
        "today.\n- **Archive** = our own saved end-of-day data."),
    "Stamp": (None,
        "The date and time printed on that data.\n\nIf it shows today's date and a recent "
        "time, those numbers are live. A date only (no time) means end-of-day data."),
    "Describes": (None, "Whether those numbers are about today or about yesterday."),
}

TIP_EOD = {
    "Bucket": (None, "The same groups as the main table: Top, Next, Rest."),
    "Return %": ("%+.2f",
        "The average move of the group's stocks on that earlier day — not today."),
    "Futures at that close": (None,
        "The **Futures now** phrase as it stood when the market closed on that day.\n\n"
        "Compare it with today's. **Example:** 'short build' here and 'long build' in the "
        "main table means traders have changed direction since yesterday."),
    "Options at that close": (None,
        "The **Options now** phrase as it stood when the market closed on that day. "
        "Compare it with today's to see if the options side has changed."),
    "Delivery % (last 4 closes)": (None,
        "**Delivery %** = out of all shares traded, how many were really taken home by "
        "buyers, instead of being bought and sold on the same day. Higher means more real "
        "buying.\n\n"
        "The four numbers are the last four finished trading days, newest first.\n\n"
        "**Example:** '60.3, 58.3, 55.8, 58.4' means 60.3% on the latest day and 58.3% the "
        "day before — rising.\n\n"
        "NSE publishes this only after the market closes, so today's number is never "
        "here."),
}

def _tip_optq(side: str) -> dict:
    call = side == "CE"
    w = "calls" if call else "puts"
    return {
        "Bucket": (None, "The same groups as the main table: Top, Next, Rest."),
        "Long build ₹ Cr": ("%.0f",
            f"New {w} OPENED while their volatility rose — buyers paying up.\n\n"
            + ("Bullish: someone is paying to bet on a rise." if call else
               "Bearish: someone is paying for protection or a bet on a fall.")),
        "Writing ₹ Cr": ("%.0f",
            f"New {w} OPENED while their volatility fell — sellers (writers) pressing.\n\n"
            + ("Bearish: writers expect the stock NOT to rise past those strikes." if call
               else "Bullish: writers expect the stock NOT to fall below those strikes.")),
        "Short covering ₹ Cr": ("%.0f",
            f"{w.capitalize()} CLOSED while their volatility rose — writers buying back.\n\n"
            + ("Bullish: call writers giving up on a ceiling." if call else
               "Bearish: put writers giving up on a floor.")),
        "Long unwind ₹ Cr": ("%.0f",
            f"{w.capitalize()} CLOSED while their volatility fell — buyers selling out.\n\n"
            + ("Bearish: call buyers giving up on a rise." if call else
               "Bullish: put buyers giving up on a fall.")),
        "Unclear ₹ Cr": ("%.0f",
            "Positions opened or closed whose side cannot be judged: volatility barely moved, "
            "the premium is too small to read, or no volatility fits the price (a stale "
            "print, deep in the money). Shown so nothing is silently dropped."),
        "Bullish − bearish ₹ Cr": ("%+.0f",
            "(The two bullish boxes) − (the two bearish boxes) for this side, in ₹ crore of "
            "delta-weighted exposure. **Example:** +250 = bullish activity outweighed bearish "
            "by ₹250 crore of stock-equivalent exposure."),
    }


TIP_OPTQ = {"CE": _tip_optq("CE"), "PE": _tip_optq("PE")}

TIP_COVERAGE = {
    "Session": (None, "The trading day."),
    "Log": (None,
        "How much of that day's 09:15–15:30 was recorded.\n\n"
        "- 🟢 **Complete** — prices AND open interest for at least 95% of the session.\n"
        "- 🟡 **Partial** — part of the session, for example the morning only.\n"
        "- 🟠 **Very little** — under 10%.\n"
        "- ⚫ **No log** — the market traded but nothing was recorded; only the official close "
        "exists."),
    "Prices": (None,
        "First and last price reading that day, and the share of the session they cover.\n\n"
        "**Example:** 09:16–11:26 (35%) means prices were recorded from 09:16 to 11:26, about "
        "a third of the session."),
    "Open interest": (None,
        "First and last open-interest reading that day, and the share of the session they "
        "cover. It can differ from prices: open interest is read less often."),
    "Not logged": (None,
        "The stretches of the session with no prices or no open interest for more than a few "
        "minutes. They cannot be filled later: NSE does not publish past intraday open "
        "interest."),
}

TIP_WINDOW = {
    "Window": (None,
        "The time period this row covers.\n\n- **last 15 min** = from 15 minutes ago "
        "until now.\n- **since 09:20 (first log)** = since this page started recording "
        "today.\n\nA period is left out if the page was not recording that far back."),
    "Index pts": ("%+.1f",
        "How many points the index moved in that period.\n\n"
        "**Example:** +12.0 means the index is 12 points higher now than at the start of "
        "the period."),
}
TIP_WINDOW_BUCKET = (
    "How many of those points came from this group in that period.\n\n"
    "**Example:** index +2, Top −20, Rest +22 means the heavy stocks are falling but the "
    "smaller stocks are holding the index up.")

TIP_FRESH = {
    "Window": (None,
        "The time period this row covers, and the clock time it starts from.\n\n"
        "**Example:** 'last 30 min (from 13:00)' = from 13:00 until the latest reading."),
    "Bucket": (None, "The same groups as the main table: Top, Next, Rest."),
    "Price %": ("%+.2f",
        "The average move of the group's stocks inside this period only — not since "
        "yesterday.\n\n**Example:** −0.30 means these stocks fell 0.30% on average during "
        "the period."),
    "Futures OI %": ("%+.2f",
        "How much the number of open futures contracts changed inside this period only.\n\n"
        "**Example:** +0.80 means 0.80% more open contracts now than at the start of the "
        "period."),
    "Futures in this window": (None,
        "What futures traders did during this period only.\n\n" + _T_FUT_WORDS +
        "- ⚪ **Mixed** — the stocks do not agree.\n\n"
        "The main table compares now with yesterday's close. This table compares now "
        "with a few minutes ago. If the two say different things, this one is the newer "
        "news.\n\n"
        "**Example:** main table says 'long build' and this says 'short build' = buyers "
        "came in earlier today, but in the last half hour new sellers are coming in.\n\n"
        f"A stock votes only if its open contracts moved more than {WIN_OI_BAND}% and its "
        f"price more than {WIN_PX_BAND}% during the period. Those two limits are our own "
        "choice and have not been tested yet."),
    "Stock votes": (None,
        "The count behind the phrase, for this period only.\n\n"
        "▲ = votes for the up side (long build, short covering). ▼ = votes for the down "
        "side (short build, long unwind). 'flat' = did not move enough to vote."),
    "Names": (None,
        "How many stocks of the group had data at both the start and the end of the "
        "period.\n\n**Example:** 9/10 means one stock is missing from this row."),
}

_EFFORT = {
    "heavy buying": "🟢 heavy buying", "heavy selling": "🔴 heavy selling",
    "heavy trade, no progress": "🟠 heavy trade, no progress",
    "quiet rise": "🟢 quiet rise", "quiet fall": "🔴 quiet fall", "quiet": "⚪ quiet",
    "rising, normal trade": "🟢 rising, normal trade",
    "falling, normal trade": "🔴 falling, normal trade", "normal": "⚪ normal"}

TIP_EFFORT = {
    "Window": (None,
        "The time period this row covers. 'since the open' = the whole day so far."),
    "Bucket": (None, "The same groups as the main table: Top, Next, Rest."),
    "Price %": ("%+.2f",
        "How much the group's price moved in this period, with heavy stocks counting more "
        "(the same way the index counts them).\n\n**Example:** +0.45 means the group "
        "added 0.45% in the period."),
    "Traded ₹ Cr": ("%.0f",
        "The money that changed hands in the group's stocks during the period, in crore "
        "rupees.\n\n**Example:** 3,250 means ₹3,250 crore traded."),
    "Value vs normal": ("%.2f",
        "Traded money in this period divided by what these stocks NORMALLY trade in the "
        "same part of the day.\n\n"
        "**Example:** 1.80 means 1.8 times the usual amount. 1.00 = normal. 0.70 = 30% "
        "quieter than usual.\n\n"
        "Each time of day has its own normal — the first half hour always trades much more "
        "than lunchtime, so they are never compared with each other."),
    "Value share %": ("%.0f",
        "This group's share of all money traded in the index's stocks during the period. "
        "Compare with 'Usual share %' in the main table: higher than usual means money is "
        "crowding into this group."),
    "What it looks like": (None,
        "Price direction joined with how hard the stocks are trading:\n\n"
        "- 🟢 **Heavy buying** — price up, traded money 1.5× normal or more.\n"
        "- 🔴 **Heavy selling** — price down, traded money 1.5× normal or more.\n"
        "- 🟠 **Heavy trade, no progress** — lots of trading but price not moving: buyers "
        "and sellers are both active and neither is winning yet.\n"
        "- 🟢/🔴 **Quiet rise / quiet fall** — price moving on thin trade (0.8× normal or "
        "less).\n- otherwise 'normal trade'.\n\n"
        "A move counts when price changes more than 0.20% in 30 minutes (scaled for other "
        "windows; 0.50% for the whole day).\n\n"
        "**Tested:** these words did NOT tell what Nifty did next — heavy buying and quiet "
        "buying were followed by the same next hour. Heavy trade only told that the next "
        "hour would swing MORE."),
}

TIP_PRESSURE = {
    "Bucket": (None, "The same groups as the main table: Top, Next, Rest."),
    "Long build ₹ Cr": ("%.0f", "Rupee value of futures positions OPENED in stocks whose "
                        "price is UP over the same span — fresh buying."),
    "Short covering ₹ Cr": ("%.0f", "Rupee value of futures positions CLOSED in stocks whose "
                            "price is UP — sellers leaving. Bullish, but an exit, not new "
                            "money: a rise made mostly of this is weaker than one of long build."),
    "Short build ₹ Cr": ("%.0f", "Rupee value of futures positions OPENED in stocks whose "
                         "price is DOWN — fresh selling."),
    "Long unwind ₹ Cr": ("%.0f", "Rupee value of futures positions CLOSED in stocks whose "
                         "price is DOWN — buyers leaving."),
    "Bullish − bearish ₹ Cr": ("%+.0f", "(Long build + short covering) − (short build + long "
                               "unwind). The last point of the line above."),
}

TIP_STOCK = {
    "Stock": (None, "The stock's NSE symbol."),
    "Bucket": (None, "Which group this stock belongs to: Top, Next or Rest."),
    "Weight %": ("%.2f",
        "How much of the index this one stock is.\n\n**Example:** 10.44 means the stock is "
        "10.44% of the index. If it rises 1%, the index rises about 0.10%."),
    "Move %": ("%+.2f", "The stock's price change today, compared with yesterday's close."),
    "Index pts": ("%+.2f",
        "How many index points this one stock has added or taken away today.\n\n"
        "**Example:** +33.3 means the index is 33 points higher because of this stock "
        "alone.\n\nA heavy stock with a small move can add more points than a light stock "
        "with a big move."),
    "Futures OI %": ("%+.2f",
        "How much this stock's open futures contracts have changed today, compared with "
        "yesterday's close.\n\n+ means new positions were added. − means positions were "
        "closed."),
    "Futures": (None,
        "What futures traders are doing in this one stock today.\n\n" + _T_FUT_WORDS +
        "- ⚪ **Flat** — open contracts changed by less than 0.5%, so there is nothing "
        "to read yet."),
    "Call OI %": ("%+.2f",
        "How much this stock's open CALL option contracts have changed today, compared "
        "with yesterday's close."),
    "Put OI %": ("%+.2f",
        "How much this stock's open PUT option contracts have changed today, compared "
        "with yesterday's close."),
    "Traded ₹ Cr": ("%.0f",
        "The value of this stock's shares traded so far today, in crore rupees.\n\n"
        "**Example:** 1,228 means ₹1,228 crore traded."),
}

TIP_BASE = {
    "State at the close": (None,
        "How the trading day ended, in one of four types:\n\n"
        "- **Broad up** — most heavy stocks AND most smaller stocks rose.\n"
        "- **Heavyweights carried** — most heavy stocks rose, most smaller stocks fell.\n"
        "- **Heavyweights lagged** — most heavy stocks fell, most smaller stocks rose.\n"
        "- **Broad down** — most heavy stocks AND most smaller stocks fell.\n\n"
        "The last row is all days together, for comparison."),
    "Sessions": ("%d",
        "How many past trading days ended in this state. More days = a more reliable "
        "average."),
    "Next open gap (bps)": ("%+.1f",
        "On average, how the index OPENED the next morning compared with that day's "
        "close.\n\n**bps** = hundredths of a percent, so 100 bps = 1%.\n\n"
        "**Example:** 13.9 means the next morning opened 0.139% higher on average."),
    "Gap up %": ("%.1f",
        "Out of 100 such days, how many times the index opened higher the next morning.\n\n"
        "**Example:** 68.5 means it opened higher about 68 times out of 100."),
    "Next day (bps)": ("%+.1f",
        "On average, how much the index moved over the whole next day (close to close).\n\n"
        "100 bps = 1%. **Example:** −3.2 means it fell 0.032% on average — almost nothing."),
    "Next day up %": ("%.1f",
        "Out of 100 such days, how many times the index closed higher the next day. "
        "Around 50 is the same as tossing a coin."),
    "Next 5 sessions (bps)": ("%+.1f",
        "On average, how much the index moved over the next 5 trading days together.\n\n"
        "100 bps = 1%. **Example:** 21.5 means +0.215% over the 5 days."),
    "5 sessions up %": ("%.1f",
        "Out of 100 such days, how many times the index was higher 5 trading days later."),
}


def _cfg(cc, spec: dict) -> dict:
    """{column: (format, tooltip)} -> Streamlit column_config."""
    return {name: (cc.NumberColumn(name, format=fmt, help=tip) if fmt
                   else cc.TextColumn(name, help=tip))
            for name, (fmt, tip) in spec.items()}


def _fmt(v, f="{:+.2f}"):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def _blank(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Float64 (nullable) for the numeric columns. A column that is all-None is otherwise an
    object column and Streamlit prints the literal word "None" in every cell — which on
    this page reads as "no data ever" rather than "not loaded yet"."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Float64")
    return df


def _dash_empty(df: pd.DataFrame, spec: dict) -> tuple:
    """(df, spec) with every numeric column that is ENTIRELY empty shown as '—' text.
    Streamlit renders a missing number as the word 'None', which reads as a bug; a dash says
    'not available'. Partly filled columns stay numeric, so their formatting is kept."""
    spec = dict(spec)
    for c, (fmt, tip) in list(spec.items()):
        if fmt and c in df.columns and df[c].isna().all():
            df[c] = "—"
            spec[c] = (None, tip)
    return df, spec


FRESH_OK, FRESH_WARN = 180.0, 360.0      # seconds: green below, orange below, red above


def out_of_step(index_pts: float, explained: float | None) -> bool:
    """True when NSE's stock rows and its index print disagree by more than 2 points AND 10%
    of the move. Each NSE feed refreshes about once a minute, but not in step — stock rows ran
    30-50s behind the index row in the same payload (15 Oct 2026). On a big day that is a
    point or two; on a small one it is most of the move: Midcap Select read +38.1 from its
    stocks against +31.7 on the index (a 20% miss), enough to reorder 'Share of move'."""
    if explained is None:
        return False
    miss = abs(index_pts - explained)
    return miss > 2.0 and miss > 0.10 * max(abs(index_pts), 1e-9)


def freshness(snap: dict, now: dt.datetime | None = None) -> dict:
    """Every clock behind the table, in one place, as ages in seconds.

    THERE ARE FOUR CLOCKS, AND THEY ARE NOT THE SAME CLOCK:
      price stamp   - the time NSE printed on its price data. Measured 15 Oct 2026: it
                      moves about every 55-60s (all five NSE price feeds do, out of step
                      with one another), however often it is asked.
      price read    - when this page last ASKED NSE. Can be 10s ago while the stamp is
                      2 minutes old; that means NSE has nothing newer, not that we are late.
      OI stamp      - the time NSE printed on its open-interest data (~every 55s).
      OI read       - when the last open-interest sweep finished here.
    Stock rows also run behind the index stamp (median lag shown separately). Showing one
    "updated" time would hide which of these is the slow one."""
    now = now or _now()

    def age(x):
        return None if x is None else max(0.0, (now - x).total_seconds())

    def level(a):
        if a is None or not snap.get("live"):
            return "closed" if a is not None else "none"
        return "ok" if a <= FRESH_OK else "warn" if a <= FRESH_WARN else "stale"

    p_read = snap.get("price_read_at")
    o_read = snap.get("deriv_ts")
    out = dict(
        price_stamp=snap.get("ts"), price_age=age(snap.get("ts")),
        price_read=p_read, price_read_age=age(p_read),
        oi_stamp=snap.get("oi_nse_to"), oi_age=age(snap.get("oi_nse_to")),
        oi_read=o_read, oi_read_age=age(o_read),
        stock_lag=snap.get("stock_lag_s"), drawn=now,
        next_price=(None if p_read is None or not snap.get("live")
                    else max(0.0, CASH_EVERY - age(p_read))),
        next_oi=(None if o_read is None or not snap.get("live")
                 else max(0.0, DERIV_EVERY - age(o_read))))
    out["price_level"] = level(out["price_age"])
    out["oi_level"] = level(out["oi_age"])
    return out


def _ago(s) -> str:
    if s is None:
        return "—"
    s = int(round(s))
    return f"{s}s ago" if s < 120 else f"{s // 60} min {s % 60:02d}s ago" if s < 3600 \
        else f"{s // 3600}h {s % 3600 // 60:02d}m ago"


_LEVEL = {"ok": ":green[●]", "warn": ":orange[●]", "stale": ":red[●]",
          "closed": ":gray[●]", "none": ":gray[○]"}


def _time_strip(st, snap: dict) -> None:
    """The four clocks, right above the table, refreshed with the page every 15s."""
    f = freshness(snap)
    with st.container(border=True):
        c = st.columns(4)
        if f["price_stamp"]:
            c[0].markdown(f"{_LEVEL[f['price_level']]} **Prices** · NSE time "
                          f"**{f['price_stamp']:%H:%M:%S}**  \n"
                          f":gray[{_ago(f['price_age'])}"
                          + (f" · stocks ~{f['stock_lag']:.0f}s older"
                             if f["stock_lag"] and f["stock_lag"] >= 5 else "") + "]")
        else:
            c[0].markdown(":gray[○] **Prices** · not loaded yet")
        if f["oi_stamp"]:
            c[1].markdown(f"{_LEVEL[f['oi_level']]} **Open interest** · NSE time "
                          f"**{f['oi_stamp']:%H:%M:%S}**  \n"
                          f":gray[{_ago(f['oi_age'])}]")
        else:
            c[1].markdown(":gray[○] **Open interest** · first reading in progress")
        if snap.get("live"):
            c[2].markdown(f"**Next check** · prices in "
                          f"**{_fmt(f['next_price'], '{:.0f}')}s**, OI in "
                          f"**{_fmt(f['next_oi'], '{:.0f}')}s**  \n"
                          f":gray[last checked {_ago(f['price_read_age'])}]")
        else:
            c[2].markdown("**Next check** · market closed  \n"
                          ":gray[final numbers for the session]")
        c[3].markdown(f"**Page drawn** · **{f['drawn']:%H:%M:%S}**  \n"
                      ":gray[redraws every 15s]")
        if f["price_level"] in ("warn", "stale") or f["oi_level"] in ("warn", "stale"):
            st.caption(":orange[A dot that is not green means NSE has not published newer "
                       "data for over 3 minutes (red: over 6). Read the table as of that "
                       "time, not as now.]")
        else:
            st.caption("NSE time = the time printed on NSE's own data. Prices and open "
                       "interest each move about once a minute, however often this page "
                       "asks — a stamp up to two minutes old is normal.")


def _provenance(snap: dict, eod_day) -> pd.DataFrame:
    """One row per family of numbers: where it comes from and the session it describes."""
    px = f"{snap['ts']:%d %b %Y %H:%M:%S}"
    oi = (f"{snap['oi_nse_to']:%d %b %Y %H:%M:%S}" if snap.get("oi_nse_to")
          else "not loaded yet")
    base = f"{snap['oi_base_date']:%d %b %Y}" if snap.get("oi_base_date") else "NSE feed"
    wd = f"{snap['weights_day']:%d %b %Y}" if snap.get("weights_day") else "—"
    out = pd.DataFrame([
        ("Index level, points, %", "NSE live feed", px, "TODAY, live"),
        ("Stock moves, Return %, Adv, Index pts", "NSE live feed", px, "TODAY, live"),
        ("Weight %", "NSE live feed (previous-close weights)", wd, "TODAY's index weights"),
        ("Activity % — traded value so far", "NSE live feed", px, "TODAY, live"),
        ("Activity % — the 'normal' it is compared with", "archive, 21-session median",
         f"up to {base}", "history (by design)"),
        ("Futures / Call / Put OI — the CHANGE", "NSE live feed", oi, "TODAY, live"),
        ("Futures / Call / Put OI — the base it is a % of", "archive end-of-day total",
         base, "YESTERDAY's close (by design)"),
        ("Futures now, Stock votes, Options now", "computed from the two rows above", oi,
         "TODAY, live"),
        ("The 'Reference only' box, Delivery %", "Daily Cash Market archive",
         f"{eod_day:%d %b %Y}" if eod_day else "—", "YESTERDAY — reference only"),
    ], columns=["Numbers", "Source", "Stamp", "Describes"])
    if snap.get("replay"):
        out["Source"] = out["Source"].str.replace("NSE live feed", "log of NSE's live feed")
        out["Describes"] = (out["Describes"]
                            .str.replace("TODAY, live", f"{snap['day']:%d %b}, last reading")
                            .str.replace("TODAY's", f"{snap['day']:%d %b}'s")
                            .str.replace("YESTERDAY", "the session before"))
    return out


def _oi_tape_rows(oi_tape: list, snap: dict) -> pd.DataFrame:
    """Each bucket's mean futures OI change at every logged sweep — is the build growing?"""
    rows = []
    buckets = {b["label"]: set(snap["stocks"].loc[snap["stocks"]["bucket"] == b["label"],
                                                  "symbol"]) for b in snap["buckets"]}
    for e in oi_tape:
        row = {"t": pd.Timestamp(e["t"])}
        for lbl, mem in buckets.items():
            v = [np.clip(x[1] / (x[0] - x[1]) * 100, -_OI_CLIP, _OI_CLIP)
                 for k, x in e["d"].items() if k in mem and x[0] - x[1] > 0]
            row[lbl] = float(np.mean(v)) if v else None
        rows.append(row)
    return pd.DataFrame(rows).set_index("t") if rows else pd.DataFrame()


_OI_CHART = {
    "Futures OI": ("fut_oi", "futures OI vs yesterday, %",
                   "Open futures contracts compared with yesterday's close, averaged over the "
                   "bucket's stocks. Rising = positions being added through the day. A fall in "
                   "the last hour is usually same-day positions being closed, not a change of "
                   "view. Read it next to the futures price below: price up while this rises "
                   "= long build; price down while this rises = short build."),
    "Call OI": ("ce", "call OI vs yesterday, %",
                "Open CALL option contracts compared with yesterday's close (every strike and "
                "expiry). Rising calls usually means traders selling calls, expecting a "
                "ceiling — this data cannot prove they were sold rather than bought."),
    "Put OI": ("pe", "put OI vs yesterday, %",
               "Open PUT option contracts compared with yesterday's close (every strike and "
               "expiry). Rising puts usually means traders selling puts, expecting a floor — "
               "this data cannot prove they were sold rather than bought."),
    "Put − Call": ("pc", "put % minus call %, points",
                   "Put OI change minus call OI change. Above zero = puts growing faster. This "
                   "line is RAW: the table's 'Options now' label also takes out the normal "
                   "monthly pattern (calls grow faster early in the month, puts near expiry), "
                   "so the two can disagree slightly early or late in the month."),
}
_PX_CHART = {
    "Futures price": ("fut_px", "futures move since yesterday's close, %",
                      "Each bucket's near-month futures price change since yesterday's close, "
                      "with heavy stocks counting more (the way the index counts them). It "
                      "tracks the cash move closely; the difference between them is the "
                      "premium, shown on the other setting."),
    "Futures premium": ("basis", "futures premium over cash, basis points (100 = 1%)",
                        "How far the near-month futures trade above the cash price, in basis "
                        "points (100 = 1%). Normally positive (the cost of carrying the stock "
                        "to expiry) and shrinking day by day toward zero at expiry. A premium "
                        "RISING through the day = futures buyers paying up more than cash "
                        "buyers; FALLING or negative = futures sellers pressing. A stock with "
                        "a dividend coming before expiry trades at a lower or negative "
                        "premium for that reason alone."),
}


def _with_segments(long: pd.DataFrame, tcol: str, group: str,
                   max_gap: pd.Timedelta = pd.Timedelta(minutes=6)) -> pd.DataFrame:
    """Number each unbroken run of points per line, so a line chart drawn with
    detail=segment STOPS at a logging gap instead of drawing a straight line across it — the
    laptop slept 11:07-15:30 on 16 Oct 2026 and the old charts joined 11:07 to the next
    reading as if the market had moved smoothly in between."""
    if long.empty:
        return long.assign(seg=pd.Series(dtype=int))
    long = long.sort_values([group, tcol]).copy()
    gap = long.groupby(group)[tcol].diff() > max_gap
    long["seg"] = gap.groupby(long[group]).cumsum().astype(int)
    long["line"] = long[group].astype(str) + "·" + long["seg"].astype(str)
    return long


def _alt_lines(st, wide: pd.DataFrame, ylab: str, height: int = 260,
               zero: bool = False, fmt: str = "+.2f") -> None:
    """A wide frame (time index x one column per line) as an Altair line chart that breaks
    at gaps. Replaces st.line_chart everywhere on this page for that reason."""
    import altair as alt
    if wide is None or wide.empty:
        return
    idx = wide.index.name or "t"
    long = (wide.reset_index().rename(columns={idx: "t"})
            .melt("t", var_name="series", value_name="v").dropna())
    long = _with_segments(long, "t", "series")
    ch = (alt.Chart(long).mark_line(strokeWidth=2)
          .encode(x=alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M")),
                  y=alt.Y("v:Q", title=ylab, scale=alt.Scale(zero=False)),
                  color=alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom")),
                  detail="line:N",
                  tooltip=[alt.Tooltip("t:T", title="time", format="%H:%M"), "series:N",
                           alt.Tooltip("v:Q", title=ylab, format=fmt)])
          .properties(height=height))
    if zero:
        ch = ch + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#888").encode(y="y:Q")
    st.altair_chart(ch, width="stretch")


_Q_COLOR = {"long build": "#1f8a4c", "short covering": "#8fd19e",
            "short build": "#c93a3a", "long unwind": "#f2a65a"}
_Q_LABEL = {"long build": "long build — fresh buying", "short covering": "short covering — shorts exiting",
            "short build": "short build — fresh selling", "long unwind": "long unwind — longs exiting"}


def _flow_chart(st, snap: dict, labels: list, buckets: dict) -> None:
    """Price on top, rupees of futures positions added / closed per block underneath."""
    import altair as alt
    syms = tuple(sorted(snap["stocks"]["symbol"]))
    _pressure_chart(st, snap, labels, buckets)
    st.markdown("**Who is doing what, block by block — futures price and positions**")
    c1, c2 = st.columns([3, 1])
    bk = c1.segmented_control("Bucket", labels, default=labels[0],
                              key=f"im_flow_bucket_{snap['code']}",
                              label_visibility="collapsed") or labels[0]
    mins = c2.segmented_control("Block", [5, 15], default=5,
                                format_func=lambda m: f"{m} min",
                                key=f"im_flow_block_{snap['code']}",
                                label_visibility="collapsed") or 5
    res = oi_flow_blocks(snap["oi_tape"], buckets, snap.get("weights") or {},
                         _archive_oi_base(syms, snap["day"]),
                         _archive_fut_close(syms, snap["day"]), snap["day"], minutes=mins)
    fl, pr = res["flow"], res["price"]
    if fl.empty or pr.empty:
        st.caption("Needs a few open-interest readings with futures prices (logged every ~1-2 "
                   "minutes from 09:15). Nothing is back-filled.")
        return
    lv = pressure_level(snap["oi_tape"], {bk: buckets[bk]},
                        _archive_oi_base(syms, snap["day"]),
                        _archive_fut_close(syms, snap["day"]), snap["day"], since="open",
                        band=PRESSURE_BAND)
    nb = net_blocks(lv["net"][bk], mins) if not lv["net"].empty else pd.DataFrame()
    g = by_side(fl[fl.bucket == bk])
    unchanged = fl[(fl.bucket == bk) & (fl.quadrant == "price unchanged")].cr.abs().sum()
    g["from"] = g["start"].dt.strftime("%H:%M")
    g["to"] = g["end"].dt.strftime("%H:%M")
    g["what"] = g["quadrant"].map(_Q_LABEL)
    px = pr[[bk]].reset_index().rename(columns={bk: "price"}).dropna()
    st.markdown(f"**{bk} — net direction through the session (bullish minus bearish)**")
    x = alt.X("end:T", title=None, axis=alt.Axis(format="%H:%M"))
    top = (alt.Chart(px).mark_line(strokeWidth=2, color="#9aa4b2")
           .encode(x=x, y=alt.Y("price:Q", title="futures move since yesterday, %",
                                scale=alt.Scale(zero=False)),
                   tooltip=[alt.Tooltip("end:T", title="time", format="%H:%M"),
                            alt.Tooltip("price:Q", title="futures % since yesterday",
                                        format="+.2f")])
           .properties(height=170))
    if not nb.empty:
        nb = nb.assign(dir=np.where(nb["step"] >= 0, "bullish block", "bearish block"),
                       side=np.where(nb["cum"] >= 0, "bullish so far", "bearish so far"),
                       when=pd.to_datetime(nb["end"]).dt.strftime("%H:%M"))
        segs = _with_segments(nb.assign(b=bk), "end", "b",
                              max_gap=pd.Timedelta(minutes=mins * 3))
        cum = (alt.Chart(segs).mark_area(opacity=0.25, color="#6aa9e9", line={"color": "#6aa9e9"})
               .encode(x=x, y=alt.Y("cum:Q", title="running total, ₹ Cr (↑ bullish)"),
                       detail="line:N",
                       tooltip=[alt.Tooltip("when:N", title="time"),
                                alt.Tooltip("cum:Q", title="running total ₹ Cr",
                                            format="+,.0f")])
               .properties(height=170))
        step = (alt.Chart(nb.dropna(subset=["step"])).mark_bar(size=max(3, 10 if mins == 15 else 6))
                .encode(x=x, y=alt.Y("step:Q", title=f"change per {mins} min, ₹ Cr"),
                        color=alt.Color("dir:N", title=None,
                                        scale=alt.Scale(domain=["bullish block", "bearish block"],
                                                        range=["#1f8a4c", "#c93a3a"]),
                                        legend=alt.Legend(orient="bottom")),
                        tooltip=[alt.Tooltip("when:N", title="block ending"),
                                 alt.Tooltip("step:Q", title="net this block ₹ Cr",
                                             format="+,.0f"),
                                 alt.Tooltip("cum:Q", title="running total ₹ Cr",
                                             format="+,.0f")])
                .properties(height=170))
        zero0 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#666").encode(y="y:Q")
        st.altair_chart(alt.vconcat(top, cum + zero0, step + zero0).resolve_scale(x="shared"),
                        width="stretch")
        st.caption(f"Middle: {bk}'s running total of bullish minus bearish futures money since "
                   "this morning's first reading — every stock's positions opened or closed and "
                   "its price move over the same span (price up: long build + short covering = "
                   "bullish; price down: short build + long unwind = bearish). Rising = buyers "
                   f"gaining all session; falling = sellers gaining. Bottom: one bar per {mins} "
                   "minutes = how much that running total moved in the block (green up, red "
                   "down). The bars add up exactly to the middle line, so a run of green bars is "
                   "the line climbing. Read it against the price on top: price rising with the "
                   "line rising = futures money backing the move; price rising while the line "
                   "falls = the move is NOT backed. Changing 5 / 15 minutes only changes how "
                   "the same line is cut into bars. One quirk: a stock whose price crosses back "
                   "over its morning level moves its whole position from one side to the other, "
                   "so a single bar can jump without much new trading — hover a bar to see it.")
    with st.expander(f"{bk} detail — the four kinds of trading behind each block"):
        _quadrant_bars(st, g, top, x, mins, unchanged, bk)
    sh = [r for r in res["sharp"] if r["bucket"] == bk]
    if sh:
        st.markdown("**Unusual blocks today**")
        for r in sh[-8:]:
            word = ("closed" if r["net"] < 0 else "opened")
            st.markdown(f"- **{r['end']:%H:%M}** — ₹{abs(r['net']):,.0f} Cr net {word} "
                        f"({r['times']:.1f}× this session's typical block); biggest part "
                        f"**{r['main']}** ₹{abs(r['main_cr']):,.0f} Cr")
        st.caption(f"'Unusual' = net positions opened or closed in a block at least "
                   f"{FLOW_SHARP_X:.0f}× the median block so far today (judged once "
                   f"{FLOW_SHARP_MIN} blocks exist). That threshold is a judgement, not "
                   "measured — this log is the first intraday stock open-interest history "
                   "there is. A large block of closing near 15:00-15:20 is usually same-day "
                   "positions being squared off.")
    else:
        st.caption(f"No unusual block in {bk} in this session (none at {FLOW_SHARP_X:.0f}× the "
                   "session's typical block).")
    _call_put_chart(st, snap, labels, buckets, res["price"])
    _option_positioning_view(st, snap, labels, buckets, res["price"])
    with st.expander("Calls and puts block by block (detail behind the line above)"):
        _options_flow_chart(st, snap, res, bk, mins, labels, buckets)


def _option_positioning_view(st, snap: dict, labels: list, buckets: dict,
                             price: pd.DataFrame) -> None:
    """Calls and puts split into long build / writing / short covering / long unwind, per
    bucket, in rupees — the options twin of the futures pressure table."""
    import altair as alt
    st.markdown("**Who is doing what in options — buyers and writers, calls and puts, by "
                "bucket**")
    if snap.get("is_expiry"):
        st.caption("Off on the monthly expiry day: expiring options are settled through the "
                   "session, so their open interest and prices move on schedule, not on views.")
        return
    mode = st.segmented_control(
        "Options judged on", ["iv", "premium"], default="iv",
        format_func=lambda v: ("volatility — time decay and the stock's move taken out"
                               if v == "iv" else "premium — as most sites show it"),
        key=f"im_opt_mode_{snap['code']}", label_visibility="collapsed") or "iv"
    op = option_positioning(snap["oi_tape"], buckets, mode)
    if not op["n"]:
        if snap["day"] < OPT_READ_FROM:
            st.info("This table starts on 3 Nov 2026, when the per-contract option read "
                    "began. Earlier sessions have only call and put totals (the chart above).")
        elif snap["oi_tape"]:
            st.warning(
                f"**No options table yet for {snap['day']:%d %b}.** The program recording "
                "this session started before the options read was added, so its readings "
                "carry call and put totals only. It fills in within a few minutes of the "
                "recorder restarting on the current code — it now does that by itself when "
                "the code changes; a recorder started earlier needs one restart (the "
                "'EqBTST Index capture' job, or closing its background python window).")
        else:
            st.caption("Fills in once open-interest readings are logged this session.")
        return
    for leg, name in (("CE", "Calls"), ("PE", "Puts")):
        bull = OPT_BULL[leg]
        rows = []
        for lbl in labels:
            a = op["last"][lbl][leg]
            net = sum(a[q][1] if q in bull else -a[q][1] for q in OPT_QUADS)
            rows.append({"Bucket": lbl, "Long build ₹ Cr": a["long build"][1],
                         "Writing ₹ Cr": a["writing"][1],
                         "Short covering ₹ Cr": a["short covering"][1],
                         "Long unwind ₹ Cr": a["long unwind"][1],
                         "Unclear ₹ Cr": a["unread"][1], "Bullish − bearish ₹ Cr": net})
        st.markdown(f"*{name}*  ·  bullish = "
                    + (" + ".join(bull)) + "  ·  bearish = "
                    + " + ".join(q for q in OPT_QUADS if q not in bull))
        st.dataframe(_blank(pd.DataFrame(rows), list(rows[0])[1:]), hide_index=True,
                     width="stretch", column_config=_cfg(st.column_config, TIP_OPTQ[leg]))
        # ITS OWN CHART, straight under its own table: calls and puts used to share one line
        # (with a futures-price panel on top), so a call-writing hour and a put-buying hour
        # cancelled into a flat line and neither could be read
        net = op["net_leg"][leg].reindex(columns=labels) if not op["net_leg"][leg].empty \
            else pd.DataFrame()
        if len(net.dropna(how="all")) >= 2:
            long_n = _with_segments(net.reset_index().melt("t", var_name="bucket",
                                                          value_name="cr").dropna(),
                                    "t", "bucket")
            colors = alt.Scale(domain=labels,
                               range=["#e8826b", "#6aa9e9", "#9aa4b2"][:len(labels)])
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#888").encode(y="y:Q")
            line = (alt.Chart(long_n).mark_line(strokeWidth=2)
                    .encode(x=alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M")),
                            y=alt.Y("cr:Q", title=f"{name.lower()}: bullish − bearish, ₹ Cr",
                                    axis=alt.Axis(tickCount=8, format=",.0f")),
                            color=alt.Color("bucket:N", scale=colors, title=None,
                                            legend=alt.Legend(orient="bottom")),
                            detail="line:N",
                            tooltip=[alt.Tooltip("t:T", title="time", format="%H:%M"),
                                     "bucket:N",
                                     alt.Tooltip("cr:Q", title=f"{name} bullish − bearish ₹ Cr",
                                                 format="+,.0f")])
                    .properties(height=400))   # 220 flattened a day's swing into a strip
            st.altair_chart(line + zero, width="stretch")
            st.caption(f"{name} through the session, each bucket's bullish − bearish (the last "
                       "column of the table above, reading by reading). Above zero = "
                       + ("call buying + calls bought back outweigh call writing + call "
                          "buyers leaving." if leg == "CE" else
                          "put writing + put buyers leaving outweigh put buying + puts bought "
                          "back."))
    st.caption(
        "Every option contract whose open interest changed since yesterday's close is put in "
        "one box. Open interest says whether positions were OPENED (up) or CLOSED (down). "
        + ("Who did it is judged on the contract's **implied volatility**, worked out from "
           "its premium and the same-month future, yesterday and now. Holding volatility "
           "fixed takes out time decay and the stock's own move, so what is left — volatility "
           "up or down — says whether buyers or sellers were pressing: opened + volatility up "
           "= buyers (long build), opened + volatility down = writers. 'Unclear' = the premium "
           f"is within {OPT_MIN_TICKS} ticks (₹{OPT_TICK * OPT_MIN_TICKS:.2f}) of what decay "
           f"and the stock's move alone would give, is under ₹{OPT_MIN_PREMIUM:g}, or is a "
           "stale print (its last trade was hours old — caught when a call and put at one "
           "strike imply very different volatilities). "
           "Time is counted on the market's clock (trading sessions; nights, weekends and "
           "holidays count as nothing) — on calendar time every Monday read as buying. "
           "**Measured** on 2.0 million stock-option changes (Aug 2024 – Oct 2026): this read "
           "points the same way as the stock's move on 36% of stock-days (the premium read: "
           "99%); build-ups split about evenly between buyers and writers on ordinary days "
           "and Mondays alike (51% / 49%), but lean to writing the day after a long holiday "
           "(37% buyers); about a quarter of the rupees come out 'Unclear'. It leans to call "
           "writing on days the stock rises (rank −0.28, both years) — partly volatility "
           "falling as stocks rise, partly, perhaps, calls sold into rallies; the data cannot "
           "tell which."
           if mode == "iv" else
           "Here who did it is judged on the **premium** alone, the way most sites read it. "
           "**Measured** on 2.0 million stock-option changes (Aug 2024 – Oct 2026): this read "
           "pointed the same way as the stock's own move on 99% of stock-days — it mostly "
           "repeats the price — and called 74% of build-ups 'writing' on days the stock did "
           "not move, because premiums lose value every day (time decay). Shown for "
           "comparison only.")
        + " ₹ = contracts × lot × future × delta: exposure that moves with the stock, so a "
        "far out-of-the-money contract counts for little. Lots are per contract month (NSE's "
        "Jan-2027 revision). Intraday open interest includes positions closed before 15:20. "
        "Describes the session; not tested as a forecast.")


def _quadrant_bars(st, g, top, x, mins, unchanged, bk) -> None:
    """The four-colour block bars (moved into a detail box: as the main view they mixed
    opposite colours in one bar and gave no direction at a glance)."""
    import altair as alt
    if g.empty:
        st.caption("No blocks yet.")
        return
    bars = (alt.Chart(g).mark_bar(size=max(3, 10 if mins == 15 else 5))
            .encode(x=x,
                    y=alt.Y("sum(side_cr):Q", title="₹ Cr  (↑ bullish / ↓ bearish)",
                            stack="zero"),
                    color=alt.Color("what:N", title=None,
                                    scale=alt.Scale(domain=[_Q_LABEL[q] for q in _Q_COLOR],
                                                    range=list(_Q_COLOR.values())),
                                    legend=alt.Legend(orient="bottom", columns=2)),
                    order=alt.Order("quadrant:N"),
                    tooltip=[alt.Tooltip("from:N", title="from"),
                             alt.Tooltip("to:N", title="to"),
                             alt.Tooltip("what:N", title="what"),
                             alt.Tooltip("sum(side_cr):Q", title="₹ Cr", format="+,.0f"),
                             alt.Tooltip("sum(stocks):Q", title="stocks")])
            .properties(height=230))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#666").encode(y="y:Q")
    st.altair_chart(alt.vconcat(top, bars + zero).resolve_scale(x="shared"),
                    width="stretch")
    st.caption(f"Each {mins}-minute block split into the four kinds of trading: above the line "
               "bullish — dark green long build (fresh buying), light green short covering "
               "(sellers leaving); below bearish — dark red short build (fresh selling), orange "
               "long unwind (buyers leaving). Dark = new money, light = exits. Each stock is "
               "judged on its own price inside the block, so one block can show both sides. "
               f"Left out: ₹{unchanged:,.0f} Cr moved while the price printed the same. These "
               "colours depend on the block length (a 15-minute block nets the ups and downs "
               "inside it differently); the net-direction chart above does not.")


_MOVE_COLOR = {"price up": "#2e9d5b", "price down": "#d64545", "price unchanged": "#8a8f98"}


def _pressure_chart(st, snap: dict, labels: list, buckets: dict) -> None:
    """All buckets: futures price on top, bullish - bearish rupees underneath (block-free)."""
    import altair as alt
    syms = tuple(sorted(snap["stocks"]["symbol"]))
    st.markdown("**Futures pressure through the session — bullish minus bearish, by bucket**")
    since = st.segmented_control("Counted from", ["close", "open"], default="close",
                                 format_func=lambda v: ("since yesterday's close" if v == "close"
                                                        else "since this morning's first reading"),
                                 key=f"im_press_since_{snap['code']}",
                                 label_visibility="collapsed") or "close"
    base, fc = _archive_oi_base(syms, snap["day"]), _archive_fut_close(syms, snap["day"])
    lv = pressure_level(snap["oi_tape"], buckets, base, fc, snap["day"], since=since,
                        band=PRESSURE_BAND)
    ser = session_series(snap["oi_tape"], [], buckets, snap.get("weights") or {}, base, fc, {},
                         snap["day"])
    # a reading where every bucket is below quorum is an all-empty row, not data: without
    # this the guard passed on 26 Oct 2026 (no lot sizes) and the table below crashed with
    # KeyError 'bucket' because `parts` had no rows at all
    net, price = lv["net"].dropna(how="all"), ser["fut_px"]
    parts = lv.get("parts", pd.DataFrame())
    if net.empty or len(net) < 2 or price.empty or parts.empty:
        st.caption("Needs a few open-interest readings with futures prices from today"
                   + (" — and lot sizes, which need yesterday's F&O close in the archive "
                      f"(archive holds {snap['oi_base_date']:%d %b})."
                      if snap.get("oi_base_date") and not net.empty else "."))
        return
    long_p = _with_segments(net[labels].reset_index().melt("t", var_name="bucket",
                                                           value_name="cr").dropna(),
                            "t", "bucket")
    long_px = _with_segments(price[labels].reset_index().melt("t", var_name="bucket",
                                                              value_name="pct").dropna(),
                             "t", "bucket")
    colors = alt.Scale(domain=labels, range=["#e8826b", "#6aa9e9", "#9aa4b2"][:len(labels)])
    x = alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M"))
    top = (alt.Chart(long_px).mark_line(strokeWidth=2)
           .encode(x=x, y=alt.Y("pct:Q", title="futures move since yesterday, %",
                                scale=alt.Scale(zero=False)),
                   color=alt.Color("bucket:N", scale=colors, title=None,
                                   legend=alt.Legend(orient="bottom")),
                   detail="line:N",
                   tooltip=[alt.Tooltip("t:T", title="time", format="%H:%M"), "bucket:N",
                            alt.Tooltip("pct:Q", title="futures %", format="+.2f")])
           .properties(height=170))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#888").encode(y="y:Q")
    bot = (alt.Chart(long_p).mark_line(strokeWidth=2)
           .encode(x=x, y=alt.Y("cr:Q", title="bullish − bearish, ₹ Cr"),
                   color=alt.Color("bucket:N", scale=colors, title=None),
                   detail="line:N",
                   tooltip=[alt.Tooltip("t:T", title="time", format="%H:%M"), "bucket:N",
                            alt.Tooltip("cr:Q", title="bullish − bearish ₹ Cr",
                                        format="+,.0f")])
           .properties(height=220))
    st.altair_chart(alt.vconcat(top, bot + zero).resolve_scale(x="shared"), width="stretch")
    last = parts.groupby("bucket").tail(1).set_index("bucket").reindex(labels)
    st.dataframe(_blank(pd.DataFrame({
        "Bucket": labels,
        "Long build ₹ Cr": last["long build"].values,
        "Short covering ₹ Cr": last["short covering"].values,
        "Short build ₹ Cr": last["short build"].values,
        "Long unwind ₹ Cr": last["long unwind"].values,
        "Bullish − bearish ₹ Cr": net[labels].iloc[-1].values}),
        ["Long build ₹ Cr", "Short covering ₹ Cr", "Short build ₹ Cr", "Long unwind ₹ Cr",
         "Bullish − bearish ₹ Cr"]), hide_index=True, width="stretch",
        column_config=_cfg(st.column_config, TIP_PRESSURE))
    st.caption("How the line is built: for every stock, its futures positions opened or closed "
               + ("since yesterday's close" if since == "close" else "since this morning's "
                  "first reading")
               + " and its futures price move over the SAME span. Price up + positions up = "
               "long build, price up + positions down = short covering (both bullish); price "
               "down + positions up = short build, price down + positions down = long unwind "
               "(both bearish). The rupee value of those positions goes on its side; the line "
               "is bullish minus bearish for the bucket. Rising = buyers gaining; falling = "
               "sellers gaining. A stock changes side only when its price moves more than "
               f"{PRESSURE_BAND:.2f}% past its starting price the other way — a heavy stock "
               "hovering on yesterday's close used to throw its whole position from one side "
               "to the other every minute (INFY ±1,300 Cr twice in six minutes on 26 Oct). "
               "It never changes with the block size below"
               + (". It uses the same four-way rule as the main table's 'Futures now', but counts "
                  "rupees instead of one vote per stock, judges price by the FUTURES move "
                  "(the table uses cash) and has no 0.5% minimum — so a nearly flat stock can "
                  "sit on a different side here than in the table (3 of 50 Nifty names on "
                  "16 Oct)." if since == "close" else ".")
               + " Describes the session; not tested as a forecast.")


def _call_put_chart(st, snap: dict, labels: list, buckets: dict, price: pd.DataFrame) -> None:
    """Calls minus puts through the session, every bucket, under the buckets' price lines."""
    import altair as alt
    st.markdown("**Call side vs put side through the session — calls minus puts, by bucket**")
    if snap.get("is_expiry"):
        st.caption("Off on the monthly expiry day: expiring options are settled through the "
                   "session and their open interest collapses on schedule.")
        return
    syms = tuple(sorted(snap["stocks"]["symbol"]))
    base = _archive_oi_base(syms, snap["day"])
    cp = call_put_series(snap["oi_tape"], buckets, base, snap["day"])
    if cp.empty or len(cp) < 2 or price.empty:
        st.caption("Needs a few open-interest readings logged during the session.")
        return
    # THE EXPIRY CYCLE, TAKEN OUT. Raw, this line read 'call side heavier' on 95% of Nifty
    # bucket-days in a cycle's first week and on 2% the day before expiry — the calendar,
    # not anyone's view. cp_cycle_rate() measures that drift in rupees from the archive.
    left = snap.get("sessions_left")
    rates = cp_cycle_rate(syms, snap["day"]) if left else None
    rate = rates.get(min(int(left), CP_RATE_CAP)) if rates else None
    usual = call_put_usual(snap["oi_tape"], buckets, base, snap["day"], rate)
    mode = st.segmented_control(
        "Calls minus puts", ["adjusted", "raw"], default="adjusted",
        format_func=lambda v: ("minus the usual expiry-cycle drift" if v == "adjusted"
                               else "as traded"),
        key=f"im_cp_mode_{snap['code']}", label_visibility="collapsed") or "adjusted"
    if mode == "adjusted" and not usual.empty:
        cp = cp.subtract(usual.reindex(cp.index)[cp.columns])
    elif mode == "adjusted":
        st.caption("The usual expiry-cycle drift is unavailable right now (needs the "
                   "archive and the sessions left to expiry) — showing the line as traded.")
        mode = "raw"
    long_cp = (cp[labels].reset_index().melt("t", var_name="bucket", value_name="cr")
               .dropna())
    long_px = (price[labels].reset_index().rename(columns={"end": "t"})
               .melt("t", var_name="bucket", value_name="pct").dropna())
    colors = alt.Scale(domain=labels, range=["#e8826b", "#6aa9e9", "#9aa4b2"][:len(labels)])
    x = alt.X("t:T", title=None, axis=alt.Axis(format="%H:%M"))
    top = (alt.Chart(long_px).mark_line(strokeWidth=2)
           .encode(x=x, y=alt.Y("pct:Q", title="futures move since yesterday, %",
                                scale=alt.Scale(zero=False)),
                   color=alt.Color("bucket:N", scale=colors, title=None,
                                   legend=alt.Legend(orient="bottom")),
                   tooltip=[alt.Tooltip("t:T", title="time", format="%H:%M"), "bucket:N",
                            alt.Tooltip("pct:Q", title="futures %", format="+.2f")])
           .properties(height=170))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#888").encode(y="y:Q")
    bot = (alt.Chart(long_cp).mark_line(strokeWidth=2)
           .encode(x=x, y=alt.Y("cr:Q", title=("calls − puts vs usual, ₹ Cr (↑ calls ahead)"
                                                if mode == "adjusted" else
                                                "calls − puts, ₹ Cr (↑ call side heavier)")),
                   color=alt.Color("bucket:N", scale=colors, title=None),
                   tooltip=[alt.Tooltip("t:T", title="time", format="%H:%M"), "bucket:N",
                            alt.Tooltip("cr:Q", title="calls − puts ₹ Cr", format="+,.0f")])
           .properties(height=230))
    st.altair_chart(alt.vconcat(top, bot + zero).resolve_scale(x="shared"), width="stretch")
    st.caption("Bottom line = call contracts opened today minus put contracts opened today, "
               "valued at the share price (₹ crore of exposure, not premium), every strike "
               "and expiry, counted from yesterday's close. Line going UP = the call side is "
               "getting heavier; going DOWN = the put side is getting heavier. "
               + (f"**Adjusted:** minus what the expiry cycle alone usually adds by this time "
                  f"of day at {left} session(s) to expiry — a typical day's "
                  f"{rate * 100:+.1f}% of the bucket's call/put exposure, measured on every "
                  "archived session since Jul 2024 and spread through the day by normal "
                  "trading volume (that spreading is an assumption). Above zero = calls "
                  "running AHEAD of their usual pace for this point of the cycle. Raw, this "
                  "line read 'call side heavier' on 95% of Nifty bucket-days in a cycle's "
                  "first week and 2% the day before expiry; adjusted, 51% and 67% (tested "
                  "on Sep 2025 – Oct 2026, fitted on earlier data). Still tilted in the "
                  "first two sessions after an expiry: adjusted Nifty read 'call side "
                  "heavier' on 71% of those days, so treat a moderate rise then as normal. "
                  "(Keying the drift on sessions SINCE expiry instead over-corrected those "
                  "days to 28–38% on three indices — tested and not used.)"
                  if mode == "adjusted" else
                  "**As traded:** includes the expiry cycle — new contracts fill with calls "
                  "faster early in the month, and calls are closed faster into expiry, so "
                  "this line leans up early in a cycle and down near its end on most days."))
    st.markdown("\n".join([
        "| Bottom line | Price (top) | Usual reading |",
        "|---|---|---|",
        "| ⬆ call side heavier | ⬇ falling | calls being written into the fall — sellers "
        "building a ceiling (bearish pressure) |",
        "| ⬇ put side heavier | ⬆ rising | puts being written into the rise — sellers "
        "building a floor (support) |",
        "| ⬆ call side heavier | ⬆ rising | ambiguous: call buying, or calls written at "
        "higher strikes |",
        "| ⬇ put side heavier | ⬇ falling | ambiguous: put buying (hedging), or puts written "
        "lower down |"]))
    st.caption("Only the first two rows are fairly clear. Telling buying from writing needs the "
               "option's own price, which this log does not keep per strike. Each bucket is "
               "drawn once at least 60% of its stocks have been read; a stock NSE skips in "
               "one reading keeps its last value. Bucket sizes differ (Rest 30 has 30 names), "
               "so compare each line's DIRECTION, not its size against another bucket. "
               "Positions opened and closed within the day are inside these numbers until "
               "they are squared off.")


def _options_flow_chart(st, snap: dict, res: dict, bk: str, mins: int,
                        labels: list | None = None, buckets: dict | None = None) -> None:
    """Calls and puts opened / closed per block, coloured by what price did meanwhile.

    It has its OWN bucket and block switches: it sits far below the futures chart's
    switches, and with only those to go by the chart did not say which bucket it was showing."""
    import altair as alt
    if labels and buckets:
        c1, c2 = st.columns([3, 1])
        bk = c1.segmented_control("Bucket", labels, default=bk,
                                  key=f"im_opt_bucket_{snap['code']}",
                                  label_visibility="collapsed") or bk
        m2 = c2.segmented_control("Block", [5, 15], default=mins,
                                  format_func=lambda m: f"{m} min",
                                  key=f"im_opt_block_{snap['code']}",
                                  label_visibility="collapsed") or mins
        if m2 != mins:
            syms = tuple(sorted(snap["stocks"]["symbol"]))
            res = oi_flow_blocks(snap["oi_tape"], buckets, snap.get("weights") or {},
                                 _archive_oi_base(syms, snap["day"]),
                                 _archive_fut_close(syms, snap["day"]), snap["day"],
                                 minutes=m2)
            mins = m2
    st.markdown(f"**{bk} — calls and puts, block by block, against the same price**")
    if snap.get("is_expiry"):
        st.caption("Off on the monthly expiry day: expiring options are settled through the "
                   "session and their open interest collapses on schedule, so the bars would "
                   "show the calendar, not traders.")
        return
    of, pr = res["options"], res["price"]
    g = of[of.bucket == bk].copy()
    if g.empty or pr.empty:
        st.caption("Needs a few open-interest readings logged during the session.")
        return
    g["from"] = g["start"].dt.strftime("%H:%M")
    g["to"] = g["end"].dt.strftime("%H:%M")
    px = pr[[bk]].reset_index().rename(columns={bk: "price"}).dropna()
    x = alt.X("end:T", title=None, axis=alt.Axis(format="%H:%M"))
    top = (alt.Chart(px).mark_line(strokeWidth=2, color="#9aa4b2")
           .encode(x=x, y=alt.Y("price:Q", title="futures move since yesterday, %",
                                scale=alt.Scale(zero=False)),
                   tooltip=[alt.Tooltip("end:T", title="time", format="%H:%M"),
                            alt.Tooltip("price:Q", title="futures %", format="+.2f")])
           .properties(height=150))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#666").encode(y="y:Q")

    def leg_chart(leg, title):
        d = g[g.leg == leg]
        return (alt.Chart(d).mark_bar(size=max(3, 10 if mins == 15 else 5))
                .encode(x=x,
                        y=alt.Y("sum(cr):Q", title=title, stack="zero"),
                        color=alt.Color("move:N", title=None,
                                        scale=alt.Scale(domain=list(_MOVE_COLOR),
                                                        range=list(_MOVE_COLOR.values())),
                                        legend=alt.Legend(orient="bottom")),
                        tooltip=[alt.Tooltip("from:N", title="from"),
                                 alt.Tooltip("to:N", title="to"),
                                 alt.Tooltip("move:N", title="price in this block"),
                                 alt.Tooltip("sum(cr):Q", title="₹ Cr (notional)",
                                             format="+,.0f"),
                                 alt.Tooltip("sum(stocks):Q", title="stocks")])
                .properties(height=170)) + zero

    st.altair_chart(alt.vconcat(top, leg_chart("Calls", "CALLS ₹ Cr (+ opened / − closed)"),
                                leg_chart("Puts", "PUTS ₹ Cr (+ opened / − closed)"))
                    .resolve_scale(x="shared"), width="stretch")
    st.caption("Bars = option contracts opened (above zero) or closed (below) in each block, "
               "every strike and expiry, valued at the share price (exposure, not premium "
               "paid). Colour = what the stock's price did in that SAME block: green up, red "
               "down, grey unchanged. Each stock is judged on its own move.")
    st.markdown(
        "\n".join([
            "| In one block | Usual reading | How sure |",
            "|---|---|---|",
            "| 🟢 **Puts opened while price rose** | put writing — sellers building a floor "
            "| fairly clear |",
            "| 🔴 **Calls opened while price fell** | call writing — sellers building a ceiling "
            "| fairly clear |",
            "| 🟢 Calls opened while price rose | call buying, or call writing at higher "
            "strikes | can't tell |",
            "| 🔴 Puts opened while price fell | put buying (hedging), or put writing lower "
            "down | can't tell |",
            "| Calls closed while price rose | call writers buying back, or buyers taking "
            "profit | can't tell |",
            "| Puts closed while price fell | put writers buying back, or buyers taking "
            "profit | can't tell |"]))
    st.caption("Why only two rows are 'fairly clear': deciding buying from writing needs the "
               "option's own price (premium up = bought, premium down = written), and this "
               "log keeps open interest per expiry, not per-strike premiums. The share price "
               "is a stand-in that only settles the two cases above.")
    sh = [r for r in res["osharp"] if r["bucket"] == bk]
    if sh:
        st.markdown("**Unusual option blocks**")
        for r in sh[-8:]:
            word = "closed" if r["net"] < 0 else "opened"
            st.markdown(f"- **{r['end']:%H:%M}** — {r['leg'].lower()} ₹{abs(r['net']):,.0f} Cr "
                        f"net {word} ({r['times']:.1f}× this session's typical block), mostly "
                        f"while {r['move']}")


def _session_charts(st, snap: dict, labels: list, buckets: dict) -> None:
    """Open interest and futures price through the session, per bucket."""
    syms = tuple(sorted(snap["stocks"]["symbol"]))
    st_ = snap["stocks"]
    cash_prev = {r.symbol: r.ltp / (1 + r.pct / 100) for r in st_.itertuples()
                 if r.ltp and r.pct is not None}
    ser = session_series(snap["oi_tape"], snap["tape"], buckets, snap.get("weights") or {},
                         _archive_oi_base(syms, snap["day"]),
                         _archive_fut_close(syms, snap["day"]), cash_prev, snap["day"])
    _flow_chart(st, snap, labels, buckets)
    oi = ser["fut_oi"]
    if oi.empty or len(oi) < 2:
        return
    exp = st.expander("Running totals since yesterday's close — open interest and futures "
                      "price, all buckets")
    st = exp                                        # the two charts below live in it
    st.caption("These lines add up the whole day, so open interest climbs on most days as "
               "same-day positions pile up; a sharp five-minute unwind barely bends them. For "
               "what is happening NOW, use the block chart above.")
    st.markdown("**Open interest through the session, by bucket**")
    pick = st.segmented_control("Open interest", list(_OI_CHART), default="Futures OI",
                                key=f"im_oi_chart_{snap['code']}",
                                label_visibility="collapsed") or "Futures OI"
    col, ylab, tip = _OI_CHART[pick]
    if snap.get("is_expiry") and col in ("ce", "pe", "pc"):
        st.caption("Options lines are off on the monthly expiry day — expiring options are "
                   "settled through the session and their open interest collapses on "
                   "schedule (the table switches them off for the same reason).")
    else:
        _alt_lines(st, ser[col][labels], ylab, zero=True)
    st.caption(tip)

    st.markdown("**Futures price through the session, by bucket**")
    have_px = not ser["fut_px"].dropna(how="all").empty
    if not have_px:
        st.caption("Futures prices have been logged with each open-interest reading since "
                   "15 Oct 2026 14:06; this fills in as readings arrive.")
        return
    pick2 = st.segmented_control("Futures", list(_PX_CHART), default="Futures price",
                                 key=f"im_px_chart_{snap['code']}",
                                 label_visibility="collapsed") or "Futures price"
    col2, ylab2, tip2 = _PX_CHART[pick2]
    _alt_lines(st, ser[col2][labels], ylab2, zero=True,
               fmt="+.0f" if col2 == "basis" else "+.2f")
    st.caption(tip2)
    exp_note = (" Today is the monthly expiry, so the NEXT month's contract is followed — the "
                "expiring one is pinned to cash." if snap.get("is_expiry") else "")
    st.caption(f"One point per open-interest reading ({len(oi)} today). A bucket is drawn "
               f"only when at least {SERIES_QUORUM:.0%} of its stocks are in a reading. Price "
               "lines follow the near-month contract all day." + exp_note +
               " A straight segment means nothing was logged then (dashboard or capture job "
               "not running, laptop asleep). Intraday open interest includes positions that "
               "are closed before 15:20.")


def _official_table(st, d, cc) -> None:
    """DCM's end-of-day read of one session, as a table."""
    st.dataframe(pd.DataFrame([{
        "Bucket": r.label,
        "Return %": r.ret_pct,
        "Futures at that close": _FUT.get(r.fut_read, "—"),
        "Options at that close": _OPT.get(r.opt_read, "—"),
        "Delivery % (last 4 closes)": ", ".join(f"{v:.1f}" for v in r.deliv_trail)
                                      if r.deliv_trail else "—",
    } for r in d.rows]), hide_index=True, width="stretch", column_config=_cfg(cc, TIP_EOD))


def _official_summary(d) -> str:
    return (f"Official close: index **{d.index_ret:+.2f}%**, equal-weight basket "
            f"**{d.equal_ret:+.2f}%**, {d.adv_all} up / {d.dec_all} down · state: "
            f"*{d.state or '—'}*")


def _draw_no_log(st, code: str, day: dt.date) -> None:
    """A session that traded but was never logged: say so, and show what does exist."""
    st.warning(f"**No minute-by-minute log for {day:%A %d %b %Y}.** Nothing was captured that "
               "day (laptop off or asleep, or the capture job not running), and NSE does not "
               "publish past intraday open interest, so it cannot be rebuilt.")
    d = official_close_read(code, day)
    if d is None:
        st.caption("The official close for this day is not in the archive yet.")
        return
    st.markdown(f"**{DISPLAY[code]} — {day:%d %b %Y}.** " + _official_summary(d))
    _official_table(st, d, st.column_config)
    st.caption("From NSE's official end-of-day files (Daily Cash Market), not from this page's "
               "log. Every date back to 2018 is on Daily Cash Market → Index & Large Cap.")


def _draw(st, code: str, snap: dict | None = None) -> None:
    replay = snap is not None
    if not replay:
        touch(code)
        snap = snapshot(code)
    if snap is None or snap.get("empty"):
        st.warning("NSE's live feed is not answering"
                   + (f" — {snap['err']}." if snap and snap.get("err") else ".")
                   + " Retrying in the background. NSE blocks data-centre addresses, so "
                     "this lane works from a home or office connection, not from the cloud VM.")
        return
    name, ix = snap["display"], snap["index"]

    # ── freshness line: which session, how old, what is still loading ────────────────
    when = f"{snap['ts']:%d %b %H:%M:%S}"
    if replay:
        cov = snap["coverage"]
        st.info(f"📼 **Recording of {snap['day']:%A %d %b %Y}, not live.** Everything below is as "
                f"of the last logged reading, **{snap['as_of']:%H:%M}**. On this page 'today' "
                "means that session and 'yesterday' the one before it.\n\n"
                + coverage_text(cov))
        if cov["grade"] != "complete":
            st.warning("This log does not cover the whole session, so the table is **not the "
                       "day's close**, and charts stop or break where logging did. The official "
                       "close is in the box below.")
        if snap.get("ban_unknown"):
            st.caption("That day's F&O ban list was not preserved, so a banned stock's votes "
                       "may not be withheld here.")
    elif snap["live"]:
        st.caption(f":green[●] **Live** · NSE feed stamped **{when}** "
                   f"({snap['age_s']:.0f}s ago) · NSE publishes new prices and open "
                   "interest about once a minute (measured); this page "
                   f"asks every {CASH_EVERY:.0f}s and {DERIV_EVERY:.0f}s")
    else:
        st.caption(f":gray[●] **Market closed** — showing the session of **{when}** "
                   "as NSE last published it.")
    if not replay:
        # the log behind the charts, always in view: a table can look complete when it is
        # only NSE's closing figures with nothing behind them (30 Oct 2026, never logged)
        _cov = session_coverage(snap["tape"], snap["oi_tape"], snap["day"], _now())
        st.caption("**This session's log:** " + coverage_text(_cov)
                   + (" The table shows NSE's own published figures; the charts have nothing "
                      "to draw until a reading is logged." if _cov["grade"] == "none" else ""))
    if snap.get("err"):
        st.caption(f":orange[Last poll failed ({snap['err']}) — showing the last good read.]")
    for n in snap["notes"]:
        st.warning(n)

    c = st.columns(5)
    c[0].metric(f"{name} (cap-weighted)", f"{ix['last']:,.2f}",
                f"{ix['pts']:+,.2f} ({ix['pct']:+.2f}%)",
                help=f"The {name} index itself: its level now, and below it today's change "
                     "in points and in percent, compared with yesterday's close.\n\n"
                     "**Cap-weighted** means big companies count more than small ones. "
                     "One heavy stock can move the index more than ten light ones.")
    c[1].metric("Equal-weight basket", f"{snap['ew']:+.2f}%",
                help=f"The simple average move of all {snap['n']} stocks in the index "
                     "today. Every stock counts the same, big or small.\n\n"
                     "It answers: how is the *typical* stock doing?\n\n"
                     "**Example:** index +1.60% but this shows +0.20% = the index is up "
                     "mostly because of a few heavy stocks; most stocks did little.")
    c[2].metric("Carry spread", f"{snap['spread']:+.2f} pp",
                help="The index move minus the average-stock move (the tile on the left). "
                     "It shows who is doing the work today.\n\n"
                     "- **Plus** = the heavy stocks are pulling the index up more than "
                     "the typical stock.\n"
                     "- **Minus** = the typical stock is doing better than the index; the "
                     "heavy stocks are holding it back.\n"
                     "- **Near zero** = heavy and light stocks are moving together.\n\n"
                     f"For {name}, a gap bigger than ±{snap['carry_band']:.2f} is a clear "
                     "difference; smaller than that is normal noise.\n\n"
                     "**pp** = percentage points. +0.30 pp means the index did 0.30% "
                     "better than the average stock.")
    c[3].metric("Advancing", f"{snap['adv']} / {snap['n']}",
                help=f"How many of the {snap['n']} index stocks are up today.\n\n"
                     f"Right now: {snap['adv']} up, {snap['dec']} down, {snap['unch']} "
                     "unchanged.\n\n"
                     "Many stocks up = a broad, healthy move. Index up with few stocks up "
                     "= a narrow move that depends on a few names.")
    c[4].metric("Sessions to expiry",
                "—" if snap["sessions_left"] is None else
                ("today" if snap["is_expiry"] else str(snap["sessions_left"])),
                help="How many trading days are left until this month's stock futures and "
                     "options contracts end"
                     + (f" ({snap['front_expiry']:%d %b %Y})" if snap["front_expiry"] else "")
                     + ". Holidays and weekends are not counted.\n\n"
                     "Why it matters: in the last few days before expiry, traders move "
                     "their positions to next month, and option numbers behave "
                     "differently. On expiry day itself the options column is switched "
                     "off.\n\n"
                     "A dash means the first futures and options reading has not arrived "
                     "yet.")
    if snap["regime"]:
        _at = ("so far" if snap["live"] else
               f"at {snap['as_of']:%H:%M}, the last reading"
               if replay and snap["coverage"]["grade"] != "complete" else "at the close")
        st.markdown(f"**{snap['regime']}**"
                    + (f" · state {_at}: *{snap['state']}*" if snap["state"] else ""))

    # ── bucket table ─────────────────────────────────────────────────────────────────
    st.subheader("Weight buckets — who is moving the index")
    if replay:
        _off = official_close_read(code, snap["day"])
        if _off is not None:
            with st.expander(f"Official close of {snap['day']:%d %b}, for comparison — "
                             "from Daily Cash Market",
                             expanded=snap["coverage"]["grade"] != "complete"):
                st.markdown(_official_summary(_off))
                _official_table(st, _off, st.column_config)
                st.caption("NSE's end-of-day files. The table below is the log's last reading. "
                           "Futures words can differ even on a full day: NSE's live futures "
                           "open interest runs high.")
    else:
        _time_strip(st, snap)
    eod_day, eod = previous_close_read(code, snap["day"] if replay else None)
    eod_rows = {r.label: r for r in eod.rows} if eod is not None else {}
    st.caption((f"Every number in this table is **{snap['day']:%d %b}'s session at "
                f"{snap['as_of']:%H:%M}**, rebuilt from the log of NSE's live feed by the same "
                "code as the live page. The session before it is in its own box further down."
                if replay else
                f"Every number in this table is **today's session ({snap['day']:%d %b})** "
                "from NSE's live feed. Yesterday's end-of-day read is kept in its own box "
                "further down so the two cannot be confused.")
               + " The sidebar **As-of close** date does not apply to this page.")
    tbl = []
    for b in snap["buckets"]:
        tbl.append({
            "Bucket": b["label"],
            "Weight %": b["weight"],
            "Index pts": b["pts"],
            "Share of move %": b["share"],
            "Return %": b["ret_ew"],
            "Adv": f"{b['adv']}/{b['n_present']}",
            "Activity %": b["pace"],
            "Value share %": b["value_share"],
            "Usual share %": b["value_share_usual"],
            "Futures OI %": b["fut_oi_pct"],
            "Futures now": _FUT.get(b["fut_read"], "—"),
            "Stock votes": b["fut_votes"],
            "Call OI %": b["ce_pct"],
            "Put OI %": b["pe_pct"],
            "Options now": ("off · expiry day" if snap["is_expiry"]
                            else _OPT.get(b["opt_read"], "—")),
            "F&O names": f"{b['fut_valid']}/{b['n_members']}",
        })
    cc = st.column_config
    _num_cols = ["Weight %", "Index pts", "Share of move %", "Return %", "Activity %",
                 "Value share %", "Usual share %", "Futures OI %", "Call OI %", "Put OI %"]
    _tdf, _tspec = _dash_empty(_blank(pd.DataFrame(tbl), _num_cols), TIP_MAIN)
    st.dataframe(_tdf, hide_index=True, width="stretch", column_config=_cfg(cc, _tspec))
    if snap.get("stuck"):
        st.caption(f"**Activity % and Value share % are '—' because NSE's traded value is "
                   f"frozen along with its prices** ({len(snap['stuck'][0])} stocks stuck"
                   + (f" since {snap['stuck'][1]}" if snap["stuck"][1] else "")
                   + "). Using the frozen figure would show trading 60–80% below normal, "
                   "which is false. No other NSE source gives live traded value per stock "
                   "(the per-stock quote page refuses scripts; the other index lists are "
                   "frozen too), so both columns come back by themselves when NSE's feed "
                   "moves again.")
    st.caption("ℹ️ Hover over any column heading for a plain-English explanation with an "
               "example.")
    if snap["have_weights"] and out_of_step(ix["pts"], snap["explained_pts"]):
        st.warning(f"NSE's stock prices and its index print are out of step right now: the "
                   f"stocks add up to **{snap['explained_pts']:+.1f}** points, the index "
                   f"shows **{ix['pts']:+.1f}**. Each NSE feed refreshes about once a minute "
                   "but not at the same moment. Bucket points and 'Share of move' are "
                   "approximate until the next refresh lines them up; the direction of each "
                   "bucket is still right.")
    if snap["have_weights"]:
        st.caption(f"Self-check: the weights rebuild **{snap['explained_pts']:+.1f}** of the "
                   f"index's **{ix['pts']:+.1f}** points (miss {snap['residual_pts']:+.1f}). "
                   "A miss of a point or two is the 1–2 minute lag between NSE's stock rows "
                   "and its index print."
                   + (f" Weights are from {snap['weights_day']:%d %b}."
                      if snap["weights_day"] and snap["weights_day"] != snap["day"] else ""))
    else:
        st.caption("NSE's weight list is unavailable, so index points are hidden; returns "
                   "and breadth below need no weights.")
    if snap["deriv_busy"] and not snap["n_deriv"]:
        st.caption("⏳ First open-interest sweep running (about 15–25 seconds for 50 "
                   "stocks). The futures and options columns fill in together when it "
                   "finishes — never stock by stock.")
    elif snap["deriv_ts"]:
        st.caption(f"Open interest: NSE stamp **{snap['oi_nse_to']:%d %b %H:%M:%S}**, read "
                   f"at {snap['deriv_ts']:%H:%M:%S} ({snap['n_deriv']} of {snap['n']} "
                   "stocks). Change is today's; it is a % of "
                   + (f"the **{snap['oi_base_date']:%d %b}** closing total "
                      f"({snap['n_archive_base']} of {snap['n_deriv']} stocks matched to "
                      "the archive)." if snap["n_archive_base"] else
                      "NSE's own previous total (archive base unavailable — option % can "
                      "read high early in the session).")
                   + (f" ⚠️ **The archive's F&O close looks a session behind** — only "
                      f"{snap['n_archive_base']} of {snap['n_deriv']} stocks match it. The "
                      "rest use NSE's own previous total for the % columns, and the rupee "
                      "charts use lot sizes saved on earlier days. It corrects itself within "
                      "10 minutes of the Daily Cash Market F&O update landing."
                      if snap["n_deriv"] and snap["n_archive_base"] < 0.5 * snap["n_deriv"]
                      else "")
                   + (" **Monthly expiry day** — futures count every contract including "
                      "the one settling today; options are switched off."
                      if snap["is_expiry"] else ""))
        rs = reconcile_summary(snap["code"])
        _dr = reconcile_day(snap["code"], snap["day"]) if replay else None
        if _dr and _dr.get("fut"):
            st.caption(f"Checked against that day's official file (log's last reading "
                       f"{_dr['last_reading'][:5]}): stock direction matched "
                       f"**{_dr['fut']['sign_agree']:.0%}** for futures OI, "
                       f"**{(_dr.get('ce') or {}).get('sign_agree', 0):.0%}** for calls, "
                       f"**{(_dr.get('pe') or {}).get('sign_agree', 0):.0%}** for puts."
                       + ("" if _dr.get("full_day") else
                          " The log stopped early, so part of any miss is the rest of the day."))
        st.caption(
            "⚠️ **Futures open interest here is provisional.** NSE's live futures figure "
            "climbs through the day and is not what its official file prints that evening "
            "(options are not affected). "
            + (f"Checked every night: on the last {rs['days']} fully logged day(s) "
               f"({rs['from']} → {rs['to']}), each stock's live futures OI direction at the "
               f"close matched the official one **{rs['fut']:.0%}** of the time — about a "
               f"coin flip — against **{rs['ce']:.0%}** for calls and **{rs['pe']:.0%}** "
               "for puts. So read 'Futures now', the futures pressure lines and the futures "
               "blocks as price direction weighted by trading, not as positions."
               if rs and rs.get("fut") is not None and rs.get("ce") is not None
               and rs.get("pe") is not None else
               "The nightly check needs a fully logged day (09:15–15:20) before it can put a "
               "number on it; first audit (15 & 19 Oct) found direction matched 42–80%."))
    elif not snap["from_dcm"]:
        st.caption("Futures and options words need the Daily Cash Market engine, which is "
                   "not reachable from here; the price decomposition is unaffected.")
    st.caption(f"Buckets: {snap['bucket_source']}.")
    with st.expander("Is this live? — where every number comes from, with its time stamp"):
        st.dataframe(_provenance(snap, eod_day), hide_index=True, width="stretch",
                     column_config=_cfg(cc, TIP_SOURCE))
        _obs = observed_refresh(snap["tape"])
        _cd = None if replay else apply_cadence()
        if _cd:
            _r = _cd["result"]
            st.caption(
                f"Refresh measured on **{_cd['measured']}** (5-second polls): NSE's price "
                f"stamp changed every **{_fmt(_r['price stamp']['median_s'], '{:.0f}')}s** "
                f"and a stock's futures OI every "
                f"**{_fmt(_r['stock OI (futures)']['median_s'], '{:.0f}')}s** (median). "
                "The poll intervals below were set from that.")
        if not replay:
            st.caption((f"NSE has been publishing a new price stamp about every **{_obs:.0f}s** "
                        "in this session's log (this page asks every "
                        f"{CASH_EVERY:.0f}s, so it cannot see anything faster). "
                        if _obs else "")
                       + f"Open interest is re-read every {DERIV_EVERY:.0f}s while this page "
                         "is on screen.")
            st.caption("Only two things here are not today's: the BASE a percentage is taken "
                       "against (yesterday's close — that is what 'change' means) and the "
                       "reference box below. NSE's feed runs 1–2 minutes behind the tape.")
    if eod_rows:
        with st.expander(f"Reference only — yesterday's end-of-day read ({eod_day:%d %b}), "
                         "from Daily Cash Market"):
            _official_table(st, eod, cc)
            st.caption("Delivery is published after the close, so it can never be live. "
                       "Compare with the table above to see what changed overnight — e.g. "
                       "short build at yesterday's close, long build today.")
    if snap["misranked"]:
        with st.expander(f"Weight vs bucket: {len(snap['misranked'])} name(s) sit across "
                         "a boundary today"):
            st.caption("Not an error. Buckets follow NSE's 6-month-average ranking so a "
                       "week's prices do not reshuffle them; today's spot weights can "
                       "disagree at the margin.")
            for m in snap["misranked"]:
                st.markdown(f"- {m}")

    for b in snap["buckets"]:
        if b["top_pts"]:
            st.markdown(f"**{b['label']}** — biggest pushes: "
                        + " · ".join(f"{s} {v:+.1f} pts" for s, v in b["top_pts"]))

    # ── intraday trend ───────────────────────────────────────────────────────────────
    labels = [b["label"] for b in snap["buckets"]]
    buckets = {l: list(snap["stocks"].loc[snap["stocks"]["bucket"] == l, "symbol"])
               for l in labels}
    st.subheader("Through the session")
    if not snap["have_weights"] or len(snap["tape"]) < 2:
        st.caption("The intraday trend is built from this dashboard's own log, one row per "
                   "poll while the market is open. It starts when this page is first opened "
                   "on a session — nothing is back-filled.")
    else:
        fr = tape_frame(snap["tape"], buckets, snap["weights"])
        mv = window_moves(fr, labels)
        if mv:
            st.dataframe(pd.DataFrame([{"Window": r["window"], "Index pts": r["index"],
                                        **{l: r[l] for l in labels}} for r in mv]),
                         hide_index=True, width="stretch",
                         column_config=_cfg(cc, {
                             **TIP_WINDOW,
                             **{l: ("%+.1f", TIP_WINDOW_BUCKET) for l in labels}}))
            st.caption("Points added in each window, by bucket. This is where a hand-over "
                       "shows: the index flat over 30 minutes while the Top bucket gives "
                       "points back and the Rest adds them.")
        er = effort_result(snap["tape"], buckets, snap["weights"],
                           _archive_norms(tuple(sorted(snap["stocks"]["symbol"])),
                                          before=snap["day"] if replay else None))
        st.markdown("**Money vs move — traded value against price, by bucket**")
        if er:
            st.dataframe(_blank(pd.DataFrame([{
                "Window": f"{r['window']} (from {r['since']})", "Bucket": r["bucket"],
                "Price %": r["price"], "Traded ₹ Cr": r["value_cr"],
                "Value vs normal": r["ratio"], "Value share %": r["value_share"],
                "What it looks like": _EFFORT.get(r["word"], "—")} for r in er]),
                ["Price %", "Traded ₹ Cr", "Value vs normal", "Value share %"]),
                hide_index=True, width="stretch", column_config=_cfg(cc, TIP_EFFORT))
            top = next((r for r in er if r["window"] == "last 30 min"), None)
            if top and top["ratio"] and top["ratio"] >= EFFORT_HEAVY:
                st.info(f"**{top['bucket']} traded {top['ratio']:.1f}× its normal in the last "
                        "30 minutes.** Measured over 393 sessions: after that, Nifty's next "
                        "hour moved about 20 basis points on average against about 15 "
                        "normally — bigger swings, in EITHER direction. It did not tell "
                        "which way.")
            st.caption("Tested before this table was built, on 393 sessions (Oct 2024 – Jun "
                       "2026): heavy buying, heavy selling and quiet moves in the Top 10 "
                       "were followed by the same next-hour Nifty move as each other — "
                       "about 50% up either way. Heavy selling was followed by a slight "
                       "bounce, not more selling. What heavy value DID foretell is a bigger "
                       "move (about 20 vs 15 basis points in the next hour). Read this table "
                       "as who is trading and how hard, not as where the index goes next.")
        else:
            st.caption("Traded value has been logged with each price row since 15 Oct 2026; "
                       "this fills in once this session's log has a few rows.")
        st.markdown("**Index points contributed by each bucket (cumulative, today)**")
        _alt_lines(st, fr[labels], "index points", zero=True, fmt="+.1f")
        with st.expander("Carry spread and breadth through the session"):
            st.markdown("**Carry spread (index − equal-weight, pp)**")
            _alt_lines(st, fr[["spread"]].rename(columns={"spread": "Carry spread"}),
                       "pp", height=200, zero=True)
            st.markdown("**Advancing stocks (% of the index)**")
            _alt_lines(st, fr[["adv_pct"]].rename(columns={"adv_pct": "Advancing %"}),
                       "%", height=200, fmt=".0f")
        st.caption(f"Log holds {len(fr)} polls from {fr.index[0]:%H:%M} to "
                   f"{fr.index[-1]:%H:%M}. A straight segment across a gap means the "
                   "dashboard was not running then.")
    rf = recent_flow(snap["oi_tape"], snap["tape"], buckets, _dcm())
    st.markdown("**Fresh positioning — futures build-up inside the last 15 / 30 / 60 minutes**")
    if rf:
        st.dataframe(_blank(pd.DataFrame([{
            "Window": f"{r['window']} (from {r['since']})", "Bucket": r["bucket"],
            "Price %": r["price"], "Futures OI %": r["oi"],
            "Futures in this window": _FUT.get(r["read"], "—"),
            "Stock votes": r["votes"], "Names": r["names"]} for r in rf]),
            ["Price %", "Futures OI %"]), hide_index=True, width="stretch",
            column_config=_cfg(cc, TIP_FRESH))
        st.caption("The main table says what the day has added up to since yesterday's "
                   "close. This says what is being done **now**. Example: main table "
                   "'long build', this table 'short build' = the day's longs are in place "
                   "but the last half hour is fresh selling.")
    else:
        st.caption("Needs at least two open-interest sweeps 15 minutes apart, logged while "
                   "the market is open and this page is on screen. Nothing is back-filled.")
    # the charts leave out the same stocks the table withholds (F&O ban: positions can only
    # close; corporate action: OI re-based) — otherwise a banned name paints every block
    # "covering/unwind" for a rule, not a view
    _w = snap.get("withheld") or set()
    _session_charts(st, snap, labels,
                    {l: [s for s in m if s not in _w] for l, m in buckets.items()})

    # ── stock by stock ───────────────────────────────────────────────────────────────
    with st.expander("Stock by stock", expanded=False):
        d = snap["stocks"].copy()
        d["_o"] = d["bucket"].map({l: i for i, l in enumerate(labels)})
        d = d.sort_values(["_o", "weight"], ascending=[True, False])
        st.dataframe(_blank(pd.DataFrame({
            "Stock": d["symbol"], "Bucket": d["bucket"], "Weight %": d["weight"],
            "Move %": d["pct"], "Index pts": d["pts"],
            "Futures OI %": d["fut_oi_pct"],
            "Futures": [_QUAD.get(v, "—") for v in d["quadrant"]],
            "Call OI %": d["ce_pct"], "Put OI %": d["pe_pct"],
            "Traded ₹ Cr": d["value_cr"],
        }), ["Weight %", "Move %", "Index pts", "Futures OI %", "Call OI %", "Put OI %",
             "Traded ₹ Cr"]), hide_index=True, width="stretch",
            column_config=_cfg(cc, TIP_STOCK))

    br = state_base_rates(code)
    if br is not None:
        with st.expander("What followed each state before — end-of-day history, not a call"):
            st.dataframe(br.rename(columns={
                "state": "State at the close", "days": "Sessions",
                "gap_bps": "Next open gap (bps)", "gap_up": "Gap up %",
                "d1_bps": "Next day (bps)", "d1_up": "Next day up %",
                "d5_bps": "Next 5 sessions (bps)", "d5_up": "5 sessions up %"}),
                hide_index=True, width="stretch", column_config=_cfg(cc, TIP_BASE))
            st.caption("These are states AT THE CLOSE; the state shown above is the session "
                       "so far and can change before 15:30. The differences between rows "
                       "do not survive the index's own close-strength (Daily Cash Market → "
                       "Index & Large Cap → evidence). Base rates, not a forecast.")


def _session_picker(st, code: str) -> dt.date | None:
    """The session on screen: None = live. The URL carries it (?session=2026-10-29), so a past
    day survives a refresh and can sit in its own browser tab beside the live one. A bad or
    unknown date falls back to live instead of raising."""
    days = list_sessions(code)
    opts = ["live"] + [d.isoformat() for d in days]
    if "im_session" not in st.session_state:
        asked = st.query_params.get("session")
        st.session_state["im_session"] = asked if asked in opts else "live"
    elif st.session_state["im_session"] not in opts:
        st.session_state["im_session"] = "live"      # that day is not listed for this index

    def label(v: str) -> str:
        if v == "live":
            return "🔴 Live / latest from NSE"
        d = dt.date.fromisoformat(v)
        cov = _day_coverage(code, d)
        extra = ("official close only" if cov["grade"] == "none" else
                 f"{_span_txt(cov['prices'])}")
        return f"{d:%d %b %Y, %a} · {_COV_WORD[cov['grade']]} · {extra}"

    pick = st.selectbox("Session", opts, format_func=label, key="im_session",
                        help="Live, or any finished session since logging began on 13 Oct 2026. "
                             "A past day is rebuilt from the log as of its last reading; each "
                             "entry says how much of 09:15–15:30 the log holds. For older dates "
                             "use Daily Cash Market → Index & Large Cap (official closes only, "
                             "back to 2018).")
    if pick == "live":
        if "session" in st.query_params:
            del st.query_params["session"]
    else:
        st.query_params["session"] = pick
    if days:
        with st.expander(f"What is stored — {len(days)} finished sessions"):
            rows = [(d, _day_coverage(code, d)) for d in days]
            st.dataframe(pd.DataFrame([{
                "Session": f"{d:%d %b %Y, %a}",
                "Log": _COV_WORD[c["grade"]],
                "Prices": _span_txt(c["prices"]),
                "Open interest": _span_txt(c["oi"]),
                "Not logged": _holes_txt(c) or "—",
            } for d, c in rows]), hide_index=True, width="stretch",
                column_config=_cfg(st.column_config, TIP_COVERAGE))
            n = sum(c["grade"] == "complete" for _, c in rows)
            st.caption(f"{n} of {len(rows)} sessions are complete. A hole is any stretch over "
                       f"{COVER_GAP_MIN} minutes with no reading. Holes cannot be filled "
                       "afterwards: NSE does not publish past intraday open interest.")
    return None if pick == "live" else dt.date.fromisoformat(pick)


def render_page(st) -> None:
    """The Index Management lane. `st` is injected so this module never imports streamlit."""
    st.title("🏛️ Index Management — who is moving the index right now")
    st.markdown("🧭 **No arrow on this page.** It shows which weight bucket is carrying the "
                "index, live. It does not forecast — the end-of-day version of every column "
                "was tested and nothing survives.")
    with st.expander("How to read this page"):
        st.markdown(HELP)
    code = st.segmented_control("Index", list(NSE_NAME), default="NIFTY",
                                format_func=lambda c: DISPLAY[c], key="im_index") or "NIFTY"
    ensure_poller()                      # capture keeps running whatever session is on screen
    day = _session_picker(st, code)
    if day is not None:
        @st.fragment
        def _past():                     # a chart switch re-runs only this, no 15s redraw
            try:
                with _replay():
                    snap = session_snapshot(code, day)
                    if snap is None:
                        _draw_no_log(st, code, day)
                    else:
                        _draw(st, code, snap)
            except Exception as exc:                         # noqa: BLE001
                st.error(f"Could not rebuild {day:%d %b}: {type(exc).__name__}: {exc}")

        _past()
        return
    touch(code)
    if _st(code)["cash"] is None:
        if _claim():
            with st.spinner("Reading NSE…"):
                poll_cash(code)          # two quick calls so the first paint is not empty
        else:
            _sync_from_disk(code)        # another process is capturing: read its log

    @st.fragment(run_every="15s")
    def _live():
        try:
            _draw(st, code)
        except Exception as exc:                             # noqa: BLE001
            st.error(f"Index Management could not draw this refresh: "
                     f"{type(exc).__name__}: {exc}")

    _live()



# ── live open interest vs the official close ─────────────────────────────────────────
# NSE's live stock-FUTURES open interest drifts up through the day and is not what the
# bhavcopy prints that evening (audit 26 Oct 2026: at 15:29 live median +1.9..+2.5% vs
# official -0.3..+0.9%; per-stock sign agreed 42-74%; ~0.2 x the day's futures volume too
# high; expiry week far worse). Options matched (corr 0.99-1.00, sign 94-100%). This keeps
# score every night so the page can say, with fresh numbers, how far to trust each column.
RECONCILE_FILE = "oi_reconcile.jsonl"
RECONCILE_FULL_DAY = dt.time(15, 20)    # a log ending earlier is kept but not scored


def _reconciled() -> set:
    p = DATA_DIR / RECONCILE_FILE
    try:
        return {(r["day"], r["code"]) for r in
                (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}
    except (OSError, ValueError):
        return set()


def reconcile_oi(days_back: int = 10, today: dt.date | None = None) -> int:
    """Score each finished logged day's LAST live reading against that day's bhavcopy, once
    per (day, index). Needs the archive to hold the day and the session before it, so it
    runs the morning after. Returns rows written."""
    from . import data
    today = today or _now().date()
    done = _reconciled()
    written = 0
    for code in NSE_NAME:
        for k in range(1, days_back + 1):
            day = today - dt.timedelta(days=k)
            if (day.isoformat(), code) in done:
                continue
            tape = [e for e in _load_jsonl(_path("oi", code, day)) if e.get("d")]
            if not tape:
                continue
            last = tape[-1]
            syms = tuple(sorted(last["d"]))
            ph = ",".join("?" * len(syms))
            try:
                with data._connect() as c:
                    prev = c.execute("select max(trade_date) from fno_bhavcopy "
                                     "where trade_date < ?", [day]).fetchone()[0]
                    have = c.execute("select count(*) from fno_bhavcopy where trade_date = ?",
                                     [day]).fetchone()[0]
                    if prev is None or not have:
                        continue                      # archive not there yet: try tomorrow
                    df = c.execute(f"""
                        select symbol, trade_date,
                               case when instrument = 'FUTSTK' then 'fut'
                                    when option_type = 'CE' then 'ce' else 'pe' end k,
                               expiry_date, sum(open_interest) oi
                        from fno_bhavcopy
                        where instrument in ('FUTSTK', 'OPTSTK') and expiry_date >= ?
                          and trade_date in (?, ?) and symbol in ({ph})
                        group by 1, 2, 3, 4""", [day, prev, day, *syms]).df()
            except Exception:                                # noqa: BLE001
                continue
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            pv = df[df.trade_date == prev].groupby(["symbol", "k"])["oi"].sum()
            ev = df[df.trade_date == day].groupby(["symbol", "k"])["oi"].sum()
            fut_prev = df[(df.trade_date == prev) & (df.k == "fut")]
            res = {"fut": [], "ce": [], "pe": []}
            for s in syms:
                F = (last.get("x") or {}).get(s, {}).get("F", {})
                byexp = {pd.Timestamp(r.expiry_date).date().isoformat(): r.oi
                         for r in fut_prev[fut_prev.symbol == s].itertuples()}
                lot = _proven_lot([(byexp.get(e), v[0] - v[1]) for e, v in F.items()])
                if not lot:
                    continue
                d = last["d"][s]
                for leg, j in (("fut", 1), ("ce", 3), ("pe", 5)):
                    b, e_ = pv.get((s, leg)), ev.get((s, leg))
                    if not b:
                        continue
                    res[leg].append((d[j] * lot / b * 100, ((e_ or 0) - b) / b * 100))
            t_last = last["t"][11:19]
            row = dict(day=day.isoformat(), code=code, last_reading=t_last,
                       full_day=t_last >= RECONCILE_FULL_DAY.strftime("%H:%M:%S"),
                       is_expiry=day.isoformat() in {e for e in (last.get("x") or {}).get(
                           syms[0], {}).get("F", {})},
                       written=_now().isoformat(timespec="seconds"))
            for leg, pairs in res.items():
                if len(pairs) < 3:
                    continue
                a = np.array(pairs)
                row[leg] = dict(n=len(a), sign_agree=float((np.sign(a[:, 0]) == np.sign(a[:, 1])).mean()),
                                live_med=float(np.median(a[:, 0])), eod_med=float(np.median(a[:, 1])),
                                corr=float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) if len(a) > 2 else None)
            if "fut" not in row:
                if not last.get("x"):
                    # a log from before per-expiry detail was stored can never be scored:
                    # record that once, or every capture start re-reads it for ten days
                    _append(DATA_DIR / RECONCILE_FILE,
                            dict(day=day.isoformat(), code=code, full_day=False,
                                 skipped="log has no per-expiry futures detail"))
                    written += 1
                continue                              # otherwise (archive behind) retry later
            _append(DATA_DIR / RECONCILE_FILE, row)
            written += 1
    return written


def reconcile_day(code: str, day: dt.date) -> dict | None:
    """The stored live-vs-official score of one session, or None."""
    p = DATA_DIR / RECONCILE_FILE
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("code") == code and r.get("day") == day.isoformat():
                    return r
    except (OSError, ValueError):
        pass
    return None


def reconcile_summary(code: str, n: int = 10) -> dict | None:
    """Mean sign agreement (live vs official close) per leg over the last `n` FULL logged
    days of this index, plus how many days that is."""
    p = DATA_DIR / RECONCILE_FILE
    try:
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    except (OSError, ValueError):
        return None
    rows = sorted((r for r in rows if r["code"] == code and r.get("full_day")),
                  key=lambda r: r["day"])[-n:]
    if not rows:
        return None
    out = {"days": len(rows), "from": rows[0]["day"], "to": rows[-1]["day"]}
    for leg in ("fut", "ce", "pe"):
        v = [r[leg]["sign_agree"] for r in rows if leg in r]
        out[leg] = float(np.mean(v)) if v else None
    return out


def capture(until: dt.time = dt.time(15, 45)) -> None:
    """Headless capture for the scheduled job: poll all four indices from now until
    `until`, then step aside so the dashboard can take over."""
    print(f"[{_now():%d %b %H:%M:%S}] capture starting (pid {os.getpid()})", flush=True)
    # KEEP THE MACHINE AWAKE WHILE CAPTURING. On 14 Oct 2026 the laptop went to idle sleep at
    # 11:02 and woke at 14:33: three and a half hours of the session were never logged.
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED blocks IDLE sleep only, for this process's lifetime
    # (cleared in `finally`, and by Windows if the process dies). Closing the lid or pressing
    # sleep still sleeps the machine.
    _awake = False
    try:
        import ctypes
        _awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001))
    except Exception:                                        # noqa: BLE001
        pass
    if _awake:
        print("  idle sleep blocked until capture stops", flush=True)
    try:
        n = compress_old_logs()
        if n:
            print(f"  compressed {n} finished-day log file(s)", flush=True)
    except Exception:                                        # noqa: BLE001
        pass
    try:
        n = reconcile_oi()
        if n:
            print(f"  scored {n} logged day(s) of live open interest against the bhavcopy",
                  flush=True)
    except Exception as exc:                                 # noqa: BLE001
        _log_error("reconcile", f"{type(exc).__name__}: {exc}")
    for code in NSE_NAME:
        _st(code)
    ensure_poller()
    # Index option chains (index_fno.py) ride on this job: same NSE feed, its own writer lock
    # and log, so they are captured from 09:15 too. A failure there never stops this capture.
    _fno = None
    try:
        from . import index_fno as _fno
        _fno.compress_old_logs()
        _fno.reconcile()                 # yesterday's live log vs NSE's corrected close
        _fno.ensure_poller()
    except Exception as exc:                                 # noqa: BLE001
        _log_error("fno", f"index_fno not started: {type(exc).__name__}: {exc}")
        _fno = None
    last = None
    # STEP ASIDE WHEN THE CODE CHANGES. This job runs all day; a fix made mid-session never
    # reached it (3 Nov 2026: the options read was added at 10:15 but the job started at
    # 09:28 kept logging without it all day). Exiting releases the lock: the dashboard
    # (restarted on the new code) takes over at once, or the next 10-minute run of the task.
    _watched = [Path(__file__), Path(__file__).with_name("index_fno.py")]

    def _mtimes():
        return tuple(p.stat().st_mtime if p.exists() else None for p in _watched)
    try:
        _code_mtime = _mtimes()
    except OSError:
        _code_mtime = None
    try:
        while _now().time() < until and _now().weekday() < 5:
            time.sleep(30)
            try:
                if _code_mtime and _mtimes() != _code_mtime:
                    print(f"[{_now():%H:%M:%S}] code changed - stepping aside for the new "
                          "version", flush=True)
                    break
            except OSError:
                pass
            owner = _MINE["owner"]
            if owner != last:
                print(f"[{_now():%H:%M:%S}] "
                      + ("capturing" if owner else "another process is capturing - standing by"),
                      flush=True)
                last = owner
    finally:
        setattr(sys, _GEN_ATTR, None)            # stops the poller thread
        if _fno is not None:
            try:
                _fno.stop_poller()
            except Exception:                                # noqa: BLE001
                pass
        time.sleep(4)
        _release()
        if _awake:
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            except Exception:                                # noqa: BLE001
                pass
        rows = {c: len(_st(c)["tape"]) for c in NSE_NAME}
        print(f"[{_now():%d %b %H:%M:%S}] capture stopped; price rows logged today: {rows}",
              flush=True)


if __name__ == "__main__":
    # python -m eqbtst.index_mgmt capture   — scheduled at 09:05 on weekdays
    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        # Run under pythonw.exe (no console window). Under cmd.exe the scheduled job opened a
        # black window in the user's session, and closing it killed the capture with
        # STATUS_CONTROL_C_EXIT — 15, 16 and 19 Oct. pythonw has no stdout, so the log is
        # written here directly instead of by a shell redirect.
        if sys.stdout is None or os.environ.get("EQBTST_CAPTURE_LOG"):
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            _f = open(DATA_DIR / "capture.log", "a", encoding="utf-8", buffering=1)
            sys.stdout = sys.stderr = _f
        capture()
        raise SystemExit(0)
    # python -m eqbtst.index_mgmt cadence [minutes]
    # Measures NSE's real refresh interval and saves it; the board adopts it by itself.
    if len(sys.argv) > 1 and sys.argv[1] == "cadence":
        mins = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
        print(f"[{_now():%d %b %H:%M:%S}] cadence run")
        try:
            status = fetch_cash("NIFTY", NseClient())["status"]
        except NseError as exc:
            raise SystemExit(f"  NSE not reachable: {exc}")
        if status != "Open":
            # nothing changes when the market is shut; a run then would "measure" nothing
            # and must not overwrite a real measurement
            raise SystemExit(f"  market is {status or 'not open'} - nothing to measure, "
                             "nothing saved")
        print(f"  polling NSE every 5s for {mins:g} minutes (Nifty 50 prices, RELIANCE OI)")
        res = measure_cadence(minutes=mins)
        for name, r in res.items():
            print(f"  {name:24s} changed {r['changes']:3d} times"
                  + (f" | every {r['median_s']:.0f}s median ({r['min_s']:.0f}-{r['max_s']:.0f}s)"
                     if r["median_s"] is not None else " | too few changes to time"))
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "cadence.json").write_text(json.dumps(
            {"measured": f"{_now():%d %b %Y %H:%M}", "minutes": mins, "result": res},
            indent=1), encoding="utf-8")
        px, oi = tuned_intervals(res)
        print(f"  saved. board will poll prices every {px or CASH_EVERY:.0f}s and sweep OI "
              f"every {oi or DERIV_EVERY:.0f}s"
              + ("" if px and oi else " (a feed changed too rarely to time: default kept)"))
