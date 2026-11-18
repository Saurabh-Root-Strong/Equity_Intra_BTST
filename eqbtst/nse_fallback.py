"""NSE-website fallback for the Fyers Data API, used once the broker's DAILY cap is spent.

WHY THIS EXISTS. Fyers' Standard plan allows 5,000 Data calls per day per LOGIN, and the
login is shared with Tradebot's capturer (~4,237/day on its own). One structure scan of this
board costs ~270 /history calls, so the cap can be gone by mid-afternoon -- after which every
quote and candle comes back refused (HTTP 429, code -360) until the daily reset. NSE's own
website serves the same session's prices with no per-day quota, so this module rebuilds what
the board needs from there.

WHAT NSE CAN AND CANNOT REPLACE -- measured 2026-11-17, not assumed:

  quotes    FULL. `equity-stock-indices?index=NIFTY 500` returns 501 rows in ONE call with
            lastPrice / previousClose / open / dayHigh / dayLow / totalTradedVolume. It
            covers 255 of the board's 259 names; the other 4 come from `getSymbolData`
            (one call each). Indices come from `allIndices` (one call).
  1D bars   FULL. The EOD archive holds every closed session; today's forming daily bar is
            exactly the quote's open/high/low/last/volume. (Built in live.py, which owns the
            archive read.)
  intraday  PARTIAL. NSE publishes only TODAY's series, one PRICE POINT PER MINUTE, no
            volume (`getSymbolChartData ... days=1D`; `5D` is a 404). So:
              - prior days come from a local cache of the broker's own 15m candles, written
                every time a Fyers fetch succeeds (cache_put below);
              - today's bars are built from the minute points: open/close exact to the
                minute, high/low are the extremes of minute SAMPLES (so slightly inside the
                true range), volume is ZERO -- unknown, not "no trades".
            Session VWAP is therefore NOT derivable from these bars; `symbol_vwap` reads NSE's
            own `averagePrice` (the true session VWAP) instead.

TIMESTAMPS. NSE's chart epochs are IST wall-clock written as if it were UTC (verified: read
naively the normal-market points span 09:15:00-15:59:56; read as UTC->IST they would start at
14:45). They are therefore converted with NO timezone shift.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from .index_mgmt import NseClient, NseError

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "candle_cache" / "15m"
CACHE_KEEP_DAYS = 90            # > the 60-day window the structure scan reads

# Fyers index symbols -> NSE index names. Same three the header strip shows, plus the bare
# form fetch_intraday is handed by _index_intraday_5m ("NIFTY50").
INDEX_NAMES = {
    "NSE:NIFTY50-INDEX": "NIFTY 50",
    "NSE:NIFTYBANK-INDEX": "NIFTY BANK",
    "NSE:FINNIFTY-INDEX": "NIFTY FIN SERVICE",
}
_BARE_INDEX = {"NIFTY50": "NIFTY 50", "NIFTYBANK": "NIFTY BANK", "FINNIFTY": "NIFTY FIN SERVICE"}

_SESSION_OPEN, _SESSION_CLOSE = dt.time(9, 15), dt.time(15, 30)

_client_lock = threading.Lock()
_CLIENT: list = []


def _client():
    """One primed NSE session for the whole process (cookies are per session)."""
    with _client_lock:
        if not _CLIENT:
            _CLIENT.append(NseClient(min_gap=0.3))
        return _CLIENT[0]


def _num(v):
    try:
        f = float(str(v).replace(",", "")) if not isinstance(v, (int, float)) else float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _ttl() -> float:
    """NSE refreshes its pages every few seconds in session; off-hours nothing moves."""
    now = dt.datetime.now()
    live = now.weekday() < 5 and _SESSION_OPEN <= now.time() <= _SESSION_CLOSE
    return 15.0 if live else 300.0


_memo: dict = {}
_memo_lock = threading.Lock()


_key_locks: dict = {}


def _memo_get(key, ttl, fn):
    """Small TTL memo shared by the threads of one scan, so 270 workers asking for the bulk
    quote cost ONE NSE call, not 270. A failed fetch is not cached.

    PER-KEY LOCK: without it, the threads that all miss at the same moment each fetch (a
    thundering herd of identical calls to a site that blocks bursty clients)."""
    with _memo_lock:
        hit = _memo.get(key)
        if hit is not None and time.time() - hit[0] < ttl:
            return hit[1]
        klock = _key_locks.setdefault(key, threading.Lock())
    with klock:
        with _memo_lock:                       # another thread may have filled it meanwhile
            hit = _memo.get(key)
            if hit is not None and time.time() - hit[0] < ttl:
                return hit[1]
        val = fn()
        with _memo_lock:
            _memo[key] = (time.time(), val)
        return val


# ── quotes ──────────────────────────────────────────────────────────────────────────────
def _bulk_rows() -> dict:
    """{symbol: NSE row} for NIFTY 500 (+ SECURITIES IN F&O when NSE serves it -- that list
    is empty outside the session)."""
    def _fetch():
        out: dict = {}
        for name in ("NIFTY 500", "SECURITIES IN F&O"):
            try:
                j = _client().get("/api/equity-stock-indices?index=" + quote(name))
            except NseError:
                continue
            for r in (j.get("data") or []) if isinstance(j, dict) else []:
                s = str(r.get("symbol") or "")
                if s and r.get("priority") != 1 and s not in out:
                    out[s] = r
        if not out:
            raise NseError("NSE bulk quote returned no rows")
        return out
    return _memo_get("bulk", _ttl(), _fetch)


def _index_rows() -> dict:
    def _fetch():
        j = _client().get("/api/allIndices")
        rows = {str(r.get("index") or ""): r for r in (j.get("data") or [])}
        if not rows:
            raise NseError("allIndices returned no rows")
        return rows
    return _memo_get("indices", _ttl(), _fetch)


def _symbol_data(sym: str) -> dict:
    """NSE's single-name quote (metaData: open/dayHigh/dayLow/previousClose/averagePrice/...)."""
    def _fetch():
        j = _client().get("/api/NextApi/apiClient/GetQuoteApi?functionName=getSymbolData"
                          f"&marketType=N&series=EQ&symbol={quote(sym)}")
        eq = (j.get("equityResponse") or [{}])[0] if isinstance(j, dict) else {}
        md = dict(eq.get("metaData") or {})
        ti = eq.get("tradeInfo") or {}
        ob = eq.get("orderBook") or {}
        md.setdefault("lastPrice", ob.get("lastPrice"))
        for k in ("totalTradedVolume", "totalTradedValue"):
            if k in ti:
                md.setdefault(k, ti[k])
        if not md:
            raise NseError(f"getSymbolData: nothing for {sym}")
        return md
    return _memo_get(("sym", sym), _ttl(), _fetch)


def _as_fyers(lp, prev, o, h, l, vol, upd=None) -> dict:
    """An NSE row in the Fyers /quotes `v` shape every caller already reads. `_upd` is NSE's
    lastUpdateTime ("YYYY-MM-DD HH:MM:SS"): on a holiday NSE still serves the last session's
    row, so a caller that needs "is this TODAY's print" must check it."""
    lp, prev = _num(lp), _num(prev)
    d = {"lp": lp, "prev_close_price": prev, "open_price": _num(o), "high_price": _num(h),
         "low_price": _num(l), "volume": _num(vol) or 0, "_src": "NSE",
         "_upd": str(upd or "")}
    if lp is not None and prev:
        d["ch"] = round(lp - prev, 2)
        d["chp"] = round((lp / prev - 1) * 100, 2)
    return d


def quotes(fy_syms: list[str]) -> dict:
    """Fyers-shaped quotes for `fy_syms` ("NSE:RELIANCE-EQ", "NSE:NIFTY50-INDEX", ...) from
    NSE. Names NSE cannot price are simply absent (the caller already counts gaps)."""
    out: dict = {}
    want_idx = [s for s in fy_syms if s in INDEX_NAMES]
    want_eq = [s for s in fy_syms if s not in INDEX_NAMES]
    if want_idx:
        try:
            ix = _index_rows()
            for s in want_idx:
                r = ix.get(INDEX_NAMES[s])
                if r:
                    out[s] = _as_fyers(r.get("last"), r.get("previousClose"), r.get("open"),
                                       r.get("high"), r.get("low"), 0)
        except NseError:
            pass
    if want_eq:
        try:
            bulk = _bulk_rows()
        except NseError:
            bulk = {}
        for s in want_eq:
            sym = s.replace("NSE:", "").replace("-EQ", "")
            r = bulk.get(sym)
            if r:
                out[s] = _as_fyers(r.get("lastPrice"), r.get("previousClose"), r.get("open"),
                                   r.get("dayHigh"), r.get("dayLow"), r.get("totalTradedVolume"),
                                   r.get("lastUpdateTime"))
                continue
            try:                                 # the few names outside NIFTY 500
                m = _symbol_data(sym)
                out[s] = _as_fyers(m.get("lastPrice"), m.get("previousClose"), m.get("open"),
                                   m.get("dayHigh"), m.get("dayLow"), m.get("totalTradedVolume"),
                                   m.get("lastUpdateTime"))
            except NseError:
                continue
    return out


def symbol_vwap(sym: str) -> float | None:
    """Today's session VWAP as NSE publishes it (`averagePrice`). None if unavailable."""
    try:
        v = _num(_symbol_data(sym).get("averagePrice"))
    except NseError:
        return None
    return v if v and v > 0 else None


def session_dates(sym: str = "RELIANCE") -> pd.DataFrame:
    """NSE's own trading calendar for ~the last month: one row per session (column ts, a
    date at 00:00), from the name's 1M daily chart. Independent of the EOD archive, which is
    what archive_staleness needs it for. Empty frame on failure."""
    def _fetch():
        j = _client().get("/api/NextApi/apiClient/GetQuoteApi?functionName="
                          f"getSymbolChartData&symbol={quote(sym)}EQN&days=1M")
        g = (j.get("grapthData") or []) if isinstance(j, dict) else []
        ts = pd.to_datetime([p[0] for p in g if isinstance(p, (list, tuple)) and p],
                            unit="ms").normalize()
        if len(ts) == 0:
            raise NseError("1M chart returned no sessions")
        return pd.DataFrame({"ts": sorted(set(ts))})
    try:
        return _memo_get(("cal", sym), 3600.0, _fetch)
    except Exception:                                           # noqa: BLE001
        return pd.DataFrame(columns=["ts"])


# ── today's minute series -> bars ─────────────────────────────────────────────────────────
def minute_points(sym: str) -> pd.DataFrame:
    """Today's (or the last session's) NORMAL-MARKET minute points: columns ts, px.
    Pre-open (PO), the closing call auction (IC) and the post-close session are dropped."""
    idx = INDEX_NAMES.get(sym) or _BARE_INDEX.get(sym)

    def _fetch():
        if idx:
            j = _client().get("/api/NextApi/apiClient/indexTrackerApi?functionName="
                              f"getIndexChart&index={quote(idx)}&flag=1D")
            g = ((j.get("data") or {}).get("grapthData") or []) if isinstance(j, dict) else []
        else:
            j = _client().get("/api/NextApi/apiClient/GetQuoteApi?functionName="
                              f"getSymbolChartData&symbol={quote(sym)}EQN&days=1D")
            g = (j.get("grapthData") or []) if isinstance(j, dict) else []
        return parse_points(g)
    return _memo_get(("min", sym), 60.0 if _ttl() < 60 else 300.0, _fetch)


def parse_points(g: list) -> pd.DataFrame:
    """[[epoch_ms, price, flag, ...], ...] -> DataFrame(ts, px), normal market only."""
    rows = [(p[0], p[1]) for p in g
            if isinstance(p, (list, tuple)) and len(p) >= 3 and p[2] == "NM"
            and _num(p[1]) and _num(p[1]) > 0]
    if not rows:
        return pd.DataFrame(columns=["ts", "px"])
    df = pd.DataFrame(rows, columns=["ms", "px"])
    df["ts"] = pd.to_datetime(df["ms"], unit="ms")          # IST wall clock -- no shift
    t = df["ts"].dt.time
    # THE CLOSING AUCTION. NSE's series stops trading points at ~15:15; 15:15-15:30 is the
    # closing call auction (flag IC), whose points are INDICATIVE prices, not trades -- one
    # sample swung 1374.8 -> 1333.5 -> 1360 inside 30s on RELIANCE. They are dropped. The
    # auction's result IS a trade -- the official close, printed as NM after 15:30 -- so once
    # it exists it is placed as the session's final point (15:29:59), and the last bar closes
    # where the day actually closed.
    after = df[t >= _SESSION_CLOSE]
    df = df[(t >= _SESSION_OPEN) & (t < _SESSION_CLOSE)]
    if not after.empty and not df.empty:
        day = df["ts"].dt.normalize().iloc[-1]
        df = pd.concat([df, pd.DataFrame({
            "ms": [0], "px": [float(after["px"].iloc[0])],
            "ts": [day + pd.Timedelta("15h29min59s")]})])
    return (df[["ts", "px"]].astype({"px": float}).drop_duplicates("ts")
            .sort_values("ts").reset_index(drop=True))


def bars_from_points(pts: pd.DataFrame, freq_min: int) -> pd.DataFrame:
    """Minute price points -> OHLC bars aligned to 09:15. volume = 0 (NSE does not publish it)."""
    if pts is None or pts.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    g = pts.set_index("ts")["px"].groupby(
        pd.Grouper(freq=f"{freq_min}min", origin="start_day", offset="9h15min"))
    b = g.agg(["first", "max", "min", "last"]).dropna().reset_index()
    b.columns = ["ts", "open", "high", "low", "close"]
    b["volume"] = 0.0
    return b


# ── 15m candle cache (prior days, from the broker) ────────────────────────────────────────
_cache_lock = threading.Lock()


def _cache_path(sym: str) -> Path:
    return CACHE_DIR / f"{sym.replace(':', '_').replace('/', '_')}.parquet"


def cache_put(sym: str, df: pd.DataFrame) -> None:
    """Merge a successful broker 15m fetch into the symbol's cache. Never raises."""
    if df is None or df.empty:
        return
    try:
        p = _cache_path(sym)
        with _cache_lock:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            old = pd.read_parquet(p) if p.exists() else None
            cols = ["ts", "open", "high", "low", "close", "volume"]
            new = df[cols].copy()
            if old is not None and not old.empty:
                # the fresh fetch wins on overlap: its last bar may have been forming before
                new = pd.concat([old[~old["ts"].isin(new["ts"])], new])
            cut = pd.Timestamp(dt.date.today() - dt.timedelta(days=CACHE_KEEP_DAYS))
            new = new[new["ts"] >= cut].sort_values("ts").reset_index(drop=True)
            tmp = p.with_suffix(".tmp")
            new.to_parquet(tmp, index=False)
            tmp.replace(p)
    except Exception:                                           # noqa: BLE001
        pass


def cache_get(sym: str) -> pd.DataFrame:
    p = _cache_path(sym)
    try:
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()
    except Exception:                                           # noqa: BLE001
        return pd.DataFrame()


def splice(cached: pd.DataFrame, today_bars: pd.DataFrame) -> pd.DataFrame:
    """Cached broker bars + NSE-built bars for the session(s) the cache does not finish.

    On the NSE series' day, cache bars are kept only BEFORE the cache's last bar: that last
    bar may have been fetched while still forming (a partial high/low/close), so the NSE build
    takes over from it. Every earlier cached bar keeps its real volume."""
    if today_bars is None or today_bars.empty:
        return cached if cached is not None else pd.DataFrame()
    if cached is None or cached.empty:
        return today_bars.reset_index(drop=True)
    day = today_bars["ts"].dt.date.max()
    same = cached[cached["ts"].dt.date == day]
    if same.empty:
        keep = cached
        nse = today_bars
    else:
        last = same["ts"].max()
        keep = cached[cached["ts"] < last]
        nse = today_bars[today_bars["ts"] >= last]
    return pd.concat([keep, nse]).sort_values("ts").reset_index(drop=True)
