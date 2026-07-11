"""
events.py — earnings/event guard. The #1 risk to the BTST edge: a name that reports
results DURING the overnight hold gaps on the earnings, not the accumulation — that is
noise, not the signal, and it can be a large adverse gap. Such names must be EXCLUDED.

Source: NSE corporate event-calendar (Financial Results / board meetings). Fetched
with a primed session, cached to a daily CSV. Degrades gracefully: if the calendar
can't be fetched, the guard reports UNAVAILABLE (the caller warns "verify manually")
rather than silently passing an earnings name.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parent.parent / "data" / "events" / "nse_events.csv"
_URL = "https://www.nseindia.com/api/event-calendar"
_HDRS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-event-calendar",
}


def fetch_nse_events() -> pd.DataFrame:
    """Pull the live NSE event calendar. Returns DataFrame[symbol, date, purpose] or
    empty on failure. Keeps only results/board-meeting events (the gap-risk ones)."""
    try:
        s = requests.Session(); s.headers.update(_HDRS)
        s.get("https://www.nseindia.com", timeout=8)              # prime cookies
        r = s.get(_URL, timeout=10)
        rows = r.json()
    except Exception:
        return pd.DataFrame(columns=["symbol", "date", "purpose"])
    out = []
    for it in rows:
        purpose = str(it.get("purpose", ""))
        desc = str(it.get("bm_desc", ""))
        if "result" not in (purpose + desc).lower() and "board meeting" not in purpose.lower():
            continue
        try:
            d = dt.datetime.strptime(it["date"], "%d-%b-%Y").date()
        except Exception:
            continue
        out.append({"symbol": it.get("symbol"), "date": d, "purpose": purpose})
    return pd.DataFrame(out)


def refresh_cache() -> int:
    """Fetch + write the cache. Returns row count (0 = fetch failed)."""
    df = fetch_nse_events()
    if df.empty:
        return 0
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return len(df)


def load_events(max_age_hours: int = 12) -> pd.DataFrame:
    """Cached events; refetch if the cache is missing or older than max_age_hours."""
    fresh = (CACHE.exists()
             and (dt.datetime.now().timestamp() - CACHE.stat().st_mtime) < max_age_hours * 3600)
    if not fresh:
        refresh_cache()
    if not CACHE.exists():
        return pd.DataFrame(columns=["symbol", "date", "purpose"])
    df = pd.read_csv(CACHE)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def guard_status() -> dict:
    df = load_events()
    return {"available": not df.empty, "n": len(df),
            "asof": (dt.datetime.fromtimestamp(CACHE.stat().st_mtime).strftime("%d-%b %H:%M")
                     if CACHE.exists() else "never")}


def upcoming(asof: dt.date, horizon_days: int = 3, events: pd.DataFrame | None = None) -> set:
    """Symbols with a results/board event in (asof, asof+horizon] — i.e. reporting
    during the overnight-to-next-day BTST hold. These must be excluded from BUY."""
    ev = load_events() if events is None else events
    if ev.empty:
        return set()
    lo, hi = asof, asof + dt.timedelta(days=horizon_days)
    m = ev[(ev["date"] > lo) & (ev["date"] <= hi)]
    return set(m["symbol"].dropna().astype(str))


def event_date(symbol: str, asof: dt.date, horizon_days: int = 3) -> dt.date | None:
    ev = load_events()
    if ev.empty:
        return None
    lo, hi = asof, asof + dt.timedelta(days=horizon_days)
    m = ev[(ev["symbol"] == symbol) & (ev["date"] > lo) & (ev["date"] <= hi)]
    return m["date"].min() if not m.empty else None
