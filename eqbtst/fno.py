"""
fno.py — near/next-month FUTURES and OPTIONS positioning per stock, read onto this board.

WHY THIS EXISTS
    The board reads the CASH tape (price, delivery, structure). It says nothing about what the
    derivatives book is doing in the same name. Daily_Cash_Market already computes that per
    symbol from the NSE F&O bhavcopy, and it is the same read its sector pages show, so the two
    screens agree by construction.

WHY IT CALLS DCM INSTEAD OF PORTING IT
    sector_tilt.py is a hand-ported copy of a DCM engine, and upstream has already re-specified
    and then REVERTED it — twice — leaving this board publishing different badges than the DCM
    page for the same sector, silently, until someone eyeballed it. That cost two re-syncs and
    two dense parity walks.
    This module does not repeat that mistake. `get_fno_expiry_breakdown_by_symbol()` returns the
    WHOLE universe in one call in ~0.5s, which is cheaper than the port would be, so there is no
    performance argument for copying it. Calling upstream directly makes drift impossible.

    THE ONE THING THAT MUST NOT BE FORGOTTEN: DCM's ConnectionManager opens DuckDB READ-WRITE
    unless CLOUD_MODE is set, and DuckDB is many-readers-OR-one-writer. Importing DCM without
    that flag takes the EXCLUSIVE lock on the archive — locking out DCM's own dashboard, its
    nightly sync, and this board's own readers. CLOUD_MODE is set BEFORE the import, every time.

WHAT THE LABELS MEAN
    FUTURES — the classic OI x price matrix, on that expiry's own contract:
        🟢 LB  Long Buildup     price up,   OI up    → new longs
        🔴 SB  Short Buildup    price down, OI up    → new shorts
        🔵 SC  Short Covering   price up,   OI down  → shorts buying back
        🟠 LU  Long Unwinding   price down, OI down  → longs leaving
        ⟳ rolling — within 3 sessions of the monthly roll, where OI change is mechanical
                     (everyone moves to the next contract) and means nothing directional.
    OPTIONS — OI x PREMIUM per leg, then combined:
        C.Buying / P.Buying    OI up,   premium up   → demand for that leg
        C.Writing / P.Writing  OI up,   premium down → supply (someone is selling it)
        C.SC / P.SC            OI down, premium up   → writers buying back
        C.LE / P.LE            OI down, premium down → buyers giving up
    Puts read INVERTED against calls: writing or closing a put is bullish, buying one is bearish.
    🔥 Bull C.Buy+P.Wrt = both legs bullish · ❄️ Bear C.Wrt+P.Buy = both bearish ·
    📊 Range C+P.Wrt = both legs WRITTEN (short volatility) · ⚡ Vol Bet C+P.Buy = both bought.
    PCR is appended as context, not as the call.

WHAT IT IS NOT — READ THIS BEFORE YOU TRADE OFF THESE COLUMNS
    1. NOT MEASURED IN THIS BOOK, AND THE NEAREST THINGS THAT WERE ARE NULL. CE/PE OI crossover
       has IC ~ 0 here (mildly contrarian if anything); EOD OI walls did not bound next-day range
       better than an ATR band; max pain did not pin (49% vs a 50% coin). So these are CONTEXT,
       exactly like the sector tilt — nothing in the engine reads them and no filter uses them.
    2. THEY ARE EOD, NOT LIVE. The F&O bhavcopy publishes after the close, so during a session
       these columns describe YESTERDAY's positioning next to a live price. `meta["stale_days"]`
       says how far behind, and the board prints it rather than letting you assume it is today.
    3. NEXT-MONTH OI CHANGE IS NOISY EARLY IN THE CYCLE. A contract with a small base can print
       +143% off a handful of lots. Read the SIGN and the label, not the magnitude.
    4. ONLY 208 OF THE ~268 NAMES ON THIS BOARD HAVE F&O AT ALL. SEBI's tightened eligibility
       phased many out (ACC's last futures bar was 2025-07-31, BATAINDIA's 2025-02-27). A "—"
       is a correct answer, not a gap in the data.
"""
from __future__ import annotations

import datetime as dt
import functools
import os
import sys

import pandas as pd

from eqbtst import config

# The four columns this board shows. DCM computes a `far` month too; it is deliberately not
# carried — far-month stock contracts are thin enough that the label is mostly noise.
COLS = ["Fut Near", "Fut Next", "Opt Near", "Opt Next"]
_SRC = {"Fut Near": "near_fut_label", "Fut Next": "next_fut_label",
        "Opt Near": "near_opt_label", "Opt Next": "next_opt_label"}

# NSE serves ~1 year of date-addressable UDiFF F&O bhavcopy and DCM retains a rolling window;
# before this there is simply nothing to read, which is a different answer from "no F&O".
_RETENTION_HINT = "F&O bhavcopy retention — the archive does not go back this far"

_EMPTY = pd.DataFrame(columns=COLS)


def _dcm():
    """Import DCM's F&O engine READ-ONLY, or None if it is not reachable.

    CLOUD_MODE must be set before the import: see the module docstring. setdefault, not
    assignment, so a caller that has deliberately chosen a mode keeps it.
    """
    os.environ.setdefault("CLOUD_MODE", "true")
    root = config.DCM_DUCKDB.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from src.analytics.fno_stocks import get_fno_expiry_breakdown_by_symbol
        return get_fno_expiry_breakdown_by_symbol
    except Exception:                                        # noqa: BLE001
        return None


def _fno_dates() -> list:
    """Every date the F&O bhavcopy actually has, newest first. One archive read, then cached."""
    from eqbtst import data
    try:
        with data._connect() as c:
            return [r[0] for r in c.execute(
                "SELECT DISTINCT trade_date FROM fno_bhavcopy ORDER BY trade_date DESC"
            ).fetchall()]
    except Exception:                                        # noqa: BLE001
        return []


@functools.lru_cache(maxsize=8)
def _breakdown(day: dt.date) -> tuple:
    fn = _dcm()
    if fn is None:
        return _EMPTY.copy(), {"date_used": None, "stale_days": None, "n": 0,
                               "reason": "DCM F&O engine not importable"}
    dates = _fno_dates()
    if not dates:
        return _EMPTY.copy(), {"date_used": None, "stale_days": None, "n": 0,
                               "reason": "no F&O bhavcopy in the archive"}
    # AS-OF, never after: the board can be pointed at a past close (or Replay), and reading the
    # newest F&O row there would be a lookahead — tomorrow's positioning beside today's price.
    usable = [d for d in dates if d <= day]
    if not usable:
        return _EMPTY.copy(), {"date_used": None, "stale_days": None, "n": 0,
                               "reason": _RETENTION_HINT}
    used = usable[0]
    try:
        df = fn(used)
    except Exception as exc:                                 # noqa: BLE001
        return _EMPTY.copy(), {"date_used": None, "stale_days": None, "n": 0,
                               "reason": f"F&O read failed: {type(exc).__name__}"}
    if df is None or df.empty:
        return _EMPTY.copy(), {"date_used": used, "stale_days": (day - used).days, "n": 0,
                               "reason": "F&O engine returned nothing for that date"}
    out = pd.DataFrame({k: df[v] for k, v in _SRC.items() if v in df.columns})
    out.index = df["symbol"].astype(str)
    for c in COLS:                                           # keep the shape stable
        if c not in out.columns:
            out[c] = "—"
    return out[COLS], {"date_used": used, "stale_days": (day - used).days,
                       "n": int(len(out)), "reason": ""}


def positioning(as_of) -> tuple:
    """(frame indexed by symbol with the 4 label columns, meta). Never raises."""
    try:
        ts = pd.Timestamp(as_of)
        # pd.Timestamp(None) is NaT, and NaT.date() returns NaT rather than raising -- so a
        # bad as-of sails past the try and dies later comparing NaT to a real date. Check it.
        if ts is pd.NaT or pd.isna(ts):
            raise ValueError("as-of is NaT")
        day = ts.date()
        if not isinstance(day, dt.date):
            raise TypeError("as-of did not resolve to a date")
    except Exception:                                        # noqa: BLE001
        return _EMPTY.copy(), {"date_used": None, "stale_days": None, "n": 0,
                               "reason": "bad as-of"}
    return _breakdown(day)


def annotate(df: pd.DataFrame, as_of) -> pd.DataFrame:
    """Add the 4 F&O columns to a board frame, matched on `symbol`.

    A name with no F&O gets "—". That is the honest answer for ~60 of this board's names, whose
    contracts NSE has retired — not a missing-data hole to be hidden.
    """
    if df is None or df.empty or "symbol" not in df.columns:
        return df
    pos, _ = positioning(as_of)
    out = df.copy()
    if pos.empty:
        for c in COLS:
            out[c] = "—"
        return out
    sym = out["symbol"].astype(str)
    for c in COLS:
        out[c] = sym.map(pos[c]).fillna("—")
    return out


def stale_note(meta: dict) -> str:
    """One line for the board: what date these columns actually describe."""
    if not meta or meta.get("date_used") is None:
        return f"⚠ F&O positioning unavailable — {meta.get('reason') or 'no data'}."
    d, s = meta["date_used"], meta.get("stale_days") or 0
    base = f"F&O positioning as of the **{d:%d %b}** close ({meta.get('n', 0)} names)"
    if s <= 0:
        return base + " — same session as the board."
    return (base + f" — **{s} day(s) behind** the board's as-of date. The bhavcopy publishes "
            "after the close, so during a session these describe yesterday's book.")


def clear_cache() -> None:
    _breakdown.cache_clear()


HELP_FUT = (
    "**FUTURES positioning** on that expiry's own contract — open interest against price, from "
    "the NSE F&O bhavcopy (EOD, via Daily_Cash_Market, same read as its sector pages).\n\n"
    "🟢 **LB** long buildup — price ↑, OI ↑ (new longs)\n"
    "🔴 **SB** short buildup — price ↓, OI ↑ (new shorts)\n"
    "🔵 **SC** short covering — price ↑, OI ↓ (shorts buying back)\n"
    "🟠 **LU** long unwinding — price ↓, OI ↓ (longs leaving)\n"
    "⚪ flat · **⟳ rolling** = within 3 sessions of the monthly roll, where the OI change is "
    "mechanical (everyone shifts to the next contract) and means nothing directional.\n\n"
    "The % is the OI change vs the previous session, same contract.\n\n"
    "⚠ **NEXT-month % is noisy early in the cycle** — a small OI base prints huge percentages "
    "off a few lots. Read the sign and the label, not the size.\n\n"
    "⚠ **EOD, not live.** The bhavcopy publishes after the close, so during a session this is "
    "YESTERDAY's book beside a live price. The caption above the table says which date.\n\n"
    "⚠ **CONTEXT ONLY — nothing in the engine reads it.** The nearest things measured in this "
    "project came back null: CE/PE OI crossover IC ≈ 0, EOD OI walls did not bound next-day "
    "range better than an ATR band, max pain did not pin (49% vs a 50% coin). "
    "**—** = the name has no F&O; SEBI's tightened eligibility retired ~60 of this board's names."
)

HELP_OPT = (
    "**OPTIONS positioning** for that expiry — open interest against PREMIUM on each leg, then "
    "combined. From the NSE F&O bhavcopy (EOD, via Daily_Cash_Market).\n\n"
    "Per leg: **Buying** = OI ↑ premium ↑ (demand) · **Writing** = OI ↑ premium ↓ (supply) · "
    "**SC** = OI ↓ premium ↑ (writers buying back) · **LE** = OI ↓ premium ↓ (buyers giving up).\n\n"
    "Puts read **inverted** against calls — writing or closing a put is bullish, buying one is "
    "bearish. Combined:\n"
    "🔥 **Bull C.Buy+P.Wrt** — both legs bullish\n"
    "❄️ **Bear C.Wrt+P.Buy** — both legs bearish\n"
    "📊 **Range C+P.Wrt** — both legs WRITTEN: short volatility, the book expects a range\n"
    "⚡ **Vol Bet C+P.Buy** — both bought: someone is paying for a move, direction unstated\n"
    "Single-sided reads show the informative leg, coloured by its true sentiment.\n\n"
    "**PCR** (put OI ÷ call OI) is appended as context, NOT the call. High PCR is conventionally "
    "read as bullish, but it is also just where the open interest happens to sit.\n\n"
    "⚠ **EOD, not live** — see the date in the caption above the table.\n\n"
    "⚠ Unlike the futures column, the options read has **no post-roll guard**: in the days around "
    "a monthly expiry the near-month legs are being closed and rolled, so the label reflects "
    "housekeeping rather than a view.\n\n"
    "⚠ **CONTEXT ONLY — nothing in the engine reads it.** "
    "**—** = the name has no F&O."
)
