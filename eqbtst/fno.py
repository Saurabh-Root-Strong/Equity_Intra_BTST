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
    CYCLE columns — the same reads taken over the MONTHLY EXPIRY CYCLE rather than one session:
        "🟢 LB +13% | +2, +3, +2, +6" — cumulative OI change since the last monthly roll on
        the SAME contract, then the last four DAILY steps, so a build still running is visually
        distinct from one that stalled a week ago. "⚠" means TODAY moved AGAINST the build.
        The CODE is PEER-RELATIVE (ranked against the other ~207 names); the number is raw.
    THE FUTURES COLUMNS CARRY THE CYCLE READ, THE OPTIONS COLUMNS THE 1-DAY READ. Not a style
    choice — see the block above COLS for the measurement behind each. Upstream also publishes
    an options CYCLE label ("🟢 Bull | CE Buy3 PE Wrt5 /6d", the cycle's daily verdicts
    counted); it is deliberately NOT carried, for the coverage reason recorded there.

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
    4. A BLANK IS UPSTREAM WITHHOLDING, NOT A GAP. Rather than print a large percentage off a
       handful of lots, upstream refuses to label a contract that has not filled up. It is
       also the reason the options CYCLE read is not carried at all (100% blank on cycle
       day 1, 66.8% on day 6).
       THIS IS NOT RARE, AND AN EARLIER VERSION OF THIS NOTE UNDERSTATED IT BADLY by
       quoting the MID-CYCLE rate (0.0% near, 0.5% next) as though it held all month. Three
       guards blank these columns and they stack around the monthly roll:
         * cycle day 0        the cumulative has no anchor yet -- 100% of names, all expiries
         * pre-expiry (3d)    the near contract is being closed out market-wide -- near only
         * fill-up guard      the next contract is still ramping -- next only, and it ratchets
                              through the cycle (measured Aug-2026: 0% -> 13% -> 33% -> 100%)
       Net: FUT NEAR IS BLANK FOR 5 CONSECUTIVE SESSIONS PER CYCLE (~25% of all sessions) and
       FUT NEXT FOR ROUGHLY 8-10 (~45%). Measured across the Jul and Aug 2026 cycles, both of
       which show the identical shape. The columns recover in full on cycle day 1.
       A "-" here is the engine declining to make a claim, and around the roll that is the
       CORRECT state -- see note 7.
    5. STILL NO MEASURED FORWARD INFORMATION. Upstream tested the cycle read at every horizon
       and found under 0.08pp. It DESCRIBES positioning; it does not predict.
    7. THE BLANK IS THE SAFE STATE; THE FIRST POPULATED DAY IS THE ONE TO DISTRUST. On
       2026-08-27 -- cycle day 1, the session these columns come back -- the archive held a
       corrupt F&O ingest: futures OI 98.6% below the prior session, traded contracts 207x
       above it, prices correct so nothing looked wrong. Every upstream guard passed it,
       because the cycle read is PEER-RELATIVE: when a whole session is wrong every name is
       wrong by the same amount, so nothing is an outlier. It rendered at stale_days=0,
       "same session as the board". Upstream now refuses such a session at the write
       (FUTSTK OI outside [0.33, 3.0]x the previous session), but the lesson stands: these
       columns fail LOUDLY as a dash and QUIETLY as a number.
    8. ONLY 208 OF THE ~268 NAMES ON THIS BOARD HAVE F&O AT ALL. SEBI's tightened eligibility
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

# DCM computes a `far` month too; deliberately not carried — far-month stock contracts are thin
# enough that the label is mostly noise, and upstream blanks it behind a volume gate anyway.
#
# FOUR columns, and the choice of WHICH read each carries is measured, not stylistic.
#
# FUTURES -> the CYCLE read. Upstream also publishes a 1-day futures label, and it is fully
# REDUNDANT: measured on 208/208 names, both expiries, the 1-day % is exactly the leading
# element of the cycle cell's own step sequence ("+2" in "LB +13% | +2, +3, +2, +6"). Its only
# unique content is its own CODE, computed from an absolute OI x price matrix while the cycle
# code is peer-relative -- so the two share the LB/SB/SC/LU vocabulary and the colour scheme
# but disagree on 31-35% of names where both are directional (LU beside SC, orange beside
# blue, same contract). Showing both doubled the width and invited a comparison that is not
# valid. The cycle read also flips direction 30.7% of the time against the daily read's 70.7%.
#
# OPTIONS -> the 1-DAY read. Here the cycle label is NOT a superset: it counts how the cycle's
# sessions voted ("Bal | CE Wrt1 PE Wrt2 /6d") and carries neither today's leg verdict nor the
# PCR level. It is also withheld early in every cycle by upstream's minimum-sessions guard --
# measured 100% blank on cycle day 1, 66.8% on day 6, 29.5% by day 12. A column that says
# nothing for the first four sessions of every month is worse than one that always reads.
COLS = ["Fut Near", "Fut Next", "Opt Near", "Opt Next"]
_SRC = {"Fut Near": "near_trend_label", "Fut Next": "next_trend_label",
        "Opt Near": "near_opt_label",   "Opt Next": "next_opt_label"}

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


HELP_FUT_CYC = (
    "**FUTURES positioning across the whole EXPIRY CYCLE** — not one session.\n\n"
    "`🟢 LB +13% | +2, +3, +2, +6`\n\n"
    "• **+13%** = cumulative OI change since the last monthly roll, on the SAME contract.\n"
    "• **| +2, +3, +2, +6** = the last four DAILY steps, so a build still running reads "
    "differently from one that stalled a week ago.\n"
    "• **⚠** = TODAY moved AGAINST the cycle build.\n"
    "• The code is **peer-relative** — ranked against the other ~207 F&O names, not an absolute "
    "cutoff. The number beside it is raw.\n\n"
    "🟢 **LB** long buildup (price ↑ OI ↑, new longs) · 🔴 **SB** short buildup "
    "(price ↓ OI ↑, new shorts) · 🔵 **SC** short covering (price ↑ OI ↓, shorts buying "
    "back) · 🟠 **LU** long unwinding (price ↓ OI ↓, longs leaving) · ⚪ flat.\n"
    "**⟳ rolling** = within 3 sessions of the monthly roll, where the OI change is mechanical "
    "(everyone shifts contract) and means nothing directional.\n\n"
    "**WHY IT SITS BESIDE THE 1-DAY COLUMN:** upstream measured the daily read flipping "
    "direction **70.7%** of the time against **30.7%** for this one. Alone the daily column is "
    "close to noise; together, this is the backdrop and the daily is what moved today.\n\n"
    "⚠ Blank early in a cycle is DELIBERATE — a contract that has not filled up would print a "
    "huge percentage off a few lots, so upstream withholds it.\n\n"
    "⚠ **CONTEXT ONLY.** Tested at every horizon upstream and found under 0.08pp of forward "
    "information — a description of positioning, not a signal. **—** = no F&O in this name."
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
