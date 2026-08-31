"""
dashboard.py — the equity BTST decision board (Streamlit).

    streamlit run eqbtst/dashboard.py

HONEST SCOPE: this reads the EOD engine (Daily_Cash_Market archive, updated
post-market). It is a "as of last close" board, NOT a live intraday tape — live
Fyers ticks are Phase 2. The BTST signal is a CLOSE decision (act ~15:15-15:30),
so an EOD board is the correct surface for it. Long-only (short overnight is proven
dead). BUY = act tonight; HOLD = open position; SELL = exit an open long next
morning; AVOID = footprint seen but rejected by liquidity or the regime gate.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from eqbtst import (arb, config, data, fno, ledger, live, mtf, scalp, screen,
                    sector_tilt)

st.set_page_config(page_title="Equity BTST Board", layout="wide", page_icon="📊")

# ── make long column tooltips SCROLLABLE ──────────────────────────────────────────────
# Streamlit's help popup has no scrollbar: text longer than the popup is simply CLIPPED, with
# no indication that anything is missing. Several columns here carry the measured evidence for
# what they show (that is the point — a number with no provenance invites over-trust), so the
# tooltips are long and were being cut mid-sentence. Cap the height and let it scroll.
# Belt-and-braces: this targets a Streamlit internal test-id, so if a future version renames it
# the CSS silently stops applying — the tooltips are ALSO kept short enough to mostly fit, and
# the full text lives in an on-page expander that never depends on this working.
# The first attempt targeted only [data-testid="stTooltipContent"], which is Streamlit's WIDGET
# tooltip (the ⓘ beside a selectbox). A DATAFRAME column-header tooltip is a different element,
# so the rule never applied and long help text stayed clipped with no scrollbar. Cast wider and
# hit the generic ARIA/baseweb tooltip containers too — whichever one Streamlit is using, one of
# these matches. Still belt-and-braces: the help strings are also kept short enough to fit.
st.markdown("""<style>
div[data-testid="stTooltipContent"],
div[data-testid="stTooltipContent"] > div,
div[data-baseweb="tooltip"],
div[data-baseweb="popover"] div[role="tooltip"],
div[role="tooltip"]{
  max-height:24rem !important; overflow-y:auto !important;
  max-width:46rem !important; white-space:pre-wrap;
}
div[role="tooltip"] p{margin-bottom:.4rem;}
</style>""", unsafe_allow_html=True)


def _cols(df, cols):
    """Only the columns that actually exist — tolerates a stale cache / older board
    that predates a new column (e.g. 'entered'), so a missing column never crashes."""
    return [c for c in cols if c in df.columns]


# ── structure labels: the engine returns terse ENUMS (BREAKOUT_UP…); humans read glyph+word.
# The ENUM stays the underlying value everywhere (filter compares s{tf} == enum, columns store
# enum) — this map is DISPLAY ONLY: format_func on the dropdowns, and a cell rewrite in _fmt.
_STRUCT_LABEL = {
    "BREAKOUT_UP":   "🚀 Breakout ↑",
    "TREND_UP":      "📈 Uptrend",
    "CONSOLIDATION": "🌀 Coiling",
    "RANGE":         "↔️ Range",
    "TREND_DOWN":    "📉 Downtrend",
    "BREAKOUT_DOWN": "💥 Breakdown ↓",
    "n/a":           "—",
    "Any":           "Any (no filter)",
}
_STRUCT_COLS = ("s15m", "s1h", "s2h", "s4h", "s1D", "s1W", "structure")


def _struct_label(v):
    """Enum → friendly label for display; unknown/missing values pass through unchanged."""
    return _STRUCT_LABEL.get(v, v)


def _tally(shown, total, what="names", extra=""):
    """Footer under a table: how many rows you are LOOKING at vs how many exist upstream.
    A table that is capped or filtered otherwise reads as 'that is all there is' — the
    denominator is what stops a 12-row view being mistaken for a 12-name universe."""
    pct = f" · {shown / total * 100:.0f}% of universe" if total else ""
    same = shown == total
    body = (f"**{shown}** {what}" if same else f"showing **{shown}** of **{total}** {what}{pct}")
    return st.caption("Σ  " + body + (f"  ·  {extra}" if extra else ""))


def _fmt(df):
    """Render-time tidy-up. A name can sit in the timeframe list (it passed the tf verdict)
    while NEVER having fired the BTST footprint — so entered / at / since% are genuinely
    absent, not broken. Python's None then printed as the literal string 'None', which reads
    like a bug rather than an answer. Show '—' (nothing fired) and leave the numerics blank."""
    d = df.copy()
    if "entered" in d.columns:
        d["entered"] = d["entered"].where(d["entered"].notna(), "—")
    # NULLABLE dtypes, not plain float. `pd.to_numeric` alone leaves numpy NaN, and this
    # Streamlit renders a NaN cell as the literal string "None" -- the exact thing the
    # em-dash above exists to prevent, just one column to the right. Reported from a
    # screenshot: `entered` read "—" while `at` and `since%` beside it read "None".
    # Float64/Int64 carry pd.NA, which crosses Arrow as a real null and renders BLANK,
    # while staying numeric so the column still sorts and formats.
    for c in ("at", "since%", "cvwap%", "rsCum%", "est_close", "vol×", "turn₹L", "wt%"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype("Float64")
    if "dn_age" in d.columns:
        d["dn_age"] = pd.to_numeric(d["dn_age"], errors="coerce").astype("Int64")
    for c in _STRUCT_COLS:                                 # terse enum -> glyph+word (display only)
        if c not in d.columns:
            continue
        bcol = "bnd" + c                                  # s15m -> bnds15m (live ±band %)
        if bcol in d.columns:
            # append the LIVE, per-name band ±% to COIL/RANGE cells only (breakout/trend have
            # no meaningful 'band'). Volatility-driven, so it auto-scales across timeframes.
            d[c] = [
                _struct_label(v) + (f"  ±{b:.1f}%"
                                    if (v in ("CONSOLIDATION", "RANGE")
                                        and isinstance(b, (int, float)) and pd.notna(b))
                                    else "")
                for v, b in zip(d[c], d[bcol])
            ]
        else:
            d[c] = d[c].map(_struct_label)
    if "setup" in d.columns:                               # tag -> icon + tag (display only)
        # SHOW THE DIRECTION ON THE TAG. "WITH-TREND CONTINUATION" alone is ambiguous — it is a
        # LONG in an uptrend and a SHORT in a downtrend (the tag is direction-agnostic; `dir`
        # decides the side). A user reading it on the SHORT tab could not see it meant a
        # DOWNTREND continuation. Same for COIL AT THE EXTREME / EXTENDED / PULLBACK. Append the
        # HTF trend arrow from `dir` (↑ up, ↓ down) so the tag reads unambiguously; NONE-dir
        # tags (squeeze, drift, range, trap) get no arrow. RANGE-TOP/FLOOR BREAK carry their
        # own direction inherently and the arrow just reinforces it.
        _dirs = d["dir"].tolist() if "dir" in d.columns else [None] * len(d)
        _arrow = {"UP": " ↑", "DOWN": " ↓"}
        d["setup"] = [
            (f"{mtf.TAG_ICON.get(v, '')} {v}{_arrow.get(dv, '')}".strip()
             if isinstance(v, str) else "—")
            for v, dv in zip(d["setup"], _dirs)]
    if "headroom" in d.columns:
        # CLEAR ROAD IS AN ANSWER, AND IT WAS RENDERING AS AN EMPTY CELL. The scan stores
        # +inf when there is NO multi-touch wall overhead, precisely so that "nothing above
        # you" stays distinguishable from "not computed" (see the inf comment in live._set).
        # A NumberColumn then printed inf as blank -- identical to a missing value -- so the
        # column's own tooltip promised a symbol the table could never show. Measured 51 of
        # 243 names on the BTST preset, i.e. 21% of the board, silently ambiguous. Rendered
        # as text so the distinction survives to the screen.
        _hv = pd.to_numeric(d["headroom"], errors="coerce")
        d["headroom"] = ["∞ clear" if np.isinf(v) else ("—" if pd.isna(v) else f"{v:.2f}")
                         for v in _hv]
    if "big_gap" in d.columns:                             # same inf treatment as headroom
        _bg = pd.to_numeric(d["big_gap"], errors="coerce")
        d["big_gap"] = ["∞" if np.isinf(v) else ("—" if pd.isna(v) else f"{v:.2f}") for v in _bg]
    return d

# ── hover tooltips: what each column is + WHY it matters ───────────────────────
LIVE_COLS = {
    "symbol": st.column_config.TextColumn("symbol", help="NSE F&O stock (cash, no theta).", pinned=True),
    "bar": st.column_config.TextColumn("bar", help="Is the CURRENT candle on your selected timeframe still forming? ⏳ forming = the bar has NOT closed yet, so clr/structure/RSI can still CHANGE — the signal can REPAINT and may look different at the bar's close (on 4h that's 13:15 or 15:30). ✓ closed = the bar is done; that read is final. A trigger fired mid-candle is provisional until the bar closes."),
    "est_close": st.column_config.NumberColumn("est_close", help="ESTIMATE OF THE OFFICIAL CLOSING PRICE — this is the number the decision is judged on, not the LTP. NSE does NOT close at the last traded price: the official close is the VOLUME-WEIGHTED AVERAGE of all trades in the final 30 minutes (15:00-15:30), and THAT is what the EOD archive stores and what the 8-year backtest gated on. Measured on 38,099 stock-days: the last-30-min VWAP tracks the official close to 1.6 bps, while the 15:30 last-traded price is off by 14.8 bps. So from 15:00 onward, clr / day% / cvwap% are recomputed on this estimate — judging them on the LTP would evaluate a DIFFERENT price than the one that was validated. Blank before 15:00 (no closing window yet, so the LTP stands and the board is a forecast).", format="%.2f"),
    "at": st.column_config.NumberColumn("at", help="The PRICE when the footprint fired (the 'entered' bar's close). Paired with since%, it tells you what the name has done since it triggered. NOTE: this is NOT your entry — the validated BTST entry is at the CLOSE (15:15-15:25), not at the trigger.", format="%.2f"),
    "since%": st.column_config.NumberColumn("since%", help="Move SINCE the footprint fired = (ltp / trigger-price - 1). POSITIVE = the name has HELD or EXTENDED since it triggered (the footprint is persisting — higher conviction). NEGATIVE = it has FADED since firing (the move is decaying; it may not hold into the close, and a name that fades back below VWAP loses the path-signature leg). IMPORTANT: this is NOT profit and NOT your P&L — you do not enter at the trigger. The BTST entry is near the CLOSE. Treat since% as a signal-QUALITY read (is it holding?), never as a return you captured.", format="%.2f"),
    "entered": st.column_config.TextColumn("entered", help="WHEN THE TRADE TRIGGERED — the wall-clock time (5-min resolution) the footprint FIRST fired today, INDEPENDENT of the timeframe you picked. If it fires at 12:30 while the 4h candle (09:15→13:15) is still forming, this reads 12:30 — not 09:15 or 13:15. The timeframe governs the structure/RSI/levels lens; the trigger is a clock event. (Qualification time, NOT the candle/scan timestamp.) Replay & the timeframe scans compute it CAUSALLY — the HH:MM the footprint first FORMED (up ≥1% AND closing the running session in the top of its range AND above session VWAP). The Live 5s snapshot has no intraday bars, so there it is FIRST-SEEN — the wall-clock time our scanner first saw the name qualify (accurate if the board ran from the open; later if you started the dashboard mid-session). Earlier + still holding = footprint persisted = higher conviction; just entered near the close = fresher/less proven.\n\n**WHAT ACTUALLY STAMPS IT.** Only names standing at **BTST-CARRY** or **FORMING** (long), or **SHORT**/**WEAK** (short). FORMING = **4 of the 5** validated legs; BTST-CARRY = **all 5** plus trailing delivery, liquidity and the close window. The five legs: strong close (clr), up on the day, volume pace, PERSISTENT relative strength (the cumulative read, not a one-day burst), and closed above session VWAP (the path signature — trended-and-held, not spiked-and-faded).\n\n**BEING UP IS NOT ENOUGH — IT IS 4 OF 5.** The biggest day% on the board can show a dash while a smaller mover stamps, because it is failing volume, RS-persistence or the VWAP path. That is the column working, not a glitch.\n\n**BEFORE 15:10 THIS COLUMN IS STRUCTURALLY SPARSE.** BTST-CARRY has a hard near-close gate, so every stamp on a morning board is FORMING — and the day-return and volume legs are the hardest to fill in the first hour. A 10:45 board showing two stamps against ten listed names is the EXPECTED shape, not a fault. The column fills through the session and only means what it says near the close.\n\nA DASH (—) means the validated footprint has NOT fired for this name today — it is in the timeframe list because it passed the (unvalidated) tf verdict, not because it formed the BTST footprint. Most often it is the VOLUME leg that fails: a name can be up 2.5%, closing strong and above VWAP on merely ordinary volume — price without participation is not accumulation. Those rejected names were measured over 8 years: 1,235 of them, worth −0.1bps. A coin flip. The dash is the signal protecting you from a chart pattern that pays nothing."),
    "time": st.column_config.TextColumn("time", help="The candle the signal is on, as open→close (e.g. 13:15-15:15 = the 2h candle spanning 13:15 to 15:15). IMPORTANT: an intraday signal only CONFIRMS at the candle's CLOSE — during a live session the current candle is still forming and the signal can repaint until it closes. Live snapshot = scan time; replay = last bar at/before your cut."),
    "sector": st.column_config.TextColumn("sector", help="Canonical sector — used for the concentration cap (≤2 names/sector, so many longs in one sector aren't one macro bet)."),
    "sector tilt": st.column_config.TextColumn("sector tilt", width="medium", help=sector_tilt.HELP),
    "ltp": st.column_config.NumberColumn("ltp", help="Last-traded price (Fyers) — LIVE, refreshed every 5s from one batch quote, on the live-snapshot board AND on the structure-scan tabs. In the TIMEFRAME tables a two-tier refresh runs: the price and everything cheap that hangs off it (day%, RS%, bar_clr, vs_vwap%, entry/stop/T1/T2, and the LONG/AVOID verdict itself) all re-derive on the live price every 5s. Only the CANDLE-derived columns (structure, RSI, tone) stay as-of the last scan — they need ~70 /history calls and only change when a BAR CLOSES anyway (on 4h, twice a day). Their age is stamped above the table; ↻ refresh to re-pull the candles.", format="%.2f"),
    "day%": st.column_config.NumberColumn("day%", help="Return vs previous close. The signal wants demand in control (≥ +1%).", format="%.2f"),
    "s15m": st.column_config.TextColumn("15m", help="Structure on 15-MINUTE bars (Kaufman efficiency over the last ~20 bars ≈ 5 hours). The fastest, noisiest frame — repaints until each 15m bar closes. Computed from one 15-min fetch (~20 days), NOT a separate API call per timeframe."),
    "s1h": st.column_config.TextColumn("1h", help="Structure on 1-HOUR bars (~20 bars ≈ 3 sessions), resampled locally from the same 15-min fetch. Includes today's forming bar → can repaint until it closes."),
    "s2h": st.column_config.TextColumn("2h", help="Structure on 2-HOUR bars (~20 bars ≈ 7 sessions), resampled from the 15-min fetch. Includes the forming bar → can repaint."),
    "s4h": st.column_config.TextColumn("4h", help="Structure on 4-HOUR bars (~20 bars ≈ 10 sessions; NSE's 6h15m session = 2 bars/day, 09:15–13:15 + a partial to 15:30). The big intraday frame. Includes the forming bar → can repaint until 13:15/15:30."),
    "s1D": st.column_config.TextColumn("1D", help="Structure on DAILY bars (last ~60 daily bars from the EOD archive). THROUGH THE LAST CLOSE — leak-free and it cannot repaint intraday, but it does not see today's bar yet. In the sister index project, Daily×Weekly BREAKOUT-from-tight-base was the one MTF pattern that survived validation — intraday MTF alignment has no such evidence and stays context."),
    "s1W": st.column_config.TextColumn("1W", help="Structure on WEEKLY bars (daily archive resampled W-FRI, ~20 weeks ≈ 5 months). The slowest frame — the regime this stock lives in. Through the last close; the current week's bar is still forming until Friday. Classic MTF read: trade in the direction of 1W/1D, time with the intraday frames — but remember the intraday lane itself is −5bps; this is tape-reading context, not a validated signal."),
    "clr": st.column_config.NumberColumn("clr", help="Close Location in Range = (close−low)/(high−low). ≥0.70 = closing strong, buyers held into the bar. Core of the accumulation footprint.", format="%.2f"),
    "character": st.column_config.TextColumn("character", help="Candle shape: marubozu_bull (strong body, best), hammer (demand rejected lows), shooting_star (supply capped highs, avoid), doji (indecision), strong/weak_close."),
    "body": st.column_config.NumberColumn("body", help="Signed body fraction of range. Positive & large = conviction green candle.", format="%.2f"),
    "vol×": st.column_config.NumberColumn("vol×", help="VOLUME PACE — is this name on pace for an N× volume day vs its own 20-day median? ≥2× required. THIS IS THE HEAVIEST-LIFTING LEG IN THE SIGNAL. Measured over 8 years: WITH the ≥2× gate = 748 trades, +20.2bps, 56% win. WITHOUT the volume leg = 1,983 trades, +7.6bps — you trade 2.6× more often for a THIRD of the return. And the names it REJECTS (every other leg passing, only volume failing) = 1,235 trades worth −0.1bps. Worthless. A coin flip. WHY: price is what people are willing to SHOW you; volume is what they are willing to PAY. A stock can drift up 2.5%, close at its high and sit above VWAP on a thin book — on a chart it looks identical to accumulation. It is not. INSTITUTIONS CANNOT BUY QUIETLY: moving real size MUST leave a volume footprint. No surge means no institution was there — you are looking at a chart pattern, not a footprint. This leg is what separates 'a stock went up' from 'someone with real money bought it'. TIME-NORMALISED so 2.0 means the same at any hour: the raw cumulative-volume/median is mechanically tiny early (only ~4% of a day's volume has traded by 09:15, ~35% by 11:00), so a genuine 2× day would read 0.27 at 09:30 and the gate could never pass before ~15:00. Dividing by the elapsed-volume fraction fixes that. At/after 15:25 the fraction is 1.0, so this EQUALS the raw ratio — the validated close decision (and the 8yr backtest) is unchanged. Blank = no volume yet (pre-open / market closed).", format="%.2f"),
    "RS%": st.column_config.NumberColumn("RS%", help="Relative strength vs Nifty today (stock% − index%). >0 = outperforming the market; persistent RS-leaders carry the overnight edge.", format="%.2f"),
    "delivTr": st.column_config.NumberColumn("delivTr", help="TRAILING delivery% (3-day avg through yesterday) from the EOD archive — the LEAK-FREE accumulation leg, known live at the close. ≥60 required for BTST-CARRY (a real footprint, not just price-action FORMING).", format="%.1f"),
    "book": st.column_config.TextColumn("book", help="THE RISK LAYER — is this name actually in tonight's book? '✓ TAKE' = it survived the sector cap (max 2 per sector, so five banks aren't one macro bet) and the top-5 cap, and is sized by the self-calibrator. '✗ capped' = it IS a valid BTST-CARRY signal, but it was dropped by the concentration cap or fell outside the top-5 by conviction. Only BTST-CARRY names get a book verdict. This board is what you act on at 15:10-15:30, so the book is constructed HERE, at the moment of the decision."),
    "wt%": st.column_config.NumberColumn("wt%", help="Portfolio weight — equal-weight across the selected names, SCALED by the self-calibration size multiplier (currently 0.30 = start small). Weights sum to the multiplier, not to 100%: the rest is CASH. 0 = capped out of the book, or the calibrator says stand aside.", format="%.1f"),
    "turn₹L": st.column_config.NumberColumn("turn₹L", help="TODAY's turnover in ₹ lacs (volume × price). Must be ≥2000 (₹20cr) for BTST-CARRY — the realism floor, so a fill is actually achievable without moving the price. IMPORTANT: this is TODAY's turnover, not yesterday's. A footprint day carries a 2× volume surge, so gating on yesterday's turnover was silently dropping 23% of validated signals — and the BEST ones (+29.3bps).", format="%.0f"),
    "rsCum%": st.column_config.NumberColumn("rsCum%", help="PERSISTENT relative strength — the 10-day CUMULATIVE RS vs Nifty (sum of daily stock−index returns), not a one-day burst. This is the VALIDATED leg: a name that has led the index for ~10 sessions carries the overnight edge; a one-day-burst laggard decays from ~+30bps to ~+19bps. Must be > 0.", format="%.2f"),
    "cvwap%": st.column_config.NumberColumn("cvwap%", help="PATH SIGNATURE — how far above the session VWAP the name is trading (%). ≥0.5% required. This separates a stock that TRENDED UP AND HELD (real accumulation) from one that SPIKED AND FADED back to VWAP (distribution into strength). A validated leg: it lifts net +26→+30bps, win 57→61%, and flips the losing 2025 year from −11.6 to +6.2. Blank = session VWAP not fetched (that leg is NOT met — no free pass).", format="%.2f"),
    "btst": st.column_config.TextColumn("btst", help="Footprint readiness x/5 — the EXACT legs the 8-year backtest was validated on: strong close (clr≥0.70) · up≥1% · volume pace≥2× · PERSISTENT 10-day RS>0 · PATH-SIGNATURE (≥0.5% above session VWAP). BTST-CARRY additionally requires trailing delivery ≥60 (delivTr), the regime gate, and the 15:10–15:30 close window. Proven identical to features.signal_mask over 474,315 rows / 8 years — so a name shown here is a name the backtest actually blessed. THE LEGS ARE NOT EQUAL. The VOLUME leg does the heaviest lifting: drop it and the edge collapses from +20.2 to +7.6bps. DELIVERY is the leg that fails most often in practice — plenty of names reach 5/5 on price and volume, then die on delivTr < 60. Strong price with weak delivery is churn, not accumulation: the shares changed hands but nobody kept them."),
    "exp_ON": st.column_config.TextColumn("exp_ON", help="Expected overnight drift IF the footprint holds (full-footprint historical average, gross — NOT a per-name forecast). Exit next-morning strength."),
    "entry": st.column_config.NumberColumn("entry", help="Suggested entry ≈ LTP. RISK GEOMETRY, not a forecast.", format="%.2f"),
    "stop": st.column_config.NumberColumn("stop", help="Protective stop = 1×ATR below (or below day-low if tighter). Defines your risk.", format="%.2f"),
    "t1": st.column_config.NumberColumn("t1", help="Target 1 = +1×ATR. An ATR-sized move, for trade management — not a predicted price.", format="%.2f"),
    "t2": st.column_config.NumberColumn("t2", help="Target 2 = +2×ATR. Stretch/runner target.", format="%.2f"),
    "risk%": st.column_config.NumberColumn("risk%", help="Distance to stop as % of price = position risk. Size so a name's stop-out is survivable.", format="%.2f"),
    "atr%": st.column_config.NumberColumn("atr%", help="Daily ATR as % of price = how much this name typically moves. Bigger = wider stops/targets.", format="%.2f"),
    "band_lo": st.column_config.NumberColumn("band_lo", help="Lower bound of the ~68% expected next-day CLOSE band (close − 0.6×ATR). Where price is LIKELY to be — a calibrated range, not a target.", format="%.1f"),
    "band_hi": st.column_config.NumberColumn("band_hi", help="Upper bound of the ~68% expected next-day CLOSE band (close + 0.6×ATR).", format="%.1f"),
    "action": st.column_config.TextColumn("action", help="BTST-CARRY = near close the full footprint holds → SHIFT this intraday name to an overnight hold (next-day move expectable; exit next-AM; delivery confirms at close). FORMING = footprint building earlier in the day. NEUTRAL = watch. AVOID = weak close / down / illiquid / regime-off. No SELL — short overnight is proven dead (20% win)."),
}


@st.cache_data(ttl=300)
def _board(date_str: str):
    return screen.board(pd.Timestamp(date_str))


@st.cache_data(ttl=30)
def _last_date():
    return data.last_trading_date()


# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Equity BTST")
# The archive can be transiently unavailable — the DCM dashboard holds DuckDB read-write
# (many-readers-OR-one-writer), or the nightly sync is mid-write. data._connect already
# raises a plain-English RuntimeError; catch it HERE so the page shows that guidance as a
# clean banner instead of a Python traceback with an "Ask ChatGPT" button. Retry a couple
# of times first — a sync lock clears in seconds.
last = None
for _try in range(3):
    try:
        _last_date.clear() if _try else None      # bypass the cached exception on retry
        last = _last_date()
        break
    except Exception as _e:
        _err = _e
        if _try < 2:
            import time as _t
            _t.sleep(1.0)
if last is None:
    st.error(str(_err))
    st.caption("This board only READS the archive — it never writes to it. The moment the "
               "other process releases the file, hit **↻ refresh** (top-left) and the board "
               "loads. Nothing here is broken.")
    if st.button("↻ retry now"):
        st.cache_data.clear()
        live.clear_universe_cache()
        sector_tilt.clear_cache()
        fno.clear_cache()
        arb.clear_cache()
        scalp.clear_cache()
        st.rerun()
    st.stop()
    # st.stop() only raises inside a real script-run context. Imported (tests, tooling) it is
    # a no-op and execution FALLS THROUGH to `last.date()` below, which then dies with a bare
    # `AttributeError: 'NoneType' has no attribute 'date'` four lines from the actual cause.
    # Re-raise so the lock names itself wherever this module is loaded.
    raise RuntimeError(str(_err))
tf = st.sidebar.radio("Timeframe",
                      ["BTST (overnight)", "Intraday", "🎬 Replay (practice)",
                       "🧮 Arbitrage"],
                      index=0)
date = st.sidebar.date_input("As-of close", value=last.date(),
                             max_value=last.date())
if st.sidebar.button("↻ refresh"):
    st.cache_data.clear()
    live.clear_universe_cache()      # also drop the EOD universe cache (picks up a new sync)
    sector_tilt.clear_cache()        # lru_cache is invisible to st.cache_data.clear()
    fno.clear_cache()
    arb.clear_cache()
    scalp.clear_cache()
test_mode = st.sidebar.checkbox("🧪 Test mode (show live board off-hours)", value=False,
                                help="Bypass the market-closed gate so you can exercise the "
                                     "UI now. Off-hours Fyers data is UNRELIABLE (indicative "
                                     "prices, junk volume) — for layout/flow testing only.")

# ── self-calibration panel (learns the KNOB, signal stays LOCKED) ───────────────
try:
    from eqbtst import calibrate as _cal

    @st.cache_data(ttl=3600)
    def _calib_now():                    # cache: never block the sidebar on a full backtest
        return _cal.calibrate()

    _cs = _cal.load_state()
    if _cs is None:                      # never run in the nightly loop yet — compute once, cached
        _cs = _calib_now()
    with st.sidebar.expander("🎛️ Self-calibration (size knob)", expanded=True):
        _mult = _cs["size_multiplier"]
        _emoji = "🟢" if _mult >= 0.8 else ("🟡" if _mult > 0 else "🔴")
        st.metric("Size multiplier", f"{_emoji} {_mult:.2f}",
                  help="Learned from the paper/live ledger, Bayesian-shrunk to the backtest "
                       "prior. Only CUTS below 1.0 when live underperforms; never levers above "
                       "backtest. 0 = stand aside (edge negative or decayed). The SIGNAL is "
                       "locked — this tunes SIZE only.")
        st.caption(f"posterior net-edge **{_cs['posterior_net_bps']:+.1f}bps** "
                   f"(±{_cs['posterior_sd_bps']:.0f}) · prior {_cs['prior']['mu0']:+.1f} · "
                   f"n={_cs['n_closed']} · {_cs['trust']}")
        st.caption(f"effective cost {_cs['cost']['effective_cost_bps']:.0f}bps "
                   f"({_cs['cost']['note']})")
        if _cs["decayed"]:
            st.warning("⚠ DECAY — edge underperforming; size forced to 0.")
except Exception as _e:
    st.sidebar.caption(f"calibration unavailable: {_e}")


# ── arbitrage spreads (CONTEXT — nothing here is wired to a decision) ──────────
# Deliberately NOT a menu of arbitrage strategies. The classic list was triaged against
# what an Indian retail cash account can execute and what this archive can compute, and
# almost none of it survives -- cash-and-carry nets ~2.6% annualised after Securities
# Transaction Tax and physical settlement, the reverse leg needs an overnight cash short
# India does not offer retail, and ETF-NAV / ADR / convertible / merger arb are either
# Authorised-Participant-only or have no data here. The reasoning, with the measured
# numbers behind every claim, lives in the arb.py module docstring. What is left is the
# CARRY SPREAD, read as a confirmation on the cash footprint rather than traded.
try:
    _mk = arb.market(date)
    with st.sidebar.expander("🧮 Arbitrage spreads (context)", expanded=False):
        _na = _mk["nifty_ann"]
        if _na == _na:                       # not NaN
            _e2 = "🟢" if _na >= 4 else ("🟡" if _na >= 0 else "🔴")
            st.metric("Cost of carry (Nifty)", f"{_e2} {_na:+.1f}%",
                      help=arb.HELP_MARKET)
            st.caption(f"{_mk['pctile']:.0f}th percentile of the last 2 years · "
                       f"median +6.2% · negative on only 5.8% of sessions")
        if _mk["roll_med"] == _mk["roll_med"]:
            st.caption(f"roll (near→next) **{_mk['roll_med']:+.1f}%** — the lending rate. "
                       f"Measured null as a signal (t +1.5); weather, not forecast.")
        if _mk["n"]:
            st.caption(f"⚠️ **{_mk['n_exdiv']} of {_mk['n']}** names carry a likely pending "
                       f"**ex-dividend** — their overnight gap averages −1.2bps against "
                       f"+8.4bps for the rest. Flagged in the `carry` column.")
        if _mk["stale_days"]:
            st.warning(f"F&O bhavcopy is **{_mk['stale_days']}d** behind "
                       f"({_mk['asof']:%d %b}) — carry is that stale.")
        st.caption("✅ in `carry` = top carry quintile — a **tie-breaker, not a trigger**. "
                   "With the close-strength footprint it measured **+20.9bps** overnight "
                   "(t +5.9) against **+10.2bps** for close-strength alone. Real ordering, "
                   "but **no combination clears the 22bps cost floor**. Prefer ✅ among names "
                   "the engine already picked; never open one because of it. "
                   "**Nothing here filters the board.**")
        st.info("📄 Full triage of all 21 classic arbitrages, the per-name carry board and "
                "the evidence behind these numbers → switch **Timeframe** to "
                "**🧮 Arbitrage**.")
except Exception as _e:
    st.sidebar.caption(f"arbitrage spreads unavailable: {_e}")


def render_price_band():
    """Compact price-band filter row, right-aligned above the table.

    Defaults come from config.PRICE_MIN_DEFAULT / PRICE_MAX_DEFAULT — a position-SIZING
    preference, so it belongs to the user, not to a measurement. Set them there.

    What the code owes you is VISIBILITY, not a decision: the band cuts BEFORE the structure
    logic, so it silently decides what the horizon dropdown may even consider. Measured on a
    243-name board, a Rs900 cap removed 137 names (56%) including RELIANCE / ICICIBANK / INFY
    and took LONG candidates from 9 to 2. Perfectly reasonable if the cap reflects real
    sizing; a trap only when forgotten. So the board always states the cut and the count."""
    sp, c1, c2 = st.columns([6, 1, 1])
    sp.markdown("**Price band (₹)** — sizing filter, off by default (Max 0 = no limit) →")
    c1.number_input("Min ₹", min_value=0.0, value=float(config.PRICE_MIN_DEFAULT),
                    step=50.0, key="price_min")
    c2.number_input("Max ₹ (0 = all)", min_value=0.0, value=float(config.PRICE_MAX_DEFAULT),
                    step=50.0, key="price_max")


def price_filter(df, col):
    """Keep rows whose price column is within the sidebar min/max band."""
    lo = st.session_state.get("price_min", 0.0)
    hi = st.session_state.get("price_max", 1e9) or 1e9
    if df.empty or col not in df.columns:
        return df
    return df[(df[col] >= lo) & (df[col] <= hi)]


# ── sector-tilt context column ────────────────────────────────────────────────────────
# WHICH CLOSE THE TILT IS READ AS-OF, per lane. This is the leakage contract, not a
# convenience: the tilt must be built from the last close that had ACTUALLY PRINTED at the
# decision instant. The EOD board decides ON a close, so it reads that same close (aligned).
# The live intraday board decides DURING a session that has not closed, so it reads the last
# COMPLETED close. Replay does the same relative to the replayed date — never that date's own
# close, which would feed the session's outcome back into a decision taken inside it.
_ASOF_EOD = pd.Timestamp(date)          # BTST tab: the signal close itself
_ASOF_LIVE = last                       # intraday: the last completed close (today has not closed)


def render_tilt_help():
    """The full `sector tilt` explainer, on the page rather than in a tooltip.

    A hover popup cannot be relied on for this: Streamlit clips it, and the clipped part is the
    honesty section (relative-not-absolute, UW-is-not-a-short, measured-and-it-does-not-help) —
    precisely the part that must not be the part you never see. No key needed: an expander is
    not a stateful widget, so the same label may appear in more than one lane."""
    with st.expander("🧭 What the **sector tilt** column means (and what it does NOT mean)"):
        st.markdown(sector_tilt.HELP_FULL)


DELIV_HELP_FULL = """
### 📦 The delivery columns, in plain English

**Delivery %** = out of everything traded in a stock that day, how much was actually *taken
home* instead of being bought and sold again the same day. High delivery means someone is
**building a position**. Low delivery means the volume was mostly day-traders passing it around.

---

#### The five numbers

They are the **five most recent readings, newest on the LEFT.** What one reading covers depends
on your **Trade horizon**:

| horizon | each number is | header says |
|---|---|---|
| Intraday · BTST | **one trading session** | `deliv 5d` |
| Swing · Positional | **one week** | `deliv 5wk` |

A one-night carry needs to see what the last few *days* did. A multi-week hold does not — daily
numbers would just be noise there.

⚠️ Because newest is on the left, the numbers run **backwards in time**. So a *rising* delivery
trend appears as *descending* numbers: `48, 43, 36, 33, 37` means delivery has been **climbing**.

---

#### A real cell, worked end to end

MANAPPURAM on 31-Jul, BTST horizon (`deliv 5d · base 30d`):

    48, 36, 44, 22, 48   Base -> (+3%)

**Step 1 — the five numbers** are just the last five sessions, latest first. Both the recent
figure and the Base come from the same formula:

    rate = SUM(delivery% x turnover)  /  SUM(turnover)

| session | delivery % | turnover (₹ lacs) | delivery % × turnover |
|---|---|---|---|
| Fri 31-Jul | 48.0% | 24,418 | 1,170,827 |
| Thu 30-Jul | 35.8% | 32,666 | 1,170,742 |
| Wed 29-Jul | 43.7% | 21,068 | 921,317 |
| Tue 28-Jul | 22.1% | 8,755 | 193,746 |
| Mon 27-Jul | 48.0% | 13,874 | 666,110 |
| **total** | | **100,781** | **4,122,742** |

    4,122,742 / 100,781  =  40.91%     <- the recent five sessions, together

Read that as: **of every rupee that traded across those five sessions, 40.9% was taken for
delivery.** Note the simple mean of the five percentages is 39.53% — a *different* number,
because it lets Tuesday (22.1% on the lightest turnover of the week) count as heavily as
Thursday's 32,666. An average of ratios is not the ratio.

**Step 2 — the Base** is the identical calculation over the **30 sessions before those five**
(12-Jun → 24-Jul):

    SUM(delivery% x turnover) = 14,600,485
    SUM(turnover)             =    366,660
                                ----------
                                    39.82%     <- this stock's normal rate

**Step 3 — compare them:**

    40.91 / 39.82 - 1  =  +3%

So MANAPPURAM is delivering **3% more than it normally does** — business as usual. Nothing to
see. The trend across the five is choppy too: one weak day (Tue 22%) inside an otherwise ~45%
week, no direction.

#### What "Base" is

**The stock's own normal delivery rate**, measured over the trading days immediately before the
window shown. Not a market average, not a fixed number — its own.

| trade horizon | baseline |
|---|---|
| Intraday | 15 sessions |
| BTST | 30 sessions |
| Swing | 45 sessions |
| Positional | 60 sessions |

Those are **trading days**, so they reach further back than they sound: 30 sessions ≈ **6 weeks**
of calendar time, 60 ≈ **3 months**.

**Why it has to be there.** Delivery % is not comparable between stocks — across this board the
normal rates run from about **15% to 66%**. So the same reading means opposite things:

| | this week | its normal | reads as |
|---|---|---|---|
| a high-delivery name | 62% | 69% | **−10%** — a *quiet* week |
| a low-delivery name | 31% | 14% | **+116%** — delivery *doubled* |

62% looks twice as impressive as 31%. It is the opposite. The **numbers give you the direction**;
the **Base % gives you the level**.

`Base -> (…)` is **not** a day-on-day change, **not** a slope, **not** a return.

---

#### How big is big

The `%` is a *relative* deviation, so it needs its own yardstick. Measured across 1,837 trading
days, this is where each stock sits among the names on the board that day:

| where it sits | typical value | how often |
|---|---|---|
| bottom 10% | **−22%** or worse | delivery draining out |
| lower quarter | −10% | |
| middle | +1% | ordinary week |
| upper quarter | +14% | |
| top 10% | **+28%** or better | genuine accumulation |

In plain terms — on a typical day about **31%** of names read above +10%, **17%** above +20%, and
only **9%** above +30%. So:

* within **±10%** — ordinary, most of the board lives here
* beyond **±20%** — top or bottom sixth
* beyond **±30%** — the extremes, roughly the top or bottom tenth

⚠️ **A fixed threshold is not a fixed rank.** The board widens and narrows: the top-10% cutoff
has ranged from **+18% to +41%** across days. On a quiet day +19% already puts a name in the top
tenth; on a wild one it takes +41%. Read the number against the rest of today's board, not
against a memorised line.

#### The small print

* **`*4d`** appears in the **weekly** view only: the week is still forming and has 4 sessions in
  it so far — the noisiest figure in the cell. **No star = the week is finished.** Daily buckets
  never carry it, because a session is done once its delivery publishes.
* A name that did not trade in a bucket shows `–`, and is never marked as forming.
* In the **daily** view the `%` compares the **last 5 sessions together** to the Base, not the
  newest single day. A one-day reading jumps a median **14.9pp per session** versus **3.6pp** for
  a five-day one — 4.1× jumpier. A number that repaints that hard every morning is not a read.
* The Base always ends where the `%` window starts, so what it overlaps differs by view: the
  daily view excludes all five sessions shown; the weekly view ends before the *current* week and
  so overlaps the four older weeks you can see. Switching horizon therefore moves the `%` for two
  reasons at once — a different baseline length **and** a different overlap.

⚠️ **A short baseline drifts with the stock.** If delivery declines slowly for two months, a
30-session base declines with it and the `%` can read near zero — the fall has *become* the new
normal. `KALYANKJIL` shows this: **+24% on a 15-session base, +16% on 30, −7% on 60**, same week.
Read the **numbers** for the trend and the **%** for "unusual versus recent". On a long slow
drift, only the numbers will tell you.

---

#### Does a high Base % actually pay? Measured — the best evidence on this board, still not wired

Joined to 8 years of this engine's own footprint triggers (n=744, regime-gated, net of 22bps),
split into thirds by the Base % **as this column computes it** (5 sessions against the 30 before
them):

| Base % third | n | net overnight | win% |
|---|---|---|---|
| low | 248 | +7.1 bps | 49.6% |
| mid | 248 | +17.4 bps | 58.1% |
| **HIGH** | 248 | **+31.3 bps** | **60.1%** |

Monotone, and the spread is large: **+24.3 bps**, night-clustered **t = +2.24** across 380
nights, **HIGH beat low in 8 of 9 years**. It survives removing the market's own overnight gap
(cross-sectional excess **+18.1 bps, t = +1.89**), so it is not a which-nights-you-traded
artifact, and it is genuinely new information rather than the delivery leg the footprint already
uses (correlation with `delivTr` only **+0.13**).

**So why is nothing wired to it?** Because those thresholds — |t| ≥ 2 and ≥ 7 of 9 years — were
written down in advance for the baseline that was shipped *at the time* (100 days). The 30-day
baseline was then chosen by testing several, and re-scored on the same 8 years. Clearing a bar
you moved the goalposts toward is not the same as passing it.

What genuinely supports it: the effect is positive at **every** baseline tested (5d +14.1, 15d
+6.3, 30d +24.3, 60d +19.4, 100d +21.3), so this is not one lucky cell in a grid — 30 is simply
the best of a set that all point the same way.

Treat a big Base % as a **tiebreaker between names you already like**. It becomes wireable on a
re-test over nights the search never saw, clearing the same bar.

**Context, not a signal.** Delivery's own forward IC is weak (~0.03–0.07). It earns its place as
one leg of the accumulation footprint — alongside close strength, volume and relative strength —
never as a reason to buy on its own.

**Two honest limits.** About **1 week in 4 is holiday-shortened** (15 of the last 61) and renders
identically to a full one. And a single huge-turnover day can dominate a bucket — measured, the
biggest day is a median 32% of weekly turnover and exceeds 70% in only **1.3%** of weeks, so it
is rare rather than routine, but a block deal can still lift one reading on its own.

**Leak-free.** NSE publishes delivery around 6pm, so today's figure does not exist at a 15:15
decision. Every value here is read through the last **completed** session.
"""


def render_deliv_help():
    """Full delivery-column note, on the page rather than in a tooltip.

    Streamlit clips a dataframe column tooltip at ~1,100 characters and gives it no scrollbar;
    two attempts to force one with CSS failed because the tooltip is an internal component this
    stylesheet cannot reliably target. Rather than keep guessing selectors, the long form lives
    here — the same fix already proven for the sector-tilt column."""
    with st.expander("📦 What the **delivery** columns mean (and what 'Base' is)"):
        st.markdown(DELIV_HELP_FULL)


def _dw(df, as_of, horizon: str | None):
    """Recompute `deliv 5wk` for the ACTIVE trade horizon, at render time.

    The baseline scales with the hold (see live.DELIV_BASE_BY_HORIZON), but the universe scan
    is cached and must not re-run just because the horizon dropdown moved — so the column is
    rebuilt here from the per-day delivery table instead. Degrades to whatever the scan already
    attached if the archive is momentarily unavailable."""
    if df is None or df.empty or "symbol" not in df.columns:
        return df
    try:
        bd = live.DELIV_BASE_BY_HORIZON.get(horizon or "btst", live._DELIV_WK_NORM)
        bk = live.DELIV_BUCKET_BY_HORIZON.get(horizon or "btst", "week")
        cells = live.deliv_weeks(as_of, base_days=bd, bucket=bk)["cell"]
        out = df.copy()
        out["deliv trend"] = out["symbol"].map(cells).fillna("—")
        return out
    except Exception as e:
        # SAY SO RATHER THAN LEAVE A STALE NUMBER. The header advertises this horizon's bucket
        # and baseline whether or not this call succeeded, so a silent fallback would print
        # last-computed values under a label describing different settings. Nothing upstream
        # pre-seeds the column any more, so writing the reason here is the only honest option.
        out = df.copy()
        out["deliv trend"] = f"— unavailable ({type(e).__name__})"
        return out


def _wt(df, as_of, side=None):
    """Attach the `sector tilt` column at RENDER time.

    Deliberately applied at the render site rather than upstream: `light` / `bb` / `bd` are
    reassigned by the 5-second price-refresh path, which rebuilds rows from the engine's own
    dicts and would drop a column added earlier. Annotating what is about to be drawn cannot
    go stale and cannot be dropped. Degrades to the unannotated frame — a locked archive must
    cost you a context column, never the board.
    """
    try:
        return sector_tilt.annotate(df, as_of, side=side)
    except Exception:
        return df
st.sidebar.caption(f"EOD archive latest: {last.date()}  •  now {dt.datetime.now():%H:%M}")
st.sidebar.caption("BTST tab = EOD engine (delivery-confirmed). Intraday tab = live Fyers.")

# ── LIVE intraday board (Fyers) ────────────────────────────────────────────────
@st.cache_data(ttl=5)
def _live_board():
    return live.quotes_board()


@st.cache_data(ttl=60)
def _tf_scan(tf: str):
    return live.tf_scan(tf)


@st.cache_data(show_spinner=False, max_entries=1)
def _uni_scan(nonce: int):
    # max_entries=1 CAPS MEMORY. There is deliberately no TTL (the nonce is the only trigger),
    # but without a size cap st.cache_data kept ONE full-universe board (~270 names x 6 TF + wall
    # lists) PER nonce for the whole session -- and the nonce only increments, so a heavy day of
    # re-scans leaked dozens of stale boards that are never read again (the VM is OOM-prone). The
    # nonce is monotonic and only the CURRENT one is ever requested, so keeping a single entry is
    # functionally identical: a fresh nonce evicts the previous board, filters still hit instantly
    # within a nonce (it is the one live entry), and the ↻ button still forces a true re-fetch by
    # also clearing live._UNISCAN_CACHE.
    # NO TTL — the nonce is the ONLY trigger. A 30-minute TTL directly contradicted the intent
    # written below: once it expired, the next interaction of ANY kind paid a 30-60s cold
    # re-scan, so typing in the price box could kick off a full universe fetch and leave the
    # previous frame on screen looking like the filter had done nothing. Exactly the surprise
    # this design set out to prevent, reintroduced by a timer. Freshness is handled where it
    # belongs: the ↻ button, the opt-in bar-close auto-refresh, and an age stamp on screen.
    # Pinned by an explicit nonce, NOT by time: the ~270-fetch scan must re-run ONLY on the ↻
    # button (which bumps the nonce), never as a side effect of moving a filter widget. Without
    # this, a filter change that happens to land after the 5-min memo bucket rolls would trigger
    # a surprise ~30s cold re-scan. Filters must be instant; scanning must be deliberate.
    sc = live.universe_mtf_scan()
    # RAISE on failure OR on an empty board, so st.cache_data NEVER stores a barren result.
    # Two poisoning paths this closes: (1) a scan before the ~06:00 token refresh -> ok=False;
    # (2) a scan in the pre-open / first seconds of the session -> ok=True but ZERO quotes ->
    # empty board. Either one, if cached, pins "no names" for the full 30-min TTL — so the board
    # stays empty even after the market opens. Raising keeps it OUT of the cache, so the very
    # next rerun (or the one-shot auto-retry below) re-fetches cleanly.
    if not sc.get("ok"):
        raise RuntimeError(sc.get("status") or "universe scan unavailable")
    if sc["board"].empty:
        raise RuntimeError("no names returned (pre-open, or a transient quote-fetch miss)")
    # (3) STRUCTURALLY BARREN: rows came back, but with NO structure on any frame. The quote
    # endpoint is cheap and usually survives while /history is rate-limited AND the DuckDB
    # archive is locked (DCM mid-sync) — that combination yields a full-looking board of ~243
    # names whose every timeframe reads '—'. It passes both guards above, so it used to be
    # cached for the whole TTL: a board that looks populated, filters to nothing, and cannot be
    # cleared by any action except ↻. The DAILY frame is the test — it comes from the archive
    # at zero API cost, so if even 1D is blank across the board, this scan is not usable.
    _bd = sc["board"]
    if "s1D" in _bd.columns and (_bd["s1D"] == "n/a").mean() > 0.9:
        raise RuntimeError("structure came back empty for every name — the EOD archive read "
                           "failed (DuckDB busy: DCM may be syncing) and/or the broker "
                           "rate-limited the history calls. Not caching this board.")
    return sc


SELL_COLS = {
    "short": st.column_config.TextColumn("short", help="Distribution readiness x/4: weak close · down ≥1% · vol surge · RS-laggard. High = supply in control."),
    "s_stop": st.column_config.NumberColumn("s_stop", help="Short stop = 1×ATR ABOVE entry. Defines risk on a short.", format="%.2f"),
    "s_t1": st.column_config.NumberColumn("s_t1", help="Short target 1 = −1×ATR (price falls). Risk geometry, not a forecast.", format="%.2f"),
    "s_t2": st.column_config.NumberColumn("s_t2", help="Short target 2 = −2×ATR.", format="%.2f"),
    "sell": st.column_config.TextColumn("sell", help="SHORT = full distribution footprint + bearish candle. WEAK = building. INTRADAY ONLY — square off before the close. Overnight short is proven -EV (win 20%). Not validated alpha; trade small."),
}

TF_COLS = {
    "bar_clr": st.column_config.NumberColumn("bar_clr", help="Close location in the LAST bar of the chosen timeframe (1h/2h). ≥0.70 = closing that bar strong."),
    "vs_vwap%": st.column_config.NumberColumn("vs_vwap%", help="Price vs session VWAP. Positive & above = buyers in control."),
    "rsi7": st.column_config.NumberColumn("rsi7", help="Fast RSI (proactive) — turns before RSI14."),
    "rsi14": st.column_config.NumberColumn("rsi14", help="Standard Wilder RSI(14)."),
    "tone": st.column_config.TextColumn("tone", help="Momentum tone: strong / neutral / rolling-over / weak (from fast-RSI slope)."),
    "structure": st.column_config.TextColumn("structure", help="Market structure on the timeframe you picked — computed over the last ~20 bars OF THAT TIMEFRAME, so it CHAINS ACROSS PRIOR DAYS (a 2h read spans ~7 trading days; a 4h read ~10). It is not today-only. Kaufman efficiency + range logic: TREND_UP/DOWN (efficient directional), RANGE (choppy), CONSOLIDATION (range contracting — coil), BREAKOUT_UP/DOWN (beyond prior range). CONTEXT ONLY — intraday structure has no validated edge; use it to understand the tape, not as a buy/sell signal."),
    "action": st.column_config.TextColumn("action", help="STAGE-2 verdict. The stock reached this table because the 1-DAY BAR let it in (day% ≥1% and day-clr ≥0.5, top 25). THIS column is the timeframe's judgement of it: LONG = the last bar OF THIS TIMEFRAME closed strong + above session VWAP + RSI not weak + RS-leader. AVOID = weak bar / below VWAP / regime-off. Long-only. NOTE: intraday direction has no validated overnight-grade edge — manage strictly by the stop."),
}

# HTF x LTF synthesis columns — the chartist read (see eqbtst/mtf.py).
SETUP_COLS = {
    "setup": st.column_config.TextColumn(
        "setup", width="medium",
        help="The HIGHER-TF × LOWER-TF read for your chosen horizon — what the two frames say "
             "TOGETHER, which neither says alone.\n\n"
             "**The ↑ / ↓ arrow is the direction of the HIGHER-TF TREND.** The same tag means "
             "opposite trades by direction: **↑ = LONG** (uptrend), **↓ = SHORT** (downtrend). "
             "WITH-TREND CONTINUATION ↑ is a bull flag; WITH-TREND CONTINUATION ↓ is a bear "
             "flag — same shape, opposite side. The `side` column follows the arrow.\n\n"
             "🎯 WITH-TREND CONTINUATION — HTF trending, LTF coiling into it. Textbook.\n"
             "🚀 RANGE-TOP BREAK — LTF breaking up AT the HTF ceiling (the only place a break "
             "can be real).\n"
             "🔻 RANGE-FLOOR BREAK — the mirror, at the HTF floor.\n"
             "↩️ PULLBACK vs HTF — LTF against the HTF trend: a dip zone if the HTF holds, an "
             "early reversal warning if it breaks.\n"
             "⚠️ EXTENDED (aligned) — both frames same direction but late; chasing.\n"
             "🌀 NESTED SQUEEZE — both compressing. Move loading, DIRECTION UNKNOWN. Wait.\n"
             "〰️ DRIFT-IN-RANGE — LTF wandering mid-box. Noise.\n"
             "🪤 FALSE-BREAK TRAP — LTF break in the MIDDLE of the HTF box. Statistically fades. "
             "Do not chase.\n\n"
             "CONTEXT, not a signal — intraday MTF alignment is unvalidated in this stack."),
    "loc": st.column_config.NumberColumn(
        "loc", format="%.2f",
        help="WHERE price is sitting inside the bigger timeframe's range — bottom to top.\n\n"
             "**0.0 = at the bottom (floor) · 1.0 = at the top (ceiling) · 0.5 = middle.**\n\n"
             "This decides if a breakout is real: near the top (≥0.72) a break UP can genuinely "
             "clear the range; near the bottom (≤0.28) a break DOWN is real. In the MIDDLE, a "
             "'break' is usually a fake — there was nothing to break through.\n\n"
             "**Example:** range ₹350–₹400, price ₹380 → loc = 0.60 (a bit above the middle). "
             "Price ₹398 → loc 0.96 (right at the ceiling — a break up here is real)."),
}

# Touch-counted dynamic support/resistance (eqbtst/indicators.py :: walls).
SR_COLS = {
    "sup": st.column_config.NumberColumn(
        "sup", format="%.2f",
        help="**SUPPORT — the nearest FLOOR below the current price.** A price level where the "
             "stock has BOUNCED UP before, so buyers tend to step in there. Not a line someone "
             "drew — a spot where price actually turned around, more than once.\n\n"
             "Blank = no clear floor below (price is in open air under it). Prices are adjusted "
             "for splits/bonuses, so they match today's scale.\n\n"
             "**Example:** `sup 355.00` — the stock keeps bouncing up from around ₹355. If it "
             "drops toward there, buyers have stepped in before, so it may hold again."),
    "sup_t": st.column_config.NumberColumn(
        "sup×", format="%d",
        help="**How many times price has BOUNCED off that support** (the ×). This is your "
             "question answered: how often the stock got held up at this floor.\n\n"
             "**1 = touched once (weak, could be luck) · 2–3 = a real floor the market keeps "
             "defending.**\n\n"
             "⚠ MORE IS NOT ALWAYS STRONGER. Measured: at **5+ touches the edge flips** — a "
             "floor hit that many times is usually being worn down and about to BREAK. So a "
             "high number means 'this level is about to matter, one way or the other', not "
             "'rock-solid floor'.\n\n"
             "**Example:** `sup 355 · sup× 3` — price bounced up off ₹355 three separate times. "
             "A defended floor. `sup× 1` at the same price = it only touched once — much weaker."),
    "res": st.column_config.NumberColumn(
        "res", format="%.2f",
        help="**RESISTANCE — the nearest CEILING above the current price.** A level where the "
             "stock has been REJECTED (turned back down) before, so sellers tend to appear "
             "there.\n\n"
             "Blank = clear sky above, no ceiling nearby — common right after a real breakout.\n\n"
             "**Example:** `res 400.00` — the stock keeps getting pushed back down from around "
             "₹400. If it rises toward there, expect sellers; it may stall or reverse."),
    "res_t": st.column_config.NumberColumn(
        "res×", format="%d",
        help="**How many times price has been REJECTED at that ceiling** (the ×). Same reading "
             "as sup×: 1 = weak, 2–3 = a defended ceiling, 5+ = worn down and about to break.\n\n"
             "**Example:** `res 400 · res× 4` — price was turned back down from ₹400 four times. "
             "A strong ceiling — but by 5+ it is more likely to finally break than hold."),
    "side": st.column_config.TextColumn(
        "side", width="small",
        help="Which side the HTF × LTF setup argues for — read off the setup's DIRECTION, "
             "not its name.\n\n"
             "This distinction matters: `WITH-TREND CONTINUATION` is the textbook setup in "
             "EITHER direction. A downtrend coiling for continuation is a **SHORT**, and it "
             "carries the identical tag to the bullish version. Same for `PULLBACK vs HTF` — "
             "a pullback inside a downtrend is a short entry, not a dip to buy.\n\n"
             "**—** = the setup takes no side (squeeze, trap, or sideways). Most of the "
             "universe sits here most of the time."),
    "big_wall": st.column_config.TextColumn(
        "big wall", width="small",
        help="**THE ONE HIGHER-FRAME WALL YOUR PAIR CANNOT SEE.** The setup, sup/res and "
             "headroom all read only the TWO frames of your horizon (e.g. 1h + 4h). But a long "
             "can sit right under a resistance ONE FRAME ABOVE the pair that it never looked "
             "at — you buy, hit the big level, and it reverses. This column shows the nearest "
             "DEFENDED (≥2-touch) wall from the ONE next confirmation frame up, in your trade's "
             "direction: a ceiling above a long, a floor below a short.\n\n"
             "Which frame, by horizon: **Intraday** (15m/1h) → **4h** · **BTST** (1h/4h) → "
             "**1D** · **Swing** (4h/1D) → **1W** · **Positional** → none (already the top "
             "frame). Just the next chart up, as a chartist checks.\n\n"
             "`1D 3745.40 ×4` = a 4-touch DAILY resistance overhead. The `big gap` column is "
             "how far, in your trigger-frame ATR (same unit as headroom). **< 0.5 = you are "
             "buying straight into a major level** — expect a fight; a break THROUGH it is the "
             "real move, so wait for the break or size for the wall. Blank = no defended "
             "bigger-frame wall in your direction (or you are on Positional, the top frame). "
             "Context, not a veto — but you must SEE the level before you trade into it."),
    "big_gap": st.column_config.TextColumn(
        "big gap",
        help="Distance to the `big wall` (the nearest 1D/1W defended level in your trade's "
             "direction), in your TRIGGER-frame ATR. ∞ = clear of any bigger-frame wall. "
             "< 0.5 = trading straight into a daily/weekly level — the pair's own headroom can "
             "say 'clear' while THIS says you are capped."),
    "at_wall": st.column_config.TextColumn(
        "at wall", width="small",
        help="**Price is TESTING a defended level RIGHT NOW.** It is sitting on a floor or "
             "ceiling it has already bounced off ≥2 times before.\n\n"
             "`RES 2896.41 x3` = price is at a ceiling it was rejected from 3 times. This is the "
             "decision moment: it either turns away again (4th rejection) or breaks through — "
             "and a break of a well-defended level is the bigger event.\n\n"
             "Blank = price is not near any such level right now.\n\n"
             "⚠ Trust this on the DAILY and WEEKLY only. On fast frames (15m–4h) the 'bounce' "
             "is just intraday chop — measured no better than a random line."),
    "long_note": st.column_config.TextColumn(
        "long evidence", width="medium",
        help="**Did this setup actually make money — over the hold YOU picked?** Tested on 9 "
             "years of this exact universe, buying at the next open. "
             "**The number compares it to buying ANY random stock.** That matters, because "
             "this market drifts up on its own: a plain positive return proves nothing. "
             "**Where the money is:** a random stock gains +0.17% overnight, then GIVES BACK "
             "-0.10% during the next day's session. So the profit is the overnight GAP, not "
             "the trend. And 5 days of holding earns the same +0.17% as that one night — four "
             "extra days add nothing. That is why the rule is: sell next day, early. "
             "**Best per hold:** overnight → 🚀 RANGE-TOP BREAK (+0.26%, worked 9 of 9 years — "
             "the most reliable result on this board). 5 days and longer → 🎯 WITH-TREND "
             "CONTINUATION (+0.41% and +0.93%). "
             "**Careful:** RANGE-TOP BREAK is the BEST overnight and LOSES money if you hold "
             "it 5 days. 🧱 COIL AT THE EXTREME is the weakest buy anywhere. "
             "That is why this tab re-sorts when you change the Trade Horizon."),
    "short_note": st.column_config.TextColumn(
        "short evidence", width="medium",
        help="**Did shorting this setup actually make money — over the hold YOU picked?** "
             "Same 9-year test, same universe, entry at the next open. "
             "**The number compares it to shorting ANY random stock.** "
             "**The short answer: nothing here works.** Not one setup beats a random short on "
             "the day after the signal, and EVERY multi-day short loses money outright — "
             "because the market drifts up while you are short. Shorting a random stock for 20 "
             "days loses 1.2%; the least-bad setup still loses 0.9%. "
             "**The one thing worth knowing:** 🔻 RANGE-FLOOR BREAK — a fresh breakdown — is "
             "the WORST thing to short the next day (-0.41% vs a random short). It BOUNCES. "
             "Over weeks it is the least-bad short, but that is a different trade entirely. "
             "**So read this tab as a list of names to AVOID or EXIT, never to short.** "
             "Note the direction was NOT flipped into buy signals: a signal that measures "
             "backwards is evidence the logic is wrong, not free money the other way."),
    "headroom": st.column_config.TextColumn(
        "headroom",
        help="**How far to the level shown in `res` (long) / `sup` (short)** — the SAME nearest "
             "level, just expressed as a distance in ATR (the unit of your stop and target). For "
             "a **LONG** it is room UP to `res`; for a **SHORT** it is room DOWN to `sup`. The "
             "LONG/SHORT tab sets which way it looks.\n\n"
             "• **∞** = CLEAR ROAD — no level that side at all.\n"
             "• **< 0.5** = you are trading RIGHT INTO the level — it sits between you and your "
             "1-ATR target. The `res×`/`sup×` count tells you how many times price turned there.\n\n"
             "**It matches `res`/`sup` by construction** — if the `res` column shows a level "
             "overhead, headroom is finite, never '∞ clear'. (It used to gate on ≥2 touches, so a "
             "single violent rejection — one big spike-and-crash — read '∞ clear' while `res` "
             "showed the level right there. Fixed.)\n\n"
             "**Example (long):** price ₹380, `res` ₹388, ATR ₹8 → headroom = (388−380)/8 = "
             "**1.0**. If `res` were ₹383 → **0.4**, you'd hit it almost at once. A short reads "
             "the same toward `sup` below.\n\n"
             "⚠ A MAP, NOT A SIGNAL. Measured (7,061 daily approaches, placebo-controlled): a "
             "real swing high is respected **69.2%** vs **70.5%** for a random line — the level "
             "does NOT cap price more than chance, and the touch count does not change that. Use "
             "it to SEE where the visible level is, never as a reason the trade will work."),
}

# Delivery-conviction columns — ported from the DCM sector-rotation view (same formulas).
DELIV_HELP_SHORT = ("DELIVERY % — share of volume taken for delivery, not squared off intraday.\n\n"
             "'48, 36, 44, 22, 48  Base -> (+3%)'\n"
             "• FIRST = most recent; the rest go BACKWARDS (a RISING trend reads DESCENDING).\n"
             "• Each number = one SESSION on Intraday/BTST, one WEEK on Swing/Positional — the "
             "header says which.\n"
             "• 'Base -> (+3%)' = vs this stock's OWN normal rate over the header's baseline "
             "('base 30d' = the 30 sessions before them).\n\n"
             "SCALE: ±10% = ordinary · ±20% = top/bottom sixth · ±30% = the extremes "
             "(~top/bottom tenth). Cutoffs drift daily.\n\n"
             "Full note + worked example: the '📦 delivery columns' expander.")

# F&O positioning columns. Tooltips live in fno.py beside the code that builds the labels,
# so a label change and its explanation cannot drift apart (they have, twice, in this file).
# Built from fno.COLS so the config can never drift out of step with the column list -- the
# two have already disagreed once (the block vanished under a filter). Each expiry shows the
# 1-DAY read then its CYCLE read, which is DCM's own ordering: backdrop beside today's move.
# Futures carry the CYCLE read, options the 1-DAY read -- so the tooltips differ by
# instrument, not by column name. See the block above fno.COLS for why.
_FNO_HELP = {"Fut Near": fno.HELP_FUT_CYC, "Fut Next": fno.HELP_FUT_CYC,
             "Opt Near": fno.HELP_OPT,     "Opt Next": fno.HELP_OPT}
# `carry` rides with the F&O block: same instrument (the futures), same status
# (CONTEXT, wired to no decision). It is a separate module because it is a SPREAD read
# -- futures against cash -- not a positioning label. See arb.py for the measurement.
ARB_COLS = {"carry": st.column_config.TextColumn(
    "carry", width="small", help=arb.HELP_CARRY)}
# How long the name has been bearish on the DAILY frame, in sessions. Numeric so the tab
# sorts by it -- "how long has this been broken" is the question you scan a short list with.
ARB_DN_AGE = st.column_config.NumberColumn(
    "dn age", width="small", format="%d",
    help=("**SESSIONS THE DAILY STRUCTURE HAS BEEN BEARISH** (TREND_DOWN or "
          "BREAKOUT_DOWN), counted back from the last close and stopping at the first "
          "session that was not.\n\n"
          "**Why an age and not a clock time.** A short here has no moment in the "
          "session. The side comes from the higher timeframe’s STRUCTURE, which changes "
          "when a bar CLOSES — usually days before today — not when price ticks. So there "
          "is no 09:15 to show; the honest answer is how long it has been in this state.\n\n"
          "**This is a DAILY read beside an intraday tab**, deliberately: it is free (the "
          "EOD archive is already loaded and back-adjusted for splits and bonuses), it "
          "works with a dead broker token, and it reaches past the broker’s ~60-day "
          "intraday limit — which would truncate exactly the names that have been bearish "
          "longest.\n\n"
          "Blank = not currently bearish on the daily frame. That is common and correct: "
          "this tab reads its side on the 1h/4h frames, so a name can be SHORT there while "
          "its DAILY structure is still a range.\n\n"
          "**Not a trade.** This list is *names to AVOID or EXIT, never to short.*"))

FNO_COLS = {
    c: st.column_config.TextColumn(
        c, width="small" if c.startswith("Fut") else "medium", help=_FNO_HELP[c])
    for c in fno.COLS
}

DELIV_COLS = {
    "deliv trend": st.column_config.TextColumn(
        "deliv trend", width="medium",
        # SHORT ON PURPOSE. Streamlit clips a dataframe column tooltip at roughly 1,100
        # characters with no scrollbar, and two attempts to force one via CSS failed because the
        # element is an internal component this stylesheet cannot reliably reach. So the tooltip
        # now carries only what fits, and the full explanation lives in the on-page expander
        # (render_deliv_help) — the pattern already proven to render in this app.
        # ORDER IS THE DEFENCE. Streamlit clips this tooltip near ~650 characters with no
        # scrollbar, and the previous version lost its SCALE line to exactly that cut. So the
        # scale now sits ABOVE everything optional: whatever survives the clip contains the
        # numbers that make the column readable. The worked example moved to the expander.
        # STALE TEXT IS WORSE THAN SHORT TEXT. This still said "Weekly", "THIS week" and "its
        # 100-day average" long after the buckets began following the trade horizon and the
        # baseline became 15/30/45/60 sessions — describing a column that no longer existed,
        # under a header that plainly said otherwise. Rewritten to read correctly in BOTH modes
        # and to name the header as the source of truth for which one is active.
        help=DELIV_HELP_SHORT),
    "wtd_deliv7": st.column_config.NumberColumn(
        "wtdDeliv7 %", format="%.1f%%",
        help="7-CALENDAR-day TURNOVER-WEIGHTED delivery % = SUM(deliv%×turnover)/SUM(turnover). "
             "The share of recent volume that took actual delivery (not intraday churn), weighted "
             "by rupees traded. High = real accumulation, not day-trading noise. Ported verbatim "
             "from the Daily_Cash_Market sector-rotation engine. Leak-free (through last close)."),
    "deliv_vs_100d": st.column_config.NumberColumn(
        "vs100D %", format="%+.1f%%",
        help="(7D wtd-delivery ÷ own 100-trading-day baseline − 1) × 100. +15% = recent delivery "
             "is 15% ABOVE this stock's OWN historical norm (conviction building); −10% = fading. "
             "0% = exactly at its own average. Compares a name to ITSELF, not to peers — so a "
             "structurally high- or low-delivery stock is judged fairly. DCM-ported."),
}


HELP_INTRADAY = """
### What am I looking at? How do I use it?

**Purpose:** find NSE-F&O cash stocks showing a *smart-money accumulation footprint*,
and act on the ONE validated edge — the **overnight BTST long** (buy near close, exit
next morning, ~+20bps net on 8yr data, LEAK-FREE). Long-only. Nothing auto-executes.

**Two lanes — do NOT mix them:**
| Lane | Signal | Hold | Validated? |
|---|---|---|---|
| **Intraday** (1h/2h/15m scan, or SELL tab) | price-action strength/weakness | **square off SAME DAY** | ❌ no proven edge — context only, trade small |
| **BTST** (🌙 BTST-CARRY box) | full accumulation footprint at the close | **overnight**, exit next AM | ✅ the real edge |

**The day, step by step:**
1. **09:15–15:00** — watch. BUY tab shows ⏳ **FORMING** names (footprint building). Delivery% (the core leg) only confirms after the close, so nothing is final yet.
2. **~15:10–15:30** — the FORMING names that still hold the full footprint flip to 🌙 **BTST-CARRY** (top green box). **These are your overnight picks.** Enter LONG near the close.
3. **Overnight → next 09:15–09:30** — exit into morning strength.

**BTST-CARRY** = the full LEAK-FREE footprint holding into the close: price legs (clr/up/
vol/RS) AND trailing delivery ≥60 (`delivTr`, sustained accumulation, known live). FORMING
= price legs there but the delivery leg isn't — so it does NOT carry. A name only carries
overnight if flagged 🌙 BTST-CARRY — never just because you bought it intraday earlier.

**Columns:** hover any header. `btst` x/4 = live footprint legs met. `Entry/Stop/T1/T2`
= ATR risk geometry (trade management), **not** a price forecast. `action=EARNINGS` =
excluded (reports results during the hold).
"""

if tf == "🧮 Arbitrage":
    # Its own LANE, not a tab inside another board, because it answers a different question
    # from every other page here: not "what do I trade tonight" but "which of the classic
    # arbitrages are even real for this account". It renders from EOD only -- no Fyers token,
    # no live quotes -- so it works when the broker session is dead, and st.stop() keeps the
    # live lanes from also drawing.
    arb.render_page(date, st)
    st.stop()
    raise SystemExit  # st.stop() is a no-op outside a script-run context (tests, tooling)

if tf == "Intraday":
    # The index strip is RESERVED here and FILLED further down. Streamlit renders in
    # execution order, so a container claims the top of the page now while the code that
    # fills it still runs after the token and market-closed gates. Rendering it inline up
    # here instead would put three tiles above an error on a dead token -- the quotes need
    # the same token the gate is about to reject.
    _index_slot = st.container()
    st.title("Live Intraday Board  ·  Fyers")
    with st.expander("❓ How to use this board — purpose, the two lanes, where BTST-CARRY is"):
        st.markdown(HELP_INTRADAY)
    tok = live.token_status()
    st.caption(tok["describe"])
    if not tok["usable"]:
        st.error("Fyers token not usable (daily ~06:00 IST expiry). Re-auth in Tradebot "
                 "(`python fyers_auth.py`), then refresh. Live board needs a fresh token.")
        st.stop()

    # market-closed gate: Fyers /quotes returns unreliable indicative prices outside
    # the session (lp far off the real close, junk volume). Don't render junk — the
    # BTST tab (EOD, delivery-confirmed) is the valid surface when the market is shut.
    if not live.market_open() and not test_mode:
        st.info("🔒 **Market closed** (live scan runs Mon–Fri 09:15–15:30 IST). Fyers "
                "quotes outside the session are unreliable indicative prices, so the live "
                "board is hidden to avoid showing junk.\n\n**For today's data-driven picks "
                "right now, switch to the BTST (overnight) tab** — that engine runs off the "
                "confirmed last close and works any time.\n\n*(To exercise the live UI now, "
                "tick 🧪 Test mode in the sidebar.)*")
        st.stop()
    if not live.market_open() and test_mode:
        st.warning("🧪 **TEST MODE** — market is closed; data below is UNRELIABLE off-hours "
                   "Fyers (indicative prices, junk volume). Layout/flow testing only.")

    # ── HEADLINE INDEX STRIP ──────────────────────────────────────────────────────────
    # Deliberately AFTER the token and market-closed gates, not under the title: both of
    # those st.stop() the page, so rendering above them would put three blank tiles on a
    # screen whose actual problem is a dead token -- an unreadable quote and an unusable
    # session would look identical.
    #
    # ITS OWN 5s FRAGMENT so the prices stay live without re-running the whole page (a full
    # rerun re-scans the universe). live.index_quotes() widens its own memo to 60s when the
    # market is shut, so this does not spend the /quotes budget on off-hours junk.
    #
    # INTRADAY LANE ONLY, on purpose. BTST and Arbitrage are EOD engines that are documented
    # to work with a dead broker session; hanging live quotes off them would break that. And
    # Replay must never see a live price at all -- that is a lookahead, and sealing the
    # future is the entire point of that lane.
    @st.fragment(run_every="5s")
    def _render_index_strip():
        rows, stamp = live.index_quotes()
        for col, r in zip(st.columns(len(rows)), rows):
            if r["lp"] is None:
                # A MISSING QUOTE IS SAID OUT LOUD. Fyers returns lp=None for an unknown
                # index name rather than erroring, so a silent blank here would read as
                # "market closed" when it actually means the symbol is wrong.
                col.metric(r["name"], "—", help="Quote unavailable for this index right now.")
            else:
                col.metric(r["name"], f"{r['lp']:,.2f}",
                           None if r["chp"] is None else f"{r['chp']:+.2f}%")
        if stamp:
            st.caption(f"indices {stamp}" + ("" if live.market_open() else
                                             " · off-hours indicative, not a real print"))
    with _index_slot:                      # fills the container reserved above the title
        _render_index_strip()

    render_price_band()          # shared by BOTH lanes (structure scan + live snapshot)
    # ── VIEW TOGGLE — the old "Timeframe → stock list" dropdown is gone. Two lanes:
    #    • Live snapshot  = the validated 5s BTST accumulation board (unchanged, falls through).
    #    • Structure scan = the WHOLE liquid universe with its 6-TF structure, narrowed ONLY by
    #      the HTF/LTF structure filter. No day-move / close-strength pre-screen (Stage-1 gone).
    _, vcol = st.columns([2, 1])
    view = vcol.radio("View", ["Live snapshot", "🔬 Structure scan"], index=0, horizontal=True,
                      key="intraday_view",
                      help=(
                          "TWO LANES.\n\n"
                          "• LIVE SNAPSHOT — the validated lane. A 5s price-action scan for the "
                          "BTST accumulation footprint (delivery confirms at the close). This is "
                          "where BTST-CARRY lives.\n\n"
                          "• STRUCTURE SCAN — structure-first research. The ENTIRE F&O universe "
                          "(~270 names, NO turnover floor; the price band still applies) is pulled "
                          "on SIX timeframes, then YOU narrow it with the Higher-TF / Lower-TF "
                          "structure and delivery filters (e.g. BREAKOUT_UP on 4h + CONSOLIDATION "
                          "on 1h). No 1-day-bar pre-screen — the filters ARE the selection. Thin "
                          "names appear too; watch turn₹L for fill risk.\n\n"
                          "HONEST: the structure lane is intraday CONTEXT. Backtested −5bps at "
                          "every bar size; MTF alignment is unvalidated. The one validated trade "
                          "is BTST-CARRY on the 1-day bar at 15:25–15:29 → Live snapshot."))

    # ---- STRUCTURE-FIRST scan: full liquid universe, narrowed by the HTF/LTF filter ----
    if view == "🔬 Structure scan":
        # ── SCALPER MODE ────────────────────────────────────────────────────────────────
        # Top-right of this lane, above everything it changes. It is a LANE SWITCH, not a
        # fifth horizon preset, and it deliberately sits ABOVE the horizon dropdown rather
        # than beside the table: a control that reorders the widgets rendered above it reads
        # as a glitch, and scalper mode replaces the whole control block, not just the frames.
        #
        # WHY IT CANNOT BE A PRESET. At a five-minute hold the delivery columns, the F&O
        # block and `carry` all describe end-of-day or overnight state that cannot change
        # inside the trade, and every one of them costs an archive read. See eqbtst/scalp.py.
        _hd, _tg = st.columns([3, 1])
        _scalper = _tg.toggle(
            "⚡ **Scalper Mode**", value=False, key="scalper_mode",
            help=("Switch this lane to a **1-minute trigger inside a 5-minute box**, for a "
                  "1-5 minute hold (10 at the outside).\n\n"
                  "**What changes:** the delivery, F&O positioning, `carry` and sector-tilt "
                  "columns all disappear -- every one is an end-of-day or multi-week number "
                  "that cannot move inside a five-minute trade. They are replaced by expected "
                  "5-minute movement, YOUR round-trip cost, and whether the first can pay for "
                  "the second.\n\n"
                  "**The cost floor also changes**, and in your favour: an intraday square-off "
                  "pays ~5-12bps, not the 22bps delivery figure the rest of this board uses "
                  "(delivery STT is 0.1% on BOTH legs; intraday is 0.025% on the sell leg "
                  "only).\n\n"
                  "⚠ It is a **VETO, not a trigger**. Direction over five minutes measured "
                  "46.2% up across 457k observations -- the board will not give you a side."))
        if _scalper:
            scalp.render_page(st)
            st.stop()
            raise SystemExit   # st.stop() is a no-op outside a script-run context
        _hd.caption(
            "**Structure-first.** The **entire F&O universe** appears — **no 1-day-bar pre-screen, "
            "no turnover floor** (the price band above still applies). Set a **Higher-TF** and/or "
            "**Lower-TF** structure (and/or a delivery slider) below; only names where the filters "
            "hold survive, and those get levels / RSI / verdict on your **Lower TF**. "
            "⚠ **Watch `turn₹L`** (today's turnover, ₹lacs) — thin names now appear too, and a "
            "tape-reading level on a low-turnover name may be unfillable. 15m/1h/2h/4h include "
            "today's forming bar (can repaint); 1D/1W are from the EOD archive (leak-free).")
        st.session_state.setdefault("uni_nonce", 0)
        rsc1, rsc2 = st.columns([3, 1])
        auto_struct = rsc1.toggle(
            "Auto-refresh structure on each 15-min bar close", value=False, key="auto_struct",
            help="OFF (default): structure is a MANUAL snapshot — hit ↻ to re-pull; filtering stays "
                 "instant. ON: re-scans ONCE right after each :00/:15/:30/:45 close (the finest "
                 "frame). Structure only CHANGES on a bar close, so this keeps it fresh live with "
                 "the minimum scans (~4/hour) and never re-scans mid-bar (which would repeat the "
                 "same result for ~270 wasted fetches). Prices/verdict already tick every 5s "
                 "regardless.")
        if rsc2.button("↻ re-scan universe",
                       help="Force a fresh full-universe pull (~270 concurrent /history calls, "
                            "~30s). The scan is otherwise PINNED — moving a filter never re-scans, "
                            "so filtering stays instant."):
            st.session_state["uni_nonce"] += 1     # only this bumps the nonce → only this re-scans
            live._UNISCAN_CACHE.clear()            # also drop the module memo so it truly re-fetches
        try:
            with st.spinner("Scanning the full F&O universe on 6 timeframes (concurrent fetch, "
                            "~30s cold; instant once pinned)…"):
                sc = _uni_scan(st.session_state["uni_nonce"])
            st.session_state.pop("_uni_retried", None)      # success — clear the retry guard
        except Exception as _e:
            # DIAGNOSE THE CAUSE, DO NOT INFER IT FROM THE CLOCK. This branch used to decide
            # what to say purely from market_open(): open -> "transient, just hit re-scan";
            # closed -> "your token may be stale, re-auth". Those are exactly inverted for the
            # case that happens EVERY TRADING MORNING. The Fyers token expires ~06:00 IST, more
            # than three hours BEFORE the 09:15 open, so the normal first failure of the day is
            # a DEAD TOKEN while the market is OPEN — and the board would tell the user it was
            # transient and to keep pressing ↻, which can never fix it. Ask the token.
            _tok = live.token_status()
            _open = live.market_open()
            if not _tok["usable"]:
                st.session_state.pop("_uni_retried", None)
                st.error(
                    f"🔑 **The Fyers token is not usable — re-authenticate.** `{_tok['describe']}`"
                    "\n\nThe token expires around **06:00 IST every day**, which is before the "
                    "09:15 open, so this is the normal state of the first scan each morning. "
                    "**↻ re-scan will not fix it** — no number of retries will.\n\n"
                    "**Fix:** close this, run `run_dashboard.bat` (it checks the token and walks "
                    "you through re-auth), then come back. The BTST (overnight) tab still works "
                    "meanwhile — it runs off the confirmed last close and needs no token.")
                st.stop()
            # Token is fine, so an empty scan really can be transient (the first seconds of the
            # session, or a quote-fetch miss). Auto-retry ONCE. Guarded so it can never loop.
            if _open and not st.session_state.get("_uni_retried"):
                st.session_state["_uni_retried"] = True
                st.session_state["uni_nonce"] += 1
                live._UNISCAN_CACHE.clear()
                st.rerun()
            st.session_state.pop("_uni_retried", None)
            st.info(
                (f"Universe scan came back empty — {_e}. Token is valid and the **market is "
                 "OPEN**, so this is a transient quote-fetch miss (or the very first seconds of "
                 "the session). It is **not cached** — just hit **↻ re-scan universe**.")
                if _open else
                (f"Universe scan unavailable — {_e}. The token is valid, so this is the market "
                 "being closed / pre-open. The live scan runs Mon–Fri 09:15–15:30 IST."))
            st.stop()
        light = price_filter(sc["board"], "ltp")     # price band applies; no turnover floor
        _sa = sc.get("scanned_at")
        _age = (dt.datetime.now() - _sa).total_seconds() if _sa else 0

        # ── AUTO-REFRESH ON THE TRIGGER FRAME'S BAR CLOSE (opt-in) ───────────────────────
        # Structure changes only when a BAR closes, and the bar that matters is your TRIGGER
        # (Lower TF) — the frame you time the entry on. So re-scan on THAT frame's close, not
        # blindly every 15 min: Intraday (15m) at :00/:15/:30/:45; BTST (1h) at :15 past each
        # hour; Swing (4h) at 13:15 and the 15:30 close; Positional (1D) not at all intraday
        # (its daily bar only closes at 15:30, and it is archive-based -- it does not even see
        # today until the nightly sync). This matches how a chartist works -- wait for YOUR
        # candle to finish -- and stops the slower horizons from re-pulling the whole universe
        # four times an hour for a bar that has not moved. All bar closes fall on the 15-min
        # grid, so a bucket keyed to the trigger frame's period catches its close exactly.
        _LTF_MIN = {"15m": 15, "1h": 60, "2h": 120, "4h": 240, "1D": 1440, "1W": 10080}
        # The preset / lower-TF widgets are defined FURTHER DOWN the script, so read the
        # trigger frame from persisted session_state (their keys survive from the prior run;
        # first run falls back to the widget defaults: btst -> 1h). Never reference the
        # not-yet-executed widget vars here.
        _pp = mtf.PRESETS.get(st.session_state.get("mtf_preset", "btst"))
        _cl = st.session_state.get("mtf_ltf", "1h")
        _trig = _pp["ltf"] if _pp else (_cl if _cl in _LTF_MIN else "15m")

        def _bar_bucket(tf, t=None):
            t = t or dt.datetime.now()
            m = _LTF_MIN.get(tf, 15)
            if m >= 1440:                                  # daily+ -> one bucket per day (no intraday re-scan)
                return f"{t:%Y-%m-%d}"
            elapsed = (t.hour * 60 + t.minute) - (9 * 60 + 15)   # minutes since the 09:15 open
            return f"{t:%Y-%m-%d}:{max(0, elapsed) // m}"

        st.session_state["scanned_bucket"] = _bar_bucket(_trig, _sa) if _sa else _bar_bucket(_trig)
        if auto_struct:
            @st.fragment(run_every="20s")
            def _auto_rescan():
                if not live.market_open():
                    return
                if _bar_bucket(_trig) != st.session_state.get("scanned_bucket"):
                    st.session_state["scanned_bucket"] = _bar_bucket(_trig)
                    st.session_state["uni_nonce"] += 1     # force a fresh pull on the next run
                    live._UNISCAN_CACHE.clear()
                    st.rerun()
            _auto_rescan()

        # ARCHIVE STALENESS — the 1D/1W frames come from the EOD archive, and if the DCM
        # nightly sync has not ingested the latest session those frames are behind the market
        # WITHOUT the scan age showing it (the scan is fresh; the DATA under it is old). Fyers
        # is the independent calendar. Warn loudly, because a name that moved big on the
        # missing session shows a verdict built on pre-move data (MOTILALOFS -7.3% on 07-24
        # read WITH-TREND CONTINUATION LONG off the 07-23 archive, price already below its
        # 'support'). Cached once/day; never blocks the board if Fyers is unreachable.
        _stale = live.archive_staleness()
        if _stale.get("ok") and _stale["stale_days"] >= 1:
            st.error(
                f"⚠ **The EOD archive is {_stale['stale_days']} trading day(s) behind the "
                f"market.** It has **{_stale['archive_date']:%d-%b}** as its latest close, but "
                f"the market has traded through **{_stale['market_date']:%d-%b}**. So every "
                f"**1D / 1W structure, level and verdict on this page is that many sessions "
                f"old** — a name that moved hard on the missing session is judged on "
                f"pre-move data (live prices still tick; the STRUCTURE under them does not). "
                f"**Fix:** run the Daily_Cash_Market nightly sync to ingest the missing "
                f"session(s), then ↻ re-scan. The Intraday frames (15m–4h) come straight from "
                f"Fyers and are unaffected.")
        # AGE IS ALWAYS SHOWN. With no TTL the board holds its scan until you re-scan, so the
        # only thing that can mislead is not knowing how old it is. Off-hours this used to be
        # hidden entirely, which is when the board sits stalest.
        if _sa:
            _amin = int(_age // 60)
            _warn = _amin >= 30 and live.market_open() and not auto_struct
            st.caption(("⚠️ " if _warn else "🕒 ")
                       + f"structure as-of **{_sa:%H:%M:%S}** ({_amin}m {int(_age % 60)}s ago)"
                       + (" — prices tick live, but STRUCTURE and LEVELS are this old. "
                          "↻ re-scan universe." if _warn else
                          " · filters re-apply instantly; ↻ re-scan universe to re-pull bars."))
        st.caption(f"scanned **{sc['n_scanned']}** liquid names · Nifty "
                   f"{sc.get('idx_ret', 0):+.2f}% · regime "
                   f"{'RISK-ON' if sc['risk_on'] else 'RISK-OFF'}"
                   + ("  ·  🔄 auto-refresh ON (next 15-min close)" if auto_struct else ""))
        # A name with no intraday candles cannot match ANY intraday structure filter — it just
        # disappears. Say how many, so the board is never silently answering from a subset.
        # A failed /quotes chunk removes FIFTY names before a single row is built, so
        # n_scanned already excludes them and the board looks complete at a smaller size.
        _qg = sc.get("n_quote_gap", 0)
        if _qg:
            st.error(f"⚠ **{_qg} names are MISSING from this scan entirely** — their quote "
                     f"batch failed twice (broker timeout or rate-limit). They are not `—` "
                     f"rows; no row exists for them at all, so every count and filter on this "
                     f"page is answering from a smaller universe. ↻ re-scan.")
        _nb = sc.get("n_blank_intraday", 0)
        if _nb:
            st.caption(f"⚠ **{_nb}** of {sc['n_scanned']} names have no intraday candles (broker "
                       "rate-limit or no data) — they show `—` on 15m/1h/2h/4h and CANNOT match "
                       "an intraday structure filter. Their 1D/1W are unaffected. ↻ refresh to "
                       "retry them.")

        # ── DELIVERY-CONVICTION FILTERS — ported from the DCM sector-rotation view ──
        # Same two sliders you use there, same formulas (turnover-weighted delivery). Default 0
        # = show everything (this lane's premise is 'all names appear, THEN you filter'); raise
        # them to demand real accumulation. These narrow the list BEFORE the per-name enrich, so
        # they also cut the fetch cost.
        dc1, dc2 = st.columns(2)
        min_wtd = dc1.slider(
            "Min stock Wtd Delivery % — filters the list below", 0, 100, 0, step=1,
            key="eqbtst_min_wtd",
            help="Hide names whose 7-day turnover-weighted delivery % is BELOW this. DCM uses 48 "
                 "as its default cut for 'genuine accumulation'; 0 here = show all.")
        min_vs = dc2.slider(
            "Min 7D vs 100D excess % — filters the list below", 0, 100, 0, step=5,
            key="eqbtst_min_vs",
            help="Hide names whose recent delivery is not at least this % ABOVE their own 100-day "
                 "norm. 0 = show all; +10 = only names delivering ≥10% above their own baseline "
                 "(conviction accelerating). Names with no 100D history are dropped when >0.")

        def _deliv_filter(df_):
            d_ = df_
            if min_wtd > 0 and "wtd_deliv7" in d_.columns:
                d_ = d_[d_["wtd_deliv7"] >= min_wtd]          # NaN >= n is False → dropped, correct
            if min_vs > 0 and "deliv_vs_100d" in d_.columns:
                d_ = d_[d_["deliv_vs_100d"] >= min_vs]
            return d_

        # ── MULTI-TIMEFRAME STRUCTURE FILTER — now THE selection mechanism ──
        with st.expander("❓ How structure is computed — the 20-bar window (read this)"):
            st.markdown(
                "**Every label reads the LAST 20 BARS of that frame** — not the whole history. "
                "So \"20 bars\" is a different amount of calendar time on each one:\n\n"
                "| Frame | Bars/day | 20 bars ≈ | Where the bars come from |\n"
                "|---|---|---|---|\n"
                "| **15m** | 25 | **~1 trading day** | Fyers, 60 calendar days of 15m candles |\n"
                "| **1h** | 6 | **~3 trading days** | same 15m candles, joined into 1h |\n"
                "| **2h** | 3 | **~7 trading days** | ″ |\n"
                "| **4h** | 2 | **~10 trading days** | ″ |\n"
                "| **1D** | 1 | **~20 trading days (~1 month)** | EOD archive (last close) |\n"
                "| **1W** | 1/wk | **~20 weeks (~5 months)** | EOD archive, last COMPLETE week |\n\n"
                "*One 15m fetch per name feeds the top four frames — 1h/2h/4h are built from it, "
                "not fetched separately.*\n\n"
                "**Why 1h is 6 bars/day and not 7:** NSE trades 09:15–15:30 = 375 minutes, which "
                "60 does not divide. The leftover 15:15–15:30 would be a 15-minute bar wearing a "
                "1h label, so it is folded into the 14:15 bar (which then spans 14:15–15:30). "
                "Same on 2h. Measured, this changed the structure label on **a third of names** — "
                "a quarter-length bar is a different animal with the same name.\n\n"
                "**1D and 1W do NOT see today.** They come from the end-of-day archive, so during "
                "a live session they are as-of the last close. The weekly also drops a "
                "part-formed week — a weekly breakout is not one until the week closes.\n\n"
                "**Why only 20 bars, when we have 8+ years?** Deliberate — structure = the "
                "**current regime, not ancient history**:\n"
                "- 🚀 **Breakout ↑** = today's close above the *prior 19 bars'* highest high. Over "
                "20 bars that means a fresh ~N-bar high. Over *thousands* of bars it would mean an "
                "*all-time* high → almost never fires, useless.\n"
                "- **Efficiency ratio** (trend detection) washes to ~0 over huge windows — every "
                "stock would read ↔️ Range.\n\n"
                "20 is the standard Kaufman window: long enough to define a range, short enough to "
                "react. The 8 years of data feed the **validated overnight edge** + delivery "
                "baselines — *not* the tape-reading structure label.\n\n"
                "**How each label is decided (by SIZE of move, not just topology):**\n"
                "- 🚀 **Breakout ↑ / 💥 Breakdown ↓** — close clears the prior 19-bar high/low "
                "by **≥ 0.5×ATR** (a real break, ~1–1.5% beyond the range on a daily frame — not "
                "a marginal poke). ATR = the frame's own volatility, so the rule stays sensible "
                "on 15m *and* 1W.\n"
                "- 📈 **Uptrend / 📉 Downtrend** — Kaufman efficiency **ER ≥ 0.40** *and* the "
                "net move covers **≥ 1×ATR** — an efficient, real directional move (not a tiny "
                "drift that happens to be smooth).\n"
                "- 🌀 **Coiling** — the latest **3-bar span** is **< 60%** of the *typical* 3-bar "
                "span in the window (the median of all of them). In plain words: *the last 3 "
                "bars covered less than 60% of the ground this name normally covers in 3 bars.* "
                "It is a **contraction**, measured against the name's own habit — which is why "
                "one rule works on a ₹50 stock and a ₹5,000 stock, on 15m and on 1W. Note it is "
                "3 bars vs the TYPICAL 3 bars, never 3 bars vs the other 17: comparing a short "
                "window to a long one fires on almost anything.\n"
                "- ↔️ **Range** — none of the above: drifting sideways, **not** tightening. "
                "Sideways is not coiled, and this is the most common label on any given day.\n\n"
                "**During market hours the coil test ignores the bar still forming.** A coil is "
                "a finished observation — you cannot say the range is contracting from a bar "
                "five minutes old, and since the window keeps the last 20 bars, a new one pushes "
                "a full-width bar out of the comparison. Measured: coil fired on 7% of the board "
                "at 12:20 and **30% at 13:20**, purely because the 13:15 bar had just opened with "
                "one candle in it. Breakout and trend still use the live bar — catching a break "
                "as it happens is the point of a live board.\n\n"
                "*(All four thresholds are tunable in config: STRUCT_BREAKOUT_ATR, "
                "STRUCT_TREND_ER, STRUCT_TREND_ATR, STRUCT_COIL. Want stronger, ~3% breakouts? "
                "raise STRUCT_BREAKOUT_ATR toward 1.0.)*\n\n"
                "**A combo reading 0 matches is normal, not broken.** 4h 🚀 Breakout ↑ together "
                "with 1h 🌀 Coiling is genuinely rare — it needs a breakout AND a pause caught in "
                "the same snapshot. Loosen one leg (4h 📈 Uptrend, or Lower TF = Any) to populate.")
        _TF_RANK = {"15m": 15, "1h": 60, "2h": 120, "4h": 240, "1D": 1440, "1W": 10080}
        _NONE = "— none —"                     # disable this whole TF leg (TF + its structure)
        _STRUCTS = ["Any", "BREAKOUT_UP", "TREND_UP", "CONSOLIDATION", "RANGE",
                    "TREND_DOWN", "BREAKOUT_DOWN"]
        # ── TRADE HORIZON PRESET ──────────────────────────────────────────────────────
        # The LTF/HTF pair is not a taste, it is the HOLD PERIOD. Each preset nests a trigger
        # frame inside a confirmation frame ~4x coarser (the classical ratio: fine enough to
        # time an entry, coarse enough for the context to mean something). Picking a horizon
        # sets BOTH frames at once and turns on the HTFxLTF setup read.
        _PRE_OPTS = ["custom"] + mtf.PRESET_ORDER
        _pre_lbl = {"custom": "⚙ Custom — pick both frames myself",
                    **{k: v["label"] for k, v in mtf.PRESETS.items()}}
        pc1, pc2, pc3 = st.columns([3, 2, 2])
        preset = pc1.selectbox("Trade horizon (sets both timeframes)", _PRE_OPTS,
                               index=_PRE_OPTS.index("btst"), key="mtf_preset",
                               format_func=lambda k: _pre_lbl[k],
                               help=(
                                   "**Pick the HOLD, and the timeframes follow.**\n\n"
                                   "A pair is only useful if the confirmation bar closes inside "
                                   "your holding period — a weekly bar cannot inform a trade you "
                                   "exit at 15:20. Each preset nests the trigger frame inside a "
                                   "confirmation frame about 4× coarser.\n\n"
                                   "| Horizon | Trigger (LTF) | Confirm (HTF) | Hold |\n"
                                   "|---|---|---|---|\n"
                                   "| Intraday | 15m | 1h | same day |\n"
                                   "| BTST | 1h | 4h | buy today → sell next day |\n"
                                   "| Swing | 4h | 1D | 2–10 sessions |\n"
                                   "| Positional | 1D | 1W | weeks |\n\n"
                                   "The higher frame gives the **box and the trend**; the lower "
                                   "frame gives the **trigger**. Where price sits inside that box "
                                   "(`loc`) is what separates a real break from a trap.\n\n"
                                   "⚠ Longer horizons stand on firmer ground — not because the "
                                   "structure is better, but because the ~22bps round-trip cost "
                                   "eats a far smaller share of a multi-day move than of a "
                                   "30-minute one. The intraday hunt in this stack is closed."))
        _P = mtf.PRESETS.get(preset)
        _hz = preset            # active trade horizon — scales the delivery baseline (_dw)
        if _P and preset != "intraday" and tf == "Intraday":
            # NAME COLLISION: the sidebar radio picks WHICH BOARD you are on (validated
            # BTST-carry board / this structure lane / replay); this dropdown picks the
            # ANALYSIS HORIZON inside this lane. Both use the word "BTST", so the sidebar can
            # read "Intraday" while the horizon reads "BTST" and look self-contradictory.
            st.caption("ℹ️ Sidebar **Intraday** = which BOARD you are on (this structure lane). "
                       f"**{_P['label'].split('·')[0].strip()}** above = the HORIZON this lane "
                       "is analysing. Different questions: the sidebar is the page, the "
                       "dropdown is the hold.")
        _setup_f = "All"
        _room_f = "All"
        if _P:
            _setup_f = pc2.selectbox(
                "Setup quality", ["All", "🎯 Textbook only", "🟢 Long-side setups",
                                  "🪤 Exclude traps"], index=0, key="mtf_setupf",
                help=("Keep only certain kinds of setup.\n\n"
                      "• **🎯 Textbook only** — the cleanest one: the big picture is trending, "
                      "and the stock has paused and gone quiet, resting before it likely "
                      "continues the same way.\n"
                      "• **🟢 Long-side** — everything pointing UP (continuation, a real "
                      "breakout, or a dip inside an uptrend).\n"
                      "• **🪤 Exclude traps** — hides the fake breakouts (a pop in the middle of "
                      "the range that usually fades).\n\n"
                      "Either way, the best setups are sorted to the top."))
            _room_f = pc3.selectbox(
                "Upper-TF S/R (one frame up)", ["All", "✅ Has room", "⚠ Tight", "🧱 Capped"],
                index=0, key="mtf_roomf",
                help=("Filter on the ONE HIGHER frame's wall (the `big wall` column) — the level "
                      "your two trading frames are blind to. Per horizon that frame is: Intraday "
                      "→ 4h, BTST → 1D, Swing → 1W, Positional → 1M. 'Room' is always in the "
                      "TRADE'S direction: for a LONG it is clear of the RESISTANCE above (room "
                      "to rise); for a SHORT it is clear of the SUPPORT below (room to fall).\n\n"
                      "• **✅ Has room** — the higher-frame wall in the trade's direction is **∞ "
                      "clear** or **≥ 1 ATR away**, so the 1×ATR target has space. This is the "
                      "'horizon setup AND the upper frame is not blocking it' list.\n"
                      "• **⚠ Tight** — **0.5 to 1.0 ATR**: the wall sits ON your 1×ATR target. "
                      "These used to belong to NEITHER option, so flipping between 'room' and "
                      "'capped' never showed them at all (3.4–8.5% of names, worst on Swing).\n"
                      "• **🧱 Capped** — the opposite: **< 0.5 ATR** from a higher-frame "
                      "wall in the trade's direction — most likely to stall or reverse into the "
                      "big level (AVOID, or wait for a clean break of it — a break THROUGH is "
                      "often the real move, so capped is not automatically 'goes the other way').\n\n"
                      "**WHAT COUNTS AS A WALL** (three fixes, 2026-08-04 — the read used to miss "
                      "the level on ~62–73% of names, now ~2–6%):\n"
                      "• a level touched **once** still counts (`×1`). It used to need two touches, "
                      "so a single violent rejection read '∞ clear' with the level plainly on the "
                      "chart. Touch count does NOT rank levels (measured), so it must not gate them.\n"
                      "• a **`~`** suffix means the level is the high/low of the frame's last two "
                      "bars. A pivot needs two bars either side of it, so the most recent two bars "
                      "can never BE a level — two WEEKS on 1W, two MONTHS on 1M. That blind spot "
                      "is exactly where 'price is testing the high right now' lives. Closed bars "
                      "only, so it cannot repaint.\n"
                      "• when price sits INSIDE a cluster of pivots, the wall is quoted at the "
                      "cluster's near EDGE, not its average — an average can fall on the wrong "
                      "side of price and turn a ceiling into a floor.\n\n"
                      "⚠ MEASURED — a CHARTIST screen, NOT a return edge. Backtested on 1,010 "
                      "footprint longs with a causal weekly-wall read: 'has room' longs earned "
                      "+7.5bps overnight vs +21.7 for 'capped' ones — room did NOT beat capped "
                      "(t=-0.66, not significant), if anything the reverse, consistent with 'a "
                      "break THROUGH the big level is the move'. So 'has room' does not predict "
                      "the stock rises; it only tells you the higher frame is not blocking it. "
                      "SHORT side same (43,042 down-structure days, causal weekly-FLOOR read): "
                      "room-to-fall did NOT beat capped (t=0.41), and EVERY class loses as a "
                      "short (room -14.5bps, capped -16.3) — the short side has no edge, room or "
                      "not. Use this to SEE the level and size the trade, not to pick winners. "
                      "The validated trade here is the overnight BTST carry (delivery "
                      "footprint), not these levels.\n\n"
                      "Composes with Setup quality: 🟢 Long-side + ✅ Has room = longs with a "
                      "clear higher frame."))
            st.caption(f"📐 **{_P['label']}** · hold: *{_P['hold']}* — {_P['note']}")
            # HOW PROVISIONAL IS THIS TAG? A forming trigger bar can relabel until it closes,
            # and the board never said so. Measured per preset (see mtf.REPAINT) rather than
            # hand-waved, because the answer differs by an order of magnitude across the four.
            _rp = mtf.REPAINT.get(preset or "", {})
            # THE BANNER MUST KEY OFF WHAT THE TRIGGER FRAME ACTUALLY IS, NOT off midday_differs==0.
            # Only a 1D/1W trigger is archive-based and truly fixed for the session. Swing's trigger
            # is 4h -- a LIVE intraday resample of today's 15m fetch: it sees today and repaints
            # until the 4h bar closes (13:15, 15:30). Routing it through the FIXED text (as
            # midday_differs==0 did) told the user the 4h bar was "a closed daily bar from the
            # archive" that "does NOT see today" -- all three claims false. Three honest cases:
            _arch_trigger = _P["ltf"] in ("1D", "1W")           # archive-based, cannot repaint
            if _arch_trigger:
                st.caption(
                    f"🔒 **This read is FIXED for the session** — the {_P['ltf']} trigger bar "
                    f"is a closed archive bar, so nothing here can repaint intraday. It also "
                    f"means it does NOT see today's session; only `ltp`, `loc` and the live "
                    f"levels move.")
            elif _rp and _rp.get("midday_differs"):
                st.caption(
                    f"🔄 **This read is PROVISIONAL — the {_P['ltf']} trigger bar is still "
                    f"printing.** Replaying full sessions: a name passes through "
                    f"**{_rp['tags_per_session']} different setup tags** in one session, the "
                    f"midday tag differs from the closing tag **{_rp['midday_differs']}% of "
                    f"the time**, and the label only settles after **{_rp['settles_pct']}% of "
                    f"the session has elapsed**. Treat it as a running commentary, not a "
                    f"decision — and remember each change of mind is another ~22bps round trip.")
            else:
                # Live INTRADAY trigger that repaints little (Swing's 4h: only ~2 bars form per
                # day). Honest middle: it is live and sees today, updates when the trigger bar
                # closes, and its forming bar can still shift the read -- but it settles fast.
                st.caption(
                    f"🟡 **This read is LIVE but slow-moving** — the {_P['ltf']} trigger is an "
                    f"intraday bar, so it DOES see today and updates when each {_P['ltf']} bar "
                    f"closes (e.g. 4h at 13:15 and 15:30). The forming bar can still shift the "
                    f"tag, but only ~2 bars print per session, so it settles far faster than the "
                    f"15m/1h triggers. Not archive-fixed; not fast-repainting either.")
        with st.expander("🧭 Higher TF vs Lower TF — the whole idea in 30 seconds"):
            st.markdown(
                "**Think of it like a map.**\n\n"
                "- **Higher TF = zoomed OUT.** The big picture — which way the stock is really "
                "going. Slow to change, but it is the direction that matters. *(This is the "
                "\"confirm\" frame.)*\n"
                "- **Lower TF = zoomed IN.** The close-up — what price is doing right now. Fast, "
                "noisy, but it is where you actually time the entry. *(This is the \"trigger\" "
                "frame.)*\n\n"
                "**Why you need both.** One frame alone lies to you. A stock can look like it is "
                "breaking out on the close-up while the big picture shows it is just wiggling in "
                "the middle of a range — a fake move. The higher frame tells you *whether the "
                "little move even means anything.*\n\n"
                "**The one number that ties them together — `loc`.** It says WHERE price sits "
                "inside the big-picture range: **0.0 = at the bottom, 1.0 = at the top.** A "
                "breakout only counts at the top (loc near 1); a breakdown only at the bottom "
                "(loc near 0). A \"break\" in the middle is the trap.\n\n"
                "**You do not have to pick these** — the **Trade Horizon** dropdown above sets "
                "both for you (BTST = 1h close-up inside a 4h big-picture). These boxes are for "
                "when you want to choose the pair yourself.\n\n"
                "*Golden rule: the Lower TF must be SMALLER than the Higher TF. Zoom in, never out.*")
        fc1, fc2, fc3, fc4 = st.columns(4)
        f_htf = fc1.selectbox("Higher TF", [_NONE, "1h", "2h", "4h", "1D", "1W"], index=3,
                              key="mtf_htf", disabled=bool(_P),
                              help=(
                                  "**Higher TF = the big picture (zoomed OUT).**\n\n"
                                  "Pick the bigger timeframe you want to judge the overall "
                                  "direction on. It changes slowly and matters more — this is the "
                                  "CONTEXT the smaller frame gets read inside.\n\n"
                                  "• **1h / 2h / 4h** — built from recent days' candles, including "
                                  "the bar forming now, so they can still CHANGE until it closes.\n"
                                  "• **1D / 1W** — from the end-of-day archive. Rock-solid, never "
                                  "change during the day — but they don't see today yet.\n\n"
                                  "This box picks the frame; the next box picks the SHAPE you want "
                                  "on it. **'— none —'** turns the Higher-TF filter off."))
        f_hst = fc2.selectbox("HTF structure", _STRUCTS, index=0, key="mtf_hst",
                              format_func=_struct_label,
                              help=(
                                  "**What SHAPE must the Higher TF be in?** Keep only stocks whose "
                                  "big-frame structure matches this. `Any` = don't filter on the "
                                  "big frame.\n\n"
                                  "The six shapes — defined by the SIZE of the move (ATR = the "
                                  "frame's own volatility unit, so the % auto-scales 15m→1W):\n"
                                  "• 🚀 **Breakout ↑** — closed ABOVE its last-20-bar high by a "
                                  "REAL margin (≥0.5×ATR ≈ 1–1.5% beyond the range on a daily "
                                  "frame) — not a 0.1% poke\n"
                                  "• 📈 **Uptrend** — already climbing efficiently (Kaufman "
                                  "ER≥0.4) AND covered ≥1×ATR of ground — momentum in progress, "
                                  "no fresh break needed\n"
                                  "• 🌀 **Coiling** — range TIGHTENING (recent 3 bars < 60% of "
                                  "the prior range) — volatility contracting, energy building\n"
                                  "• ↔️ **Range** — oscillating sideways in its band: no "
                                  "efficient direction, no break, not tightening\n"
                                  "• 📉 **Downtrend** — efficient fall (mirror of Uptrend)\n"
                                  "• 💥 **Breakdown ↓** — closed BELOW its 20-bar low by "
                                  "≥0.5×ATR (mirror of Breakout)"))
        f_ltf = fc3.selectbox("Lower TF", [_NONE, "15m", "1h", "2h", "4h", "1D"], index=2,
                              key="mtf_ltf", disabled=bool(_P),
                              help=(
                                  "**Lower TF = the close-up (zoomed IN).**\n\n"
                                  "Pick the smaller timeframe where you actually TIME the entry, "
                                  "inside the big frame's context. Same chart, zoomed in.\n\n"
                                  "It does two jobs:\n"
                                  "1. Filters on the close-up SHAPE (next box).\n"
                                  "2. Your **entry / stop / target levels are built on THIS "
                                  "frame** — smaller frame means tighter levels.\n\n"
                                  "Keep it SMALLER than the Higher TF (e.g. Higher 4h, Lower 1h). "
                                  "Zoom in, never out.\n\n"
                                  "**'— none —'** turns the filter off; the levels then fall back "
                                  "to your Higher TF."))
        f_lst = fc4.selectbox("LTF structure", _STRUCTS, index=0, key="mtf_lst",
                              format_func=_struct_label,
                              help=(
                                  "**What SHAPE must the Lower TF be in?** Keep only stocks whose "
                                  "small-frame structure matches this. `Any` = don't filter on the "
                                  "small frame. (Same six shapes as HTF structure.)\n\n"
                                  "**The whole idea — a worked example:**\n"
                                  "Higher TF **4h = 🚀 Breakout ↑** + Lower TF **1h = 🌀 Coiling** "
                                  "→ the stock BROKE OUT on the big frame and is now COILING on the "
                                  "small one: it made its move, then paused to gather energy — a "
                                  "classic continuation setup. A name shows ONLY if BOTH boxes "
                                  "match.\n\n"
                                  "⚠ This is tape-reading CONTEXT, not a proven edge — intraday "
                                  "alignment is unvalidated. The validated trade is BTST-CARRY."))
        # A leg is ON only if BOTH its TF is a real frame (not '— none —') AND its structure is a
        # shape (not 'Any'). Either one off = that leg does not filter.
        # A preset OVERRIDES the two frame boxes (they render disabled, showing the pair). The
        # SHAPE boxes stay live — they are an optional extra cut on top of the setup read.
        if _P:
            f_htf, f_ltf = _P["htf"], _P["ltf"]
        _htf_on = (f_htf != _NONE) and (f_hst != "Any")
        _ltf_on = (f_ltf != _NONE) and (f_lst != "Any")
        # LEVELS/RSI/verdict need a REAL frame even if the Lower TF is '— none —': fall back to
        # the Higher TF, then to 1h. (The Lower TF is both a filter leg AND the entry frame.)
        levels_tf = f_ltf if f_ltf != _NONE else (f_htf if f_htf != _NONE else "1h")

        if (_htf_on and _ltf_on and _TF_RANK[f_ltf] >= _TF_RANK[f_htf]):
            st.warning(f"⚠ your 'lower' TF ({f_ltf}) is not below your 'higher' TF ({f_htf}) — "
                       "the filter still applies, but the HTF/LTF logic is inverted.")

        # STATUS LINE — a TF pick does NOTHING until its structure box is a shape (and the TF
        # itself is not '— none —'). Say so, so a selected frame next to 'Any' never reads as an
        # active-but-empty filter.
        if not (_htf_on or _ltf_on or min_wtd > 0 or min_vs > 0):
            st.caption("⚪ **No filter active.** Showing all names. A timeframe filters only once "
                       "BOTH its **frame** (not '— none —') AND its **structure** (not 'Any') are "
                       f"set. When you do filter, **{levels_tf}** builds the levels/RSI/verdict.")
        else:
            _bits = []
            if _htf_on:
                _bits.append(f"Higher **{f_htf}** = {_struct_label(f_hst)}")
            if _ltf_on:
                _bits.append(f"Lower **{f_ltf}** = {_struct_label(f_lst)}")
            if min_wtd > 0:
                _bits.append(f"wtdDeliv ≥ **{min_wtd}%**")
            if min_vs > 0:
                _bits.append(f"vs100D ≥ **{min_vs}%**")
            st.caption("🟢 **Active filter:** " + "  ·  ".join(_bits)
                       + f"  →  levels/verdict on **{levels_tf}**.")

        def _mtf_filter(df_):
            d_ = df_
            if _htf_on and f"s{f_htf}" in d_.columns:
                d_ = d_[d_[f"s{f_htf}"] == f_hst]
            if _ltf_on and f"s{f_ltf}" in d_.columns:
                d_ = d_[d_[f"s{f_ltf}"] == f_lst]
            return d_

        # SETUP READ — free (pure arithmetic on boxes the scan already carried). Applied to the
        # WHOLE universe so the default view is already sorted best-context-first.
        _setup_on = False
        _room_on = False
        _n_setup = None
        _census = None
        if _P:
            light = live.add_setup(light, ltf=_P["ltf"], htf=_P["htf"])
            # CENSUS OF THE WHOLE TAPE — from the FULL SCAN, not from `light`. Taken after the
            # setup filter it once reported "1 setup type across 7 names", which describes your
            # filter and not the market; that was fixed. But `light` has ALREADY been cut by the
            # PRICE BAND (applied the moment the board is loaded), so with a Rs900 cap the census
            # read "6 setup types across all 106 scanned names" while 243 were scanned — it
            # silently described 44% of the market and called that "all". A price cap is a
            # position-SIZING choice; it must never decide what the tape is doing. The census is
            # therefore built from sc["board"] directly. Cost is nil — add_setup is arithmetic
            # over boxes the scan already carried.
            _census = live.add_setup(sc["board"], ltf=_P["ltf"], htf=_P["htf"])[
                ["setup", "setup_read", "turn₹L", "symbol", "dir"]].copy()
            if _setup_f == "🎯 Textbook only":
                light, _setup_on = light[light["setup"] == "WITH-TREND CONTINUATION"], True
            elif _setup_f == "🟢 Long-side setups":
                # DIRECTION, not tag: a continuation/pullback tag reads identically in a
                # downtrend, so filtering on the tag alone served short setups as longs.
                light, _setup_on = light[light["side"] == "LONG"], True
            elif _setup_f == "🔴 Short-side setups":
                light, _setup_on = light[light["side"] == "SHORT"], True
            elif _setup_f == "🪤 Exclude traps":
                light, _setup_on = light[~light["setup"].isin(mtf.AVOID_TAGS)], True
            # UPPER-TF S/R FILTER — on the big-wall (one frame up), the level the pair is blind
            # to. big_gap is still NUMERIC here (float, inf for clear); _fmt stringifies it only
            # at render. "Room" = the higher frame is not capping the trade (clear or >=1 ATR);
            # "Capped" = a defended higher-frame wall sits <0.5 ATR in the trade's direction.
            # A SEPARATE flag from setup quality so the funnel can attribute each cut honestly:
            # picking "Has room" with Setup quality = All must NOT read as a "setup" cut.
            _n_setup = len(light)                       # count AFTER setup quality, BEFORE room
            if _room_f != "All" and "big_gap" in light.columns:
                _bg = pd.to_numeric(light["big_gap"], errors="coerce")
                if _room_f == "✅ Has room":
                    light, _room_on = light[np.isinf(_bg) | (_bg >= 1.0)], True
                elif _room_f == "⚠ Tight":
                    # 0.5-1.0 ATR. These names belonged to NEITHER of the old two options, so
                    # flipping between them never showed them (measured 3.4-8.5% of the universe,
                    # worst on Swing). Not a third opinion -- the missing third of the range.
                    light, _room_on = light[(_bg >= 0.5) & (_bg < 1.0)], True
                elif _room_f == "🧱 Capped":
                    light, _room_on = light[_bg < 0.5], True
            light = light.sort_values(["setup_rank", "turn₹L"], ascending=[True, False])

        after_deliv = _deliv_filter(light)          # stage the chain so each cut is VISIBLE
        filtered = _mtf_filter(after_deliv)
        active = _htf_on or _ltf_on or _setup_on or _room_on or (min_wtd > 0) or (min_vs > 0)
        # `deliv 5wk` sits immediately after `side`: once you know WHICH WAY a setup points, the
        # next question is whether anyone is actually accumulating into it, and that read is
        # useless twenty columns to the right. The per-side evidence note (`long/short evidence`)
        # takes its old slot beside the other delivery columns — it is a long text field, so it
        # scans poorly in the middle of numeric columns and costs nothing sitting further out.
        # `turn₹L` sits LAST. It is a fill-risk CHECK, not something you scan row by row: you
        # consult it once you have a candidate, to ask whether the level you just read is
        # actually tradeable in that name. Holding a prime column for it pushed the reads you
        # do scan further right. NOTE it is still load-bearing — the universe carries no
        # turnover floor, so thin names appear, and the caption above the table keeps saying so.
        # F&O positioning sits IMMEDIATELY AFTER the delivery read, by request and because the
        # two answer the same question from opposite books: delivery is who took stock in the
        # CASH market, near/next futures+options is what the DERIVATIVES book did in the same
        # name on the same day. Twenty columns apart they would never be compared.
        # `side` SITS LAST, with turn₹L. The board splits into LONG / SHORT / No-side TABS,
        # so within any table every row carries the same value — the column is constant and
        # tells you what the tab heading already said, while occupying a prime slot and
        # pushing the delivery and F&O reads toward the right edge. Kept (it confirms the
        # split, and the no-preset table has no tabs) but demoted.
        light_cols = (["symbol", "sector", "sector tilt", "ltp", "day%"]
                      + (["setup", "deliv trend"] + fno.COLS + ["carry"] +
                         ["loc", "at_wall", "sup", "sup_t", "res",
                          "res_t", "headroom", "big_wall", "big_gap"]
                         if _P else ["deliv trend"] + fno.COLS + ["carry"])
                      + ["wtd_deliv7", "deliv_vs_100d",
                         "s15m", "s1h", "s2h", "s4h", "s1D", "s1W", "turn₹L", "side"])

        # WHAT THE TAPE LOOKS LIKE RIGHT NOW — the census of setups across the whole universe,
        # with the full read for each. A tag in a cell is a label; this is what it MEANS.
        if _P and _census is not None and not _census.empty:
            # GROUP BY (tag, DIRECTION), not tag alone. A directional tag means opposite things
            # up vs down: WITH-TREND CONTINUATION ↑ is a bull flag (LONG), ↓ is a bear flag
            # (SHORT); COIL AT THE EXTREME ↑ is a flag at the highs, ↓ a base at the lows. The
            # old census lumped them under one heading and showed ONE read — so a group of 28
            # 'COIL AT THE EXTREME' could be a mix of longs and shorts with the read of whichever
            # sorted first. Split them, arrow the header, and show each direction's own read.
            _arrow = {"UP": " ↑", "DOWN": " ↓"}
            _key = _census["setup"] + _census["dir"].map(lambda d: _arrow.get(d, ""))
            _vc = _key.value_counts()
            _band_cut = len(_census) - len(light) if not _setup_on else None
            with st.expander(f"🔭 What the {_P['ltf']} × {_P['htf']} tape says right now — "
                             f"{len(_vc)} setup types across all {len(_census)} scanned names"
                             + (f"  (incl. {_band_cut} your price band hides)"
                                if _band_cut else "")):
                for _lbl, _cnt in _vc.items():
                    _r = _census[_key == _lbl]
                    _tag = _r["setup"].iloc[0]
                    _ex = ", ".join(_r.nlargest(min(4, len(_r)), "turn₹L")["symbol"])
                    st.markdown(f"**{mtf.TAG_ICON.get(_tag, '')} {_lbl}** — {_cnt} names  \n"
                                f"{_r['setup_read'].iloc[0]}  \n"
                                f"*most liquid:* {_ex}")
                st.caption("⚠ Setup quality is a CHARTIST ranking, not an expected return. "
                           "Intraday multi-TF alignment has no validated edge in this stack; the "
                           "one validated trade here is the overnight BTST carry, which is "
                           "selected by delivery + close-strength, not by these shapes.")

        if not active:
            _cut = sc["n_scanned"] - len(light)
            st.info(
                (f"**All {len(light)} scanned names** are in the tabs below, split by what the "
                 f"**{_P['ltf']} × {_P['htf']}** read says about each one."
                 if not _cut else
                 f"**{len(light)} of {sc['n_scanned']} scanned names** shown — your **price "
                 f"band is cutting {_cut}** (Max ₹ = 0 shows the whole universe). "
                 f"MEASURED: a price cap is not a neutral cut on this edge — over 8 years of "
                 f"footprint triggers the overnight payoff falls MONOTONICALLY with price "
                 f"(≤₹900: **+33.8bps**, n=358 · >₹900: **+7.8bps**, n=390 · diff t=3.04), "
                 f"cheap beat rich in **8 of 8 years**, and it is not a liquidity artifact — "
                 f"the gap is WIDEST in the most-traded third. Keeping a cap here is a "
                 f"selection choice that has paid.")
                + " Raise a **delivery** slider or pick a **structure** to narrow further; "
                  "levels, RSI and a verdict are added on your Lower TF once you filter.")
            _cfg = {**LIVE_COLS, **TF_COLS, **DELIV_COLS, **SETUP_COLS, **SR_COLS, **FNO_COLS, **ARB_COLS}
            # NAME THE ACTIVE BASELINE IN THE HEADER. It now moves with the trade horizon, so a
            # bare "deliv trend" would leave the reader guessing which yardstick the % is against
            # — and the answer changes the sign on real names (KALYANKJIL reads +24% on a 15-day
            # base and −7% on a 60-day one).
            _bd = live.DELIV_BASE_BY_HORIZON.get(_hz or "btst", live._DELIV_WK_NORM)
            _bk = live.DELIV_BUCKET_BY_HORIZON.get(_hz or "btst", "week")
            # help MUST come from the shared constant. Reading it back off the config object
            # (`DELIV_COLS[...].help`) silently returned None -- st.column_config returns a
            # plain DICT, so the attribute never existed, hasattr was False, and the column
            # lost its tooltip entirely while still rendering perfectly.
            _cfg["deliv trend"] = st.column_config.TextColumn(
                f"deliv {'5d' if _bk == 'day' else '5wk'} · base {_bd}d",
                width="medium", help=DELIV_HELP_SHORT)
            render_tilt_help()
            render_deliv_help()
            # WHICH SESSION THE F&O COLUMNS DESCRIBE. The bhavcopy publishes after the close, so
            # on the Intraday tab these sit beside a LIVE price while describing yesterday's
            # book. Saying the date is the difference between context and a silent lie.
            _fmeta = fno.positioning(_ASOF_LIVE)[1]
            if _fmeta.get("stale_days"):
                st.warning("🧮 " + fno.stale_note(_fmeta))
            else:
                st.caption("🧮 " + fno.stale_note(_fmeta))

            def _side_table(df_, note=None, extra_cols=(), timed=False):
                if df_.empty:
                    st.caption("No name is on this side right now. That is a reading of the "
                               "tape, not an error — in a one-way market one side empties.")
                    return
                if note:
                    st.warning(note)
                _c = list(light_cols)
                # `entered` ON THE LONG TAB ONLY. universe_mtf_scan is a STRUCTURE scan --
                # it never evaluates the BTST footprint, so it ships no trigger time and this
                # table had no `entered` column at all while every other board had one
                # (reported from a screenshot). The trigger needs today's 5m bars, one fetch
                # per name, so it is computed per-tab rather than for all ~250 scanned: the
                # full fan-out trips the same rate limit that already produces the
                # "unreadable, not neutral" rows.
                #
                # NOT ON THE SHORT TAB, and this is not a cost decision -- the column CANNOT
                # fill there. `entered` is the ACCUMULATION footprint trigger and every leg
                # is long-directional: day_ret >= RET_TH (+1% UP on the day), close-in-range
                # >= CLR_TH, close >= CVWAP_TH above session VWAP, cumulative RS > RS_MIN,
                # volume on pace for VOL_TH x normal. A name the structure calls SHORT is
                # down on the day, so `day_ret >= +1%` is false at every bar and the trigger
                # is None by construction. Shipped that way once: three columns of "—" on the
                # SHORT tab, bought with ~17 network fetches per 5-minute bucket.
                # There is also nothing to time. The short side's own banner says it: +4.4bps
                # against a ~22bps round trip, "a weakness screen, not an entry signal". A
                # short ENTRY time would be timing a trade this board says not to take.
                if timed == "SHORT":
                    # `entered` HERE MEANS: the bar TODAY at which this became a SHORT --
                    # structure recomputed at each of today's lower-frame bar closes and put
                    # through the same synthesize()/side_of() the live board runs. It is NOT
                    # the accumulation footprint (that is long-only by construction, which is
                    # why this column was blank), and it is NOT a fixed-box price replay
                    # (that stamps a fake 09:15 on the structure-driven majority).
                    # Blank = the flip predates today; `dn age` carries that duration.
                    df_ = live.add_short_entry_times(df_, _P["htf"], _P["ltf"],
                                                    symbols=list(df_["symbol"]))
                    if "ltp" in _c:
                        _base = _c.index("ltp")
                        for _i, _e in enumerate(("entered", "at", "since%")):
                            if _e in df_.columns and _e not in _c:
                                _c.insert(_base + _i, _e)
                    # `dn age` is the COMPANION to that time, not a replacement. Most shorts
                    # turned before today, so `entered` is legitimately blank for them -- the
                    # side is decided by the higher frame's structure ENUM, which changes on a
                    # BAR CLOSE, often days back. The age answers those rows: sessions the
                    # DAILY structure has been bearish. Free (the EOD archive is already
                    # warmed for the 1D/1W reads and back-adjusted), works with a dead broker
                    # token, and reaches past the broker's ~60-day intraday limit, which
                    # truncates exactly the names that have been bearish longest.
                    # Together: `entered` = it happened TODAY at HH:MM; `dn age` = it has been
                    # broken for N sessions. Blank in both = short on the intraday frames only.
                    df_ = live.add_downtrend_age(df_, symbols=list(df_["symbol"]))
                    if "ltp" in _c and "dn_age" not in _c:
                        _c.insert(_c.index("ltp"), "dn_age")
                elif timed:
                    df_ = live.add_trigger_times(df_, symbols=list(df_["symbol"]))
                    # sits immediately BEFORE ltp -- when-then-what. `at`/`since%` ride along
                    # so the entry price lands next to the live one.
                    # ANCHOR ONCE. Re-reading _c.index("ltp") inside the loop walks the
                    # anchor right as each insert shifts it, which lands `at`/`since%` AFTER
                    # ltp instead of before it.
                    if "ltp" in _c:
                        _base = _c.index("ltp")
                        for _i, _e in enumerate(("entered", "at", "since%")):
                            if _e in df_.columns and _e not in _c:
                                _c.insert(_base + _i, _e)
                for _e in extra_cols:                 # per-side columns (e.g. the short verdict)
                    if _e in df_.columns and _e not in _c:
                        # LANDS WITH THE DELIVERY BLOCK, not beside `side`. It used to sit right
                        # after the side column, which pushed the level columns out and put a
                        # wide sentence in the middle of the numbers you scan. `deliv 5wk` holds
                        # that slot now; the evidence note reads fine further right.
                        _anchor = next((c for c in ("deliv_vs_100d", "wtd_deliv7", "side")
                                        if c in _c), None)
                        _c.insert(_c.index(_anchor) + 1 if _anchor else 1, _e)
                df_ = arb.annotate(fno.annotate(_wt(_dw(df_, _ASOF_LIVE, _hz), _ASOF_LIVE), _ASOF_LIVE), _ASOF_LIVE)
                st.dataframe(_fmt(df_)[_cols(df_, _c)], use_container_width=True,
                             hide_index=True,
                             column_config={**_cfg, "dn_age": ARB_DN_AGE})
                # Denominator = the pool the SIDES were split from, not the raw scan. Quoting
                # the scan made "3 of 243 (1%)" appear under tabs that summed to 106, implying
                # the split ran over the whole universe when a price band had already cut it.
                _tally(len(df_), len(light), "names",
                       f"of {sc['n_scanned']} scanned" if len(light) != sc["n_scanned"] else "")

            # ── LONG / SHORT split by the setup's OWN direction, not by its tag ──────────
            # Only shown when a horizon preset is active, because without one there is no
            # HTF x LTF read to take a side from — and inventing a side from a single
            # timeframe label is how a downtrend gets bought.
            if _P and "side" in light.columns:
                # ── PRICE TICKS HERE TOO, 5s, WHILE STRUCTURE STAYS PINNED ────────────────
                # These tabs are what the board opens on, and they had no fragment at all: the
                # ltp column was frozen at scan time while its own tooltip promised "LIVE,
                # refreshed every 5s on every tab". One batch quote re-prices the whole board
                # (243 names = 5 /quotes requests, not 243 /history), then add_setup re-derives
                # everything that is a FUNCTION of price — loc, the setup tag, the side, and the
                # nearest support/resistance/headroom. The 20-bar boxes, wall lists and ATRs are
                # untouched: those only change when a BAR closes, which is what ↻ is for.
                @st.fragment(run_every="5s")
                def _side_tabs(base=light):
                    light_live = base
                    if live.market_open():
                        _re = live.refresh_light_prices(base)
                        light_live = live.add_setup(_re, ltf=_P["ltf"], htf=_P["htf"])
                        st.caption(f"💹 prices live ({dt.datetime.now():%H:%M:%S}) · structure & "
                                   f"walls pinned at {_sa:%H:%M}" if _sa else "💹 prices live")
                    light = light_live
                    # THE LONG SIDE IS RE-SORTED BY HOLD TOO. Measured: RANGE-TOP BREAK is the
                    # best OVERNIGHT long (+32.1bps excess, 9 of 9 years) and the WORST 5-day one
                    # (-24.7, and negative in absolute terms). WITH-TREND CONTINUATION is its
                    # mirror. TAG_RANK ranks continuation first, which is right for Swing and
                    # Positional and backwards for BTST.
                    _lo = light[light["side"] == "LONG"].copy()
                    if not _lo.empty:
                        _lo["long_note"] = [mtf.long_verdict(t, preset) for t in _lo["setup"]]
                        _lo["_lrank"] = [mtf.long_rank(t, preset) for t in _lo["setup"]]
                        _lo = _lo.sort_values(["_lrank", "turn₹L"], ascending=[True, False])
                    _no = light[light["side"] == "—"]
                    # THE SHORT SIDE IS RE-SORTED BY WHAT WAS MEASURED, NOT BY TEXTBOOK QUALITY.
                    # setup_rank is a chartist ranking; on the short side it is inverted (see
                    # mtf.SHORT_RANK). Sorting the SHORT tab by it put the tag that measured
                    # +0.47% AGAINST a short at the top of every list, and the one tag that
                    # worked at the bottom.
                    _sh = light[light["side"] == "SHORT"].copy()
                    if not _sh.empty:
                        _sh["short_note"] = [mtf.short_verdict(t, preset) for t in _sh["setup"]]
                        _sh["_srank"] = [mtf.short_rank(t, preset) for t in _sh["setup"]]
                        _sh = _sh.sort_values(["_srank", "turn₹L"], ascending=[True, False])
                    tL, tS, tN = st.tabs([f"🟢 LONG ({len(_lo)})", f"🔴 SHORT ({len(_sh)})",
                                          f"⚪ No side ({len(_no)})"])
                    with tL:
                        st.caption("Setups pointing UP: an uptrend coiling for continuation, a break "
                                   "at the top of the higher-TF range, or a pullback INTO an uptrend.")
                        _side_table(_lo, extra_cols=("long_note",), timed=True)
                    with tS:
                        _cash_ok = mtf.SHORTABLE_IN_CASH.get(preset, False)
                        _edge = mtf.SHORT_EDGE_BPS.get(preset, 0.0)
                        if _cash_ok:
                            _side_table(_sh, extra_cols=("short_note",), timed="SHORT", note=(
                                "⚠ **Intraday only — square off before the close.** Measured on this "
                                "universe (43,042 down-structure days, 2018–2026): a same-day short "
                                f"earns **{_edge:+.1f}bps** before the ~22bps round-trip cost. That is "
                                "the BEST case on this board and it is still under the cost floor — "
                                "a weakness screen, not an entry signal."))
                        else:
                            _side_table(_sh, extra_cols=("short_note",), timed="SHORT", note=(
                                f"🛑 **This horizon cannot hold a short — twice over.**\n\n"
                                f"**1. Mechanically:** Indian cash equity has no overnight short — it "
                                f"must be squared off the SAME DAY. The *{_P['hold']}* hold is "
                                f"unreachable in cash. These are all F&O names so a stock FUTURE "
                                f"exists, but that is a different instrument: margin, lot size, "
                                f"expiry and rollover.\n\n"
                                f"**2. Economically — the part that matters more:** measured over "
                                f"43,042 down-structure days, a short held to this horizon earns "
                                f"**{_edge:+.1f}bps BEFORE costs.** The downtrend is real but it is an "
                                f"INTRADAY move (−4.4bps in-session); overnight the same names gap "
                                f"**+10.8bps AGAINST a short**, and only 32.5% of nights gap down at "
                                f"all. The overnight gap that IS the long edge here is a nightly toll "
                                f"for a short — with a tail that runs +438bps in the worst 1%.\n\n"
                                f"Read this list as **names to AVOID or EXIT**, never to short."))
                        # PER-TAG EVIDENCE. The horizon warning above is about the INSTRUMENT;
                        # this is about the SETUP itself, and it is the more damaging result.
                        if not _sh.empty:
                            _anti = _sh[_sh["setup"].isin(mtf.SHORT_ANTI_PREDICTIVE)]
                            _ok = _sh[_sh["setup"].isin(mtf.SHORT_VALIDATED)]
                            st.error(
                                f"🔬 **{len(_anti)} of these {len(_sh)} names sit on a setup measured "
                                f"ANTI-PREDICTIVE on the short side.** Reconstructing this exact "
                                f"pipeline over 468,661 observations (2018–2026), shorts "
                                f"OUTPERFORMED the universe by +0.57% over 20 days — the side is "
                                f"inverted, not merely weak.\n\n"
                                f"Per setup (excess vs market; **positive = the short LOST**):\n"
                                f"• ⚠️ EXTENDED (aligned) **+1.62%** — an extended downtrend is an "
                                f"OVERSOLD name; shorting it is selling the low\n"
                                f"• ↩️ PULLBACK vs HTF **+0.50%**\n"
                                f"• 🎯 WITH-TREND CONTINUATION **+0.47%** — a downtrend that COILS is "
                                f"a base forming, not a continuation\n"
                                f"• 🔻 RANGE-FLOOR BREAK **−1.09%** — the only short setup that "
                                f"worked ({len(_ok)} here now)\n\n"
                                f"In a structurally rising market, most 'bearish' structure is a "
                                f"bottoming pattern.")
                        st.caption("Short P&L by hold, before the ~22bps cost floor (measured, "
                                   "n=43,042): intraday **+4.4bps** · overnight **−5.1** · 5-day "
                                   "**−13.6**. It decays monotonically with hold length — the exact "
                                   "opposite of the long side, where a longer horizon amortises cost.")
                    with tN:
                        # A DATA GAP IS NOT A MARKET READING. Names whose intraday candles failed
                        # come back "HTF warming" and would otherwise sit here indistinguishable
                        # from names the tape genuinely has nothing to say about. Measured: 34 of
                        # 243 on the intraday-frame presets, and 0 on positional — the archive
                        # feeds 1D/1W, so the gap is frame-specific, not name-specific.
                        _warm = int(_no["setup"].astype(str).str.contains("warming").sum())
                        st.caption("Squeezes, traps and sideways names — the setup takes no side. "
                                   "Most of the universe lives here most of the time, and that is "
                                   "the honest default: no trade.")
                        if _warm:
                            st.info(f"⏳ **{_warm} of these {len(_no)} are UNREADABLE, not neutral** "
                                    f"— no candles came back for the {_P['ltf']}/{_P['htf']} frames "
                                    f"(broker rate-limit), so they could not be judged either way. "
                                    f"↻ refresh retries them. The **Positional** horizon reads "
                                    f"1D/1W from the EOD archive and can judge all of them.")
                        _side_table(_no)
                _side_tabs()
            else:
                _lt = arb.annotate(fno.annotate(_wt(_dw(light, _ASOF_LIVE, _hz), _ASOF_LIVE), _ASOF_LIVE), _ASOF_LIVE)
                st.dataframe(_fmt(_lt)[_cols(_lt, light_cols)], use_container_width=True,
                             hide_index=True, column_config=_cfg)
                _tally(len(light), sc["n_scanned"], "names",
                       "no filter active" if len(light) == sc["n_scanned"]
                       else "price band is the only cut")
            st.stop()

        # ── FILTER FUNNEL — show WHERE names drop, so a 0 is diagnosable (which stage killed
        # it?), not a mystery. Only stages you actually engaged appear.
        _funnel = [f"scanned **{sc['n_scanned']}**"]
        if _setup_on:
            _funnel.append(f"setup ({_setup_f}) → **{_n_setup if _n_setup is not None else len(light)}**")
        if _room_on:
            _funnel.append(f"upper-TF ({_room_f}) → **{len(light)}**")
        if (st.session_state.get("price_max") or 0) or (st.session_state.get("price_min") or 0):
            _funnel.append(f"price band → **{len(light)}**")
        if min_wtd > 0 or min_vs > 0:
            _funnel.append(f"delivery (wtd≥{min_wtd} · vs100D≥{min_vs}) → **{len(after_deliv)}**")
        if _htf_on or _ltf_on:
            _legs = " · ".join(
                ([f"HTF {f_htf}={_struct_label(f_hst)}"] if _htf_on else [])
                + ([f"LTF {f_ltf}={_struct_label(f_lst)}"] if _ltf_on else []))
            _funnel.append(f"structure ({_legs}) → **{len(filtered)}**")
        st.caption("🔎 funnel:  " + "  →  ".join(_funnel))
        if filtered.empty:
            st.info("No name survives this combination right now. That is an answer, not an error "
                    "— read the funnel above to see WHICH stage emptied it, then loosen that leg "
                    "(a slider to 0, or a structure box to 'Any').")
            st.stop()

        # CAP the per-name enrich (cost bound). Sort by TURNOVER, not day% — day%-desc kept the
        # top gainers and would drop the very names a Downtrend/Breakdown filter is looking for
        # (they have LOW day%). Liquidity-first keeps the most FILLABLE matches, any direction.
        _MAXE = 60
        _cap_key = "turn₹L" if "turn₹L" in filtered.columns else "day%"
        capped = filtered.sort_values(_cap_key, ascending=False).head(_MAXE)
        if len(filtered) > _MAXE:
            st.caption(f"⚠ {len(filtered)} matches — reading the top **{_MAXE}** by turnover "
                       "(most fillable) for levels/verdict. Tighten a leg to see the rest.")
        with st.spinner(f"Reading {len(capped)} matches on {levels_tf} bars for levels & verdict…"):
            enr = live.enrich_mtf(capped, ltf=levels_tf, risk_on=sc["risk_on"],
                                  idx_ret=sc.get("idx_ret", 0.0))
        enr = price_filter(enr, "ltp")
        # The F&O labels are EOD and do not move on a 5s price tick, so they are attached ONCE
        # here rather than inside the refresh fragment -- everything downstream (including
        # live.refresh_prices) inherits them.
        enr = arb.annotate(fno.annotate(enr, _ASOF_LIVE), _ASOF_LIVE)
        if enr.empty:
            st.info("Matches found, but none could be read on the Lower TF (thin candles). "
                    "Try a coarser Lower TF.")
            st.stop()

        # `deliv trend` sits after `side` here TOO. The pre-filter table already placed it
        # there; leaving the enriched lists with it out in the delivery block meant the
        # column JUMPED across the table the moment a filter was applied — same board,
        # same row, different position depending on whether you had narrowed it.
        # THE F&O BLOCK TRAVELS WITH IT, for the same reason and by the same mistake: added
        # to the pre-filter list only, the four columns VANISHED the moment any filter was
        # applied (reported on "Has room"). Any column added to one list must be added to
        # both, or the board silently shows a different shape once you narrow it.
        _sc = (["setup", "deliv trend"] + fno.COLS + ["carry"] +
               ["loc", "at_wall", "sup", "sup_t", "res",
                "res_t", "headroom", "big_wall", "big_gap"]) if _P else []

        def _day_by_setup(cols):
            # ltp + day% belong BESIDE the setup, not buried after the risk columns: you read the
            # shape, the price, and the day's move together, and day% is a validated footprint leg
            # (the signal wants day_ret >= +1%). Relocates them right after `setup` when a preset
            # is active; with no setup column (custom TF) they stay where they were.
            #
            # `sector` + `sector tilt` RIDE ALONG, immediately after day%, because that is the
            # question day% raises: this name moved — is its whole SECTOR moving with it, or is it
            # alone? Left in declaration order the pair landed at column ~20, off the right edge of
            # the table, which for a context column is the same as not existing. The two travel
            # TOGETHER so the badge is never orphaned from the sector it describes: "🔴 UW #19/24"
            # with no sector name beside it is a verdict about nothing you can see.
            # `entered` LEADS the relocated block. It is declared before `ltp` in both lists,
            # but this very function used to hop ltp/day%/sector over it -- which left the fire
            # TIME stranded past the whole S/R block while the price it fired against sat at
            # column 3. Reported from a screenshot. Reading order is when-then-what: a name
            # that triggered at 10:55 is a different read from one that triggered at 09:30,
            # and that question comes BEFORE the price. Rows that never fired show "—", so it
            # costs one narrow column.
            if "setup" in cols:
                move = [c for c in ("entered", "ltp", "day%", "sector", "sector tilt")
                        if c in cols]
                cols = [c for c in cols if c not in move]
                i = cols.index("setup") + 1
                cols = cols[:i] + move + cols[i:]
            return cols

        long_cols = _day_by_setup(["symbol", *_sc, "entered", "at", "since%", "time", "bar", "sector", "sector tilt", "ltp",
                     "day%", "wtd_deliv7", "deliv_vs_100d",
                     "s15m", "s1h", "s2h", "s4h", "s1D", "s1W",
                     "bar_clr", "character", "vs_vwap%", "rsi7", "rsi14", "tone", "RS%",
                     "entry", "stop", "t1", "t2", "atr%", "action", "turn₹L", "side"])
        sell_cols_tf = _day_by_setup(["symbol", *_sc, "entered", "at", "since%", "time", "bar", "sector", "sector tilt", "ltp",
                        "day%", "wtd_deliv7", "deliv_vs_100d",
                        "s15m", "s1h", "s2h", "s4h", "s1D", "s1W",
                        "bar_clr", "character", "vs_vwap%", "rsi7", "rsi14", "tone", "RS%",
                        "entry", "s_stop", "s_t1", "s_t2", "atr%", "sell", "turn₹L", "side"])

        @st.fragment(run_every="5s")
        def _struct_panel():
            bb = live.refresh_prices(enr, risk_on=sc["risk_on"]) if live.market_open() else enr
            st.caption(f"💹 prices live ({dt.datetime.now():%H:%M:%S}) · structure/levels as-of "
                       f"{_sa:%H:%M}" if (_sa and live.market_open())
                       else "market closed — last-session values")
            # SPLIT BY THE SETUP SIDE, consistent with the filter chain and the pre-filter tabs
            # (the user selected these names by side + room, which are SIDE concepts). `action`
            # / `sell` — the intraday FOOTPRINT verdict — stay as columns, so you see whether the
            # footprint agrees with the MTF side. The old split was by `action` and, when no
            # footprint fired (every off-hours session), fell back to dumping ALL matches under
            # LONG — which is why NESTED SQUEEZE / RANGE-BOUND (no-side) rows appeared as longs.
            _has_side = "side" in bb.columns
            tb, tsh = st.tabs([f"🟢 LONG ({levels_tf} bars)", f"🔴 SHORT ({levels_tf} bars)"])
            with tb:
                lo = bb[bb["side"] == "LONG"] if _has_side else bb[bb["action"] == "LONG"]
                if lo.empty:
                    st.caption("No LONG-side setup among the matches. That is a reading of the "
                               "tape, not an error — loosen a filter to see more.")
                else:
                    lo = _wt(_dw(lo, _ASOF_LIVE, _hz), _ASOF_LIVE, "LONG")
                    st.dataframe(_fmt(lo)[_cols(lo, long_cols)], use_container_width=True,
                                 hide_index=True, column_config={**LIVE_COLS, **TF_COLS, **DELIV_COLS, **SETUP_COLS, **SR_COLS, **FNO_COLS, **ARB_COLS})
                    _tally(len(lo), sc["n_scanned"], "names",
                           f"{len(filtered)} matched the filter · {len(bb)} read on {levels_tf}")
            with tsh:
                st.warning("⚠ **Intraday short only — SQUARE OFF BEFORE THE CLOSE.** Overnight "
                           "short is proven -EV (win 20%); intraday direction has no validated "
                           "edge either. Weakness screen, not alpha — trade small, manage by s_stop.")
                sh = (bb[bb["side"] == "SHORT"] if _has_side
                      else bb[bb["sell"].isin(["SHORT", "WEAK"])].sort_values("sell"))
                if sh.empty:
                    st.caption("No SHORT-side setup among the matches. A reading of the tape, "
                               "not an error.")
                else:
                    sh = _wt(_dw(sh, _ASOF_LIVE, _hz), _ASOF_LIVE, "SHORT")
                    st.dataframe(_fmt(sh)[_cols(sh, sell_cols_tf)], use_container_width=True,
                                 hide_index=True,
                                 column_config={**LIVE_COLS, **TF_COLS, **DELIV_COLS, **SETUP_COLS, **SR_COLS, **SELL_COLS, **FNO_COLS, **ARB_COLS})
                    _tally(len(sh), sc["n_scanned"], "names",
                           f"{len(filtered)} matched the filter · {len(bb)} read on {levels_tf}")

        _struct_panel()
        st.caption("Entry/Stop/T1/T2 = ATR risk geometry on your Lower TF, NOT a forecast. "
                   "Structure-first is a research lens; the validated trade is BTST-CARRY.")
        st.stop()

    # ---- live snapshot (5s price-action scan) — Buy / Sell tabs ---------------
    st.caption("Live price-action scan, refreshing every 5s. **BUY** = accumulation "
               "footprint (validated overnight edge; delivery confirms at close). **SELL** "
               "= intraday distribution/weakness (square off same day — overnight short is "
               "proven -EV).")

    @st.fragment(run_every="5s")
    def _live_panel():
        # THE LOCK GUARD HAS TO LIVE HERE TOO. The one at the top of the script only covers
        # page LOAD; this fragment re-runs every 5 seconds and touches the archive on each
        # tick (quotes_board -> last_trading_date). A writer that grabs the DuckDB mid-session
        # therefore blew past that guard and rendered a raw traceback with an "Ask ChatGPT"
        # button — the same failure the top-of-script handler exists to prevent, just later.
        try:
            lb = _live_board()
        except Exception as e:
            st.error(f"⚠ **Archive unavailable right now.** {e}")
            st.caption("This board only READS the archive. The live prices above are "
                       "unaffected — it is the EOD-derived fields (prev close, volume "
                       "baseline, ATR, delivery) that need it. It clears on its own the "
                       "moment the other process lets go; this panel retries every 5s.")
            return
        if not lb["ok"] or lb["board"].empty:
            st.warning("No live quotes right now (market closed / pre-open). Scan "
                       "populates 09:15–15:30.")
            return
        if not lb.get("market_open", True):
            st.warning("⚠ **MARKET CLOSED** (outside Mon–Fri 09:15–15:30 IST). This is the "
                       "**last-session snapshot** — `day%`, `Nifty today`, `character` are "
                       "STALE, and `vol×` is blank (no volume today). Live values build "
                       "during the session. For actionable picks now, use the **BTST tab**.")
        # ARCHIVE-STALENESS GUARD — if our EOD baseline is not the broker's baseline, then
        # prev_close / vol_med20 / atr14 / deliv_trail are from the WRONG session and every
        # signal below is corrupted. This must be impossible to miss.
        _ah = lb.get("archive") or {}
        if _ah.get("stale"):
            st.error(
                f"🛑 **STALE EOD ARCHIVE — DO NOT TRADE THIS BOARD.** "
                f"{_ah['mismatch']}/{_ah['checked']} names ({_ah['pct']}%) disagree with the "
                "broker's previous close. The nightly DCM sync has not run for the last "
                "session, so `prev_close`, `vol×` baseline, `ATR` and `delivTr` are all from "
                "the WRONG day — every signal here is corrupted. Run the DCM EOD sync, then "
                "hit ↻ refresh.")
        bd = price_filter(lb["board"], "ltp")
        # Annotate ONCE on the parent frame so every sub-table (carry / forming / short)
        # inherits the columns. Attaching per-table is how the structure lane ended up with
        # the block in one list and not the other.
        bd = arb.annotate(fno.annotate(bd, _ASOF_LIVE), _ASOF_LIVE)
        counts = bd["action"].value_counts().to_dict()
        scounts = bd["sell"].value_counts().to_dict()
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Regime", "RISK-ON" if lb["risk_on"] else "RISK-OFF")
        h2.metric("Nifty today", f"{lb.get('idx_ret', 0):+.2f}%")
        h3.metric("BUY (carry/forming)",
                  f"{counts.get('BTST-CARRY', 0) + counts.get('FORMING', 0)}")
        h4.metric("SELL (short/weak)",
                  f"{scounts.get('SHORT', 0) + scounts.get('WEAK', 0)}")
        render_tilt_help()

        # F&O block follows the DELIVERY column where there is one (`delivTr` here), and the
        # sector tilt otherwise — the short list carries no delivery read, so the two
        # DCM-sourced context blocks sit together instead.
        buy_cols = ["symbol", "entered", "at", "since%", "time", "sector", "sector tilt", "ltp", "est_close", "day%", "clr", "character", "vol×",
                    "RS%", "rsCum%", "cvwap%", "delivTr"] + fno.COLS + ["carry"] + ["turn₹L", "btst", "book", "wt%", "exp_ON", "band_lo", "band_hi",
                    "entry", "stop", "t1", "t2", "risk%", "atr%", "action"]
        sell_cols = ["symbol", "entered", "at", "since%", "time", "sector", "sector tilt"] + fno.COLS + ["carry"] + ["ltp", "day%", "clr", "character", "vol×",
                     "RS%", "short", "entry", "s_stop", "s_t1", "s_t2", "atr%", "sell"]
        t_buy, t_sell = st.tabs(["🟢 BUY (long — validated overnight edge)",
                                 "🔴 SELL (intraday short — square off same day)"])
        with t_buy:
            carry = bd[bd["action"] == "BTST-CARRY"].sort_values("btst", ascending=False)
            forming = bd[bd["action"] == "FORMING"].sort_values("btst", ascending=False)
            # 🌙 BTST-CARRY — its own prominent box: THIS is the overnight signal at ~15:15
            st.markdown("#### 🌙 BTST-CARRY — hold overnight")
            if carry.empty:
                st.caption("None yet. Names appear here in the **15:10–15:30** window when a "
                           "full footprint holds into the close. **These are the overnight "
                           "picks** — enter LONG near close, exit next morning.")
            else:
                st.success(f"{len(carry)} name(s) ready to carry overnight — act 15:15–15:30, "
                           "exit next 09:15–09:30.")
                carry = _wt(carry, _ASOF_LIVE, "LONG")
                st.dataframe(_fmt(carry)[_cols(carry, buy_cols)], use_container_width=True, hide_index=True,
                             column_config={**LIVE_COLS, **FNO_COLS, **ARB_COLS})
            # ⏳ FORMING — watch list, may flip to CARRY near the close
            st.markdown("#### ⏳ FORMING — building (watch)")
            if forming.empty:
                b = _wt(bd.sort_values("clr", ascending=False).head(10), _ASOF_LIVE, "LONG")
                st.caption("No footprint building yet — top-10 by close-strength meanwhile:")
                st.dataframe(_fmt(b)[_cols(b, buy_cols)], use_container_width=True, hide_index=True,
                             column_config={**LIVE_COLS, **FNO_COLS, **ARB_COLS})
            else:
                st.caption(f"{len(forming)} building — may flip to 🌙 BTST-CARRY near the close.")
                forming = _wt(forming, _ASOF_LIVE, "LONG")
                st.dataframe(_fmt(forming)[_cols(forming, buy_cols)], use_container_width=True, hide_index=True,
                             column_config={**LIVE_COLS, **FNO_COLS, **ARB_COLS})
        with t_sell:
            st.warning("⚠ **Intraday short only — SQUARE OFF BEFORE THE CLOSE.** Holding "
                       "these short OVERNIGHT is proven -EV (net −42bps, win 20% on 8yr — "
                       "weak names bounce overnight). Intraday direction has no validated "
                       "edge either; this is a price-action *weakness* screen, not alpha. "
                       "Trade small, manage by s_stop.")
            s = bd[bd["sell"].isin(["SHORT", "WEAK"])].sort_values("short", ascending=False)
            if s.empty:
                st.caption("No distribution/weakness names live right now.")
            else:
                s = _wt(s, _ASOF_LIVE, "SHORT")
                st.dataframe(_fmt(s)[_cols(s, sell_cols)], use_container_width=True, hide_index=True,
                             column_config={**LIVE_COLS, **SELL_COLS, **FNO_COLS, **ARB_COLS})
        st.caption(f"updated {dt.datetime.now():%H:%M:%S} • hover any header for what+why. "
                   "VWAP · RSI7/14 · tone appear in the table when you pick a 1h/2h/15m "
                   "timeframe above.")
    _live_panel()
    st.stop()
if tf == "🎬 Replay (practice)":
    st.title("🎬 Replay / Practice — the board at any past time")
    with st.expander("❓ What is this — backtest & practice, causally (no lookahead)"):
        st.markdown(
            "Freeze the live board at any **past date + time**. Only data up to that "
            "minute is used — **no lookahead**, so it's honest practice. At **1pm** you see "
            "⏳ FORMING names; move the time to **15:15** to see which became 🌙 **BTST-CARRY**. "
            "This is how you learn the FORMING→CARRY flow before risking money.\n\n"
            "First load of a date is slow (fetches ~250 names' candles); after that, "
            "scrubbing the time is instant. Delivery% is still EOD — replay shows the "
            "price-action state, the BTST tab is the delivery-confirmed truth.")
    tok = live.token_status()
    st.caption(tok["describe"])
    if not tok["usable"]:
        st.error("Fyers token not usable — re-auth in Tradebot (`python fyers_auth.py`).")
        st.stop()
    rc1, rc2, rc3 = st.columns([1, 2, 1])
    rdate = rc1.date_input("Date", value=_last_date().date(), max_value=_last_date().date(),
                           key="replay_date")
    rtime = rc2.select_slider("Time (IST)",
                              options=[f"{h:02d}:{m:02d}" for h in range(9, 16)
                                       for m in (0, 15, 30, 45)
                                       if (h, m) >= (9, 15) and (h, m) <= (15, 30)],
                              value="13:00", key="replay_time")
    rtf = rc3.selectbox("Timeframe", ["15m", "1h", "2h", "4h"], index=0, key="replay_tf",
                        help="Candle timeframe the VWAP/RSI/structure read is computed on. "
                             "15m = native fine bars; 1h/2h/4h resample the causal bars up "
                             "to your cut. Coarse tf = fewer bars early in the day (structure "
                             "may read n/a until enough bars form).")

    @st.cache_data(ttl=3600)
    def _replay(dstr, t, tf_):
        return live.replay_board(dstr, t, tf=tf_)

    with st.spinner(f"Reconstructing the board as of {rdate} {rtime} ({rtf}; first load of a day "
                    "fetches ~250 names)…"):
        rb = _replay(pd.Timestamp(rdate).strftime("%Y-%m-%d"), rtime, rtf)
    if not rb["ok"] or rb["board"].empty:
        st.info("No data for that date/time (not a trading day, or candles unavailable). "
                "Try a recent trading day.")
        st.stop()
    render_price_band()
    bd = price_filter(rb["board"], "ltp")
    near_close = rtime >= "15:10"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("As of", f"{rb['date']} {rtime}")
    m2.metric("Regime", "RISK-ON" if rb["risk_on"] else "RISK-OFF")
    m3.metric("Nifty so far", f"{rb.get('idx_ret') or 0:+.2f}%")
    carry_n = int((bd["action"] == "BTST-CARRY").sum())
    form_n = int((bd["action"] == "FORMING").sum())
    m4.metric("🌙 CARRY / ⏳ FORMING", f"{carry_n} / {form_n}")
    if not near_close:
        st.info(f"It's **{rtime}** — before the 15:10 window, so names show as ⏳ **FORMING** "
                "(building). Move the slider to **15:15** to see which flip to 🌙 BTST-CARRY.")
    render_tilt_help()
    buy_cols = ["symbol", "entered", "at", "since%", "time", "sector", "sector tilt"] + fno.COLS + ["carry"] + ["ltp", "day%", "structure", "clr", "character",
                "vs_vwap%", "rsi7", "rsi14", "tone", "vol×", "RS%", "rsCum%", "cvwap%", "btst", "entry",
                "stop", "t1", "t2", "atr%", "action"]
    sell_cols_r = ["symbol", "entered", "at", "since%", "time", "sector", "sector tilt"] + fno.COLS + ["carry"] + ["ltp", "day%", "structure", "clr", "character",
                   "vs_vwap%", "rsi7", "rsi14", "tone", "vol×", "RS%", "entry",
                   "s_stop", "s_t1", "s_t2", "atr%", "sell"]
    # THE REPLAY AS-OF IS THE DAY BEFORE, NOT THE REPLAYED DAY. A replay reconstructs a
    # decision taken INSIDE session `rdate`, which had not closed at that moment — reading
    # that session's own close would feed its outcome back into the decision. This is the
    # same class of lookahead that retracted two "edges" in this stack, so it is resolved by
    # the module, not by hand.
    # NOTE the fallback is None, NOT rdate. If the prior close cannot be resolved, the honest
    # outcome is NO tilt column ("— tilt unavailable"); falling back to rdate would quietly
    # substitute the one value this whole comment exists to forbid.
    _asof_replay = sector_tilt.last_close_before(pd.Timestamp(rdate))
    # SAME LOOKAHEAD RULE FOR F&O. The bhavcopy for `rdate` publishes AFTER that session's
    # close, so reading it inside a replay of that session is exactly the leak the comment
    # above forbids. Anchor on the prior close; a None as-of yields "—", never rdate.
    bd = arb.annotate(fno.annotate(bd, _asof_replay), _asof_replay)
    rt_long, rt_short = st.tabs(["🟢 LONG (BTST-CARRY / FORMING)", "🔴 SHORT (intraday)"])
    with rt_long:
        long_side = bd[bd["action"].isin(["BTST-CARRY", "FORMING"])]
        if long_side.empty:
            st.caption("None building at this time — top-10 by close-strength so far:")
            long_side = bd.sort_values("clr", ascending=False).head(10)
        long_side = _wt(long_side, _asof_replay, "LONG")
        st.dataframe(_fmt(long_side)[_cols(long_side, buy_cols)], use_container_width=True, hide_index=True,
                     column_config={**LIVE_COLS, **TF_COLS, **FNO_COLS, **ARB_COLS})
        st.caption("Practice: note the FORMING names now, scrub to 15:15, see which held into "
                   "🌙 BTST-CARRY — those were the overnight picks. VWAP/RSI/tone point-in-time.")
    with rt_short:
        st.warning("⚠ **Intraday short only** (square off same day). Overnight short is "
                   "proven -EV; intraday direction is unvalidated. Weakness screen, not alpha.")
        sh = bd[bd["sell"].isin(["SHORT", "WEAK"])].sort_values("sell")
        if sh.empty:
            st.caption("No distribution/weakness names at this time.")
        else:
            sh = _wt(sh, _asof_replay, "SHORT")
            _c = [c for c in sell_cols_r if c in sh.columns]
            st.dataframe(_fmt(sh)[_c], use_container_width=True, hide_index=True,
                         column_config={**LIVE_COLS, **TF_COLS, **SELL_COLS, **FNO_COLS, **ARB_COLS})
    st.stop()
# ── BTST board ─────────────────────────────────────────────────────────────────
# Guarded for the same reason as the live panel: the top-of-script check proves the archive
# was readable a moment ago, not that it still is. A writer grabbing the DuckDB between the
# two lines would traceback here, on the tab that matters most.
try:
    b = _board(pd.Timestamp(date).strftime("%Y-%m-%d"))
except Exception as _e:
    st.error(f"⚠ **Archive unavailable right now.** {_e}")
    st.caption("This board only READS the archive — it never writes to it. Hit **↻ refresh** "
               "once the other process releases the file; nothing here is broken.")
    st.stop()
risk_on = b["risk_on"]

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
c1.title("Equity BTST Board")
with st.expander("❓ What is this tab — the validated overnight edge"):
    st.markdown(
        "**LEAK-FREE accumulation footprint**, every leg knowable at the 15:15 close: "
        "strong close + **trailing** delivery% (3-day avg through *yesterday* — NSE publishes "
        "today's only at ~6pm, so the signal uses the prior days) + volume surge + up-day + "
        "close>VWAP + persistent RS-leader. Regime-gated (Nifty>50MA), sector-capped, "
        "earnings-guarded, liquid names only.\n\n**Edge (leak-free, 8yr):** ~+20bps net/night "
        "gross of a soft 2025 (−3bps that year); long-only. **`delivTd`** (today's delivery) "
        "is a POST-HOC quality check, NOT a signal input.\n\n**Plan:** LONG near the close, "
        "exit next 09:15–09:30. Paper-first.\n\n**Intraday tab** = the live version (watch "
        "footprints form → 🌙 BTST-CARRY at ~15:10).")
c2.metric("Regime", "RISK-ON" if risk_on else "RISK-OFF",
          "Nifty > 50MA" if risk_on else "Nifty < 50MA")
c3.metric("Footprint hits", b["n_footprint"])
c4.metric("Deployable edge", "≈ +20 bps", "leak-free net/night, long-only")

st.caption(f"As of close **{pd.Timestamp(date).date()}** • BTST long-only • "
           "act ~15:15–15:30, exit next-morning strength • paper-first, nothing auto-executes.")

# ── sector rotation context — WITH the measurement, because the number is surprising ──
try:
    _tl, _tm = sector_tilt.sector_tilt(_ASOF_EOD)
    if _tm.get("available"):
        _ow = ", ".join(_tl.index[_tl["tilt"] == "OVERWEIGHT"][:6])
        _uw = ", ".join(_tl.index[_tl["tilt"] == "UNDERWEIGHT"][:6])
        with st.expander(f"🧭 Sector rotation context — {_tm['n_ow']} overweight / "
                         f"{_tm['n_uw']} underweight of {_tm['n_sectors']} sectors "
                         f"(DCM 1–2wk tilt, as-of {_tm['as_of']})"):
            st.markdown(
                f"**Money is rotating INTO:** {_ow or '—'}  \n"
                f"**Rotating OUT OF:** {_uw or '—'}  \n"
                f"Nifty backdrop: **{_tm['state']}** · {_tm['verdict']} · "
                f"size hint {_tm['size_hint']:.2f} · {_tm['divergence']} · "
                f"sector dispersion {_tm['dispersion']:.2f}")
            _ms = sector_tilt.load_measurement()
            if _ms is None:
                st.info("This column has **not been measured against the overnight payoff on "
                        "this machine yet** — so treat it as pure context. Run "
                        "`python -m eqbtst.cli tilt-history` then `tilt-measure` to find out "
                        "whether it carries anything, and this box will report the result.")
            else:
                _bk = {b["tilt"]: b for b in _ms.get("buckets", [])}
                _rows = "".join(
                    f"| {_ic} {_t} | {_bk[_t]['n']} | **{_bk[_t]['net_ON']:+.1f} bps** | "
                    f"{_bk[_t]['win%']:.1f}% |\n"
                    for _t, _ic in (("OVERWEIGHT", "🟢"), ("NEUTRAL", "⚪"),
                                    ("UNDERWEIGHT", "🔴"), ("WATCH", "👁"))
                    if _t in _bk)
                st.warning(
                    "**MEASURED, AND IT RUNS BACKWARDS FOR THIS BOOK — read this before you "
                    "use the column.** DCM advertises the tilt as a 1–2 WEEK signal, but its "
                    "headline daily-IC t≈9 was measured on a panel weighted by the SAME day's "
                    "turnover — a lookahead, since fixed upstream but never re-measured, so "
                    "treat that number as void. What IS measured is the line below. Joined to "
                    f"this engine's own footprint triggers (n={_ms['n_signals']}, "
                    f"regime-gated, {_ms['cost_bps']:.0f}bps cost), the overnight payoff goes "
                    "the OTHER way:\n\n"
                    "| sector tilt | n | net overnight | win% |\n|---|---|---|---|\n" + _rows +
                    f"\nOW − UW = **{_ms['diff_ON']:+.1f} bps** (t = {_ms['t_ON']:+.2f}; "
                    f"night-clustered t = {_ms['t_clustered']:+.2f}).\n\n"
                    "**But most of the headline is not a sector call.** OVERWEIGHT and "
                    f"UNDERWEIGHT signals fire on almost disjoint nights "
                    f"({_ms['nights_ow']} vs {_ms['nights_uw']}, overlapping on only "
                    f"{_ms['n_paired_nights']}), and a night is dominated by market-wide "
                    "overnight beta. Splitting it:\n\n"
                    f"* **{_ms['diff_night']:+.1f} bps** (t {_ms['t_night']:+.2f}) is simply "
                    "WHICH NIGHTS each bucket traded — timing, not sector information.\n"
                    f"* **{_ms['diff_xs']:+.1f} bps** (t {_ms['t_xs']:+.2f}) is genuinely "
                    "cross-sectional (excess over that same night's universe gap).\n"
                    f"* Same-night OW vs UW, the strictest control, is "
                    f"{_ms['diff_paired']:+.1f} bps (t {_ms['t_paired']:+.2f}) on only "
                    f"{_ms['n_paired_nights']} nights — too few to resolve either way.\n\n"
                    "**Nothing is wired to this.** The pre-registered rule asked whether "
                    "OVERWEIGHT beats UNDERWEIGHT; it does not "
                    f"({_ms['yrs_ow_wins']} of {_ms['yrs_tested']} years), so the column is "
                    "**DISPLAY-ONLY**. The inversion is a NEW hypothesis, not a licence: about "
                    "a third of it is night-timing, its significance leans on one year, and "
                    "flipping a rule after seeing its sign is exactly how two earlier 'edges' "
                    "here were retracted. It needs its own pre-registered out-of-sample test "
                    "before it touches selection or size.")
                st.caption(f"Measured {_ms.get('measured_on','?')} · "
                           "`python -m eqbtst.cli tilt` for the full ranking · `tilt-measure` "
                           "to re-run · OVERWEIGHT is RELATIVE strength, not a forecast, and "
                           "UNDERWEIGHT is never a short.")
            st.markdown("---")
            st.markdown(sector_tilt.HELP_FULL)
except Exception as _e:
    st.caption(f"sector tilt unavailable: {_e}")

# earnings guard status
g = b.get("guard", {})
if not g.get("available"):
    st.warning("⚠ **Earnings guard OFF** — NSE event calendar unreachable. A BUY name could "
               "be reporting results tonight (an earnings gap, not the edge). **Verify each "
               "pick manually** before acting.")
elif b.get("n_earnings"):
    st.success(f"✅ Earnings guard ON (NSE calendar {g.get('asof')}, {g.get('n')} events) — "
               f"excluded {b['n_earnings']} footprint name(s) reporting during the hold.")
else:
    st.caption(f"✅ Earnings guard ON (NSE calendar {g.get('asof')}) — no BUY name reports "
               "during the hold.")

if not risk_on:
    st.warning("Regime RISK-OFF (Nifty below its 50-day MA). The edge is net-negative "
               "outside a Nifty uptrend — **no new longs tonight.** Footprint names below "
               "are shown as AVOID for transparency.")

# BUY board
render_price_band()
buys = price_filter(b["buys"], "close_price")
st.subheader("🟢 BUY — tonight's long candidates" if risk_on else "🟢 BUY — (suppressed, regime off)")
if buys.empty:
    st.info("No BUY candidates: " + ("no name cleared the footprint + liquidity."
            if risk_on else "regime is risk-off."))
else:
    show = buys.copy()
    show["day%"] = (100 * show["ret"]).round(1)
    show[">vwap%"] = (100 * show["close_vs_vwap"]).round(2)
    show["RS10%"] = (100 * show["rs_idx_cum"]).round(1)
    show["wt%"] = (100 * show["weight"]).round(0)
    show = show.rename(columns={"deliv_trail": "delivTr", "deliv_per": "delivTd",
                                "vol_ratio": "vol×", "close_price": "entry≈"})
    show["band (68%)"] = show.apply(
        lambda r: f"{r['band_lo']:.1f} – {r['band_hi']:.1f}"
        if pd.notna(r.get("band_lo")) else "—", axis=1)
    show["range (74%)"] = show.apply(
        lambda r: f"{r['range_lo']:.1f} – {r['range_hi']:.1f}"
        if pd.notna(r.get("range_lo")) else "—", axis=1)
    show = arb.annotate(fno.annotate(_wt(show, _ASOF_EOD, "LONG"), _ASOF_EOD), _ASOF_EOD)
    cols = (["action", "symbol", "sector", "sector tilt", "entry≈", "band (68%)", "range (74%)",
             "exp_move%", "clr", "delivTr", "delivTd"] + fno.COLS + ["carry"] +
            ["vol×", "day%", ">vwap%", "RS10%", "wt%"])
    cols = _cols(show, cols)
    st.dataframe(show[cols].round(2), use_container_width=True, hide_index=True,
                 column_config={
                     **FNO_COLS, **ARB_COLS,
                     "symbol": st.column_config.TextColumn("symbol", pinned=True),
                     "delivTr": st.column_config.NumberColumn(
                         "delivTr", help="TRAILING delivery% (3-day avg through yesterday) — "
                         "the LEAK-FREE signal leg, known at the 15:15 close. ≥60 = sustained "
                         "accumulation.", format="%.1f"),
                     "delivTd": st.column_config.NumberColumn(
                         "delivTd", help="TODAY's delivery% — POST-HOC confirmation only (NSE "
                         "publishes it ~6pm, AFTER entry). A quality check on what you bought, "
                         "NOT a signal input.", format="%.1f"),
                     "band (68%)": st.column_config.TextColumn(
                         "band (68%)", help="Calibrated ~68% expected NEXT-DAY CLOSE range "
                         "(close ± 0.6×ATR). Where price is LIKELY to be — not a target. "
                         "The one validated forecast product is RANGE, not direction."),
                     "range (74%)": st.column_config.TextColumn(
                         "range (74%)", help="~74% full next-day HIGH/LOW range (close ± 1×ATR). "
                         "Price likely stays within this the whole next session."),
                     "exp_move%": st.column_config.NumberColumn(
                         "exp_move%", help="The 68% band as ±% of price — the expected move size.",
                         format="%.2f"),
                     "sector tilt": st.column_config.TextColumn(
                         "sector tilt", width="medium", help=sector_tilt.HELP)})
    st.caption("**band (68%)** = where price likely CLOSES next day · **range (74%)** = where it "
               "likely stays all next session. Calibrated on the F&O universe — a RANGE, not a "
               "point forecast. Plan: LONG near close, exit next-morning; size for a ~-8% shock gap.")

# AVOID
av = b["avoid"]
if not av.empty:
    with st.expander(f"⚪ AVOID — {len(av)} footprint name(s) rejected (why)"):
        a = av.copy()
        a["day%"] = (100 * a["ret"]).round(1)
        a = a.rename(columns={"deliv_per": "deliv%", "vol_ratio": "vol×"})
        a = _wt(a, _ASOF_EOD, "LONG")
        st.dataframe(a[_cols(a, ["symbol", "sector", "sector tilt", "reason", "clr", "deliv%",
                                 "vol×", "day%", "turnover_lacs"])].round(1),
                     use_container_width=True, hide_index=True,
                     column_config={"sector tilt": st.column_config.TextColumn(
                         "sector tilt", width="medium", help=sector_tilt.HELP)})

# ── paper ledger ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Paper ledger")
led = ledger.state()
lc1, lc2 = st.columns(2)
op = led["open"]
lc1.markdown("**HOLD — open positions** (exit next-morning = SELL)")
if op.empty:
    lc1.caption("none open.")
else:
    o = op.rename(columns={"entry_px": "entry", "deliv_per": "deliv%"})
    lc1.dataframe(o[["date", "symbol", "entry", "clr", "deliv%", "score"]],
                  use_container_width=True, hide_index=True)

s = led["summary"]
lc2.markdown("**Realized (closed, net of cost)**")
if not s:
    lc2.caption("no closed positions yet.")
else:
    m1, m2, m3 = lc2.columns(3)
    m1.metric("trades", s["trades"])
    m2.metric("win", f"{100*s['win']:.0f}%")
    m3.metric("mean", f"{s['mean_bps']:+.1f} bps")
    lc2.caption(f"total {s['total_bps']:+.0f} bps • worst {s['worst_bps']:+.0f} bps • "
                "backtest ≈ +16–19 bps; paper must track it.")

st.caption("SELL = exit an open long (there is no short entry — overnight short is "
           "proven dead, win ~20%). Run `python -m eqbtst.cli reconcile` next morning "
           "to fill exits.")
