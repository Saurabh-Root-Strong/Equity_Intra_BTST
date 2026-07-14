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

import pandas as pd
import streamlit as st

from eqbtst import config, data, ledger, live, screen

st.set_page_config(page_title="Equity BTST Board", layout="wide", page_icon="📊")


def _cols(df, cols):
    """Only the columns that actually exist — tolerates a stale cache / older board
    that predates a new column (e.g. 'entered'), so a missing column never crashes."""
    return [c for c in cols if c in df.columns]


def _fmt(df):
    """Render-time tidy-up. A name can sit in the timeframe list (it passed the tf verdict)
    while NEVER having fired the BTST footprint — so entered / at / since% are genuinely
    absent, not broken. Python's None then printed as the literal string 'None', which reads
    like a bug rather than an answer. Show '—' (nothing fired) and leave the numerics blank."""
    d = df.copy()
    if "entered" in d.columns:
        d["entered"] = d["entered"].where(d["entered"].notna(), "—")
    for c in ("at", "since%", "cvwap%", "rsCum%", "est_close", "vol×", "turn₹L", "wt%"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")   # None -> NaN -> renders blank
    return d

# ── hover tooltips: what each column is + WHY it matters ───────────────────────
LIVE_COLS = {
    "symbol": st.column_config.TextColumn("symbol", help="NSE F&O stock (cash, no theta).", pinned=True),
    "bar": st.column_config.TextColumn("bar", help="Is the CURRENT candle on your selected timeframe still forming? ⏳ forming = the bar has NOT closed yet, so clr/structure/RSI can still CHANGE — the signal can REPAINT and may look different at the bar's close (on 4h that's 13:15 or 15:30). ✓ closed = the bar is done; that read is final. A trigger fired mid-candle is provisional until the bar closes."),
    "est_close": st.column_config.NumberColumn("est_close", help="ESTIMATE OF THE OFFICIAL CLOSING PRICE — this is the number the decision is judged on, not the LTP. NSE does NOT close at the last traded price: the official close is the VOLUME-WEIGHTED AVERAGE of all trades in the final 30 minutes (15:00-15:30), and THAT is what the EOD archive stores and what the 8-year backtest gated on. Measured on 38,099 stock-days: the last-30-min VWAP tracks the official close to 1.6 bps, while the 15:30 last-traded price is off by 14.8 bps. So from 15:00 onward, clr / day% / cvwap% are recomputed on this estimate — judging them on the LTP would evaluate a DIFFERENT price than the one that was validated. Blank before 15:00 (no closing window yet, so the LTP stands and the board is a forecast).", format="%.2f"),
    "at": st.column_config.NumberColumn("at", help="The PRICE when the footprint fired (the 'entered' bar's close). Paired with since%, it tells you what the name has done since it triggered. NOTE: this is NOT your entry — the validated BTST entry is at the CLOSE (15:15-15:25), not at the trigger.", format="%.2f"),
    "since%": st.column_config.NumberColumn("since%", help="Move SINCE the footprint fired = (ltp / trigger-price - 1). POSITIVE = the name has HELD or EXTENDED since it triggered (the footprint is persisting — higher conviction). NEGATIVE = it has FADED since firing (the move is decaying; it may not hold into the close, and a name that fades back below VWAP loses the path-signature leg). IMPORTANT: this is NOT profit and NOT your P&L — you do not enter at the trigger. The BTST entry is near the CLOSE. Treat since% as a signal-QUALITY read (is it holding?), never as a return you captured.", format="%.2f"),
    "entered": st.column_config.TextColumn("entered", help="WHEN THE TRADE TRIGGERED — the wall-clock time (5-min resolution) the footprint FIRST fired today, INDEPENDENT of the timeframe you picked. If it fires at 12:30 while the 4h candle (09:15→13:15) is still forming, this reads 12:30 — not 09:15 or 13:15. The timeframe governs the structure/RSI/levels lens; the trigger is a clock event. (Qualification time, NOT the candle/scan timestamp.) Replay & the timeframe scans compute it CAUSALLY — the HH:MM the footprint first FORMED (up ≥1% AND closing the running session in the top of its range AND above session VWAP). The Live 5s snapshot has no intraday bars, so there it is FIRST-SEEN — the wall-clock time our scanner first saw the name qualify (accurate if the board ran from the open; later if you started the dashboard mid-session). Earlier + still holding = footprint persisted = higher conviction; just entered near the close = fresher/less proven. A DASH (—) means the validated footprint has NOT fired for this name today — it is in the timeframe list because it passed the (unvalidated) tf verdict, not because it formed the BTST footprint. Most often it is the VOLUME leg that fails: a name can be up 2.5%, closing strong and above VWAP on merely ordinary volume — price without participation is not accumulation."),
    "time": st.column_config.TextColumn("time", help="The candle the signal is on, as open→close (e.g. 13:15-15:15 = the 2h candle spanning 13:15 to 15:15). IMPORTANT: an intraday signal only CONFIRMS at the candle's CLOSE — during a live session the current candle is still forming and the signal can repaint until it closes. Live snapshot = scan time; replay = last bar at/before your cut."),
    "sector": st.column_config.TextColumn("sector", help="Canonical sector — used for the concentration cap (≤2 names/sector, so many longs in one sector aren't one macro bet)."),
    "ltp": st.column_config.NumberColumn("ltp", help="Last-traded price (Fyers) — LIVE, refreshed every 5s on every tab (one batch quote). In the TIMEFRAME tables a two-tier refresh runs: the price and everything cheap that hangs off it (day%, RS%, bar_clr, vs_vwap%, entry/stop/T1/T2, and the LONG/AVOID verdict itself) all re-derive on the live price every 5s. Only the CANDLE-derived columns (structure, RSI, tone) stay as-of the last scan — they need ~70 /history calls and only change when a BAR CLOSES anyway (on 4h, twice a day). Their age is stamped above the table; ↻ refresh to re-pull the candles.", format="%.2f"),
    "day%": st.column_config.NumberColumn("day%", help="Return vs previous close. The signal wants demand in control (≥ +1%).", format="%.2f"),
    "clr": st.column_config.NumberColumn("clr", help="Close Location in Range = (close−low)/(high−low). ≥0.70 = closing strong, buyers held into the bar. Core of the accumulation footprint.", format="%.2f"),
    "character": st.column_config.TextColumn("character", help="Candle shape: marubozu_bull (strong body, best), hammer (demand rejected lows), shooting_star (supply capped highs, avoid), doji (indecision), strong/weak_close."),
    "body": st.column_config.NumberColumn("body", help="Signed body fraction of range. Positive & large = conviction green candle.", format="%.2f"),
    "vol×": st.column_config.NumberColumn("vol×", help="Volume PACE vs own 20-day median daily volume — TIME-NORMALISED: 'is this name on pace for an N× volume day?'. ≥2× = real participation surge. This matters: raw cumulative-volume/median is mechanically tiny in the morning (only ~4% of a day's volume has traded by 09:15, ~35% by 11:00), so a genuine 2× day would read 0.27 at 09:30 and fail the 2.0 gate — the volume leg could never pass before ~15:00. Dividing by the elapsed-volume fraction makes 2.0 mean the same thing at any hour. At/after 15:25 the fraction is 1.0, so this EQUALS the raw ratio — the validated close decision (and the 8yr backtest) is unchanged. Blank = no volume yet (pre-open / market closed).", format="%.2f"),
    "RS%": st.column_config.NumberColumn("RS%", help="Relative strength vs Nifty today (stock% − index%). >0 = outperforming the market; persistent RS-leaders carry the overnight edge.", format="%.2f"),
    "delivTr": st.column_config.NumberColumn("delivTr", help="TRAILING delivery% (3-day avg through yesterday) from the EOD archive — the LEAK-FREE accumulation leg, known live at the close. ≥60 required for BTST-CARRY (a real footprint, not just price-action FORMING).", format="%.1f"),
    "book": st.column_config.TextColumn("book", help="THE RISK LAYER — is this name actually in tonight's book? '✓ TAKE' = it survived the sector cap (max 2 per sector, so five banks aren't one macro bet) and the top-5 cap, and is sized by the self-calibrator. '✗ capped' = it IS a valid BTST-CARRY signal, but it was dropped by the concentration cap or fell outside the top-5 by conviction. Only BTST-CARRY names get a book verdict. This board is what you act on at 15:10-15:30, so the book is constructed HERE, at the moment of the decision."),
    "wt%": st.column_config.NumberColumn("wt%", help="Portfolio weight — equal-weight across the selected names, SCALED by the self-calibration size multiplier (currently 0.30 = start small). Weights sum to the multiplier, not to 100%: the rest is CASH. 0 = capped out of the book, or the calibrator says stand aside.", format="%.1f"),
    "turn₹L": st.column_config.NumberColumn("turn₹L", help="TODAY's turnover in ₹ lacs (volume × price). Must be ≥2000 (₹20cr) for BTST-CARRY — the realism floor, so a fill is actually achievable without moving the price. IMPORTANT: this is TODAY's turnover, not yesterday's. A footprint day carries a 2× volume surge, so gating on yesterday's turnover was silently dropping 23% of validated signals — and the BEST ones (+29.3bps).", format="%.0f"),
    "rsCum%": st.column_config.NumberColumn("rsCum%", help="PERSISTENT relative strength — the 10-day CUMULATIVE RS vs Nifty (sum of daily stock−index returns), not a one-day burst. This is the VALIDATED leg: a name that has led the index for ~10 sessions carries the overnight edge; a one-day-burst laggard decays from ~+30bps to ~+19bps. Must be > 0.", format="%.2f"),
    "cvwap%": st.column_config.NumberColumn("cvwap%", help="PATH SIGNATURE — how far above the session VWAP the name is trading (%). ≥0.5% required. This separates a stock that TRENDED UP AND HELD (real accumulation) from one that SPIKED AND FADED back to VWAP (distribution into strength). A validated leg: it lifts net +26→+30bps, win 57→61%, and flips the losing 2025 year from −11.6 to +6.2. Blank = session VWAP not fetched (that leg is NOT met — no free pass).", format="%.2f"),
    "btst": st.column_config.TextColumn("btst", help="Footprint readiness x/5 — the EXACT legs the 8-year backtest was validated on: strong close (clr≥0.70) · up≥1% · volume pace≥2× · PERSISTENT 10-day RS>0 · PATH-SIGNATURE (≥0.5% above session VWAP). BTST-CARRY additionally requires trailing delivery ≥60 (delivTr), the regime gate, and the 15:10–15:30 close window. Proven identical to features.signal_mask over 474,315 rows / 8 years — so a name shown here is a name the backtest actually blessed."),
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
last = _last_date()
tf = st.sidebar.radio("Timeframe",
                      ["BTST (overnight)", "Intraday", "🎬 Replay (practice)"],
                      index=0)
date = st.sidebar.date_input("As-of close", value=last.date(),
                             max_value=last.date())
if st.sidebar.button("↻ refresh"):
    st.cache_data.clear()
    live.clear_universe_cache()      # also drop the EOD universe cache (picks up a new sync)
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


def render_price_band():
    """Compact price-band filter row, right-aligned above the table."""
    sp, c1, c2 = st.columns([6, 1, 1])
    sp.markdown("**Price band (₹)** — filter the list by stock price →")
    c1.number_input("Min ₹", min_value=0.0, value=0.0, step=50.0, key="price_min")
    c2.number_input("Max ₹", min_value=0.0, value=900.0, step=50.0, key="price_max")


def price_filter(df, col):
    """Keep rows whose price column is within the sidebar min/max band."""
    lo = st.session_state.get("price_min", 0.0)
    hi = st.session_state.get("price_max", 1e9) or 1e9
    if df.empty or col not in df.columns:
        return df
    return df[(df[col] >= lo) & (df[col] <= hi)]
st.sidebar.caption(f"EOD archive latest: {last.date()}  •  now {dt.datetime.now():%H:%M}")
st.sidebar.caption("BTST tab = EOD engine (delivery-confirmed). Intraday tab = live Fyers.")

# ── LIVE intraday board (Fyers) ────────────────────────────────────────────────
@st.cache_data(ttl=5)
def _live_board():
    return live.quotes_board()


@st.cache_data(ttl=60)
def _tf_scan(tf: str):
    return live.tf_scan(tf)


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

if tf == "Intraday":
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

    render_price_band()
    # timeframe dropdown — right above the table, right-aligned
    _, ddcol = st.columns([3, 1])
    scan_tf = ddcol.selectbox("Timeframe → stock list",
                              ["4h", "2h", "1h", "15m", "Live snapshot"], index=1,
                              key="scan_tf",
                              help=(
                                  "WHAT THE TIMEFRAME ACTUALLY DOES — two stages.\n\n"
                                  "① THE 1-DAY BAR SOURCES THE STOCKS. The whole universe "
                                  "(~250 names) is screened on TODAY'S OWN CANDLE, TWICE:\n"
                                  "   • LONG screen  — day% ≥ +1% AND day-clr ≥ 0.5 → top 25 "
                                  "by strength\n"
                                  "   • SHORT screen — day% ≤ −1% AND day-clr ≤ 0.5 → top 12 "
                                  "by weakness\n"
                                  "Both must clear the ₹20cr turnover floor (a level you "
                                  "cannot get filled at is worse than no level). Nothing "
                                  "reaches the table unless the 1-day bar lets it in.\n\n"
                                  "② THE TIMEFRAME JUDGES THEM. Its candles — today's PLUS "
                                  "prior days' (a 2h read spans ~15 days, a 4h read ~30) — give "
                                  "structure (Kaufman efficiency), RSI/tone, the last bar's "
                                  "close-strength, and the ATR that sets your stop/T1/T2. Those "
                                  "decide LONG / NEUTRAL / AVOID.\n\n"
                                  "SO: the timeframe NEVER SOURCES a stock — it only JUDGES and "
                                  "STYLES one. Switch 2h→4h and you get the SAME candidates, "
                                  "re-judged, with a WIDER stop (4h ATR ≫ 15m ATR). The dropdown "
                                  "changes your stop width and the tape view, not the stock pool.\n\n"
                                  "TWO CLOCKS in this table. LIVE every 5s (one batch quote): "
                                  "ltp, day%, RS%, bar_clr, vs_vwap%, entry/stop/T1/T2, and the "
                                  "VERDICT itself. AS-OF THE LAST SCAN: structure, rsi, tone — "
                                  "they need ~70 /history calls and only move when a BAR CLOSES "
                                  "anyway. Their age is stamped above the table.\n\n"
                                  "NSE trades 6h15m, so a 4h bar = 09:15–13:15 then a partial "
                                  "13:15–15:30 (only ~2 bars/day — which is why coarse frames "
                                  "chain across prior days).\n\n"
                                  "HONEST: this whole lane is intraday CONTEXT. Backtested at "
                                  "every timeframe: 15m −6.2 · 1h −5.0 · 2h −5.4 · 4h −5.2 bps. "
                                  "It loses money at every bar size. The one VALIDATED trade is "
                                  "BTST-CARRY on the 1-day bar at 15:10–15:30 → 'Live snapshot'.")
                              )

    # ---- timeframe-driven stock list (1h / 2h / 15m bars) ----------------------
    if scan_tf != "Live snapshot":
        st.caption(
            f"**Two stages.** ① The **1-DAY BAR sources the stocks** — the whole universe is "
            f"screened on today's own candle, **twice**: the LONG screen takes `day% ≥ +1% "
            f"AND day-clr ≥ 0.5` (top 25 by strength), the SHORT screen takes the mirror "
            f"`day% ≤ −1% AND day-clr ≤ 0.5` (top 12 by weakness). Both must clear the "
            f"₹20cr turnover floor. ② The **{scan_tf} candles then JUDGE them** — "
            f"today's + prior days' bars give structure / RSI / tone, and the ATR that sets "
            f"your stop. **The timeframe never SOURCES a stock — it only judges and styles "
            f"one.** Switch {scan_tf}→4h: *same candidates*, re-judged, wider stop.")
        with st.spinner(f"Scanning on {scan_tf} bars…"):
            sc = _tf_scan(scan_tf)
        if not sc["ok"] or sc["board"].empty:
            st.info(f"No {scan_tf} candidates — market closed / pre-open, or nothing "
                    "clears the footprint on this timeframe. During a live session this "
                    "fills. For today's delivery-confirmed picks use the BTST tab.")
        else:
            b = price_filter(sc["board"], "ltp")
            # STALENESS — this table does NOT tick. The tf scan is heavy (~70 /history
            # calls), so unlike the 5s Live snapshot it is a FROZEN SNAPSHOT: Streamlit
            # only re-runs on interaction, so ltp — and every level derived from it
            # (entry/stop/T1/T2) — can be arbitrarily old while looking perfectly actionable.
            _sa = sc.get("scanned_at")
            _age = (dt.datetime.now() - _sa).total_seconds() if _sa else 0
            _mkt = live.market_open()
            if _sa:
                if _mkt and _age > 1800:
                    st.warning(
                        f"⚠ **structure/RSI are {int(_age // 60)}m old** (scanned "
                        f"{_sa:%H:%M}). Prices, levels and the verdict are live (5s), but the "
                        "CANDLE-derived columns need a re-pull. Hit **↻ refresh**.")
                else:
                    st.caption(
                        f"🕒 **Two clocks.** LIVE every 5s: `ltp` · `day%` · `RS%` · "
                        f"`bar_clr` · `vs_vwap%` · `entry/stop/T1/T2` · **the verdict**. "
                        f"As-of the **{_sa:%H:%M:%S}** scan ({int(_age)}s ago): `structure` · "
                        f"`rsi` · `tone` — these need candles and only move when a bar closes. "
                        "↻ refresh to re-pull them.")
            st.caption(f"scanned {sc['n_scanned']} shortlisted names · Nifty "
                       f"{sc.get('idx_ret', 0):+.2f}% · regime "
                       f"{'RISK-ON' if sc['risk_on'] else 'RISK-OFF'}")
            long_cols = ["symbol", "entered", "at", "since%", "time", "bar", "sector", "ltp", "day%", "structure",
                         "bar_clr", "character", "vs_vwap%", "rsi7", "rsi14", "tone", "RS%",
                         "entry", "stop", "t1", "t2", "atr%", "action"]
            sell_cols_tf = ["symbol", "entered", "at", "since%", "time", "bar", "sector", "ltp", "day%", "structure",
                            "bar_clr", "character", "vs_vwap%", "rsi7", "rsi14", "tone", "RS%",
                            "entry", "s_stop", "s_t1", "s_t2", "atr%", "sell"]

            # ── TWO-TIER REFRESH ────────────────────────────────────────────────────
            # The heavy half (structure / RSI / tone / bar_clr) only changes when a BAR
            # CLOSES — on 4h, twice a day — and costs ~70 /history calls. The cheap half
            # (price, and every level anchored to it) changes every tick and costs ONE
            # batch quote. Freezing the whole table froze the wrong half: it pinned a
            # precise-looking entry/stop/T1/T2 to a price that had stopped moving.
            # So: prices tick here every 5s; structure stays as-of the stamped scan.
            @st.fragment(run_every="5s")
            def _tf_panel():
                bb = live.refresh_prices(b, risk_on=sc['risk_on']) if live.market_open() else b
                st.caption(f"💹 prices live ({dt.datetime.now():%H:%M:%S}) · structure as-of "
                           f"{_sa:%H:%M}" if (_sa and live.market_open())
                           else "market closed — last-session values")
                tb, ts = st.tabs([f"🟢 LONG ({scan_tf} bars)", f"🔴 SHORT ({scan_tf} bars)"])
                with tb:
                    lo = bb[bb["action"] == "LONG"]
                    _lo = lo if not lo.empty else bb
                    st.dataframe(_fmt(_lo)[_cols(_lo, long_cols)],
                                 use_container_width=True, hide_index=True,
                                 column_config={**LIVE_COLS, **TF_COLS})
                    if lo.empty:
                        st.caption("No LONG on this timeframe — showing the shortlist.")
                with ts:
                    st.warning("⚠ **Intraday short only — SQUARE OFF BEFORE THE CLOSE.** "
                               "Overnight short is proven -EV (win 20%); intraday direction "
                               "has no validated edge either. Weakness screen, not alpha — "
                               "trade small, manage by s_stop.")
                    sh = bb[bb["sell"].isin(["SHORT", "WEAK"])].sort_values("sell")
                    if sh.empty:
                        st.caption("No distribution/weakness names on this timeframe.")
                    else:
                        st.dataframe(_fmt(sh)[_cols(sh, sell_cols_tf)], use_container_width=True,
                                     hide_index=True,
                                     column_config={**LIVE_COLS, **TF_COLS, **SELL_COLS})

            _tf_panel()
            st.caption("Entry/Stop/T1/T2 = ATR risk geometry on this timeframe, NOT a "
                       "forecast. Long-only is the validated edge; SHORT is intraday context.")
        st.stop()

    # ---- live snapshot (5s price-action scan) — Buy / Sell tabs ---------------
    st.caption("Live price-action scan, refreshing every 5s. **BUY** = accumulation "
               "footprint (validated overnight edge; delivery confirms at close). **SELL** "
               "= intraday distribution/weakness (square off same day — overnight short is "
               "proven -EV).")

    @st.fragment(run_every="5s")
    def _live_panel():
        lb = _live_board()
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
        counts = bd["action"].value_counts().to_dict()
        scounts = bd["sell"].value_counts().to_dict()
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Regime", "RISK-ON" if lb["risk_on"] else "RISK-OFF")
        h2.metric("Nifty today", f"{lb.get('idx_ret', 0):+.2f}%")
        h3.metric("BUY (carry/forming)",
                  f"{counts.get('BTST-CARRY', 0) + counts.get('FORMING', 0)}")
        h4.metric("SELL (short/weak)",
                  f"{scounts.get('SHORT', 0) + scounts.get('WEAK', 0)}")

        buy_cols = ["symbol", "entered", "at", "since%", "time", "sector", "ltp", "est_close", "day%", "clr", "character", "vol×",
                    "RS%", "rsCum%", "cvwap%", "delivTr", "turn₹L", "btst", "book", "wt%", "exp_ON", "band_lo", "band_hi",
                    "entry", "stop", "t1", "t2", "risk%", "atr%", "action"]
        sell_cols = ["symbol", "entered", "at", "since%", "time", "sector", "ltp", "day%", "clr", "character", "vol×",
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
                st.dataframe(_fmt(carry)[_cols(carry, buy_cols)], use_container_width=True, hide_index=True,
                             column_config=LIVE_COLS)
            # ⏳ FORMING — watch list, may flip to CARRY near the close
            st.markdown("#### ⏳ FORMING — building (watch)")
            if forming.empty:
                b = bd.sort_values("clr", ascending=False).head(10)
                st.caption("No footprint building yet — top-10 by close-strength meanwhile:")
                st.dataframe(_fmt(b)[_cols(b, buy_cols)], use_container_width=True, hide_index=True,
                             column_config=LIVE_COLS)
            else:
                st.caption(f"{len(forming)} building — may flip to 🌙 BTST-CARRY near the close.")
                st.dataframe(_fmt(forming)[_cols(forming, buy_cols)], use_container_width=True, hide_index=True,
                             column_config=LIVE_COLS)
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
                st.dataframe(_fmt(s)[_cols(s, sell_cols)], use_container_width=True, hide_index=True,
                             column_config={**LIVE_COLS, **SELL_COLS})
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
    buy_cols = ["symbol", "entered", "at", "since%", "time", "sector", "ltp", "day%", "structure", "clr", "character",
                "vs_vwap%", "rsi7", "rsi14", "tone", "vol×", "RS%", "rsCum%", "cvwap%", "btst", "entry",
                "stop", "t1", "t2", "atr%", "action"]
    sell_cols_r = ["symbol", "entered", "at", "since%", "time", "sector", "ltp", "day%", "structure", "clr", "character",
                   "vs_vwap%", "rsi7", "rsi14", "tone", "vol×", "RS%", "entry",
                   "s_stop", "s_t1", "s_t2", "atr%", "sell"]
    rt_long, rt_short = st.tabs(["🟢 LONG (BTST-CARRY / FORMING)", "🔴 SHORT (intraday)"])
    with rt_long:
        long_side = bd[bd["action"].isin(["BTST-CARRY", "FORMING"])]
        if long_side.empty:
            st.caption("None building at this time — top-10 by close-strength so far:")
            long_side = bd.sort_values("clr", ascending=False).head(10)
        st.dataframe(_fmt(long_side)[_cols(long_side, buy_cols)], use_container_width=True, hide_index=True,
                     column_config={**LIVE_COLS, **TF_COLS})
        st.caption("Practice: note the FORMING names now, scrub to 15:15, see which held into "
                   "🌙 BTST-CARRY — those were the overnight picks. VWAP/RSI/tone point-in-time.")
    with rt_short:
        st.warning("⚠ **Intraday short only** (square off same day). Overnight short is "
                   "proven -EV; intraday direction is unvalidated. Weakness screen, not alpha.")
        sh = bd[bd["sell"].isin(["SHORT", "WEAK"])].sort_values("sell")
        if sh.empty:
            st.caption("No distribution/weakness names at this time.")
        else:
            _c = [c for c in sell_cols_r if c in sh.columns]
            st.dataframe(_fmt(sh)[_c], use_container_width=True, hide_index=True,
                         column_config={**LIVE_COLS, **TF_COLS, **SELL_COLS})
    st.stop()
# ── BTST board ─────────────────────────────────────────────────────────────────
b = _board(pd.Timestamp(date).strftime("%Y-%m-%d"))
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
    cols = ["action", "symbol", "sector", "entry≈", "band (68%)", "range (74%)",
            "exp_move%", "clr", "delivTr", "delivTd", "vol×", "day%", ">vwap%", "RS10%", "wt%"]
    st.dataframe(show[cols].round(2), use_container_width=True, hide_index=True,
                 column_config={
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
                         format="%.2f")})
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
        st.dataframe(a[["symbol", "sector", "reason", "clr", "deliv%", "vol×",
                        "day%", "turnover_lacs"]].round(1),
                     use_container_width=True, hide_index=True)

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
