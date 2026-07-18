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

from eqbtst import config, data, ledger, live, mtf, screen

st.set_page_config(page_title="Equity BTST Board", layout="wide", page_icon="📊")


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
    for c in ("at", "since%", "cvwap%", "rsCum%", "est_close", "vol×", "turn₹L", "wt%"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")   # None -> NaN -> renders blank
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
        d["setup"] = [f"{mtf.TAG_ICON.get(v, '')} {v}".strip() if isinstance(v, str) else "—"
                      for v in d["setup"]]
    return d

# ── hover tooltips: what each column is + WHY it matters ───────────────────────
LIVE_COLS = {
    "symbol": st.column_config.TextColumn("symbol", help="NSE F&O stock (cash, no theta).", pinned=True),
    "bar": st.column_config.TextColumn("bar", help="Is the CURRENT candle on your selected timeframe still forming? ⏳ forming = the bar has NOT closed yet, so clr/structure/RSI can still CHANGE — the signal can REPAINT and may look different at the bar's close (on 4h that's 13:15 or 15:30). ✓ closed = the bar is done; that read is final. A trigger fired mid-candle is provisional until the bar closes."),
    "est_close": st.column_config.NumberColumn("est_close", help="ESTIMATE OF THE OFFICIAL CLOSING PRICE — this is the number the decision is judged on, not the LTP. NSE does NOT close at the last traded price: the official close is the VOLUME-WEIGHTED AVERAGE of all trades in the final 30 minutes (15:00-15:30), and THAT is what the EOD archive stores and what the 8-year backtest gated on. Measured on 38,099 stock-days: the last-30-min VWAP tracks the official close to 1.6 bps, while the 15:30 last-traded price is off by 14.8 bps. So from 15:00 onward, clr / day% / cvwap% are recomputed on this estimate — judging them on the LTP would evaluate a DIFFERENT price than the one that was validated. Blank before 15:00 (no closing window yet, so the LTP stands and the board is a forecast).", format="%.2f"),
    "at": st.column_config.NumberColumn("at", help="The PRICE when the footprint fired (the 'entered' bar's close). Paired with since%, it tells you what the name has done since it triggered. NOTE: this is NOT your entry — the validated BTST entry is at the CLOSE (15:15-15:25), not at the trigger.", format="%.2f"),
    "since%": st.column_config.NumberColumn("since%", help="Move SINCE the footprint fired = (ltp / trigger-price - 1). POSITIVE = the name has HELD or EXTENDED since it triggered (the footprint is persisting — higher conviction). NEGATIVE = it has FADED since firing (the move is decaying; it may not hold into the close, and a name that fades back below VWAP loses the path-signature leg). IMPORTANT: this is NOT profit and NOT your P&L — you do not enter at the trigger. The BTST entry is near the CLOSE. Treat since% as a signal-QUALITY read (is it holding?), never as a return you captured.", format="%.2f"),
    "entered": st.column_config.TextColumn("entered", help="WHEN THE TRADE TRIGGERED — the wall-clock time (5-min resolution) the footprint FIRST fired today, INDEPENDENT of the timeframe you picked. If it fires at 12:30 while the 4h candle (09:15→13:15) is still forming, this reads 12:30 — not 09:15 or 13:15. The timeframe governs the structure/RSI/levels lens; the trigger is a clock event. (Qualification time, NOT the candle/scan timestamp.) Replay & the timeframe scans compute it CAUSALLY — the HH:MM the footprint first FORMED (up ≥1% AND closing the running session in the top of its range AND above session VWAP). The Live 5s snapshot has no intraday bars, so there it is FIRST-SEEN — the wall-clock time our scanner first saw the name qualify (accurate if the board ran from the open; later if you started the dashboard mid-session). Earlier + still holding = footprint persisted = higher conviction; just entered near the close = fresher/less proven. A DASH (—) means the validated footprint has NOT fired for this name today — it is in the timeframe list because it passed the (unvalidated) tf verdict, not because it formed the BTST footprint. Most often it is the VOLUME leg that fails: a name can be up 2.5%, closing strong and above VWAP on merely ordinary volume — price without participation is not accumulation. Those rejected names were measured over 8 years: 1,235 of them, worth −0.1bps. A coin flip. The dash is the signal protecting you from a chart pattern that pays nothing."),
    "time": st.column_config.TextColumn("time", help="The candle the signal is on, as open→close (e.g. 13:15-15:15 = the 2h candle spanning 13:15 to 15:15). IMPORTANT: an intraday signal only CONFIRMS at the candle's CLOSE — during a live session the current candle is still forming and the signal can repaint until it closes. Live snapshot = scan time; replay = last bar at/before your cut."),
    "sector": st.column_config.TextColumn("sector", help="Canonical sector — used for the concentration cap (≤2 names/sector, so many longs in one sector aren't one macro bet)."),
    "ltp": st.column_config.NumberColumn("ltp", help="Last-traded price (Fyers) — LIVE, refreshed every 5s on every tab (one batch quote). In the TIMEFRAME tables a two-tier refresh runs: the price and everything cheap that hangs off it (day%, RS%, bar_clr, vs_vwap%, entry/stop/T1/T2, and the LONG/AVOID verdict itself) all re-derive on the live price every 5s. Only the CANDLE-derived columns (structure, RSI, tone) stay as-of the last scan — they need ~70 /history calls and only change when a BAR CLOSES anyway (on 4h, twice a day). Their age is stamped above the table; ↻ refresh to re-pull the candles.", format="%.2f"),
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
        st.rerun()
    st.stop()
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
    """Compact price-band filter row, right-aligned above the table.

    Max defaults to ₹900 (user preference). price_filter reads Max=0 as 'no cap' (0 or 1e9), so
    set Max to 0 to see every name above ₹900 (most index heavyweights)."""
    sp, c1, c2 = st.columns([6, 1, 1])
    sp.markdown("**Price band (₹)** — filter by stock price (Max 0 = no limit) →")
    c1.number_input("Min ₹", min_value=0.0, value=0.0, step=50.0, key="price_min")
    c2.number_input("Max ₹ (0 = all)", min_value=0.0, value=900.0, step=50.0, key="price_max")


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


@st.cache_data(ttl=1800, show_spinner=False)
def _uni_scan(nonce: int):
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
        help="WHERE the price sits inside the HIGHER-TF range box. 0.00 = at the box LOW, "
             "1.00 = at the box HIGH, 0.50 = dead middle.\n\n"
             "This is the variable that decides whether a lower-TF breakout means anything: "
             "≥0.72 (at the ceiling) or ≤0.28 (at the floor) = a break can genuinely resolve "
             "the range. Anywhere in the middle = the same break is a trap."),
}

# Delivery-conviction columns — ported from the DCM sector-rotation view (same formulas).
DELIV_COLS = {
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
        st.caption(
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
            # An empty/failed scan is NOT cached (it raised). If the market is open this is almost
            # always a transient first-seconds-of-session quote miss — auto-retry ONCE with a fresh
            # nonce before bothering the user. Guarded so it can never loop.
            if live.market_open() and not st.session_state.get("_uni_retried"):
                st.session_state["_uni_retried"] = True
                st.session_state["uni_nonce"] += 1
                live._UNISCAN_CACHE.clear()
                st.rerun()
            _open = live.market_open()
            st.session_state.pop("_uni_retried", None)
            st.info(
                (f"Universe scan came back empty — {_e}. **Market is OPEN**, so this is a "
                 "transient quote-fetch miss (or the very first seconds of the session). It is "
                 "**not cached** — just hit **↻ re-scan universe**.") if _open else
                (f"Universe scan unavailable — {_e}. Market is closed / pre-open, or the Fyers "
                 "token is stale (~06:00 daily expiry). Re-auth, then hit **↻ re-scan**."))
            st.stop()
        light = price_filter(sc["board"], "ltp")     # price band applies; no turnover floor
        _sa = sc.get("scanned_at")
        _age = (dt.datetime.now() - _sa).total_seconds() if _sa else 0

        # ── AUTO-REFRESH aligned to the 15-min bar close (opt-in) ───────────────────────
        # Structure changes ONLY when a bar closes; the 15m frame is the finest, so its
        # boundary (:00/:15/:30/:45) is the natural cadence. This fragment ticks lightly and
        # triggers ONE full re-scan just after each boundary — never mid-bar (same result,
        # wasted fetches). `scanned_b15` remembers the bucket the current scan belongs to so
        # the first tick after a fresh scan does not immediately re-fire.
        def _b15(t=None):
            t = t or dt.datetime.now()
            return f"{t:%H}:{(t.minute // 15) * 15:02d}"
        st.session_state["scanned_b15"] = _b15(_sa) if _sa else _b15()
        if auto_struct:
            @st.fragment(run_every="20s")
            def _auto_rescan():
                if not live.market_open():
                    return
                if _b15() != st.session_state.get("scanned_b15"):
                    st.session_state["scanned_b15"] = _b15()
                    st.session_state["uni_nonce"] += 1     # force a fresh pull on the next run
                    live._UNISCAN_CACHE.clear()
                    st.rerun()
            _auto_rescan()

        if _sa and live.market_open() and _age > 300 and not auto_struct:
            st.caption(f"🕒 structure as-of **{_sa:%H:%M:%S}** ({int(_age // 60)}m {int(_age % 60)}s "
                       "ago) — ↻ re-scan to re-pull. Prices on matched names tick live below.")
        st.caption(f"scanned **{sc['n_scanned']}** liquid names · Nifty "
                   f"{sc.get('idx_ret', 0):+.2f}% · regime "
                   f"{'RISK-ON' if sc['risk_on'] else 'RISK-OFF'}"
                   + ("  ·  🔄 auto-refresh ON (next 15-min close)" if auto_struct else ""))
        # A name with no intraday candles cannot match ANY intraday structure filter — it just
        # disappears. Say how many, so the board is never silently answering from a subset.
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
                "**`structure()` always looks at the LAST 20 bars of each frame** "
                "(`lookback=20`, Kaufman efficiency + range). Not the whole history — the last "
                "20 bars *of that timeframe*. So the calendar span differs per frame:\n\n"
                "| Frame | Bars/day (NSE 6h15m) | 20 bars ≈ prior days | Fetched (raw) |\n"
                "|---|---|---|---|\n"
                "| **1h** | ~6.5 | **~3 trading days** | 15m over 20 cal-days, resampled |\n"
                "| **2h** | ~3.5 | **~6–7 trading days** | ″ |\n"
                "| **4h** | 2 | **~10 trading days** | ″ |\n"
                "| **1D** | 1 | **~20 trading days (~1 month)** | EOD archive `tail(60)`, uses last 20 |\n"
                "| **1W** | 1/wk | **~20 weeks (~5 months)** | EOD archive, W-FRI resample |\n\n"
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
                "- 🌀 **Coiling** — recent 3-bar range **< 60%** of the prior range → volatility "
                "contracting.\n"
                "- ↔️ **Range** — none of the above: oscillating in its band.\n\n"
                "*(All four thresholds are tunable in config: STRUCT_BREAKOUT_ATR, "
                "STRUCT_TREND_ER, STRUCT_TREND_ATR, STRUCT_COIL. Want stronger, ~3% breakouts? "
                "raise STRUCT_BREAKOUT_ATR toward 1.0.)*\n\n"
                "**So your combo can read 0 matches and that's normal:** e.g. 4h 🚀 Breakout ↑ ∩ "
                "1h 🌀 Coiling is rare — a breakout, then a pause, caught in the same snapshot. "
                "Loosen one leg (4h 📈 Uptrend, or LTF = Any) to populate.")
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
        pc1, pc2 = st.columns([3, 2])
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
        _setup_f = "All"
        if _P:
            _setup_f = pc2.selectbox(
                "Setup quality", ["All", "🎯 Textbook only", "🟢 Long-side setups",
                                  "🪤 Exclude traps"], index=0, key="mtf_setupf",
                help=("Filter on the **HTF × LTF setup tag** (the chartist read), not on raw "
                      "shapes.\n\n"
                      "• **🎯 Textbook only** — `WITH-TREND CONTINUATION`: higher-TF trending, "
                      "lower-TF coiling into it. The classical continuation setup.\n"
                      "• **🟢 Long-side** — continuation + range-top break + pullback-with-trend.\n"
                      "• **🪤 Exclude traps** — drops `FALSE-BREAK TRAP` (a lower-TF break in the "
                      "MIDDLE of the higher-TF box — statistically fades).\n\n"
                      "Rows sort best-setup-first regardless."))
            st.caption(f"📐 **{_P['label']}** · hold: *{_P['hold']}* — {_P['note']}")
        fc1, fc2, fc3, fc4 = st.columns(4)
        f_htf = fc1.selectbox("Higher TF", [_NONE, "1h", "2h", "4h", "1D", "1W"], index=3,
                              key="mtf_htf", disabled=bool(_P),
                              help=(
                                  "**HIGHER TIMEFRAME — the big picture.**\n\n"
                                  "Pick the LARGE timeframe you want to judge the trend on. This is "
                                  "the CONTEXT: where is the stock in its bigger move? A bigger "
                                  "frame changes slowly and matters more.\n\n"
                                  "• **1h / 2h / 4h** — built live from today's + recent days' "
                                  "candles. They include the bar still forming now, so they can "
                                  "CHANGE (repaint) until that bar closes.\n"
                                  "• **1D / 1W** — from the end-of-day archive, through the LAST "
                                  "close. Rock-solid: they never change during the day (but they "
                                  "don't see today yet).\n\n"
                                  "This box picks the frame; the next box picks the SHAPE you want "
                                  "on it. Pick **'— none —'** to switch the Higher-TF leg OFF "
                                  "entirely (same as setting its structure to 'Any')."))
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
                                  "**LOWER TIMEFRAME — the zoom-in.**\n\n"
                                  "Pick the SMALL timeframe where you want to TIME the entry inside "
                                  "the big frame's context. Think of it as zooming into the same "
                                  "chart.\n\n"
                                  "It does two jobs:\n"
                                  "1. Filters on the small-frame SHAPE (next box).\n"
                                  "2. Your **entry / stop / T1 / T2 levels are built on THIS "
                                  "frame** (its ATR sets the stop width). Smaller frame = tighter "
                                  "levels.\n\n"
                                  "Keep it BELOW the Higher TF (e.g. Higher 4h, Lower 1h). Intraday "
                                  "frames can repaint until the bar closes.\n\n"
                                  "**'— none —'** switches the Lower-TF FILTER off. Levels/RSI/verdict "
                                  "still need a frame, so they fall back to your Higher TF (or 1h if "
                                  "that is also none)."))
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
        if _P:
            light = live.add_setup(light, ltf=_P["ltf"], htf=_P["htf"])
            _keep = {"🎯 Textbook only": {"WITH-TREND CONTINUATION"},
                     "🟢 Long-side setups": mtf.LONG_TAGS}.get(_setup_f)
            if _keep is not None:
                light, _setup_on = light[light["setup"].isin(_keep)], True
            elif _setup_f == "🪤 Exclude traps":
                light, _setup_on = light[~light["setup"].isin(mtf.AVOID_TAGS)], True
            light = light.sort_values(["setup_rank", "turn₹L"], ascending=[True, False])

        after_deliv = _deliv_filter(light)          # stage the chain so each cut is VISIBLE
        filtered = _mtf_filter(after_deliv)
        active = _htf_on or _ltf_on or _setup_on or (min_wtd > 0) or (min_vs > 0)
        light_cols = (["symbol", "sector", "ltp", "turn₹L", "day%"]
                      + (["setup", "loc"] if _P else [])
                      + ["wtd_deliv7", "deliv_vs_100d",
                         "s15m", "s1h", "s2h", "s4h", "s1D", "s1W"])

        # WHAT THE TAPE LOOKS LIKE RIGHT NOW — the census of setups across the whole universe,
        # with the full read for each. A tag in a cell is a label; this is what it MEANS.
        if _P and "setup" in light.columns and not light.empty:
            _vc = light["setup"].value_counts()
            with st.expander(f"🔭 What the {_P['ltf']} × {_P['htf']} tape says right now — "
                             f"{len(_vc)} setup types across {len(light)} names"):
                for _tag, _cnt in _vc.items():
                    _r = light[light["setup"] == _tag]
                    _ex = ", ".join(_r.nlargest(min(4, len(_r)), "turn₹L")["symbol"])
                    st.markdown(f"**{mtf.TAG_ICON.get(_tag, '')} {_tag}** — {_cnt} names  \n"
                                f"{_r['setup_read'].iloc[0]}  \n"
                                f"*most liquid:* {_ex}")
                st.caption("⚠ Setup quality is a CHARTIST ranking, not an expected return. "
                           "Intraday multi-TF alignment has no validated edge in this stack; the "
                           "one validated trade here is the overnight BTST carry, which is "
                           "selected by delivery + close-strength, not by these shapes.")

        if not active:
            st.info(f"**{len(light)} names** in the universe. Pick a **HTF/LTF structure** or "
                    "raise a **delivery** slider above to select — then the matches get levels, "
                    "RSI and a verdict on your Lower TF. Showing structure + delivery only until "
                    "you filter.")
            st.dataframe(_fmt(light)[_cols(light, light_cols)], use_container_width=True,
                         hide_index=True, column_config={**LIVE_COLS, **TF_COLS, **DELIV_COLS, **SETUP_COLS})
            _tally(len(light), sc["n_scanned"], "names",
                   "no filter active" if len(light) == sc["n_scanned"]
                   else "price band is the only cut")
            st.stop()

        # ── FILTER FUNNEL — show WHERE names drop, so a 0 is diagnosable (which stage killed
        # it?), not a mystery. Only stages you actually engaged appear.
        _funnel = [f"scanned **{sc['n_scanned']}**"]
        if _setup_on:
            _funnel.append(f"setup ({_setup_f}) → **{len(light)}**")
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
        if enr.empty:
            st.info("Matches found, but none could be read on the Lower TF (thin candles). "
                    "Try a coarser Lower TF.")
            st.stop()

        _sc = ["setup", "loc"] if _P else []
        long_cols = ["symbol", *_sc, "entered", "at", "since%", "time", "bar", "sector", "ltp",
                     "turn₹L", "day%", "wtd_deliv7", "deliv_vs_100d",
                     "s15m", "s1h", "s2h", "s4h", "s1D", "s1W",
                     "bar_clr", "character", "vs_vwap%", "rsi7", "rsi14", "tone", "RS%",
                     "entry", "stop", "t1", "t2", "atr%", "action"]
        sell_cols_tf = ["symbol", *_sc, "entered", "at", "since%", "time", "bar", "sector", "ltp",
                        "turn₹L", "day%", "wtd_deliv7", "deliv_vs_100d",
                        "s15m", "s1h", "s2h", "s4h", "s1D", "s1W",
                        "bar_clr", "character", "vs_vwap%", "rsi7", "rsi14", "tone", "RS%",
                        "entry", "s_stop", "s_t1", "s_t2", "atr%", "sell"]

        @st.fragment(run_every="5s")
        def _struct_panel():
            bb = live.refresh_prices(enr, risk_on=sc["risk_on"]) if live.market_open() else enr
            st.caption(f"💹 prices live ({dt.datetime.now():%H:%M:%S}) · structure/levels as-of "
                       f"{_sa:%H:%M}" if (_sa and live.market_open())
                       else "market closed — last-session values")
            tb, tsh = st.tabs([f"🟢 LONG ({levels_tf} bars)", f"🔴 SHORT ({levels_tf} bars)"])
            with tb:
                lo = bb[bb["action"] == "LONG"]
                _lo = lo if not lo.empty else bb
                st.dataframe(_fmt(_lo)[_cols(_lo, long_cols)], use_container_width=True,
                             hide_index=True, column_config={**LIVE_COLS, **TF_COLS, **DELIV_COLS, **SETUP_COLS})
                _tally(len(_lo), sc["n_scanned"], "names",
                       f"{len(filtered)} matched the filter · {len(bb)} read on {levels_tf}"
                       + ("" if lo.empty else f" · {len(lo)} LONG"))
                if lo.empty:
                    st.caption("No LONG verdict among the matches — showing all matches.")
            with tsh:
                st.warning("⚠ **Intraday short only — SQUARE OFF BEFORE THE CLOSE.** Overnight "
                           "short is proven -EV (win 20%); intraday direction has no validated "
                           "edge either. Weakness screen, not alpha — trade small, manage by s_stop.")
                sh = bb[bb["sell"].isin(["SHORT", "WEAK"])].sort_values("sell")
                if sh.empty:
                    st.caption("No distribution/weakness names among the matches.")
                else:
                    st.dataframe(_fmt(sh)[_cols(sh, sell_cols_tf)], use_container_width=True,
                                 hide_index=True,
                                 column_config={**LIVE_COLS, **TF_COLS, **DELIV_COLS, **SETUP_COLS, **SELL_COLS})
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
