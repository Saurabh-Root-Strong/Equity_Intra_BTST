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

# ── hover tooltips: what each column is + WHY it matters ───────────────────────
LIVE_COLS = {
    "symbol": st.column_config.TextColumn("symbol", help="NSE F&O stock (cash, no theta)."),
    "sector": st.column_config.TextColumn("sector", help="Canonical sector — used for the concentration cap (≤2 names/sector, so many longs in one sector aren't one macro bet)."),
    "ltp": st.column_config.NumberColumn("ltp", help="Live last-traded price (Fyers).", format="%.2f"),
    "day%": st.column_config.NumberColumn("day%", help="Return vs previous close. The signal wants demand in control (≥ +1%).", format="%.2f"),
    "clr": st.column_config.NumberColumn("clr", help="Close Location in Range = (close−low)/(high−low). ≥0.70 = closing strong, buyers held into the bar. Core of the accumulation footprint.", format="%.2f"),
    "character": st.column_config.TextColumn("character", help="Candle shape: marubozu_bull (strong body, best), hammer (demand rejected lows), shooting_star (supply capped highs, avoid), doji (indecision), strong/weak_close."),
    "body": st.column_config.NumberColumn("body", help="Signed body fraction of range. Positive & large = conviction green candle.", format="%.2f"),
    "vol×": st.column_config.NumberColumn("vol×", help="Volume vs own 20-day median. ≥2× = real participation surge. Blank = no volume yet (pre-open / market closed).", format="%.2f"),
    "RS%": st.column_config.NumberColumn("RS%", help="Relative strength vs Nifty today (stock% − index%). >0 = outperforming the market; persistent RS-leaders carry the overnight edge.", format="%.2f"),
    "btst": st.column_config.TextColumn("btst", help="BTST-readiness x/4: how many LIVE overnight-footprint legs are met (strong close · up≥1% · vol≥2× · RS-leader). Higher near the close = the day looks like accumulation that historically drifts up overnight → a next-day move is more expectable. VWAP-hold + delivery% are the 2 further confirmations."),
    "exp_ON": st.column_config.TextColumn("exp_ON", help="Expected overnight drift IF the footprint holds (full-footprint historical average, gross — NOT a per-name forecast). Exit next-morning strength."),
    "entry": st.column_config.NumberColumn("entry", help="Suggested entry ≈ LTP. RISK GEOMETRY, not a forecast.", format="%.2f"),
    "stop": st.column_config.NumberColumn("stop", help="Protective stop = 1×ATR below (or below day-low if tighter). Defines your risk.", format="%.2f"),
    "t1": st.column_config.NumberColumn("t1", help="Target 1 = +1×ATR. An ATR-sized move, for trade management — not a predicted price.", format="%.2f"),
    "t2": st.column_config.NumberColumn("t2", help="Target 2 = +2×ATR. Stretch/runner target.", format="%.2f"),
    "risk%": st.column_config.NumberColumn("risk%", help="Distance to stop as % of price = position risk. Size so a name's stop-out is survivable.", format="%.2f"),
    "atr%": st.column_config.NumberColumn("atr%", help="Daily ATR as % of price = how much this name typically moves. Bigger = wider stops/targets.", format="%.2f"),
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
                      ["BTST (overnight)", "Intraday", "🎬 Replay (practice)", "BTST+1"],
                      index=0)
date = st.sidebar.date_input("As-of close", value=last.date(),
                             max_value=last.date())
if st.sidebar.button("↻ refresh"):
    st.cache_data.clear()
test_mode = st.sidebar.checkbox("🧪 Test mode (show live board off-hours)", value=False,
                                help="Bypass the market-closed gate so you can exercise the "
                                     "UI now. Off-hours Fyers data is UNRELIABLE (indicative "
                                     "prices, junk volume) — for layout/flow testing only.")
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
    "action": st.column_config.TextColumn("action", help="LONG = strong bar + above VWAP + RSI not weak + RS-leader, on this timeframe. AVOID = weak bar / below VWAP / regime-off. Long-only. NOTE: intraday direction has no validated overnight-grade edge — manage strictly by the stop."),
}


HELP_INTRADAY = """
### What am I looking at? How do I use it?

**Purpose:** find NSE-F&O cash stocks showing a *smart-money accumulation footprint*,
and act on the ONE validated edge — the **overnight BTST long** (buy near close, exit
next morning, ~+16–19bps net on 8yr data). Long-only. Nothing auto-executes.

**Two lanes — do NOT mix them:**
| Lane | Signal | Hold | Validated? |
|---|---|---|---|
| **Intraday** (1h/2h/15m scan, or SELL tab) | price-action strength/weakness | **square off SAME DAY** | ❌ no proven edge — context only, trade small |
| **BTST** (🌙 BTST-CARRY box) | full accumulation footprint at the close | **overnight**, exit next AM | ✅ the real edge |

**The day, step by step:**
1. **09:15–15:00** — watch. BUY tab shows ⏳ **FORMING** names (footprint building). Delivery% (the core leg) only confirms after the close, so nothing is final yet.
2. **~15:10–15:30** — the FORMING names that still hold the full footprint flip to 🌙 **BTST-CARRY** (top green box). **These are your overnight picks.** Enter LONG near the close.
3. **Overnight → next 09:15–09:30** — exit into morning strength.

**Where is BTST-CARRY?** BUY tab → the top **🌙 BTST-CARRY** box. Empty until ~15:10 or
when the market's closed. A name only carries overnight if it's flagged there — never
just because you bought it intraday earlier.

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

    # timeframe dropdown — right above the table, right-aligned
    _, ddcol = st.columns([3, 1])
    scan_tf = ddcol.selectbox("Timeframe → stock list",
                              ["Live snapshot", "1h", "2h", "15m"], index=0,
                              key="scan_tf",
                              help="Live snapshot = current price-action scan (5s). "
                                   "1h/2h/15m = scan the universe on that bar timeframe.")

    # ---- timeframe-driven stock list (1h / 2h / 15m bars) ----------------------
    if scan_tf != "Live snapshot":
        st.caption(f"Scanning the liquid universe on **{scan_tf} bars** — pre-filtered to "
                   "today's up-and-strong names, then ranked by the footprint on this "
                   "timeframe (bar close-strength · above-VWAP · RSI · RS). Long-only. "
                   "Refresh to re-scan.")
        with st.spinner(f"Scanning on {scan_tf} bars…"):
            sc = _tf_scan(scan_tf)
        if not sc["ok"] or sc["board"].empty:
            st.info(f"No {scan_tf} candidates — market closed / pre-open, or nothing "
                    "clears the footprint on this timeframe. During a live session this "
                    "fills. For today's delivery-confirmed picks use the BTST tab.")
        else:
            st.caption(f"scanned {sc['n_scanned']} shortlisted names · Nifty "
                       f"{sc.get('idx_ret', 0):+.2f}% · regime "
                       f"{'RISK-ON' if sc['risk_on'] else 'RISK-OFF'}")
            st.dataframe(sc["board"], use_container_width=True, hide_index=True,
                         column_config={**LIVE_COLS, **TF_COLS})
            st.caption("Entry/Stop/T1/T2 = ATR risk geometry on this timeframe, NOT a "
                       "forecast. Intraday direction has no overnight-grade validated "
                       "edge — trade small, manage by the stop.")
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
        bd = lb["board"]
        counts = bd["action"].value_counts().to_dict()
        scounts = bd["sell"].value_counts().to_dict()
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Regime", "RISK-ON" if lb["risk_on"] else "RISK-OFF")
        h2.metric("Nifty today", f"{lb.get('idx_ret', 0):+.2f}%")
        h3.metric("BUY (carry/forming)",
                  f"{counts.get('BTST-CARRY', 0) + counts.get('FORMING', 0)}")
        h4.metric("SELL (short/weak)",
                  f"{scounts.get('SHORT', 0) + scounts.get('WEAK', 0)}")

        buy_cols = ["symbol", "sector", "ltp", "day%", "clr", "character", "vol×",
                    "RS%", "btst", "exp_ON", "entry", "stop", "t1", "t2", "risk%",
                    "atr%", "action"]
        sell_cols = ["symbol", "sector", "ltp", "day%", "clr", "character", "vol×",
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
                st.dataframe(carry[buy_cols], use_container_width=True, hide_index=True,
                             column_config=LIVE_COLS)
            # ⏳ FORMING — watch list, may flip to CARRY near the close
            st.markdown("#### ⏳ FORMING — building (watch)")
            if forming.empty:
                b = bd.sort_values("clr", ascending=False).head(10)
                st.caption("No footprint building yet — top-10 by close-strength meanwhile:")
                st.dataframe(b[buy_cols], use_container_width=True, hide_index=True,
                             column_config=LIVE_COLS)
            else:
                st.caption(f"{len(forming)} building — may flip to 🌙 BTST-CARRY near the close.")
                st.dataframe(forming[buy_cols], use_container_width=True, hide_index=True,
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
                st.dataframe(s[sell_cols], use_container_width=True, hide_index=True,
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
    rc1, rc2 = st.columns([1, 2])
    rdate = rc1.date_input("Date", value=_last_date().date(), max_value=_last_date().date(),
                           key="replay_date")
    rtime = rc2.select_slider("Time (IST)",
                              options=[f"{h:02d}:{m:02d}" for h in range(9, 16)
                                       for m in (0, 15, 30, 45)
                                       if (h, m) >= (9, 15) and (h, m) <= (15, 30)],
                              value="13:00", key="replay_time")

    @st.cache_data(ttl=3600)
    def _replay(dstr, t):
        return live.replay_board(dstr, t)

    with st.spinner(f"Reconstructing the board as of {rdate} {rtime} (first load of a day "
                    "fetches ~250 names)…"):
        rb = _replay(pd.Timestamp(rdate).strftime("%Y-%m-%d"), rtime)
    if not rb["ok"] or rb["board"].empty:
        st.info("No data for that date/time (not a trading day, or candles unavailable). "
                "Try a recent trading day.")
        st.stop()
    bd = rb["board"]
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
    buy_cols = ["symbol", "sector", "ltp", "day%", "clr", "character", "vs_vwap%", "rsi7",
                "rsi14", "tone", "vol×", "RS%", "btst", "entry", "stop", "t1", "t2",
                "atr%", "action"]
    st.markdown("#### 🌙 BTST-CARRY / ⏳ FORMING (long footprint at this time)")
    long_side = bd[bd["action"].isin(["BTST-CARRY", "FORMING"])]
    if long_side.empty:
        st.caption("None building at this time — top-10 by close-strength so far:")
        long_side = bd.sort_values("clr", ascending=False).head(10)
    st.dataframe(long_side[buy_cols], use_container_width=True, hide_index=True,
                 column_config={**LIVE_COLS, **TF_COLS})
    st.caption("Practice: note the FORMING names now, scrub to 15:15, see which held into "
               "🌙 BTST-CARRY — those were the overnight picks. VWAP/RSI/tone are point-in-time.")
    st.stop()
if tf == "BTST+1":
    st.title("BTST+1 — proven dead")
    st.error("Holding the position into day 2 is negative every year on 8yr data "
             "(the overnight gap mean-reverts intraday on day 2). Not offered. "
             "Capture the overnight gap only.")
    st.stop()

# ── BTST board ─────────────────────────────────────────────────────────────────
b = _board(pd.Timestamp(date).strftime("%Y-%m-%d"))
risk_on = b["risk_on"]

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
c1.title("Equity BTST Board")
with st.expander("❓ What is this tab — the validated overnight edge"):
    st.markdown(
        "**This is the confirmed BTST engine** (runs off the last close, after NSE "
        "publishes delivery%). It ranks the accumulation footprint — strong close + high "
        "delivery% + delivery-spike + volume surge + up-day + close>VWAP + persistent "
        "RS-leader — regime-gated (Nifty>50MA), sector-capped, earnings-guarded, liquid "
        "names only.\n\n**Edge:** ~+16–19bps net/night, long-only, 8yr-validated. **Plan:** "
        "LONG near the close (15:15–15:30), exit next 09:15–09:30. Paper-first — nothing "
        "auto-executes.\n\n**Intraday tab** = the live version of this (watch footprints "
        "form → 🌙 BTST-CARRY at ~15:10). **This tab** = the delivery-confirmed truth after "
        "the bell.")
c2.metric("Regime", "RISK-ON" if risk_on else "RISK-OFF",
          "Nifty > 50MA" if risk_on else "Nifty < 50MA")
c3.metric("Footprint hits", b["n_footprint"])
c4.metric("Deployable edge", "≈ +16–19 bps", "net/night, liquid, long-only")

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
buys = b["buys"]
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
    show = show.rename(columns={"deliv_per": "deliv%", "deliv_spike": "spike",
                                "vol_ratio": "vol×", "close_price": "entry≈"})
    cols = ["action", "symbol", "sector", "entry≈", "clr", "deliv%", "spike",
            "vol×", "day%", ">vwap%", "RS10%", "score", "wt%"]
    st.dataframe(show[cols].round(2), use_container_width=True, hide_index=True)
    st.caption("Plan: LONG near close, equal-weight, exit next-morning strength "
               "(VWAP reclaim / RSI roll). Size each for a ~-8% shock gap (unhedged "
               "overnight beta). Exclude any name reporting results tonight (event risk).")

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
