"""
arb.py — the arbitrage spreads, read onto this cash-equity board.

WHY THIS EXISTS, AND WHY IT IS NOT A LIST OF 21 STRATEGIES
    The classic arbitrage menu (index cash-and-carry, calendars, ETF-NAV, triangular,
    pairs, dispersion, convertible, merger, ADR, index-rebalance, closing-auction) was
    triaged against TWO hard constraints: what an Indian retail cash-equity account can
    actually execute, and what this archive can actually compute. Most of the menu fails
    one or both. What survived is not a trade at all — it is the CARRY SPREAD, used as a
    confirmation layer on the cash signal this board already trades.

    Every number quoted below was measured on this archive: 273 F&O names, 94,117
    stock-days, 451 sessions, 2024-07-26 .. 2026-08-14 (the F&O bhavcopy starts
    2024-07-24; the cash spine goes back to 2018 but futures do not).

WHY CASH-AND-CARRY IS NOT A TRADE FOR THIS ACCOUNT
    The Nifty near-month annualised basis has a MEDIAN of 6.23% here (5th pct -0.15%,
    95th 12.84%). Single-stock near-month basis medians 5.56%; the near-to-next calendar
    spread medians 6.31%. Those are not mispricings. That IS the cost of carry — the
    Mumbai Interbank Offered Rate plus a stock-lending fee — and it is what the Rs 2 lakh
    crore of Indian arbitrage-fund AUM (Kotak / Edelweiss / ICICI equity arbitrage) is
    harvesting. It compensates capital, not skill.

    The retail arithmetic kills it outright. Over a typical 25-day cycle a 6.2% annualised
    basis is 0.43% gross. Against that: Securities Transaction Tax (STT) 0.1% on the cash
    buy, stamp duty and exchange charges and Goods and Services Tax (GST) on top, STT
    0.02% on the futures sell, and then — since SEBI moved single-stock derivatives to
    PHYSICAL SETTLEMENT in October 2019 — the expiry delivery leg attracts another 0.1%.
    Round trip lands near 0.25%. Net is roughly 0.18% over 25 days, about 2.6% annualised,
    on 100% of the cash outlay plus futures margin. A liquid fund beats it. And the
    REVERSE trade (futures cheap, short the cash) is not available at all: India has no
    retail overnight short in cash equity, and the Securities Lending and Borrowing (SLB)
    window is too thin and too expensive to stand in for one.

    So the basis is not harvested here. It is READ.

WHAT THE MENU LOOKS LIKE AFTER THE TRIAGE
    NOT EXECUTABLE — structural, no amount of code fixes it:
      ETF-NAV and Index-ETF-Futures triangular — creation and redemption in India is
        Authorised-Participant-only in crore-sized baskets. Indian ETF premia are real and
        can be large (the overseas-fund premia that opened when SEBI's offshore investment
        limit froze creation) but retail can only PAY the premium, never arb it.
      American Depositary Receipt (ADR) / cross-listed — the Infosys and HDFC Bank ADR
        premia persist BECAUSE capital controls make fungibility one-way. The spread is
        the control, not an error.
      Convertible, capital-structure, and fixed-income relative value — India has no
        liquid convertible market and the corporate bond secondary is over-the-counter and
        thin. Nothing in this archive prices either leg.
      Merger / risk arbitrage — real under the SEBI Substantial Acquisition of Shares and
        Takeovers (SAST) open-offer regime, but it needs a deal feed and Competition
        Commission of India / National Company Law Tribunal timeline modelling, and the
        holding period is 6-18 months. Wrong instrument, wrong horizon, no data.
      Closing-auction / Volume-Weighted Average Price (VWAP) / flow — needs tick and order
        book. This archive is End Of Day (EOD).
      Dispersion, correlation, volatility, and volatility-surface arbitrage — implied
        volatility IS computable from the option closes here, but execution is a multi-leg
        option book across 200 names. Also already tested in this book: the
        volatility-risk-premium harvest modelled +47 and was RETRACTED as an artifact, and
        the real-price iron fly was thin and fragile.
      Index rebalancing — genuinely real (NSE announces changes ahead, passive funds must
        trade the effective close), but no constituent-change history exists here and the
        sample is a handful of names per semi-annual review.

    COMPUTABLE, MEASURED, DEAD:
      Calendar spread (near vs next), both index and single stock — as a forward signal it
        is nothing. Sorted into quintiles against the next overnight gap: +2.27bps,
        t = +1.49, and not monotone. Conditioned on the close-strength gate it INVERTS
        (cheap-roll +8.16bps t +2.17 against rich-roll +6.49bps t +0.88). Displayed as
        context; wired to no decision.
      Dividend arbitrage — not an edge here, a TRAP, and finding that out is most of the
        value in this module. See the block on _EXDIV_THR.
      Pairs / statistical / sector-stock / index-constituent arbitrage — these need a
        short leg. Cash cannot, but the 273 single-stock futures can, so they are the only
        genuinely open door on the list. NOT MEASURED YET. Nothing here claims they work.

    THE ONE THING THAT PAYS ON THIS BOARD — carry as a confirmation, not a trade. See below.

CARRY AS A CONFIRMATION LAYER
    The headline result is a trap, and it has to be said first. Sorting the universe by
    annualised basis against the next session's close-to-close return prints +31.97bps
    top-minus-bottom at t = +13.49. That number is mostly FALSE. Two tells: it does not
    decay (skip a single day and it collapses to +2.98bps, t = +1.28 — information decays,
    one-off accounting adjustments vanish exactly like that), and 40.5% of the
    most-negative-basis bucket turns out to be an ex-dividend date. A stock about to go
    ex-dividend has a futures price that already excludes the dividend, so its basis is
    deeply negative; on the ex-date the CASH price drops by the dividend and the raw
    return records a loss that was never a loss. It is not momentum in disguise — the
    correlation of basis with the same day's own return is 0.002 and the placebo of
    today's return alone gives t = +0.42 — it is dividend accounting.

    With ex-dividend days removed the residual is real but SMALL, and it matters ENORMOUSLY
    which removal rule is used. The diagnosis above used the NEXT session's basis snap-back
    to identify ex-dates — fine for proving the artifact exists, useless for trading, because
    it is not knowable at decision time. Every number below uses the CAUSAL rule the board
    actually ships (_EXDIV_THR, today's cross-section only):

        close-strength alone (CLR >= 0.66)             +10.21bps   t +3.27   n=26,861
        close-strength AND carry quintile 1-2           +5.44bps   t +1.64   n=10,509
        close-strength AND carry quintile 4-5          +15.40bps   t +4.59   n=10,105
        close-strength AND carry quintile 5            +20.91bps   t +5.85   n= 4,957
        carry quintile 5 alone (no close-strength)     +16.03bps   t +4.86   n=18,851

    THE ORDERING IS REAL AND NOT ONE CELL CLEARS THE 22bps COST FLOOR. The best is +20.91bps,
    short by 1.1bps. An earlier revision of this module reported +24.35bps for that cell and
    stated that it cleared the floor; that figure came from the ORACLE ex-dividend rule and is
    withdrawn. Scoring a signal with data unavailable at decision time is how two earlier
    'edges' in this book were retracted, and it is not repeated here.

    So carry does not create a trade. It ORDERS a list the engine has already chosen, and it
    REMOVES names that are about to go ex-dividend. See the WHAT DO I ACTUALLY DO section on
    the page for the two concrete instructions and the four measured do-nots.

    HORIZON — the read is ONE NIGHT. Held to the next open it is +20.91bps (t +5.85); to the
    next close +15.10bps (t +2.26); +2d +7.87bps (t +0.81); +3d +4.08bps (t +0.35). Gone.
    This independently reproduces the board's existing 'sell into the open' rule.

    NO SHORT SIDE. Carry does not invert. A weak close with carry quintile 1 gives
    -5.13bps (t -0.64) — nothing — and weak-closing names with RICH carry still gap
    +14.66bps UP (t +3.78). Nothing here supports a short.

    It holds up where it should. Split in half by time, +23.11bps (t +6.05) and +25.78bps
    (t +4.75). Demeaned WITHIN each symbol, so no static per-stock tilt can carry it,
    +18.19bps (t +4.32) — so roughly two thirds is a genuine time-varying signal and one
    third is a stable name preference. It is strongest in the most liquid quartile
    (+21.2bps top quintile), so it is not a thin-name filling artifact. Stacking delivery
    excess on top HURT it (+13.28bps), so it does not compose with everything.

WHAT THIS IS NOT
    1. NOT AN ARBITRAGE YOU CAN PUT ON. Nothing in this module is executable as a spread
       trade from a retail cash account. It reads a spread that other people trade.
    2. THE CARRY CONFIRMATION IS NOT WIRED TO ANY DECISION. Like the sector tilt and the
       F&O columns, it is CONTEXT. No filter consumes it, the engine does not read it.
       Its measured lift is +14bps over a 22bps cost floor, which is inside the noise of
       one bad fill.
    3. IT IS EOD, NOT LIVE. The F&O bhavcopy publishes after the close, so during a
       session the carry describes YESTERDAY's futures against a live cash price. The
       staleness is printed, never assumed away.
    4. THE EX-DIVIDEND FLAG IS A PROBABILITY, NOT A CALENDAR. There is no corporate-action
       table here. It is inferred from the basis itself, and it is right about 44% of the
       time against a noisy proxy for truth — but the flagged names' overnight gap is
       -1.2bps (t -0.35) against +8.4bps (t +2.57) for everything else, which is the
       separation that matters. Treat a flag as "do not carry this one overnight", never
       as "this stock pays a dividend tomorrow".
    5. ONLY THE F&O NAMES HAVE A CARRY AT ALL. A name with no futures has no basis, and a
       blank is the correct answer.
"""
from __future__ import annotations

import datetime as dt
import functools

import numpy as np
import pandas as pd

from . import data

# Minimum days to expiry. Inside the last two sessions the basis collapses toward zero
# mechanically (convergence), so an annualised read there is division by almost nothing.
_MIN_DTE = 3

# Implied dividend, as a percent of spot, above which a name is flagged as a likely
# pending ex-date. Chosen by sweeping the threshold against a next-session basis snap-back
# as ground truth: 0.15 flags 21.8% of the board, 0.50 flags 7.3%. 0.25 sits at the knee —
# it flags 13.3% (about 28 names a day), and those names' overnight gap is -1.2bps against
# +8.4bps for the rest. Below 0.20 the flag starts eating clean names; above 0.30 recall
# falls without precision improving much (44.2% at 0.30 against 43.6% here).
_EXDIV_THR = 0.25

# The carry quintile that counts as CONFIRMED. Measured on the causal rule: quintile 5 with
# close-strength is +20.91bps (t +5.85); widening to 4-5 dilutes it to +15.40bps. NEITHER
# clears the 22bps cost floor -- 5 is chosen because it is the strongest ORDERING, not because
# it is profitable on its own. A ✅ is a tie-breaker, never a trigger.
_CONFIRM_Q = 5

HELP_CARRY = (
    "**Carry** — the single-stock futures basis, annualised: how much more (or less) than "
    "the cash price the futures market will pay to hold this name to expiry.\n\n"
    "`(near-month futures ÷ cash close − 1) × 365 ÷ DTE (Days To Expiry)`\n\n"
    "Across this archive the median is **+5.6%** — that is the cost of carry (roughly the "
    "Mumbai Interbank Offered Rate plus a stock-lending fee), not a mispricing, and it is "
    "what arbitrage funds harvest. You cannot: a retail cash-and-carry nets about 2.6% "
    "annualised after Securities Transaction Tax (STT) and physical settlement, which a "
    "liquid fund beats.\n\n"
    "**Read it as positioning, not as an arbitrage.** RICH carry = the derivatives book is "
    "paying up to be long alongside you. CHEAP or negative carry = either heavy short "
    "demand, or — far more often — a dividend the futures have already priced out.\n\n"
    "✅ **confirm** = top carry quintile, and it is a **TIE-BREAKER, NOT A TRIGGER**. "
    "Measured only alongside the close-strength footprint: that pair is **+20.91bps** "
    "overnight (t +5.85) against **+10.21bps** for close-strength alone and **+5.44bps** "
    "when the derivatives book is NOT paying to carry the name. The ordering is real, but "
    "**no combination clears the 22bps cost floor** — +20.91 against 22. So prefer ✅ among "
    "names the engine has already picked; never open a position because of one. Carry alone, "
    "with no footprint, is +16.03bps — also under the floor. There is no short side: a weak "
    "close with cheap carry is −5.13bps (t −0.64).\n\n"
    "⚠️ **ex-div?** = carry sits more than 0.25% of spot below the day's median. Usually a "
    "pending ex-dividend date: the cash price drops by the dividend and your overnight "
    "carry eats it. Flagged names average **-1.2bps** overnight against **+8.4bps** for "
    "the rest. Inferred from the basis, not from a corporate-action calendar.\n\n"
    "EOD (End Of Day) — the F&O bhavcopy publishes after the close, so intraday this is "
    "yesterday's futures against a live cash price."
)

HELP_MARKET = (
    "**The market's cost of carry** — Nifty near-month futures against the Nifty 50 spot, "
    "annualised.\n\n"
    "Median here is **+6.2%**; it is negative on only **5.8%** of sessions. This is the "
    "rate leveraged longs are paying to stay long the index, so it doubles as a crude "
    "risk-appetite gauge: rich carry = longs bidding for leverage, flat or negative = "
    "hedging demand overwhelming it, which happens around stress and around large index "
    "dividend clusters.\n\n"
    "**Roll** is the near-to-next calendar spread, annualised — effectively the "
    "stock-lending rate the board is paying to hold positions past expiry (median "
    "**+6.3%**).\n\n"
    "Neither predicts anything on this board. Sorted against the next overnight gap, the "
    "roll spread gives +2.3bps at t +1.5 and is not monotone. Shown as the weather, not as "
    "a forecast."
)


def _fmt_pct(x: float) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:+.1f}%"


@functools.lru_cache(maxsize=8)
def _raw(asof: str) -> pd.DataFrame:
    """Near and next single-stock futures against the cash close, for ONE session.

    Keyed on the ISO date so a re-render is free. Everything downstream is derived from
    this frame, and nothing in it reads a bar later than `asof` — the whole module is
    as-of by construction because it only ever selects a single trade_date.
    """
    sql = """
        with f as (
          select symbol, expiry_date, close_price fut, open_interest oi, contracts,
                 row_number() over (partition by symbol order by expiry_date) rn,
                 date_diff('day', trade_date, expiry_date) dte
          from fno_bhavcopy
          where instrument = 'FUTSTK' and trade_date = ? and expiry_date >= ?
        ),
        s as (
          select symbol, close_price spot, turnover_lacs
          from daily_data
          where trade_date = ? and series = 'EQ' and close_price > 0
        )
        select f.symbol, f.rn, f.dte, f.fut, f.oi, f.contracts, s.spot, s.turnover_lacs
        from f join s using (symbol)
        where f.rn <= 2
    """
    with data._connect() as c:
        d = c.execute(sql, [asof, asof, asof]).df()
    if d.empty:
        return d
    near = d[d.rn == 1].set_index("symbol")
    nxt = d[d.rn == 2].set_index("symbol")
    out = pd.DataFrame({
        "spot": near.spot, "fut": near.fut, "dte": near.dte, "oi": near.oi,
        "turnover_lacs": near.turnover_lacs,
        "fut_next": nxt.fut.reindex(near.index), "dte_next": nxt.dte.reindex(near.index),
    })
    return out.reset_index()


def carry(asof: dt.date | str) -> pd.DataFrame:
    """Per-symbol carry read for one session.

    Columns: symbol, basis_pct, basis_ann, roll_ann, div_impl, exdiv, q (1-5 carry
    quintile), confirm, tag. Empty frame if that session has no futures bhavcopy.
    """
    key = pd.Timestamp(asof)
    if pd.isna(key):
        return pd.DataFrame()
    d = _raw(key.strftime("%Y-%m-%d"))
    if d.empty:
        return pd.DataFrame()

    d = d[(d.dte >= _MIN_DTE) & d.fut.notna() & d.spot.gt(0)].copy()
    if d.empty:
        return pd.DataFrame()

    d["basis_pct"] = (d.fut / d.spot - 1.0) * 100.0
    d["basis_ann"] = d.basis_pct * 365.0 / d.dte
    # The roll is the SAME kind of number one expiry out: what the next contract charges
    # over the near one, annualised across the gap between the two expiries.
    span = (d.dte_next - d.dte).replace(0, np.nan)
    d["roll_ann"] = (d.fut_next / d.fut - 1.0) * 100.0 * 365.0 / span

    # Implied dividend, CAUSAL: today's basis against the day's own cross-sectional median
    # carry. Deliberately not a per-symbol historical mean — a name's own history is
    # contaminated by its own past ex-dates, and the cross-section prices the same interest
    # rate for everyone on the same day. Median, not mean, because the ex-date names are
    # exactly the outliers being detected.
    med = float(d.basis_pct.median())
    d["div_impl"] = med - d.basis_pct
    d["exdiv"] = d.div_impl > _EXDIV_THR

    # Quintile on the annualised basis. Ranked first so ties (dead futures printing the
    # same close) cannot collapse a bin and silently drop names.
    if d.basis_ann.nunique() >= 5:
        d["q"] = pd.qcut(d.basis_ann.rank(method="first"), 5, labels=False) + 1
    else:
        d["q"] = np.nan
    d["confirm"] = (d.q >= _CONFIRM_Q) & ~d.exdiv

    def _tag(r) -> str:
        if r.exdiv:
            # Show the IMPLIED DIVIDEND, not the annualised basis. Annualising a
            # dividend-distorted basis produces a number like -68% (the 365/DTE multiplier
            # scaling a one-off cash payment), which reads as a catastrophe rather than as
            # a 2% payout. The dividend, as a percent of spot, is the number you act on:
            # it is roughly what the cash price drops on the ex-date.
            return f"⚠️ ex-div? ~{r.div_impl:.1f}%"
        if r.confirm:
            return f"✅ {_fmt_pct(r.basis_ann)}"
        return _fmt_pct(r.basis_ann)

    d["tag"] = d.apply(_tag, axis=1)
    return d[["symbol", "basis_pct", "basis_ann", "roll_ann", "div_impl", "exdiv",
              "q", "confirm", "tag", "dte", "turnover_lacs"]].sort_values("symbol")


def market(asof: dt.date | str) -> dict:
    """Index-level carry: Nifty basis, its percentile against history, universe roll."""
    key = pd.Timestamp(asof)
    out = {"nifty_ann": np.nan, "pctile": np.nan, "roll_med": np.nan,
           "carry_med": np.nan, "n_exdiv": 0, "n": 0, "stale_days": None, "asof": None}
    if pd.isna(key):
        return out
    iso = key.strftime("%Y-%m-%d")
    sql = """
        with f as (
          select trade_date, close_price fut,
                 date_diff('day', trade_date, expiry_date) dte,
                 row_number() over (partition by trade_date order by expiry_date) rn
          from fno_bhavcopy
          where instrument = 'FUTIDX' and symbol = 'NIFTY' and expiry_date >= trade_date
            and trade_date <= ?
        ),
        i as (select trade_date, close_val spot from index_data where index_name = ?)
        select f.trade_date, f.fut, f.dte, i.spot
        from f join i using (trade_date)
        where f.rn = 1 and f.dte >= ?
        order by f.trade_date
    """
    from . import config
    with data._connect() as c:
        ni = c.execute(sql, [iso, config.NIFTY_INDEX_NAME, _MIN_DTE]).df()
    if not ni.empty:
        ni["ann"] = (ni.fut / ni.spot - 1.0) * 100.0 * 365.0 / ni.dte
        cur = ni.iloc[-1]
        out["nifty_ann"] = float(cur.ann)
        out["pctile"] = float((ni.ann <= cur.ann).mean() * 100.0)
        out["asof"] = pd.Timestamp(cur.trade_date)
        out["stale_days"] = int((key.normalize() - pd.Timestamp(cur.trade_date).normalize()).days)

    d = carry(key)
    if not d.empty:
        out["roll_med"] = float(d.roll_ann.median(skipna=True))
        out["carry_med"] = float(d.basis_pct.median())
        out["n_exdiv"] = int(d.exdiv.sum())
        out["n"] = int(len(d))
    return out


def annotate(df: pd.DataFrame, asof: dt.date | str, col: str = "carry") -> pd.DataFrame:
    """Attach the carry tag to a board table, matched on its `symbol` column.

    Degrades to an em-dash rather than raising: a missing bhavcopy, a locked archive or a
    non-F&O name must not take the board down over a context column.
    """
    if df is None or df.empty or "symbol" not in df.columns:
        return df
    out = df.copy()
    try:
        d = carry(asof)
    except Exception:
        d = pd.DataFrame()
    if d.empty:
        out[col] = "—"
        return out
    out[col] = out.symbol.map(d.set_index("symbol").tag).fillna("—")
    return out


def nifty_carry_history(asof: dt.date | str, days: int = 500) -> pd.DataFrame:
    """Nifty near-month annualised basis, one row per session up to `asof`.

    The chart behind the 'cost of carry' metric. Strictly `<= asof` so the replay lane and
    a back-dated as-of both draw the curve the board would actually have seen.
    """
    key = pd.Timestamp(asof)
    if pd.isna(key):
        return pd.DataFrame()
    from . import config
    sql = """
        with f as (
          select trade_date, close_price fut,
                 date_diff('day', trade_date, expiry_date) dte,
                 row_number() over (partition by trade_date order by expiry_date) rn
          from fno_bhavcopy
          where instrument = 'FUTIDX' and symbol = 'NIFTY'
            and expiry_date >= trade_date and trade_date <= ?
        ),
        i as (select trade_date, close_val spot from index_data where index_name = ?)
        select f.trade_date, f.fut, f.dte, i.spot
        from f join i using (trade_date)
        where f.rn = 1 and f.dte >= ?
        order by f.trade_date desc limit ?
    """
    with data._connect() as c:
        d = c.execute(sql, [key.strftime("%Y-%m-%d"), config.NIFTY_INDEX_NAME,
                            _MIN_DTE, int(days)]).df()
    if d.empty:
        return d
    d["carry_ann"] = (d.fut / d.spot - 1.0) * 100.0 * 365.0 / d.dte
    d["trade_date"] = pd.to_datetime(d.trade_date)
    return d.sort_values("trade_date")[["trade_date", "carry_ann"]]


# ── the strategy triage, as DATA ──────────────────────────────────────────────
# The full menu, each row carrying the verdict FOR AN INDIAN RETAIL CASH ACCOUNT reading
# this archive. Kept as a table rather than prose so the page cannot quietly disagree with
# the module docstring, and so a verdict that changes has exactly one place to change.
#
# STATUS vocabulary, and it is deliberately blunt:
#   BLOCKED   — structurally impossible for this account. No amount of code fixes it.
#   NO DATA   — the legs are not priced in this archive, and cannot be.
#   NULL      — computable here, MEASURED, and found to carry nothing.
#   OPEN      — genuinely available and NOT yet measured. No claim either way.
#   CONTEXT   — shipped, but as a read, not as a trade.
_S = [
    (1, "Index Futures–Cash Arbitrage", "BLOCKED",
     "The real Indian arb trade — and the spread IS the carry rate, not a mispricing. "
     "Nifty basis medians **+6.23%** annualised here. Over a 25d cycle that is 0.43% gross "
     "against ~0.25% of cost (STT 0.1% cash buy + 0.02% futures sell + another 0.1% on the "
     "physical-settlement leg since Oct-2019), leaving ~**2.6% annualised** on 100% of the "
     "cash outlay. A liquid fund beats it. The reverse leg needs an overnight cash short, "
     "which India does not offer retail."),
    (2, "Futures Calendar Arbitrage", "NULL",
     "Median roll **+6.31%** annualised = the lending rate, again not a mispricing. Tested "
     "as a forward signal: **+2.27bps, t +1.49**, not monotone. Shown on this page as "
     "weather; wired to nothing."),
    (3, "Stock Futures Calendar Arbitrage", "NULL",
     "Same spread per name, plus physical-settlement obligations on BOTH legs at expiry. "
     "Conditioned on the close-strength gate it inverts (cheap-roll +8.16bps t +2.17 "
     "against rich-roll +6.49bps t +0.88) — i.e. noise."),
    (4, "ETF–NAV Arbitrage", "BLOCKED",
     "Indian ETF premia are real and can be very large (the overseas-fund premia that "
     "opened when SEBI's offshore limit froze creation). But creation and redemption is "
     "Authorised-Participant-only in crore-sized baskets. Retail can only PAY the premium."),
    (5, "Index–ETF–Futures Triangular", "BLOCKED",
     "Needs the ETF leg from #4. Same gate."),
    (6, "Pairs Trading", "OPEN",
     "Needs a short leg. Cash cannot hold one overnight in India — but the **273 single-stock "
     "futures** can, which makes this one of the few genuinely open doors on the list. "
     "**Not measured in this book.** Nothing here claims it works."),
    (7, "Statistical Arbitrage", "OPEN",
     "Same short-leg logic as #6. Note the archive is short: F&O starts **2024-07-24**, so "
     "roughly 2 years — thin for a cross-sectional model with many parameters."),
    (8, "Sector–Stock Statistical Arbitrage", "OPEN",
     "The sector spine already exists (canonical sectors + the tilt engine), so the residual "
     "is computable today. Related work in the wider book found a within-sector stock pick "
     "does carry information at 10 days. Unmeasured HERE, at this horizon."),
    (9, "Index–Constituent Arbitrage", "OPEN",
     "Basket-versus-index replication. Executable via stock futures, but 50 legs of cost on "
     "a spread that #1 already shows is roughly the carry rate."),
    (10, "Dispersion Trading", "BLOCKED",
     "Implied volatility IS computable from the option closes here, but the trade is a "
     "multi-leg option book across ~200 names, rebalanced. Not a retail cash account."),
    (11, "Correlation Arbitrage", "BLOCKED", "Same execution gate as #10."),
    (12, "Volatility Arbitrage", "NULL",
     "Already tested in this book: the volatility-risk-premium harvest modelled +47 and was "
     "**RETRACTED as an artifact**; the real-price iron fly was thin and fragile with "
     "confidence intervals straddling zero."),
    (13, "Volatility Surface Arbitrage", "BLOCKED",
     "Needs a live surface and multi-leg execution. This archive is End Of Day."),
    (14, "Convertible Arbitrage", "NO DATA",
     "India has effectively no liquid convertible market. Neither leg is priced here."),
    (15, "Capital Structure Arbitrage", "NO DATA",
     "Needs credit spreads against equity. The Indian corporate bond secondary is "
     "over-the-counter and thin; nothing in this archive prices it."),
    (16, "Fixed-Income Relative Value", "NO DATA",
     "Not an equity strategy, and no bond curve exists in this archive."),
    (17, "Merger / Risk Arbitrage", "NO DATA",
     "Genuinely real in India under the SEBI Substantial Acquisition of Shares and Takeovers "
     "(SAST) open-offer regime and delisting reverse book-building. But it needs a deal feed "
     "plus Competition Commission of India / National Company Law Tribunal timeline "
     "modelling, and holds 6–18 months. Wrong horizon for an overnight board."),
    (18, "ADR / Cross-Listed Arbitrage", "BLOCKED",
     "The Infosys and HDFC Bank ADR premia persist **because** capital controls make "
     "fungibility one-way. The spread is the control, not an error to be corrected."),
    (19, "Dividend Arbitrage", "CONTEXT",
     "Not an edge here — a **TRAP**, and catching it is most of what this page is worth. "
     "It is why the raw basis signal prints t +13.5 and why that number is false. Shipped "
     "INVERTED: as the ⚠️ ex-dividend exclusion, not as a trade."),
    (20, "Index Rebalancing Arbitrage", "NO DATA",
     "Real and documented (NSE announces changes ahead; passive funds must trade the "
     "effective close). But no constituent-change history exists in this archive, and the "
     "sample is a handful of names per semi-annual review."),
    (21, "Closing-Auction / VWAP / Flow", "BLOCKED",
     "Needs tick and order-book data. This archive is End Of Day, and retail has no "
     "Volume-Weighted Average Price guarantee to arbitrage against."),
]
_STATUS_ICON = {"BLOCKED": "🔴 blocked", "NO DATA": "⚫ no data", "NULL": "🟡 measured null",
                "OPEN": "🔵 open, unmeasured", "CONTEXT": "🟢 shipped as context"}


def strategies() -> pd.DataFrame:
    """The 21-strategy triage as a frame: #, strategy, status, why."""
    return pd.DataFrame(
        [{"#": n, "strategy": s, "status": _STATUS_ICON[k], "_k": k, "why": w}
         for n, s, k, w in _S])


def render_page(asof, st) -> None:
    """The Arbitrage page. `st` is injected so this module never imports streamlit."""
    st.title("🧮 Arbitrage — what is real in India, and what this board can use")
    st.caption(f"As-of close **{pd.Timestamp(asof):%d %b %Y}** · archive: 273 F&O names, "
               f"94,117 stock-days, 451 sessions (F&O bhavcopy begins 2024-07-24)")

    st.error(
        "**Nothing on this page is a trade you can put on.** Every classic arbitrage was "
        "triaged against two hard limits — what an Indian retail cash account can execute, "
        "and what this archive can price. Almost none survive. What survives is a SPREAD "
        "OTHER PEOPLE TRADE, read as confirmation on the cash footprint this board already "
        "has. Treat the whole page as context.")

    with st.expander("👉 **SO WHAT DO I ACTUALLY DO?** — read this before anything else",
                     expanded=True):
        st.markdown(
            "**You do not arbitrage anything. You do not buy or sell off this page at all.** "
            "There is no leg-against-leg trade here, no buy-this-sell-that. The carry number "
            "is a spread that arbitrage funds and proprietary desks trade with balance sheet "
            "and stock-lending access you do not have — and after Securities Transaction Tax "
            "and physical settlement it nets a retail account about **2.6% annualised**, which "
            "a liquid fund beats while you sleep.\n\n"
            "This page changes **exactly two things**, and both are modifications to the BTST "
            "board's existing long list. Neither creates a position.\n\n"
            "---\n"
            "#### 1. ⚠️ ex-dividend → REMOVE the name from tonight's list  *(the real win)*\n"
            "This is the only instruction here worth acting on, and it works because it is a "
            "**subtraction**, not a new trade. Flagged names average **−1.2bps** overnight "
            "against **+8.4bps** for everything else. Dropping them costs nothing, needs no "
            "extra fill, and removes a set that loses money. If you take one thing from this "
            "page, take this.\n\n"
            "#### 2. ✅ confirm → a TIE-BREAKER on names the engine already chose\n"
            "When the BTST board hands you more long candidates than you intend to fund, "
            "prefer the ✅ ones. Measured, the ordering is real: +20.91bps against +10.21bps "
            "for close-strength alone, and +5.44bps for names the derivatives book is NOT "
            "paying to carry.\n\n"
            "**But it does not clear the 22bps cost floor** (+20.91 against 22). So a ✅ is "
            "**never a reason to open a position you were not already going to open.** It "
            "orders a list. It does not lengthen one.\n\n"
            "---\n"
            "#### What NOT to do — each of these is measured, not cautionary boilerplate\n"
            "- **Do not buy a high-carry name on its own.** Carry quintile 5 with no "
            "close-strength footprint = +16.03bps, under the floor. APLAPOLLO at +27.4% carry "
            "is not a buy signal; it is a name the futures market is paying up to hold.\n"
            "- **Do not short low carry.** It does not work in either direction. Weak close "
            "with carry quintile 1 = **−5.13bps, t −0.64** — nothing. And weak-closing names "
            "with RICH carry still gap **+14.66bps UP** (t +3.78). Carry has no short side "
            "here.\n"
            "- **Do not hold past the open.** The whole effect is the overnight gap, and it is "
            "in your hands at the next open. By close+2d it is +7.87bps (t +0.81) and by "
            "close+3d +4.08bps (t +0.35) — gone. This matches the board's existing exit.\n"
            "- **Do not read a rich Nifty carry as a market call.** It is the cost of "
            "leverage, not a forecast.\n\n"
            "#### Scale check, so nobody mistakes this for money\n"
            "The best cell is **+20.91bps ≈ 0.21%** on a one-night hold, before the 22bps it "
            "costs to get in and out. On a ₹1,00,000 position that is about **₹209 gross, "
            "negative net**. The honest summary of this whole page is: **it tells you which "
            "names to DROP, and in what order to fund the rest. It does not find you a "
            "trade.**")

    st.subheader("How long the read lasts (✅ confirm + close-strength, cumulative)")
    st.dataframe(pd.DataFrame([
        {"held until": "next OPEN (the board's exit)", "cumulative": "+20.91bps", "t": "+5.85",
         "verdict": "the whole effect — already in hand"},
        {"held until": "next CLOSE (+1d)", "cumulative": "+15.10bps", "t": "+2.26",
         "verdict": "giving it back"},
        {"held until": "+2d", "cumulative": "+7.87bps", "t": "+0.81", "verdict": "dead"},
        {"held until": "+3d", "cumulative": "+4.08bps", "t": "+0.35", "verdict": "dead"},
        {"held until": "+5d", "cumulative": "+4.73bps", "t": "+0.31", "verdict": "dead"},
    ]), hide_index=True, use_container_width=True)
    st.caption("Independently confirms the board's existing rule: **sell into the next open.** "
               "Holding a second night has never paid in this book.")

    mk = market(asof)
    c1, c2, c3, c4 = st.columns(4)
    na = mk["nifty_ann"]
    c1.metric("Cost of carry (Nifty)", "—" if na != na else f"{na:+.1f}%", help=HELP_MARKET)
    c2.metric("Percentile (2yr)", "—" if mk["pctile"] != mk["pctile"] else f"{mk['pctile']:.0f}th",
              help="Where today's carry sits against the last two years. Median is +6.2%; "
                   "it is negative on only 5.8% of sessions.")
    c3.metric("Roll (near→next)", "—" if mk["roll_med"] != mk["roll_med"] else f"{mk['roll_med']:+.1f}%",
              help="The stock-lending rate. Measured null as a signal (t +1.5).")
    c4.metric("Ex-dividend flags", f"{mk['n_exdiv']} / {mk['n']}",
              help="Names whose carry sits >0.25% of spot below the day's median — usually "
                   "a pending ex-date. Their overnight gap averages −1.2bps.")
    if mk["stale_days"]:
        st.warning(f"F&O bhavcopy is **{mk['stale_days']}d** behind ({mk['asof']:%d %b}) — "
                   f"the carry on this page is that stale.")

    hist = nifty_carry_history(asof)
    if not hist.empty:
        st.line_chart(hist.set_index("trade_date")["carry_ann"], height=180)
        st.caption("Nifty near-month annualised basis. This is what leveraged longs pay to "
                   "stay long the index — a crude risk-appetite gauge. It is what arbitrage "
                   "funds harvest, and what a retail account cannot.")

    st.markdown("---")
    st.subheader("The menu, triaged for an Indian retail cash account")
    stg = strategies()
    counts = stg._k.value_counts()
    st.caption(" · ".join(f"**{counts.get(k, 0)}** {_STATUS_ICON[k]}"
                          for k in ["BLOCKED", "NO DATA", "NULL", "OPEN", "CONTEXT"]))
    pick = st.multiselect("Filter by status", list(_STATUS_ICON.values()),
                          default=list(_STATUS_ICON.values()))
    st.dataframe(stg[stg.status.isin(pick)][["#", "strategy", "status", "why"]],
                 hide_index=True, use_container_width=True,
                 column_config={
                     "#": st.column_config.NumberColumn("#", width="small"),
                     "strategy": st.column_config.TextColumn("strategy", width="medium"),
                     "status": st.column_config.TextColumn("status", width="small"),
                     "why": st.column_config.TextColumn("why — measured, not asserted",
                                                        width="large")})

    st.markdown("---")
    st.subheader("Why the raw basis signal is a trap")
    st.markdown(
        "Sorting the universe by annualised basis against the next session's close-to-close "
        "return prints **+31.97bps, t = +13.49**. That number is mostly **false**, and the "
        "two tells are worth knowing by heart:\n\n"
        "1. **It does not decay.** Skip a single day — measure t+1 to t+2 instead — and it "
        "collapses to **+2.98bps, t = +1.28**. Real information decays smoothly. A one-off "
        "accounting adjustment vanishes exactly like that.\n"
        "2. **40.5% of the most-negative-basis bucket is an ex-dividend date.** A stock about "
        "to go ex-dividend has a futures price that already excludes the dividend, so its "
        "basis is deeply negative. On the ex-date the CASH price drops by the dividend and "
        "the raw return records a loss that was never a loss.\n\n"
        "It is **not** momentum in disguise: the correlation of basis with the same day's own "
        "return is **0.002**, and the placebo of today's return alone gives **t = +0.42**. "
        "With ex-dividend names removed by the causal rule the board actually ships, the "
        "residual is real but small — carry quintile 5 on its own is **+16.03bps** (t +4.86) "
        "on the overnight gap, **below this board's 22bps cost floor**. It is not a trade.")

    st.subheader("What carry adds — and why it still does not clear the cost floor")
    st.caption("Every row is the overnight GAP, date-clustered, ex-dividend names removed by "
               "the SAME causal rule the board actually ships (not the next-day rule used to "
               "diagnose the artifact — that one is not available at decision time).")
    ev = pd.DataFrame([
        {"read": "close-strength footprint alone (CLR ≥ 0.66)", "gap": "+10.21bps",
         "t": "+3.27", "n": "26,861", "vs 22bps cost": "🔴 under"},
        {"read": "+ carry quintile 1–2 (book disagrees)", "gap": "+5.44bps", "t": "+1.64",
         "n": "10,509", "vs 22bps cost": "🔴 under"},
        {"read": "+ carry quintile 4–5", "gap": "+15.40bps", "t": "+4.59", "n": "10,105",
         "vs 22bps cost": "🔴 under"},
        {"read": "+ carry quintile 5 (✅ confirm) — BEST CELL", "gap": "+20.91bps",
         "t": "+5.85", "n": "4,957", "vs 22bps cost": "🔴 under, by 1.1bps"},
        {"read": "carry quintile 5 alone, ignoring the close", "gap": "+16.03bps",
         "t": "+4.86", "n": "18,851", "vs 22bps cost": "🔴 under"},
    ])
    st.dataframe(ev, hide_index=True, use_container_width=True)
    st.error(
        "**NOT ONE COMBINATION CLEARS 22bps.** The best cell on this board is +20.91bps. An "
        "earlier version of this page claimed +24.35bps and said it cleared the floor — that "
        "number came from removing ex-dividend names using the NEXT session's basis, which is "
        "not knowable at decision time. Under the causal rule the board can actually run, the "
        "lift is smaller and it lands **under** the floor. Corrected rather than kept, because "
        "a signal that clears its cost only when scored with tomorrow's data is exactly the "
        "kind of thing this project has had to retract twice before.")
    st.caption("What is nonetheless solid: it is **not decaying** — quarterly, +22.4 / +18.9 / "
               "+8.6 / +20.7 / +23.0 / +24.0 / +18.8 / +30.3 / +20.5bps, no trend, though "
               "several quarters sit at t < 1.3. It survives within-symbol demeaning "
               "(+18.19bps, t +4.32), so it is not a static per-stock tilt. It is strongest in "
               "the most LIQUID quartile, so not a thin-name filling artifact. It holds across "
               "the expiry cycle (DTE 3-7 +23.8, 8-15 +20.8, 16-25 +23.3) and weakens only far "
               "out (26-40d +13.0, t +1.84). Stacking delivery excess on top HURT it "
               "(+13.28bps). And it is **one night only** — see the horizon table above.")

    st.markdown("---")
    st.subheader("Today's carry board")
    d = carry(asof)
    if d.empty:
        st.info("No futures bhavcopy for this session — nothing to price.")
        return
    only = st.radio("Show", ["All", "✅ confirm only", "⚠️ ex-dividend only"],
                    horizontal=True, index=0)
    v = d if only == "All" else (d[d.confirm] if only.startswith("✅") else d[d.exdiv])
    v = v.assign(**{
        "basis %": v.basis_pct.round(2), "carry % (ann)": v.basis_ann.round(1),
        "roll % (ann)": v.roll_ann.round(1),
        # BLANK unless the name is actually flagged. On a rich-carry row div_impl is
        # NEGATIVE by construction (carry above the day's median), and printing "-0.60"
        # under a header reading "implied div %" invites reading it as a negative dividend.
        # It is evidence for the ex-dividend call and means nothing otherwise.
        "implied div %": v.div_impl.where(v.exdiv).round(2),
        "turnover ₹L": v.turnover_lacs.round(0), "DTE": v.dte,
    }).sort_values("basis_ann", ascending=False)
    st.dataframe(
        v[["symbol", "basis %", "carry % (ann)", "roll % (ann)", "implied div %",
           "q", "tag", "DTE", "turnover ₹L"]],
        hide_index=True, use_container_width=True, height=430,
        column_config={
            "basis %": st.column_config.NumberColumn(
                "basis %", format="%+.2f",
                help="Futures over cash, raw. Not annualised — this is the actual spread."),
            "carry % (ann)": st.column_config.NumberColumn(
                "carry % ann", format="%+.1f",
                help="The same spread annualised over Days To Expiry. Median across the "
                     "archive is +5.6%."),
            "roll % (ann)": st.column_config.NumberColumn(
                "roll % ann", format="%+.1f",
                help="Next month over near month, annualised — the lending rate. Measured "
                     "null as a signal."),
            "implied div %": st.column_config.NumberColumn(
                "implied div %", format="%.2f",
                help="Only filled on ⚠️ flagged rows. How far this name's carry sits BELOW "
                     "the day's median, as a percent of spot — roughly what the cash price "
                     "drops on the ex-date. Blank means not flagged; on a rich-carry name "
                     "this quantity is negative and meaningless, so it is not shown."),
            "q": st.column_config.NumberColumn(
                "quintile", format="%d",
                help="Carry quintile against the other F&O names TODAY. 5 = richest. Only "
                     "quintile 5, and only alongside the close-strength footprint, measured "
                     "above the cost floor."),
            "tag": st.column_config.TextColumn("carry", help=HELP_CARRY)})
    st.caption(f"{len(v)} of {len(d)} names · **{int(d.confirm.sum())}** ✅ confirm · "
               f"**{int(d.exdiv.sum())}** ⚠️ ex-dividend · the ~60 board names with no "
               f"futures are absent because they have no basis at all.")
    st.caption("The ex-dividend flag is **inferred from the basis, not read from a "
               "corporate-action calendar** — there is no such table in this archive. It is "
               "right about 44% of the time against a noisy proxy for truth, but flagged "
               "names' overnight gap is **−1.2bps (t −0.35)** against **+8.4bps (t +2.57)** "
               "for everything else, which is the separation that matters. Read a flag as "
               "'do not carry this one overnight', never as 'this stock pays tomorrow'.")


def clear_cache() -> None:
    _raw.cache_clear()
