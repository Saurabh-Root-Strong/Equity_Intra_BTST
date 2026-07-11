"""
screen.py — tonight's ranked accumulation candidates.

Combines the LOCKED conviction footprint (features.signal_mask) with the mandatory
regime gate (regime) and cross-sectional top-N selection. This is the nightly
output the trader (or the paper ledger) acts on: LONG these names near the close,
exit into next morning's strength. Long-only, overnight only.
"""
from __future__ import annotations

import pandas as pd

from . import config, data, events, features, indicators, portfolio, regime


def screen(date: pd.Timestamp | None = None, top_n: int | None = None) -> pd.DataFrame:
    """Return the ranked LONG candidates for `date`'s close (default: latest EOD).

    Columns: symbol, close_price, clr, deliv_per, deliv_spike, vol_ratio, ret,
    score, risk_on. Empty (with attrs['risk_on']) when the gate is off or no name
    clears the footprint.
    """
    date = pd.Timestamp(date) if date is not None else data.last_trading_date()
    top_n = top_n or config.TOP_N

    # need LOOKBACK+ days of history for the rolling medians
    start = (date - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    df = data.load_eod(start=start, end=date.strftime("%Y-%m-%d"))
    df = features.add_features(df)
    df = features.add_relative_strength(df, data.load_nifty(start=start, end=date.strftime("%Y-%m-%d")))

    day = df[df["trade_date"] == date].copy()
    risk_on = regime.is_risk_on(date)

    cand = day[features.signal_mask(day)].copy()
    if not cand.empty:
        earn = events.upcoming(date.date(), horizon_days=3)   # earnings guard (graceful)
        cand = cand[~cand["symbol"].isin(earn)]
    if not cand.empty:
        cand["score"] = features.conviction_score(cand)
        cand = cand.sort_values("score", ascending=False)
        cand = portfolio.select(cand)          # sector cap + top-N + equal weights
    else:
        cand["sector"] = []; cand["weight"] = []
    cand["risk_on"] = risk_on

    cols = ["trade_date", "symbol", "sector", "close_price", "clr", "deliv_trail",
            "deliv_per", "vol_ratio", "ret", "close_vs_vwap", "rs_idx_cum",
            "score", "weight", "risk_on"]
    out = cand[cols].reset_index(drop=True) if not cand.empty else \
        pd.DataFrame(columns=cols)
    out.attrs["risk_on"] = risk_on
    out.attrs["date"] = date
    return out


def board(date: pd.Timestamp | None = None, top_n: int | None = None) -> dict:
    """Full decision board for the dashboard: BUY list (selected, sized), AVOID
    list (footprint-passers rejected only by liquidity or the regime gate, with a
    reason), and the regime flag. Long-only — there is no SELL entry; SELL means
    exiting an open long (handled by the ledger)."""
    date = pd.Timestamp(date) if date is not None else data.last_trading_date()
    top_n = top_n or config.TOP_N
    start = (date - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    df = data.load_eod(start=start, end=date.strftime("%Y-%m-%d"))
    df = features.add_features(df)
    df = features.add_relative_strength(df, data.load_nifty(start=start, end=date.strftime("%Y-%m-%d")))
    day = df[df["trade_date"] == date].copy()
    risk_on = regime.is_risk_on(date)
    sectors = data.load_sectors()

    core = day[features.signal_mask(day, require_liquidity=False)].copy()
    core["sector"] = core["symbol"].map(lambda s: sectors.get(s, f"_{s}"))
    core["score"] = features.conviction_score(core) if not core.empty else []
    core = core.sort_values("score", ascending=False)

    liquid = core[core["turnover_lacs"] >= config.LIQ_MIN_LACS]
    thin = core[core["turnover_lacs"] < config.LIQ_MIN_LACS].copy()

    # EARNINGS GUARD: a name reporting results during the overnight hold gaps on the
    # earnings, not the accumulation. Exclude it from BUY. Degrades gracefully.
    guard = events.guard_status()
    earn = events.upcoming(date.date(), horizon_days=3) if guard["available"] else set()
    earners = liquid[liquid["symbol"].isin(earn)].copy()
    liquid = liquid[~liquid["symbol"].isin(earn)]

    if risk_on and not liquid.empty:
        buys = portfolio.select(liquid)
        buys["action"] = "BUY"
        bands = buys.apply(lambda r: indicators.band(r["close_price"], r.get("atr14", 0)),
                           axis=1)
        for col in ("band_lo", "band_hi", "range_lo", "range_hi", "exp_move%"):
            buys[col] = [b.get(col) for b in bands]
    else:
        buys = liquid.head(0).assign(sector=[], weight=[], action=[])

    avoids = []
    if not earners.empty:
        e = earners.copy(); e["reason"] = "EARNINGS — results during the hold"; avoids.append(e)
    if not thin.empty:
        t = thin.copy(); t["reason"] = "illiquid (<₹20cr turnover)"; avoids.append(t)
    if not risk_on and not liquid.empty:
        l = liquid.copy(); l["reason"] = "regime RISK-OFF (Nifty<50MA)"; avoids.append(l)
    avoid = pd.concat(avoids, ignore_index=True) if avoids else core.head(0).assign(reason=[])

    return {"date": date, "risk_on": risk_on, "buys": buys, "avoid": avoid,
            "n_footprint": len(core), "guard": guard, "n_earnings": len(earners)}


def format_screen(out: pd.DataFrame) -> str:
    date = out.attrs.get("date")
    risk_on = out.attrs.get("risk_on", False)
    L = ["=" * 78,
         f"  ACCUMULATION SCREEN — close {pd.Timestamp(date).date()}   "
         f"regime: {'RISK-ON (Nifty>50MA) — full size' if risk_on else 'RISK-OFF — STAND ASIDE / watch only'}",
         "=" * 78]
    if not risk_on:
        L.append("  Gate is OFF. The edge is net-negative outside a Nifty uptrend. No new longs.")
    if out.empty:
        L.append("  No names clear the conviction footprint. Stand aside.")
    else:
        L.append(f"  {'SYMBOL':<12}{'sector':<20}{'clr':>5}{'delivTr':>8}{'delivTd':>8}"
                 f"{'vol×':>6}{'day%':>7}{'>vwap%':>8}{'RS10%':>7}{'wt':>6}")
        for r in out.itertuples():
            L.append(f"  {r.symbol:<12}{str(r.sector)[:19]:<20}{r.clr:>5.2f}{r.deliv_trail:>8.1f}"
                     f"{r.deliv_per:>8.1f}{r.vol_ratio:>6.1f}{100*r.ret:>+6.1f}%"
                     f"{100*r.close_vs_vwap:>+7.2f}%{100*r.rs_idx_cum:>+6.1f}%{r.weight:>6.0%}")
        L.append(f"\n  → LONG near close, equal-weight, exit next-morning strength (VWAP/RSI)."
                 f"\n    Overnight only — NEVER hold into day 2. Long-only (short side proven "
                 f"dead). Paper-first.")
    return "\n".join(L)
