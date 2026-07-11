"""
backtest.py — walk-forward validation of the accumulation-overnight edge, baked
into the package so the LOCKED rule can be re-checked any time (regime-gated,
net-of-cost, per year). No lookahead: features use only past medians; the exit is
the strictly-next session's open.

This reproduces the evidence in the README. Run it before trusting the paper
ledger, and re-run periodically to confirm the edge has not decayed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data, features, regime


def _prepare(start: str | None = None) -> pd.DataFrame:
    df = data.load_eod(start=start)
    df = features.add_features(df)
    df = features.add_relative_strength(df, data.load_nifty(start=start))
    g = df.groupby("symbol", group_keys=False)
    df["next_open"] = g["open_price"].shift(-1)
    df["next_close"] = g["close_price"].shift(-1)
    df["ON"] = (df["next_open"] / df["close_price"] - 1) * 1e4          # overnight gap
    df["D2"] = (df["next_close"] / df["next_open"] - 1) * 1e4           # day-2 intraday
    reg = regime.nifty_regime()
    df = df.merge(reg, on="trade_date", how="left")
    df["yr"] = df["trade_date"].dt.year
    return df.dropna(subset=["clr", "deliv_per", "vol_ratio", "ON", "up"])


def run(start: str | None = None, gated: bool = True, top_n: int | None = None) -> pd.DataFrame:
    """Per-year net-of-cost expectancy for the LOCKED signal. `gated` applies the
    regime gate; `top_n` (if set) keeps only the best N names per day."""
    df = _prepare(start)
    sig = features.signal_mask(df)
    if gated:
        sig = sig & df["up"]
    sel = df[sig].copy()
    if top_n:
        sel["score"] = features.conviction_score(sel)
        sel = sel.sort_values(["trade_date", "score"], ascending=[True, False]) \
                 .groupby("trade_date").head(top_n)
    cost = config.COST_BPS
    rows = []
    for yr, s in sel.groupby("yr"):
        if len(s) < 20:
            continue
        net = s["ON"] - cost
        rows.append({
            "year": int(yr), "n": len(s),
            "gross_ON": round(s["ON"].mean(), 1),
            "net_ON": round(net.mean(), 1),
            "win%": round(100 * (net > 0).mean(), 1),
            "total_bps": round(net.sum(), 0),
            "D2_net": round((s["D2"] - 6).mean(), 1),   # day-2 hold (MIS cost) — should be < 0
        })
    return pd.DataFrame(rows)


def format_run(tbl: pd.DataFrame, title: str) -> str:
    L = [f"\n  {title}",
         f"  {'year':>5}{'n':>6}{'grossON':>9}{'netON':>7}{'win%':>7}{'total':>9}{'D2net':>7}"]
    for _, r in tbl.iterrows():
        L.append(f"  {int(r['year']):>5}{int(r['n']):>6}{r['gross_ON']:>+9.1f}{r['net_ON']:>+7.1f}"
                 f"{r['win%']:>6.1f}%{r['total_bps']:>+9.0f}{r['D2_net']:>+7.1f}")
    if len(tbl):
        tot_n = tbl["n"].sum()
        wm = np.average(tbl["net_ON"], weights=tbl["n"])
        L.append(f"  {'ALL':>5}{tot_n:>6}{'':>9}{wm:>+7.1f}   (n-weighted net overnight bps)")
    return "\n".join(L)
