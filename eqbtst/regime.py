"""
regime.py — the mandatory Nifty regime gate.

Validation: gating the signal to a Nifty uptrend (close > 50-day MA) revives net
overnight edge to positive in 4 of 5 recent years (2022 +5.7, 2023 +11.1,
2024 +2.7, 2026 +2.7 bps net; only 2025 stayed negative). Ungated, the signal is
net ~0-to-negative in 2024–26. The gate is NOT optional.
"""
from __future__ import annotations

import pandas as pd

from . import config, data


def nifty_regime(nf: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-date regime flag: `up` = Nifty close above its 50-day MA (risk-on)."""
    if nf is None:
        nf = data.load_nifty()
    nf = nf.sort_values("trade_date").copy()
    nf["ma"] = nf["close_val"].rolling(config.REGIME_MA).mean()
    nf["up"] = nf["close_val"] > nf["ma"]
    return nf[["trade_date", "up"]]


def is_risk_on(date: pd.Timestamp, nf: pd.DataFrame | None = None) -> bool:
    """Was the regime risk-on as of `date`'s close? (archive lookup — use for the EOD
    board and for replay, where `date` IS the session being judged.)"""
    reg = nifty_regime(nf)
    row = reg[reg["trade_date"] == pd.Timestamp(date)]
    return bool(row["up"].iloc[0]) if not row.empty else False


def is_risk_on_live(live_close: float | None, nf: pd.DataFrame | None = None) -> bool:
    """Regime gate for a LIVE session — TODAY's Nifty vs TODAY's 50-day MA.

    The archive only runs through YESTERDAY, so an archive lookup would gate today's
    trade on yesterday's regime. That is not the rule the backtest validated: it gates on
    the SIGNAL DAY's own close. The regime flips on ~6.7% of sessions, and ~7.5% of
    validated signal-days land on a flip — precisely at regime turns, which is exactly
    when the gate is load-bearing. So reconstruct today's MA: the last (REGIME_MA-1)
    archive closes plus today's live Nifty. At 15:10-15:30 the index is effectively final,
    so this reproduces the backtested gate.

    Falls back to False (stand aside) if the index price is unavailable — never a free pass.
    """
    if not live_close:
        return False
    if nf is None:
        nf = data.load_nifty()
    prior = nf.sort_values("trade_date")["close_val"].to_numpy(float)[-(config.REGIME_MA - 1):]
    if len(prior) < config.REGIME_MA - 1:
        return False
    ma = (prior.sum() + float(live_close)) / config.REGIME_MA
    return bool(float(live_close) > ma)
