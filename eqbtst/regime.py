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
    """Was the regime risk-on as of `date`'s close?"""
    reg = nifty_regime(nf)
    row = reg[reg["trade_date"] == pd.Timestamp(date)]
    return bool(row["up"].iloc[0]) if not row.empty else False
