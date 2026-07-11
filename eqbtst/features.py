"""
features.py — the accumulation footprint. Turns EOD OHLC + volume + delivery
into the LOCKED conviction features and a cross-sectional score.

All features are causal: rolling medians use .shift(1) so a day's own value never
leaks into its own baseline. No lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach conviction features per (symbol, trade_date). Input must be sorted
    by symbol, trade_date (data.load_eod already is)."""
    df = df.sort_values(["symbol", "trade_date"]).copy()
    g = df.groupby("symbol", group_keys=False)

    rng = (df["high_price"] - df["low_price"]).replace(0, np.nan)
    df["clr"] = (df["close_price"] - df["low_price"]) / rng          # close location in range
    df["body"] = (df["close_price"] - df["open_price"]) / rng        # signed body fraction
    df["ret"] = df["close_price"] / df["prev_close"] - 1             # day return

    # path signature: how far above session VWAP (avg_price) the day CLOSED. A
    # trended/accumulation day closes well above VWAP; a spike-and-fade closes back
    # near it. The one intraday-shape stat available from the EOD archive.
    vwap = df["avg_price"].where(df["avg_price"] > 0)
    df["close_vs_vwap"] = (df["close_price"] - vwap) / vwap
    df["vwap_in_range"] = (vwap - df["low_price"]) / rng

    vol_med = g["ttl_trd_qnty"].transform(
        lambda s: s.shift(1).rolling(config.LOOKBACK).median())
    df["vol_ratio"] = df["ttl_trd_qnty"] / vol_med                  # participation surge

    deliv_med = g["deliv_per"].transform(
        lambda s: s.shift(1).rolling(config.LOOKBACK).median())
    df["deliv_spike"] = df["deliv_per"] - deliv_med                 # abnormal accumulation

    return df


def add_relative_strength(df: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    """Attach persistent relative strength vs the index. `nifty` needs columns
    trade_date, close_val. rs_idx = daily (stock ret − index ret); rs_idx_cum =
    its RS_LOOKBACK-day cumulative sum (persistent leadership, not a one-day burst).

    THIS (Part X): for BTST, weight PERSISTENT relative strength far above a single
    short-lived burst. Validated: persistent-RS names carry the overnight edge;
    burst-only laggards decay to ~+19bps net.
    """
    nf = nifty.sort_values("trade_date").copy()
    nf["idx_ret"] = nf["close_val"].pct_change()
    df = df.merge(nf[["trade_date", "idx_ret"]], on="trade_date", how="left")
    df = df.sort_values(["symbol", "trade_date"])
    df["rs_idx"] = df["ret"] - df["idx_ret"]
    df["rs_idx_cum"] = df.groupby("symbol", group_keys=False)["rs_idx"].transform(
        lambda s: s.rolling(config.RS_LOOKBACK).sum())
    return df


def signal_mask(df: pd.DataFrame, require_liquidity: bool = True) -> pd.Series:
    """The LOCKED conviction stack — the smart-money accumulation footprint.

    Strong close (buyers held into the bell) + high delivery% (shares actually
    taken, not churned) + delivery ABOVE its own baseline (fresh accumulation) +
    volume surge (real participation) + up day (demand in control) + a PATH-
    PERSISTENT close well above session VWAP (trended, not spike-and-fade) +
    PERSISTENT relative-strength leader. Long-only.

    require_liquidity=False drops only the turnover floor, so the dashboard can
    surface footprint-passers that are AVOID solely because they are too thin.
    """
    m = (
        (df["clr"] >= config.CLR_TH)
        & (df["deliv_per"] >= config.DELIV_TH)
        & (df["deliv_spike"] >= config.DELIV_SPIKE)
        & (df["vol_ratio"] >= config.VOL_TH)
        & (df["ret"] >= config.RET_TH)
        & (df["close_vs_vwap"] >= config.CVWAP_TH)
        & (df["rs_idx_cum"] > config.RS_MIN)
    )
    if require_liquidity:
        m = m & (df["turnover_lacs"] >= config.LIQ_MIN_LACS)
    return m


def conviction_score(df: pd.DataFrame) -> pd.Series:
    """Cross-sectional rank score (0–5) for ranking candidates on a given night.
    Higher = stronger accumulation footprint. Percentile-ranked WITHIN the day
    so it is scale-free across the universe; computed per trade_date by the caller
    when a single night is passed, or globally for backtest ranking."""
    r = df["clr"].rank(pct=True)
    r = r + df["deliv_per"].rank(pct=True)
    r = r + df["deliv_spike"].clip(lower=0).rank(pct=True)
    r = r + df["vol_ratio"].clip(upper=6).rank(pct=True)
    r = r + df["ret"].clip(lower=0).rank(pct=True)
    if "rs_idx_cum" in df.columns:              # prefer the strongest persistent leaders
        r = r + df["rs_idx_cum"].rank(pct=True)
    return r
