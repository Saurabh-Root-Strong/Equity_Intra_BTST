"""
regime.py — the mandatory Nifty regime gate.

Validation, re-measured 2026-08-08 (the previous numbers in this docstring were
stale and could not be reproduced by any `backtest.run` setting — they understated
2024 by roughly 10x — so they have been replaced with a fresh run):

  gated (close > 50-day MA), net of COST_BPS, 848 signals over 9 years
      per-row mean          +23.6 bps      naive t +6.07
      DATE-CLUSTERED mean   +18.8 bps/day  t +4.50   (572 distinct days)
      block-bootstrap 95% CI  [+9.7, +27.3] bps, p(<=0) < 0.0001
  ungated
      +9.7 bps, t +1.52, positive in 6 of 9 years

  The gate roughly doubles the edge and the naive t-stat overstates it: signals
  cluster (1.48 names/day), so cluster by DATE, not by row.

WHERE THE EDGE ACTUALLY LIVES — per year, date-clustered:
      2018 +4.0 (t0.3)   2019  -1.4 (t-0.1)   2020 +51.8 (t3.7)
      2021 +43.7 (t2.8)  2022 +10.9 (t0.9)    2023  +7.0 (t0.9)
      2024 +29.5 (t2.3)  2025  -0.7 (t-0.1)   2026  +4.9 (t0.4)
  Only 2020, 2021 and 2024 clear t>2. Six of nine years are indistinguishable
  from zero and two are mildly negative. This is a strong-uptrend-year edge, not
  an all-weather one — size it accordingly and do not expect it every year.
  (A per-ROW view reports 9/9 years positive; that is day-count weighting and it
  disagrees with the clustered view on 2019 and 2025. Trust the clustered one.)

THE 50 IS NOT FITTED — this is the reassuring part. Sweeping the MA length:
      MA   10    20    30    40    50    60    75   100   150   200
      net +23.1 +23.0 +23.7 +24.2 +23.6 +25.0 +27.1 +25.9 +26.6 +25.0  bps
      t    6.06  5.73  5.83  6.23  6.07  6.44  6.50  6.36  6.53  6.21
  Every length works; 50 is mid-pack (75 is the best in-sample, so 50 was clearly
  not cherry-picked). Alternative "market is up" definitions do just as well:
  MA50-rising +27.9, 20d-return>0 +23.6, 50d-return>0 +26.9, close>MA200 +25.0.
  A result that is flat across the whole parameter surface is an effect, not a
  curve fit — but it also means the gate is "is the market rising", nothing more
  specific. Do not read significance into the number 50.

The gate is NOT optional.
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


def is_risk_on_live(live_close: float | None, nf: pd.DataFrame | None = None,
                    max_stale_days: int | None = None) -> bool:
    """Regime gate for a LIVE session — TODAY's Nifty vs TODAY's 50-day MA.

    The archive only runs through YESTERDAY, so an archive lookup would gate today's
    trade on yesterday's regime. That is not the rule the backtest validated: it gates on
    the SIGNAL DAY's own close. The regime flips on ~6.7% of sessions, and ~7.5% of
    validated signal-days land on a flip — precisely at regime turns, which is exactly
    when the gate is load-bearing. So reconstruct today's MA: the last (REGIME_MA-1)
    archive closes plus today's live Nifty. At 15:10-15:30 the index is effectively final,
    so this reproduces the backtested gate.

    Falls back to False (stand aside) if the index price is unavailable — never a free pass.

    TWO GUARDS, both added 2026-08-08 after audit:

    1. DOUBLE-COUNT. Rows dated today-or-later are dropped before the window is taken.
       Without this, any call made AFTER the nightly sync has written today's close (or
       on a weekend/holiday, when the broker still returns the last close) used a window
       that ALREADY contained today and then added it again — counting today twice and
       silently dropping the 50th-oldest day. The MA error is exactly
       (close[t] - close[t-49]) / 50, i.e. the 50-day change over 50, so in an uptrend
       the MA came out too HIGH and the gate read risk-OFF when it should read risk-ON.
       Measured on 3,245 sessions it flipped the verdict on 0.55% of them (10 on->off,
       8 off->on) — and those flips sit on MA crossings, exactly the turns this gate
       exists to catch.

    2. STALENESS. `max_stale_days` (opt-in; live callers should pass ~5) rejects a window
       whose newest close is older than that many calendar days. Without it a failed sync
       still produced a confident True/False off a stale window — the archive has 9 gaps
       longer than 4 days, the worst 18 days.
    """
    if not live_close or config.REGIME_MA < 2:
        return False
    if nf is None:
        nf = data.load_nifty()
    s = nf.sort_values("trade_date")

    today = pd.Timestamp.today().normalize()
    s = s[pd.to_datetime(s["trade_date"]) < today]      # guard 1: never count today twice
    if s.empty:
        return False

    if max_stale_days is not None:
        last = pd.Timestamp(s["trade_date"].iloc[-1]).normalize()
        if (today - last).days > max_stale_days:        # guard 2: stale archive
            return False

    prior = s["close_val"].to_numpy(float)[-(config.REGIME_MA - 1):]
    if len(prior) < config.REGIME_MA - 1:
        return False
    ma = (prior.sum() + float(live_close)) / config.REGIME_MA
    return bool(float(live_close) > ma)
