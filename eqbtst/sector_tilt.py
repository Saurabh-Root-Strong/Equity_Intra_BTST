"""
sector_tilt.py — the DCM 1–2 week forward SECTOR tilt, read onto this board.

WHY THIS EXISTS
    The board tells you what ONE NAME's tape is doing. It says nothing about whether the
    name's SECTOR is the one money is rotating into or out of. Daily_Cash_Market already
    computes that, and it is the ONE sector call that survived deep validation there
    (src/analytics/sector_forward_tilt.py): cross-sectional sector MOMENTUM — relative
    strength vs Nifty over 10 sessions — predicts 1–2 WEEK forward sector returns with
    daily-IC t ≈ 9, Monte-Carlo p < 0.002 vs 600 random portfolios, cost-robust to 40bps.

WHAT IT IS NOT — READ THIS BEFORE YOU TRADE OFF THE COLUMN
    1. HORIZON MISMATCH. The tilt is measured over 10 TRADING DAYS. This engine holds
       OVERNIGHT (close → next open) and NEVER into day 2. Whether the sector edge accrues
       in the overnight GAP (which this book can collect) or during the DAY sessions (which
       it cannot — day-2 holds are net-negative every year, see config.py) is a SEPARATE
       empirical question. `measure_overnight()` in this module answers it; until it has,
       the column is CONTEXT, not a filter, and nothing in the engine reads it.
    2. IT IS A RELATIVE SIGNAL. OVERWEIGHT means "strong versus the other sectors", not
       "going up". A sector can be the best-ranked one in the market and still be falling.
       The validated edge in THIS project is long-only absolute return with a Nifty regime
       gate; cross-sectional neutrality kills it (measured −36bps). So an OVERWEIGHT badge
       is not permission and an UNDERWEIGHT badge is not a short.
    3. UNDERWEIGHT IS NOT A SHORT CALL, in DCM's own words — a sector basket cannot be
       shorted cheaply, and this engine's short side is separately proven dead (net
       −41.9bps, win 20.2%). On a SHORT row the badge is agreement-of-context only.

FAITHFULNESS TO DCM
    Every threshold below is MIRRORED from sector_forward_tilt.py, not re-tuned. The tilt
    LABEL is reproduced exactly: composite rank (0.60·rs_2w + 0.25·rs_1w + 0.15·dv5d),
    the WATCH carve-out, the thin-sector demotion and the momentum-persistence demotion are
    all ported. tests/test_sector_tilt.py pins our labels against DCM's own get_forward_tilt
    on sample dates so the port cannot silently drift.

    ONE DOCUMENTED divergence, which does not touch the tilt label:
      • The debounced regime state is computed over the FULL loaded Nifty history in one
        pass, where DCM re-derives it from a trailing 30-day window per call. Ours has the
        longer memory; both are causal. This affects the advisory regime banner (verdict /
        size_hint / est_rel_bps scaling) and NOT the tilt label — DCM's own recalibration
        left `momentum_inverts` False on every branch, so the regime no longer gates labels.
        Measured over 43 sampled dates spanning 2018-2026: ZERO divergence in state, verdict
        or size_hint, so the longer memory is a difference without an effect so far.
    `robz` is still not computed — but its NaN-ness IS reproduced, because DCM drops names
    whose MAD is 0, and that changes the constituent SET (see _breadth). An earlier version of
    this docstring asserted robz "feeds nothing"; that was wrong and cost one sector's
    accumulation breadth on a real date. A column being unused as a VALUE does not make it
    unused as a FILTER.

LEAKAGE CONTRACT — the whole reason this module takes an explicit `as_of`
    Everything reads data <= as_of ONLY, and the caller must pass the last close that had
    ACTUALLY PRINTED at the decision instant:
      • BTST tab      → as_of = the signal close itself. The tilt is built from that same
                        close, so it is exactly aligned (no leak, no lag).
      • Intraday live → as_of = the last COMPLETED close (yesterday). Today has not closed.
      • Replay        → as_of = the trading day BEFORE the replay date. Using the replay
                        date's own close would feed that session's outcome into an intraday
                        decision — the exact class of lookahead that has retracted two
                        "edges" in this stack already.
    `last_close_before()` exists so callers never hand-roll that.

    The per-day cache is keyed on the as_of DATE PARAMETER, like live.deliv_momentum — never
    on dt.date.today(). Caches keyed on today() are what poison Replay.
"""
from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd

from . import config, data

# ── MIRRORED FROM DCM sector_forward_tilt.py — do not tune here ───────────────────────
_MOM_2W        = 10       # trading days for 2-week momentum / relative strength (DISPLAY only)
_MOM_1W        = 5        # trading days for 1-week momentum (DISPLAY only)
# ── FORMATION WINDOWS — THE RANKING FACTORS (re-derived upstream 2026-07-31) ──────────
# The engine used to rank on 10-day relative strength. On the corrected lagged-weight panel
# that nets ~+0.39% per leg over a 10-15 day hold (t+1.7); ranking on LONG formation and
# holding the SAME 2-3 weeks nets +0.70% (t+2.5) and is positive in all five eras. Upstream
# deliberately selected on the WORST horizon in the band and the WORST era, not the best cell,
# because adjacent horizons swung wildly from sampling noise.
_MOM_3M        = 60       # ~3-month relative strength (ranking factor)
_MOM_6M        = 120      # ~6-month relative strength (ranking factor)
_DV_BASE       = 100      # trailing window for the delivery-flow baseline
_DV_FLOW       = 5        # short delivery-flow window
_MIN_HIST      = 12       # min sector daily rows before a sector is ranked
_MIN_SECTORS   = 8        # need a real cross-section to rank at all
# DCM's own SQL lookbacks, in CALENDAR days: `trade_date > as_of - N` (so the set is the last
# N-1 days inclusive). Both warmups are DERIVED from these inside _engine rather than passed in
# by each caller — two call sites choosing their own warmup is a drift bug waiting to happen,
# and a warmup even slightly longer than DCM's changes which names clear the delivery-history
# gate, which moved accumulation breadth on a real date.
# 400, not the old 260: the ranking factor is now 6-MONTH relative strength (120 trading days),
# and a 260-day window leaves only ~175 rows per sector — enough to compute rs_6m at the last
# date but not to reproduce upstream's, because `shift(120)` then lands on a different row.
# Symptom when this was still 260: most sectors matched EXACTLY while a handful drifted, which
# reads like a formula bug and is actually a window bug.
_PANEL_CAL     = 400      # sector return/turnover panel window (upstream: > as_of - 400)
_DELIV_CAL     = 210      # per-symbol delivery panel window
_MIN_LIQ_NAMES = 5        # below this a sector is "thin" (noisy rs/breadth)

# 0.50 rank(rs_6m) + 0.50 rank(rs_3m). The old 0.60/0.25/0.15 (rs_2w/rs_1w/dv5d) was
# IC-fitted on the biased same-day-turnover panel. On corrected data over a 2-3 week hold,
# delivery flow is a DRAG (pure dv5d fwd20 t-1.2; adding it cut the blend's t from +2.4 to
# +1.9) and short formation is dominated at every hold >= 3 weeks. rs_2w / rs_1w / dv5d are
# still computed and DISPLAYED — they are simply no longer ranked on.
_W_RS6M, _W_RS3M = 0.50, 0.50

_OW_RANK       = 0.75     # composite rank >= this → OVERWEIGHT
_UW_RANK       = 0.25     # composite rank <= this → UNDERWEIGHT
_WATCH_BREADTH = 0.55     # accumulation breadth above this ...
_WATCH_RS_MAX  = 0.35     # ... while momentum rank is still weak → WATCH (contrarian, held out)

# RECALIBRATED upstream 2026-07-31 (290 -> 75). The old 290 came from a "~1.9%/10d tercile
# spread" measured on a SAME-DAY-turnover panel — i.e. on inflated returns. On the lagged panel
# the overweight basket beats an equal-weight sector benchmark by +0.31%/10d, and OW−UW spans
# 0.42–0.72% over a rank spread of ~0.74, giving 57–97 bps per unit of rank; 75 is the midpoint.
# Display estimate only: est_rel_bps is a monotone rescaling of rank and adds nothing to it.
_REL_SLOPE_BPS = 110.0    # bps of 10d RELATIVE return per unit of (rank − 0.5)

# THE WEIGHT MUST NOT KNOW THE RETURN IT IS WEIGHTING. A stock's SAME-DAY turnover explodes on
# the day it jumps, so weighting a daily sector return by same-day turnover correlates the weight
# with the outcome: measured upstream at +0.717%/day of fake drift versus +0.025%/day on lagged
# weights, inflating rs_2w by ~6.6pp and reordering the buy list (3 of 6 top names change).
# PRIOR-SESSION turnover is knowable at entry and is what a real basket would hold. Same defect
# family as the Smart-Money-vs-Tilt panel that had to be retracted for exactly this reason.
_LAG_LEADIN_DAYS = 15     # extra lead-in so the first in-window day still has a prior session

_DISP_MIN      = 1.5      # rs_2w cross-sectional std below this = nothing to rotate on

# delivery layer (mirrored from DCM sector_signal_v2.py)
_ACCUM_R5      = 1.05     # per-stock 5d/100d delivery-% ratio above this AND rising = accumulating
_DELIV_MIN_HIST = 40      # per-stock min days of delivery history for a robust baseline
_DELIV_SLOPE_WIN = 15     # bars the per-stock delivery slope is read over

# regime read (Nifty) — a reliability lever on SIZE, not on the label
_VOL_HI_PCT    = 0.80
_PULLBACK_5D   = -3.0
_MED_TREND_WIN = 40
_EMA_SLOPE_WIN = 10
_REGIME_CONFIRM = 3
_ER_STRONG     = 0.50
_ER_CHOPPY     = 0.30
_TQ_MULT       = {"strong": 1.00, "moderate": 0.90, "choppy": 0.80}
_MULT_UP, _MULT_HIVOL, _MULT_CHOP = 1.00, 1.00, 0.70
_MULT_DOWN, _MULT_REVERSAL, _MULT_BULLTRAP = 0.50, 0.40, 0.70

# momentum-persistence gate
# DCM bails to a documented "market context unavailable" default when it has loaded fewer than
# 60 index rows. That threshold is EVALUATED PER as_of, so a date only 30 sessions into the
# archive reads UNKNOWN — even though a long frame containing that date can compute EMAs for it.
# Reproduced by masking, not by frame length, or the history file would carry a fabricated
# regime for early dates that DCM itself declines to call.
_REG_MIN_ROWS = 60
_REG_UNKNOWN = {"reg_state": "UNKNOWN", "reg_verdict": "SELECTIVE", "reg_size_hint": 0.5,
                "reg_conf_mult": 1.0, "reg_divergence": "n/a", "reg_med_trend": "UNKNOWN",
                "reg_trend_strength": "moderate", "reg_er20": float("nan")}

_PERS_LOOKBACK_CAL = 620  # ~2 trading years, CALENDAR window (matches DCM's SQL)
_PERS_FWD          = 10   # forward horizon persistence is measured over
_PERS_MIN_OBS      = 30   # min realized forward windows before a sector's persistence is used

# DCM's analytics.min_turnover_lacs (config/settings.yaml) — the floor deciding which names
# contribute to a sector aggregate. Mirrored so both sides aggregate the SAME cross-section.
# NOTE this is ₹1cr, deliberately far below this project's own LIQ_MIN_LACS (₹20cr): that one
# is a FILL-REALISM floor for names WE trade, this one defines what the SECTOR is.
TILT_MIN_TURNOVER_LACS = 100.0

_EXCLUDE_SECTORS = ("ETF", "Others")
_SERIES = ("EQ", "SM", "ST")

TILT_HISTORY = config.LEDGER.parent / "sector_tilt_history.parquet"
# The board quotes this measurement. It is PERSISTED rather than hardcoded into the UI text so
# the two cannot drift: a re-run on new data updates what the board says, and if the file is
# missing the board says "not measured yet" instead of quoting a number from memory.
TILT_MEASUREMENT = config.LEDGER.parent / "sector_tilt_measurement.json"

_COLS = ["trade_date", "sector", "score", "rank", "rank_pos", "n_sectors", "tilt",
         "rs_6m", "rs_3m", "rs_2w", "rs_1w", "dv5d", "accum_breadth", "deliv_slope", "n_liq", "thin",
         "divergence", "persistence", "revert", "est_rel_bps", "confidence",
         "reg_state", "reg_verdict", "reg_size_hint", "reg_conf_mult", "reg_divergence",
         "reg_med_trend", "reg_trend_strength", "reg_er20", "dispersion"]


# ── loaders (read-only; the DCM archive is never written by this project) ──────────────
def _sector_panel(start: str, end: str) -> pd.DataFrame:
    """Daily sector return weighted by PRIOR-SESSION turnover, + delivery ₹-value.

    LAG() runs over the UNFILTERED per-symbol series so the weight is that stock's own previous
    session, then the liquidity screen is applied to the CURRENT day — matching DCM exactly. The
    delivery ₹-value leg keeps same-day turnover (it is a level, not a weighted return).
    """
    ph = ",".join("?" * len(_SERIES))
    xh = ",".join("?" * len(_EXCLUDE_SECTORS))
    lead = (pd.Timestamp(start) - pd.Timedelta(days=_LAG_LEADIN_DAYS)).strftime("%Y-%m-%d")
    sql = f"""
        WITH base AS (
            SELECT s.sector, b.trade_date, b.turnover_lacs, b.deliv_per,
                   -- WINSORIZE the per-stock daily move at +/-25%. An uncapped print (a
                   -- split, a bonus, an illiquid spike) is not a return and it distorts the
                   -- whole sector's multi-month momentum — the factor now looks back six
                   -- months, so one bad print poisons 120 bars rather than 10. Upstream
                   -- measured the clip lifting top-3 picks from +0.48% to ~+1.0%/12d.
                   LEAST(GREATEST((b.close_price - b.prev_close)
                                  / NULLIF(b.prev_close, 0) * 100, -25), 25)      AS r,
                   LAG(b.turnover_lacs) OVER (PARTITION BY b.symbol
                                              ORDER BY b.trade_date)              AS w_lag
            FROM daily_data b
            INNER JOIN v_sector_master s ON b.symbol = s.symbol
            WHERE b.series IN ({ph})
              AND s.sector NOT IN ({xh})
              AND b.trade_date >= ? AND b.trade_date <= ?
        )
        SELECT sector, trade_date,
               SUM(turnover_lacs * deliv_per / 100.0) / 100.0                AS daily_dv_cr,
               SUM(w_lag * r) / NULLIF(SUM(CASE WHEN r IS NOT NULL
                                           THEN w_lag END), 0)               AS wtd_ret_pct
        FROM base
        WHERE turnover_lacs >= ? AND w_lag IS NOT NULL
          AND trade_date >= ?
        GROUP BY sector, trade_date
        ORDER BY sector, trade_date
    """
    params = [*_SERIES, *_EXCLUDE_SECTORS, lead, end, TILT_MIN_TURNOVER_LACS, start]
    with data._connect() as c:
        df = c.execute(sql, params).df()
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _liquid_counts(start: str, end: str) -> pd.DataFrame:
    """Liquid constituent count per (sector, date) → the `thin` flag."""
    ph = ",".join("?" * len(_SERIES))
    xh = ",".join("?" * len(_EXCLUDE_SECTORS))
    sql = f"""
        SELECT s.sector, b.trade_date, COUNT(*) AS n_liq
        FROM daily_data b
        INNER JOIN v_sector_master s ON b.symbol = s.symbol
        WHERE b.series IN ({ph})
          AND s.sector NOT IN ({xh})
          AND b.turnover_lacs >= ?
          AND b.trade_date >= ? AND b.trade_date <= ?
        GROUP BY s.sector, b.trade_date
    """
    params = [*_SERIES, *_EXCLUDE_SECTORS, TILT_MIN_TURNOVER_LACS, start, end]
    with data._connect() as c:
        df = c.execute(sql, params).df()
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _deliv_panel(start: str, end: str) -> pd.DataFrame:
    """Per-symbol delivery-% + turnover panel (for the accumulation-breadth overlay)."""
    ph = ",".join("?" * len(_SERIES))
    xh = ",".join("?" * len(_EXCLUDE_SECTORS))
    sql = f"""
        SELECT b.symbol, s.sector, b.trade_date, b.deliv_per, b.turnover_lacs
        FROM daily_data b
        INNER JOIN v_sector_master s ON b.symbol = s.symbol
        WHERE b.series IN ({ph})
          AND s.sector NOT IN ({xh})
          AND b.trade_date >= ? AND b.trade_date <= ?
          AND b.deliv_per IS NOT NULL
        ORDER BY b.symbol, b.trade_date
    """
    params = [*_SERIES, *_EXCLUDE_SECTORS, start, end]
    with data._connect() as c:
        df = c.execute(sql, params).df()
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _nifty(end: str) -> pd.DataFrame:
    """Full Nifty history through `end`. NOT windowed: DCM's realized-vol percentile is an
    EXPANDING rank over everything it loaded (its loader has no start bound), so truncating
    the front would shift the high-vol regime flag."""
    sql = ("select trade_date, close_val, pct_chg from index_data "
           "where index_name=? and trade_date<=? order by trade_date")
    with data._connect() as c:
        df = c.execute(sql, [config.NIFTY_INDEX_NAME, end]).df()
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    px = df["close_val"].astype(float)
    df["nret"] = (df["pct_chg"].astype(float) if df["pct_chg"].notna().any()
                  else px.pct_change() * 100)
    return df


# ── factor maths ──────────────────────────────────────────────────────────────────────
def _compound(pct: pd.Series, n: int) -> pd.Series:
    """Compounded % return over the trailing n rows, aligned to the last row."""
    cr = np.log1p(pct / 100.0).cumsum()
    return np.expm1(cr - cr.shift(n)) * 100.0


def _grouped_compound(df: pd.DataFrame, n: int) -> pd.Series:
    """Per-sector trailing compounded return, computed on each sector's OWN row sequence.

    Deliberately NOT a pivot: a pivot introduces NaN on any (sector, date) the archive is
    missing, and cumsum propagates that NaN forward forever. DCM groups by sector and walks
    each group's own rows, so a missing day shortens the calendar span rather than voiding
    the factor. This reproduces that.
    """
    lr = np.log1p(df["wtd_ret_pct"] / 100.0)
    g = lr.groupby(df["sector"], sort=False)
    cr = g.cumsum()
    prev = cr.groupby(df["sector"], sort=False).shift(n)
    return np.expm1(cr - prev) * 100.0


def _rolling_slope(wide: pd.DataFrame, win: int) -> pd.DataFrame:
    """OLS slope of each column over a trailing `win`-row window (x = 0..win-1).

    Reproduces DCM's `_slope_per_col` including its NaN handling, which is not the textbook
    one and matters: missing days are dropped from the NUMERATOR but the denominator stays
    the full Σ(x−x̄)² of the whole window. So

        slope = [ Σ_notna y·xd − nanmean(y)·Σ_notna xd ] / Σ_all xd²

    computed as a sum of `win` shifted frames — exact and vectorised. Writing the textbook
    complete-windows-only version instead was measurably different (it shifted accumulation
    breadth by ~1-2pp per sector), which is exactly the kind of quiet drift the port exists
    to avoid.
    """
    xd = np.arange(win, dtype=float) - (win - 1) / 2.0
    denom = float((xd ** 2).sum())
    ok = wide.notna()
    y0 = wide.fillna(0.0)
    s_yxd = sum(y0.shift(win - 1 - j) * xd[j] for j in range(win))
    s_xd_ok = sum(ok.shift(win - 1 - j).astype(float) * xd[j] for j in range(win))
    s_y = sum(y0.shift(win - 1 - j) for j in range(win))
    cnt = sum(ok.shift(win - 1 - j).astype(float) for j in range(win))
    mean_y = s_y / cnt.where(cnt > 0)
    return (s_yxd - mean_y * s_xd_ok) / denom


def _mad_zero(dp: pd.DataFrame, med: pd.DataFrame) -> pd.DataFrame:
    """Where is the trailing-window MAD of delivery-% exactly zero?

    THIS IS A FILTER, NOT A STATISTIC. DCM computes `robz = (today-med)/(1.4826*MAD)` and then
    `dropna(subset=["robz","ratio5"])`. A name whose delivery-% is effectively pinned (SOLEX
    sits at 100%) has MAD 0 → divide by zero → robz NaN → DCM DROPS it from its sector's
    accumulation breadth. Skipping this because "robz feeds nothing" inflated Renewables'
    constituent count 17 → 18 and moved that sector's breadth on a live date.

    MAD == 0 ⟺ MORE THAN HALF the window equals the median. Two cheaper tests were tried and
    BOTH were wrong: IQR == 0 is not equivalent (a 60%-identical block can sit entirely above
    the 25th percentile), and even "the median coincides with a quartile" fails as a necessary
    prefilter because pandas interpolates a quantile between neighbouring order statistics. So
    this computes the exact trailing MAD — looped over DATES but vectorised across SYMBOLS,
    which keeps it around a second on a full year and removes the whole class of near-miss.
    """
    v = dp.to_numpy(float)
    res = np.zeros(v.shape, dtype=bool)
    for t in range(_DELIV_MIN_HIST, len(v)):
        w = v[max(0, t - _DV_BASE):t, :]             # STRICTLY before t, like DCM's hist100
        if len(w) < _DELIV_MIN_HIST:
            continue
        with warnings.catch_warnings():              # all-NaN columns are expected
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m = np.nanmedian(w, axis=0)
            mad = np.nanmedian(np.abs(w - m), axis=0)
        res[t] = (mad == 0.0)                        # NaN == 0 is False → not flagged
    return pd.DataFrame(res, index=dp.index, columns=dp.columns)


def _breadth(dpanel: pd.DataFrame) -> pd.DataFrame:
    """Per-(date, sector) accumulation breadth + median delivery slope, bottom-up.

    Ported from DCM sector_signal_v2.get_robust_delivery_signals: a stock is "accumulating"
    when its 5-day delivery-% mean runs >5% above its own trailing-100-day mean AND its
    15-day delivery slope is positive — a MULTI-DAY read, not a one-day spike. Breadth is the
    fraction of a sector's liquid constituents in that state. Only stocks liquid ON THAT DATE
    count, exactly as DCM restricts its panel to names liquid on the as-of day.
    """
    if dpanel.empty:
        return pd.DataFrame(columns=["trade_date", "sector", "accum_breadth", "deliv_slope"])
    dp = dpanel.pivot_table("deliv_per", "trade_date", "symbol").sort_index()
    to = dpanel.pivot_table("turnover_lacs", "trade_date", "symbol").reindex(
        index=dp.index, columns=dp.columns)

    # min_periods matches DCM's gate: it needs >=40 days of delivery history, NOT a full 100
    # (its `hist100.mean()` averages whatever is there). Requiring 100 would silently void
    # every recently-listed name instead of ranking it.
    base = dp.rolling(_DV_BASE, min_periods=_DELIV_MIN_HIST).mean().shift(1)
    enough = dp.notna().rolling(_DV_BASE, min_periods=1).sum().shift(1) >= _DELIV_MIN_HIST
    med = dp.rolling(_DV_BASE, min_periods=_DELIV_MIN_HIST).median().shift(1)
    ratio5 = dp.rolling(_DV_FLOW, min_periods=1).mean() / base.replace(0, np.nan)
    slope = _rolling_slope(dp, _DELIV_SLOPE_WIN) / med.replace(0, np.nan)
    # ── DELIBERATE, DOCUMENTED DEVIATION FROM DCM (the only one that changes a number) ──
    # `accum` tests `slope > 0`. On a delivery series that is EXACTLY FLAT for 15 sessions the
    # true slope is 0, and the computed value is pure rounding noise — DCM's summation order
    # gives +2.0e-18 for DLINKINDIA on 2024-08-27, this module's gives -1.1e-17, so the two
    # disagree about whether a flat line is "rising". Neither is more correct as arithmetic;
    # both are wrong as FINANCE, because a flat delivery trend is not accumulation. Snapping
    # sub-epsilon slopes to exactly 0 makes the answer deterministic and makes it NO. A real
    # 15-day normalised slope in this data is O(1e-3), so 1e-12 cannot mask signal — it only
    # removes a coin flip. Cost: one constituent's accum flag, ~1.6pp of one sector's breadth.
    slope = slope.where(slope.abs() > 1e-12, 0.0)

    usable = (enough & dp.notna() & (to >= TILT_MIN_TURNOVER_LACS)
              & ratio5.notna() & ~_mad_zero(dp, med))
    accum = ((ratio5 > _ACCUM_R5) & (slope > 0)).astype(float).where(usable)
    slope_u = slope.where(usable)

    sec = dpanel.drop_duplicates("symbol").set_index("symbol")["sector"]
    sec = sec.reindex(dp.columns)
    # long-form group-by on (date, sector): the wide frames are dates × symbols, so stack
    # `accum` is non-null wherever the stock is usable (a NaN slope reads as "not rising",
    # exactly as DCM's `slope > 0` treats it), so it is the base index; the raw slope is
    # aligned onto it and may stay NaN — the sector median skips those.
    acc_s = accum.stack().dropna()
    long = pd.DataFrame({"accum": acc_s,
                         "slope": slope_u.stack().reindex(acc_s.index)})
    if long.empty:
        return pd.DataFrame(columns=["trade_date", "sector", "accum_breadth", "deliv_slope"])
    long.index.names = ["trade_date", "symbol"]
    long = long.reset_index()
    long["sector"] = long["symbol"].map(sec)
    g = long.groupby(["trade_date", "sector"])
    out = pd.DataFrame({
        "accum_breadth": g["accum"].mean(),
        "deliv_slope": g["slope"].median(),
        "_n": g.size(),
    }).reset_index()
    # DCM needs >=5 usable constituents before it publishes a sector's delivery aggregate
    out = out[out["_n"] >= _MIN_LIQ_NAMES].drop(columns="_n")
    return out


def _persistence(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-(date, sector) momentum persistence: trailing mean forward RELATIVE edge, causal.

    edge_t = sector's realized fwd-10 return − the cross-sectional MEDIAN sector fwd-10.
    persistence at date T = mean of edge over the trailing 620 CALENDAR days. Because edge is
    NaN for the last _PERS_FWD rows (the forward window has not completed), a date T only ever
    averages edges realized at or before T − 10 sessions — no lookahead by construction.
    >0 ⇒ high ranks kept outperforming (trust the overweight); <0 ⇒ they faded (demote it).
    """
    empty = pd.DataFrame(columns=["trade_date", "sector", "persistence", "pers_n"])
    if panel.empty:
        return empty
    ret = panel.pivot_table("wtd_ret_pct", "trade_date", "sector").sort_index()
    if len(ret) < _PERS_FWD + _PERS_MIN_OBS:
        return empty
    cr = np.log1p(ret / 100.0).cumsum()
    fwd = np.expm1(cr.shift(-_PERS_FWD) - cr) * 100.0
    edge = fwd.sub(fwd.median(axis=1), axis=0)
    # ── CAUSALITY, and it is subtle enough that a bulk backfill got it wrong once ──────
    # DCM is causal by ACCIDENT OF SHAPE: its panel ends at as_of, so `fwd` is NaN for the
    # last _PERS_FWD rows and its mean-over-all-rows silently averages only forward windows
    # that had actually completed by as_of. Computing the same thing over a RANGE removes
    # that accident — the frame now extends past date T, so edge_T..edge_{T-9} ARE populated
    # from returns that had not happened yet at T, and a plain 620D rolling mean eats them.
    # Measured: that leak moved `persistence` on every date and FLIPPED the OVERWEIGHT →
    # NEUTRAL revert demotion on ~7% of them, i.e. it changed the label the board renders.
    # So subtract the trailing _PERS_FWD ROWS explicitly: window = (T-620d, T-10 rows], which
    # is exactly the set DCM averages. Now bulk == pointwise, enforced by a test.
    win = f"{_PERS_LOOKBACK_CAL}D"
    tot_s, tot_n = edge.rolling(win).sum(), edge.rolling(win).count()
    # the tail must fill to 0, not NaN: on the pointwise path those last rows are ALREADY NaN
    # (nothing to subtract), and pandas returns NaN for an all-NaN sum window — which would
    # propagate through the subtraction and void persistence completely.
    tail_s = edge.rolling(_PERS_FWD, min_periods=1).sum().fillna(0.0)
    tail_n = edge.rolling(_PERS_FWD, min_periods=1).count().fillna(0.0)
    n = (tot_n - tail_n)
    pers = (tot_s - tail_s) / n.where(n > 0)
    pers = pers.where(n >= _PERS_MIN_OBS)
    out = pd.DataFrame({"persistence": pers.stack().dropna()})
    out.index.names = ["trade_date", "sector"]
    out = out.reset_index()
    nn = n.stack().dropna().rename("pers_n").reset_index()
    nn.columns = ["trade_date", "sector", "pers_n"]
    out = out.merge(nn, on=["trade_date", "sector"], how="left")
    return out[out["pers_n"] >= _PERS_MIN_OBS]


# ── regime (advisory: scales SIZE and est_rel_bps, never the tilt label) ───────────────
def _expanding_pct(s: pd.Series) -> pd.Series:
    """Expanding percentile of each value within the history up to and including it.

    Reproduces DCM's `(vol20 <= vol20.iloc[i]).mean()` EXACTLY, including the part that looks
    like a bug and must be copied anyway: the denominator is i+1 — every row so far, NaN
    warm-up rows included (NaN <= v is False, so they sit in the denominator and never in the
    numerator). Excluding them, as a first cut of this function did, biases the percentile
    UPWARD by ~1% on a 2000-row series, which is enough to cross the 0.80 HIGH_VOL threshold
    on borderline days and hand back a different regime — a silent divergence from the source.
    """
    v = s.to_numpy(float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if not np.isfinite(v[i]):
            continue
        w = v[:i + 1]
        out[i] = float(np.sum(np.less_equal(w, v[i], where=np.isfinite(w),
                                           out=np.zeros(len(w), dtype=bool))) / (i + 1))
    return pd.Series(out, index=s.index)


def _confirmed_states(close: pd.Series, e20: pd.Series, e50: pd.Series,
                      volpct: pd.Series) -> pd.Series:
    """Debounced persistent regime per date (UP/DOWN/HIGH_VOL/CHOP), one causal pass.

    Raw EMA-stack labels whipsaw (DCM's 8yr audit: median run 3 days, 42% flip straight
    back), so a SWITCH is only accepted after the new label has held _REGIME_CONFIRM
    consecutive days; shorter flips keep the prior confirmed regime.
    """
    lab = []
    for px, a, b, vp in zip(close, e20, e50, volpct):
        if px < a < b:
            lab.append("DOWN")
        elif np.isfinite(vp) and vp >= _VOL_HI_PCT:
            lab.append("HIGH_VOL")
        elif px > a > b:
            lab.append("UP")
        else:
            lab.append("CHOP")
    conf, pending, cnt = [], lab[0] if lab else "CHOP", 0
    last = pending
    for v in lab:
        if v == pending:
            cnt += 1
        else:
            pending, cnt = v, 1
        if cnt >= _REGIME_CONFIRM:
            last = pending
        conf.append(last)
    return pd.Series(conf, index=close.index)


def _regime_history(nf: pd.DataFrame) -> pd.DataFrame:
    """Per-date Nifty posture: state, confidence multiplier, verdict, size hint, divergence."""
    cols = ["trade_date", "reg_state", "reg_verdict", "reg_size_hint", "reg_conf_mult",
            "reg_divergence", "reg_med_trend", "reg_trend_strength", "reg_er20"]
    if nf.empty:
        return pd.DataFrame(columns=cols)
    if len(nf) < _REG_MIN_ROWS:
        return pd.DataFrame({"trade_date": nf["trade_date"], **_REG_UNKNOWN})
    nf = nf.sort_values("trade_date").reset_index(drop=True)
    close = nf["close_val"].astype(float)
    ret = nf["nret"].astype(float) / 100.0
    e20 = close.ewm(span=20, adjust=False).mean()
    e50 = close.ewm(span=50, adjust=False).mean()
    vol20 = ret.rolling(20).std()
    volpct = _expanding_pct(vol20)
    ret_5d = _compound(nf["nret"], 5)
    ret_20d = _compound(nf["nret"], 20)
    ret_med = _compound(nf["nret"], _MED_TREND_WIN)
    ema_slope = e20 - e20.shift(_EMA_SLOPE_WIN)

    # Kaufman efficiency ratio (20d): |net move| / path length. High = clean trend.
    path = close.diff().abs().rolling(20).sum()
    er20 = (close - close.shift(20)).abs() / path.replace(0, np.nan)
    strength = np.where(er20 >= _ER_STRONG, "strong",
                        np.where(er20 < _ER_CHOPPY, "choppy", "moderate"))
    strength = pd.Series(strength, index=close.index).where(er20.notna(), "moderate")

    base = _confirmed_states(close, e20, e50, volpct)
    reversal = (ret_5d <= _PULLBACK_5D) & (ret_20d > 0)

    med_up = (ret_med > 0) & (ema_slope > 0)
    med_dn = (ret_med < 0) & (ema_slope < 0)
    short_up = ret_5d > 0
    med_trend = np.where(med_up, "UP", np.where(med_dn, "DOWN", "FLAT"))
    diverg = np.where(short_up & med_dn, "BULLTRAP",
             np.where((~short_up) & med_up, "DIP_IN_UP",
             np.where(short_up & med_up, "ALIGNED_UP",
             np.where((~short_up) & med_dn, "ALIGNED_DN", "MIXED"))))

    state = np.where(reversal, "REVERSAL", base.to_numpy())
    mult = np.select(
        [state == "REVERSAL", state == "TRENDING_DOWN", state == "DOWN",
         state == "HIGH_VOL", state == "UP"],
        [_MULT_REVERSAL, _MULT_DOWN, _MULT_DOWN, _MULT_HIVOL, _MULT_UP],
        default=_MULT_CHOP).astype(float)
    verdict = np.where(np.isin(state, ["UP", "HIGH_VOL"]), "ACT", "SELECTIVE")
    size = np.select(
        [state == "REVERSAL", state == "DOWN", state == "HIGH_VOL", state == "UP"],
        [0.3, 0.4, 0.75, 1.0], default=0.5).astype(float)
    # DCM's labels for the two EMA-stack states
    state = np.where(state == "UP", "TRENDING_UP", np.where(state == "DOWN", "TRENDING_DOWN",
                     np.where(state == "CHOP", "CHOPPY", state)))

    # trend-quality nudge applies in ACT regimes only
    tq = strength.map(_TQ_MULT).to_numpy(float)
    size = np.where(verdict == "ACT", size * tq, size)

    # divergence overlays DOWNGRADE only — never upgrade
    bull = diverg == "BULLTRAP"
    mult = np.where(bull, mult * _MULT_BULLTRAP, mult)
    size = np.where(bull, np.minimum(size, 0.5), size)
    verdict = np.where(bull & (verdict == "ACT"), "SELECTIVE", verdict)
    aligned_dn = (diverg == "ALIGNED_DN") & (verdict != "ACT")
    size = np.where(aligned_dn, np.minimum(size, 0.3), size)
    mult = np.where(aligned_dn, np.minimum(mult, _MULT_DOWN), mult)

    out = pd.DataFrame({
        "trade_date": nf["trade_date"], "reg_state": state, "reg_verdict": verdict,
        "reg_size_hint": np.round(size, 2), "reg_conf_mult": mult,
        "reg_divergence": diverg, "reg_med_trend": med_trend,
        "reg_trend_strength": strength.to_numpy(), "reg_er20": er20.to_numpy(),
    })
    # rows with fewer than _REG_MIN_ROWS of index history BEFORE them read UNKNOWN, per above
    warm = np.arange(len(out)) < _REG_MIN_ROWS - 1
    for k, v in _REG_UNKNOWN.items():
        out.loc[warm, k] = v
    return out


# ── the engine (one code path for the live column AND the backtest) ────────────────────
def _engine(start: str, end: str) -> pd.DataFrame:
    """Per-(date, sector) tilt for every date in [start, end]. Causal at every date.

    Warmups are derived from DCM's own window constants, so a single-date call and a bulk
    range call see the SAME trailing history for any shared date (enforced by
    test_bulk_history_equals_pointwise).
    """
    s0 = pd.Timestamp(start)
    panel_warmup = (s0 - pd.Timedelta(days=_PANEL_CAL - 1)).strftime("%Y-%m-%d")
    deliv_warmup = (s0 - pd.Timedelta(days=_DELIV_CAL - 1)).strftime("%Y-%m-%d")
    panel = _sector_panel(panel_warmup, end)
    if panel.empty:
        return pd.DataFrame(columns=_COLS)
    panel = panel.sort_values(["sector", "trade_date"]).reset_index(drop=True)

    # A sector needs _MIN_HIST rows IN THE TRAILING WINDOW before it is ranked — DCM's
    # `len(g) < _MIN_HIST` on its 260-day panel. Counted over calendar days, NOT as a position
    # within whatever frame happens to be loaded: a cumcount would let the same date qualify in
    # a bulk run and fail in a pointwise one, right when a new sector is introduced.
    panel["_one"] = 1.0
    seq = (panel.set_index("trade_date").groupby("sector", sort=False)["_one"]
           .rolling(f"{_PANEL_CAL - 1}D").sum().reset_index(level=0, drop=True).to_numpy())
    panel = panel.drop(columns="_one")
    panel["mom_2w"] = _grouped_compound(panel, _MOM_2W)
    panel["mom_1w"] = _grouped_compound(panel, _MOM_1W)
    # long formation — the RANKING factors. NaN until a sector has that much history; the
    # rank step below fills those with a neutral 0.5 rather than dropping the sector.
    panel["mom_3m"] = _grouped_compound(panel, _MOM_3M)
    panel["mom_6m"] = _grouped_compound(panel, _MOM_6M)
    gdv = panel.groupby("sector", sort=False)["daily_dv_cr"]
    # min_periods=1, matching DCM's `dv.iloc[:-1].tail(100).mean()` — it averages WHATEVER
    # history exists and gates only on _MIN_HIST rows of the sector. Demanding a full 100-row
    # baseline (the first cut here) voided dv5d for the first ~100 sessions of the archive, so
    # the composite went NaN and early-2018 dates carried NO TILT AT ALL while DCM ranked them
    # — which quietly dropped those signals out of the measurement's 2018 bucket.
    dv_base = gdv.transform(lambda s: s.rolling(_DV_BASE, min_periods=1).mean().shift(1))
    dv_flow = gdv.transform(lambda s: s.rolling(_DV_FLOW, min_periods=1).mean())
    panel["dv5d"] = dv_flow / dv_base.where(dv_base > 0)
    panel = panel[seq >= _MIN_HIST]

    nf = _nifty(end)
    reg = _regime_history(nf)
    # AS-OF (forward-fill) lookup, not an exact-date map. DCM reads `_compound(...).iloc[-1]`
    # off a frame filtered `trade_date <= as_of` — i.e. the LAST AVAILABLE index row at or
    # before as_of, which is a ffill, not a zero. An exact map with fillna(0) silently turned
    # relative strength into ABSOLUTE momentum on any session index_data is missing (found on
    # 2025-05-27). Ranks happen to be invariant to that constant shift, so it never moved a
    # tilt label — it just printed a wrong rs_2w, which is worse: a number that looks fine.
    if nf.empty:
        panel["rs_2w"] = panel["mom_2w"]
        panel["rs_1w"] = panel["mom_1w"]
        panel["rs_3m"] = panel["mom_3m"]
        panel["rs_6m"] = panel["mom_6m"]
    else:
        nd = pd.DatetimeIndex(nf["trade_date"])
        n2 = pd.Series(_compound(nf["nret"], _MOM_2W).to_numpy(), index=nd)
        n1 = pd.Series(_compound(nf["nret"], _MOM_1W).to_numpy(), index=nd)
        pd_ = pd.DatetimeIndex(panel["trade_date"])
        panel["rs_2w"] = panel["mom_2w"] - n2.reindex(pd_, method="ffill").to_numpy()
        panel["rs_1w"] = panel["mom_1w"] - n1.reindex(pd_, method="ffill").to_numpy()
        n3 = pd.Series(_compound(nf["nret"], _MOM_3M).to_numpy(), index=nd)
        n6 = pd.Series(_compound(nf["nret"], _MOM_6M).to_numpy(), index=nd)
        panel["rs_3m"] = panel["mom_3m"] - n3.reindex(pd_, method="ffill").fillna(0.0).to_numpy()
        panel["rs_6m"] = panel["mom_6m"] - n6.reindex(pd_, method="ffill").fillna(0.0).to_numpy()

    brd = _breadth(_deliv_panel(deliv_warmup, end))
    panel = panel.merge(brd, on=["trade_date", "sector"], how="left")
    # EXACTLY DCM's persistence window: its SQL filters `trade_date > as_of - 620` with a
    # 15-day lead-in for the LAG, so the panel spans 619 days back inclusive. A wider window
    # (this used 660) changes which rows the LAG chain starts from for intermittently-traded
    # symbols, which moved persistence in the 6th decimal and broke parity.
    pers = _persistence(_sector_panel(
        (pd.Timestamp(start) - pd.Timedelta(days=_PERS_LOOKBACK_CAL - 1)).strftime("%Y-%m-%d"),
        end))
    panel = panel.merge(pers.drop(columns=["pers_n"], errors="ignore"),
                        on=["trade_date", "sector"], how="left")
    nliq = _liquid_counts(start, end)
    panel = panel.merge(nliq, on=["trade_date", "sector"], how="left")

    # keep only the requested window now that every trailing factor is formed
    panel = panel[(panel["trade_date"] >= pd.Timestamp(start))
                  & (panel["trade_date"] <= pd.Timestamp(end))].copy()
    if panel.empty:
        return pd.DataFrame(columns=_COLS)

    panel["n_liq"] = panel["n_liq"].fillna(0).astype(int)
    panel["thin"] = panel["n_liq"] < _MIN_LIQ_NAMES

    # ── cross-sectional ranks, one independent cross-section per date ──────────────
    byd = panel.groupby("trade_date")
    r_rs2 = byd["rs_2w"].rank(pct=True)
    r_dv5 = byd["dv5d"].rank(pct=True)
    # a sector without 6-month history is ranked MID-PACK, not dropped — matching upstream,
    # which fills a missing formation rank with 0.5 rather than letting the sector vanish
    r_6m = byd["rs_6m"].rank(pct=True).fillna(0.5)
    r_3m = byd["rs_3m"].rank(pct=True).fillna(0.5)
    panel["score"] = _W_RS6M * r_6m + _W_RS3M * r_3m
    panel["rank"] = panel.groupby("trade_date")["score"].rank(pct=True)
    panel["rank_pos"] = panel.groupby("trade_date")["score"].rank(ascending=False,
                                                                 method="min")
    panel["n_sectors"] = panel.groupby("trade_date")["sector"].transform("size")
    panel["divergence"] = r_dv5 - r_rs2
    panel = panel[panel["n_sectors"] >= _MIN_SECTORS]
    if panel.empty:
        return pd.DataFrame(columns=_COLS)

    # ── the tilt label ────────────────────────────────────────────────────────────
    br = panel["accum_breadth"].fillna(0.0)
    rr = panel["rank"]
    tilt = np.where((br >= _WATCH_BREADTH) & (rr <= _WATCH_RS_MAX), "WATCH",
            np.where(rr >= _OW_RANK, "OVERWEIGHT",
            np.where(rr <= _UW_RANK, "UNDERWEIGHT", "NEUTRAL")))
    tilt = pd.Series(tilt, index=panel.index).where(rr.notna(), None)
    # thin sectors cannot be a confident overweight; historically-reverting ones fade
    panel["revert"] = panel["persistence"] < 0            # NaN → False (unknown ⇒ keep)
    tilt = tilt.mask((tilt == "OVERWEIGHT") & panel["thin"], "NEUTRAL")
    tilt = tilt.mask((tilt == "OVERWEIGHT") & panel["revert"], "NEUTRAL")
    panel["tilt"] = tilt

    # ── dispersion: is there anything to rotate on at all today? ───────────────────
    disp = panel.groupby("trade_date")["rs_2w"].transform("std")
    panel["dispersion"] = disp
    # AS-OF join, for the same reason rs_2w uses one: DCM reads its regime off the LAST index
    # row at or before as_of, so on a session index_data is missing it reports the prior day's
    # regime — not a blank. An exact-date merge produced NaN state/verdict/size there (found on
    # 2025-05-27), which then rendered as the literal string "nan" on the board.
    if reg.empty:
        for c in reg.columns:
            if c != "trade_date":
                panel[c] = np.nan
    else:
        panel = pd.merge_asof(panel.sort_values("trade_date"),
                              reg.sort_values("trade_date"),
                              on="trade_date", direction="backward")
    low_disp = disp.to_numpy() < _DISP_MIN
    panel["reg_size_hint"] = np.where(low_disp,
                                      (panel["reg_size_hint"] * 0.5).round(2),
                                      panel["reg_size_hint"])
    panel["reg_verdict"] = np.where(low_disp & (panel["reg_verdict"] == "ACT"),
                                    "SELECTIVE", panel["reg_verdict"])

    cm = panel["reg_conf_mult"].fillna(1.0)
    panel["est_rel_bps"] = ((panel["rank"] - 0.5) * _REL_SLOPE_BPS * cm).round(0)
    panel["confidence"] = (cm * np.where(panel["thin"], 0.5, 1.0)
                              * np.where(panel["revert"], 0.6, 1.0)).round(2)
    return panel.reindex(columns=_COLS).sort_values(
        ["trade_date", "score"], ascending=[True, False]).reset_index(drop=True)


# ── public API ────────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=4)
def _tilt_cached(as_of_key: str) -> pd.DataFrame:
    """One archive read per as-of DATE (not per today()) — Replay-safe by construction."""
    return _engine(start=as_of_key, end=as_of_key)


def clear_cache() -> None:
    """Drop the per-date cache. The dashboard's ↻ calls st.cache_data.clear(), which does NOT
    reach an lru_cache — so without this a re-sync of the SAME trade date (a partial nightly
    sync followed by a full one) would serve the first, incomplete read for the life of the
    process. Mirrors live.clear_universe_cache()."""
    _tilt_cached.cache_clear()


def sector_tilt(as_of) -> tuple[pd.DataFrame, dict]:
    """Per-sector tilt as of `as_of`'s close, plus the regime meta for that date.

    `as_of` MUST be the last close that had actually printed at the decision instant —
    see the leakage contract in the module docstring. Returns (per-sector frame indexed by
    sector, regime dict). Both are empty/defaulted when the archive cannot support a read,
    so a caller never has to special-case a missing day.
    """
    key = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    err = None
    try:
        df = _tilt_cached(key)
    except Exception as e:                     # archive locked / schema drift / no such date
        # DEGRADE, BUT SAY WHY. Swallowing this to a bare "unavailable" makes a locked DuckDB,
        # a renamed column and a non-trading day all look identical — so a genuine breakage
        # reads as an ordinary empty day and nobody investigates. The reason travels in meta.
        df, err = pd.DataFrame(columns=_COLS), f"{type(e).__name__}: {e}"
    if df.empty:
        return df, {"as_of": key, "available": False,
                    "error": err or "no ranked sector cross-section for this date"}
    meta = {
        "as_of": key, "available": True,
        "state": df["reg_state"].iloc[0], "verdict": df["reg_verdict"].iloc[0],
        "size_hint": float(df["reg_size_hint"].iloc[0]),
        "conf_mult": float(df["reg_conf_mult"].iloc[0]),
        "divergence": df["reg_divergence"].iloc[0],
        "med_trend": df["reg_med_trend"].iloc[0],
        "trend_strength": df["reg_trend_strength"].iloc[0],
        "dispersion": float(df["dispersion"].iloc[0]),
        "n_sectors": int(df["n_sectors"].iloc[0]),
        "n_ow": int((df["tilt"] == "OVERWEIGHT").sum()),
        "n_uw": int((df["tilt"] == "UNDERWEIGHT").sum()),
    }
    return df.set_index("sector"), meta


def last_close_before(d) -> pd.Timestamp | None:
    """The last archived trading close STRICTLY BEFORE `d` — the correct as_of for any
    INTRADAY decision taken during session `d` (that session has not closed yet, so its own
    close is not knowable). Hand-rolling this is how replay lookahead gets introduced."""
    sql = ("select max(trade_date) from daily_data where series='EQ' and trade_date < ?")
    try:
        with data._connect() as c:
            v = c.execute(sql, [pd.Timestamp(d).strftime("%Y-%m-%d")]).fetchone()[0]
    except Exception:
        return None
    return pd.Timestamp(v) if v is not None else None


# ── display ───────────────────────────────────────────────────────────────────────────
_BADGE = {"OVERWEIGHT": "🟢 OW", "UNDERWEIGHT": "🔴 UW",
          "NEUTRAL": "⚪ NEUTRAL", "WATCH": "👁 WATCH"}


def badge(row: pd.Series | None, side: str | None = None) -> str:
    """One cell: the sector's tilt, its rank position, and whether it agrees with the row.

    Honest about absence — a sector with too few liquid names, or one the cross-section
    could not rank, reads as a dash with the REASON, never as a neutral verdict. A missing
    read and a genuine NEUTRAL are different answers and the cell keeps them different.
    """
    if row is None or not len(row):
        return "— sector not ranked"
    tilt = row.get("tilt")
    if not isinstance(tilt, str):
        return "— sector not ranked"
    txt = _BADGE.get(tilt, tilt)
    pos, n = row.get("rank_pos"), row.get("n_sectors")
    if pd.notna(pos) and pd.notna(n):
        txt += f" #{int(pos)}/{int(n)}"
    if side in ("LONG", "SHORT") and tilt in ("OVERWEIGHT", "UNDERWEIGHT"):
        agree = (tilt == "OVERWEIGHT") == (side == "LONG")
        txt += f" · {'with' if agree else 'against'} your {side}"
    if bool(row.get("thin")):
        # A THIN SECTOR STILL HAS A VERDICT — flag it, do not hide it. Suppressing the label
        # outright (an earlier version of this function) discarded a read the engine really
        # made: DCM withholds only the OVERWEIGHT call on a thin sector (it demotes that to
        # NEUTRAL upstream), while NEUTRAL / UNDERWEIGHT / WATCH stand. Rendering all of them
        # as a dash overstated the uncertainty and lost information.
        txt += f"  ⚠thin({int(row.get('n_liq') or 0)})"
    return txt


def annotate(df: pd.DataFrame, as_of, side: str | None = None,
             side_col: str = "side") -> pd.DataFrame:
    """Add the `sector tilt` column to a finished board table.

    Applied to the RENDERED table rather than inside each row-builder on purpose: there are
    five row-builders across the live / timeframe / replay / EOD lanes and any new one would
    silently miss the column. One annotation site per table cannot drift.

    `side` forces the agreement read (the LONG/SHORT tabs know their own side); otherwise it
    is taken from `side_col` per row when present.
    """
    if df is None or df.empty or "sector" not in df.columns:
        return df
    if as_of is None:                       # caller could not resolve a causal as-of close
        out = df.copy()
        out["sector tilt"] = "— as-of close unresolved"
        return out
    tilt, _meta = sector_tilt(as_of)
    out = df.copy()
    if tilt.empty:
        why = _meta.get("error") or ""
        out["sector tilt"] = ("— tilt unavailable" if not why
                              else f"— tilt unavailable ({why[:60]})")
        return out
    sides = ([side] * len(out) if side is not None
             else (out[side_col].tolist() if side_col in out.columns else [None] * len(out)))
    out["sector tilt"] = [
        badge(tilt.loc[s] if s in tilt.index else None, sd)
        for s, sd in zip(out["sector"], sides)
    ]
    return out


# ── Phase 2: the history, and the measurement that decides whether this is edge ────────
def build_history(start: str = "2017-06-01", end: str | None = None,
                  path=TILT_HISTORY) -> pd.DataFrame:
    """Backfill the per-(date, sector) tilt for the whole archive → parquet.

    Chunked by calendar year with a warmup overlap, because the delivery panel is per-SYMBOL
    (~2000 names × 8 years) and pulling it whole would be a >1GB frame on a 2GB VM. Each chunk
    is computed with enough trailing history for every rolling factor to be fully formed, then
    only the in-range dates are kept — so a chunk boundary is NOT a discontinuity.

    This is the SAME `_engine` the live column calls. That is the point: the column you read
    on the board and the column that gets measured below cannot disagree.
    """
    end = end or data.last_trading_date().strftime("%Y-%m-%d")
    yrs = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    out = []
    for y in yrs:
        c0 = max(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(start))
        c1 = min(pd.Timestamp(f"{y}-12-31"), pd.Timestamp(end))
        if c0 > c1:
            continue
        chunk = _engine(start=c0.strftime("%Y-%m-%d"), end=c1.strftime("%Y-%m-%d"))
        print(f"  {y}: {len(chunk):>6} sector-days  "
              f"({chunk['trade_date'].nunique() if len(chunk) else 0} dates)")
        out.append(chunk)
    hist = (pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=_COLS))
    if not hist.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        hist.to_parquet(path, index=False)
    return hist


def load_history(path=TILT_HISTORY) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No tilt history at {path}. Build it first:  python -m eqbtst.cli tilt-history")
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _welch(a: pd.Series, b: pd.Series) -> float:
    """Two-sample t-statistic with unequal variances (a − b)."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


# THE RULE IS WRITTEN DOWN BEFORE THE NUMBER IS SEEN. Two "edges" in this stack were
# retracted after the fact because the test was chosen once the data was on screen. So:
# the tilt becomes a GATE only if BOTH conditions hold, and stays display-only otherwise.
_GATE_RULE = ("PRE-REGISTERED: wire the tilt as a gate ONLY IF (a) OVERWEIGHT − UNDERWEIGHT "
              "net overnight bps > 0 with |t| >= 2.0, AND (b) OW beats UW in >= 5 of the 8 "
              "years. Otherwise it stays DISPLAY-ONLY — permanently.")


def measure_overnight(start: str = "2018-01-01", gated: bool = True) -> dict:
    """Does the sector tilt condition the OVERNIGHT payoff of the footprint signal?

    The one question that decides whether this column is edge or decoration. For every
    historical footprint trigger, join the trigger's SECTOR tilt as of that same close (the
    aligned, causal read the BTST tab shows) and split the overnight gap by tilt bucket.

    Also splits ON (overnight gap — the part this book collects) against D2 (the next day's
    intraday — the part it structurally cannot, day-2 holds being net-negative every year).
    If the tilt's spread lives in D2 and not in ON, the signal is real and this engine still
    cannot monetise it. That distinction is the entire point of the split.
    """
    from . import backtest, features            # local: backtest imports config/data already

    df = backtest._prepare(start)
    # The MARKET-WIDE overnight gap for each night, over the liquid universe. This is the
    # `beta` term in this project's own edge decomposition (~+16bps universe drift + ~+10bps
    # selection), and it is the confound that matters here: OVERWEIGHT and UNDERWEIGHT signals
    # fire on almost DISJOINT sets of nights (191 vs 100, overlapping on 18), so a raw bucket
    # comparison is mostly a BETWEEN-NIGHT comparison and can be won purely by trading better
    # nights. Subtracting it isolates the genuinely CROSS-SECTIONAL part of any difference.
    night = (df[df["turnover_lacs"] >= config.LIQ_MIN_LACS]
             .groupby("trade_date")["ON"].mean().rename("univ_ON"))
    sig = features.signal_mask(df)
    if gated:
        sig = sig & df["up"]
    sel = df[sig].copy()
    sectors = data.load_sectors()
    # KNOWN LIMITATION: v_sector_master is a CURRENT map with no history, so a 2018 trigger is
    # bucketed by its 2026 sector. Reclassifications therefore blur early-year membership. DCM
    # has no historical sector map to join, so this is a floor on precision, not a fixable bug.
    sel["sector"] = sel["symbol"].map(lambda s: sectors.get(s, f"_{s}"))

    hist = load_history()
    sel = sel.merge(hist[["trade_date", "sector", "tilt", "rank", "est_rel_bps", "thin"]],
                    on=["trade_date", "sector"], how="left")
    sel = sel.merge(night, on="trade_date", how="left")
    cost = config.COST_BPS
    sel["net"] = sel["ON"] - cost
    sel["excess"] = sel["ON"] - sel["univ_ON"]        # cross-sectional, night factor removed
    sel["yr"] = sel["trade_date"].dt.year

    rows = []
    for t in ("OVERWEIGHT", "NEUTRAL", "UNDERWEIGHT", "WATCH"):
        s = sel[sel["tilt"] == t]
        if s.empty:
            continue
        rows.append({
            "tilt": t, "n": len(s),
            "gross_ON": round(s["ON"].mean(), 1),
            "net_ON": round(s["net"].mean(), 1),
            "win%": round(100 * (s["net"] > 0).mean(), 1),
            "t_vs_0": round(float(s["net"].mean() / (s["net"].std(ddof=1) / np.sqrt(len(s)))), 2)
                      if len(s) > 5 and s["net"].std(ddof=1) > 0 else float("nan"),
            "D2_net": round((s["D2"] - 6).mean(), 1),
        })
    buckets = pd.DataFrame(rows)
    unmatched = int(sel["tilt"].isna().sum())

    ow, uw = sel[sel["tilt"] == "OVERWEIGHT"], sel[sel["tilt"] == "UNDERWEIGHT"]
    diff_on = (float(ow["ON"].mean() - uw["ON"].mean())
               if len(ow) and len(uw) else float("nan"))
    diff_d2 = (float(ow["D2"].mean() - uw["D2"].mean())
               if len(ow) and len(uw) else float("nan"))
    t_on, t_d2 = _welch(ow["ON"], uw["ON"]), _welch(ow["D2"], uw["D2"])

    per_yr = []
    for y, g in sel.groupby("yr"):
        a, b = g[g["tilt"] == "OVERWEIGHT"]["net"], g[g["tilt"] == "UNDERWEIGHT"]["net"]
        if len(a) < 5 or len(b) < 5:
            per_yr.append({"year": int(y), "n_ow": len(a), "n_uw": len(b),
                           "ow_net": round(a.mean(), 1) if len(a) else None,
                           "uw_net": round(b.mean(), 1) if len(b) else None,
                           "ow_wins": None})
            continue
        per_yr.append({"year": int(y), "n_ow": len(a), "n_uw": len(b),
                       "ow_net": round(a.mean(), 1), "uw_net": round(b.mean(), 1),
                       "ow_wins": bool(a.mean() > b.mean())})
    years = pd.DataFrame(per_yr)
    yrs_ok = int(years["ow_wins"].fillna(False).sum()) if len(years) else 0
    yrs_tested = int(years["ow_wins"].notna().sum()) if len(years) else 0

    # ── is the difference cross-sectional, or just a different set of NIGHTS? ──────────
    diff_night = (float(ow["univ_ON"].mean() - uw["univ_ON"].mean())
                  if len(ow) and len(uw) else float("nan"))
    diff_xs = (float(ow["excess"].mean() - uw["excess"].mean())
               if len(ow) and len(uw) else float("nan"))
    t_night, t_xs = _welch(ow["univ_ON"], uw["univ_ON"]), _welch(ow["excess"], uw["excess"])

    # cluster-robust SE by NIGHT: overnight returns share a market-wide gap, so trades on the
    # same night are not independent observations and a plain t overstates precision.
    sub = sel[sel["tilt"].isin(["OVERWEIGHT", "UNDERWEIGHT"])]
    t_clust, n_clust = float("nan"), 0
    if len(sub) > 20 and sub["trade_date"].nunique() > 5:
        X = np.column_stack([np.ones(len(sub)), (sub["tilt"] == "OVERWEIGHT").to_numpy(float)])
        y = sub["ON"].to_numpy(float)
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        r = y - X @ b
        xtx = np.linalg.inv(X.T @ X)
        meat = np.zeros((2, 2))
        for _d, idx in sub.groupby("trade_date").indices.items():
            s = X[idx].T @ r[idx]
            meat += np.outer(s, s)
        se = float(np.sqrt(np.diag(xtx @ meat @ xtx)[1]))
        t_clust, n_clust = (b[1] / se if se > 0 else float("nan")), sub["trade_date"].nunique()

    # the STRICTEST control: same-night OW vs UW. Removes the night factor by construction,
    # but only the overlapping nights qualify, so it is usually badly underpowered.
    pair = (sub.groupby(["trade_date", "tilt"])["ON"].mean().unstack().dropna())
    n_pair = len(pair)
    if n_pair > 2:
        pd_ = pair["OVERWEIGHT"] - pair["UNDERWEIGHT"]
        t_pair = float(pd_.mean() / (pd_.std(ddof=1) / np.sqrt(n_pair))) if pd_.std(ddof=1) else float("nan")
        diff_pair = float(pd_.mean())
    else:
        t_pair = diff_pair = float("nan")

    passes = bool(np.isfinite(diff_on) and diff_on > 0 and np.isfinite(t_on)
                  and abs(t_on) >= 2.0 and yrs_ok >= 5)
    return {"buckets": buckets, "years": years, "n_signals": len(sel),
            "unmatched": unmatched, "cost_bps": cost, "gated": gated,
            "diff_ON": round(diff_on, 1), "t_ON": round(t_on, 2),
            "diff_D2": round(diff_d2, 1), "t_D2": round(t_d2, 2),
            "diff_night": round(diff_night, 1), "t_night": round(t_night, 2),
            "diff_xs": round(diff_xs, 1), "t_xs": round(t_xs, 2),
            "t_clustered": round(t_clust, 2), "n_nights_clustered": n_clust,
            "nights_ow": int(ow["trade_date"].nunique()), "nights_uw": int(uw["trade_date"].nunique()),
            "n_paired_nights": n_pair, "diff_paired": round(diff_pair, 1),
            "t_paired": round(t_pair, 2),
            "yrs_ow_wins": yrs_ok, "yrs_tested": yrs_tested,
            "measured_on": pd.Timestamp.today().strftime("%Y-%m-%d"),
            "rule": _GATE_RULE, "gate_verdict": "WIRE AS GATE" if passes else "DISPLAY-ONLY"}


def save_measurement(m: dict, path=TILT_MEASUREMENT) -> None:
    """Persist the scalar findings + the bucket table so the board renders MEASURED numbers."""
    import json
    out = {k: v for k, v in m.items() if k not in ("buckets", "years")}
    out["buckets"] = m["buckets"].to_dict("records") if len(m["buckets"]) else []
    out["years"] = m["years"].to_dict("records") if len(m["years"]) else []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


def load_measurement(path=TILT_MEASUREMENT) -> dict | None:
    """The persisted measurement, or None if it has never been run on this machine."""
    import json
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def format_measurement(m: dict) -> str:
    L = ["", "=" * 84,
         f"  SECTOR TILT vs THE OVERNIGHT PAYOFF   "
         f"{'regime-gated' if m['gated'] else 'UNGATED'} · cost {m['cost_bps']:.0f}bps · "
         f"n={m['n_signals']} triggers ({m['unmatched']} with no ranked sector)",
         "=" * 84,
         f"  {'tilt':<13}{'n':>6}{'grossON':>9}{'netON':>8}{'win%':>7}{'t':>7}{'D2net':>8}"]
    for _, r in m["buckets"].iterrows():
        L.append(f"  {r['tilt']:<13}{int(r['n']):>6}{r['gross_ON']:>+9.1f}{r['net_ON']:>+8.1f}"
                 f"{r['win%']:>6.1f}%{r['t_vs_0']:>7.2f}{r['D2_net']:>+8.1f}")
    L += ["",
          f"  OW − UW  OVERNIGHT (what this book can collect) : {m['diff_ON']:+.1f} bps  "
          f"t={m['t_ON']:+.2f}   (night-clustered t={m['t_clustered']:+.2f}, "
          f"{m['n_nights_clustered']} nights)",
          f"  OW − UW  DAY-2 INTRADAY (what it cannot)        : {m['diff_D2']:+.1f} bps  "
          f"t={m['t_D2']:+.2f}",
          "  If the spread sits in DAY-2 and not OVERNIGHT, the sector signal is real and this",
          "  engine still cannot monetise it — the hold is close→next-open, never into day 2.",
          "",
          "  ── IS IT CROSS-SECTIONAL, OR JUST BETTER NIGHTS? ──────────────────────────────",
          f"  OW fired on {m['nights_ow']} nights, UW on {m['nights_uw']} — overlapping on only "
          f"{m['n_paired_nights']}. So the raw number above is mostly a BETWEEN-NIGHT",
          "  comparison, and this project's own decomposition says a night is dominated by "
          "market-wide",
          "  overnight BETA (~+16bps) rather than by selection. Splitting the raw difference:",
          f"    WHICH NIGHTS  (universe gap on each bucket's nights) : {m['diff_night']:+.1f} bps  "
          f"t={m['t_night']:+.2f}   ← timing, NOT a sector signal",
          f"    CROSS-SECTIONAL (excess over that night's universe)  : {m['diff_xs']:+.1f} bps  "
          f"t={m['t_xs']:+.2f}   ← the only part a sector call could own",
          f"    SAME-NIGHT OW vs UW (strictest, {m['n_paired_nights']} nights)          : "
          f"{m['diff_paired']:+.1f} bps  t={m['t_paired']:+.2f}   ← usually too few nights to resolve",
          "",
          f"  {'year':>6}{'n_OW':>7}{'n_UW':>7}{'OW net':>9}{'UW net':>9}{'OW wins':>9}"]
    for _, r in m["years"].iterrows():
        ow = f"{r['ow_net']:+.1f}" if r["ow_net"] is not None else "—"
        uw = f"{r['uw_net']:+.1f}" if r["uw_net"] is not None else "—"
        w = "—" if r["ow_wins"] is None else ("yes" if r["ow_wins"] else "no")
        L.append(f"  {int(r['year']):>6}{int(r['n_ow']):>7}{int(r['n_uw']):>7}{ow:>9}{uw:>9}{w:>9}")
    L += ["", f"  {m['rule']}",
          f"  → OW beat UW in {m['yrs_ow_wins']} of {m['yrs_tested']} tested years.",
          f"  → VERDICT: {m['gate_verdict']}", ""]
    if m["gate_verdict"] == "DISPLAY-ONLY":
        L.append("  The column stays CONTEXT. Do not wire it into selection, ranking or size.")
    return "\n".join(L)


# SHORT tooltip. Streamlit's help popup CLIPS long text (it has no scrollbar of its own), so
# the hover version has to stand alone at a readable length; the full evidence lives in HELP_FULL,
# rendered in an on-page expander. Keeping the honesty warnings in BOTH — they are the part a
# reader must not miss, so they cannot be the part that gets cut off.
HELP = (
    "WHICH SIDE OF THE SECTOR ROTATION THIS NAME IS ON — from the Daily_Cash_Market 1–2 week "
    "forward sector tilt.\n\n"
    "🟢 OW = sector in the top quartile of relative momentum · 🔴 UW = bottom quartile · "
    "⚪ NEUTRAL = the middle · 👁 WATCH = heavy delivery accumulation but momentum has not "
    "turned yet. '#15/24' = the sector's strength rank among the sectors ranked today "
    "(#1 = strongest). '⚠thin(3)' = fewer than 5 liquid names, so the rank is noisy.\n\n"
    "CONTEXT ONLY — nothing in the engine reads it. It does not filter, rank or size anything:\n"
    "• The tilt is validated over 10 TRADING DAYS. This engine holds ONE NIGHT. Measured on 8yr "
    "of this engine's own signals, it does NOT help — it runs BACKWARDS (weak sectors paid more), "
    "and about a third of even that is just which NIGHTS each bucket traded, not the sector.\n"
    "• RELATIVE, NOT ABSOLUTE: the best-ranked sector in the market can still be falling.\n"
    "• 🔴 UW IS NOT A SHORT SIGNAL. This engine's short side is separately proven dead.\n\n"
    "See the '🧭 Sector rotation context' expander for the full numbers."
)

HELP_FULL = (
    "WHICH SIDE OF THE ROTATION THIS NAME'S SECTOR IS ON — read from the Daily_Cash_Market "
    "1–2 week forward sector tilt, the one sector call that survived deep validation there "
    "(cross-sectional sector momentum vs Nifty, daily-IC t≈9, Monte-Carlo p<0.002 vs 600 "
    "random portfolios, cost-robust to 40bps). 🟢 OW = the sector ranks in the top quartile "
    "of relative momentum; 🔴 UW = bottom quartile; ⚪ NEUTRAL = the middle; 👁 WATCH = heavy "
    "delivery accumulation but momentum has NOT turned yet (contrarian, deliberately held "
    "out of the active tilt because momentum is the validated timer). '#2/14' is the "
    "sector's rank among the ranked sectors today.\n\n"
    "THREE THINGS THIS COLUMN IS NOT, and they matter more than the badge:\n"
    "(1) NOT MEASURED FOR THIS HOLD. The tilt is validated over 10 TRADING DAYS. This engine "
    "holds OVERNIGHT and never into day 2 (day-2 holds are net-negative every year). Whether "
    "the sector edge lands in the overnight GAP — the only part this book collects — or in "
    "the day sessions is a separate question, and until it is measured this column is "
    "CONTEXT. Nothing in the engine reads it: it does not filter, rank or size anything.\n"
    "(2) RELATIVE, NOT ABSOLUTE. OVERWEIGHT means 'strong versus other sectors', not 'going "
    "up'. The best-ranked sector in the market can still be falling, and this project's edge "
    "is long-only ABSOLUTE return under a Nifty regime gate — cross-sectional neutrality was "
    "measured to kill it (−36bps). A green badge is context, not permission.\n"
    "(3) UNDERWEIGHT IS NOT A SHORT. In DCM's own words a sector basket cannot be shorted "
    "cheaply; and this engine's overnight short side is separately proven dead (−41.9bps, win "
    "20.2% — weak closes BOUNCE overnight). On a SHORT row 'with your SHORT' means the "
    "context agrees, nothing more.\n\n"
    "A dash is an answer, not a gap: '— thin' = fewer than 5 liquid names in the sector, so "
    "its rank is noise and an overweight is withheld; '— sector not ranked' = the name's "
    "sector is outside the ranked cross-section (ETF/Others/unmapped). As-of the last "
    "COMPLETED close — during a live session that is yesterday, because today has not closed."
)
