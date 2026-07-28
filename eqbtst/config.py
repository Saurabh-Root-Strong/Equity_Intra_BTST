"""
config.py — LOCKED parameters for the delivery-conviction overnight equity edge.

Every threshold here was fixed by the 8-year validation on the DCM cash-market
archive (2018–2026, 270 F&O stocks). They are LOCKED: no in-sample tuning. The
honest read from that validation (see README "Evidence"):

  * The tradeable edge is an OVERNIGHT gap: a smart-money accumulation footprint
    (strong close + high delivery% + delivery spike + volume surge on an up day)
    drifts UP into the next morning. GROSS win ~58–66%, persists into 2025–26.
  * It is COST-BOUND. At a 22 bps retail BTST round-trip the naive signal is net
    ~0 to negative in 2024–26. It only clears cost when (a) regime-gated to a
    Nifty uptrend and (b) executed better than the blind close/open prints.
  * Holding into DAY 2 is DEAD (buy-next-open/sell-next-close is negative every
    year). Capture the overnight gap ONLY. LONG ONLY (shorts fight the drift).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- data source: the Daily_Cash_Market DuckDB archive (read-only) ------------
# ENV-OVERRIDABLE so the VM (different filesystem layout) needs no code edit -- set
# EQBTST_DCM_DUCKDB / EQBTST_TRADEBOT_DIR there; the local dev paths are the fallback.
DCM_DUCKDB = Path(os.environ.get(
    "EQBTST_DCM_DUCKDB", r"d:/Python Projects/Daily_Cash_Market/data/market_data.duckdb"))
NIFTY_INDEX_NAME = "Nifty 50"          # index_data.index_name for the regime gate

# --- live feed: reuse the Tradebot Fyers auth/token (no re-plumbing) -----------
TRADEBOT_DIR = Path(os.environ.get("EQBTST_TRADEBOT_DIR", r"d:/Python Projects/Tradebot"))
FYERS_HISTORY_URL = "https://api-t1.fyers.in/data/history"
FYERS_QUOTES_URL = "https://api-t1.fyers.in/data/quotes"
NIFTY_FYERS = "NSE:NIFTY50-INDEX"      # live index for the regime / relative strength

# --- LOCKED conviction signal (the accumulation footprint) --------------------
# LEAK-FREE by construction: every leg is knowable at the 15:15 close. Today's OHLC/
# volume are known at the close; delivery% is NOT (NSE publishes it ~6pm), so the
# delivery leg uses the TRAILING average through t-1 — real accumulation persists
# across days, and trailing delivery nets the same edge without the look-ahead.
CLR_TH        = 0.70   # close in top 30% of day range   -> strong close
DELIV_TRAIL_TH = 60.0  # trailing N-day AVG delivery% (through t-1) -> sustained accumulation
DELIV_TRAIL_WIN = 3    # trailing window for the delivery average (days, all published by t-1)
VOL_TH        = 2.0    # volume vs own 20d median          -> participation surge
RET_TH        = 0.01   # up at least +1% on the day        -> demand in control
# today's delivery% (DELIV_TH) is a POST-HOC confirmation only — it lands ~6pm, after
# entry — so it is shown as a quality check in the EOD tab, never a signal input.
DELIV_TH    = 60.0     # today's delivery% threshold, for the post-hoc confirmation badge
CVWAP_TH    = 0.005   # close >=0.5% above session VWAP (avg_price) -> PATH-persistent
RS_LOOKBACK = 10      # window for cumulative relative strength (trading days)
RS_MIN      = 0.0     # require 10d cumulative RS vs Nifty > 0 -> PERSISTENT leader
LIQ_MIN_LACS = 2000.0 # >= Rs 20cr traded that day -> fillable without moving price
LOOKBACK    = 20      # rolling window for the medians (trading days)

# --- structure classifier (CONTEXT label; magnitude-gated, ATR-normalised) --------
# The label is defined by SIZE of move, not just topology — a marginal new high is NOT
# a breakout. Thresholds are in ATR (the frame's OWN volatility unit) so they auto-scale:
# ~0.5xATR on a daily frame is ~1-1.5% above the range; the same rule on 15m or 1W stays
# sensible. Raise STRUCT_BREAKOUT_ATR toward 1.0 to demand a stronger (~2.5-3% on daily)
# break; lower it to catch earlier ones.
STRUCT_LOOKBACK     = 20    # bars of history the label reads (per frame)
STRUCT_BREAKOUT_ATR = 0.5   # close must clear the prior 19-bar range by >= this x ATR
STRUCT_TREND_ER     = 0.40  # Kaufman efficiency ratio for a clean, low-noise trend
STRUCT_TREND_ATR    = 1.0   # AND the window's net move must cover >= this x ATR (a REAL move)
STRUCT_COIL         = 0.60  # recent 3-bar range < this x the prior range => contracting (coil)

# ── TOUCH-COUNTED SUPPORT / RESISTANCE ───────────────────────────────────────────────
# These decide WHETHER TWO PIVOTS ARE "THE SAME LEVEL", so they decide the touch count --
# the whole value of the feature ("how many times price rejected here"). They were measured
# wrong. The old tol=0.2*ATR is TIGHTER THAN A SINGLE DAY'S RANGE, so two rejections a week
# apart at what a chartist calls one level almost never landed within tolerance: 67% of all
# levels came out 1-touch, i.e. the count never accumulated. A human reads a level as a ZONE
# ~2-3% wide; on a daily frame ATR is ~3%, so that is ~0.6-1.0 ATR, not 0.2. Measured across
# 180 names: at 0.6, multi-touch levels go from 33% -> 65% and 3+ touch from 9% -> 39%,
# without collapsing into 2-3 mega-zones (that starts near 1.0). Verified on MOTILALOFS
# against a hand-drawn chart: 976 x5, 905 x3, 816 x2 -- the exact walls the eye picks, where
# the old setting split each into 1-touch pivots. SR_TOL_ATR is used in BOTH clustering sites
# (indicators.sr_levels AND live._live_levels' live re-cluster); they MUST agree or the second
# re-splits what the first merged.
SR_TOL_ATR   = 0.6    # two pivots within this x ATR = the same level (zone half-width ~= this)
# A level PERSISTS far longer than a trend regime, so its window is longer than STRUCT_LOOKBACK.
# 40 bars (~2 months on daily) cut off levels the chart still respects -- MOTILALOFS' own May
# support/resistance, tested again in July, sat just outside it. 60 bars (~3 months daily,
# ~2.5 days on 15m, ~14 months on weekly) is the standard "what you see on the chart" window.
SR_LOOKBACK  = 60     # bars of history S/R levels are drawn from (NOT the 20-bar label window)
# LIQ_MIN_LACS is the REALISM floor. Stress test: the headline +30bps net was partly
# flattered by thin names where the fill is not real. On genuinely liquid names
# (>=Rs20cr turnover) the honest deployable edge is ~+16-19bps net, win ~62%. Thin
# names are excluded because their +30 is a mirage slippage would eat.
#
# EXECUTION assumption is LOCKED to the only fillable one: ENTER near the close
# (15:15-15:25, ~= close_price), EXIT next morning (~= next open). Do NOT "improve"
# entry by assuming a VWAP fill — on a strong-close signal day VWAP sits well below
# the close, so a VWAP entry is an impossible look-ahead (it backtests to 97% win).
# RS_MIN is the validated relative-strength refiner: a name that has PERSISTENTLY
# outperformed the index over ~10 sessions (not a one-day burst) carries the
# overnight edge better. On 8yr it holds net ~+30bps while lifting the weak years
# (2025 +0.8->+5.9, 2026 +4.5->+15.1); burst-only laggards fall to +19bps. LOCKED.
# CVWAP_TH is the validated "path signature" refiner (a la Stock A spike-fade vs
# Stock B accumulation): a green day that CLOSES well above its own VWAP trended up
# and held, vs one that spiked and faded to VWAP. On 8yr it lifts net +26->+30bps,
# win 57->61%, and FLIPS the losing 2025 year positive (-11.6 -> +6.2). LOCKED.

# --- regime gate (mandatory; gating revives net edge in 4 of 5 recent years) --
REGIME_MA   = 50      # Nifty > its 50-day MA => risk-on => full size

# --- direction: LONG ONLY (proven, not a preference) --------------------------
# The overnight equity drift is structurally long-biased. The mirror-image SHORT
# (distribution footprint, gated to a downtrend) was tested on 8yr: net -41.9bps,
# win 20.2% — weak-close names BOUNCE overnight ~80% of the time. Shorting overnight
# fights the tape. No short side. (A same-day intraday short is a different animal,
# needs ticks, and faces the same cost floor — Phase 2 question, not this engine.)
LONG_ONLY = True

# --- selection, concentration & sizing ----------------------------------------
TOP_N          = 5    # trade at most the best N names per night
MAX_PER_SECTOR = 2    # N longs in one sector = one overnight bet; cap the correlation
MAX_HOLD       = "overnight"   # capture close -> next-morning. NEVER day-2.
# gap-tail: this is unhedged overnight equity beta. Worst historical signal night
# ran to a multi-% gap-down. Size each name so a ~-8% shock gap is survivable, and
# cap gross overnight exposure. Equal-weight across the selected names by default.

# --- expected-move band (the ONLY validated forecast product: RANGE, not a target) --
# Calibrated on the F&O universe (last yr, |move|/ATR percentiles): where price is
# LIKELY to be, not a prediction of where it goes. ±0.6 ATR contains ~68% of next-day
# CLOSES; ±1.0 ATR contains ~74% of the full next-day HIGH/LOW range; the overnight
# (BTST exit) move is small, ~±0.2 ATR at 68%.
BAND_CLOSE_68 = 0.6   # ± this × ATR = ~68% next-day close band
BAND_RANGE    = 1.0   # ± this × ATR = ~74% next-day full-range band
BAND_ON_68    = 0.2   # ± this × ATR = ~68% overnight (close->next open) band

# --- costs (report gross AND net; the binding constraint) ---------------------
COST_BPS    = 22.0    # realistic retail BTST round-trip (STT + brokerage + slippage)

# --- universe -----------------------------------------------------------------
# F&O single-stock universe from the DCM F&O bhavcopy (the liquid, shortable-
# free, no-theta cash names). ~270 symbols, the practical "Nifty 200 F&O" set.
FNO_INSTRUMENTS = ("FUTSTK", "OPTSTK")

# --- paper ledger (Phase 1 is PAPER ONLY — nothing auto-executes) -------------
LEDGER = Path(__file__).resolve().parent.parent / "data" / "validation" / "paper_ledger.parquet"
SIM_LEDGER = Path(__file__).resolve().parent.parent / "data" / "validation" / "sim_ledger.parquet"


# --- intraday volume PACE profile (time-of-day normalisation) ---------------
# Median cumulative %% of a day's volume completed by each 5-min mark (120 F&O stocks x 2yr).
# Used to TIME-NORMALIZE the intraday volume surge: cum_vol/med_daily is meaningless at
# 10am (only ~22%% of the day has traded), so a genuine 2x-volume day would read 0.44 and
# fail the 2.0 gate. Dividing by this fraction gives PACE: a true 2x day reads 2.0 at any
# hour. At 15:25 the fraction is 1.0, so the CLOSE decision is mathematically unchanged
# (the 8yr backtest used the full-day ratio -- that integrity is preserved).
VOL_PROFILE = {
    "09:15": 0.0424, "09:20": 0.0674, "09:25": 0.0874, "09:30": 0.1073,
    "09:35": 0.1232, "09:40": 0.1383, "09:45": 0.1546, "09:50": 0.1695,
    "09:55": 0.1827, "10:00": 0.1960, "10:05": 0.2104, "10:10": 0.2216,
    "10:15": 0.2340, "10:20": 0.2464, "10:25": 0.2586, "10:30": 0.2696,
    "10:35": 0.2813, "10:40": 0.2935, "10:45": 0.3047, "10:50": 0.3175,
    "10:55": 0.3289, "11:00": 0.3413, "11:05": 0.3519, "11:10": 0.3609,
    "11:15": 0.3707, "11:20": 0.3812, "11:25": 0.3899, "11:30": 0.4004,
    "11:35": 0.4121, "11:40": 0.4228, "11:45": 0.4328, "11:50": 0.4428,
    "11:55": 0.4521, "12:00": 0.4648, "12:05": 0.4739, "12:10": 0.4842,
    "12:15": 0.4954, "12:20": 0.5052, "12:25": 0.5165, "12:30": 0.5269,
    "12:35": 0.5365, "12:40": 0.5454, "12:45": 0.5553, "12:50": 0.5642,
    "12:55": 0.5740, "13:00": 0.5853, "13:05": 0.5948, "13:10": 0.6050,
    "13:15": 0.6134, "13:20": 0.6230, "13:25": 0.6335, "13:30": 0.6413,
    "13:35": 0.6512, "13:40": 0.6604, "13:45": 0.6715, "13:50": 0.6812,
    "13:55": 0.6911, "14:00": 0.7022, "14:05": 0.7110, "14:10": 0.7234,
    "14:15": 0.7340, "14:20": 0.7450, "14:25": 0.7563, "14:30": 0.7684,
    "14:35": 0.7815, "14:40": 0.7944, "14:45": 0.8083, "14:50": 0.8215,
    "14:55": 0.8367, "15:00": 0.8608, "15:05": 0.8853, "15:10": 0.9115,
    "15:15": 0.9436, "15:20": 0.9742, "15:25": 1.0000,
}


# ── PRICE BAND DEFAULTS — YOURS TO SET ────────────────────────────────────────────────
# What the board's Min/Max Rs boxes start at each session. This is a POSITION-SIZING
# preference (how many shares fit your capital), so it is your call, not a derived value —
# set either to whatever you want, or 0 to disable that side of the band.
#
#     PRICE_MAX_DEFAULT = 0      -> no cap, every scanned name reaches the table
#     PRICE_MAX_DEFAULT = 900    -> only names <= Rs900
#
# ONE THING TO KNOW WHEN YOU PICK A NUMBER, because it is invisible otherwise: the band cuts
# BEFORE the structure logic runs, so it decides what the horizon dropdown is even allowed to
# consider. Measured on a live 243-name board, a Rs900 cap removed 137 names (56%) — and
# those included RELIANCE, ICICIBANK, INFY, AXISBANK, TECHM, BAJFINANCE, the most liquid
# names in the universe — taking LONG candidates from 9 to 2 and SHORT from 13 to 7. That is
# a fine trade-off if the cap reflects real position sizing; it is only a problem when it is
# forgotten. The board now states the cut and the count on screen whenever the band is on.
PRICE_MIN_DEFAULT = 0.0
PRICE_MAX_DEFAULT = 900.0
# YOUR number, and it is a good one. Measured over 8 years of footprint triggers, the
# overnight payoff falls MONOTONICALLY with share price -- this is a real cross-sectional
# effect, not a sizing preference:
#     <250      n=116  +47.3bps   t=3.79   win 66.4%
#     250-500   n=111  +27.6      t=2.21
#     500-900   n=131  +27.1      t=1.93
#     900-2000  n=215   +9.2      t=1.69
#     2000-5000 n=133   +4.4      t=0.70
#     >5000     n= 42  +11.2      t=0.94
#   <=900 +33.8 (n=358)  vs  >900 +7.8 (n=390)   diff +26.0, t=3.04
# Cheap beat rich in 8 of 8 YEARS. NOT a liquidity artifact: the gap survives inside every
# turnover tercile and is WIDEST in the thickest one (+42.2). It also survives tick-aware
# costs (NSE Rs0.05 tick is 5.8bps on a Rs250 name vs 0.15bps on a Rs2000+ one): +28.6 vs
# +7.2. The one real caution is the <250 bucket, which pays ~11.6bps of round-trip tick drag
# -- still the best bucket after paying it, but sizing there needs limit orders, not market.
