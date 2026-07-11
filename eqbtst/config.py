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

from pathlib import Path

# --- data source: the Daily_Cash_Market DuckDB archive (read-only) ------------
DCM_DUCKDB = Path(r"d:/Python Projects/Daily_Cash_Market/data/market_data.duckdb")
NIFTY_INDEX_NAME = "Nifty 50"          # index_data.index_name for the regime gate

# --- live feed: reuse the Tradebot Fyers auth/token (no re-plumbing) -----------
TRADEBOT_DIR = Path(r"d:/Python Projects/Tradebot")
FYERS_HISTORY_URL = "https://api-t1.fyers.in/data/history"
FYERS_QUOTES_URL = "https://api-t1.fyers.in/data/quotes"
NIFTY_FYERS = "NSE:NIFTY50-INDEX"      # live index for the regime / relative strength

# --- LOCKED conviction signal (the accumulation footprint) --------------------
CLR_TH      = 0.70    # close in top 30% of day range  -> strong close
DELIV_TH    = 60.0    # delivery % of traded volume     -> real ownership taken
DELIV_SPIKE = 10.0    # delivery% minus own 20d median   -> ABNORMAL accumulation
VOL_TH      = 2.0     # volume vs own 20d median          -> participation surge
RET_TH      = 0.01    # up at least +1% on the day        -> demand in control
CVWAP_TH    = 0.005   # close >=0.5% above session VWAP (avg_price) -> PATH-persistent
RS_LOOKBACK = 10      # window for cumulative relative strength (trading days)
RS_MIN      = 0.0     # require 10d cumulative RS vs Nifty > 0 -> PERSISTENT leader
LIQ_MIN_LACS = 2000.0 # >= Rs 20cr traded that day -> fillable without moving price
LOOKBACK    = 20      # rolling window for the medians (trading days)
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

# --- costs (report gross AND net; the binding constraint) ---------------------
COST_BPS    = 22.0    # realistic retail BTST round-trip (STT + brokerage + slippage)

# --- universe -----------------------------------------------------------------
# F&O single-stock universe from the DCM F&O bhavcopy (the liquid, shortable-
# free, no-theta cash names). ~270 symbols, the practical "Nifty 200 F&O" set.
FNO_INSTRUMENTS = ("FUTSTK", "OPTSTK")

# --- paper ledger (Phase 1 is PAPER ONLY — nothing auto-executes) -------------
LEDGER = Path(__file__).resolve().parent.parent / "data" / "validation" / "paper_ledger.parquet"
SIM_LEDGER = Path(__file__).resolve().parent.parent / "data" / "validation" / "sim_ledger.parquet"
