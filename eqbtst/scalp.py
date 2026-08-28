"""SCALPER MODE — a 1-minute trigger inside a 5-minute box, for a 1-10 minute hold.

WHY THIS IS A SEPARATE LANE AND NOT A FIFTH HORIZON PRESET
----------------------------------------------------------
Adding "1m/5m" to mtf.PRESETS would have been three lines. It would also have been wrong,
because at a five-minute hold almost every column on the structure board stops describing
anything that can happen inside the trade:

  * `wtd_deliv7`, `deliv_vs_100d`   delivery is settled at the END of the session. It cannot
                                    change between 10:31 and 10:36.
  * the F&O block                   the bhavcopy publishes after the close, so it describes
                                    YESTERDAY's derivatives book.
  * `carry`                         an overnight, one-night-only annualised basis read
                                    (see arb.py). A scalper never holds the night.
  * `sector tilt`                   a multi-WEEK rotation read rendered beside a trade that
                                    is over before the next update of it exists.

None of those are wrong; they are answers to questions a scalper is not asking, and every one
costs an archive read and a column of width. So scalper mode drops the whole positioning /
delivery / carry stack and spends the space on the only two things that decide a five-minute
trade: HOW FAR the name can travel in five minutes, and WHAT THAT TRIP COSTS.

THE COST FLOOR IS NOT 22 bps HERE — THE ONE PIECE OF GOOD NEWS
---------------------------------------------------------------
config.COST_BPS = 22 is a DELIVERY round trip, and it is ~22 mostly because delivery STT
(Securities Transaction Tax) is 0.1% on BOTH legs = 20bps on its own. An intraday square-off
does not pay that: STT is 0.025% on the SELL leg only. Rebuilt from the statutory schedule:

      position       total   brokerage  exch  sebi   gst   stt  stamp  spread
      Rs   50,000   11.60bps      6.00  0.59  0.02  1.19  2.50   0.30    1.00
      Rs  100,000    9.24         4.00  0.59  0.02  0.83  2.50   0.30    1.00
      Rs  200,000    6.88         2.00  0.59  0.02  0.47  2.50   0.30    1.00
      Rs  500,000    5.47         0.80  0.59  0.02  0.25  2.50   0.30    1.00
      Rs 1,000,000   5.00         0.40  0.59  0.02  0.18  2.50   0.30    1.00

So a scalp faces roughly 5-12bps, not 22 -- and the number moves with POSITION SIZE, because
the Rs20-per-order brokerage is a fixed cost being amortised. That is why this lane asks for
your position size: at Rs50k the flat brokerage alone is 6bps and eats the trade; at Rs5L it
is 0.8bps and nearly vanishes. Same setup, opposite verdict.

AND HERE IS THE BAD NEWS, MEASURED
-----------------------------------
480,000 one-minute bars, the 40 most liquid F&O names, 32 sessions (2026-07-13..2026-08-25 --
the broker caps 1-minute history at exactly 12,000 candles per symbol, so 32 sessions is ALL
there is). Absolute forward move, in bps:

      horizon   median   p75    p90    P(|move| > 10bps)
      1 min       2.9     6.2   11.0        11.7%
      5 min       6.7    13.5   23.7        35.0%
      10 min      9.4    18.8   32.9        47.8%

Read the 5-minute row against the cost table. The MEDIAN five-minute move on the most liquid
names in the market is 6.7bps. At a Rs1L position the round trip is 9.24bps. So on a typical
name at a typical moment, A TRADER WHO CALLED THE DIRECTION CORRECTLY EVERY SINGLE TIME would
still lose money on a five-minute hold. That is not a statement about skill; it is arithmetic
on the size of the thing being divided.

DIRECTION IS A COIN FLIP AT THIS HORIZON
-----------------------------------------
Same panel, causal, 456,960 observations: P(next 5 min is up) = 46.2%, mean forward move
+0.11bps, IC(this bar's range, next 5 min SIGNED) = +0.024. Nothing. (A tempting
"autocorrelation" of +0.45 appears if you regress the next 1 minute against the next 5 -- that
is WINDOW OVERLAP, the 1-minute return is literally a component of the 5-minute one, and it is
not a signal.) This matches every other directional result in this stack: the 60m price-action
study, the band-fade retraction, and the intraday hunt config.py already calls closed.

Because direction is a coin flip, expected value on any single scalp is exactly MINUS THE
COST, whatever the chart looks like. Which fixes what this lane can honestly be:

      SIZE is predictable.   DIRECTION is not.

      IC(trailing 10-bar mean 1m range, |next 5 min|) = +0.399   <- large, and stable
      IC(relative volume,               |next 5 min|) = +0.227

      by trailing-range quintile:   Q0 mean |5m| 6.30bps  (P>10bps 19.5%)
                                    Q4 mean |5m| 17.67bps (P>10bps 56.3%)
      and the direction split is FLAT across those quintiles (45.4% .. 46.7% up) --
      liveliness buys you SIZE, never a side.

That last point survives the strongest test available: split the board's OWN side call by
expected move and the signed edge stays flat while the move size TRIPLES.

      exp-move quintile   n        signed e5   t_clustered   mean |5m move|
      Q0                  21,763   -0.035bps      -0.21          7.0bps
      Q2                  21,749   -0.218          -1.16         10.4
      Q4                  21,759   +0.399          +0.70         22.3

Three times the movement, and not one basis point of direction to show for it.

AND THE SETUP TAG ITSELF EARNS NOTHING — MEASURED, AND ONCE MEASURED WRONG
---------------------------------------------------------------------------
The 1m x 5m read this lane renders was replayed CAUSALLY over the same panel: 475,840
evaluations, structure rebuilt from candles truncated to each 1-minute bar close, run through
the real indicators.struct_full -> mtf.synthesize -> mtf.side_of pipeline. Forward 5-minute
return signed the way the trade would be taken, error bars CLUSTERED BY SESSION (bars inside
one day are nowhere near independent, and the naive t-stat is 3-6x too generous):

      tag                       side       n      e5    t_clus  sess+   net @Rs2L
      PULLBACK vs HTF           LONG   1,130  +3.118     1.99    69%      -3.77
      WITH-TREND CONTINUATION   LONG   4,968  +1.532     1.58    69%      -5.35
      RANGE-TOP BREAK           LONG   5,842  +0.522     0.74    50%      -6.36
      WITH-TREND CONTINUATION   SHORT  4,096  +0.351     0.32    53%      -6.53
      EXTENDED (aligned)        LONG  14,743  -0.038    -0.07    59%      -6.92
      COIL AT THE EXTREME       LONG  30,739  -0.092    -0.34    50%      -6.98
      PULLBACK vs HTF           SHORT  1,204  -0.133    -0.11    53%      -7.02
      COIL AT THE EXTREME       SHORT 33,297  -0.399    -1.94    47%      -7.28
      EXTENDED (aligned)        SHORT 13,645  -1.344    -2.78    28%      -8.23
      RANGE-FLOOR BREAK         SHORT  5,414  -1.670    -4.12    31%      -8.55

NOT ONE CELL CLEARS COST. The best of them is +3.1bps gross against a 6.88bps floor at Rs2L
(-2.35bps even at Rs5L, the most favourable size on the sheet), and it is the SMALLEST cell on
the board at n=1,130. The LONG/SHORT split separates nothing at all: LONG minus SHORT is
-0.04bps at five minutes, t = -0.33. The short side is separately and significantly NEGATIVE
(clustered t -3.24, only 19% of sessions positive), which independently reproduces
mtf.SHORT_EDGE_BPS from a completely different direction.

⚠ THE FIRST VERSION OF THIS TABLE WAS A LOOKAHEAD, and it is worth recording what it looked
like, because it looked GREAT: WITH-TREND CONTINUATION at +16.6bps, t = 33, 100% of sessions
positive, every cell an order of magnitude larger. The bug was one line -- each 1-minute bar
was indexed to the 5-minute bar CONTAINING it (searchsorted(..., "right") - 1) rather than the
last bar to have CLOSED, so the confirm frame's newest bar carried up to four minutes of the
future against a five-minute forward return. The tell was already in the table before the bug
was found: RANGE-TOP BREAK came out NEGATIVE at -5.9bps while continuation tags were hugely
positive, which is the signature of a peeked bar (the move is already inside the window, so
whatever comes next is the fade), not of a chart pattern. The shipped path carries the same
guard in _closed(), for the same reason.

SO THIS BOARD IS A VETO, NOT A TRIGGER
---------------------------------------
It answers one question per name: CAN a five-minute move in this name, right now, pay for its
own round trip? If the answer is no, the setup is irrelevant -- a textbook coil on a name that
travels 4bps in five minutes is not a trade at any hit rate. That is a SUBTRACTION, which is
the only kind of thing that has ever worked in this project (cf. the ex-dividend drop in
arb.py). YOU supply the direction from the tape. The board refuses to.

`p pays` is therefore P(|move| > cost), NOT P(profit). It is a NECESSARY condition and not a
sufficient one, and the page says so in those words.

WHICH NAMES TO EVEN SCAN — RANK BY RANGE, NOT BY TURNOVER
----------------------------------------------------------
The obvious default (scan the most liquid names) is measurably backwards. Across the same 40
names, the FREE end-of-day archive proxy `atr14 / close` correlates with realised mean
|5-minute move| at Spearman +0.902 -- while TURNOVER correlates -0.10. The megacaps are the
least scalpable names in the market:

      HDFCBANK  atr 1.03%  ->  5.85bps mean |5m move|   (rank 40 of 40)
      RELIANCE  atr 1.42%  ->  6.54bps                  (rank 39)
      HFCL      atr 3.88%  -> 19.15bps                  (rank  1)

      top-10 by TURNOVER: mean |5m| 10.08bps
      top-10 by ATR%    : mean |5m| 15.44bps   (+53%)

So the scan ranks by daily ATR%, from the archive, at zero API cost -- and uses turnover only
as a FILLABILITY FLOOR, never as a ranking key. CAVEAT, stated because the sample cannot
support more: that +0.902 was measured INSIDE the top-40 liquidity band. Ranking the whole
242-name universe by ATR% alone would pull in names whose spread eats the move, which is
exactly what the turnover floor is there to stop.

THE READ IS MAXIMALLY PROVISIONAL, BY CONSTRUCTION
---------------------------------------------------
mtf.REPAINT already measures that the 15m-trigger Intraday preset does not settle its tag
until the session is 92% over. A 1-minute trigger is the far end of that same law: the tag can
change every sixty seconds, and it is SUPPOSED to. That is not a defect to be smoothed away --
a scalper wants the freshest possible read -- but it does mean the setup tag here is a
snapshot, never a filter you can leave running.

WHAT THE SCAN COSTS, AND WHY THERE IS A SIZE SLIDER
----------------------------------------------------
One 1-minute /history fetch per name, rate-paced at live._HIST_GAP (0.33s -> ~180 req/min, a
MEASURED ceiling; see live.py). So scan wall-clock is ~0.33s x N:

      25 names ->  ~8s      40 -> ~13s      80 -> ~26s      242 -> ~80s

A 1-minute trigger frame wants a re-scan every 60 seconds. The full universe takes EIGHTY, so
the full universe cannot be scalped -- the scan would be permanently behind its own trigger
bar. Hence the slider, and hence the page printing the estimate beside it.

This module never imports streamlit: `render_page(st)` takes it as an argument, so the whole
page is testable headless (tests/test_scalp.py).
"""
from __future__ import annotations

import datetime as dt
import threading
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config, indicators, live, mtf

# ── HORIZON ──────────────────────────────────────────────────────────────────────────
# One preset. The pair is a 1m trigger inside a 5m box: the classical ~4x nesting ratio, and
# both are NATIVE broker resolutions (verified against /history -- 375 and 75 bars per session
# respectively, so no resampling, no stub, nothing inferred).
PRESETS: dict[str, dict] = {
    "scalp": {
        "label": "Scalp  ·  1m trigger / 5m confirm",
        "ltf": "1m", "htf": "5m", "hold": "1-5 minutes, 10 at the outside",
        "note": "The fastest pair the broker can serve. Direction at this horizon is a coin "
                "flip (46.2% up, IC +0.024 over 457k observations), so this board will not "
                "give you one. It tells you whether the name can MOVE far enough in five "
                "minutes to pay its own round trip, and leaves the side to you and the tape.",
    },
}
PRESET_ORDER = ["scalp"]

# The ONE-FRAME-UP context frame, for S/R only -- it never enters the side decision. Neither
# is a native broker resolution, so both are resampled from the 1-minute series. 375 minutes
# does not divide by either (375/7 = 53.6, 375/10 = 37.5), so BOTH end the session on a stub
# bar, which _resample folds into the bar before it. See _resample for why that matters.
UP_FRAMES = {"7m": 7, "10m": 10}
UP_DEFAULT = "10m"

# ── COST MODEL ───────────────────────────────────────────────────────────────────────
# Statutory intraday (MIS) equity schedule. NOT config.COST_BPS, which is the DELIVERY round
# trip and is ~22 mostly because delivery STT is 0.1% on both legs.
STT_SELL = 0.00025          # 0.025%, SELL leg only  (delivery: 0.1% on BOTH legs)
NSE_TXN = 0.0000297         # 0.00297% per leg
SEBI_FEE = 0.000001         # Rs10 per crore, per leg
STAMP_BUY = 0.00003         # 0.003%, BUY leg only
GST_RATE = 0.18             # on brokerage + exchange txn + SEBI fee
BROK_FLAT = 20.0            # Rs per executed order (discount broker)
BROK_PCT = 0.0003           # or 0.03% of turnover, whichever is LOWER
TICK = 0.05                 # NSE cash tick above Rs1 -- the floor under any spread
DEFAULT_SPREAD_TICKS = 2.0  # crossing the book once each way. OPTIMISTIC on a thin name.
DEFAULT_POSITION = 200_000  # Rs

# ── P(|move| > cost) ─────────────────────────────────────────────────────────────────
# Non-parametric, from the distribution of |next 5 min| / (trailing 10-bar mean 1m range) over
# 460,576 observations. A parametric fit was rejected: the ratio is right-skewed, and a
# normal/lognormal fit misprices exactly the tail the trade lives in.
#
# The ratio is remarkably STABLE, which is what makes one table legitimate for all names:
#     across the 40 names    median ratio 0.667 .. 0.894   (median of medians 0.821)
#     across vol quintiles   median ratio 0.975 (quietest) .. 0.739 (liveliest)
# The mild decline with volatility makes the table slightly OPTIMISTIC on the liveliest names
# and slightly pessimistic on the quietest -- an error of a few percentage points, and in the
# conservative direction for the names this board is least sure about.
_RATIO_X = (0.25, 0.40, 0.50, 0.60, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00)
_RATIO_P = (81.4, 72.3, 66.5, 61.0, 53.3, 42.3, 33.5, 26.5, 16.8, 10.8, 7.1)

# Expected |5-minute move| from the trailing 10-bar mean 1-minute range, both in bps. OLS over
# the same panel: |f5| = 2.271 + 0.854 * rv10, calibration error -1% to +5% through the middle
# eight deciles. The INTERCEPT is load-bearing: a pure ratio form (1.10 x rv10) under-predicts
# the quietest decile by 33%, i.e. it says a dead name is deader than it is -- the one
# direction of error that would hide a real trade rather than manufacture one.
EXP5_A, EXP5_B = 2.271, 0.854

# TIME OF DAY IS ALREADY INSIDE THE RANGE TERM. Raw, the opening 15 minutes carry a mean |5m
# move| of 20.75bps against 8.2-8.9 at midday -- a 2.4x swing that looks like it demands its
# own multiplier. It does not: after removing the rv10 fit, the residual by session slot runs
# -0.9 to +1.1bps against a 10.15bps mean. The opening is livelier because its BARS are wider,
# and rv10 sees that directly. No time-of-day term is applied, deliberately.

EVIDENCE = {
    "bars": 480_000, "names": 40, "sessions": 32,
    "span": "2026-07-13..2026-08-25",
    "cap": "the broker serves exactly 12,000 one-minute candles per symbol",
    "abs_move_bps": {1: 2.9, 5: 6.7, 10: 9.4},          # medians
    "p_up_5m": 46.18, "mean_fwd5_bps": 0.11, "ic_dir": 0.024,
    "ic_size_rv10": 0.399, "ic_size_rvol": 0.227,
    "rank_spearman_atr": 0.902, "rank_spearman_turnover": -0.101,
    # causal MTF replay (see the module docstring). n = evaluations, e5 = signed forward
    # 5-minute return in bps, t = CLUSTERED BY SESSION.
    "mtf_evals": 475_840,
    "mtf_best_cell": ("PULLBACK vs HTF", "LONG", 1_130, 3.118, 1.99),
    "mtf_long_minus_short_bps": -0.043, "mtf_long_minus_short_t": -0.33,
    "mtf_short_t_clustered": -3.24,
    "mtf_cells_clearing_cost": 0,
}

# The causal MTF replay's cell table as DATA (tag, side, n, e5 bps, t clustered, % sessions
# positive), so the page renders the same numbers the docstring quotes and the two cannot
# drift apart. Ordered best-first, which is also worst-first for net-of-cost: every row is
# negative once the round trip is paid.
MTF_CELLS = (
    ("PULLBACK vs HTF", "LONG", 1_130, 3.118, 1.99, 69),
    ("WITH-TREND CONTINUATION", "LONG", 4_968, 1.532, 1.58, 69),
    ("RANGE-TOP BREAK", "LONG", 5_842, 0.522, 0.74, 50),
    ("WITH-TREND CONTINUATION", "SHORT", 4_096, 0.351, 0.32, 53),
    ("EXTENDED (aligned)", "LONG", 14_743, -0.038, -0.07, 59),
    ("COIL AT THE EXTREME", "LONG", 30_739, -0.092, -0.34, 50),
    ("PULLBACK vs HTF", "SHORT", 1_204, -0.133, -0.11, 53),
    ("COIL AT THE EXTREME", "SHORT", 33_297, -0.399, -1.94, 47),
    ("EXTENDED (aligned)", "SHORT", 13_645, -1.344, -2.78, 28),
    ("RANGE-FLOOR BREAK", "SHORT", 5_414, -1.670, -4.12, 31),
)

_SESSION_END = pd.Timedelta("15h30min")
_STUB_FRAC = 0.5
_LB = config.STRUCT_LOOKBACK          # 20 bars on both frames -- what the replay measured
_RV_BARS = 10                         # trailing window for the range/volume energy reads
_FETCH_DAYS = 5                       # 5 x 375 = 1,875 one-minute bars: ample for 20-bar
                                      # windows on every frame up to 10m, and a small payload.

_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def clear_cache() -> None:
    """Drop every memo this module holds. Wired to the dashboard's cache-clear controls."""
    with _CACHE_LOCK:
        _CACHE.clear()
    scalp_universe.cache_clear()


def _bucket1(now: dt.datetime | None = None) -> str:
    """The current ONE-minute bar bucket. The 5-minute bucket the rest of the board caches on
    is useless here -- it would pin a 1-minute trigger frame for five of its own bars."""
    n = now or dt.datetime.now()
    return f"{n:%Y-%m-%d %H:%M}"


# ── COST ─────────────────────────────────────────────────────────────────────────────
def cost_parts(value: float = DEFAULT_POSITION, price: float = 1000.0,
               spread_ticks: float = DEFAULT_SPREAD_TICKS) -> dict:
    """Every line of the intraday round trip separately, in bps of turnover.

    Shown on the page rather than summarised, so the number is auditable instead of asserted
    -- and so it is obvious WHICH line the position-size slider is actually moving."""
    def _num(x, fallback, floor):
        # NaN CANNOT BE CLAMPED BY max(). Every comparison against NaN is False, so
        # max(nan, 1.0) returns nan and the whole cost model comes back nan -- which renders
        # as a dash in `cost` and, worse, as a dash in `pays?`, the one column the page tells
        # you to treat as final. A silent dash on the veto column is the most expensive
        # failure this page has, so junk falls back to the default rather than propagating.
        try:
            v = float(x)
        except (TypeError, ValueError):
            return fallback
        if v != v or v <= floor:            # NaN, inf-, or non-positive
            return fallback
        return v

    value = _num(value, DEFAULT_POSITION, 0.0)
    price = _num(price, 1000.0, 0.0)
    try:
        spread_ticks = float(spread_ticks)
        if spread_ticks != spread_ticks or spread_ticks < 0:
            spread_ticks = DEFAULT_SPREAD_TICKS
    except (TypeError, ValueError):
        spread_ticks = DEFAULT_SPREAD_TICKS
    brok = 2 * min(BROK_FLAT, BROK_PCT * value)
    txn = 2 * NSE_TXN * value
    sebi = 2 * SEBI_FEE * value
    return {
        "brokerage": 1e4 * brok / value,
        "exchange": 1e4 * txn / value,
        "sebi": 1e4 * sebi / value,
        "gst": 1e4 * GST_RATE * (brok + txn + sebi) / value,
        "stt": 1e4 * STT_SELL,
        "stamp": 1e4 * STAMP_BUY,
        "spread": 1e4 * spread_ticks * TICK / price,
    }


def round_trip_bps(value: float = DEFAULT_POSITION, price: float = 1000.0,
                   spread_ticks: float = DEFAULT_SPREAD_TICKS) -> float:
    """Intraday square-off round trip, in bps of turnover. See cost_parts for the split."""
    return float(sum(cost_parts(value, price, spread_ticks).values()))


def p_pays(exp_range_bps: float, cost: float) -> float:
    """P(|move over the next 5 minutes| > cost), in percent, interpolated on the measured
    ratio table.

    NOT P(profit). The move must also go YOUR WAY, and this board has no view on that --
    direction at this horizon is 46.2% up over 457k observations, i.e. a coin flip. This is
    the NECESSARY condition (can the name travel far enough to pay for the trip at all) and
    never the sufficient one."""
    try:
        r, c = float(exp_range_bps), float(cost)
    except (TypeError, ValueError):
        return float("nan")
    if not (r > 0) or not (c > 0) or r != r or c != c:
        return float("nan")
    x = c / r
    if x <= _RATIO_X[0]:
        return float(_RATIO_P[0])
    if x >= _RATIO_X[-1]:
        return float(_RATIO_P[-1])
    return float(np.interp(x, _RATIO_X, _RATIO_P))


def exp_move_bps(rv10_bps: float) -> float:
    """Expected |5-minute move| in bps from the trailing 10-bar mean 1-minute range."""
    try:
        v = float(rv10_bps)
    except (TypeError, ValueError):
        return float("nan")
    if v != v or v < 0:
        return float("nan")
    return EXP5_A + EXP5_B * v


# ── WHICH NAMES TO SCAN ──────────────────────────────────────────────────────────────
MIN_TURN_CR_DEFAULT = 50.0        # Rs crore of median daily turnover -- a FILLABILITY floor


@lru_cache(maxsize=16)
def scalp_universe(n: int = 30, min_turn_cr: float = MIN_TURN_CR_DEFAULT) -> tuple:
    """The n most SCALPABLE names, ranked by daily ATR% from the EOD archive. Zero API cost.

    ATR% RANKS (Spearman +0.902 against realised mean |5-minute move|); turnover only GATES.
    Sorting by liquidity instead picks the LEAST scalpable names in the market (turnover ranks
    -0.10), which is why the obvious default was rejected. Returns a tuple, so the lru_cache
    key stays hashable and the result cannot be mutated by a caller."""
    try:
        u = live.liquid_universe(None)
    except Exception:                                        # noqa: BLE001
        return ()
    if u is None or u.empty:
        return ()
    u = u.copy()
    ref = pd.to_numeric(u.get("ref_close"), errors="coerce")
    u["atr_pct"] = 100 * pd.to_numeric(u.get("atr14"), errors="coerce") / ref
    u["turn_cr"] = pd.to_numeric(u.get("vol_med20"), errors="coerce") * ref / 1e7
    u = u.dropna(subset=["atr_pct", "turn_cr"])
    if u.empty:
        return ()
    n = max(int(n), 1)
    gated = u[u["turn_cr"] >= float(min_turn_cr)]
    # NEVER RETURN AN EMPTY BOARD BECAUSE THE FLOOR WAS SET TOO HIGH. A floor that filters
    # everything is a user error, not a reading of the market, and an empty board is
    # indistinguishable from "nothing is scalpable today" -- a completely different statement.
    if gated.empty:
        gated = u.nlargest(min(len(u), n), "turn_cr")
    return tuple(gated.nlargest(n, "atr_pct")["symbol"])


def scan_seconds(n: int) -> float:
    """Wall-clock estimate for a scan of n names, from the MEASURED rate pacer.

    This is the NETWORK FLOOR and deliberately nothing else: it is the one term with a hard
    external ceiling (~180 requests/minute), and it is what the "cannot keep up with a
    1-minute bar" warning has to key off. The per-name run walk adds CPU on top -- measured
    7.8s actual against a 6.6s estimate at 20 names, so roughly +20% -- which is why the
    warning threshold sits at 55s rather than 60."""
    return round(max(int(n), 0) * live._HIST_GAP, 1)


# ── FRAMES ───────────────────────────────────────────────────────────────────────────
def _resample(f: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample the 1-minute series to `minutes`, aligned to the 09:15 open, folding a
    trailing SESSION STUB into the bar before it.

    live.merge_session_stubs is deliberately NOT reused. It early-returns on any frame of 15
    minutes or less, which is right for every frame that existed when it was written (1, 5 and
    15 all divide the 375-minute session exactly, so no stub is possible) and wrong for the two
    this lane adds: 375/7 = 53.6 and 375/10 = 37.5, so a 7m board ends every day on a FOUR
    minute bar and a 10m board on a FIVE minute one. The structure classifier compares bar
    ranges against each other, so a half-width bar wearing a full-width label is the same
    defect that was manufacturing coils on the 1h and 2h frames (measured there at 34% and 37%
    of names changing label). Folding is also the honest chartist read: the stub is the tail of
    the previous bar, not a bar of its own.

    Written as a LOCAL rule rather than a widened guard, so a validated function keeps its
    measured behaviour on the frames it was actually measured on."""
    if f is None or f.empty or minutes <= 1:
        return f
    r = (f.set_index("ts")
         .groupby(pd.Grouper(freq=f"{minutes}min", origin="start_day", offset="9h15min"))
         .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
              close=("close", "last"), volume=("volume", "sum"))
         .dropna(subset=["close"]).reset_index())
    if r.empty or 375 % minutes == 0:
        return r
    rows: list[dict] = []
    for _, row in r.iterrows():
        d = row.to_dict()
        ts = d["ts"]
        avail = min(ts + pd.Timedelta(minutes=minutes), ts.normalize() + _SESSION_END) - ts
        if rows and avail < pd.Timedelta(minutes=_STUB_FRAC * minutes):
            p = rows[-1]
            p["high"] = max(p["high"], d["high"])
            p["low"] = min(p["low"], d["low"])
            p["close"] = d["close"]
            p["volume"] = p["volume"] + d["volume"]
        else:
            rows.append(d)
    return pd.DataFrame(rows)


def fetch_1m(sym: str, days: int = _FETCH_DAYS) -> pd.DataFrame:
    """One name's 1-minute candles, rate-paced, DE-DUPLICATED, corporate-action adjusted.

    The de-dup is not defensive tidying. The broker returns byte-identical duplicate candles
    for some ranges -- verified on RELIANCE, where a same-day range came back 50 rows / 25
    unique at 15m and 750 / 375 at 1m, with 24 of 50 rows exact duplicates. A doubled series
    leaves the range BOX intact (min and max do not care), so nothing looks wrong on the
    chart, while every volume figure doubles and a 20-bar structure window silently covers
    only TEN real bars."""
    f = live.fetch_intraday(sym, tf="1m", lookback_days=days)
    if f is None or f.empty:
        return pd.DataFrame()
    f = f.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    try:
        f = indicators.adjust_corporate_actions(f)
    except Exception:                                        # noqa: BLE001
        pass
    return f


# ── PER-NAME SCAN ────────────────────────────────────────────────────────────────────
def _closed(frame: pd.DataFrame, minutes: int, now: dt.datetime | None = None) -> pd.DataFrame:
    """Drop a bar that has not finished printing yet.

    A bar stamped T on an m-minute frame covers [T, T+m) and is only KNOWN at T+m. The
    resampler happily emits the current, part-formed bar -- its high, low and close describe
    however much of the bar has happened so far. On the TRIGGER frame that is exactly what a
    scalper wants (it is the live tape). On the CONFIRM frame it is not: the confirm frame is
    supposed to be the settled context the trigger fires inside, and a part-formed 5-minute bar
    is neither settled nor comparable to the nineteen full bars beside it in the window.

    This is also the defect that made the first version of the offline replay print
    WITH-TREND CONTINUATION at +16.6bps with a t-stat of 33 and 100% of sessions positive:
    indexing each 1-minute bar to the 5-minute bar CONTAINING it handed the structure window up
    to four minutes of future against a five-minute forward return. Same mistake, one live and
    one offline, so the guard belongs in the shipped path and not only in the study."""
    if frame is None or frame.empty:
        return frame
    n = now or dt.datetime.now()
    cutoff = pd.Timestamp(n) - pd.Timedelta(minutes=minutes)
    out = frame[frame["ts"] <= cutoff]
    # If NOTHING has closed yet (the first minutes of the session, or an off-hours render
    # where `now` sits far past the last bar's session), fall back to dropping just the final
    # bar. Returning empty would render the name as unreadable rather than as early.
    return out if len(out) else frame.iloc[:-1]


def _bps_range(f: pd.DataFrame) -> pd.Series:
    return 1e4 * (f["high"] - f["low"]) / f["close"]


# ── WHEN DID THIS BECOME THE SIDE IT IS NOW? ─────────────────────────────────────────
# MEASURED FIRST, because the obvious implementation is wrong at this resolution.
#
# Every other board in this project answers `entered` as "the first time today". That is
# right for a footprint that fires once a session. It is badly wrong here: replayed over
# 475,840 causal evaluations, a scalp side lasts a MEDIAN OF TWO MINUTES (p75 6, p90 20) and
# flips a median of TWENTY-ONE times per name per session. "First LONG today" and "the LONG
# you are looking at" are different on 86% of sided bars, and when they differ the median gap
# is NINETY-NINE MINUTES. A column reading 09:22 for a flip that happened forty seconds ago
# is not a rounding error, it is the opposite of the answer.
#
# So this returns the start of the CURRENT UNBROKEN RUN -- "this has been LONG continuously
# since 14:22" -- which is the only version that answers a scalper's actual question: is this
# a fresh flip, or one I have already missed?
#
# ⚠ FRESH IS NOT A BUY SIGNAL, and the tooltip says so. Bucketing the same panel by run age,
# the flip bar ITSELF is the worst cell on the board (-0.488bps, clustered t -1.97) and no
# bucket clears zero convincingly (age 3-5min +0.331 t +0.99; age 60+ +0.469 t +1.13). The
# column is a STALENESS read -- how much of the move you have already missed -- not an entry
# trigger.
#
# COST: the walk stops at the first bar that disagrees, and runs are short -- median 10 bars
# back, p99 113, hard-capped at today's open. The 5-minute structure is only recomputed when
# the 5-minute index actually moves, so it is ~1.2 struct_full calls per bar walked.
_MAX_WALK = 240             # bars; the session is 375, so this is a guard not a policy


def _side_at(f, h5, i: int, lb: int = None):
    """The board's LONG/SHORT/— verdict as it stood at the close of 1-minute bar `i`.

    CAUSAL BY CONSTRUCTION. A 5-minute bar stamped T covers [T, T+5) and is only known at
    T+5; acting at the close of 1-minute bar i means acting at ts_i + 1min, so the newest
    fully-known 5-minute bar is the last one with T <= ts_i - 4min. Indexing to the 5-minute
    bar CONTAINING ts_i instead -- the obvious form -- is exactly the lookahead that made the
    offline study print +16.6bps at t=33. Same rule as _closed(), applied at a past instant."""
    lb = lb or _LB
    if i < 4:
        return None, None
    lw = f.iloc[max(0, i - lb + 1):i + 1]
    if len(lw) < 5:
        return None, None
    cut = f["ts"].iat[i] - pd.Timedelta(minutes=4)
    hpos = int(np.searchsorted(h5["ts"].to_numpy(), np.datetime64(cut), side="right")) - 1
    if hpos < 4:
        return None, None
    hw = h5.iloc[max(0, hpos - lb + 1):hpos + 1]
    if len(hw) < 5:
        return None, None
    sl = indicators.struct_full(lw)          # a PAST bar has closed: forming=False
    sh = indicators.struct_full(hw)
    px = float(lw["close"].iloc[-1])
    syn = mtf.synthesize(
        {"struct": sh["struct"], "hi": sh.get("hi"), "lo": sh.get("lo"),
         "n": int(sh.get("n", 0))},
        {"struct": sl["struct"], "hi": sl.get("hi"), "lo": sl.get("lo"),
         "n": int(sl.get("n", 0))}, px)
    return mtf.side_of(syn["tag"], syn.get("dir", "NONE")), px


def entered_run(f, h5, sess_start_i: int, side: str, ltp: float) -> tuple:
    """(entered, at, since%) for the CURRENT run of `side`. See the note above.

    The newest bar's side is taken as GIVEN rather than recomputed: _scan_one evaluates it
    with forming=True (it is the live tape) and this walk uses forming=False (those bars have
    closed). Recomputing bar 0 under the other convention could disagree with the side the row
    is actually displaying, and then `entered` would time a flip that never happened.

    `entered` carries a leading '>=' when the run reaches the first bar we can evaluate today
    -- the flip then predates our window and the honest answer is a bound, not a time."""
    if side not in ("LONG", "SHORT") or f is None or len(f) < 6:
        return None, np.nan, np.nan
    last = len(f) - 1
    start = last                                    # bar 0 is, by definition, in the run
    floor = max(sess_start_i, last - _MAX_WALK, 4)
    bounded = False
    for i in range(last - 1, floor - 1, -1):
        sd, _ = _side_at(f, h5, i)
        if sd is None:                              # ran out of readable window
            bounded = True
            break
        if sd != side:
            break
        start = i
    else:
        bounded = start <= floor
    ts = f["ts"].iat[start]
    px = float(f["close"].iat[start])
    since = (100 * (ltp / px - 1)) if px > 0 else np.nan
    if side == "SHORT" and since == since:
        since = -since                              # a short profits when price FALLS
    # +0.0 COLLAPSES NEGATIVE ZERO. round(-0.0001, 2) is -0.0, which renders as "-0.00" and
    # reads as a losing trade on a name that has not moved at all -- and on a one-minute-old
    # run that is the MOST common case on this board, not an edge case.
    return (("≥" if bounded else "") + f"{ts:%H:%M}"), round(px, 2), round(since, 2) + 0.0


def _scan_one(sym: str, up_min: int, position: float, spread_ticks: float,
              now: dt.datetime | None = None) -> dict | None:
    """One name's scalp read. NEVER RAISES -- a thrown exception inside the pool would drop
    the name silently, which on a 25-name board is 4% of the universe vanishing unremarked."""
    try:
        f = fetch_1m(sym)
        if f is None or len(f) < _LB + 2:
            return None
        today = f["ts"].dt.date.max()
        sess = f[f["ts"].dt.date == today]
        if sess.empty:
            return None

        # TRIGGER frame: 1-minute, INCLUDING the bar still printing. That live bar is the
        # scalper's tape and struct_full's `forming` flag already handles the one test (the
        # coil) that a part-printed bar would corrupt.
        ltf_lab = indicators.struct_full(f.tail(_LB), forming=True)
        # CONFIRM + CONTEXT frames: closed bars only. See _closed.
        h5 = _closed(_resample(f, 5), 5, now)
        hup = _closed(_resample(f, up_min), up_min, now)
        if h5 is None or len(h5) < 5:
            return None
        htf_lab = indicators.struct_full(h5.tail(_LB))

        px = float(sess["close"].iloc[-1])
        syn = mtf.synthesize(
            {"struct": htf_lab["struct"], "hi": htf_lab.get("hi"), "lo": htf_lab.get("lo"),
             "n": int(htf_lab.get("n", 0))},
            {"struct": ltf_lab["struct"], "hi": ltf_lab.get("hi"), "lo": ltf_lab.get("lo"),
             "n": int(ltf_lab.get("n", 0))}, px)
        tag = syn["tag"]
        side = mtf.side_of(tag, syn.get("dir", "NONE"))

        # ── ENERGY. The trailing range is the whole predictive content of this board.
        # CLOSED bars only: the forming bar's range is partial by definition, so including it
        # drags the estimate down by up to a full bar's worth right when the tape is moving.
        closed1 = _closed(sess, 1, now)
        base = closed1 if len(closed1) >= _RV_BARS else sess
        rv10 = float(_bps_range(base.tail(_RV_BARS)).mean())
        vmed = float(f["volume"].median() or 0)
        rvol = (float(base["volume"].tail(_RV_BARS).mean()) / vmed) if vmed > 0 else np.nan

        exp5 = exp_move_bps(rv10)
        cost = round_trip_bps(position, px, spread_ticks)
        pays = p_pays(exp5, cost)

        # ── GEOMETRY. ON THE CONFIRM FRAME, NOT THE TRIGGER FRAME.
        # The first version sized stops from the 1-MINUTE ATR, which on a liquid name is
        # roughly three basis points -- a stop INSIDE the spread, guaranteed to be taken out
        # by noise before the trade has an opinion. The unit has to match the HOLD, and the
        # hold is one to five minutes, so the 5-minute box is the right frame. It also fixed
        # `room`, which was reporting 33 and 58 "ATR" of headroom -- arithmetically true and
        # completely unreadable, because it was counting one-minute ATRs.
        atr5 = float(indicators.atr(h5.tail(60), 14) or 0.0)
        sr = {}
        try:
            if hup is not None and len(hup) >= 5:
                sr = indicators.sr_levels(hup, spot=px) or {}
        except Exception:                                    # noqa: BLE001
            sr = {}
        sup, res = sr.get("support"), sr.get("resistance")
        head_up, head_dn = sr.get("head_up"), sr.get("head_dn")
        head = head_up if side == "LONG" else (head_dn if side == "SHORT" else None)
        # ROOM IN THE UNIT THE TRADE IS DENOMINATED IN: how many EXPECTED FIVE-MINUTE MOVES
        # of clear road there are before the one-frame-up wall. "2.4 moves" answers the
        # scalper's actual question (can this trip finish before it hits something?) in a way
        # that neither rupees nor a foreign frame's ATR does. inf = no multi-touch wall that
        # way, which is CLEAR ROAD -- the opposite of unknown.
        head_bps = (1e4 * float(head) / px) if (head is not None and px > 0) else None
        room = (head_bps / exp5) if (head_bps is not None and exp5 == exp5 and exp5 > 0) \
            else np.inf

        # WHEN this became the side it is now (start of the current unbroken run).
        _sess_i = int(f.index[f["ts"].dt.date == today][0]) if len(sess) else 0
        entered, ent_px, since = entered_run(f, h5, _sess_i, side, px)

        vw = indicators.vwap(sess)
        prev = f[f["ts"].dt.date < today]
        pc = float(prev["close"].iloc[-1]) if len(prev) else float(sess["open"].iloc[0])
        turn_l = float((sess["volume"] * sess["close"]).sum() / 1e5)

        return {
            "symbol": sym, "setup": tag, "dir": syn.get("dir", "NONE"), "side": side,
            "setup_rank": mtf.TAG_RANK.get(tag, 99), "setup_read": syn.get("read", ""),
            "loc": syn.get("loc"), "ltp": round(px, 2),
            "day%": round(100 * (px / pc - 1), 2) if pc > 0 else np.nan,
            "entered": entered, "at": ent_px, "since%": since,
            "exp5m": round(exp5, 1), "cost": round(cost, 1),
            "p pays": round(pays, 0) if pays == pays else np.nan,
            "pays": _pays_tag(pays),
            "rvol": round(rvol, 2) if rvol == rvol else np.nan,
            "spread": round(1e4 * TICK / px, 2) if px > 0 else np.nan,
            "1m": ltf_lab["struct"], "5m": htf_lab["struct"],
            "vs_vwap%": round(100 * (px / vw - 1), 2) if vw else np.nan,
            "sup": round(float(sup), 2) if sup else np.nan,
            "res": round(float(res), 2) if res else np.nan,
            "room": round(room, 2) if room == room and room != np.inf else np.inf,
            "stop": round(px - atr5, 2) if atr5 > 0 else np.nan,
            "t1": round(px + atr5, 2) if atr5 > 0 else np.nan,
            "s_stop": round(px + atr5, 2) if atr5 > 0 else np.nan,
            "s_t1": round(px - atr5, 2) if atr5 > 0 else np.nan,
            "atr₹": round(atr5, 2) if atr5 > 0 else np.nan,
            "turn₹L": round(turn_l, 1),
            "bars": int(len(sess)),
        }
    except Exception:                                        # noqa: BLE001
        return None


# ── THE VERDICT ──────────────────────────────────────────────────────────────────────
# Thresholds on P(|5-min move| > cost), the ONE calibrated number on the board. They are cut
# points on a continuum, not discovered classes -- the column shows the raw percentage beside
# the icon precisely so the icon cannot become the whole read.
PAYS_OK, PAYS_MARGINAL = 60.0, 45.0


def _pays_tag(p: float) -> str:
    if p != p:
        return "—"
    if p >= PAYS_OK:
        return f"✅ {p:.0f}%"
    if p >= PAYS_MARGINAL:
        return f"⚠ {p:.0f}%"
    return f"⛔ {p:.0f}%"


# ── THE SCAN ─────────────────────────────────────────────────────────────────────────
# `entered` / `at` / `since%` sit IMMEDIATELY AFTER day%, by request and because that is the
# reading order the pair belongs to: the day's move, then when this side actually started and
# how much of it has already happened. Pushed right, a staleness column is the same as absent.
COLS = ["setup", "ltp", "day%", "entered", "at", "since%", "exp5m", "cost", "pays", "rvol",
        "spread", "1m", "5m", "vs_vwap%", "sup", "res", "room", "atr₹", "turn₹L", "side"]

HELP = {
    "setup": "The 1m x 5m chartist read: the 5-minute frame is the BOX, the 1-minute frame "
             "is the TRIGGER, and `room` says where price sits relative to the frame above. "
             "RANKS SETUP QUALITY, IT DOES NOT PREDICT RETURNS -- direction over the next five "
             "minutes measured 46.2% up across 457k observations, i.e. a coin flip. "
             "Replayed causally over 475,840 "
             "evaluations, NOT ONE tag/side cell clears the cost floor: the best is "
             "PULLBACK vs HTF LONG at +3.1bps gross against a 6.9bps round trip, and "
             "LONG minus SHORT is -0.04bps (t -0.33). Use it to place a stop, never to "
             "pick a name.",
    "entered": "When this name became the side it is showing now -- the start of its CURRENT "
               "UNBROKEN run, not the first time it took this side today.\n\n"
               "That distinction is the whole column. Measured over 475,840 causal "
               "evaluations, a scalp side lasts a **median of two minutes** and flips a median "
               "of **21 times per session**; 'first today' and 'the one you are looking at' "
               "differ on 86% of rows, by a median of 99 minutes. A '≥' prefix means the run "
               "reaches the first bar readable today, so the flip predates the window.\n\n"
               "⚠ FRESH IS NOT A BUY SIGNAL. Bucketed by run age, the flip bar itself is the "
               "WORST cell on the board (-0.49bps, clustered t -1.97). Read this as staleness "
               "-- how much of the move you already missed -- not as a trigger.",
    "at": "Price at the bar this side started. The reference `since%` is measured from.",
    "since%": "Move since this side began, SIGNED THE WAY THE TRADE WOULD BE: positive means "
              "the side has been right so far, on a short as well as a long. A large positive "
              "number is a warning, not an endorsement -- the move is behind you, and the "
              "median run only lasts two minutes.",
    "exp5m": "Expected SIZE of the next five minutes' move, in basis points, ignoring "
             "direction. From the trailing 10-bar mean 1-minute range: |move| = 2.271 + "
             "0.854 x range, fitted on 460,576 observations (IC +0.399, calibration error "
             "-1% to +5% through the middle eight deciles). This is the one number on the "
             "board with real predictive content.",
    "cost": "YOUR round-trip cost in bps at the position size set above -- brokerage, STT "
            "(Securities Transaction Tax), exchange + SEBI fees, GST, stamp duty and two "
            "ticks of spread. It MOVES WITH POSITION SIZE: the Rs20-per-order brokerage is "
            "6bps of a Rs50k trade and 0.8bps of a Rs5L one. This is an INTRADAY square-off, "
            "so it is ~5-12bps and NOT the 22bps delivery figure the BTST board uses.",
    "pays": "P(the next five minutes moves further than your cost) -- the ✅/⚠/⛔ is just a "
            "cut on that percentage.\n\n"
            "**This is NOT the probability of profit.** The move also has to go YOUR WAY, and "
            "this board has no view on that. It is a NECESSARY condition: below it, even a "
            "perfectly-called direction cannot pay for the trip. ⛔ means do not trade this "
            "name right now no matter how good the chart looks.",
    "rvol": "Volume over the last 10 one-minute bars against this name's own median minute. "
            "Participation is the second-best size predictor after range (IC +0.227 vs "
            "+0.399) and they overlap, so read it as confirmation, not as an independent leg.",
    "spread": "One NSE tick (Rs0.05) as bps of this price -- a hard FLOOR under the real "
              "spread, never the spread itself. On a Rs150 name that floor is 3.3bps, which "
              "is half a typical five-minute move before anyone has quoted you anything.",
    "room": "Clear road to the one-frame-up wall in the trade's direction, measured in "
            "EXPECTED FIVE-MINUTE MOVES. 2.0 means the wall is about two of this name's "
            "typical five-minute trips away; below 1.0 the trade runs into the level before "
            "it has finished. `inf` = no multi-touch wall that way, which is clear road, not "
            "unknown.\n\nContext for placing the exit, NOT a return signal: measured on "
            "longer horizons in this stack, 'has room' did not beat 'capped' on either side.",
    "1m": "Kaufman structure on the TRIGGER frame, including the bar still printing -- so it "
          "can and should change every sixty seconds.",
    "5m": "Kaufman structure on the CONFIRM frame, CLOSED BARS ONLY. A part-formed 5-minute "
          "bar is not comparable to the nineteen full bars beside it in the window, and "
          "including it leaks up to four minutes of the future into the read.",
    "atr₹": "ATR in rupees on the 5-minute CONFIRM frame -- the unit `stop` and `t1` are "
            "built from. Deliberately not the 1-minute ATR: that is ~3bps on a liquid name, "
            "which puts the stop INSIDE the spread where noise takes it out before the trade "
            "has an opinion. Risk geometry, not a forecast.",
}


def scan(n: int = 30, up_frame: str = UP_DEFAULT, position: float = DEFAULT_POSITION,
         spread_ticks: float = DEFAULT_SPREAD_TICKS,
         min_turn_cr: float = MIN_TURN_CR_DEFAULT, nonce: int = 0) -> dict:
    """Scan the n most scalpable names on 1m x 5m. Memoised per ONE-MINUTE bucket.

    The 5-minute memo the rest of the board uses would pin a 1-minute trigger frame for five
    of its own bars, which is the whole thing this lane exists to avoid."""
    ts = live.token_status()
    if not ts.get("usable"):
        return {"ok": False, "status": ts.get("describe", "token unusable"),
                "board": pd.DataFrame(), "n_scanned": 0, "syms": ()}

    up_min = UP_FRAMES.get(up_frame, UP_FRAMES[UP_DEFAULT])
    key = (_bucket1(), int(n), up_min, round(float(position), 2),
           round(float(spread_ticks), 3), round(float(min_turn_cr), 2), int(nonce))
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if hit is not None:
        return {"ok": True, **hit}

    syms = scalp_universe(int(n), float(min_turn_cr))
    if not syms:
        return {"ok": False, "status": "universe unavailable (archive not reachable)",
                "board": pd.DataFrame(), "n_scanned": 0, "syms": ()}

    from concurrent.futures import ThreadPoolExecutor
    now = dt.datetime.now()
    with ThreadPoolExecutor(max_workers=live._SCAN_WORKERS) as ex:
        rows = list(ex.map(lambda s: _scan_one(s, up_min, position, spread_ticks, now), syms))
    board = pd.DataFrame([r for r in rows if r])
    if not board.empty:
        # THIS STREAMLIT RENDERS A MISSING VALUE AS THE LITERAL STRING "None" in an
        # object-dtype column, and as a blank in a NULLABLE numeric one. The main board hit
        # exactly this on `at` / `since%` / `dn age` and shipped a table full of the word
        # "None". `entered` is text, so it gets the board's own em dash; the two numerics get
        # nullable dtypes so pandas hands Arrow a real null.
        board["entered"] = board["entered"].fillna("—").replace({None: "—", "": "—"})
        for c in ("at", "since%"):
            board[c] = pd.to_numeric(board[c], errors="coerce").astype("Float64")
        # SORT BY WHAT THE BOARD ACTUALLY KNOWS. Sorting by setup_rank would put the prettiest
        # CHART on top, and the chart is the part with no measured forward content. Expected
        # move is the part that does, so the liveliest tradeable names lead and setup quality
        # breaks ties within them.
        board = board.sort_values(["exp5m", "setup_rank"], ascending=[False, True])

    out = {"status": ts.get("describe", ""), "board": board, "n_scanned": len(syms),
           "n_read": int(len(board)), "syms": syms, "scanned_at": now,
           "up_frame": up_frame, "up_min": up_min,
           "n_blank": int(len(syms) - len(board))}
    with _CACHE_LOCK:
        if len(_CACHE) > 6:                     # never hold more than a few one-minute buckets
            _CACHE.clear()
        _CACHE[key] = out
    return {"ok": True, **out}


# ── PAGE ─────────────────────────────────────────────────────────────────────────────
def col_config(st) -> dict:
    """Streamlit column_config for the scalp board, DERIVED from HELP so the two can never
    drift. The F&O block shipped once with a hand-written config holding fewer entries than
    the column list, and the columns it missed rendered with no tooltip at all."""
    cfg = {c: st.column_config.TextColumn(c, help=HELP[c]) for c in HELP}
    cfg["exp5m"] = st.column_config.NumberColumn("exp 5m bps", format="%.1f", help=HELP["exp5m"])
    cfg["cost"] = st.column_config.NumberColumn("cost bps", format="%.1f", help=HELP["cost"])
    cfg["rvol"] = st.column_config.NumberColumn("rvol", format="%.2f", help=HELP["rvol"])
    cfg["spread"] = st.column_config.NumberColumn("tick bps", format="%.2f", help=HELP["spread"])
    cfg["room"] = st.column_config.NumberColumn("room ATR", format="%.2f", help=HELP["room"])
    cfg["pays"] = st.column_config.TextColumn("pays?", width="small", help=HELP["pays"])
    cfg["entered"] = st.column_config.TextColumn("entered", width="small",
                                                 help=HELP["entered"])
    cfg["at"] = st.column_config.NumberColumn("at", format="%.2f", help=HELP["at"])
    cfg["since%"] = st.column_config.NumberColumn("since%", format="%.2f",
                                                  help=HELP["since%"])
    return cfg


def _actually_do(st) -> None:
    """The instruction, first, in the imperative -- before any table. arb.py learned this the
    hard way: a page of measured context that never says what to DO gets read as a signal."""
    with st.expander("❓ **SO WHAT DO I ACTUALLY DO?** — read this before the table", True):
        st.markdown(
            "**This board is a VETO, not a trigger. It never tells you which way to trade.**\n\n"
            "1. **Read `pays?` first, and treat ⛔ as final.** It is "
            "`P(the next five minutes moves further than your round trip)`. On a ⛔ name a "
            "*perfectly* called direction still loses money — the move is smaller than the "
            "bill. No chart pattern fixes that. Drop it and look at the next row.\n"
            "2. **Set your real position size above.** Cost is not a constant: the flat "
            "₹20-per-order brokerage is **6bps** of a ₹50,000 trade and **0.8bps** of a "
            "₹5,00,000 one. The same name is ⛔ at one size and ✅ at another, and that is a "
            "true statement about your trading, not a bug.\n"
            "3. **You supply the direction.** From the tape, the level, the order book — "
            "wherever you actually get it. Measured here on 457,000 observations, the next "
            "five minutes is **46.2% up** with a mean of **+0.11bps**: this board holds no "
            "directional information and will not pretend to.\n"
            "4. **Use `setup`, `sup`/`res` and `room` to place the stop**, not to pick the "
            "name. They are chartist context with no measured forward return at this horizon.\n"
            "5. **Square off.** Every number here is an intraday cost. Carrying overnight "
            "moves you onto the delivery schedule (STT 0.1% on *both* legs) and roughly "
            "triples the bill.")
        st.caption("Why it is built this way: SIZE is predictable at this horizon "
                   "(IC +0.399 on the trailing range), DIRECTION is not (IC +0.024). So the "
                   "board sells you the half it can actually measure.")


def _setup_evidence(st, cost: float) -> None:
    """What the `setup` column is actually worth, as a table, so the page can never claim more
    than the replay measured. Rendered from MTF_CELLS -- the same tuple the module docstring
    quotes, so the two cannot drift apart."""
    rows = "\n".join(
        f"| {t} | {sd} | {n:,} | {e:+.3f}bps | {tt:+.2f} | {sp}% | **{e - cost:+.2f}bps** |"
        for t, sd, n, e, tt, sp in MTF_CELLS)
    with st.expander("🔬 What the `setup` column is actually worth (all 10 cells, measured)"):
        st.markdown(
            f"Replayed causally over **{EVIDENCE['mtf_evals']:,} evaluations** — structure "
            f"rebuilt from candles truncated to each 1-minute bar close, then run through the "
            f"real `struct_full → synthesize → side_of` pipeline. `t` is **clustered by "
            f"session**, because bars inside one day are nowhere near independent and the "
            f"naive t-stat runs 3–6× too generous.\n\n"
            f"| setup | side | n | fwd 5m | t (clustered) | sessions + | "
            f"**net of your cost** |\n"
            f"|---|---|---:|---:|---:|---:|---:|\n{rows}\n\n"
            f"**Not one cell clears the {cost:.2f}bps floor.** The best of them is also the "
            f"*smallest* cell on the board (n=1,130). LONG minus SHORT is "
            f"{EVIDENCE['mtf_long_minus_short_bps']:+.3f}bps "
            f"(t {EVIDENCE['mtf_long_minus_short_t']:+.2f}) — the side split separates "
            f"nothing at all. The short side is separately and significantly negative "
            f"(clustered t {EVIDENCE['mtf_short_t_clustered']:+.2f}, 19% of sessions "
            f"positive), which independently reproduces `mtf.SHORT_EDGE_BPS`.\n\n"
            f"⚠ The first version of this table was a **lookahead**, and it looked superb — "
            f"WITH-TREND CONTINUATION at +16.6bps, t=33, 100% of sessions positive, every "
            f"cell an order of magnitude larger. Each 1-minute bar had been indexed to the "
            f"5-minute bar *containing* it rather than the last one to have **closed**, which "
            f"handed the confirm frame up to four minutes of the future against a five-minute "
            f"forward return. The shipped scan carries the same guard (`_closed`), so the live "
            f"board cannot repeat it.")


def render_page(st, key_prefix: str = "scalp") -> None:
    """Render the whole scalper lane. `st` is passed IN, never imported, so the page is
    testable headless (tests/test_scalp.py drives it with a stub)."""
    st.subheader("⚡ Scalper mode — 1m trigger / 5m confirm")
    st.caption(
        f"Hold **{PRESETS['scalp']['hold']}**. The delivery, F&O positioning, carry and "
        f"sector-tilt columns are GONE here — every one of them is an end-of-day or "
        f"multi-week number that cannot change inside a five-minute trade. What replaces them "
        f"is the only pair that decides a scalp: **how far this name can travel in five "
        f"minutes**, and **what the round trip costs you**.")

    _actually_do(st)

    # ── CONTROLS ─────────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    n = c1.slider("Names to scan", 10, 242, 30, 5, key=f"{key_prefix}_n",
                  help=("One 1-minute /history fetch per name, rate-paced at 0.33s (a MEASURED "
                        "ceiling — 180 requests/minute; a burst above it returns 429s that "
                        "silently become blank rows).\n\n"
                        "**A 1-minute trigger frame wants a re-scan every 60 seconds, and the "
                        "full 242-name universe takes ~80 — so it can never keep up with its "
                        "own trigger bar.** Keep this under ~150 if you want the board fresher "
                        "than the frame you are trading.\n\n"
                        "Names are ranked by daily ATR% from the archive, NOT by turnover: "
                        "measured across 40 names, ATR% ranks realised 5-minute movement at "
                        "Spearman +0.902 while turnover ranks it at −0.10. The megacaps are "
                        "the least scalpable names in the market."))
    _secs = scan_seconds(n)
    c1.caption(f"≈ **{_secs:.0f}s** per scan · "
               + ("✅ fits inside a 1-minute bar" if _secs <= 55 else
                  f"⚠ **longer than the 1-minute bar** — the board will lag the frame you are "
                  f"trading by {_secs - 60:.0f}s or more"))

    pos = c2.number_input("Position size (₹)", 10_000, 10_000_000, DEFAULT_POSITION, 10_000,
                          key=f"{key_prefix}_pos",
                          help=("The single biggest lever on this page. Brokerage is ₹20 per "
                                "executed order, so it is a FIXED cost being amortised: 6bps "
                                "of ₹50k, 2bps of ₹2L, 0.8bps of ₹5L. Every `cost` and `pays?` "
                                "cell moves with this number."))
    spread = c3.number_input("Spread (ticks, round trip)", 0.0, 20.0, DEFAULT_SPREAD_TICKS,
                             0.5, key=f"{key_prefix}_spread",
                             help=("How many ₹0.05 ticks you expect to give up crossing the "
                                   "book, both legs together. **2.0 is optimistic** and "
                                   "assumes a genuinely liquid name; on anything thin, raise "
                                   "it. This is the one input the board cannot measure for "
                                   "you — Fyers /history serves no bid/ask."))
    up = c4.selectbox("Upper-TF S/R (one frame up)", list(UP_FRAMES),
                      index=list(UP_FRAMES).index(UP_DEFAULT), key=f"{key_prefix}_up",
                      help=("The context frame the `sup`/`res`/`room` columns are drawn from. "
                            "It never enters the LONG/SHORT decision.\n\n"
                            "⚠ Neither 7m nor 10m is a native broker resolution, so both are "
                            "resampled from the 1-minute series — and 375 minutes divides by "
                            "neither (375/7 = 53.6, 375/10 = 37.5). Both therefore end the "
                            "session on a stub bar (4 and 5 minutes), which is folded into "
                            "the bar before it; an unfolded half-width bar reads as a fake "
                            "coil."))

    st.session_state.setdefault(f"{key_prefix}_nonce", 0)
    r1, r2 = st.columns([1, 5])
    if r1.button("↻ re-scan", key=f"{key_prefix}_rescan",
                 help="Force a fresh pull. The scan is otherwise pinned to the current "
                      "one-minute bucket, so moving a slider re-prices instantly without "
                      "re-fetching a single candle."):
        st.session_state[f"{key_prefix}_nonce"] += 1
        clear_cache()

    # ── COST, AUDITABLE ──────────────────────────────────────────────────────────────
    parts = cost_parts(pos, 1000.0, spread)
    total = sum(parts.values())
    r2.caption(f"💸 round trip at ₹{pos:,.0f} on a ₹1,000 name = **{total:.2f}bps** · "
               + " · ".join(f"{k} {v:.2f}" for k, v in parts.items()))
    with st.expander("💸 Where the cost number comes from (and why it is not 22bps)"):
        _keys = ("brokerage", "exchange", "sebi", "gst", "stt", "stamp", "spread")
        _rows = []
        for v in (50_000, 100_000, 200_000, 500_000, 1_000_000):
            p = cost_parts(v, 1000.0, spread)
            _rows.append("| ₹{:,} | **{:.2f}** | ".format(v, sum(p.values()))
                         + " | ".join(f"{p[k]:.2f}" for k in _keys) + " |")
        st.markdown(
            "The **22bps** figure everywhere else in this project (`config.COST_BPS`) is a "
            "**delivery** round trip, and it is ~22 almost entirely because delivery STT is "
            "**0.1% on both legs = 20bps on its own**. An intraday square-off does not pay "
            "that — STT is **0.025% on the sell leg only**.\n\n"
            "| position | total | " + " | ".join(_keys) + " |\n"
            + "|---" * (len(_keys) + 2) + "|\n"
            + "\n".join(_rows)
            + "\n\n⚠ The `spread` column above assumes a ₹1,000 name. The board's own `cost` "
              "column uses each name's real price, so a ₹150 stock carries a far larger tick "
              "cost — one tick is 3.3bps there against 0.5bps here.")

    _setup_evidence(st, total)

    # ── SCAN ─────────────────────────────────────────────────────────────────────────
    with st.spinner(f"Scanning {n} names on 1-minute bars (~{_secs:.0f}s)…"):
        sc = scan(n=n, up_frame=up, position=pos, spread_ticks=spread,
                  nonce=st.session_state[f"{key_prefix}_nonce"])
    if not sc.get("ok"):
        st.error(f"🔑 Scalp scan unavailable — `{sc.get('status')}`. The 1-minute feed needs a "
                 f"live Fyers token; the BTST (overnight) tab runs off the archive and does "
                 f"not.")
        return
    board = sc["board"]
    if board is None or board.empty:
        st.info("Nothing came back readable. On a 1-minute frame that usually means the "
                "session has only just opened and no name has 20 closed bars yet.")
        return

    _sa = sc.get("scanned_at")
    _age = (dt.datetime.now() - _sa).total_seconds() if _sa else 0
    st.caption(f"⏱ scanned **{sc['n_read']} of {sc['n_scanned']}** names at "
               f"{_sa:%H:%M:%S} (**{_age:.0f}s ago**) · S/R from the **{sc['up_frame']}** frame"
               + (f" · {sc['n_blank']} unreadable" if sc.get("n_blank") else ""))
    if _age > 90:
        st.warning(f"⚠ This board is **{_age:.0f} seconds old** and you are trading a 1-minute "
                   f"bar. Hit ↻ re-scan — the setup tags have had {_age / 60:.0f}+ chances to "
                   f"change since this was pulled.")

    _pp = pd.to_numeric(board.get("p pays"), errors="coerce")
    tradeable = int((_pp >= PAYS_OK).sum())
    vetoed = int((_pp < PAYS_MARGINAL).sum())
    st.info(f"**{tradeable} of {sc['n_read']} names can currently pay for their own round "
            f"trip** at ₹{pos:,.0f} (`pays?` ✅ ≥{PAYS_OK:.0f}%) · **{vetoed} are vetoed "
            f"outright** (⛔ <{PAYS_MARGINAL:.0f}%). Remember what that percentage is: P(the "
            f"move is BIG enough), not P(it goes your way). Direction here measured **46.2% "
            f"up** over 457k observations.")

    cfg = col_config(st)
    _long = COLS + ["stop", "t1"]
    _short = COLS + ["s_stop", "s_t1"]

    def _tab(df_, cols, empty_note):
        if df_.empty:
            st.caption(empty_note)
            return
        show = ["symbol"] + [c for c in cols if c in df_.columns]
        st.dataframe(df_[show], use_container_width=True, hide_index=True, column_config=cfg)

    _side = board.get("side")
    tl, tsh, tn = st.tabs([f"🟢 LONG ({int((_side == 'LONG').sum())})",
                           f"🔴 SHORT ({int((_side == 'SHORT').sum())})",
                           f"⚪ No side ({int((_side == '—').sum())})"])
    with tl:
        _tab(board[board["side"] == "LONG"], _long,
             "No LONG-side setup right now. A reading of the tape, not an error.")
    with tsh:
        st.warning("⚠ **Square off before the close.** Overnight short is proven -EV in this "
                   "stack (win 20%), and an intraday short pays the SAME cost as a long with "
                   "the same coin-flip direction — there is no short edge here either.")
        _tab(board[board["side"] == "SHORT"], _short,
             "No SHORT-side setup right now. A reading of the tape, not an error.")
    with tn:
        st.caption("Setups with no side: squeezes, traps and drift. Shown because `exp5m` is "
                   "still meaningful — a name can be worth watching for movement before it "
                   "has a shape.")
        _tab(board[board["side"] == "—"], _long, "Nothing sideless right now.")

    st.caption(
        f"📏 Measured on {EVIDENCE['bars']:,} one-minute bars · {EVIDENCE['names']} names · "
        f"{EVIDENCE['sessions']} sessions ({EVIDENCE['span']}). That window is not a choice: "
        f"{EVIDENCE['cap']}, so ~32 sessions is the entire history available at this "
        f"resolution. Everything on this page is therefore calibrated on ONE market regime, "
        f"and should be re-measured once the broker has served enough new days to make a "
        f"second one.")
