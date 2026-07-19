"""Multi-timeframe HTF x LTF synthesis — the classical chartist read, ported from the
TradeBoard scout (Tradebot/tradeboard.py :: synthesize) and adapted to cash equity.

THE IDEA (why two timeframes, not one)
--------------------------------------
One frame gives you a label; it cannot tell you whether that label MEANS anything. A 15m
breakout is a completely different animal depending on where it happens inside the 1h box:

  * at the TOP of the higher-TF range  -> the only place a break can resolve the range
  * in the MIDDLE of it                -> statistically a false break; the trap that pays
                                          the people on the other side

So the higher timeframe supplies the BOX (and the trend), the lower timeframe supplies the
TRIGGER, and `loc` (where price sits in the HTF box, 0 = at the low, 1 = at the high) is the
variable that decides which of the two you are actually looking at. That is the whole read.

HORIZON PRESETS
---------------
The LTF/HTF pair is not a preference, it is the HOLD PERIOD. A pair whose HTF bar takes a
week to close cannot inform a trade you exit at 15:20. Each preset nests the trigger frame
inside a confirmation frame roughly 4x coarser -- the classical ratio (fine enough to time,
coarse enough to mean something).

HONESTY -- READ THIS
--------------------
This is CONTEXT, not a signal. Nothing here is a validated edge:
  * intraday MTF alignment was measured in the sister project and produced no edge
    (60m price-action study: impulse +0.9bps = the cost floor; the band-fade "edge" died to
    an honest-fill audit).
  * the one validated thing in this whole stack is the overnight BTST carry, and it comes
    from the delivery/close-strength footprint, NOT from structure.
  * the Daily x Weekly breakout-from-tight-base result (sister project) is the closest thing
    to support here, and it is universe-dependent and bull-concentrated.
The tags rank SETUP QUALITY as a chartist would read it. They do not predict returns. Longer
presets sit on firmer ground than shorter ones purely because the cost floor eats less of a
multi-day move than of a 30-minute one.
"""
from __future__ import annotations

# LTF = trigger/entry frame, HTF = confirmation frame. hold = intended horizon.
PRESETS: dict[str, dict] = {
    "intraday": {
        "label": "Intraday  ·  15m trigger / 1h confirm",
        "ltf": "15m", "htf": "1h", "hold": "square off same day",
        "note": "Fastest pair here. Also the WEAKEST ground: a same-day move must clear "
                "~22bps of round-trip cost, and the intraday directional hunt in this stack "
                "is closed (no variant beat the cost floor). Read it, do not mechanise it.",
    },
    "btst": {
        "label": "BTST  ·  1h trigger / 4h confirm",
        "ltf": "1h", "htf": "4h", "hold": "buy today, sell NEXT DAY (one night)",
        "note": "Matches the ONE validated edge in this project (overnight carry) in HORIZON "
                "only. The edge itself comes from the delivery + close-strength footprint, "
                "not from these bars -- use structure to size and time, not to select. "
                "WHERE THE MONEY IS: the measured payoff is the OVERNIGHT GAP, i.e. it is "
                "already in your hands at the next open. The following day session added "
                "nothing in testing, and holding a SECOND night (day-2) was dead. So sell "
                "next day -- early in it, not late.",
    },
    "swing": {
        "label": "Swing  ·  4h trigger / 1D confirm",
        "ltf": "4h", "htf": "1D", "hold": "2-10 sessions",
        "note": "Cost is ~1/10th of the move you are hunting, so structure has room to pay. "
                "1D is a CLOSED-bar frame (cannot repaint, but does not see today).",
    },
    "positional": {
        "label": "Positional  ·  1D trigger / 1W confirm",
        "ltf": "1D", "htf": "1W", "hold": "weeks",
        "note": "The nearest neighbour to the one measured multi-TF result in this stack: "
                "Daily x Weekly breakout-from-a-tight-base (+1.86%/10d, t=3.2, sister "
                "project) -- which INVERTS when the weekly is down, so the weekly gate is "
                "the whole trade. Both bars are closed-bar: no repaint.",
    },
}
PRESET_ORDER = ["intraday", "btst", "swing", "positional"]

# ── HOW PROVISIONAL IS THE READ? (measured, not asserted) ────────────────────────────
# The trigger bar is still PRINTING, so its label can change until it closes. Measured by
# replaying a full session on 49 names: at every 15-minute checkpoint both frames were
# rebuilt from candles TRUNCATED to that instant -- exactly what a scan fired at that moment
# would have seen -- and the tag re-derived.
#
#   preset      distinct tags per name per session    midday tag != closing tag   settles at
#   Intraday    3-5 for 44 of 49 names                57%                         92% of session
#   BTST        2-3 for 40 of 49 names                53%                         79% of session
#   Swing       1 (the daily trigger bar cannot repaint intraday -- the archive has no
#   Positional   bar for today at all, so these two frames are FIXED all session)
#
# Read the Intraday row again: the label does not settle until the session is 92% over,
# which is when it stops being actionable. That is a structural reason the intraday hunt
# keeps dying at the cost floor, arrived at from a completely different direction than the
# cost studies -- you cannot act on a read that will change 3-5 times before the close, and
# every change is another 22bps round trip. It also means the 15-minute auto-rescan is
# largely re-reading an unfinished bar rather than resolving new information.
#
# The law: the faster the trigger frame, the more PROVISIONAL the tag. Slower presets are
# steadier not because they are smarter but because their bar has already closed.
REPAINT = {
    "intraday":   {"tags_per_session": "3-5", "midday_differs": 57, "settles_pct": 92},
    "btst":       {"tags_per_session": "2-3", "midday_differs": 53, "settles_pct": 79},
    "swing":      {"tags_per_session": "1", "midday_differs": 0, "settles_pct": 0},
    "positional": {"tags_per_session": "1", "midday_differs": 0, "settles_pct": 0},
}

_TREND_UP_S = {"BREAKOUT_UP", "TREND_UP"}
_TREND_DN_S = {"BREAKOUT_DOWN", "TREND_DOWN"}
_RANGE_S = {"CONSOLIDATION", "RANGE"}

G, R, A, B, N = "#22c55e", "#f87171", "#fbbf24", "#40c4ff", "#94a3b8"

# Setup-quality rank (best chartist context -> worst). Sorting the board by this puts the
# textbook setups on top and the traps at the bottom. It ranks CONTEXT, not expected return.
TAG_RANK = {
    "WITH-TREND CONTINUATION": 0,      # HTF trend + LTF coil = the textbook with-trend setup
    "RANGE-TOP BREAK": 1,              # resolution watch, at the only location it can be real
    "RANGE-FLOOR BREAK": 1,
    "PULLBACK vs HTF": 2,              # dip zone with the HTF trend
    "EXTENDED (aligned)": 3,           # aligned but late -- chase risk
    "DRIFT-IN-RANGE": 4,               # wait
    "NESTED SQUEEZE": 5,               # wait, direction unknown, but energy IS building
    "RANGE-BOUND (no setup)": 6,       # nothing happening at all
    "FALSE-BREAK TRAP": 7,             # avoid
    "HTF warming": 8, "LTF warming": 8, "n/a": 9,
}

TAG_ICON = {
    "WITH-TREND CONTINUATION": "🎯", "RANGE-TOP BREAK": "🚀", "RANGE-FLOOR BREAK": "🔻",
    "PULLBACK vs HTF": "↩️", "EXTENDED (aligned)": "⚠️", "DRIFT-IN-RANGE": "〰️",
    "NESTED SQUEEZE": "🌀", "FALSE-BREAK TRAP": "🪤", "RANGE-BOUND (no setup)": "😴",
    "HTF warming": "⏳", "LTF warming": "⏳", "n/a": "—",
}


def synthesize(htf: dict, ltf: dict, spot: float) -> dict:
    """HTF x LTF structure confluence read. Returns {tag, read, color, loc}.

    `htf`/`ltf` are indicators.struct_full() payloads (need: struct, hi, lo, n).
    `loc` = where spot sits in the HTF box: 0.0 at the low, 1.0 at the high. It is the
    decisive variable whenever the HTF is a range -- see the module docstring."""
    hs = (htf or {}).get("struct", "n/a")
    ls = (ltf or {}).get("struct", "n/a")
    if hs == "n/a" or (htf or {}).get("n", 0) < 6:
        return {"tag": "HTF warming", "color": N, "loc": None, "dir": "NONE",
                "read": "higher timeframe has too few closed bars — no multi-TF read yet."}
    if ls == "n/a":
        return {"tag": "LTF warming", "color": N, "loc": None, "dir": "NONE",
                "read": "lower timeframe has too few closed bars — no trigger frame."}

    hi, lo = htf.get("hi"), htf.get("lo")
    loc = None
    if hi and lo and hi > lo and spot:
        loc = max(0.0, min(1.0, (float(spot) - float(lo)) / (float(hi) - float(lo))))
    near_hi = loc is not None and loc >= 0.72
    near_lo = loc is not None and loc <= 0.28

    # ── HTF is a RANGE / CONSOLIDATION — location decides everything ──────────────
    if hs in _RANGE_S:
        if ls in _RANGE_S:
            # A squeeze requires an actual volatility CONTRACTION on at least one frame.
            # Lumping plain RANGE (oscillating, not tightening) in with CONSOLIDATION would
            # stamp "NESTED SQUEEZE" on every quietly sideways name — measured at 85 of 140 on
            # a live board, which makes the tag meaningless. Sideways is not coiled.
            if "CONSOLIDATION" not in (hs, ls):
                return {"tag": "RANGE-BOUND (no setup)", "color": N, "loc": loc, "dir": "NONE",
                        "read": "both timeframes are simply oscillating sideways — no trend, no "
                                "break, and NO volatility contraction. Nothing is loading; this "
                                "is the default resting state of a stock, not a setup."}
            both = hs == ls == "CONSOLIDATION"
            return {"tag": "NESTED SQUEEZE", "color": A, "loc": loc, "dir": "NONE",
                    "read": ("volatility CONTRACTING on both timeframes — the tightest version "
                             "of this setup. " if both else
                             "volatility CONTRACTING on one timeframe while the other stays "
                             "range-bound — a coil building inside a bigger box. ")
                            + "A move is loading but the DIRECTION IS UNKNOWN. Stand aside, mark "
                              "the HTF box high/low, trade the break — do not predict it."}
        if ls == "BREAKOUT_UP":
            if near_hi:
                return {"tag": "RANGE-TOP BREAK", "color": G, "loc": loc, "dir": "UP",
                        "read": "LTF breaking UP at the HTF ceiling — the ONLY location where an "
                                "LTF breakout can be real. Needs to HOLD above, ideally with "
                                "volume/delivery confirming, or it snaps back into the box."}
            return {"tag": "FALSE-BREAK TRAP", "color": R, "loc": loc, "dir": "NONE",
                    "read": "LTF pop UP in the MIDDLE of the HTF range — statistically fades back "
                            "into the box. Do not chase; the HTF HIGH is the line that matters."}
        if ls == "BREAKOUT_DOWN":
            if near_lo:
                return {"tag": "RANGE-FLOOR BREAK", "color": R, "loc": loc, "dir": "DOWN",
                        "read": "LTF breaking DOWN at the HTF floor — only here can it be real. "
                                "Must hold below, else it snaps back up into the range."}
            return {"tag": "FALSE-BREAK TRAP", "color": R, "loc": loc, "dir": "NONE",
                    "read": "LTF drop in the MIDDLE of the HTF range — statistically reverts. "
                            "Do not chase; the HTF LOW is the line that matters."}
        return {"tag": "DRIFT-IN-RANGE", "color": N, "loc": loc, "dir": "NONE",
                "read": "LTF drifting inside the HTF box — noise until it reaches an edge. "
                        "Read the HTF high/low as the only levels that matter."}

    # ── HTF is TRENDING / BROKEN OUT ─────────────────────────────────────────────
    htf_up = hs in _TREND_UP_S
    d = "UP" if htf_up else "DOWN"
    if ls in _RANGE_S:
        return {"tag": "WITH-TREND CONTINUATION", "color": G if htf_up else R, "loc": loc,
                "dir": "UP" if htf_up else "DOWN",
                "read": f"HTF {d}, LTF coiling — a pullback loading for CONTINUATION. The "
                        f"textbook with-trend setup. Trigger = the LTF breaking {d}; the idea "
                        f"is invalid if it breaks the other way."}
    ltf_up = ls in _TREND_UP_S
    if ltf_up == htf_up:
        return {"tag": "EXTENDED (aligned)", "color": A, "loc": loc, "dir": "UP" if htf_up else "DOWN",
                "read": f"HTF and LTF BOTH {d} — aligned, but late. Entering here is chasing; "
                        f"wait for the LTF to coil (a pullback) instead."}
    return {"tag": "PULLBACK vs HTF", "color": A, "loc": loc, "dir": "UP" if htf_up else "DOWN",
            "read": f"LTF {'DOWN' if htf_up else 'UP'} against an HTF {d} — a dip/rally zone "
                    f"WITH the higher-TF trend IF that structure holds; an early REVERSAL "
                    f"warning if the HTF level breaks. Watch the HTF pivot, not the LTF bar."}


# Setups that argue for a SIDE at all. Everything else is wait/avoid.
#
# DIRECTION IS NOT IN THE TAG. "WITH-TREND CONTINUATION" is the textbook setup in EITHER
# direction — a downtrend coiling for continuation is a SHORT, and the tag reads identically.
# Splitting long/short on the tag alone therefore served bearish continuations as long
# candidates (and pullbacks in a downtrend as dip-buys, which is precisely the trade that
# catches a falling knife). Split on `dir`, which synthesize() now returns explicitly.
DIRECTIONAL_TAGS = {"WITH-TREND CONTINUATION", "RANGE-TOP BREAK", "RANGE-FLOOR BREAK",
                    "PULLBACK vs HTF", "EXTENDED (aligned)"}
AVOID_TAGS = {"FALSE-BREAK TRAP"}
WAIT_TAGS = {"NESTED SQUEEZE", "DRIFT-IN-RANGE", "RANGE-BOUND (no setup)"}


def side_of(tag: str, direction: str) -> str:
    """LONG / SHORT / — for one row. A setup only takes a side when it is BOTH directional
    and pointing somewhere; traps and squeezes take none."""
    if tag in AVOID_TAGS or tag not in DIRECTIONAL_TAGS:
        return "—"
    return {"UP": "LONG", "DOWN": "SHORT"}.get(direction, "—")


# ── WHAT A SHORT ACTUALLY EARNS, BY HOLD LENGTH ──────────────────────────────────────
# Measured on this universe's own archive: 495,607 daily observations, 2018-2026, corporate
# actions back-adjusted, structure labelled causally with the same 20-bar rules the board
# uses. Population = every bar labelled TREND_DOWN or BREAKDOWN on the daily frame
# (n=43,042) -- i.e. exactly the "stock is in a downtrend on a 1D basis" case.
#
# Decomposition of the next day, from the SHORT's point of view (bps):
#
#     overnight GAP      +10.8     <- the short PAYS this, every night
#     intraday move       -4.4     <- the down move happens HERE, in the session
#     -----------------------
#     close-to-close      +5.1
#
# So the downtrend is real, but it is an INTRADAY phenomenon. Overnight, equities carry a
# structural upward drift that shows up in EVERY structure bucket -- +13.0bps even in
# TREND_DOWN, +1.9bps even after a BREAKDOWN -- and only 32.5% of nights gap down at all.
# The same overnight gap that IS the validated long edge in this project is, for a short,
# a nightly toll plus an unbounded tail: the worst 1% of gaps runs +438bps against you and
# 4.8% of nights gap more than 2% against you.
#
# Net short P&L before the ~22bps round-trip cost, by hold:
SHORT_EDGE_BPS = {"intraday": +4.4, "btst": -5.1, "swing": -13.6, "positional": -13.6}
# It decays MONOTONICALLY with hold length -- the exact opposite of the long side, where a
# longer horizon helps because the cost floor is amortised over a bigger move. Even the best
# case (+4.4bps, intraday) sits well under the 22bps cost floor, so no short here is
# tradeable; the difference is between "not tradeable" and "structurally paying to hold".


# ── DOES THE BOARD'S OWN SIDE LOGIC ACTUALLY SEPARATE WINNERS FROM LOSERS? ───────────
# Ran the REAL pipeline over history rather than trusting it: daily structure -> weekly
# structure (last COMPLETE week only) -> synthesize() -> side_of(), causally, on the
# corporate-action-adjusted archive. 468,661 observations, 2018-2026. This reconstructs the
# POSITIONAL preset exactly (1D trigger / 1W confirm); the intraday-frame presets cannot be
# validated this way because the broker only serves ~60 days of intraday history.
#
# Forward 20-day return, as EXCESS over the same-period universe baseline (+1.74%):
#
#     side        n         excess
#     LONG    81,129        +0.93%
#     SHORT   34,343        +0.57%     <- shorts OUTPERFORM. The side is INVERTED.
#     none   353,189        -0.27%
#
# Two things are true at once. The setup logic DOES find movers -- both sides beat the
# do-nothing bucket, so the tags are not noise. But the DIRECTION assignment is right on the
# long side and backwards on the short side. Per tag, on the short side:
#
#     EXTENDED (aligned) DOWN      n= 8,906   +1.62%   worst offender
#     PULLBACK vs HTF    DOWN      n=   753   +0.50%
#     WITH-TREND CONT.   DOWN      n=20,220   +0.47%
#     RANGE-FLOOR BREAK            n= 4,464   -1.09%   the ONLY one that works short
#
# The chartist reading of why: in a structurally rising market, a downtrend that has become
# EXTENDED is an oversold name at the end of its decline, and a downtrend that COILS is a
# base forming. Both are bottoming patterns. Shorting them is selling the low. Only a fresh
# break of the range FLOOR is genuinely bearish, and it is the smallest bucket.
#
# Long side for contrast (all four tags positive), but NOT stable: year-by-year LONG excess
# runs -1.36, +0.63, -0.58, +0.31, +0.95, +1.05, +0.93, +0.01, -0.95 -- five up, four down.
# Positive on average, sign-unstable in practice. Consistent with everything else in this
# stack: structure is context, not an edge.
SIDE_EXCESS_20D = {
    ("LONG", "EXTENDED (aligned)"): +1.36, ("LONG", "PULLBACK vs HTF"): +1.01,
    ("LONG", "WITH-TREND CONTINUATION"): +0.81, ("LONG", "RANGE-TOP BREAK"): +0.36,
    ("SHORT", "EXTENDED (aligned)"): +1.62, ("SHORT", "PULLBACK vs HTF"): +0.50,
    ("SHORT", "WITH-TREND CONTINUATION"): +0.47, ("SHORT", "RANGE-FLOOR BREAK"): -1.09,
}
# Short setups measured ANTI-PREDICTIVE (the name rose more than the market). Positive excess
# on a SHORT means the trade lost. Only RANGE-FLOOR BREAK survived.
SHORT_ANTI_PREDICTIVE = {"EXTENDED (aligned)", "PULLBACK vs HTF", "WITH-TREND CONTINUATION"}
SHORT_VALIDATED = {"RANGE-FLOOR BREAK"}


# ── THE SHORT SIDE MUST NOT BE SORTED BY TAG_RANK ────────────────────────────────────
# TAG_RANK encodes CHARTIST textbook quality, and on the short side the measurement says
# that ordering is upside down. WITH-TREND CONTINUATION is TAG_RANK 0 -- the best-looking
# setup on the board -- and it measured +0.47% excess against a short (n=20,220), i.e. it
# LOST. RANGE-FLOOR BREAK, the only tag that actually worked short (-1.09%), sits at rank 1
# and therefore sorted BELOW it.
#
# On the live board that produced an inverted list: of the SHORT names, 88% / 95% / 100% /
# 50% were anti-predictive tags (intraday / BTST / swing / positional), and the top TEN rows
# as sorted were 10 of 10 WITH-TREND CONTINUATION on three of the four presets, with the
# single validated RANGE-FLOOR BREAK buried underneath. The warning box above the table said
# "these are names to AVOID, never to short" while the table beneath it ranked the worst ones
# first -- and people act on order, not on prose.
#
# So the SHORT tab sorts by MEASURED short outcome (most negative excess = best short) and
# every row carries its own verdict. Lower is better, same convention as TAG_RANK.
SHORT_RANK = {t: i for i, t in enumerate(
    sorted((t for t in DIRECTIONAL_TAGS if ("SHORT", t) in SIDE_EXCESS_20D),
           key=lambda t: SIDE_EXCESS_20D[("SHORT", t)]))}


def short_verdict(tag: str) -> str:
    """Per-row honesty marker for the SHORT tab, from the 468,661-observation study."""
    ex = SIDE_EXCESS_20D.get(("SHORT", tag))
    if tag in SHORT_VALIDATED:
        return f"✅ works short ({ex:+.2f}%)"
    if tag in SHORT_ANTI_PREDICTIVE:
        return f"⛔ measured BACKWARDS ({ex:+.2f}% = the short lost)"
    return "— unrated"


# Can this horizon actually be SHORTED in the cash segment? No: Indian cash equity has no
# overnight short — an unsold short must be squared off the same day (delivery you do not own
# cannot be carried). Anything beyond intraday needs the futures leg, and the overnight short
# was separately measured -EV in this project. So the short side is a WEAKNESS SCREEN whose
# execution constraint changes with the horizon, and the UI must say which.
SHORTABLE_IN_CASH = {"intraday": True, "btst": False, "swing": False, "positional": False}
