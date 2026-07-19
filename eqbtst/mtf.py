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


# Can this horizon actually be SHORTED in the cash segment? No: Indian cash equity has no
# overnight short — an unsold short must be squared off the same day (delivery you do not own
# cannot be carried). Anything beyond intraday needs the futures leg, and the overnight short
# was separately measured -EV in this project. So the short side is a WEAKNESS SCREEN whose
# execution constraint changes with the horizon, and the UI must say which.
SHORTABLE_IN_CASH = {"intraday": True, "btst": False, "swing": False, "positional": False}
