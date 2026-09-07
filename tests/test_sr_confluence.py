"""S/R confluence flag — offline unit tests. No DB, no network.

The flag's whole value rests on it being SELECTIVE: the two frames of a horizon are
resampled from one price series, so a loose match fires on everything and means nothing
(measured: 196 of 197 names at an ATR-scale tolerance, with a shuffled control at 148).
These tests pin the three conditions that make it separable — same price on both frames,
inside the tolerance, and price actually AT the level on the side the trade needs.
"""
import io

import numpy as np
import pandas as pd

from eqbtst import config, live


def _kind(walls, built):
    """(level, touches) -> (level, touches, n_lows, n_highs), all touches of one kind.
    `built` is "L" (swing lows = a demand shelf) or "H" (swing highs = a supply ceiling)."""
    return [(x, t, t if built == "L" else 0, 0 if built == "L" else t) for x, t in walls]


def _board(ltf_walls, htf_walls, ltp=100.0, atr=2.0, side="LONG",
           ltf="1h", htf="4h", built="L", built_htf=None):
    """One-row board shaped like universe_mtf_scan's output, enough for add_setup.

    `built` says what kind of swing formed the levels — the distinction the board was blind
    to until walls_kind: "L" makes them demand shelves (price turned UP there), "H" makes
    them old ceilings. `built_htf` overrides it for the confirm frame only, so a test can
    make the two frames agree on the PRICE and disagree on what the level actually is."""
    row = {
        "symbol": "X", "sector": "Test", "ltp": ltp, "day%": 0.0, "turn₹L": 100.0,
        "_pc": ltp, "_vol_med20": 1e6, "_rs_cum9": 0.0,
        f"sr_wall{ltf}": ltf_walls, f"sr_wall{htf}": htf_walls,
        f"sr_wallk{ltf}": _kind(ltf_walls, built),
        f"sr_wallk{htf}": _kind(htf_walls, built_htf or built),
        f"sr_atr{ltf}": atr, f"sr_atr{htf}": atr * 2,
        f"sr_wallx{ltf}": [], f"sr_wallx{htf}": [],
        f"sr_blind{ltf}": (float("nan"), float("nan")),
        f"sr_blind{htf}": (float("nan"), float("nan")),
    }
    # boxes: an HTF uptrend with an LTF range -> a directional tag, so `side` is not "—"
    for tf, st_ in ((ltf, "CONSOLIDATION"), (htf, "TREND_UP" if side == "LONG" else "TREND_DOWN")):
        row[f"s{tf}"] = st_
        row[f"box_h{tf}"], row[f"box_l{tf}"], row[f"box_n{tf}"] = 110.0, 90.0, 20
    return pd.DataFrame([row])


def _conf(b, ltf="1h", htf="4h", **kw):
    out = live.add_setup(b, ltf=ltf, htf=htf, **kw)
    return out["sr_conf"].iloc[0], out["conf_gap"].iloc[0]


def test_fires_when_both_frames_hold_the_same_floor_under_price():
    # 99.5 is 0.25 ATR under a 100.0 spot, and both frames have it.
    txt, gap = _conf(_board([(99.5, 3)], [(99.5, 2)]))
    assert "SUP 99.50" in txt and "🛡️" in txt        # a real shelf, not a flip
    assert "3↓/0↑" in txt                            # 3 turns up, 0 turns down
    assert "1h+4h" in txt
    assert gap == 0.25


def test_touches_are_MAXed_across_frames_never_summed():
    # One swing seen twice by two resamples of one series is not two swings.
    txt, _ = _conf(_board([(99.5, 3)], [(99.5, 4)]))
    assert "×4" in txt          # not ×7


def test_higher_frame_disagreeing_on_the_price_kills_it():
    # A 4h level 4% away is a DIFFERENT level, not a confirmation of the 1h one. Deliberately
    # beyond the LOOSEST setting the slider offers, so this stays true at every tolerance.
    txt, gap = _conf(_board([(99.5, 3)], [(95.6, 4)]))
    assert txt == ""
    assert np.isinf(gap)


def test_confluence_reads_THE_PAIR_and_never_the_one_frame_up_wall():
    """🧲 S/R aligned and `Upper-TF S/R` answer DIFFERENT questions and must never share a
    wall list. Confluence = the horizon's OWN two frames (ltf + htf) agreeing. big_wall = the
    single frame ABOVE the pair, which the pair is blind to. Wiring confluence to the upper
    frame would silently turn one control into the other, and the two disagree constantly."""
    # 1h + 4h agree at 99.50; the frame ABOVE the pair (1D) has a level somewhere else.
    b = _board([(99.5, 3)], [(99.5, 2)])
    b["sr_wall1D"] = [[(80.0, 9)]]
    b["sr_wallx1D"] = [[(80.0, 9, 80.0, 80.0)]]
    b["sr_atr1D"] = 2.0
    b["sr_blind1D"] = [(float("nan"), float("nan"))]
    out = live.add_setup(b, ltf="1h", htf="4h")
    assert "SUP 99.50" in out["sr_conf"].iloc[0]                # the PAIR decided it
    # ...and moving the upper frame's wall onto price does NOT create a confluence, because
    # the upper frame is not one of the two frames being compared.
    b2 = _board([(99.5, 3)], [(97.0, 2)])                        # the pair does NOT agree
    b2["sr_wall1D"] = [[(99.5, 9)]]                              # but the frame above does
    b2["sr_wallx1D"] = [[(99.5, 9, 99.5, 99.5)]]
    b2["sr_atr1D"] = 2.0
    b2["sr_blind1D"] = [(float("nan"), float("nan"))]
    assert live.add_setup(b2, ltf="1h", htf="4h")["sr_conf"].iloc[0] == ""


def test_tolerance_is_adjustable_and_the_loose_end_is_reachable():
    """The user picks how closely the frames must agree. A 1% disagreement must be rejected
    at the default and accepted at the 2% end — otherwise the slider is decorative."""
    # Both levels BELOW price (a floor is confirmed by a floor, never by a ceiling on the
    # other side of the tape), ~1% of price apart.
    b = _board([(99.5, 3)], [(98.5, 4)])
    assert _conf(b)[0] == ""                        # default 0.50% -> not the same level
    tight = live.add_setup(b.copy(), ltf="1h", htf="4h",
                           conf_tol_bps=config.SR_CONF_TOL_MIN_BPS)
    loose = live.add_setup(b.copy(), ltf="1h", htf="4h",
                           conf_tol_bps=config.SR_CONF_TOL_MAX_BPS)
    assert tight["sr_conf"].iloc[0] == ""
    assert "SUP 99.50" in loose["sr_conf"].iloc[0]
    # The level is still quoted at the TRIGGER frame's price, whatever the tolerance.
    assert loose["_conf_px"].iloc[0] == 99.5


def test_loosening_can_only_ever_add_names_never_drop_one():
    """Monotonicity. A wider tolerance is a superset — if it ever removed a name the slider
    would be unreadable (loosening would have to be tried in both directions)."""
    b = _board([(99.5, 3)], [(99.9, 4)])
    fired = [live.add_setup(b.copy(), ltf="1h", htf="4h", conf_tol_bps=t)["sr_conf"].iloc[0] != ""
             for t in (25.0, 50.0, 100.0, 150.0, 200.0)]
    assert fired == sorted(fired)                   # False... then True, never back again


def test_tolerance_boundary_is_price_relative_not_atr_relative():
    tol = config.SR_CONF_TOL_BPS / 1e4 * 100.0        # bps of a 100.00 spot
    inside, _ = _conf(_board([(99.5, 3)], [(99.5 + tol * 0.5, 2)]))
    outside, _ = _conf(_board([(99.5, 3)], [(99.5 + tol * 3, 2)]))
    assert inside != ""
    assert outside == ""
    # ...and the ATR-scale match a naive implementation would use must NOT fire: 0.6 ATR is
    # 1.20 on this board, ~60x the price tolerance. This is the whole tautology guard.
    assert _conf(_board([(99.5, 3)], [(99.5 + 0.6 * 2.0, 2)]))[0] == ""


def test_price_must_be_AT_the_level_not_merely_somewhere_in_the_box():
    near = config.SR_CONF_NEAR_ATR * 2.0              # 0.5 ATR = 1.00 on this board
    at, _ = _conf(_board([(100.0 - near * 0.5, 3)], [(100.0 - near * 0.5, 2)]))
    far, gap = _conf(_board([(100.0 - near * 3, 3)], [(100.0 - near * 3, 2)]))
    assert at != ""
    assert far == "" and np.isinf(gap)


def test_direction_follows_the_trade_long_wants_a_floor_short_a_ceiling():
    # The SAME wall pair, read from each side. A long needs it UNDER price; a short OVER.
    below, above = [(99.5, 3)], [(100.5, 3)]
    assert "SUP" in _conf(_board(below, below, side="LONG"))[0]
    assert _conf(_board(above, above, side="LONG"))[0] == ""      # ceiling is not a long's level
    assert "RES" in _conf(_board(above, above, side="SHORT", built="H"))[0]
    assert _conf(_board(below, below, side="SHORT", built="H"))[0] == ""


def test_nearest_agreeing_level_wins_when_several_align():
    txt, gap = _conf(_board([(99.8, 2), (99.2, 5)], [(99.8, 2), (99.2, 5)]))
    assert "99.80" in txt        # 0.1 ATR away beats the stronger x5 at 0.4 ATR
    assert gap == 0.10


def test_missing_or_degenerate_inputs_never_raise_and_never_claim_a_level():
    for ltf_w, htf_w, atr in (([], [(99.5, 2)], 2.0),
                              ([(99.5, 2)], [], 2.0),
                              ([(99.5, 2)], [(99.5, 2)], 0.0)):
        b = _board(ltf_w, htf_w, atr=atr)
        txt, gap = _conf(b)
        assert txt == "" and np.isinf(gap)


def test_refresh_clears_the_flag_once_price_trades_through_the_level():
    """A support is only a support while it is UNDERFOOT. The live tick must drop the flag
    rather than keep advertising a floor that price has fallen through."""
    b = live.add_setup(_board([(99.5, 3)], [(99.5, 2)]), ltf="1h", htf="4h")
    assert b["sr_conf"].iloc[0] != ""
    px = b["_conf_px"].iloc[0]
    assert px == 99.5
    # Re-derive exactly as refresh_prices does, with price now BELOW the level.
    for new_ltp, expect in ((99.6, True), (99.4, False), (105.0, False)):
        gap = (new_ltp - px) / 2.0
        live_ok = bool(0 <= gap <= config.SR_CONF_NEAR_ATR)
        assert live_ok is expect


# ── WHAT BUILT THE LEVEL — the difference between a floor and a broken ceiling ───────
def test_a_support_built_from_swing_LOWS_is_a_shelf():
    """The textbook object: price came down to it and turned UP, on both frames."""
    txt, _ = _conf(_board([(99.5, 3)], [(99.5, 2)], built="L"))
    assert "🛡️" in txt and "SUP 99.50" in txt
    assert "flip" not in txt


def test_a_support_built_from_swing_HIGHS_is_labelled_a_FLIP_not_a_shelf():
    """An old ceiling price has broken above. Still a level under price — but it has never
    once held price up, and calling it plain 'support' is the conflation this read ends."""
    txt, _ = _conf(_board([(99.5, 3)], [(99.5, 2)], built="H"))
    assert "SUP-flip" in txt and "🔄" in txt
    assert "0↓/3↑" in txt                    # every turn there was DOWN, none up
    # the split is the TRIGGER frame's own (3), never the two frames summed (3+2=5):
    # the same "one swing seen twice" inflation that touches are MAXed to avoid.
    assert "×3" in txt


def test_the_two_frames_must_agree_on_WHAT_the_level_is_not_only_its_price():
    """A 1h swing low and a 4h swing high at the same rupee do not confirm each other —
    one is a floor, the other a ceiling. Same price, opposite meaning -> MIXED, so the
    default SHELF filter rejects it."""
    b = _board([(99.5, 3)], [(99.5, 2)], built="L", built_htf="H")
    assert _conf(b, conf_kind="SHELF")[0] == ""
    assert _conf(b, conf_kind="FLIP")[0] == ""
    txt, _ = _conf(b, conf_kind="ANY")       # visible, but honestly labelled unresolved
    assert "SUP?" in txt and "➖" in txt


def test_kind_filter_selects_and_ANY_keeps_everything():
    shelf = _board([(99.5, 3)], [(99.5, 2)], built="L")
    flip = _board([(99.5, 3)], [(99.5, 2)], built="H")
    assert _conf(shelf, conf_kind="SHELF")[0] != ""
    assert _conf(shelf, conf_kind="FLIP")[0] == ""
    assert _conf(flip, conf_kind="FLIP")[0] != ""
    assert _conf(flip, conf_kind="SHELF")[0] == ""
    assert _conf(shelf, conf_kind="ANY")[0] != ""
    assert _conf(flip, conf_kind="ANY")[0] != ""


def test_short_side_mirrors_it_a_real_ceiling_is_built_from_swing_HIGHS():
    """Role decides which pivot kind counts as genuine: highs make a real ceiling over a
    short, exactly as lows make a real floor under a long."""
    above = [(100.5, 3)]
    real = _conf(_board(above, above, side="SHORT", built="H"))[0]
    flipped = _conf(_board(above, above, side="SHORT", built="L"))[0]
    assert "RES 100.50" in real and "🛡️" in real
    assert "RES-flip" in flipped


def test_a_board_without_the_kind_list_still_works_and_says_it_does_not_know():
    """Backward compatibility: a board built before walls_kind existed has no sr_wallk. It
    must not crash, and must not silently claim the level is a verified shelf."""
    b = _board([(99.5, 3)], [(99.5, 2)])
    b = b.drop(columns=["sr_wallk1h", "sr_wallk4h"])
    txt, gap = _conf(b, conf_kind="ANY")
    assert "SUP?" in txt and gap == 0.25
    assert "↓/" not in txt                   # no pivot split invented from missing data


# ── AUDIT REGRESSIONS ────────────────────────────────────────────────────────────────
def test_the_confirming_level_must_be_on_the_SAME_SIDE_of_price():
    """A 4h CEILING cannot confirm a 1h FLOOR. Matching on price distance alone let one do
    exactly that whenever the two sat within tolerance across the tape — measured on the live
    board at 9% of matches at 0.50% tolerance and 22% at 2.00%, growing with the width."""
    # price 100.00; 1h floor at 99.50; the only 4h level is at 101.40, ABOVE price.
    b = _board([(99.5, 3)], [(101.4, 4)])
    for tol in (25.0, 50.0, 100.0, 200.0):
        assert _conf(b.copy(), conf_tol_bps=tol, conf_kind="ANY")[0] == "", tol
    # with a 4h level on the correct side, the same geometry DOES confirm
    ok = _board([(99.5, 3)], [(99.4, 4)])
    assert "SUP 99.50" in _conf(ok, conf_kind="ANY")[0]


def test_a_no_side_row_can_show_a_CEILING_not_only_a_floor():
    """No-side rows are where this filter actually lives (a name parked on a shelf is not
    trending), and they used to look DOWN unconditionally — so a resistance confluence was
    unreachable there. Live board: 10 of 30 confluence names were ceilings."""
    def _noside(walls, built):
        b = _board(walls, walls, built=built)
        b.loc[0, "s1h"] = "RANGE"
        b.loc[0, "s4h"] = "RANGE"
        return b
    up = _conf(_noside([(100.5, 3)], "H"), conf_kind="ANY")[0]     # ceiling overhead
    dn = _conf(_noside([(99.5, 3)], "L"), conf_kind="ANY")[0]      # floor underfoot
    assert "RES 100.50" in up
    assert "SUP 99.50" in dn


def test_no_side_picks_the_NEARER_of_the_two_when_price_sits_between_levels():
    """Given both a floor and a ceiling in range, the row reports the one price is actually
    standing on — not whichever the loop happened to reach first."""
    b = _board([(99.9, 3), (100.6, 3)], [(99.9, 3), (100.6, 3)], built="L")
    b.loc[0, "s1h"] = "RANGE"
    b.loc[0, "s4h"] = "RANGE"
    txt, gap = _conf(b, conf_kind="ANY")
    assert "99.90" in txt            # 0.05 ATR away beats the ceiling 0.30 ATR away
    assert gap == 0.05


# ── THE 🧲 CLOCK: `entered` becomes "when did price arrive at the level" ─────────────
def _cndl(lows, highs, tf="1h", start="2026-09-04 09:15"):
    freq = {"15m": "15min", "1h": "60min", "4h": "240min", "1D": "D"}[tf]
    ts = pd.date_range(start, periods=len(lows), freq=freq)
    return pd.DataFrame({"ts": ts, "open": highs, "high": highs, "low": lows,
                         "close": [(a + b) / 2 for a, b in zip(lows, highs)],
                         "volume": [1] * len(lows)})


def test_arrival_is_the_start_of_the_current_stay_in_the_zone():
    d = _cndl([105, 104, 100.2, 100.1, 100.3], [106, 105, 101, 100.8, 100.9])
    lab, px, bars = live._level_arrival(d, 100.0, 2.0, "1h")   # zone 99.00-101.00
    assert lab == "11:15" and bars == 3


def test_leaving_the_zone_and_returning_starts_a_NEW_test():
    """A level price walked away from and came back to has been tested TWICE. Reporting the
    first arrival would date the stay before a departure the trader can see on the chart."""
    d = _cndl([100.2, 100.1, 105, 100.3, 100.2], [101, 100.8, 106, 100.9, 100.7])
    lab, _, bars = live._level_arrival(d, 100.0, 2.0, "1h")
    assert lab == "12:15" and bars == 2      # the second approach, not the first


def test_a_stay_that_has_ENDED_is_marked_was_never_shown_as_current():
    """Price was on the shelf and has left. Reporting that visit bare would read as a live
    stay — the column would claim price has been at the level since 09:15 when it walked away
    at 11:15. Past visits are useful; past visits disguised as current ones are not."""
    d = _cndl([100.2, 100.1, 105], [101, 100.8, 106])
    lab, _, bars = live._level_arrival(d, 100.0, 2.0, "1h")
    assert lab.startswith("was ") and bars == 2
    # With no in-zone bar anywhere there is nothing to date at all.
    empty = _cndl([105, 104], [106, 105])
    assert live._level_arrival(empty, 100.0, 2.0, "1h")[0] == "just now"


def test_a_wick_into_the_shelf_counts_as_a_test_of_it():
    """Range intersection, not close-inside: the rejection bars a chartist watches for are
    exactly the ones that wick in and close back out."""
    d = _cndl([104, 99.5], [105, 103])
    lab, _, bars = live._level_arrival(d, 100.0, 2.0, "1h")
    assert lab == "10:15" and bars == 1


def test_label_follows_the_trigger_frame_daily_stamps_a_DATE():
    """'09:15' is meaningless on a 1D trigger — Positional must stamp the session."""
    d = _cndl([105, 100.2, 100.1], [106, 101, 100.8], tf="1D")
    lab, _, _ = live._level_arrival(d, 100.0, 2.0, "1D")
    assert lab == "05-Sep"


def test_a_stay_crossing_a_session_prefixes_the_day():
    d = _cndl([100.2] * 30, [100.8] * 30, tf="1h")
    lab, _, bars = live._level_arrival(d, 100.0, 2.0, "1h")
    assert lab.startswith("04 ") and bars == 30


def test_arrival_clock_degrades_quietly_on_bad_input():
    d = _cndl([100.2, 100.1], [101, 100.8])
    assert live._level_arrival(None, 100.0, 2.0, "1h") == (None, None, 0)
    assert live._level_arrival(d, float("nan"), 2.0, "1h") == (None, None, 0)
    assert live._level_arrival(d, 100.0, 0.0, "1h") == (None, None, 0)
    assert live._level_arrival(d, 100.0, float("nan"), "1h") == (None, None, 0)


def test_the_clock_and_the_flag_use_the_SAME_zone():
    """If the arrival zone and the confluence zone ever diverged, a row could show a level it
    is 'at' with no arrival time, or an arrival time for a level it is not at."""
    import inspect
    src = inspect.getsource(live._level_arrival)
    assert "SR_CONF_NEAR_ATR" in src


def test_the_clock_reads_the_SAME_price_the_flag_fired_on():
    """The flag fires on `ltp`; the clock reads candle ranges. Off-hours the newest bar is a
    zero-range indicative stub (measured live: WIPRO L=H=C=189.98 against a 184.16 quote, a
    3.1% gap; GAIL 9.1%; SUNTV 10.8%), so nothing intersected the zone and every row read
    'just now'. Widening the last bar by the live price is the same rule refresh_prices
    already applies to bar_clr, and it makes the two agree by construction."""
    stub = _cndl([105, 104, 100.2, 100.1, 189.98], [106, 105, 101, 100.8, 189.98])
    # Without the live price the clock can only report the last visit, correctly marked past.
    assert live._level_arrival(stub, 100.0, 2.0, "1h")[0] == "was 04 11:15"
    lab, _, bars = live._level_arrival(stub, 100.0, 2.0, "1h", ltp=100.0)
    assert lab == "11:15" and bars == 3                                      # real arrival


def test_a_fresh_mid_bar_arrival_counts_one_bar_not_zero():
    """Price walked into the zone inside the forming bar: the stay is one bar old, dated at
    that bar, not an unhelpful 'just now'."""
    d = _cndl([105, 104, 103], [106, 105, 104])
    lab, _, bars = live._level_arrival(d, 100.0, 2.0, "1h", ltp=100.2)
    assert lab == "11:15" and bars == 1
    # ...and with no live price supplied there is genuinely nothing to date it with
    assert live._level_arrival(d, 100.0, 2.0, "1h")[0] == "just now"


def test_the_clock_never_mutates_the_callers_candles():
    """deep_state's frame is shared with every other column on the row. Widening the last bar
    in place would silently rewrite the high/low that structure, RSI and bar_clr read."""
    d = _cndl([105, 104, 103], [106, 105, 104])
    before = d["high"].tolist(), d["low"].tolist()
    live._level_arrival(d, 100.0, 2.0, "1h", ltp=100.2)
    assert (d["high"].tolist(), d["low"].tolist()) == before


# ── SCOPE: "whole universe" must actually bypass the band, at BOTH sites ────────────
def test_whole_universe_scope_bypasses_the_price_band_everywhere():
    """Filters COMMUTE, so a scope control that merely re-ordered the stages would return a
    byte-identical list and be decoration. It has to drop the band -- and the band is applied
    TWICE (once building the pool, once after the per-name enrich). Missing the second one
    would silently undo the bypass on exactly the rows it was meant to reveal, while the
    funnel still claimed the universe had been searched."""
    dash = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    # the pool is built unbanded in universe mode
    assert 'light = sc["board"].copy() if _sr_universe else price_filter(sc["board"], "ltp")' in dash
    # ...and the post-enrich re-application is guarded by the same flag
    i = dash.index('enr = price_filter(enr, "ltp")')
    assert "if not _sr_universe:" in dash[i - 400:i],         "the post-enrich price_filter must be skipped in whole-universe mode"
    # the scope is read BEFORE the pool is built, or the branch above cannot see it
    assert dash.index('mtf_confscope') < dash.index('light = sc["board"].copy()') or         dash.index('_conf_scope = str(') < dash.index('light = sc["board"].copy()')
    # ...and in whole-universe mode the S/R read is the ONLY thing deciding the list:
    # every other filter is overridden, not merely re-ordered.
    assert '_setup_f, _room_f = "All", "All"' in dash,         "setup-quality and upper-TF must be overridden in whole-universe mode"
    assert "after_deliv = light if _sr_universe else _deliv_filter(light)" in dash,         "the delivery sliders must be skipped in whole-universe mode"
    assert "filtered = after_deliv if _sr_universe else _mtf_filter(after_deliv)" in dash,         "the structure boxes must be skipped in whole-universe mode"
    # the horizon is NOT a filter -- it defines which two frames the question is about
    assert 'ltf=_P["ltf"], htf=_P["htf"]' in dash


def test_whole_universe_names_every_control_it_overrode():
    """A control that silently stops biting is worse than one that is absent, so the funnel
    lists each overridden filter by name -- which means the REQUESTED values have to be
    captured before they are forced to 'All'."""
    dash = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    assert "_setup_f_req, _room_f_req = _setup_f, _room_f" in dash
    assert dash.index("_setup_f_req, _room_f_req") < dash.index('_setup_f, _room_f = "All", "All"')
    for named in ("price band", "setup quality", "upper-TF", "delivery", "structure"):
        assert f'"{named}"' in dash, f"funnel must be able to name {named}"


def test_scope_defaults_to_the_current_list():
    """Opening the board must not silently widen what every other filter was set against."""
    dash = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    assert 'st.session_state.get("mtf_confscope", "NONE")' in dash


def test_filtered_panel_has_a_NO_SIDE_tab_like_the_prefilter_panel():
    """The pre-filter panel splits LONG / SHORT / No side; the FILTERED panel split only two
    ways, so every row whose setup takes no direction was computed, enriched, counted in the
    funnel and then dropped at render. That is fatal for the S/R filter specifically: a name
    standing on a level both frames agree on is almost by definition going sideways, so its
    main output had nowhere to appear. Reported as 'whole universe shows the same as current
    list' — it did not, both just collapsed to the same one or two directional rows."""
    dash = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    i = dash.index("def _struct_panel")
    body = dash[i:dash.index("        _struct_panel()", i)]
    assert "tb, tsh, tno = st.tabs(" in body, "the filtered panel must split three ways"
    assert "No side (" in body
    assert "with tno:" in body
    # the no-side frame must be the COMPLEMENT of the two directional ones, so nothing can
    # fall between the cracks the way it did before.
    assert "_no_n = _conf_order(bb[~bb.index.isin(_lo_n.index.union(_sh_n.index))])" in body
    # counts in the labels — this defect was invisible precisely because they were absent.
    # Asserted on the COUNT EXPRESSIONS, not on a fixed label string: the labels themselves
    # change with the split rule (structure tag vs level side), the counts must not.
    assert "{len(_lo_n)}" in body and "{len(_sh_n)}" in body and "{len(_no_n)}" in body


def test_every_enriched_row_lands_in_exactly_one_tab():
    """LONG, SHORT and No side must partition the enriched frame — no row shown twice, none
    dropped. Pinned as set algebra on the index rather than on labels."""
    import pandas as _pd
    bb = _pd.DataFrame({"side": ["LONG", "SHORT", "—", "LONG", "—"]})
    lo = bb[bb["side"] == "LONG"]
    sh = bb[bb["side"] == "SHORT"]
    no = bb[~bb.index.isin(lo.index.union(sh.index))]
    assert len(lo) + len(sh) + len(no) == len(bb)
    assert set(lo.index) | set(sh.index) | set(no.index) == set(bb.index)
    assert not (set(lo.index) & set(sh.index)) and not (set(no.index) & set(lo.index))


def test_sr_columns_lead_the_row_when_the_filter_decides_the_list():
    """If 🧲 chose which names are on the board, `S/R aligned` is the answer to "why is this
    here" — and it was sitting ~15 columns right, behind the F&O block, off the edge of the
    screen. Reported as "it should come based on support/resistance": the filter was right,
    the table was not saying so. Same defect the `sector tilt` relocation already fixed."""
    dash = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    i = dash.index("def _day_by_setup")
    # SCAN THE WHOLE FUNCTION, never a fixed byte count — a comment added inside it would
    # otherwise push the assertion out of the window and fail a test about unchanged code.
    body = dash[i:dash.index("long_cols = _day_by_setup(", i)]
    assert "if _conf_f:" in body and '"sr_conf", "conf_gap"' in body
    # hoisted to just after `symbol`, i.e. BEFORE the setup relocation runs
    assert body.index('_sr = [c for c in ("sr_conf", "conf_gap")') < body.index('if "setup" in cols:')


def test_column_lists_cannot_render_a_duplicate_column():
    """The column lists are built by concatenating conditional fragments, so the moment one
    fragment hoists a column to the front while another still declares it later, the name
    appears twice — and df[[...twice...]] hands Streamlit two identical columns rather than
    raising. Caught for real when the S/R columns were promoted to lead the row while still
    sitting inside the shared structure block. _cols must de-duplicate, first mention wins."""
    import pandas as _pd
    src = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    i = src.index("def _cols(df, cols):")
    j = src.index("def _struct_label", i)          # next top-level def after it
    ns = {}
    exec(src[i:j], ns)
    df = _pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    out = ns["_cols"](df, ["a", "b", "c", "a", "b", "zzz"])
    assert out == ["a", "b", "c"], out
    assert len(df[out].columns) == len(set(df[out].columns))


def test_the_real_light_cols_list_survives_dedup():
    """light_cols legitimately mentions sr_conf/conf_gap twice once the filter hoists them —
    the hoist is conditional, the structure block is not. Proves the render path is safe."""
    import pandas as _pd
    src = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    i = src.index("def _cols(df, cols):")
    j = src.index("def _struct_label", i)
    ns = {}
    exec(src[i:j], ns)
    cols = ["symbol", "sr_conf", "conf_gap", "sector", "setup",
            "loc", "at_wall", "sr_conf", "conf_gap", "sup"]
    df = _pd.DataFrame({c: [0] for c in set(cols)})
    out = ns["_cols"](df, cols)
    assert out.count("sr_conf") == 1 and out.count("conf_gap") == 1
    assert out.index("sr_conf") == 1, "the hoisted position must win, not the later one"


# ── THE LEVEL DECIDES THE SIDE (not the structure tag) ──────────────────────────────
def test_setup_tag_must_not_gate_which_side_is_searched():
    """The structure tag used to decide which side of price the confluence was allowed to look
    at, so a name reading LONG could never surface a resistance confluence however plainly
    price sat under a ceiling. Measured on the live board: 143 matched with LONG 0 / SHORT 4 /
    No side 56, and 7 names were invisible outright."""
    # tag says LONG (HTF uptrend), but the only aligned level is a CEILING overhead
    b = _board([(100.5, 3)], [(100.5, 2)], side="LONG", built="H")
    assert _conf(b.copy(), conf_kind="ANY")[0] == ""            # gated: never searched
    out = live.add_setup(b.copy(), ltf="1h", htf="4h", conf_kind="ANY", conf_any_side=True)
    assert "RES 100.50" in out["sr_conf"].iloc[0]               # found once ungated
    assert out["_conf_side"].iloc[0] == "RES"


def test_conf_any_side_can_only_add_matches_never_remove_one():
    """Ungating is a superset: every level the tag-gated search could find is still found."""
    for side, walls, built in (("LONG", [(99.5, 3)], "L"), ("SHORT", [(100.5, 3)], "H")):
        b = _board(walls, walls, side=side, built=built)
        gated = live.add_setup(b.copy(), ltf="1h", htf="4h", conf_kind="ANY")
        free = live.add_setup(b.copy(), ltf="1h", htf="4h", conf_kind="ANY",
                              conf_any_side=True)
        if gated["sr_conf"].iloc[0]:
            assert free["sr_conf"].iloc[0] == gated["sr_conf"].iloc[0]


def test_tabs_split_on_the_LEVEL_side_when_the_filter_is_on():
    """price on a floor = long candidate, price under a ceiling = short candidate. Splitting
    on the structure tag instead put ~93% of matches under 'No side', because a name parked on
    a shelf is range-bound BY THAT MEASURE — which is the whole point of the filter."""
    dash = io.open("eqbtst/dashboard.py", encoding="utf-8").read()
    i = dash.index("def _struct_panel")
    body = dash[i:dash.index("        _struct_panel()", i)]
    assert '_by_level = _conf_f and "_conf_side" in bb.columns' in body
    assert '_lo_n = _conf_order(bb[bb["_conf_side"] == "SUP"])' in body
    assert '_sh_n = _conf_order(bb[bb["_conf_side"] == "RES"])' in body
    # ...and the tab says which rule produced it, so the meaning is never ambiguous
    assert "at SUPPORT" in body and "at RESISTANCE" in body
    # the structure split must still exist for when the filter is OFF
    assert 'bb["side"] == "LONG"' in body and 'bb["side"] == "SHORT"' in body
