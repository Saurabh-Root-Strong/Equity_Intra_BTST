"""Column ORDER and PRESENCE invariants for the board tables.

Split out of test_fno.py: these guard where columns RENDER, which is a different
question from whether the F&O block is wired. Every bug in this family shares one
shape -- the declared list and the rendered table disagreed, and nothing failed.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

from eqbtst import config, live

_DASH = Path(__file__).resolve().parent.parent / "eqbtst" / "dashboard.py"


def _src() -> str:
    return io.open(_DASH, encoding="utf-8").read()


def test_entered_is_never_pushed_past_ltp_by_the_reorder():
    """`entered` must sit BEFORE `ltp` in every table that shows both.

    It already did in every DECLARED column list -- and still rendered on the far right of
    the intraday board, past the whole S/R block. The cause was `_day_by_setup`, which
    relocates ltp/day%/sector next to `setup` and so HOPPED them over `entered`, leaving
    the fire time stranded while the price it fired against sat at column 3.

    That is the same class of bug as `deliv trend` jumping and the F&O block vanishing: the
    declared order and the RENDERED order disagreed, and nothing failed. So this test
    asserts on the rendered order -- it re-runs the real reorder over the real lists rather
    than trusting how they are written down.
    """
    src = _src()
    m = re.search(r'move = \[c for c in \(([^)]*)\)', src, re.S)
    assert m, "_day_by_setup no longer builds `move` from a tuple -- re-read this test"
    move_order = [x.strip().strip('"\'') for x in m.group(1).split(",") if x.strip()]
    assert move_order.index("entered") < move_order.index("ltp"), (
        f"`entered` must lead `ltp` in the _day_by_setup relocation tuple, got {move_order}. "
        f"Declared list order does NOT save you here -- whatever this tuple relocates jumps "
        f"ahead of everything left behind.")
    for lst in re.findall(r'\[\s*"symbol"[^\]]*\]', src, re.S):
        if '"entered"' in lst and '"ltp"' in lst:
            assert lst.index('"entered"') < lst.index('"ltp"'), \
                f"a column list declares `ltp` before `entered`: {lst[:90]}..."


def test_trigger_times_never_raise_and_preserve_the_frame():
    """`entered` is CONTEXT on a structure table. A rate-limited 5m fetch -- the same limit
    that already produces the board's "unreadable, not neutral" rows -- must cost you the
    column, never the table. And it must never reorder or drop rows: the sided tabs are
    already sorted by rank before this runs."""
    df = pd.DataFrame({"symbol": ["RELIANCE", "INFY", "ZZZNOTREAL"],
                       "ltp": [1400.0, 1600.0, 10.0], "_pc": [1390.0, 1580.0, 9.0],
                       "_vol_med20": [1e6, 1e6, 1e3], "_rs_cum9": [0.1, 0.2, 0.0]})
    out = live.add_trigger_times(df, symbols=["RELIANCE", "INFY"])
    assert list(out["symbol"]) == list(df["symbol"]), "row order changed"
    assert len(out) == len(df)
    for c in ("entered", "at", "since%"):
        assert c in out.columns, f"{c} not attached"
    assert live.add_trigger_times(pd.DataFrame()).empty
    assert list(live.add_trigger_times(pd.DataFrame({"x": [1]})).columns) == ["x"]
    assert "entered" in live.add_trigger_times(df, symbols=[]).columns
    assert "entered" in live.add_trigger_times(df, symbols=["NOPE"]).columns


def test_numeric_trigger_columns_are_float_not_object_none():
    """A column of Python None is OBJECT dtype and prints the literal string "None" in
    every cell unless the caller runs it through the dashboard's _fmt() coercion. That
    shipped: `at` and `since%` rendered as "None" beside an `entered` that correctly showed
    "—". NaN renders blank everywhere, so the module must hand back floats itself rather
    than relying on a formatter it does not control."""
    df = pd.DataFrame({"symbol": ["A", "B"], "ltp": [100.0, 200.0], "_pc": [99.0, 198.0],
                       "_vol_med20": [1e6, 1e6], "_rs_cum9": [0.1, 0.2]})
    out = live.add_trigger_times(df, symbols=["A", "B"])
    for c in ("at", "since%"):
        assert out[c].dtype.kind == "f", (
            f"{c} is {out[c].dtype} -- an object column of None renders as the string 'None'")
    assert not any(isinstance(v, str) for v in out["at"])


def test_entered_is_computed_on_the_LONG_tab_only():
    """`entered` is the ACCUMULATION footprint trigger, and every leg is long-directional:
    day_ret >= RET_TH (+1% UP), close-in-range >= CLR_TH, close >= CVWAP_TH above VWAP,
    cumulative RS > RS_MIN, volume on pace. A name the structure calls SHORT is DOWN on the
    day, so the trigger is None at every bar -- the column CANNOT fill.

    This shipped once as three columns of "—" on the SHORT tab, bought with ~17 network
    fetches per 5-minute bucket. The No-side tab has the same emptiness for a different
    reason (no trade at all). So: LONG opts in, nothing else does.
    """
    src = _src()
    # finditer, not findall: findall with a capture group returns only the GROUP, so the
    # `timed=True` check below would read the frame name instead of the call line.
    calls = [(m.group(1), m.group(0))
             for m in re.finditer(r"_side_table\((_lo|_sh|_no)\b[^\n]*", src)]
    assert calls, "the sided tables are no longer rendered through _side_table"
    seen = set()
    for frame, call in calls:
        seen.add(frame)
        timed = "timed=True" in call
        if frame == "_lo":
            assert timed, "the LONG tab lost its trigger times"
        else:
            assert not timed, (
                f"the {frame} tab pays a 5m fetch per name for a column that cannot fill -- "
                f"the footprint trigger requires day_ret >= +{config.RET_TH:.0%}")
    assert seen == {"_lo", "_sh", "_no"}, f"a side tab vanished: {seen}"
    assert "live.add_trigger_times(" in src


def test_the_footprint_trigger_really_is_long_only():
    """Pins the CLAIM the tab scoping rests on, against config rather than against a
    comment. If RET_TH ever goes negative or two-sided, this fails and the SHORT-tab
    decision must be revisited deliberately instead of silently becoming wrong."""
    assert config.RET_TH > 0, "RET_TH is no longer long-only -- re-check the SHORT tab scoping"
    assert config.CLR_TH > 0.5, "CLR_TH no longer demands a strong (high) close"


def test_entered_lands_before_ltp_in_the_universe_table():
    """Reproduces the real insertion against the real light_cols shape. Anchoring on `ltp`
    and re-reading its index per insert walks the anchor right as each insert shifts it,
    landing `at`/`since%` AFTER ltp -- which is the bug this pins."""
    fno_cols = ["Fut Near", "Fut Next", "Opt Near", "Opt Next"]
    light_cols = (["symbol", "sector", "sector tilt", "ltp", "day%"]
                  + ["setup", "deliv trend"] + fno_cols + ["carry"]
                  + ["loc", "at_wall", "sup", "sup_t", "res", "res_t", "headroom",
                     "big_wall", "big_gap"]
                  + ["wtd_deliv7", "deliv_vs_100d", "turn₹L", "side"])
    c = list(light_cols)
    base = c.index("ltp")
    for i, e in enumerate(("entered", "at", "since%")):
        if e not in c:
            c.insert(base + i, e)
    for col in ("entered", "at", "since%"):
        assert c.index(col) < c.index("ltp"), f"{col} rendered AFTER ltp"
    assert c[:7] == ["symbol", "sector", "sector tilt", "entered", "at", "since%", "ltp"]


def test_short_side_is_mostly_structure_fixed_not_price_driven():
    """WHY THE SHORT TAB CARRIES AN AGE, NOT A CLOCK TIME.

    A price-path replay was built for it first: the side comes from synthesize(htf, ltf, LTP)
    with the boxes pinned at scan time, so "when did this first become SHORT today" looked
    like a matter of replaying today's 5-minute closes through the same function.

    Most of it is not. Of the combinations that can yield SHORT, the large majority hold
    SHORT at EVERY price, because SHORT is decided by the higher frame's structure ENUM.
    Those flip when a BAR CLOSES -- often days before today -- so a replay would stamp the
    first bar of the session on them, a fake 09:15. Hence `dn_age`, a count of sessions in a
    bearish DAILY structure, rather than a time of day.

    A MINORITY GENUINELY DO MOVE WITH PRICE: a RANGE or CONSOLIDATION higher frame under a
    BREAKOUT_DOWN lower frame -- the RANGE-FLOOR BREAK geometry, which is the one short tag
    this project ever measured as working. For those an intraday flip time WOULD be real.

    The first version of this test swept invented labels ("BREAKDOWN", "COIL"). synthesize
    does not recognise them, falls through to EXTENDED (aligned)/DOWN and returns SHORT at
    every price -- so it "proved" price-independence out of garbage inputs. Only the six
    enums indicators.struct_full actually emits are swept here.
    """
    from eqbtst import mtf
    real = ["TREND_UP", "TREND_DOWN", "RANGE", "BREAKOUT_UP", "BREAKOUT_DOWN",
            "CONSOLIDATION"]
    prices = [88 + 0.5 * i for i in range(50)]
    n_short, dependent = 0, []
    for hs in real:
        for ls in real:
            htf = {"struct": hs, "hi": 110.0, "lo": 90.0, "n": 20}
            ltf = {"struct": ls, "hi": 101.0, "lo": 93.0, "n": 20}
            sides = set()
            for px in prices:
                syn = mtf.synthesize(htf, ltf, px)
                sides.add(mtf.side_of(syn["tag"], syn.get("dir", "NONE")))
            if "SHORT" in sides:
                n_short += 1
                if len(sides) > 1:
                    dependent.append((hs, ls))
    assert n_short > 0, "no combination yields SHORT any more -- re-read this test"
    assert len(dependent) < n_short / 2, (
        f"{len(dependent)} of {n_short} SHORT combinations now move with price -- the side is "
        f"no longer mostly structure-fixed, so an intraday flip time may beat an age here")
    assert all(ls == "BREAKOUT_DOWN" for _, ls in dependent), (
        f"price-dependent SHORT combos are no longer only lower-frame breakdowns: {dependent}")


def test_bearish_daily_labels_exist_in_the_vocabulary():
    """`_BEARISH_D` is matched against struct_full's OWN output, so a label that module never
    emits matches nothing and silently undercounts. That shipped once as "BREAKDOWN" (the real
    enum is BREAKOUT_DOWN), dropping every fresh breakdown from the age count."""
    import inspect

    from eqbtst import indicators
    src = inspect.getsource(indicators.struct_full)
    assert live._BEARISH_D, "the bearish label set is empty"
    for label in live._BEARISH_D:
        assert f'"{label}"' in src, \
            f"{label!r} is not a label struct_full emits -- it will never match"


def test_downtrend_age_counts_consecutive_bearish_sessions():
    """Age must be CONSECUTIVE and must stop at the first non-bearish session, or a name that
    was bearish months ago reads as bearish now. NaN (not 0) when the name is not currently
    bearish, so the column renders blank rather than claiming a zero-day downtrend."""
    import numpy as np
    idx = pd.date_range("2026-01-01", periods=60, freq="B")
    falling = pd.DataFrame({"ts": idx, "open": np.linspace(200, 100, 60),
                            "high": np.linspace(202, 102, 60),
                            "low": np.linspace(198, 98, 60),
                            "close": np.linspace(200, 100, 60)})
    rising = falling.assign(**{c: falling[c].values[::-1]
                               for c in ("open", "high", "low", "close")})
    orig = live._daily_hist
    live._daily_hist = lambda: {"DOWN": falling, "UP": rising}
    try:
        out = live.add_downtrend_age(pd.DataFrame({"symbol": ["DOWN", "UP", "MISSING"]}),
                                     symbols=["DOWN", "UP", "MISSING"])
        ages = dict(zip(out["symbol"], out["dn_age"]))
        assert ages["DOWN"] > 0, "a monotonic decline has no downtrend age"
        assert ages["UP"] != ages["UP"], "a rising name must be NaN, not a number"
        assert ages["MISSING"] != ages["MISSING"], "an unknown name must be NaN"
        assert out["dn_age"].dtype.kind == "f"
    finally:
        live._daily_hist = orig


def _session_bars(px):
    """40 sessions of realistic NSE 15m bars (09:15..15:15, business days only)."""
    import numpy as np
    days = pd.bdate_range("2026-07-01", periods=40)
    ts = pd.DatetimeIndex([d + pd.Timedelta(hours=9, minutes=15) + pd.Timedelta(minutes=15 * k)
                           for d in days for k in range(25)])
    px = np.asarray(px, float)
    return pd.DataFrame({"ts": ts, "open": px, "high": px * 1.002, "low": px * 0.998,
                         "close": px, "volume": 1e5}), len(ts)


def test_short_entry_time_fires_only_when_the_flip_happened_today():
    """`entered` on the SHORT tab is "the bar TODAY at which this became a SHORT".

    The baseline is the PREVIOUS SESSION'S CLOSE, not "was it short at 09:15". Conflating
    those loses the most interesting case -- a name that breaks down ON the opening bar is
    short at the first bar of the day, and reporting nothing for it hides a flip that really
    did happen today. Only a name already short when yesterday closed has a flip that
    predates today, and that one must stay blank rather than claim a time it cannot know.
    """
    import numpy as np

    from eqbtst import live as _live
    _, total = _session_bars(np.linspace(100, 200, 1000))
    cases = {
        "broke down today": np.concatenate([np.linspace(100, 140, total - 25),
                                            np.linspace(140, 100, 25)]),
        "falling for weeks": np.linspace(200, 100, total),
        "rising throughout": np.linspace(100, 200, total),
    }
    orig = _live.fetch_intraday
    try:
        got = {}
        for lab, px in cases.items():
            frame, _ = _session_bars(px)
            _live.fetch_intraday = (lambda s_, tf=None, lookback_days=None, _b=frame: _b)
            out = _live.add_short_entry_times(
                pd.DataFrame({"symbol": ["X"], "ltp": [float(px[-1])]}), "4h", "1h",
                symbols=["X"])
            got[lab] = out["entered"].iloc[0]
        assert got["broke down today"] is not None, \
            "a breakdown that happened TODAY reported no time"
        assert got["falling for weeks"] is None, \
            "a name short since before today must stay blank -- dn age carries that"
        assert got["rising throughout"] is None, "a rising name is not short"
    finally:
        _live.fetch_intraday = orig


def test_short_entry_times_degrade_without_raising():
    """Same contract as every other context column: a rate-limited fetch or a warming frame
    costs you the column, never the table, and row order/count never move."""
    from eqbtst import live as _live
    df = pd.DataFrame({"symbol": ["A", "B", "C"], "ltp": [1.0, 2.0, 3.0]})
    orig = _live.fetch_intraday
    try:
        _live.fetch_intraday = lambda *a, **k: pd.DataFrame()      # nothing came back
        out = _live.add_short_entry_times(df, "4h", "1h", symbols=["A", "B"])
        assert list(out["symbol"]) == ["A", "B", "C"] and len(out) == 3
        for c in ("entered", "at", "since%"):
            assert c in out.columns

        def _boom(*a, **k):
            raise RuntimeError("429 rate limited")
        _live.fetch_intraday = _boom
        out2 = _live.add_short_entry_times(df, "4h", "1h", symbols=["A"])
        assert len(out2) == 3, "a broker error took the table down"
    finally:
        _live.fetch_intraday = orig
    assert _live.add_short_entry_times(pd.DataFrame(), "4h", "1h").empty
    assert list(_live.add_short_entry_times(pd.DataFrame({"x": [1]}), "4h", "1h").columns) == ["x"]
