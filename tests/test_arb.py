"""
test_arb.py — the carry/arbitrage context layer.

The bar these tests defend is not "does it compute". It is the three ways this module
could quietly go wrong: it could take a WRITE lock on the shared archive (which locks out
the DCM dashboard and this board's own readers), it could read a bar later than the as-of
it was handed (the lookahead class this project has already shipped once and had to fix),
and it could drift out of step with the tables it feeds — the F&O block already grew a
wiring tripwire for exactly that reason, and `carry` rides in the same lists.
"""
from __future__ import annotations

import pathlib
import re

import pandas as pd
import pytest

from eqbtst import arb, data


@pytest.fixture(scope="module")
def last():
    return data.last_trading_date()


def test_never_opens_the_archive_read_write():
    """DuckDB is many-readers-OR-one-writer. A write handle here would lock out the DCM
    dashboard, its nightly sync, and every other reader on this board. arb.py must go
    through data._connect, which pins read_only=True — never open duckdb itself."""
    src = pathlib.Path(arb.__file__).read_text(encoding="utf-8")
    assert "duckdb.connect" not in src, \
        "arb.py opens DuckDB directly -- route through data._connect (read_only=True)"
    assert "data._connect" in src


def test_carry_reads_only_the_asof_session(last):
    """The whole module is as-of by construction because every query pins ONE trade_date.
    If a `<=` ever creeps in, this catches it: an as-of far in the past must not return
    today's names."""
    src = pathlib.Path(arb.__file__).read_text(encoding="utf-8")
    assert "trade_date = ?" in src, "the per-symbol query no longer pins a single session"
    back = pd.Timestamp(last) - pd.Timedelta(days=400)
    old, new = arb.carry(back), arb.carry(last)
    if not old.empty and not new.empty:
        assert not old.equals(new), "an as-of 400 days back returned the current session"


def test_pre_history_asof_degrades_quietly(last):
    """The F&O bhavcopy starts 2024-07-24. Before that there is no carry at all, and the
    board must render an em-dash rather than raise -- a context column may never be able
    to take the page down."""
    assert arb.carry("1990-01-01").empty
    df = pd.DataFrame({"symbol": ["RELIANCE", "INFY"]})
    out = arb.annotate(df, "1990-01-01")
    assert list(out["carry"]) == ["—", "—"]
    m = arb.market("1990-01-01")
    assert m["n"] == 0


def test_nat_asof_is_not_silently_a_date():
    """pd.Timestamp(None) returns NaT rather than raising, so a bad as-of flows straight
    into a date comparison and reads as a valid session. Both entry points must reject it."""
    assert arb.carry(None).empty
    assert arb.market(None)["n"] == 0


def test_non_fno_names_get_a_dash_not_a_gap(last):
    """~60 board names have no futures at all (SEBI eligibility retired them). A blank is
    the correct answer, not a missing value that renders as NaN."""
    df = pd.DataFrame({"symbol": ["RELIANCE", "ZZZNOTREAL"]})
    out = arb.annotate(df, last)
    assert out.loc[out.symbol == "ZZZNOTREAL", "carry"].iloc[0] == "—"


def test_expiry_week_is_excluded(last):
    """Inside the last two sessions the basis converges to zero mechanically, so the
    annualised read is a division by almost nothing. _MIN_DTE guards it."""
    d = arb.carry(last)
    if not d.empty:
        assert (d.dte >= arb._MIN_DTE).all()


def test_exdiv_and_confirm_are_mutually_exclusive(last):
    """A name flagged as a likely ex-date must never also be a ✅ confirm. The confirm is
    an instruction to carry the position overnight; the flag says the cash price is about
    to drop by the dividend. Shipping both on one row would be the worst possible read."""
    d = arb.carry(last)
    if not d.empty:
        assert not (d.exdiv & d.confirm).any()
        assert not d[d.exdiv].tag.str.startswith("✅").any()


def test_carry_is_wired_into_every_table_that_shows_fno():
    """`carry` rides in the same column lists as the F&O block, and each needs THREE
    things: the name in the list, the frame annotated so the data exists, and a
    column_config so the tooltip renders. Adding a table and wiring only one or two is the
    failure this pins -- it renders as a blank column with no error.

    Counted against the F&O block rather than as absolute numbers, so the two can only
    ever move together. Comments are stripped first: the F&O tripwire was broken once by a
    comment that merely MENTIONED the symbol it was counting."""
    src = pathlib.Path(__file__).parent.parent / "eqbtst" / "dashboard.py"
    txt = "\n".join(ln for ln in src.read_text(encoding="utf-8").splitlines()
                    if not ln.lstrip().startswith("#"))
    fno_lists = len(re.findall(r'fno\.COLS\s*\+', txt))
    arb_lists = txt.count('+ ["carry"]')
    assert arb_lists == fno_lists, (
        f'{arb_lists} `["carry"]` entries against {fno_lists} fno.COLS lists -- a table '
        f"shows the F&O block without the carry column (or the reverse)")
    assert len(re.findall(r"arb\.annotate\(", txt)) == len(re.findall(r"fno\.annotate\(", txt)), \
        "a frame is annotated with F&O positioning but not with carry -- blank column"
    assert txt.count("**ARB_COLS") == txt.count("**FNO_COLS"), \
        "a table renders the carry column with no column_config -- tooltip is lost"


def test_carry_annotate_uses_the_same_asof_as_fno():
    """Replay must annotate with the PRIOR close, never the session being replayed --
    that is the lookahead already fixed once in this project. Pinning arb to whatever
    as-of fno was handed makes the two impossible to drift apart."""
    src = pathlib.Path(__file__).parent.parent / "eqbtst" / "dashboard.py"
    txt = src.read_text(encoding="utf-8")
    # GREEDY between the two calls on purpose: the inner frame expression carries as-of
    # arguments of its own (`_dw(df_, _ASOF_LIVE, _hz)`), and a lazy match stops at the
    # first of them instead of at fno.annotate's own last argument.
    for m in re.finditer(r"arb\.annotate\(fno\.annotate\([^\n]+,\s*(\w+)\),\s*(\w+)\)", txt):
        assert m.group(1) == m.group(2), \
            f"carry annotated as-of {m.group(2)} but F&O as-of {m.group(1)}"
    assert not re.search(r"arb\.annotate\([^)]*\brdate\b", txt), \
        "replay must annotate carry with _asof_replay (prior close), never rdate"


def test_thresholds_are_named_constants_not_literals():
    """The ex-dividend threshold and the confirm quintile were both chosen by measurement
    (0.25% at the knee of a precision/recall sweep; quintile 5 because widening to 4-5
    drops the read under the cost floor). Inlining either hides that they are tunable and
    invites a silent re-tune."""
    assert arb._EXDIV_THR == 0.25
    assert arb._CONFIRM_Q == 5
    assert arb._MIN_DTE == 3


class _StubST:
    """Records every Streamlit call instead of drawing. render_page takes `st` as an
    argument precisely so it can be exercised headless -- the page is the largest block of
    new code here and would otherwise only ever be tested by looking at it."""

    def __init__(self, parent=None):
        # Children (columns, sidebar) SHARE the root's log -- st.metric is called on the
        # column objects, not on `st`, so a child with its own list would hide every metric
        # on the page from the assertions below.
        self.calls = parent.calls if parent else []
        self.text = parent.text if parent else []

    # `with st.expander(...)` needs a context manager, so the stub is one and returns
    # itself from every layout call.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        # `st.column_config.NumberColumn(...)` is attribute access on an attribute, not a
        # call -- returning a plain function here makes it an AttributeError. Hand back
        # another stub so the whole namespace is walkable.
        if name in ("column_config", "sidebar"):
            return _StubST(self)

        def f(*a, **k):
            if name in ("expander", "container", "spinner", "form", "empty"):
                self.calls.append(name)
                return _StubST(self)
            self.calls.append(name)
            self.text.extend(x for x in a if isinstance(x, str))
            if name == "columns":
                return [_StubST(self) for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))]
            if name == "multiselect":
                return k.get("default", [])
            if name == "radio":
                return a[1][0] if len(a) > 1 else "All"
            return None
        return f


def test_page_renders_end_to_end(last):
    """Exercises every branch of the page against the real archive: metrics, the chart,
    the triage table, the evidence table and the per-name carry board."""
    stub = _StubST()
    arb.render_page(last, stub)
    for required in ("title", "dataframe", "metric", "subheader"):
        assert required in stub.calls, f"page never called st.{required}"
    assert stub.calls.count("dataframe") >= 3, \
        "page should draw the triage, the evidence and the carry board"


def test_page_survives_a_session_with_no_futures():
    """Before 2024-07-24 there is no F&O bhavcopy. The page must still render its triage
    and say so, not raise -- an as-of on the cash spine's early history is a legal input."""
    stub = _StubST()
    arb.render_page("1990-01-01", stub)
    assert "title" in stub.calls


def test_triage_covers_the_whole_menu_with_known_verdicts():
    """All 21 classic strategies, each with a status from the fixed vocabulary. A new row
    with a typo'd status would silently vanish from the page's filter."""
    s = arb.strategies()
    assert len(s) == 21
    assert list(s["#"]) == list(range(1, 22))
    assert set(s["_k"]) <= set(arb._STATUS_ICON)
    # Every verdict must carry a reason. Two rows are legitimately one-liners because they
    # DEFER to another row (#5 needs #4's ETF leg, #11 shares #10's execution gate), so a
    # cross-reference counts as reasoning; anything else must argue its own case.
    thin = s[(s.why.str.len() <= 40) & ~s.why.str.contains(r"#\d+", regex=True)]
    assert thin.empty, f"triage rows with no reasoning: {list(thin.strategy)}"
    # the survivor must be exactly one, and it must be the dividend row -- inverted into
    # the ex-dividend exclusion rather than shipped as a trade
    ctx = s[s._k == "CONTEXT"]
    assert len(ctx) == 1 and "Dividend" in ctx.strategy.iloc[0]


def test_arbitrage_lane_is_wired_and_stops_the_other_boards():
    """The page is its own sidebar lane. Without st.stop() the live boards would draw
    underneath it, and the lane needs no Fyers token so it must not fall through into the
    token gate."""
    src = pathlib.Path(__file__).parent.parent / "eqbtst" / "dashboard.py"
    txt = src.read_text(encoding="utf-8")
    assert '"🧮 Arbitrage"' in txt, "the Arbitrage lane is not in the sidebar radio"
    m = re.search(r'if tf == "🧮 Arbitrage":(.+?)\n\nif tf == "Intraday"', txt, re.S)
    assert m, "the Arbitrage lane block is missing or moved"
    assert "arb.render_page(date, st)" in m.group(1)
    assert "st.stop()" in m.group(1), "the lane never stops -- live boards draw underneath"


def test_the_withdrawn_oracle_number_never_comes_back():
    """+24.35bps for the confirm cell was measured by removing ex-dividend names using the
    NEXT session's basis snap-back -- not knowable at decision time. The causal rule the
    board ships gives +20.91bps, which does NOT clear the 22bps cost floor.

    That distinction is the whole difference between a tie-breaker and a trade, and it is
    the exact failure mode ('scored with data the decision would not have had') behind two
    earlier retractions in this book. So pin it: the shippable number must be present, the
    withdrawn one may appear ONLY as a retraction, and nothing may claim the floor is
    cleared."""
    for f in (pathlib.Path(arb.__file__),
              pathlib.Path(__file__).parent.parent / "eqbtst" / "dashboard.py"):
        src = f.read_text(encoding="utf-8")
        if "24.35" in src or "24.4bps" in src:
            assert re.search(r"withdrawn|earlier (revision|version)", src), (
                f"{f.name} quotes the oracle-scored 24.35bps without marking it withdrawn")
        # Look for an AFFIRMATIVE "clears the floor" claim. The negation sits BEFORE the
        # verb in normal English ("no combination clears the 22bps cost floor"), so a
        # lookahead cannot see it -- inspect the preceding window instead.
        # `it\b` matters: without the boundary this also fires inside "clears ITS cost",
        # which appears in the retraction paragraph itself.
        for m in re.finditer(r"\bclears (it\b|the 22bps cost floor)", src):
            before = src[max(0, m.start() - 60):m.start()].lower()
            assert re.search(r"\b(no|not|never|neither|nor|without)\b", before), (
                f"{f.name} affirmatively claims the cost floor is cleared -- no measured "
                f"cell does. Context: ...{src[max(0, m.start()-60):m.end()+20]}...")
    src = pathlib.Path(arb.__file__).read_text(encoding="utf-8")
    assert "+20.91bps" in src, "the shippable causal number is gone from the module"
    assert "TIE-BREAKER, NOT A TRIGGER" in src


def test_page_answers_what_do_i_do_before_showing_any_number():
    """The first version of this page showed a carry table and never said what to do with
    it, which is what prompted the rewrite. The instruction block must come BEFORE the
    carry board, and it must name both actions and the no-short finding."""
    src = pathlib.Path(arb.__file__).read_text(encoding="utf-8")
    i_do = src.index("SO WHAT DO I ACTUALLY DO")
    i_board = src.index("Today's carry board")
    assert i_do < i_board, "the carry board is drawn before the instructions"
    for claim in ("REMOVE the name", "TIE-BREAKER", "Do not short low carry",
                  "Do not hold past the open"):
        assert claim in src, f"the instruction block no longer states: {claim}"


def test_module_states_it_is_not_a_tradeable_arbitrage():
    """The single most dangerous way to read this module is as a menu of arbitrage trades
    that can be put on from a retail cash account. They cannot. The docstring carries the
    arithmetic; this pins that it stays there."""
    src = pathlib.Path(arb.__file__).read_text(encoding="utf-8")
    assert "NOT AN ARBITRAGE YOU CAN PUT ON" in src
    assert "cost floor" in src
