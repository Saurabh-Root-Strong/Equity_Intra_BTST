"""F&O positioning columns — the guards that matter, not the labels themselves.

The labels are DCM's; this module only reads them. What can break here is the plumbing:
the archive lock, the as-of discipline, and graceful degradation.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from eqbtst import config, fno


def test_dcm_is_imported_read_only():
    """THE one that must never regress.

    DCM's ConnectionManager opens DuckDB READ-WRITE unless CLOUD_MODE is set, and DuckDB is
    many-readers-OR-one-writer. Importing it without the flag takes the EXCLUSIVE lock on the
    shared archive -- locking out DCM's own dashboard, its nightly sync, and this board's own
    readers. This board must only ever READ.
    """
    fno._dcm()
    assert os.environ.get("CLOUD_MODE", "").lower() == "true"
    if not config.DCM_DUCKDB.exists():
        pytest.skip("archive not reachable")
    import sys
    sys.path.insert(0, str(config.DCM_DUCKDB.parent.parent))
    try:
        from src.data.connection import _read_only
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"DCM not importable: {e}")
    assert _read_only() is True, "DCM would take a WRITE lock on the shared archive"


def test_never_reads_past_the_as_of_date():
    """The board can be pointed at a past close, or Replay. Serving the NEWEST F&O row there
    would put tomorrow's positioning beside today's price."""
    if not config.DCM_DUCKDB.exists():
        pytest.skip("archive not reachable")
    from eqbtst import data
    try:
        last = data.last_trading_date()
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"archive unavailable: {e}")
    back = pd.Timestamp(last) - pd.Timedelta(days=90)
    _, meta = fno.positioning(back)
    if meta["date_used"] is None:
        pytest.skip("no F&O data that far back")
    assert meta["date_used"] <= back.date()


def test_degrades_to_dashes_rather_than_raising():
    """Every failure path -- no archive, pre-retention date, junk input -- must return the
    columns filled with an em dash. A board that raises is worse than one that says nothing."""
    for bad in ["1990-01-01", None, "not-a-date"]:
        frame, meta = fno.positioning(bad)
        assert list(frame.columns) == fno.COLS
        assert meta["date_used"] is None or meta["n"] >= 0
    df = pd.DataFrame({"symbol": ["AAA", "BBB"]})
    out = fno.positioning("1990-01-01")[0]
    assert out.empty
    ann = fno.annotate(df, "1990-01-01")
    assert list(ann[fno.COLS].iloc[0]) == ["—"] * len(fno.COLS)


def test_names_without_fno_get_a_dash_not_a_guess():
    """~60 of this board's names have no F&O -- NSE retired the contracts at an expiry.
    A dash is the correct answer; anything else invents positioning that does not exist."""
    if not config.DCM_DUCKDB.exists():
        pytest.skip("archive not reachable")
    from eqbtst import data
    try:
        last = data.last_trading_date()
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"archive unavailable: {e}")
    df = pd.DataFrame({"symbol": ["KOTAKBANK", "ZZZNOTREAL"]})
    out = fno.annotate(df, last)
    assert list(out[out.symbol == "ZZZNOTREAL"][fno.COLS].iloc[0]) == ["—"] * len(fno.COLS)
    assert set(out.columns) >= set(fno.COLS)


def test_annotate_preserves_row_count_and_order():
    if not config.DCM_DUCKDB.exists():
        pytest.skip("archive not reachable")
    from eqbtst import data
    try:
        last = data.last_trading_date()
    except Exception as e:                                   # noqa: BLE001
        pytest.skip(f"archive unavailable: {e}")
    df = pd.DataFrame({"symbol": ["WIPRO", "KOTAKBANK", "ZZZ"], "x": [1, 2, 3]})
    out = fno.annotate(df, last)
    assert len(out) == 3 and list(out["symbol"]) == ["WIPRO", "KOTAKBANK", "ZZZ"]
    assert list(out["x"]) == [1, 2, 3]


def test_fno_block_is_in_BOTH_column_lists_at_the_same_place():
    """The board builds its columns TWICE -- `light_cols` before any filter, `_sc` after one.

    Adding a column to only one is a bug this board has now shipped twice: first `deliv trend`
    (it JUMPED across the table when a filter was applied), then the F&O block (it VANISHED
    entirely the moment "Has room" was selected). Both lists live in dashboard.py and nothing
    but this test ties them together.

    Asserted on the SOURCE, because building the real frames needs a live broker session.
    """
    import io
    import re
    from pathlib import Path
    src = io.open(Path(__file__).resolve().parent.parent / "eqbtst" / "dashboard.py",
                  encoding="utf-8").read()

    # both lists must anchor the F&O block immediately after `deliv trend`
    pat = r'"deliv trend"\]\s*\+\s*fno\.COLS'
    hits = re.findall(pat, src)
    assert len(hits) >= 2, (
        f"expected the F&O block anchored after 'deliv trend' in BOTH the pre-filter and the "
        f"enriched column lists, found {len(hits)}"
    )
    # and the enriched frame must actually carry them, or the columns render empty
    # fno.annotate may be WRAPPED by another context annotator (arb.annotate adds the carry
    # column around the same frame), so match it anywhere on the right-hand side rather than
    # pinning it to the first call. What must stay true is that `enr` is annotated at all.
    assert re.search(r"enr\s*=\s*(?:\w+\.annotate\(\s*)*fno\.annotate\(", src), \
        "enriched frame is never annotated -- the four columns would render blank under a filter"


def test_no_side_path_also_carries_the_fno_block():
    """The no-preset render path has its own column list too."""
    import io
    import re
    from pathlib import Path
    src = io.open(Path(__file__).resolve().parent.parent / "eqbtst" / "dashboard.py",
                  encoding="utf-8").read()
    assert re.search(r'\["deliv trend"\]\s*\+\s*fno\.COLS', src), \
        "the no-preset column list dropped the F&O block"


def test_every_table_that_shows_the_fno_block_can_actually_render_it():
    """A column needs THREE things wired, in three different places, to appear:

        1. its name in that table's column list
        2. the frame annotated so the data exists
        3. FNO_COLS in that table's column_config, or it renders with no tooltip

    Miss (1) and the column vanishes -- which is exactly what happened on "Has room": the
    block was in the pre-filter list and not the enriched one. Miss (2) and it renders blank.
    Miss (3) and it renders bare. None of these fail loudly; the table just looks different.

    So pin the COUNTS. Adding a table without the block, or a list without a config, moves a
    number here and forces the author to decide deliberately rather than discover it later
    from a screenshot.
    """
    import io
    import re
    from pathlib import Path
    raw = io.open(Path(__file__).resolve().parent.parent / "eqbtst" / "dashboard.py",
                  encoding="utf-8").read()
    # STRIP COMMENTS FIRST. The first version of this test counted prose: a comment that
    # merely MENTIONED fno.COLS inflated the total, so editing documentation broke the test.
    # A tripwire that fires on comments trains you to ignore it.
    src = "\n".join(ln.split("#", 1)[0] for ln in raw.splitlines())
    lists = len(re.findall(r"fno\.COLS", src))
    annot = len(re.findall(r"fno\.annotate\(", src))
    cfgs = len(re.findall(r"\*\*FNO_COLS", src))
    assert (lists, annot, cfgs) == (9, 6, 10), (
        f"F&O column wiring moved: {lists} fno.COLS references / {annot} annotate calls / "
        f"{cfgs} configs (expected 9 / 6 / 10). If you ADDED a table, wire all three and "
        f"update this count. If a number DROPPED, a table just lost the block silently."
    )


def test_replay_annotates_with_the_prior_close_not_the_replayed_session():
    """The F&O bhavcopy for session D publishes AFTER D's close.

    A replay reconstructs a decision taken INSIDE session D, so reading D's own bhavcopy
    feeds that session's outcome back into the decision -- the same lookahead class that
    retracted two 'edges' in this stack. The replay must anchor on the prior close.
    """
    import io
    import re
    from pathlib import Path
    src = io.open(Path(__file__).resolve().parent.parent / "eqbtst" / "dashboard.py",
                  encoding="utf-8").read()
    assert re.search(r"fno\.annotate\(bd,\s*_asof_replay\)", src), \
        "replay must annotate with _asof_replay (prior close), never rdate"
    assert not re.search(r"fno\.annotate\([^)]*\brdate\b", src), \
        "replay is annotating with the replayed session itself — lookahead"


def test_column_config_is_derived_from_COLS_not_hand_listed():
    """FNO_COLS must be BUILT from fno.COLS, not typed out.

    The two disagreed once already: the column list gained entries the config never got, so
    the columns rendered with no tooltip. Deriving one from the other makes that unrepresentable
    -- and it is why growing 4 -> 8 columns needed no config edit at all.
    """
    import io
    import re
    from pathlib import Path
    from eqbtst import fno
    src = io.open(Path(__file__).resolve().parent.parent / "eqbtst" / "dashboard.py",
                  encoding="utf-8").read()
    assert re.search(r"for c in fno\.COLS", src), "FNO_COLS is no longer derived from fno.COLS"
    # and every column must have a tooltip mapped, or it renders bare
    assert re.search(r"_FNO_HELP\s*=\s*\{", src)
    for c in fno.COLS:
        assert f'"{c}"' in src, f"{c} has no entry in the help map"
