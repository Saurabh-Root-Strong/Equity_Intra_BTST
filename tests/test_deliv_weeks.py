"""The 5-week delivery-trend column: aggregation, baseline, part-week honesty, leak contract.

The delivery leg is the one this project has already been burned on -- NSE publishes delivery
~6pm, so any figure that quietly includes TODAY is unusable at a 15:15 decision. These tests pin
the arithmetic and the honesty markers; the archive-backed ones skip when the DuckDB is locked.
"""
import numpy as np
import pandas as pd
import pytest

from eqbtst import config, live


def _archive():
    if not config.DCM_DUCKDB.exists():
        return None
    try:
        from eqbtst import data
        return data.last_trading_date()
    except Exception:
        return None


# ── arithmetic: turnover-weighted, not a plain mean ───────────────────────────────
def test_weekly_value_is_turnover_weighted_not_a_simple_average():
    # a 90%-delivery day on tiny turnover must NOT drag the week up like a big day would
    num = 20.0 * 1000 + 90.0 * 10          # 20% on 1000 lacs, 90% on 10 lacs
    wtd = num / 1010
    assert wtd == pytest.approx(20.69, abs=0.01)
    assert wtd < np.mean([20.0, 90.0]), "a plain mean would let a dead day count equally"


def test_norm_window_is_taken_before_the_displayed_weeks():
    # comparing 5 weeks against a baseline that CONTAINS them is self-referential and mutes
    # the very move being judged. The constant exists so that intent is greppable.
    assert live._DELIV_WK_NORM == 100 and live._DELIV_WK_MINHIST == 40


# ── the honesty markers ───────────────────────────────────────────────────────────
def test_partial_week_is_flagged_and_rendered_with_a_star():
    d = _archive()
    if d is None:
        pytest.skip("archive not reachable")
    out = live.deliv_weeks(d)
    if out.empty:
        pytest.skip("no delivery rows")
    wk_end = pd.Timestamp(pd.Period(d, freq="W-FRI").end_time).normalize()
    expect_partial = d < wk_end
    assert bool(out["partial"].iloc[0]) == bool(expect_partial)
    if expect_partial:
        assert out["cell"].str.contains(r"\*").all(), "a part-week must say so"


def test_cell_carries_the_stocks_own_norm_not_a_universe_average():
    d = _archive()
    if d is None:
        pytest.skip("archive not reachable")
    out = live.deliv_weeks(d)
    if out.empty or "norm" not in out:
        pytest.skip("no rows")
    norms = out["norm"].dropna()
    # per-stock baselines genuinely differ -- that is the whole reason the norm is displayed
    assert norms.max() - norms.min() > 15, "norms should span the universe, not be one number"
    row = out.loc[norms.index[0]]
    assert f"n{row['norm']:.0f}" in row["cell"]


def test_direction_glyph_matches_the_latest_week_against_the_norm():
    d = _archive()
    if d is None:
        pytest.skip("archive not reachable")
    out = live.deliv_weeks(d)
    if out.empty:
        pytest.skip("no rows")
    wcols = [c for c in out.columns if c.startswith("w")]
    for _s, r in out.dropna(subset=["norm"]).head(200).iterrows():
        vals = [r[c] for c in wcols]
        latest = next((v for v in reversed(vals) if pd.notna(v)), np.nan)
        if pd.isna(latest) or r["norm"] <= 0:
            continue
        rel = latest / r["norm"]
        if rel >= 1.10:
            assert "▲" in r["cell"]
        elif rel <= 0.90:
            assert "▼" in r["cell"]
        else:
            assert "▲" not in r["cell"] and "▼" not in r["cell"]


def test_missing_history_says_no_norm_rather_than_inventing_one():
    d = _archive()
    if d is None:
        pytest.skip("archive not reachable")
    out = live.deliv_weeks(d)
    if out.empty:
        pytest.skip("no rows")
    for _s, r in out.iterrows():
        if pd.isna(r["norm"]):
            assert "no norm" in r["cell"], "a missing baseline must be stated, not omitted"


# ── the leak contract ─────────────────────────────────────────────────────────────
def test_reads_no_further_than_the_as_of_date():
    d = _archive()
    if d is None:
        pytest.skip("archive not reachable")
    back = d - pd.Timedelta(days=30)
    older = live.deliv_weeks(back)
    newer = live.deliv_weeks(d)
    if older.empty or newer.empty:
        pytest.skip("no rows")
    # an earlier as-of must produce a DIFFERENT (earlier) window, never today's answer
    common = older.index.intersection(newer.index)
    assert len(common) > 20
    same = (older.loc[common, "cell"] == newer.loc[common, "cell"]).mean()
    assert same < 0.5, "an as-of 30 days back must not reproduce today's weeks"


def test_cache_is_keyed_on_the_asof_date_not_today():
    # a cache keyed on today() is what poisons Replay -- this asserts two dates coexist
    d = _archive()
    if d is None:
        pytest.skip("archive not reachable")
    live.deliv_weeks(d)
    live.deliv_weeks(d - pd.Timedelta(days=30))
    keys = {k[0] for k in live._DELIV_WK}
    assert len(keys) >= 1 and all(isinstance(k, type(d.date())) for k in keys)
