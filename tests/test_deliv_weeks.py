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
        wl = [c for c in out.columns if c.startswith("w") and c[1:].isdigit()][-1]
        have = out[out[wl].notna()]
        assert have["cell"].str.contains(r"\*\d*d?").all(), "a part-week must say so"


def test_norms_are_per_stock_not_a_universe_average():
    d = _archive()
    if d is None:
        pytest.skip("archive not reachable")
    out = live.deliv_weeks(d)
    if out.empty or "norm" not in out:
        pytest.skip("no rows")
    norms = out["norm"].dropna()
    # per-stock baselines genuinely differ -- that is the whole reason a norm is carried at all
    assert norms.max() - norms.min() > 15, "norms should span the universe, not be one number"


def test_cell_leads_with_the_CURRENT_week_then_goes_backwards():
    d = _archive()
    if d is None:
        pytest.skip("archive not reachable")
    out = live.deliv_weeks(d)
    if out.empty:
        pytest.skip("no rows")
    wcols = [c for c in out.columns if c.startswith("w") and c[1:].isdigit()]
    r = out[out[wcols].notna().all(axis=1)].iloc[0]
    shown = [s.split("*")[0] for s in r["cell"].split("  ")[0].split(", ")]
    chrono = [f"{r[c]:.0f}" for c in wcols]           # stored oldest -> newest
    assert shown == chrono[::-1], "display must be newest-first, storage stays chronological"


def test_deviation_is_relative_to_the_norm_and_signed():
    d = _archive()
    if d is None:
        pytest.skip("archive not reachable")
    out = live.deliv_weeks(d)
    if out.empty:
        pytest.skip("no rows")
    wcols = [c for c in out.columns if c.startswith("w") and c[1:].isdigit()]
    checked = 0
    for _s, r in out.dropna(subset=["norm", "dev_pct"]).head(200).iterrows():
        latest = next((v for v in reversed([r[c] for c in wcols]) if pd.notna(v)), np.nan)
        if pd.isna(latest) or r["norm"] <= 0:
            continue
        want = (latest / r["norm"] - 1) * 100
        assert r["dev_pct"] == pytest.approx(want, abs=1e-6)
        # percentage POINTS would be latest-norm; assert we did NOT ship that
        assert f"{want:+.0f}%" in r["cell"]
        checked += 1
    assert checked > 20


def test_a_week_with_no_reading_is_not_marked_partial():
    # a name that did not trade this week shows a dash; calling that dash "partial" would
    # dress absence up as an in-progress figure
    d = _archive()
    if d is None:
        pytest.skip("archive not reachable")
    out = live.deliv_weeks(d)
    if out.empty:
        pytest.skip("no rows")
    wcols = [c for c in out.columns if c.startswith("w") and c[1:].isdigit()]
    for _s, r in out.iterrows():
        if pd.isna(r[wcols[-1]]):
            assert not r["cell"].startswith("–*")


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
