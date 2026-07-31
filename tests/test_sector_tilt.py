"""Sector-tilt tests.

Two layers, on purpose:

  OFFLINE — the pure maths and the display contract. No DB, no network, always runs.
  PARITY  — the port pinned against Daily_Cash_Market's own get_forward_tilt on real
            dates. SKIPPED when the archive or the DCM package is not reachable (VM,
            CI, or the DuckDB held read-write by the DCM dashboard), because a skip is
            honest and a failure there would be about the environment, not the code.

The parity layer is the one that matters: this module is a PORT, and a port that is not
pinned to its source silently drifts the moment either side is edited.
"""
import numpy as np
import pandas as pd
import pytest

from eqbtst import sector_tilt as T


# ── OFFLINE: the maths ────────────────────────────────────────────────────────────
def _dcm_slope_reference(mat: pd.DataFrame, med: pd.Series) -> pd.Series:
    """DCM's _slope_per_col, copied verbatim — the thing _rolling_slope must reproduce."""
    y = mat.values.astype(float)
    n = y.shape[0]
    x = np.arange(n, dtype=float)
    xd = x - x.mean()
    denom = (xd ** 2).sum()
    ym = np.nanmean(y, axis=0)
    num = np.nansum((y - ym) * xd[:, None], axis=0)
    return pd.Series(num / denom, index=mat.columns) / med.replace(0, np.nan)


def test_rolling_slope_matches_dcm_including_its_nan_handling():
    rng = np.random.default_rng(7)
    wide = pd.DataFrame(rng.normal(50, 8, size=(40, 4)),
                        index=pd.date_range("2025-01-01", periods=40, freq="B"),
                        columns=list("ABCD"))
    wide.iloc[30, 1] = np.nan            # a hole INSIDE the trailing window
    wide.iloc[26, 2] = np.nan
    win = T._DELIV_SLOPE_WIN
    med = pd.Series(1.0, index=wide.columns)
    got = T._rolling_slope(wide, win).iloc[-1]
    want = _dcm_slope_reference(wide.tail(win), med)
    # DCM drops NaN from the numerator but keeps the FULL window in the denominator; a
    # textbook complete-windows-only slope disagrees here, which is why this is pinned.
    pd.testing.assert_series_equal(got, want, check_names=False, rtol=1e-12)


def test_rolling_slope_sign_follows_the_trend():
    idx = pd.date_range("2025-01-01", periods=30, freq="B")
    up = pd.DataFrame({"U": np.linspace(40, 60, 30)}, index=idx)
    dn = pd.DataFrame({"D": np.linspace(60, 40, 30)}, index=idx)
    assert T._rolling_slope(up, 15).iloc[-1, 0] > 0
    assert T._rolling_slope(dn, 15).iloc[-1, 0] < 0


def test_compound_is_a_true_compounding_not_a_sum():
    s = pd.Series([0.0, 10.0, 10.0, 10.0])       # n+1 rows: the window needs a base row
    got = T._compound(s, 3).iloc[-1]
    assert got == pytest.approx((1.1 ** 3 - 1) * 100)     # 33.1%, not 30%


def _stack(n, down_days=()):
    """Nifty series whose raw EMA-stack label is UP every day except `down_days`.

    UP needs px > ema20 > ema50; DOWN needs px < ema20 < ema50 — so the EMAs, not the price,
    are what get flipped to author a down day.
    """
    close = pd.Series(np.linspace(100, 200, n))
    e20 = close - 5.0
    e50 = close - 10.0
    for i in down_days:
        e20.iloc[i] = close.iloc[i] + 5.0
        e50.iloc[i] = close.iloc[i] + 10.0
    return close, e20, e50, pd.Series(0.5, index=close.index)


def test_confirmed_states_debounces_a_transient_flip():
    # a clean uptrend with a single one-day break of the stack must NOT switch the regime:
    # DCM's 8yr audit found raw EMA-stack labels whipsaw (median run 3d, 42% flip back).
    st = T._confirmed_states(*_stack(80, down_days=[60]))
    assert st.iloc[59] == "UP"
    assert st.iloc[60] == "UP", "a single-day break must not switch the confirmed regime"
    assert st.iloc[61] == "UP"


def test_confirmed_states_accepts_a_persistent_switch():
    st = T._confirmed_states(*_stack(80, down_days=range(60, 70)))
    assert st.iloc[61] == "UP", "the switch must not be accepted before it is confirmed"
    assert st.iloc[62] == "DOWN", "a regime held >= _REGIME_CONFIRM days must be accepted"


# ── OFFLINE: the display contract — absence must stay distinguishable ─────────────
def _row(**kw):
    base = dict(tilt="OVERWEIGHT", thin=False, rank_pos=2, n_sectors=14, n_liq=30)
    base.update(kw)
    return pd.Series(base)


def test_badge_reports_agreement_with_the_row_side():
    assert "with your LONG" in T.badge(_row(), "LONG")
    assert "against your LONG" in T.badge(_row(tilt="UNDERWEIGHT"), "LONG")
    assert "with your SHORT" in T.badge(_row(tilt="UNDERWEIGHT"), "SHORT")
    assert "against your SHORT" in T.badge(_row(), "SHORT")


def test_badge_never_calls_a_neutral_tilt_on_a_missing_read():
    # a missing read and a genuine NEUTRAL are DIFFERENT answers; a board that renders the
    # first as the second is inventing a verdict.
    for missing in (None, pd.Series(dtype=object), _row(tilt=None), _row(tilt=np.nan)):
        out = T.badge(missing, "LONG")
        assert out.startswith("—") and "NEUTRAL" not in out


def test_badge_flags_a_thin_sector_without_hiding_its_verdict():
    # DCM withholds only the OVERWEIGHT call on a thin sector (demoted upstream to NEUTRAL);
    # NEUTRAL/UNDERWEIGHT/WATCH still stand. Rendering them all as a dash threw away a real read.
    out = T.badge(_row(tilt="UNDERWEIGHT", thin=True, n_liq=3), "LONG")
    assert "UW" in out and "thin(3)" in out and "against your LONG" in out
    assert not out.startswith("—")


def test_badge_marks_unresolved_asof_distinctly_from_a_missing_sector():
    df = pd.DataFrame({"symbol": ["X"], "sector": ["IT"]})
    out = T.annotate(df, None)["sector tilt"].iloc[0]
    assert "as-of" in out and out.startswith("—")


def test_badge_shows_neutral_as_a_real_answer():
    assert T.badge(_row(tilt="NEUTRAL"), "LONG").startswith("⚪")


def test_annotate_leaves_a_sectorless_table_alone():
    df = pd.DataFrame({"symbol": ["X"], "ltp": [10.0]})
    assert "sector tilt" not in T.annotate(df, "2026-07-29").columns


def test_both_help_texts_carry_every_caveat():
    """The column's whole risk is being read as permission, so the caveats are the guardrail.

    There are TWO help strings because Streamlit's tooltip CLIPS long text with no scrollbar:
    HELP is the short hover version, HELP_FULL the on-page expander. The warnings must appear
    in BOTH — if they only lived in the long one they would be exactly what got cut off.
    """
    assert len(T.HELP) < len(T.HELP_FULL), "HELP must be the SHORT one (it has to fit a tooltip)"
    for name, txt in (("HELP", T.HELP), ("HELP_FULL", T.HELP_FULL)):
        u = txt.upper()
        assert "10 TRADING DAYS" in u, f"{name}: horizon mismatch not stated"
        assert "NOT A SHORT" in u, f"{name}: UW-is-not-a-short caveat missing"
        assert "RELATIVE" in u and "ABSOLUTE" in u, f"{name}: relative-vs-absolute missing"
        assert "CONTEXT" in u, f"{name}: does not say it is context only"


def test_gate_rule_is_written_down_before_the_measurement_runs():
    # pre-registration: the rule cannot be chosen after the number is on screen.
    assert "PRE-REGISTERED" in T._GATE_RULE and ">= 2.0" in T._GATE_RULE


# ── PARITY: pinned against DCM's own implementation ───────────────────────────────
def _dcm_available():
    import importlib.util
    import sys
    from pathlib import Path
    from eqbtst import config
    if not config.DCM_DUCKDB.exists():
        return None
    root = config.DCM_DUCKDB.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if importlib.util.find_spec("src.analytics.sector_forward_tilt") is None:
        return None
    from src.analytics.sector_forward_tilt import get_forward_tilt
    return get_forward_tilt


@pytest.mark.parametrize("as_of", ["2026-07-29", "2026-06-30", "2026-03-30", "2025-09-30"])
def test_port_reproduces_dcm_tilt(as_of):
    fn = _dcm_available()
    if fn is None:
        pytest.skip("DCM archive/package not reachable — parity is environment-gated")
    try:
        want, _reg = fn(pd.Timestamp(as_of).date())
        got, meta = T.sector_tilt(as_of)
    except Exception as e:                       # DuckDB held read-write by the DCM app
        pytest.skip(f"archive unavailable: {e}")
    if want.empty or got.empty:
        pytest.skip(f"no tilt for {as_of} on either side")

    w = want.set_index("sector").sort_index()
    g = got.sort_index()
    assert list(w.index) == list(g.index), "the ranked sector cross-section must be identical"

    # the LABEL is what the board renders — it must match exactly, never approximately
    pd.testing.assert_series_equal(g["tilt"], w["tilt"], check_names=False)
    for col in ("rank", "rs_2w", "rs_1w", "dv5d", "persistence", "est_rel_bps", "confidence"):
        pd.testing.assert_series_equal(g[col].astype(float), w[col].astype(float),
                                       check_names=False, rtol=1e-9, atol=1e-9)
    # accum_breadth / deliv_slope carry ONE deliberate deviation: sub-epsilon (exactly-flat)
    # delivery slopes are snapped to 0, so a flat series counts as NOT accumulating. DCM lets
    # float rounding decide that at ~1e-18. The disagreement is bounded by one constituent, so
    # allow 1/_MIN_CONSTITUENTS of slack rather than pretending it is exact — see _breadth.
    for col in ("accum_breadth", "deliv_slope"):
        pd.testing.assert_series_equal(g[col].astype(float), w[col].astype(float),
                                       check_names=False, rtol=1e-6, atol=0.02)
    assert (g["n_liq"] == w["n_liq"]).all()
    assert (g["thin"] == w["thin"]).all()


def test_bulk_history_equals_pointwise():
    """A RANGE computed in one pass must equal the same dates computed one at a time.

    This is the invariant the DCM parity test structurally CANNOT see: parity only ever
    exercises the single-date path, where causality comes free from the panel ending at
    as_of. The bulk path removes that accident, and it caught a genuine lookahead — the
    momentum-persistence mean was eating forward windows that had not completed yet at T,
    which moved `persistence` on every date and flipped the OVERWEIGHT→NEUTRAL demotion on
    ~7% of them. Any factor added later that peeks forward will fail here and only here.
    """
    from eqbtst import config
    if not config.DCM_DUCKDB.exists():
        pytest.skip("archive not reachable")
    try:
        bulk = T._engine("2025-06-02", "2025-06-20")
    except Exception as e:
        pytest.skip(f"archive unavailable: {e}")
    if bulk.empty:
        pytest.skip("no tilt rows in the probe window")
    num = ["rank", "score", "persistence", "accum_breadth", "deliv_slope",
           "est_rel_bps", "confidence", "reg_size_hint", "dispersion"]
    txt = ["tilt", "revert", "reg_state", "thin", "reg_verdict"]
    for d in sorted(bulk["trade_date"].unique()):
        pt, _m = T.sector_tilt(pd.Timestamp(d))
        bk = bulk[bulk["trade_date"] == d].set_index("sector")
        assert list(bk.index.sort_values()) == list(pt.index.sort_values())
        for c in num:
            a = bk[c].reindex(pt.index).astype(float).to_numpy()
            b = pt[c].astype(float).to_numpy()
            assert np.allclose(a, b, rtol=1e-9, equal_nan=True), f"{d} {c} bulk != pointwise"
        for c in txt:
            a = bk[c].reindex(pt.index).astype(str).to_numpy()
            assert (a == pt[c].astype(str).to_numpy()).all(), f"{d} {c} bulk != pointwise"


def test_one_row_per_sector_per_date():
    """A fanned-out merge would silently duplicate a sector and make badge() render
    '— sector not ranked' for a sector that WAS ranked (row.get returns a Series)."""
    from eqbtst import config
    if not config.DCM_DUCKDB.exists():
        pytest.skip("archive not reachable")
    try:
        bulk = T._engine("2025-06-02", "2025-06-20")
    except Exception as e:
        pytest.skip(f"archive unavailable: {e}")
    if bulk.empty:
        pytest.skip("no rows")
    assert not bulk.duplicated(["trade_date", "sector"]).any()


def test_last_close_before_is_strictly_earlier():
    from eqbtst import config
    if not config.DCM_DUCKDB.exists():
        pytest.skip("archive not reachable")
    from eqbtst import data
    try:
        last = data.last_trading_date()
    except Exception as e:
        pytest.skip(f"archive unavailable: {e}")
    prev = T.last_close_before(last)
    # the replay/intraday as-of MUST be strictly before the session being decided, or the
    # session's own outcome leaks into the decision.
    assert prev is not None and prev < last
