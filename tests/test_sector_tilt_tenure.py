"""The `Days -> N` tenure badge on the sector tilt column.

Tenure is computed from THIS module's own `_engine` history, not from DCM and not by
porting DCM's rebuild path. Upstream needs that second path only because its runtime keeps
the as-of row alone and cannot reproduce the WATCH / thin overlays without extra queries —
which is why it suppresses the UNDERWEIGHT streak when the breadth history is missing.
`_engine(start, end)` is already a date-range panel that models both overlays, so the
streak is an exact groupby over data we can have for +0.79s.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eqbtst import sector_tilt as ST


def _hist(**cols) -> pd.DataFrame:
    """Long-format (trade_date, sector, tilt) history from {sector: [oldest..newest]}."""
    n = len(next(iter(cols.values())))
    dates = pd.bdate_range("2026-01-01", periods=n)
    rows = [{"trade_date": d, "sector": sec, "tilt": v}
            for sec, seq in cols.items() for d, v in zip(dates, seq)]
    return pd.DataFrame(rows)


def test_counts_only_the_trailing_run():
    h = _hist(A=["OVERWEIGHT", "NEUTRAL", "OVERWEIGHT", "OVERWEIGHT", "OVERWEIGHT"])
    out = ST._tenure_days(h, pd.Series({"A": "OVERWEIGHT"}))
    assert out["A"] == 3          # the earlier OW does NOT carry across the NEUTRAL


def test_neutral_and_watch_are_never_established():
    """NEUTRAL is the residual bucket — every overlay demotes into it, so a NEUTRAL
    streak measures 'nothing else fired', not a call being held."""
    h = _hist(A=["NEUTRAL"] * 5, B=["WATCH"] * 5)
    out = ST._tenure_days(h, pd.Series({"A": "NEUTRAL", "B": "WATCH"}))
    assert out["A"] == 0 and out["B"] == 0


def test_underweight_is_established_here():
    """Unlike upstream, which suppresses UW when it cannot reproduce WATCH."""
    h = _hist(A=["UNDERWEIGHT"] * 4)
    assert ST._tenure_days(h, pd.Series({"A": "UNDERWEIGHT"}))["A"] == 4


def test_a_gap_breaks_the_streak():
    """A sector that fell out of the ranked cross-section did not hold the call through it.

    B carries every date so the SESSION still exists in the panel — that is the production
    shape, and it is what makes A's gap a real NaN cell rather than a vanished row. A date
    absent for EVERY sector is a non-trading day and must NOT break a streak; a sector
    absent from a date other sectors have must.
    """
    h = _hist(A=["UNDERWEIGHT", "UNDERWEIGHT", np.nan, "UNDERWEIGHT"],
              B=["NEUTRAL"] * 4)
    assert ST._tenure_days(h, pd.Series({"A": "UNDERWEIGHT"}))["A"] == 1


def test_a_missing_session_does_not_break_a_streak():
    """No row for anyone = no session. The call was held across it."""
    h = _hist(A=["UNDERWEIGHT"] * 4)
    assert ST._tenure_days(h, pd.Series({"A": "UNDERWEIGHT"}))["A"] == 4


def test_unknown_sector_and_empty_history_are_zero_not_error():
    assert ST._tenure_days(_hist(A=["OVERWEIGHT"]), pd.Series({"ZZ": "OVERWEIGHT"}))["ZZ"] == 0
    out = ST._tenure_days(pd.DataFrame(), pd.Series({"A": "OVERWEIGHT"}))
    assert out["A"] == 0


@pytest.mark.parametrize("days,expect", [(0, False), (1, True), (7, True)])
def test_badge_renders_days_only_when_established(days, expect):
    row = pd.Series({"tilt": "OVERWEIGHT", "rank_pos": 1, "n_sectors": 24,
                     "days_in_tilt": days})
    txt = ST.badge(row, "LONG")
    assert ("Days" in txt) is expect
    # 0 must never print as "Days -> 0" — that would assert the call was made today
    assert "Days \u2192 0" not in txt


def test_badge_flags_a_one_day_call_as_new():
    row = pd.Series({"tilt": "UNDERWEIGHT", "rank_pos": 20, "n_sectors": 24,
                     "days_in_tilt": 1})
    assert "Days \u2192 1 NEW" in ST.badge(row, "SHORT")


def test_badge_clamps_a_very_long_streak():
    row = pd.Series({"tilt": "OVERWEIGHT", "rank_pos": 1, "n_sectors": 24,
                     "days_in_tilt": ST._TENURE_LOOKBACK + 40})
    assert f"Days \u2192 {ST._TENURE_LOOKBACK}+" in ST.badge(row, "LONG")


def test_badge_survives_a_frame_without_the_column():
    """An older cached frame predating this feature must not crash the column."""
    row = pd.Series({"tilt": "OVERWEIGHT", "rank_pos": 1, "n_sectors": 24})
    assert "Days" not in ST.badge(row, "LONG")


# ── regressions found in the post-ship audit ──────────────────────────────────────────
def test_a_non_trading_as_of_returns_nothing_not_the_previous_session():
    """`_tilt_cached` slices the as-of row out of the tenure history. That slice must be
    the REQUESTED date: on a Sunday the history's last row is Friday's, and returning it
    would answer a question nobody asked, stamped with a date it does not describe
    (meta["as_of"] carries the requested key). The single-date engine it replaced returned
    empty here, which is what makes the board print "— tilt unavailable".
    """
    import pandas as pd
    for weekend in ("2026-08-29", "2026-08-30"):        # Sat, Sun
        df, meta = ST.sector_tilt(weekend)
        assert df.empty, f"{weekend} is not a session; it must not borrow Friday's tilt"
        assert meta["available"] is False
