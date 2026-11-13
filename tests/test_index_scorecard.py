"""The scorecard: point-in-time readings from the log, forward targets, locked grading."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from eqbtst import index_fno as f
from eqbtst import index_scorecard as sc
from test_index_fno import EXP, _session

DAY = dt.date(2026, 11, 6)


def test_at_takes_last_reading_before_the_stamp_and_sees_holes():
    rows = _session(30, gap_after=10)                  # 09:16..09:26, then 10:27..10:45
    r = sc._at(rows, dt.datetime(2026, 11, 6, 9, 20))
    assert r["t"] == "2026-11-06T09:20:00"
    assert sc._at(rows, dt.datetime(2026, 11, 6, 10, 0)) is None     # 34 min stale = hole
    assert sc._at(rows, dt.datetime(2026, 11, 6, 9, 10)) is None     # before the first
    assert sc._at(rows, dt.datetime(2026, 11, 6, 10, 30))["t"] == "2026-11-06T10:30:00"


def test_signals_are_point_in_time_and_signed_by_lore():
    rows = _session(40)
    ks = f.window([k[0] for k in rows[-1]["k"]], 25000.0)
    sig = sc.signals_at(rows[:20], rows[19], ks, 25000.0, None, False)
    assert set(sig) == set(sc.SIGNALS)
    # 25000 PE OI builds while its premium... the session helper's chg is +1 (price up) so
    # the put is 'Long build-up' = bearish for the index; calls at 25200 build with chg +1 =
    # bullish. Strike says is defined and integer; cp_doi: puts added more than calls -> +
    assert isinstance(sig["best_says"], int)
    assert sig["cp_doi"] > 0
    assert sig["pcr_ladder"] is not None and sig["mp_gap"] is not None
    assert sig["checked_says"] is None or isinstance(sig["checked_says"], int)
    # the same stamp read with MORE rows must not change (nothing after the stamp is used)
    sig2 = sc.signals_at(rows[:20], rows[19], ks, 25000.0, None, False)
    assert sig2 == sig
    # expiry day: the IV check is off
    sig3 = sc.signals_at(rows[:20], rows[19], ks, 25000.0, None, True)
    assert sig3["checked_says"] is None and sig3["chain_iv"] is None


def test_score_day_rows_per_stamp_with_forward_targets(tmp_path, monkeypatch):
    # a log from 09:16 to 15:35, one reading a minute, spot drifting +1/min
    rows = _session(380, drift=1.0)
    monkeypatch.setattr(f, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sc.f, "DATA_DIR", tmp_path)
    f._READ.clear()
    p = tmp_path / f"chain_NIFTY_{DAY:%Y%m%d}.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(f, "day_open", lambda code, day, rows: (25000.0, "test"))
    monkeypatch.setattr(f, "close_base", lambda code, day, e: None)
    monkeypatch.setattr(f, "prev_close", lambda code, day: 24990.0)
    monkeypatch.setattr(f, "sessions_left", lambda day, e: 1)
    recs = sc.score_day("NIFTY", DAY, EXP.isoformat())
    assert [r["stamp"] for r in recs] == list(sc.STAMPS)
    r = {x["stamp"]: x for x in recs}
    # 12:00 -> 12:30 is +30 points on 25164 = +0.119%; to the 15:30 reading = +210 points
    assert r["12:00"]["u"] == pytest.approx(25000 + 164)
    assert r["12:00"]["fwd_30"] == pytest.approx(30 / 25164 * 100, rel=1e-3)
    assert r["12:00"]["fwd_close"] == pytest.approx(210 / 25164 * 100, rel=1e-3)
    assert r["15:00"]["fwd_60"] is None                    # 16:00 is after the close
    assert r["15:00"]["fwd_close"] == pytest.approx(30 / 25344 * 100, rel=1e-3)
    assert all(x["v"] == sc.VERSION and x["ladder"] == 17 for x in recs)
    # score_all writes them once and skips them next time
    monkeypatch.setattr(f, "logged_days", lambda code: [DAY] if code == "NIFTY" else [])
    monkeypatch.setattr(f, "logged_expiries", lambda code, day: [EXP.isoformat()])
    n1 = sc.score_all(today=dt.date(2026, 11, 7))
    n2 = sc.score_all(today=dt.date(2026, 11, 7))
    assert (n1, n2) == (1, 0)
    assert sc.scored_days("NIFTY") == 1 and len(sc.load("NIFTY")) == len(sc.STAMPS)


def test_grade_counts_a_day_once_and_follows_the_lore_sign():
    recs = []
    for i in range(12):
        day = f"2026-11-{i + 1:02d}"
        for e, bump in (("2026-11-17", 0.0), ("2026-11-23", 5.0)):   # two expiries a day
            recs.append(dict(day=day, e=e, stamp="12:00", expiry_day=False,
                             chain_iv=(-1.0 if i % 2 else 1.0) + bump,       # − = fear fades
                             net_bb=float(i - 6), fwd_close=(0.5 if i % 2 else -0.5)))
    df = pd.DataFrame(recs)
    g = sc.grade(df, "chain_iv", "12:00", "fwd_close", False)
    assert g["n"] == 12 and g["n_called"] == 12            # one row per day, nearest expiry
    assert g["hit"] == 1.0                                  # −chain_iv agrees with the move
    assert g["up"] == pytest.approx(0.5) and g["down"] == pytest.approx(-0.5)
    assert sc.grade(df, "net_bb", "09:45", "fwd_close", False)["n"] == 0
    assert sc.grade(df, "net_bb", "12:00", "fwd_close", True)["n"] == 0
    t = sc.grade_table(df, "fwd_close", None)
    assert set(t["signal"]) == set(sc.SIGNALS) and len(t) == len(sc.SIGNALS) * len(sc.STAMPS)
    ic = sc._spearman(np.arange(10.0), np.arange(10.0))
    assert ic == pytest.approx(1.0) and sc._spearman(np.ones(3), np.ones(3)) is None
