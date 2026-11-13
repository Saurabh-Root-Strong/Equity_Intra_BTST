"""Index F&O Management (eqbtst/index_fno.py) — offline tests, no NSE, no DCM."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from eqbtst import index_fno as f

EXP = dt.date(2026, 11, 9)
DAY = dt.date(2026, 11, 6)


def _side(K, spot, call, oi=1000.0, doi=100.0, vol=5000.0, iv=15.0, chg=1.0):
    intr = max(spot - K, 0) if call else max(K - spot, 0)
    ltp = intr + 50.0
    return {"strikePrice": K, "lastPrice": ltp, "change": chg, "openInterest": oi,
            "changeinOpenInterest": doi, "totalTradedVolume": vol,
            "impliedVolatility": iv, "buyPrice1": ltp - 0.5, "sellPrice1": ltp + 0.5,
            "underlyingValue": spot}


def payload(spot=25030.0, lo=24000, hi=26000, step=50, ts="06-Nov-2026 10:00:00",
            oi=None, doi=None, iv=None, vol=None, exp="09-Nov-2026", drop_pe=()):
    oi, doi, iv, vol = oi or {}, doi or {}, iv or {}, vol or {}
    data = []
    for K in range(lo, hi + 1, step):
        r = {"strikePrice": K, "expiryDates": exp}
        for s in ("CE", "PE"):
            if s == "PE" and K in drop_pe:
                continue
            r[s] = _side(K, spot, s == "CE", oi.get((s, K), 1000.0), doi.get((s, K), 100.0),
                         vol.get((s, K), 5000.0), iv.get((s, K), 15.0))
        data.append(r)
    return {"records": {"data": data, "timestamp": ts, "underlyingValue": spot,
                        "expiryDates": [exp]},
            "filtered": {"data": data, "CE": {"totOI": 5e5, "totVol": 1e7},
                         "PE": {"totOI": 8e5, "totVol": 9e6}}}


# ── parsing ─────────────────────────────────────────────────────────────────────────────
def test_parse_chain_basic():
    ch = f.parse_chain(payload(), EXP)
    assert ch["spot"] == 25030.0 and ch["ts"] == dt.datetime(2026, 11, 6, 10, 0)
    assert len(ch["strikes"]) == 41 and ch["tot"] == [5e5, 1e7, 8e5, 9e6]
    assert ch["ce"][25000.0][2] == 1000.0


def test_parse_chain_zero_means_missing():
    j = payload()
    s = j["records"]["data"][10]["CE"]
    s["lastPrice"], s["impliedVolatility"], s["buyPrice1"] = 0, 0, 0
    ch = f.parse_chain(j, EXP)
    v = dict(zip(f.FIELDS, ch["ce"][float(s["strikePrice"])]))
    assert v["ltp"] is None and v["iv"] is None and v["bid"] is None and v["oi"] == 1000.0


def test_parse_chain_rejects_other_expiry_rows_and_thin_chain():
    j = payload(lo=24900, hi=25100)                      # only 5 strikes
    with pytest.raises(f.im.NseError):
        f.parse_chain(j, EXP)
    j = payload()
    for r in j["records"]["data"][:20]:
        r["expiryDates"] = "17-Nov-2026"
    ch = f.parse_chain(j, EXP)
    assert len(ch["strikes"]) == 21
    for r in j["records"]["data"][:30]:                  # 11 left: too thin to show 17
        r["expiryDates"] = "17-Nov-2026"
    with pytest.raises(f.im.NseError):
        f.parse_chain(j, EXP)


def test_parse_chain_empty_raises():
    with pytest.raises(f.im.NseError):
        f.parse_chain({"records": {"data": []}}, EXP)


def test_one_sided_strike_not_atm_candidate():
    ch = f.parse_chain(payload(spot=25010, drop_pe=(25000,)), EXP)
    assert 25000.0 not in ch["strikes"]
    assert f.pick_atm(ch["strikes"], ch["spot"]) == 25050.0 or \
        f.pick_atm(ch["strikes"], ch["spot"]) == 24950.0


def test_parse_expiries_and_monthly():
    ex = f.parse_expiries({"expiryDates": ["17-Nov-2026", "09-Nov-2026", "23-Nov-2026",
                                           "01-Dec-2026", "29-Dec-2026", "bad"]})
    assert ex[0] == dt.date(2026, 11, 9) and len(ex) == 5
    assert not f.is_monthly(dt.date(2026, 11, 9), ex)
    assert f.is_monthly(dt.date(2026, 11, 23), ex)
    cap = f.captured_expiries(ex, DAY)
    assert cap == [dt.date(2026, 11, 9), dt.date(2026, 11, 17), dt.date(2026, 11, 23)]
    # monthly-only index: nearest IS the monthly -> two expiries
    mo = [dt.date(2026, 11, 23), dt.date(2026, 12, 29), dt.date(2027, 1, 25)]
    assert f.captured_expiries(mo, DAY) == mo[:2]
    # expiry day: today's expiry still captured; a past one never
    assert f.captured_expiries(ex, dt.date(2026, 11, 9))[0] == dt.date(2026, 11, 9)
    assert f.captured_expiries(ex, dt.date(2026, 11, 10))[0] == dt.date(2026, 11, 17)


# ── strikes ─────────────────────────────────────────────────────────────────────────────
def test_pick_atm_tie_goes_lower():
    assert f.pick_atm([24950.0, 25000.0, 25050.0], 25025.0) == 25000.0
    assert f.pick_atm([24950.0, 25000.0, 25050.0], 25026.0) == 25050.0
    assert f.pick_atm([], 25000) is None


def test_window_uses_listed_strikes_and_edges():
    ks = [float(k) for k in range(24000, 25001, 50)] + [25100.0, 25200.0, 25300.0]
    w = f.window(ks, 25000.0, 8)
    assert w[0] == 24600.0 and w[-1] == 25300.0 and len(w) == 12     # only 3 listed above
    assert len(f.window(ks, 24500.0, 8)) == 17
    assert f.window(ks, 24990.0, 2) == [24900.0, 24950.0, 25000.0, 25100.0, 25200.0]


def test_max_pain_known_answer():
    ce = {100.0: 0, 110.0: 1000, 120.0: 0}
    pe = {100.0: 0, 110.0: 0, 120.0: 0}
    assert f.max_pain(ce, pe) in (100.0, 110.0)
    ce = {100.0: 10, 110.0: 10, 120.0: 500}
    pe = {100.0: 500, 110.0: 10, 120.0: 10}
    assert f.max_pain(ce, pe) == 110.0
    assert f.max_pain({}, {}) is None


def test_compact_roundtrip():
    ch = f.parse_chain(payload(lo=20000, hi=30000), EXP)
    row = f.compact(ch, keep=5)
    ks = [k[0] for k in row["k"]]
    assert row["a"] == 25050.0
    # 5 listed either side + every 100-pt strike within 21 steps (2,100 pts) of ATM
    assert set(f.window(ch["strikes"], 25050.0, 5)) <= set(ks)
    assert {float(k) for k in range(23000, 27101, 100)} <= set(ks)
    assert 22900.0 not in ks and 24975.0 not in ks and 23050.0 not in ks
    ce, pe = f.row_sides(row)
    assert set(ce) == set(pe) and ce[25050.0]["oi"] == 1000.0
    json.dumps(row)                                      # serialisable


# ── reads ───────────────────────────────────────────────────────────────────────────────
def test_premium_read_quadrants():
    assert f.premium_read("CE", 5, 10) == "long build"
    assert f.premium_read("CE", -5, 10) == "writing"
    assert f.premium_read("CE", 5, -10) == "short covering"
    assert f.premium_read("CE", -5, -10) == "long unwind"
    assert f.premium_read("CE", 0, 10) is None and f.premium_read("CE", None, 1) is None


def test_iv_read_and_lean():
    assert f.iv_read(500, 1.0, 1000) == "buyers adding"
    assert f.iv_read(500, -1.0, 1000) == "writers adding"
    assert f.iv_read(-500, 1.0, 1000) == "writers covering"
    assert f.iv_read(-500, -1.0, 1000) == "buyers exiting"
    assert f.iv_read(500, 0.2, 1000) == "unclear"
    assert f.iv_read(5, 2.0, 1000) == "little change"
    assert f.iv_read(None, 1.0, 1000) is None and f.iv_read(np.nan, 1.0, 1000) is None
    assert f.lean("PE", "writers adding") == "🟢" and f.lean("CE", "writers adding") == "🔴"
    assert f.lean("PE", "writing") == "🟢" and f.lean("CE", "long build") == "🟢"
    assert f.lean("CE", "unclear") == "⚪"


def _session(n=30, start=dt.datetime(2026, 11, 6, 9, 16), gap_after=None, drift=5.0):
    """n compact rows a minute apart: spot drifts up, 25000 PE OI builds, IV falls."""
    rows = []
    for i in range(n):
        t = start + dt.timedelta(minutes=i + (60 if gap_after is not None and i > gap_after
                                               else 0))
        spot = 25000 + drift * i
        oi = {("PE", 25000): 1000.0 + 100 * i, ("CE", 25200): 2000.0 + 50 * i}
        doi = {("PE", 25000): 100.0 + 100 * i, ("CE", 25200): 100.0 + 50 * i}
        iv = {("PE", 25000): 16.0 - 0.1 * i}
        vol = {("PE", 25000): 1000.0 * (i + 1)}
        j = payload(spot=spot, ts=t.strftime("%d-%b-%Y %H:%M:%S"), oi=oi, doi=doi, iv=iv,
                    vol=vol)
        rows.append(f.compact(f.parse_chain(j, EXP)))
    return rows


def test_ladder_iv_read_from_first_reading():
    rows = _session()
    ks = f.window([k[0] for k in rows[-1]["k"]], rows[-1]["a"])
    tab = f.ladder(rows[-1], rows[0], ks, "iv")
    r = tab[tab["Strike"] == 25000.0].iloc[0]
    assert r["PE ΔOI since first"] == 2900.0 and r["PE ΔIV"] == pytest.approx(-2.9)
    assert r["PE read"] == "🟢 writers adding"
    assert tab["atm"].sum() == 1
    tab2 = f.ladder(rows[-1], None, ks, "premium")
    assert tab2["PE ΔIV"].isna().all()


def test_headline_numbers():
    rows = _session()
    last = rows[-1]
    ks = f.window([k[0] for k in last["k"]], last["a"])
    h = f.headline(last, rows[0], ks)
    assert h["atm"] == last["a"] and h["straddle"] == pytest.approx(
        2 * 50 + abs(last["u"] - last["a"]))
    assert h["pcr_all"] == pytest.approx(1.6) and h["pe_oi"] == 25000.0
    assert h["mp"] is not None


def test_session_series_fixed_strikes_and_units():
    rows = _session()
    df = f.frame(rows)
    ks = f.window([k[0] for k in rows[-1]["k"]], rows[-1]["a"])
    ss = f.session_series(df, rows, ks, lot=65, base="yday")
    assert len(ss) == 30
    # calls: 17 strikes, 16 at doi 100 + 25200 at 100+50i ; ₹ Cr = contracts*65*spot/1e7
    i = 29
    calls = (16 * 100 + 100 + 50 * i) * 65 * rows[0]["u"] / 1e7   # first reading's level
    assert ss["calls"].iloc[-1] == pytest.approx(calls)
    first = f.session_series(df, rows, ks, lot=None, base="first")
    assert first["puts"].iloc[0] == 0 and first["puts"].iloc[-1] == pytest.approx(2900.0)
    assert (ss["calls − puts"] == ss["calls"] - ss["puts"]).all()
    assert ss["straddle"].notna().all() and ss["atm iv"].notna().all()


def test_session_series_quorum_blanks_partial_readings():
    rows = _session(5)
    rows[2]["k"] = rows[2]["k"][:15]                     # most of the window missing
    df = f.frame(rows)
    ks = f.window([k[0] for k in rows[-1]["k"]], rows[-1]["a"])
    ss = f.session_series(df, rows, ks, lot=65)
    assert np.isnan(ss["calls"].iloc[2]) and not np.isnan(ss["calls"].iloc[3])


def test_strike_panels_volume_blocks_never_negative():
    rows = _session(20)
    # NSE volume going backwards once (a bad print) must not plot negative volume
    for k in rows[10]["k"]:
        if k[0] == 25000.0:
            k[1 + f.NF + 4] = 10.0
    df = f.frame(rows)
    p = f.strike_panels(df, [25000.0], "Volume per block")
    assert (p["v"].dropna() >= 0).all() and len(p)
    p2 = f.strike_panels(df, [25000.0, 25050.0], "Implied volatility")
    assert set(p2["side"]) == {"CE", "PE"}


def test_oi_blocks_sum_to_days_change():
    rows = _session(40)
    df = f.frame(rows)
    blk = f.oi_blocks(df, [25000.0], 15)
    pe = blk[(blk["side"] == "PE") & (blk["K"] == 25000.0)]
    # day's change at last reading = doi of last row = 100 + 100*39
    assert pe["chg"].sum() == pytest.approx(100 + 100 * 39)


def test_frame_skips_malformed():
    rows = _session(2)
    rows.append({"t": "bad"})
    rows.append({"t": "2026-11-06T10:00:00", "k": [[25000.0, 1, 2]]})
    df = f.frame(rows)
    assert df["t"].nunique() == 2


def test_fut_series_and_word():
    rows = [{"t": "2026-11-06T09:20:00", "s": 25000.0,
             "F": {"2026-11-23": [25050.0, 24900.0, 2.0e5, 5000.0, 1e5],
                   "2026-12-29": [25200.0, 25000.0, 2e4, 100.0, 1e4]}},
            {"t": "2026-11-23T09:20:00", "s": 25000.0,                 # expiry day -> next
             "F": {"2026-11-23": [25001.0, 24900.0, 1e5, -1e5, 1e5],
                   "2026-12-29": [25200.0, 25000.0, 2e4, 100.0, 1e4]}}]
    fs = f.fut_series(rows)
    assert fs["expiry"].tolist() == ["2026-11-23", "2026-12-29"]
    assert fs["premium bps"].iloc[0] == pytest.approx(20.0)
    assert f.fut_word(5, 10) == "long build" and f.fut_word(-5, 10) == "short build"
    assert f.fut_word(5, -10) == "short covering" and f.fut_word(-5, -10) == "long unwind"
    assert f.fut_word(0, 10) is None


def test_parse_fut():
    j = {"timestamp": "06-Nov-2026 10:00:00",
         "data": [{"instrumentType": "FUTIDX", "expiryDate": "23-Nov-2026", "lastPrice": 25008,
                   "prevClose": 24588, "openInterest": 203350, "changeinOpenInterest": 8864,
                   "totalTradedVolume": 125137, "underlyingValue": 25030.35},
                  {"instrumentType": "OPTIDX", "expiryDate": "09-Nov-2026"}]}
    r = f.parse_fut(j)
    assert r["t"] == "2026-11-06T10:00:00" and list(r["F"]) == ["2026-11-23"]
    with pytest.raises(f.im.NseError):
        f.parse_fut({"data": [{"instrumentType": "OPTIDX"}]})


# ── capture: dedupe, session clock, lock ───────────────────────────────────────────────
@pytest.fixture
def tmpdir_fno(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "DATA_DIR", tmp_path)
    monkeypatch.setattr(f, "_S", {})
    monkeypatch.setattr(f, "_READ", {})
    return tmp_path


class _Client:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = 0

    def get(self, path):
        self.calls += 1
        return self.pages.pop(0) if len(self.pages) > 1 else self.pages[0]


def test_poll_chain_logs_once_per_nse_stamp(tmpdir_fno, monkeypatch):
    monkeypatch.setattr(f, "_now", lambda: dt.datetime(2026, 11, 6, 10, 0, 30))
    moved = {("CE", 25000): 1500.0}
    c = _Client([payload(ts="06-Nov-2026 10:00:00"), payload(ts="06-Nov-2026 10:00:00"),
                 payload(ts="06-Nov-2026 10:01:00", oi=moved),
                 payload(ts="06-Nov-2026 10:02:00", oi=moved)])   # clock moved, numbers did not
    for _ in range(4):
        assert f.poll_chain("NIFTY", EXP, c)
    rows = f.chain_log("NIFTY", DAY, EXP.isoformat())
    assert [r["t"][11:16] for r in rows] == ["10:00", "10:01"]
    # a restarted writer reads the log and does not re-log the same numbers
    monkeypatch.setattr(f, "_S", {})
    f.poll_chain("NIFTY", EXP, _Client([payload(ts="06-Nov-2026 10:03:00", oi=moved)]))
    f._READ.clear()
    assert len(f.chain_log("NIFTY", DAY, EXP.isoformat())) == 2
    assert EXP.isoformat() in f.load_latest("NIFTY")["chain"]


def test_poll_chain_outside_session_not_logged(tmpdir_fno, monkeypatch):
    monkeypatch.setattr(f, "_now", lambda: dt.datetime(2026, 11, 6, 15, 40))
    f.poll_chain("NIFTY", EXP, _Client([payload(ts="06-Nov-2026 15:37:23")]))
    assert f.chain_log("NIFTY", DAY, EXP.isoformat()) == []
    assert f.load_latest("NIFTY")["chain"][EXP.isoformat()]["t"].startswith("2026-11-06T15:37")


def test_poll_chain_bad_payload_keeps_last_good(tmpdir_fno, monkeypatch):
    monkeypatch.setattr(f, "_now", lambda: dt.datetime(2026, 11, 6, 10, 0, 30))
    f.poll_chain("NIFTY", EXP, _Client([payload(ts="06-Nov-2026 10:00:00")]))
    assert not f.poll_chain("NIFTY", EXP, _Client([{"records": {"data": []}}]))
    assert f.load_latest("NIFTY")["chain"][EXP.isoformat()]["t"].endswith("10:00:00")


def test_claim_respects_live_owner(tmpdir_fno):
    (tmpdir_fno / "poller.lock").write_text(json.dumps({"pid": -1, "beat": 9e12}))
    assert not f._claim()
    (tmpdir_fno / "poller.lock").write_text(json.dumps({"pid": -1, "beat": 0}))
    assert f._claim()
    f._release()
    assert not (tmpdir_fno / "poller.lock").exists()


def test_cycle_polls_captured_expiries_only_in_hours(tmpdir_fno, monkeypatch):
    calls = []
    monkeypatch.setattr(f, "market_window", lambda now=None: True)
    monkeypatch.setattr(f, "_now", lambda: dt.datetime(2026, 11, 6, 10, 0))
    monkeypatch.setattr(f, "poll_expiries", lambda code, client=None: f._st(code).update(
        expiries=[dt.date(2026, 11, 9), dt.date(2026, 11, 17), dt.date(2026, 11, 23),
                  dt.date(2026, 12, 29)], exp_at=9e12, exp_day=DAY) or f._st(code)["expiries"])
    monkeypatch.setattr(f, "poll_chain", lambda code, e, client=None: calls.append((code, e))
                        or f._st(code)["chain_at"].__setitem__(e.isoformat(), 9e12))
    monkeypatch.setattr(f, "poll_fut", lambda code, client=None: f._st(code).update(fut_at=9e12))
    f._cycle()
    nifty = [e for c, e in calls if c == "NIFTY"]
    assert nifty == [dt.date(2026, 11, 9), dt.date(2026, 11, 17), dt.date(2026, 11, 23)]
    calls.clear()
    f._cycle()
    assert calls == []                                    # cadence respected
    # outside hours: nothing unless someone is looking
    monkeypatch.setattr(f, "_now", lambda: dt.datetime(2026, 11, 6, 20, 0))
    monkeypatch.setattr(f, "_S", {})
    f._cycle()
    assert calls == []


def test_viewer_in_another_process_drives_off_hours_fetch(tmpdir_fno, monkeypatch):
    """The writer may be the scheduled job; a page open in the dashboard must still get an
    off-hours chain, and an expiry outside the captured three must still be read."""
    calls = []
    monkeypatch.setattr(f, "market_window", lambda now=None: False)
    monkeypatch.setattr(f, "_now", lambda: dt.datetime(2026, 11, 6, 20, 0))
    monkeypatch.setattr(f, "poll_expiries", lambda code, client=None: f._st(code).update(
        expiries=[dt.date(2026, 11, 9), dt.date(2026, 11, 17), dt.date(2026, 11, 23),
                  dt.date(2026, 12, 29)], exp_at=9e12, exp_day=DAY) or f._st(code)["expiries"])
    monkeypatch.setattr(f, "poll_chain", lambda code, e, client=None: calls.append((code, e))
                        or f._st(code)["chain_at"].__setitem__(e.isoformat(), 9e12))
    monkeypatch.setattr(f, "poll_fut", lambda code, client=None: f._st(code).update(fut_at=9e12))
    f._cycle()
    assert calls == []
    # "another process" = a view file this process's memory knows nothing about
    f.im._write_json(tmpdir_fno / "view_BANKNIFTY.json",
                     {"at": __import__("time").time(), "exp": "2026-12-29"})
    f._cycle()
    assert [c for c, _ in calls] and {c for c, _ in calls} == {"BANKNIFTY"}
    assert dt.date(2026, 12, 29) in [e for _, e in calls]
    # touch() writes the file
    f.touch("NIFTY", dt.date(2026, 11, 17))
    assert json.loads((tmpdir_fno / "view_NIFTY.json").read_text())["exp"] == "2026-11-17"


def test_compress_old_logs(tmpdir_fno):
    p = f._path("chain", "NIFTY", dt.date(2026, 11, 5))
    p.write_text('{"t": "2026-11-05T10:00:00", "e": "2026-11-09"}\n', encoding="utf-8")
    today = f._path("chain", "NIFTY", DAY)
    today.write_text("{}\n", encoding="utf-8")
    assert f.compress_old_logs(DAY) == 1
    assert not p.exists() and today.exists()
    assert f.chain_log("NIFTY", dt.date(2026, 11, 5), "2026-11-09")[0]["t"].startswith("2026")
    assert dt.date(2026, 11, 5) in f.logged_days("NIFTY")


# ── audit 6 Nov 2026: model IV, corrected close, heal, holidays, races ─────────────────
def _b76_payload(F=25000.0, sig=0.18, T=5 / 252, spot=None, ts="06-Nov-2026 10:00:00",
                 skew=0.0, nse_iv=(14.0, 21.0)):
    """A chain priced by Black-76 off forward F (spot may differ — carry / stale index)."""
    spot = spot or F
    data = []
    for K in range(24000, 26001, 50):
        s = sig + skew * (25000 - K) / 1000
        c, _ = f.im.b76(F, K, T, s, True)
        p, _ = f.im.b76(F, K, T, s, False)
        r = {"strikePrice": K, "expiryDates": "09-Nov-2026"}
        for side, px, iv in (("CE", float(c), nse_iv[0]), ("PE", float(p), nse_iv[1])):
            r[side] = {"strikePrice": K, "lastPrice": round(px, 2), "change": 1.0,
                       "openInterest": 1000, "changeinOpenInterest": 10,
                       "totalTradedVolume": 100, "impliedVolatility": iv,
                       "buyPrice1": round(px - 0.05, 2), "sellPrice1": round(px + 0.05, 2)}
        data.append(r)
    return {"records": {"data": data, "timestamp": ts, "underlyingValue": spot},
            "filtered": {}}


def test_model_iv_recovers_vol_and_forward_despite_bad_spot(monkeypatch):
    """NSE prices IV off spot; with spot 60 pts off the forward its call and put IVs split.
    The model takes the forward from parity and recovers one IV for both sides."""
    monkeypatch.setattr(f.im, "_trading_years", lambda t, e: 5 / 252)
    ch = f.parse_chain(_b76_payload(F=25000.0, spot=25060.0), EXP)
    row = f.compact(ch)
    m = f.model_iv(row, EXP, dt.datetime(2026, 11, 6, 10, 0))
    assert m["F"] == pytest.approx(25000.0, abs=0.5)
    row.update(m)
    ce, pe = f.row_sides(row)
    for K in (24700.0, 25000.0, 25300.0):
        assert ce[K]["IV"] == pytest.approx(18.0, abs=0.15)
        assert pe[K]["IV"] == pytest.approx(18.0, abs=0.15)
        assert ce[K]["iv"] == 14.0                       # NSE's figure kept, not used
    h = f.headline(row, None, f.window(sorted(ce), row["a"]))
    assert h["atm_iv"] == pytest.approx(18.0, abs=0.15) and h["fwd"] == m["F"]


def test_model_iv_empty_without_time_or_parity(monkeypatch):
    monkeypatch.setattr(f.im, "_trading_years", lambda t, e: 0.0)
    row = f.compact(f.parse_chain(_b76_payload(), EXP))
    assert f.model_iv(row, EXP, dt.datetime(2026, 11, 9, 15, 31)) == {}
    monkeypatch.setattr(f.im, "_trading_years", lambda t, e: 5 / 252)
    assert f.model_iv(row, EXP, None) == {}
    row["u"] = 30000.0                                   # forward 17% off spot: broken
    assert f.model_iv(row, EXP, dt.datetime(2026, 11, 6, 10)) == {}


def test_iv_never_mixes_sources_between_rows(monkeypatch):
    monkeypatch.setattr(f.im, "_trading_years", lambda t, e: 5 / 252)
    old = f.compact(f.parse_chain(_b76_payload(), EXP))            # NSE IV only
    new = dict(f.compact(f.parse_chain(_b76_payload(), EXP)))
    new.update(f.model_iv(new, EXP, dt.datetime(2026, 11, 6, 10)))
    ks = f.window([k[0] for k in new["k"]], new["a"])
    tab = f.ladder(new, old, ks, "iv")
    assert tab["CE ΔIV"].isna().all()                    # 18 (model) - 14 (NSE) is not a change
    assert f.headline(new, old, ks)["atm_iv0"] is None


def test_smile_uses_out_of_the_money_side(monkeypatch):
    monkeypatch.setattr(f.im, "_trading_years", lambda t, e: 5 / 252)
    row = f.compact(f.parse_chain(_b76_payload(skew=0.02), EXP))
    row.update(f.model_iv(row, EXP, dt.datetime(2026, 11, 6, 10)))
    sm = f.smile_points(row, [24800.0, 25000.0, 25200.0])
    assert sm[24800.0] > sm[25000.0] > sm[25200.0]      # put skew comes through


def test_close_snapshot_and_reconcile(tmpdir_fno, monkeypatch):
    """Session rows go to the log; a chain stamped after 15:35 goes to the close file only;
    reconcile scores the last session reading against it."""
    clock = {"now": dt.datetime(2026, 11, 6, 15, 25)}
    monkeypatch.setattr(f, "_now", lambda: clock["now"])
    oi_live = {("CE", K): 1100.0 for K in range(24000, 26001, 50)}
    doi_live = {("CE", K): 100.0 for K in range(24000, 26001, 50)}
    f.poll_chain("NIFTY", EXP, _Client([payload(ts="06-Nov-2026 15:25:00", oi=oi_live,
                                                doi=doi_live)]))
    clock["now"] = dt.datetime(2026, 11, 6, 15, 41)
    oi_off = {("CE", K): 980.0 for K in range(24000, 26001, 50)}       # NSE's correction
    doi_off = {("CE", K): -20.0 for K in range(24000, 26001, 50)}      # sign flips
    f.poll_chain("NIFTY", EXP, _Client([payload(ts="06-Nov-2026 15:40:00", oi=oi_off,
                                                doi=doi_off)]))
    assert len(f.chain_log("NIFTY", DAY, EXP.isoformat())) == 1
    close = f.load_close("NIFTY", DAY)
    assert close["chain"][EXP.isoformat()]["t"].endswith("15:40:00")
    rec = f.reconcile_day("NIFTY", DAY)[0]
    assert rec["sign"] == pytest.approx(50.0)           # calls all flipped, puts all agree
    assert rec["bias_CE"] == pytest.approx((1100 / 980 - 1) * 100, abs=0.01)
    assert rec["bias_PE"] == 0.0
    clock["now"] = dt.datetime(2026, 11, 7, 9, 0)
    assert f.reconcile() == 1 and f.reconcile() == 0     # once per day
    s = f.reconcile_summary("NIFTY")
    assert s["days"] == 1 and s["sign"] == pytest.approx(50.0)


def test_heal_ignores_viewed_and_expired_chains(tmpdir_fno, monkeypatch):
    """First version reset the NSE session every 2 minutes all day once any chain held in
    memory went old (an expiry viewed once, yesterday's expired contract)."""
    resets = []
    monkeypatch.setattr(f, "_reset_client", lambda why: resets.append(why))
    monkeypatch.setattr(f, "_now", lambda: dt.datetime(2026, 11, 6, 11, 0))
    monkeypatch.setattr(f, "_HEAL", {"at": 0.0, "refresh_seen": 9e12})
    s = f._st("NIFTY")
    fresh = dt.datetime(2026, 11, 6, 10, 59, 30).isoformat()
    s["chain"] = {"2026-11-09": {"f": fresh}, "2026-12-29": {"f": "2026-11-06T09:30:00"},
                  "2026-11-05": {"f": "2026-11-05T15:30:00"}}
    s["nearest"] = "2026-11-09"
    s["changed_at"] = {"2026-11-09": __import__("time").time() - 30}
    f._maybe_heal()
    assert resets == []
    # nearest expiry frozen (stamp ticking, numbers not) -> heal
    s["changed_at"]["2026-11-09"] = __import__("time").time() - f.HEAL_AFTER - 5
    f._maybe_heal()
    assert len(resets) == 1 and "frozen" in resets[0]


def test_cycle_prunes_expired_and_slows_on_holiday(tmpdir_fno, monkeypatch):
    calls = []
    monkeypatch.setattr(f, "_now", lambda: dt.datetime(2026, 11, 10, 11, 0))   # a holiday
    monkeypatch.setattr(f, "poll_expiries", lambda code, client=None: f._st(code).update(
        expiries=[dt.date(2026, 11, 17), dt.date(2026, 11, 23)], exp_at=9e12,
        exp_day=dt.date(2026, 11, 10)) or f._st(code)["expiries"])
    monkeypatch.setattr(f, "poll_chain", lambda code, e, client=None: calls.append((code, e))
                        or f._st(code)["chain_at"].__setitem__(e.isoformat(),
                                                               __import__("time").time()))
    monkeypatch.setattr(f, "poll_fut", lambda code, client=None: f._st(code).update(
        fut_at=__import__("time").time()))
    s = f._st("NIFTY")
    s["chain"] = {"2026-11-09": {"t": "2026-11-09T15:40:00"},
                  "2026-11-17": {"t": "2026-11-09T15:40:00"}}   # NSE still serving Monday
    f._cycle()
    assert "2026-11-09" not in s["chain"]                 # expired: out of memory
    # the nearest chain is a day old after 09:20 -> polled every 5 min, not every minute
    s["chain_at"]["2026-11-17"] = __import__("time").time() - 120
    calls.clear()
    f._cycle()
    assert ("NIFTY", dt.date(2026, 11, 17)) not in calls


def test_chain_log_dedupes_double_writer(tmpdir_fno):
    p = f._path("chain", "NIFTY", DAY)
    for t, x in (("2026-11-06T10:00:00", 1), ("2026-11-06T10:01:00", 2),
                 ("2026-11-06T10:00:00", 3)):
        f.im._append(p, {"t": t, "e": "2026-11-09", "x": x})
    rows = f.chain_log("NIFTY", DAY, "2026-11-09")
    assert [r["x"] for r in rows] == [3, 2]


def test_rupee_lines_use_one_index_level():
    """A pure index move with no position change must not move the ₹ lines."""
    rows = _session(10, drift=50.0)                       # index +450 pts, OI fixed below
    for r in rows:
        for k in r["k"]:
            k[1 + 3] = 100.0                              # CE doi constant
    df = f.frame(rows)
    ks = f.window([k[0] for k in rows[-1]["k"]], rows[0]["a"], 4)
    ss = f.session_series(df, rows, ks, lot=65)
    assert ss["calls"].dropna().nunique() == 1


# ── strike gap (user, 6 Nov: Nifty OI sits on the hundreds) ─────────────────────────────
def test_grid_and_listed_step():
    ks = [float(k) for k in range(24000, 26001, 50)]
    g = f.grid(ks, 100)
    assert g[:3] == [24000.0, 24100.0, 24200.0] and len(g) == 21
    assert f.grid(ks, None) == ks
    assert f.grid([24050.0, 24150.0], 100) == [24050.0, 24150.0]     # never an empty ladder
    assert f.listed_step(ks, 25000.0) == 50.0
    mid = [float(k) for k in range(13500, 14501, 25)]
    assert f.listed_step(mid, 14000.0) == 25.0
    assert f.on_grid(25000.0, 100) and not f.on_grid(25050.0, 100) and f.on_grid(1.0, None)


def test_ladder_and_headline_on_100_grid(monkeypatch):
    monkeypatch.setattr(f.im, "_trading_years", lambda t, e: 5 / 252)
    row = f.compact(f.parse_chain(_b76_payload(spot=25030.0, F=25030.0), EXP))
    row.update(f.model_iv(row, EXP, dt.datetime(2026, 11, 6, 10)))
    ce, pe = f.row_sides(row)
    assert row["a"] == 25050.0                            # NSE's listed ATM
    atm = f.grid_atm(row, 100)
    assert atm == 25000.0                                 # 30 from 25,000 vs 70 from 25,100
    ks = f.window(f.grid(set(ce) & set(pe), 100), atm, 8)
    assert ks[0] == 24200.0 and ks[-1] == 25800.0 and len(ks) == 17
    assert all(k % 100 == 0 for k in ks)
    tab = f.ladder(row, None, ks, "iv", atm)
    assert tab.loc[tab["atm"], "Strike"].tolist() == [25000.0]
    assert tab["CE ΔIV"].dtype == float                   # NaN, never the word 'None'
    h = f.headline(row, None, ks, 100)
    assert h["atm"] == 25000.0
    assert h["straddle"] == pytest.approx(f._px(ce[25000.0]) + f._px(pe[25000.0]))


def test_session_series_straddle_on_grid():
    rows = _session(5)                                    # spot 25000..25020
    df = f.frame(rows)
    ks = f.window(f.grid([k[0] for k in rows[-1]["k"]], 100), 25000.0, 4)
    ss = f.session_series(df, rows, ks, lot=65, step=100)
    # payload premium = intrinsic + 50 (mid of a tight book): ATM 25,000 at every reading
    exp = [100.0 + (25000 + 5 * i - 25000) for i in range(5)]
    assert ss["straddle"].tolist() == pytest.approx(exp)


# ── test-data findings (Fyers 5-6 Nov import): stale prices at a gap open ──────────────
def test_untraded_contract_has_no_price():
    assert f._px({"ltp": 1345.5, "vol": 0.0}) is None            # 5 Nov Fin 26100 CE 09:15
    assert f._px({"ltp": 335.3, "vol": 66.0}) == 335.3
    assert f._px({"ltp": 9.0, "vol": 0.0, "bid": 99.0, "ask": 101.0}) == 100.0  # tight book


def test_model_row_without_forward_is_blank_not_nse(monkeypatch):
    monkeypatch.setattr(f.im, "_trading_years", lambda t, e: 5 / 252)
    row = f.compact(f.parse_chain(_b76_payload(), EXP))
    for k in row["k"]:
        k[1 + 4] = 0.0                                   # no call traded, no book
        k[1 + 6] = k[1 + 7] = None
    m = f.model_iv(row, EXP, dt.datetime(2026, 11, 6, 9, 15))
    assert m == {}
    row.update(m or {"v": []})
    ce, pe = f.row_sides(row)
    assert f.iv_source(row) == "model"
    assert all(v["IV"] is None for v in ce.values())     # never NSE's 14.0 instead


def test_itm_side_iv_dropped_when_it_disagrees(monkeypatch):
    monkeypatch.setattr(f.im, "_trading_years", lambda t, e: 5 / 252)
    j = _b76_payload(F=25000.0)
    for r in j["records"]["data"]:
        if r["strikePrice"] == 25300:                    # ITM put, stale: priced 40 rupees rich
            r["PE"]["lastPrice"] += 150
            r["PE"]["buyPrice1"] = r["PE"]["sellPrice1"] = 0
    row = f.compact(f.parse_chain(j, EXP))
    row.update(f.model_iv(row, EXP, dt.datetime(2026, 11, 6, 10)))
    ce, pe = f.row_sides(row)
    assert pe[25300.0]["IV"] is None and ce[25300.0]["IV"] == pytest.approx(18.0, abs=0.2)
    assert pe[24700.0]["IV"] == pytest.approx(18.0, abs=0.2)     # OTM side always kept
    assert ce[24700.0]["IV"] == pytest.approx(18.0, abs=0.2)     # consistent ITM kept


def test_baseline_is_each_strikes_first_valid_iv():
    def r(t, ivs):
        k = [[K, 10, 1, 100, 1, 5, None, None, None, 10, 1, 100, 1, 5, None, None, None]
             for K in (24900.0, 25000.0, 25100.0)]
        return {"t": t, "u": 25000.0, "a": 25000.0, "k": k,
                "v": [[K, ivs.get(K), ivs.get(K)] for K in (24900.0, 25000.0, 25100.0)]}
    rows = [r("2026-11-05T09:15:00", {}), r("2026-11-05T09:20:00", {25000.0: 14.0}),
            r("2026-11-05T09:28:00", {24900.0: 15.0, 25000.0: 13.0, 25100.0: 12.5})]
    b = f.baseline_iv(rows, [24900.0, 25000.0, 25100.0])
    assert b["CE"] == {25000.0: 14.0, 24900.0: 15.0, 25100.0: 12.5} and b["atm"] == 14.0
    tab = f.ladder(rows[-1], rows[0], [24900.0, 25000.0, 25100.0], "iv", 25000.0, b)
    assert tab.loc[tab.Strike == 25000.0, "CE ΔIV"].iloc[0] == pytest.approx(-1.0)


def test_fyers_import_converts_units_and_tags(tmp_path, monkeypatch):
    from eqbtst import index_fno_fyers as fy
    monkeypatch.setattr(f, "DATA_DIR", tmp_path)
    monkeypatch.setattr(f, "_READ", {})
    monkeypatch.setattr(f.im, "_trading_years", lambda t, e: 5 / 252)
    monkeypatch.setattr(fy, "_nearest_expiry", lambda code, day: EXP)
    monkeypatch.setattr(fy, "_official", lambda code, day, e: pd.DataFrame(
        columns=["k", "s", "oi", "doi"]))
    recs, cds = [], []
    for i, ts in enumerate(pd.to_datetime(["2026-11-06 03:44:10", "2026-11-06 03:46:00",
                                           "2026-11-06 03:47:00"], utc=True)):
        for K in range(24000, 26001, 50):
            for side in ("CE", "PE"):
                recs.append(dict(ts=ts, symbol="NSE:NIFTY50-INDEX", strike=K, side=side,
                                 ltp=100.0 + i, ltpch=1.0, oi=65 * (1000 + i), oich=65 * i,
                                 volume=65 * (10 + i), delta=0.5, iv=15.0, expiry=0))
        cds.append(dict(ts=ts, symbol="NSE:NIFTY50-INDEX", resolution="1sec", close=25010.0))
    src = tmp_path / "src"
    src.mkdir()
    pd.DataFrame(recs).to_parquet(src / "2026-11-06_chain_snapshots.parquet")
    pd.DataFrame(cds).to_parquet(src / "2026-11-06_candles.parquet")
    res = fy.convert_day(dt.date(2026, 11, 6), src)
    rows = res["rows"]["NIFTY"]
    assert [r["t"][11:19] for r in rows] == ["09:16:00", "09:17:00"]   # 09:14 dropped, IST
    ce, _ = f.row_sides(rows[0])
    assert ce[25000.0]["oi"] == 1001 and ce[25000.0]["doi"] == 1 and ce[25000.0]["vol"] == 11
    assert all(r["src"] == "fyers" and "v" in r for r in rows)
    assert res["report"]["NIFTY"]["not_whole_lots"] == 0


def test_no_fyers_calls():
    import inspect
    src = inspect.getsource(f)
    import re
    # the page may SAY "Fyers" (the test-data banner); it must never import or call it
    assert not re.search(r"import\s+.*\b(fyers\w*|live)\b|from\s+\.\s+import\s+.*\blive\b|fyers_apiv3", src)


# ── the page renders end to end (Streamlit AppTest, synthetic log) ─────────────────────
def test_page_renders_live_and_past(tmpdir_fno, monkeypatch):
    from streamlit.testing.v1 import AppTest
    now = dt.datetime(2026, 11, 6, 9, 50)
    monkeypatch.setattr(f, "_now", lambda: now)
    monkeypatch.setattr(f, "ensure_poller", lambda: None)
    monkeypatch.setattr(f, "_claim", lambda: False)
    monkeypatch.setattr(f, "_lot", lambda *a: (65, "test"))
    monkeypatch.setattr(f, "sessions_left", lambda d, e: 1)
    rows = _session(30)
    for r in rows:
        f.im._append(f._path("chain", "NIFTY", DAY), r)
    for r in _session(10, start=dt.datetime(2026, 11, 5, 9, 16)):
        f.im._append(f._path("chain", "NIFTY", dt.date(2026, 11, 5)), r)
    fut = {"t": "2026-11-06T09:49:00", "s": 25100.0,
           "F": {"2026-11-23": [25150.0, 24900.0, 2e5, 5e3, 1e5]}}
    f.im._append(f._path("fut", "NIFTY", DAY), fut)
    f.im._write_json(tmpdir_fno / "latest_NIFTY.json", {
        "expiries": ["2026-11-09", "2026-11-17", "2026-11-23"],
        "chain": {"2026-11-09": rows[-1]}, "fut": fut})

    def app():
        import streamlit as st
        from eqbtst import index_fno
        index_fno.render_page(st)

    at = AppTest.from_function(app, default_timeout=60).run()
    assert not at.exception, at.exception
    assert not [e for e in at.error], [e.value for e in at.error]
    assert any("strikes — calls left" in e.label for e in at.expander)
    assert len(at.dataframe) >= 2
    # every metric shown, switches work
    at.segmented_control(key="if_mode_NIFTY").set_value("premium read").run()
    assert not at.exception
    at.segmented_control(key="if_metric_NIFTY").set_value("Volume per block").run()
    assert not at.exception
    at.segmented_control(key="if_pblk_NIFTY_v").set_value(60).run()
    assert not at.exception
    at.segmented_control(key="if_metric_NIFTY").set_value("Open interest").run()
    at.segmented_control(key="if_pblk_NIFTY").set_value("each reading").run()
    assert not at.exception
    at.segmented_control(key="if_pblk_NIFTY").set_value(30).run()
    assert not at.exception
    at.segmented_control(key="if_blk_NIFTY").set_value(60).run()
    assert not at.exception
    at.segmented_control(key="if_bmetric_NIFTY").set_value("Volume per 5 min").run()
    assert not at.exception
    at.segmented_control(key="if_dby_NIFTY").set_value("IV").run()
    assert not at.exception
    at.segmented_control(key="if_dblk_NIFTY").set_value(60).run()
    assert not at.exception
    at.segmented_control(key="if_dblk_NIFTY").set_value(5).run()
    assert not at.exception
    at.segmented_control(key="if_anchor2_NIFTY").set_value("live ATM").run()
    assert not at.exception
    at.select_slider(key="if_nside_NIFTY").set_value(20).run()
    assert not at.exception and not at.error
    for gap in ("listed", 50, 100, 200, "auto"):
        at.segmented_control(key="if_step_NIFTY").set_value(gap).run()
        assert not at.exception and not at.error, gap
    assert any("Best / Better / Good" in x.value for x in at.subheader)
    for m in ("Open interest", "Volume per block", "OI change"):
        at.segmented_control(key="if_bmetric_NIFTY").set_value(m).run()
        assert not at.exception and not at.error, m
    at.segmented_control(key="if_bbase_NIFTY").set_value("since this morning's first reading").run()
    assert not at.exception and not at.error
    # past session
    at.selectbox(key="if_day_NIFTY").set_value(dt.date(2026, 11, 5)).run()
    assert not at.exception and not at.error


def test_no_real_poller_from_tests_or_a_bare_import():
    """7 Nov 2026: test_core imports eqbtst.dashboard for a helper, and that import started
    BOTH real pollers -- NSE calls from a test run, the live writer locks taken from the
    running board, and a sweep thread that made test_index_mgmt flaky."""
    import pathlib
    import threading
    before = {t.name for t in threading.enumerate()}
    f.ensure_poller()
    f.im.ensure_poller()
    started = {t.name for t in threading.enumerate()} - before
    assert not {"eqbtst-index-fno", "eqbtst-index-mgmt"} & started
    src = (pathlib.Path(__file__).parent.parent / "eqbtst" / "dashboard.py").read_text(
        encoding="utf-8")
    i = src.index("if st.runtime.exists():")
    assert i < src.index("index_mgmt.ensure_poller()") and i < src.index("index_fno.ensure_poller()")


# ── strike buckets Best / Better / Good (user, 7 Nov 2026) ─────────────────────────────
def test_bucket_map_6_4_6_around_atm():
    ks = [float(k) for k in range(24200, 25801, 100)]          # 17 strikes, centre 25,000
    b = f.bucket_map(ks, 25000.0)
    best = sorted(k for k, v in b.items() if v == "Best")
    better = sorted(k for k, v in b.items() if v == "Better")
    good = sorted(k for k, v in b.items() if v == "Good")
    assert best == [float(k) for k in range(24700, 25301, 100)]  # ATM + 3 either side
    assert better == [24500.0, 24600.0, 25400.0, 25500.0]
    assert good == [24200.0, 24300.0, 24400.0, 25600.0, 25700.0, 25800.0]
    # narrower ladder: no Good; wider: everything past step 5 is Good
    assert set(f.bucket_map(ks[4:13], 25000.0).values()) == {"Best", "Better"}
    wide = [float(k) for k in range(23000, 27001, 100)]
    assert f.bucket_map(wide, 25000.0)[23000.0] == "Good"
    assert f.bucket_map([], 25000.0) == {}


def test_bucket_table_and_series_add_up():
    rows = _session(20)
    df = f.frame(rows)
    last = rows[-1]
    ks = f.window(f.grid([k[0] for k in last["k"]], 100), f.grid_atm(last, 100), 8)
    bmap = f.bucket_map(ks, f.grid_atm(last, 100))
    bt = f.bucket_table(last, rows[0], bmap, 65)
    ce, pe = f.row_sides(last)
    assert bt["Call OI"].sum() == pytest.approx(sum(ce[k]["oi"] for k in ks))
    assert bt["Put ΔOI"].sum() == pytest.approx(sum(pe[k]["doi"] for k in ks))
    best = bt.set_index("Bucket").loc["Best"]
    assert best["PCR"] == pytest.approx(best["Put OI"] / best["Call OI"])
    assert list(bt["Bucket"]) == ["Best", "Better", "Good"]
    # the bucket lines add up to the all-strikes line at every reading
    bs = f.bucket_series(df, bmap, 65)
    whole = f.session_series(df, rows, ks, 65)
    tot = bs.groupby("t")[["calls", "puts"]].sum()
    assert tot["calls"].to_numpy() == pytest.approx(whole["calls"].to_numpy())
    assert tot["puts"].to_numpy() == pytest.approx(whole["puts"].to_numpy())


def test_lean_words():
    assert f._lean(100, 300) == "🟢 put side heavier"
    assert f._lean(300, 100) == "🔴 call side heavier"
    assert f._lean(100, 105) == "⚪ balanced"
    assert f._lean(None, 5) == "—" and f._lean(0, 0) == "—"


def test_bucket_table_rupees_equal_the_lines_last_point():
    """Audit 7 Nov: the table used the live index level, the lines the first reading's, so on
    5 Nov Nifty Best said -78,435 Cr in the table and -77,744 Cr at the line's end."""
    rows = _session(20, drift=40.0)                       # the index moves +760 pts
    df = f.frame(rows)
    last = rows[-1]
    ks = f.window(f.grid([k[0] for k in last["k"]], 100), f.grid_atm(last, 100), 8)
    bmap = f.bucket_map(ks, f.grid_atm(last, 100))
    bt = f.bucket_table(last, rows[0], bmap, 65).set_index("Bucket")
    bs = f.bucket_series(df, bmap, 65)
    end = bs[bs["t"] == bs["t"].max()].set_index("bucket")["calls − puts"]
    for b in end.index:
        assert bt.loc[b, "Calls − puts ₹ Cr"] == pytest.approx(end[b])


def test_auto_gap_reaches_the_priced_move():
    """100 unless 8 x 100 falls short of the ATM straddle's move; then 200. Decided from the
    first COMPUTABLE straddle; 100 when none is."""
    def row(u, straddle, traded=True):
        k = []
        for K in range(int(u) - 2000, int(u) + 2001, 100):
            px = straddle / 2 if K == round(u / 100) * 100 else 10.0
            vol = 5.0 if traded else 0.0
            k.append([float(K), px, 0, 100, 1, vol, 15, None, None, px, 0, 100, 1, vol, 15, None, None])
        return {"t": "2026-11-05T09:20:00", "u": u, "a": round(u / 100) * 100, "k": k}
    assert f.auto_gap([row(25000.0, 250.0)])[0] == 100                 # Nifty: ±1.0%
    assert f.auto_gap([row(58000.0, 1450.0)])[0] == 200                # Bank monthly: ±2.5%
    assert f.auto_gap([row(58000.0, 600.0)])[0] == 100                 # Bank near expiry
    assert f.auto_gap([row(58000.0, 9000.0)])[0] == 200                # beyond: widest allowed
    assert f.auto_gap([row(58000.0, 1450.0, traded=False)]) [0] == 100  # untraded: default
    g, em, _ = f.auto_gap([row(58000.0, 1450.0, False), row(58000.0, 1450.0)])
    assert g == 200 and em == pytest.approx(2.5)                       # first computable
    assert f.auto_gap([], None)[0] == 100


# ── ladder + buckets fixed at the 09:15 OPEN (user, 7 Nov 2026) ─────────────────────────
def test_open_anchor_example_23780():
    """User's example: the index opens at 23,780 -> centre 23,800 -> 8 strikes either side on a
    100-point gap -> buckets decided from there for the day."""
    ks = [float(k) for k in range(22000, 25601, 50)]
    g = f.grid(ks, 100)
    centre = f.pick_atm(g, 23780.0)
    w = f.window(g, centre, 8)
    assert centre == 23800.0 and w[0] == 23000.0 and w[-1] == 24600.0 and len(w) == 17
    b = f.bucket_map(w, centre)
    assert [k for k in w if b[k] == "Best"] == [float(k) for k in range(23500, 24101, 100)]


def test_day_open_sources_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "_dcm_open", lambda code, day: None)
    monkeypatch.setattr(f.im, "DATA_DIR", tmp_path)
    day = dt.date(2026, 11, 6)
    r = lambda t, u, **kw: dict({"t": f"2026-11-06T{t}", "u": u}, **kw)  # noqa: E731
    # 1. NSE open stored by the writer wins, whatever the first reading says
    o, src = f.day_open("NIFTY", day, [r("09:16:47", 24399.0), r("10:00:00", 24450.0,
                                                                 o=24538.2, os="nse")])
    assert o == 24538.2 and "official" in src
    # 2. Index Management's same-day read (zero NSE calls)
    f.im._write_json(tmp_path / "latest_NIFTY.json", {"cash": {
        "ts": "2026-11-06T09:16:46", "index": {"open": 24538.2}}})
    assert f.day_open("NIFTY", day, [r("09:16:47", 24399.0)])[0] == 24538.2
    (tmp_path / "latest_NIFTY.json").unlink()
    # 3. first reading only when it IS the open (by 09:20)
    o, src = f.day_open("NIFTY", day, [r("09:16:47", 24399.0)])
    assert o == 24399.0 and "first reading" in src
    # 4. a late start is never passed off as the open
    o, src = f.day_open("NIFTY", day, [r("11:00:05", 24800.0)])
    assert o is None and "11:00" in src
    # yesterday's Index Management read does not count for today
    f.im._write_json(tmp_path / "latest_NIFTY.json", {"cash": {
        "ts": "2026-11-05T15:30:00", "index": {"open": 24378.05}}})
    assert f.day_open("NIFTY", day, [r("11:00:05", 24800.0)])[0] is None


def test_log_keeps_the_open_ladder_after_a_big_move():
    """Bank Nifty on a 200 gap: the open ladder reaches 1,600 points; after a 1,000-point run
    those strikes must still be logged."""
    ch = f.parse_chain(payload(spot=59000.0, lo=54000, hi=62000, step=100), EXP)
    ks = [k[0] for k in f.compact(ch)["k"]]
    assert 56200.0 not in ks                              # beyond 2,100 of today's ATM
    ks2 = [k[0] for k in f.compact(ch, anchor=58000.0)["k"]]
    assert {float(k) for k in range(56400, 59601, 200)} <= set(ks2)   # open 58,000 +-1,600


def test_writer_stamps_the_open_on_rows(tmpdir_fno, monkeypatch):
    monkeypatch.setattr(f, "_now", lambda: dt.datetime(2026, 11, 6, 10, 0, 30))
    monkeypatch.setattr(f, "nse_open", lambda code, day, client=None: 24538.2)
    f.poll_chain("NIFTY", EXP, _Client([payload(ts="06-Nov-2026 10:00:00")]))
    row = f.chain_log("NIFTY", DAY, EXP.isoformat())[0]
    assert row["o"] == 24538.2 and row["os"] == "nse"
    # before 09:16 the open is not asked for
    monkeypatch.setattr(f, "_S", {})
    monkeypatch.setattr(f, "_now", lambda: dt.datetime(2026, 11, 6, 9, 15, 20))
    assert f._day_open_writer("NIFTY", f._now()) is None


# ── time strip (user, 9 Nov: Index F&O lacked Index Management's clocks) ───────────────
def test_freshness_clocks_and_levels(monkeypatch):
    now = dt.datetime(2026, 11, 9, 10, 0, 41)
    latest = {"t": "2026-11-09T09:59:58", "f": "2026-11-09T10:00:31", "c": "2026-11-09T10:00:31"}
    fut = {"t": "2026-11-09T09:58:48", "f": "2026-11-09T10:00:05"}
    fr = f.freshness(latest, fut, [{"t": "2026-11-09T09:20:53"}], True, now)
    assert fr["chain_age"] == 43 and fr["chain_level"] == "ok" and fr["open"]
    assert fr["next_chain"] == pytest.approx(f.CHAIN_EVERY - 10)
    assert fr["next_fut"] == pytest.approx(f.FUT_EVERY - 36)
    assert fr["fut_level"] == "ok" and fr["log_from"] == "09:20" and fr["logged"] == 1
    # NSE's data was 5 min old when read (frozen feed) -> orange; 10 min -> red
    old = dict(latest, t="2026-11-09T09:55:00")
    assert f.freshness(old, fut, [], True, now)["chain_level"] == "warn"
    older = dict(latest, t="2026-11-09T09:50:00")
    assert f.freshness(older, fut, [], True, now)["chain_level"] == "stale"
    # 9 Nov false alarm: futures stamp 3 min 10 s old, but it was 66 s old when read 124 s
    # ago and the next read is due -> green, not orange
    fut2 = {"t": "2026-11-09T09:57:31", "f": "2026-11-09T09:58:37"}
    assert f.freshness(latest, fut2, [], True, now)["fut_level"] == "ok"
    # the writer died: last read 8 min ago -> red even though the data was fresh then
    dead = {"t": "2026-11-09T09:52:00", "f": "2026-11-09T09:52:40", "c": None}
    assert f.freshness(dead, fut, [], True, now)["chain_level"] == "stale"
    # after the bell, or a past session: grey, no next check
    shut = f.freshness(latest, fut, [], True, dt.datetime(2026, 11, 9, 16, 30))
    assert shut["chain_level"] == "closed" and shut["next_chain"] is None and not shut["open"]
    assert f.freshness(latest, fut, [], False, now)["chain_level"] == "closed"
    assert f.freshness({}, None, [], True, now)["chain_level"] == "none"


def test_chain_error_logged_once_an_hour(tmpdir_fno, monkeypatch):
    monkeypatch.setattr(f, "_now", lambda: dt.datetime(2026, 11, 9, 10, 0))
    bad = _Client([{"records": {"data": [{"strikePrice": 27000, "expiryDates": "29-Dec-2026",
                                          "CE": _side(27000, 27200, True)}]}}])
    for _ in range(3):
        assert not f.poll_chain("FINNIFTY", dt.date(2026, 12, 29), bad)
    log = (tmpdir_fno / "errors.log").read_text(encoding="utf-8")
    assert log.count("FINNIFTY 2026-12-29") == 1


def test_panel_titles_carry_level_and_change():
    """9 Nov: on one shared axis the bucket PCRs (0.47 / 0.72 / 1.32) were flat lines; each
    bucket now gets its own panel and the header says the level and the change."""
    idx = pd.to_datetime(["2026-11-09 09:20", "2026-11-09 10:00", "2026-11-09 11:00"])
    wide = pd.DataFrame({"Best": [0.46, 0.50, 0.47], "Good": [1.26, np.nan, 1.32],
                         "Better": [np.nan, np.nan, np.nan]}, index=idx)
    t = f.panel_titles(wide)
    assert t["Best"] == "Best · 0.47 now · +0.01 since 09:20"
    assert t["Good"] == "Good · 1.32 now · +0.06 since 09:20"
    assert t["Better"] == "Better"


def test_index_column_marks_one_bucket_and_distances():
    ks = list(range(24300, 26000, 100))                 # 17 strikes, centre 25,100
    bmap = f.bucket_map(ks, 25100)
    w = f.spot_vs_buckets(25312.4, bmap)               # nearest 25,300 -> Best
    assert w["Best"] == "◀ index here"
    assert w["Better"].startswith("188 pts up to 25,500")
    assert w["Good"].startswith("388 pts up to 25,700")
    w = f.spot_vs_buckets(25460, bmap)                 # nearest 25,500 -> Better
    assert w["Better"] == "◀ index here" and "down to 25,400" in w["Best"]
    assert sum(v == "◀ index here" for v in f.spot_vs_buckets(25450, bmap).values()) == 1
    w = f.spot_vs_buckets(26100, bmap)                 # off the ladder: no bucket holds it
    assert "◀ index here" not in w.values() and w["Good"].startswith("⚠ index past the ladder")
    assert set(f.spot_vs_buckets(None, bmap).values()) == {"—"}


def test_blocks_count_from_0915():
    t = pd.Series(pd.to_datetime(["2026-11-10 09:15:00", "2026-11-10 10:14:59",
                                  "2026-11-10 10:15:00", "2026-11-10 15:29:00"]))
    assert [x.strftime("%H:%M") for x in f.block_start(t, 60)] == ["09:15", "09:15", "10:15",
                                                                   "15:15"]
    assert [x.strftime("%H:%M") for x in f.block_start(t, 30)] == ["09:15", "09:45", "10:15",
                                                                   "15:15"]


def test_volume_blocks_keep_opening_trades_and_sum_to_day():
    rows = _session(70)                                # 09:16 .. 10:25, cumulative 1,000/min
    df = f.frame(rows)
    p = f.strike_panels(df, [25000.0], "Volume per block", 60)
    pe = p[p["side"] == "PE"].set_index("t")["v"]
    # first hour 09:15-10:15 includes the 09:15-09:16 trades of the first reading
    assert pe.iloc[0] == 59_000 and pe.iloc[1] == 11_000 and pe.sum() == 70_000
    # a late start (first reading 10:40) must NOT dump the whole morning into one bar
    late = f.frame(_session(30, start=dt.datetime(2026, 11, 6, 10, 40)))
    q = f.strike_panels(late, [25000.0], "Volume per block", 60)
    assert q[q["side"] == "PE"]["v"].sum() == 29_000


def test_level_blocks_take_last_reading_of_each_block():
    rows = _session(40)                                # 09:16 .. 09:55
    df = f.frame(rows)
    p = f.strike_panels(df, [25000.0], "Open interest", 15)
    pe = p[p["side"] == "PE"].sort_values("t")
    # blocks 09:15 / 09:30 / 09:45: last readings 09:29, 09:44, 09:55 (still running = now)
    assert [x.strftime("%H:%M") for x in pe["t"]] == ["09:29", "09:44", "09:55"]
    assert pe["v"].iloc[-1] == 1000.0 + 100 * 39
    raw = f.strike_panels(df, [25000.0], "Open interest")
    assert len(raw[raw["side"] == "PE"]) == 40


def test_index_domain_is_the_days_own_range():
    t = pd.date_range("2026-11-09 09:16", periods=4, freq="min")
    s = pd.Series([24950.0, 25000.0, 25110.0, 25008.0], index=t)
    lo, hi = f.index_domain(s, 25076.4)
    assert 24930 < lo < 24950 and 25110 < hi < 25130          # not a 1,000-point axis
    lo, hi = f.index_domain(pd.Series([25000.0] * 3, index=t[:3]))
    assert hi - lo == pytest.approx(25000 * 0.001 * 1.16)     # flat day still has room
    assert f.index_domain(pd.Series(dtype=float)) is None


def test_build_state_four_quadrants_and_side_lean():
    bs = f.build_state
    assert bs("CE", 500, 10, 10_000) == ("Long build-up", 1)
    assert bs("CE", 500, -10, 10_000) == ("Short build-up", -1)       # call writing = bearish
    assert bs("CE", -500, 10, 10_000) == ("Short covering", 1)
    assert bs("CE", -500, -10, 10_000) == ("Long unwinding", -1)
    assert bs("PE", 500, 10, 10_000) == ("Long build-up", -1)         # put buying = bearish
    assert bs("PE", 500, -10, 10_000) == ("Short build-up", 1)        # put writing = bullish
    assert bs("PE", -500, 10, 10_000) == ("Short covering", -1)
    assert bs("PE", -500, -10, 10_000) == ("Long unwinding", 1)
    assert bs("CE", 100, 10, 10_000) == ("Quiet", 0)                  # 1% < 2% floor
    assert bs("CE", 500, 0.3, 10_000, "IV") == ("Unclear", 0)          # IV move under band
    assert bs("CE", 500, None, 10_000) == (None, 0)
    assert bs("CE", 500, float("nan"), 10_000) == (None, 0)


def test_strike_states_blocks_sum_to_days_change():
    rows = _session(40)
    df = f.frame(rows)
    st_ = f.strike_states(df, [25000.0], 15)
    pe = st_[st_["side"] == "PE"]
    assert len(pe) == 3 and pe["d_oi"].sum() == pytest.approx(100 + 100 * 39)
    assert set(pe["state"]) <= set(f.STATE_CODE)
    bb = f.bull_bear(st_)
    assert list(bb.columns) == ["block", "bull", "bear", "net"] and len(bb) == 3
    assert (bb["net"] == bb["bull"] - bb["bear"]).all()


def test_drill_table_rows_high_to_low_with_markers():
    rows = _session(10)
    df = f.frame(rows)
    ks = [24900.0, 25000.0, 25100.0]
    tab = f.drill_table(rows[-1], df, ks, 25000.0, "premium", 25010.0)
    assert list(tab["Strike"]) == ["25,100", "25,000 ★ ◀ index", "24,900"]
    for c in ("Call state", "Put state", "Strike says", "Calls − puts ΔOI", "Put Δ premium"):
        assert c in tab.columns
    tiv = f.drill_table(rows[-1], df, ks, 25000.0, "IV", 25010.0)
    assert "Put Δ IV" in tiv.columns
    r = tiv[tiv["Strike"].str.startswith("25,000")].iloc[0]
    assert r["Put Δ IV"] == pytest.approx(-0.9)                    # 16.0 -> 15.1


def test_strike_says_same_state_both_sides():
    assert f._strike_says(-1, 1, "Short build-up", "Short build-up").startswith(
        "⚪ writers on both sides")
    assert f._strike_says(1, -1, "Long build-up", "Long build-up").startswith("⚪ buyers")
    assert f._strike_says(1, 1, "Long build-up", "Short build-up") == "🟢 bullish (both sides)"
    assert f._strike_says(-1, 0, "Short build-up", "Quiet") == "🔴 leaning bearish"
    assert f._strike_says(0, 0, "Quiet", "Quiet") == "⚪ quiet"


def _smile_df(spots, extra=None):
    """Sticky-moneyness smile IV_K = 20 - 0.002 (K - S): an index move alone re-prices every
    strike along the smile with nobody trading. `extra` = {(t_index, K): vol pts added}."""
    recs = []
    for i, S in enumerate(spots):
        t = pd.Timestamp("2026-11-09 09:20") + pd.Timedelta(minutes=i)
        for K in (24800.0, 24900.0, 25000.0, 25100.0, 25200.0):
            iv = 20 - 0.002 * (K - S) + (extra or {}).get((i, K), 0.0)
            for side in ("CE", "PE"):
                recs.append(dict(t=t, u=S, a=25000.0, K=K, side=side, ltp=100.0, chg=1.0,
                                 oi=10_000.0 + 500 * i, doi=500.0 * (i + 1),
                                 vol=1000.0 * (i + 1), iv=iv, bid=99.0, ask=101.0, IV=iv))
    return pd.DataFrame(recs)


def test_iv_pressure_removes_the_slide_along_the_smile():
    df = _smile_df([25000, 25050, 25100])                  # index +100, no one trades vol
    raw = df[(df.K == 25000) & (df.side == "CE")]["IV"]
    assert raw.iloc[-1] - raw.iloc[0] == pytest.approx(0.2)   # raw IV says 'buyers'
    p = f.iv_pressure(df).dropna()
    assert p.abs().max() == pytest.approx(0.0, abs=1e-9)   # pressure: nothing
    df2 = _smile_df([25000, 25050, 25100], {(2, 25000.0): 0.6})
    p2 = f.iv_pressure(df2)
    assert p2.loc[(pd.Timestamp("2026-11-09 09:22"), 25000.0, "CE")] > 0.5


def test_strike_states_iv_uses_pressure_and_reports_volume():
    df = _smile_df([25000, 25050, 25100])
    st_ = f.strike_states(df, [25000.0], 15, "IV")
    r = st_[st_["side"] == "CE"].iloc[0]
    assert r["state"] == "Unclear"                          # slide alone is not buying
    assert r["vol"] == 3000.0 and r["fresh"] == pytest.approx(r["d_oi"] / 3000.0)
    tab = f.drill_table({"t": "x", "k": []}, df, [25000.0], 25000.0, "IV", 25100.0)
    assert "Call volume" in tab.columns and "Call fresh %" in tab.columns


def test_day_flow_adds_up_who_did_it():
    st_ = pd.DataFrame([
        dict(block=1, K=25000.0, side="CE", d_oi=500.0, state="Long build-up", lean=1),
        dict(block=2, K=25000.0, side="CE", d_oi=-200.0, state="Long unwinding", lean=-1),
        dict(block=3, K=25000.0, side="CE", d_oi=300.0, state="Long build-up", lean=1),
        dict(block=1, K=25000.0, side="PE", d_oi=400.0, state="Short build-up", lean=1),
        dict(block=2, K=25000.0, side="PE", d_oi=50.0, state="Quiet", lean=0)])
    g = f.day_flow(st_).set_index(["side", "word"])
    assert g.loc[("CE", "Buyers in"), "v"] == 800 and g.loc[("CE", "Buyers out"), "v"] == 200
    assert g.loc[("CE", "Buyers out"), "signed"] == -200
    assert g.loc[("PE", "Writers in"), "signed"] == 400                # put writing = up
    assert ("PE", "·") not in g.index                                  # quiet left out


def test_prev_close_reads_index_mgmt_same_day(tmp_path, monkeypatch):
    monkeypatch.setattr(f.im, "DATA_DIR", tmp_path)
    (tmp_path / "latest_NIFTY.json").write_text(json.dumps({"cash": {
        "ts": "2026-11-11T12:24:33", "index": {"last": 25330.05, "prev": 25044.65}}}),
        encoding="utf-8")
    assert f.prev_close("NIFTY", dt.date(2026, 11, 11)) == 25044.65


def test_check_label_rules():
    cl = f.check_label
    V = 100_000
    assert cl("Long build-up", 5_000, V, 0.9, False)[0].startswith("✅")         # paid up
    assert cl("Long build-up", 5_000, V, -0.9, False)[1] == "Short build-up"    # IV says writing
    assert cl("Short build-up", 5_000, V, -0.9, False)[0].startswith("✅")
    assert cl("Short covering", -5_000, V, 0.9, False)[0].startswith("✅")
    assert cl("Long unwinding", -5_000, V, 0.9, False)[1] == "Short covering"
    assert cl("Long build-up", 5_000, V, 0.2, False)[0] == "≈ index move only"
    assert cl("Long build-up", 5_000, V, 0.9, False, band=1.5)[0].startswith("≈ inside")
    assert cl("Long build-up", 1_000, V, 0.9, False)[0].startswith("≈ churn")   # 1% of volume
    assert cl("Long build-up", 500, 2_000, 0.9, False)[0].startswith("— too thin")
    assert cl("Long build-up", 5_000, V, 0.9, True)[0].startswith("— not checked")
    assert cl("Long build-up", 5_000, V, None, False)[0] == "— no IV to check"
    assert cl("Quiet", 5_000, V, 0.9, False) == ("—", None)


def test_day_check_removes_slide_and_chain_move():
    # sticky-moneyness smile, index +100 and the WHOLE surface 1 pt cheaper; one strike paid up
    def row(S, shift, bump=None):
        k = []
        for K in range(24600, 25501, 100):
            iv = 20 - 0.002 * (K - S) + shift + ((bump or {}).get(K, 0.0))
            ce = [100.0, 1.0, 1000.0, 10.0, 50_000.0, iv, 99.0, 101.0]
            pe = [100.0, -1.0, 1000.0, 10.0, 50_000.0, iv, 99.0, 101.0]
            k.append([float(K), *ce, *pe])
        return {"t": "2026-11-11T10:00:00", "u": float(S), "a": 25000.0, "e": "2026-11-17", "k": k}
    base, now = row(25000, 0.0), row(25100, -1.0, {25000: 1.2})
    chk = f.day_check(pd.DataFrame(), now, base)
    assert chk["base"] == "close" and chk["common"] == pytest.approx(-1.0, abs=1e-6)
    own, rel = chk["by"][("CE", 25000.0)]
    assert rel == pytest.approx(1.2, abs=1e-6)                      # only the bump is left
    assert abs(chk["by"][("CE", 25300.0)][1]) < 1e-6


def test_close_base_only_previous_session(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "load_close", lambda code, d: (
        {"chain": {"2026-11-17": {"u": 1.0, "v": [[1, 2, 3]]}}} if d == dt.date(2026, 11, 6)
        else {}))
    monkeypatch.setattr(f, "prev_session", lambda code, d: dt.date(2026, 11, 9))
    # the newest snapshot is from 6 Nov but the previous session was 9 Nov: no base
    assert f.close_base("NIFTY", dt.date(2026, 11, 11), "2026-11-17") is None
    monkeypatch.setattr(f, "prev_session", lambda code, d: dt.date(2026, 11, 6))
    assert f.close_base("NIFTY", dt.date(2026, 11, 11), "2026-11-17") is not None


def test_day_check_uses_forward_when_both_rows_have_it():
    def row(S, F, shift=0.0):
        k = []
        for K in range(24600, 25501, 100):
            iv = 20 - 0.002 * (K - F) + shift
            leg = [100.0, 1.0, 1000.0, 10.0, 50_000.0, iv, 99.0, 101.0]
            k.append([float(K), *leg, *leg])
        return {"t": "2026-11-11T10:00:00", "u": float(S), "F": float(F), "a": 25000.0,
                "e": "2026-11-17", "k": k}
    # spot moved +60 but the forward +100: the smile slid with the FORWARD, nothing else
    chk = f.day_check(pd.DataFrame(), row(25060, 25100), row(25000, 25000))
    assert max(abs(v[0]) for v in chk["by"].values()) < 1e-6


def test_net_driver_level_and_who_moved_it():
    S = pd.DataFrame([
        # block 1: calls +500 (writers in, bearish), puts +100 (writers in, bullish)
        dict(K=25000.0, block=1, side="CE", d_oi=500.0, state="Short build-up", lean=-1),
        dict(K=25000.0, block=1, side="PE", d_oi=100.0, state="Short build-up", lean=1),
        # block 2: calls +50, puts -400 (buyers out, bullish) -> puts drove it
        dict(K=25000.0, block=2, side="CE", d_oi=50.0, state="Quiet", lean=0),
        dict(K=25000.0, block=2, side="PE", d_oi=-400.0, state="Long unwinding", lean=1),
        # block 3: only calls logged
        dict(K=25000.0, block=3, side="CE", d_oi=-100.0, state="Long unwinding", lean=-1)])
    nd = f.net_driver(S).set_index("block")
    assert list(nd["level"]) == [400.0, 850.0, 750.0]                 # cumulative calls − puts
    assert nd.loc[1, "side"] == "CE" and nd.loc[1, "lean"] == -1 and nd.loc[1, "code"] == "CW+"
    assert nd.loc[2, "side"] == "PE" and nd.loc[2, "lean"] == 1 and nd.loc[2, "code"] == "PB−"
    assert nd.loc[3, "dP"] == 0.0 and nd.loc[3, "side"] == "CE"       # missing side counts zero
    assert f.net_driver(pd.DataFrame()).empty


def _cells(blocks, big=None, oi=100_000.0):
    """A ladder of 17 strikes x 2 sides: every cell changes 500 contracts, except `big`
    = {(block_index, K, side): d_oi}."""
    recs = []
    t0 = pd.Timestamp("2026-11-11 09:15")
    for i in range(blocks):
        b = t0 + pd.Timedelta(minutes=15 * i)
        for K in range(24300, 26000, 100):
            for side in ("CE", "PE"):
                d = (big or {}).get((i, float(K), side), 500.0)
                recs.append(dict(block=b, t=b + pd.Timedelta(minutes=14), K=float(K),
                                 side=side, oi=oi, d_oi=d, d_px=1.0, d_iv=np.nan, vol=5000.0,
                                 fresh=0.1, state="Long build-up" if d > 0 else "Long unwinding",
                                 lean=(1 if d > 0 else -1) if side == "CE" else
                                 (-1 if d > 0 else 1)))
    return pd.DataFrame(recs)


def test_unusual_blocks_flags_only_the_outsized_change():
    st_ = _cells(6, {(0, 25000.0, "CE"): 90_000.0,          # first block: overnight, never judged
                     (3, 25200.0, "PE"): 40_000.0,          # 80x the ladder's blocks, 2.4% of put OI
                     (4, 25200.0, "PE"): -25_000.0,         # takes most of it back
                     (4, 24300.0, "CE"): 900.0})            # 1.8x: nothing
    al = f.unusual_blocks(st_, 15)
    assert [(r.K, r.side) for r in al.itertuples()] == [(25200.0, "PE"), (25200.0, "PE")]
    r = al.iloc[0]
    assert r["ratio"] == pytest.approx(80.0) and r["share"] == pytest.approx(40_000 / 1.7e6)
    assert bool(r["undone"]) is True and bool(al.iloc[1]["undone"]) is False
    lines = f.alert_lines(al, 25100.0, None, False)
    assert lines[0].startswith("🟢 **10:15 · 25,200 · put buyers out -25,000**")   # newest first
    assert "half taken back next block" in lines[1] and "80× the day's big blocks" in lines[1]
    assert "block still running" in f.alert_lines(al, 25100.0, al["block"].max(), False)[0]
    assert "expiry day" not in f.alert_lines(al, 25100.0, None, True)[0]   # 10:15: not settling
    late = al.copy()
    late["block"] = pd.Timestamp("2026-11-11 14:30")
    assert "expiry day: positions settling" in f.alert_lines(late, 25100.0, None, True)[0]


def test_unusual_blocks_needs_a_baseline_and_skips_gaps_and_thin_wings():
    assert f.unusual_blocks(_cells(2, {(1, 25000.0, "CE"): 90_000.0}), 15).empty   # no yardstick
    g = _cells(6, {(3, 25000.0, "CE"): 40_000.0})
    g.loc[(g["block"] == g["block"].unique()[3]) & (g["K"] == 25000.0) & (g["side"] == "CE"),
          "t"] += pd.Timedelta(minutes=45)                   # that reading came after a hole
    assert f.unusual_blocks(g, 15).empty
    thin = _cells(6, {(3, 24300.0, "PE"): 9_000.0}, oi=100_000.0)   # 18x but 0.5% of put OI
    assert f.unusual_blocks(thin, 15).empty
    assert f.unusual_blocks(pd.DataFrame(), 15).empty
