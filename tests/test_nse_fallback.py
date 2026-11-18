"""Fyers daily-cap handling: the call counter, the cap flag, and the NSE fallback.

Offline: no Fyers and no NSE traffic -- the network edges are monkeypatched."""
import datetime as dt

import pandas as pd
import pytest

from eqbtst import config, live
from eqbtst import nse_fallback as nf


@pytest.fixture(autouse=True)
def _fresh_counter(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "_CALLS_FILE", tmp_path / "fyers_calls.json")
    monkeypatch.setattr(live, "_tradebot_local_calls", lambda: None)
    monkeypatch.setattr(config, "EQBTST_FYERS_BUDGET", 0)
    live._CALLS.clear()
    live._PENDING.update({"calls": 0, "by_kind": {}})
    live._FLUSH["at"] = 0.0
    live._SRC["nse_at"] = None
    monkeypatch.setattr(nf, "CACHE_DIR", tmp_path / "cache")
    nf._memo.clear()
    yield
    live._CALLS.clear()


_CAP = {"code": -360, "s": "error",
        "message": "You've reached the Standard plan limit of 5,000 requests/day."}


def _ms(t: str) -> int:
    """NSE writes IST wall-clock as if UTC: a naive epoch of the IST time."""
    return int(pd.Timestamp(t).value // 1_000_000)


# ── NSE minute series ────────────────────────────────────────────────────────────────────
def test_parse_points_keeps_normal_market_and_places_the_official_close():
    g = [[_ms("2026-11-17 09:05:00"), 100, "PO"],
         [_ms("2026-11-17 09:15:00"), 101, "NM"],
         [_ms("2026-11-17 09:16:00"), 102, "NM"],
         [_ms("2026-11-17 15:15:08"), 103, "NM"],
         [_ms("2026-11-17 15:26:00"), 90, "IC"],          # indicative auction quote: not a trade
         [_ms("2026-11-17 15:31:09"), 104, "NM"],         # official close
         [_ms("2026-11-17 15:59:56"), 104, "NM"]]
    p = nf.parse_points(g)
    assert p["ts"].iloc[0] == pd.Timestamp("2026-11-17 09:15:00")      # no timezone shift
    assert 90 not in p["px"].tolist() and 100 not in p["px"].tolist()
    assert p["ts"].iloc[-1] == pd.Timestamp("2026-11-17 15:29:59")
    assert p["px"].iloc[-1] == 104


def test_bars_from_points_align_to_0915_and_carry_no_volume():
    pts = pd.DataFrame({"ts": pd.to_datetime(["2026-11-17 09:15", "2026-11-17 09:20",
                                              "2026-11-17 09:29", "2026-11-17 09:30"]),
                        "px": [10.0, 12.0, 9.0, 11.0]})
    b = nf.bars_from_points(pts, 15)
    assert b["ts"].tolist() == [pd.Timestamp("2026-11-17 09:15"), pd.Timestamp("2026-11-17 09:30")]
    assert b.iloc[0][["open", "high", "low", "close"]].tolist() == [10, 12, 9, 9]
    assert (b["volume"] == 0).all()


def test_splice_replaces_the_cached_forming_bar_and_keeps_prior_days():
    cached = pd.DataFrame({"ts": pd.to_datetime(["2026-11-16 15:15", "2026-11-17 09:15",
                                                 "2026-11-17 09:30"]),
                           "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 5.0})
    today = pd.DataFrame({"ts": pd.to_datetime(["2026-11-17 09:15", "2026-11-17 09:30",
                                                "2026-11-17 09:45"]),
                          "open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0, "volume": 0.0})
    f = nf.splice(cached, today)
    assert f["ts"].is_unique and len(f) == 4
    assert f.set_index("ts").loc["2026-11-17 09:15", "volume"] == 5.0      # real volume kept
    assert f.set_index("ts").loc["2026-11-17 09:30", "close"] == 2.0       # forming bar replaced


def test_cache_put_merges_and_round_trips():
    a = pd.DataFrame({"ts": pd.to_datetime([dt.date.today()] * 1) + pd.Timedelta("9h15min"),
                      "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0})
    nf.cache_put("ABC", a)
    nf.cache_put("ABC", a.assign(close=1.7))
    got = nf.cache_get("ABC")
    assert len(got) == 1 and got["close"].iloc[0] == 1.7


# ── the counter and the cap ──────────────────────────────────────────────────────────────
def test_cap_reply_is_recorded_and_left_becomes_exactly_zero():
    for _ in range(3):
        live._count_call("history")
    assert live.data_quota_exhausted() is None
    b = live.budget_status()
    assert b["ours"] == 3 and b["left"] == 4997 and not b["left_is_exact"]
    assert live._note_quota(_CAP)
    b = live.budget_status()
    assert b["capped"] and b["left"] == 0 and b["left_is_exact"]
    assert b["ours_at_cap"] == 3 and b["others_at_cap"] == 4997
    assert live._quota_skip()


def test_count_survives_a_restart(tmp_path):
    live._count_call("quotes")
    live._count_call("quotes")
    with live._calls_lock:
        live._flush_calls(force=True)
    live._CALLS.clear()                                  # a fresh process
    live._PENDING.update({"calls": 0, "by_kind": {}})
    assert live.budget_status()["ours"] == 2


def test_fyers_get_sends_nothing_once_capped(monkeypatch):
    live._note_quota(_CAP)
    monkeypatch.setattr(live.requests, "get", lambda *a, **k: pytest.fail("network call sent"))
    with pytest.raises(live.FyersCapped):
        live._fyers_get(config.FYERS_QUOTES_URL, {}, 5, "quotes")


def test_own_budget_switches_off_fyers(monkeypatch):
    monkeypatch.setattr(config, "EQBTST_FYERS_BUDGET", 2)
    live._count_call("history")
    assert not live._quota_skip()
    live._count_call("history")
    assert live._quota_skip() and live.budget_status()["budget_spent"]


def test_yesterdays_cap_does_not_blank_today():
    live._note_quota(_CAP)
    live._CALLS["cap_at"] = (dt.datetime.now() - dt.timedelta(days=1)).isoformat()
    assert live.data_quota_exhausted() is None and not live._quota_skip()


# ── the fallback paths ───────────────────────────────────────────────────────────────────
def test_quotes_fall_back_to_nse_when_capped(monkeypatch):
    live._note_quota(_CAP)
    monkeypatch.setattr(nf, "quotes", lambda syms: {s: {"lp": 10.0, "_src": "NSE"} for s in syms})
    q = live._fetch_quotes(["NSE:AAA-EQ", "NSE:BBB-EQ"])
    assert set(q) == {"NSE:AAA-EQ", "NSE:BBB-EQ"} and live._QUOTE_GAP[0] == 0
    assert live.data_source() == "NSE"


def test_coarse_frame_is_blank_without_cached_history(monkeypatch):
    live._note_quota(_CAP)
    today = pd.Timestamp(dt.date.today())
    monkeypatch.setattr(nf, "minute_points", lambda s: pd.DataFrame(
        {"ts": [today + pd.Timedelta(minutes=555 + i) for i in range(120)], "px": 100.0}))
    assert live.fetch_intraday("ZZZ", tf="1h").empty               # no starved label
    f15 = live.fetch_intraday("ZZZ", tf="15m")
    assert len(f15) == 8 and f15.attrs["src"] == "NSE"


def test_daily_frame_is_archive_plus_todays_nse_bar(monkeypatch):
    live._note_quota(_CAP)
    today = dt.date.today()
    arch = pd.DataFrame({"ts": pd.to_datetime([today - dt.timedelta(days=2),
                                               today - dt.timedelta(days=1)]),
                         "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5})
    monkeypatch.setattr(live, "_daily_hist", lambda: {"ABC": arch})
    monkeypatch.setattr(nf, "quotes", lambda syms: {syms[0]: {
        "lp": 1.8, "open_price": 1.6, "high_price": 1.9, "low_price": 1.55, "volume": 1e6,
        "_upd": f"{today.isoformat()} 15:59:00"}})
    d = live.fetch_intraday("ABC", tf="1D")
    assert len(d) == 3 and d["close"].iloc[-1] == 1.8 and d.attrs["src"] == "NSE"


def test_daily_frame_skips_a_holiday_quote(monkeypatch):
    live._note_quota(_CAP)
    today = dt.date.today()
    arch = pd.DataFrame({"ts": pd.to_datetime([today - dt.timedelta(days=1)]),
                         "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5})
    monkeypatch.setattr(live, "_daily_hist", lambda: {"ABC": arch})
    monkeypatch.setattr(nf, "quotes", lambda syms: {syms[0]: {
        "lp": 1.5, "open_price": 1.0, "_upd": "2020-01-01 15:59:00"}})   # stale print
    assert len(live.fetch_intraday("ABC", tf="1D")) == 1


def test_unknown_vwap_takes_no_side():
    s = {"bar_clr": 0.95, "clr": 0.95, "above_vwap": True, "tone": "strong",
         "rs_vs_index": 1.0, "_vwap_unknown": True}
    assert live._tf_action(s, True) == "NEUTRAL"
    assert live._tf_sell_action({**s, "bar_clr": 0.0, "above_vwap": False,
                                 "tone": "weak", "rs_vs_index": -1}, True) == "—"


def test_scan_does_not_label_coarse_frames_from_today_only(monkeypatch):
    """mtf_structure resamples 15m -> 1h/2h/4h itself; today's NSE bars alone must not
    produce a 1h label (that leak labelled 114 of 259 names on the first live run)."""
    today = pd.Timestamp(dt.date.today())
    f = pd.DataFrame({"ts": [today + pd.Timedelta(minutes=555 + 15 * i) for i in range(25)],
                      "open": 100.0, "close": [100.0 + (i % 3) for i in range(25)],
                      "volume": 0.0})
    f["high"], f["low"] = f["close"] + 1, f["close"] - 1
    f.attrs["src"] = "NSE"
    monkeypatch.setattr(live, "fetch_intraday", lambda *a, **k: f.copy())
    monkeypatch.setattr(live, "_daily_hist", lambda: {})
    out = live.mtf_structure("TODAYONLY_TEST")
    assert out["1h"] == "n/a" and out["2h"] == "n/a" and out["4h"] == "n/a"
    assert out["15m"] != "n/a"


def test_a_later_refusal_restarts_the_skip_window():
    """The skip window must run from the LATEST refusal; the display keeps the FIRST."""
    old = (dt.datetime.now() - dt.timedelta(minutes=30)).isoformat(timespec="seconds")
    live._note_quota(_CAP)
    first = live._CALLS["cap_first"]
    live._CALLS["cap_at"] = old                                 # last refusal 30 min ago
    assert not live._quota_skip()                               # window expired: probe allowed
    live._note_quota(_CAP)                                      # the probe is refused again
    assert live._quota_skip()                                   # window restarted ...
    assert live._read_calls_file()["cap_at"] > old              # ... on disk too
    assert live._CALLS["cap_first"] == first                    # display keeps the first


# ── audit fixes ─────────────────────────────────────────────────────────────────────────
def _five_min_today(vol=1000.0):
    t0 = pd.Timestamp(dt.date.today()) + pd.Timedelta("9h15min")
    ts = [t0 + pd.Timedelta(minutes=5 * i) for i in range(75)]
    f = pd.DataFrame({"ts": ts, "open": 100.0, "high": 101.0, "low": 99.0,
                      "close": 100.0, "volume": vol})
    return f


def test_no_close_estimate_under_the_closing_auction(monkeypatch):
    monkeypatch.setattr(live, "fetch_intraday", lambda *a, **k: _five_min_today())
    live._SESS_CACHE.clear()
    assert live._session_extras("AUCT1", 100.0).get("close30") is None
    monkeypatch.setattr(config, "CLOSING_AUCTION_FROM", dt.date(2999, 1, 1))   # pre-auction era
    live._SESS_CACHE.clear()
    assert live._session_extras("AUCT2", 100.0).get("close30") is not None


def test_bare_index_is_requested_as_an_index(monkeypatch):
    seen = {}

    def fake(url, params, timeout, kind):
        seen["sym"] = params["symbol"]
        return {"s": "no_data"}
    monkeypatch.setattr(live, "_fyers_get", fake)
    monkeypatch.setattr(live, "_hist_pace", lambda: None)
    live.fetch_intraday("NIFTY50", tf="5m", lookback_days=2)
    assert seen["sym"] == "NSE:NIFTY50-INDEX"
    live.fetch_intraday("RELIANCE", tf="5m", lookback_days=2)
    assert seen["sym"] == "NSE:RELIANCE-EQ"


def _bars(days):
    rows = []
    for d in days:
        for i in range(25):
            rows.append({"ts": pd.Timestamp(d) + pd.Timedelta("9h15min") + pd.Timedelta(minutes=15 * i),
                         "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("age,keep", [(1, True), (6, False)])
def test_stale_prior_history_is_not_spliced(monkeypatch, age, keep):
    today = dt.date.today()
    prev = today - dt.timedelta(days=1)
    from eqbtst import paper_hz
    monkeypatch.setattr(paper_hz, "load_bars", lambda s, tf: _bars([today - dt.timedelta(days=age)]))
    monkeypatch.setattr(live, "_prev_session", lambda t: prev)
    live._PRIOR_MEMO.clear()
    assert (not live._prior_15m(f"STALE{age}").empty) is keep


def test_staleness_check_uses_nses_calendar_under_the_fallback(monkeypatch):
    from eqbtst import data as _data
    today = pd.Timestamp(dt.date.today())
    arch = today - pd.Timedelta(days=3)
    monkeypatch.setattr(_data, "last_trading_date", lambda: arch)
    f = pd.DataFrame({"ts": [arch], "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                      "volume": 0.0})
    f.attrs["src"] = "NSE"
    monkeypatch.setattr(live, "fetch_intraday", lambda *a, **k: f)
    monkeypatch.setattr(live, "drop_phantom_bars", lambda x: x)
    monkeypatch.setattr(nf, "session_dates", lambda s="RELIANCE": pd.DataFrame(
        {"ts": [arch, arch + pd.Timedelta(days=1), arch + pd.Timedelta(days=2), today]}))
    live._STALE_CACHE.clear()
    out = live.archive_staleness()
    live._STALE_CACHE.clear()
    assert out["ok"] and out["stale_days"] == 2          # not a self-comparison "0"


def test_memo_fetches_once_under_concurrency():
    import threading
    calls = []

    def slow():
        calls.append(1)
        import time as _t
        _t.sleep(0.05)
        return 42
    ths = [threading.Thread(target=lambda: nf._memo_get("herd", 60, slow)) for _ in range(8)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    assert len(calls) == 1
