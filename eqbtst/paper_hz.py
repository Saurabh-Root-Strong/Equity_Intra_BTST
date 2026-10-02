"""
paper_hz.py — the Paper-trade lane on ALL FOUR trade horizons, on a focused high-volatility
Nifty 100 universe, with the Upper-TF S/R read the live board filters on.

WHY A SECOND ENGINE AND NOT MORE ROWS IN paper.FRAMES. paper.py was built on one premise:
"the intraday pairs need broker history, which serves ~60 days". That premise is false for
this machine -- Tradebot's download_historical.py already holds TWO YEARS of 15-minute bars
for the whole F&O universe (verified: RELIANCE 2024-06-14 -> 2026-06-12, 494 sessions), and
the broker tops the tail up in 90-day chunks. So 15m/1h/4h can be simulated honestly. What
changes with them is not the arithmetic but the CLOCK: bars close at different times on each
frame, a session ends, and an Intraday trade must be squared off. This module owns that clock.

THE UNIVERSE: the 20 most volatile names in the Nifty 100, by ATR% (mean true range / close,
trailing 120 sessions). Not beta, not "high-beta" blog lists -- ATR% is the unit a stop and a
target are measured in, and scalp.py measured it ranking realised move at Spearman +0.90.
Two honest modes:
  * POINT-IN-TIME (default) -- re-ranked every month on data that ENDED before the month
    began. A name only trades in months it was genuinely in the top 20 at the time.
  * TODAY'S LIST -- the current 20 across the whole history. LOOK-AHEAD: those names are in
    the list BECAUSE they were volatile, so the past is judged by a filter it could not know.
    Offered because "which trades did my 20 names give" is a fair question; labelled as biased.
Both inherit one bias neither can remove: the Nifty 100 list is TODAY's membership
(survivorship). NSE publishes no free historical constituent file.

THE CLOCK -- ONE RULE FOR EVERY FRAME. Each bar carries an `end` timestamp. A higher-frame bar
is visible at trigger bar i only if its end <= trigger bar i's end. So a 4h bar that opened at
09:15 is invisible to the 10:15 1h bar, a weekly bar is visible from its Friday 15:30, a
monthly bar from its last calendar day. Weekly/monthly ends are PERIOD ends, not "last row in
the data" -- otherwise the running month would count as closed on the archive's last day.

FILLS, AND THE TWO CHEATS THIS FIXES VERSUS THE OLD WALK:
  * GAPS. A stop is a stop-market order: if the bar OPENS beyond it, you are filled at the open,
    not at the stop. The old walk filled at the stop price, i.e. it capped every overnight gap
    loss at exactly -1R -- on the high-volatility names this page is now about, the single most
    flattering assumption available. Symmetrically a target gapped through fills at the open.
  * THE END OF THE TAPE. A trade still running when the data ends is OPEN, not a time-stop exit.
    It is marked to market in its own table and excluded from every statistic.
Unchanged: entry at the next bar's OPEN; a bar containing both stop and target is a STOP;
one position per name at a time; cost charged in R.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data, indicators, mtf

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_CSV = ROOT / "data" / "universe" / "nifty100.csv"
NIFTY100_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv"
CACHE_15M = ROOT / "data" / "intraday15"
TRADEBOT_15M = Path(config.TRADEBOT_DIR) / "data" / "historical" / "15min"
# Where each intraday source lives: Tradebot's 2-year download, plus this project's top-up cache.
_BAR_DIRS = {"15m": (TRADEBOT_15M, CACHE_15M),
             "5m": (Path(config.TRADEBOT_DIR) / "data" / "historical" / "5min",
                    ROOT / "data" / "intraday5")}
_BAR_MIN = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}

N_NAMES = 20
VOL_WINDOW = 120                         # sessions of ATR% behind the ranking
SESSION_END = pd.Timedelta("15h30min")

HORIZONS = {k: {"label": v["label"], "ltf": v["ltf"], "htf": v["htf"],
                "up": mtf.one_frame_up(v["htf"]), "hold": v["hold"]}
            for k, v in mtf.PRESETS.items()}
# SML-INTRADAY -- 5m trigger / 15m confirm, Upper-TF 1h (one rung up the same ladder: the
# frame above 15m is 1h, exactly as 1h -> 4h for Intraday). A paper-lane horizon only: the live
# board's structure scan is built from one 15-minute fetch and has no 5m frame, so adding it to
# mtf.PRESETS would put a horizon on the board that the board cannot compute.
# The pair is 3x, not the 4x the other presets nest -- 5m inside 20m would need a frame no chart
# or broker uses; 15m is the frame traders actually confirm a 5m entry on.
HORIZONS = {"sml_intraday": {"label": "Sml-Intraday  ·  5m trigger / 15m confirm", "ltf": "5m",
                             "htf": "15m", "up": mtf.one_frame_up("15m"),
                             "hold": "square off same day, minutes to an hour"},
            **HORIZONS}
HORIZON_ORDER = ["sml_intraday", *mtf.PRESET_ORDER]
# Which bars each horizon is built from. Every coarser frame is RESAMPLED from that one source,
# so all frames of a horizon are the same series at different resolutions.
SOURCE = {"sml_intraday": "5m", "intraday": "15m", "btst": "15m", "swing": "15m",
          "positional": "1D"}
# Time stop in TRIGGER bars. BTST: six 1h bars = one session (the day's last 1h bar spans
# 14:15-15:30 after the stub merge), so "enter today, out by the same hour tomorrow".
# Sml-Intraday: twelve 5m bars = one hour.
DEFAULT_HOLD = {"sml_intraday": 12, "intraday": 12, "btst": 6, "swing": 10, "positional": 20}
HOLD_RANGE = {"sml_intraday": (3, 60), "intraday": (2, 25), "btst": (2, 30), "swing": (2, 40),
              "positional": (5, 60)}
# Intraday is squared off at the session's last bar: carrying it turns an MIS trade into
# delivery (STT both legs) and a same-day stop into an overnight gap.
SQUARE_OFF = {"sml_intraday": True, "intraday": True, "btst": False, "swing": False,
              "positional": False}
# The same S/R windows the board uses per frame (live.mtf_structure / _set).
_SR_LB = {"1D": config.SR_DAILY_LOOKBACK}
_UP_LB = {"4h": config.SR_LOOKBACK, "1D": config.SR_DAILY_LOOKBACK, "1W": config.SR_LOOKBACK,
          "1M": config.SR_LOOKBACK}
_STRUCT_TAIL = config.STRUCT_LOOKBACK + 20     # struct_full reads only the last lookback bars
ROOM_BUCKETS = ("✅ Has room", "⚠ Tight", "🧱 Capped", "n/a")


# ── universe ─────────────────────────────────────────────────────────────────────────
def nifty100(refresh: bool = False) -> list[str]:
    """Current Nifty 100 constituents: NSE's own file, cached under data/universe/.

    A failed refresh falls back to the cached copy -- the list changes twice a year, so a
    stale list is a far smaller error than an empty page."""
    if refresh or not UNIVERSE_CSV.exists():
        try:
            import requests
            r = requests.get(NIFTY100_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            r.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            if "Symbol" in df.columns and len(df) >= 90:
                UNIVERSE_CSV.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(UNIVERSE_CSV, index=False)
        except Exception:                                      # noqa: BLE001
            pass
    if not UNIVERSE_CSV.exists():
        return []
    return pd.read_csv(UNIVERSE_CSV)["Symbol"].astype(str).str.strip().tolist()


def universe_asof() -> str | None:
    return (dt.datetime.fromtimestamp(UNIVERSE_CSV.stat().st_mtime).strftime("%Y-%m-%d")
            if UNIVERSE_CSV.exists() else None)


def daily_adjusted(symbols, start: str = "2018-01-01", end: str | None = None) -> dict:
    """{sym: daily OHLC (ts, open, high, low, close)} from the EOD archive, SPLIT-ADJUSTED.

    The archive is unadjusted (KOTAKBANK 2132 -> 421 across its 1:5). Unadjusted, a split is a
    -80% bar: a short opened the day before books a gigantic fake target, a long a fake gap
    stop, and every level and structure label near it is fiction. paper.simulate reads the raw
    archive -- this lane does not."""
    # A DCM writer (sync, backfill, its dashboard) holds DuckDB exclusively for seconds at a
    # time. Retry briefly before giving up, so one write burst does not cost a whole run.
    import time
    for _k in range(12):
        try:
            df = data.load_eod(start=start, end=end)
            break
        except RuntimeError:
            if _k == 11:
                raise
            time.sleep(5)
    df = df[df["symbol"].isin(list(symbols))]
    out = {}
    for s, g in df.groupby("symbol"):
        g = (g.sort_values("trade_date")
              .rename(columns={"trade_date": "ts", "open_price": "open", "high_price": "high",
                               "low_price": "low", "close_price": "close"})
              [["ts", "open", "high", "low", "close"]].reset_index(drop=True))
        g["ts"] = pd.to_datetime(g["ts"]).dt.normalize()
        out[s] = indicators.adjust_corporate_actions(g)
    return out


def _tr_pct(d: pd.DataFrame) -> pd.Series:
    c = d["close"].astype(float)
    pc = c.shift()
    tr = np.maximum(d["high"] - d["low"],
                    np.maximum((d["high"] - pc).abs(), (d["low"] - pc).abs()))
    return pd.Series((tr / c).to_numpy(), index=d["ts"])


def vol_rank(daily: dict, as_of=None, n: int = N_NAMES, window: int = VOL_WINDOW) -> pd.DataFrame:
    """Rank by mean ATR% over the `window` sessions STRICTLY BEFORE as_of.

    A name needs 80% of the window to be ranked at all -- a fresh listing's first weeks are
    not a volatility estimate, they are an IPO."""
    cut = pd.Timestamp(as_of) if as_of is not None else None
    rows = []
    for s, d in daily.items():
        x = _tr_pct(d)
        if cut is not None:
            x = x[x.index < cut]
        x = x.dropna().tail(window)
        if len(x) < 0.8 * window:
            continue
        rows.append({"symbol": s, "atr%": round(100 * float(x.mean()), 2),
                     "close": float(d["close"].iloc[-1])})
    r = pd.DataFrame(rows)
    if r.empty:
        return r
    r = r.sort_values("atr%", ascending=False).reset_index(drop=True)
    r["rank"] = np.arange(1, len(r) + 1)
    return r.head(n)


def pit_members(daily: dict, start, end, n: int = N_NAMES, window: int = VOL_WINDOW) -> dict:
    """{month Period: set(symbols)} -- each month's top n, ranked on data before it began."""
    out = {}
    for m in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M"):
        r = vol_rank(daily, as_of=m.start_time, n=n, window=window)
        out[m] = set(r["symbol"]) if not r.empty else set()
    return out


# ── intraday history (15m and 5m) ────────────────────────────────────────────────────
def load_15m(sym: str) -> pd.DataFrame:
    return load_bars(sym, "15m")


def load_bars(sym: str, tf: str = "15m") -> pd.DataFrame:
    """Tradebot's 2-year download UNION this project's top-up cache, de-duplicated.

    De-duplication is not hygiene: the broker returns byte-identical duplicate candles when a
    range starts and ends on one date, and a doubled bar halves every structure window."""
    parts = []
    tb, cache = _BAR_DIRS[tf]
    for p in (tb / f"NSE_{sym}_EQ_{_BAR_MIN[tf]}min.parquet", cache / f"{sym}.parquet"):
        if p.exists():
            try:
                parts.append(pd.read_parquet(p))
            except Exception:                                  # noqa: BLE001
                pass
    if not parts:
        return pd.DataFrame()
    f = pd.concat(parts, ignore_index=True)
    f["ts"] = pd.to_datetime(f["ts"])
    f = (f.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True))
    t = f["ts"] - f["ts"].dt.normalize()
    # SESSION ONLY. The 5m download carries after-hours prints to 18:55 -- not bars anyone
    # could trade, and a structure window that includes them reads a different market.
    f = f[(t >= pd.Timedelta("9h15min")) & (t < SESSION_END)]
    if "volume" not in f.columns:
        f["volume"] = 0.0
    return f[["ts", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def topup_15m(symbols, now: dt.datetime | None = None, progress=None) -> dict:
    return topup_bars(symbols, "15m", now=now, progress=progress)


def topup_bars(symbols, tf: str = "15m", now: dt.datetime | None = None, progress=None) -> dict:
    """Fetch the missing tail of each name's 15m / 5m history from the broker (needs a token).

    90-day chunks (the broker's minute-resolution cap is 100). A candle that has not finished
    printing is dropped -- a forming bar stored as history is a bar that never existed."""
    import requests
    from . import live
    now = now or dt.datetime.now()
    cache = _BAR_DIRS[tf][1]
    mins = _BAR_MIN[tf]
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, sym in enumerate(symbols):
        f = load_bars(sym, tf)
        start = (f["ts"].max().date() if not f.empty else now.date() - dt.timedelta(days=700))
        new = []
        d0 = start
        while d0 <= now.date():
            d1 = min(d0 + dt.timedelta(days=89), now.date())
            live._hist_pace()
            try:
                r = requests.get(config.FYERS_HISTORY_URL,
                                 headers={"Authorization": live._auth_header(), "version": "3"},
                                 params={"symbol": live.fy_symbol(sym), "resolution": str(mins),
                                         "date_format": "1", "range_from": d0.isoformat(),
                                         "range_to": d1.isoformat(), "cont_flag": "1"},
                                 timeout=20).json()
            except Exception:                                  # noqa: BLE001
                r = {}
            if r.get("s") == "ok" and r.get("candles"):
                x = pd.DataFrame(r["candles"],
                                 columns=["ts", "open", "high", "low", "close", "volume"])
                x["ts"] = (pd.to_datetime(x["ts"], unit="s", utc=True)
                           .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None))
                new.append(x)
            d0 = d1 + dt.timedelta(days=1)
        added = 0
        if new:
            x = pd.concat(new, ignore_index=True)
            x = x[x["ts"] + pd.Timedelta(minutes=mins) <= pd.Timestamp(now)]
            p = cache / f"{sym}.parquet"
            old = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=x.columns)
            m = (pd.concat([old, x], ignore_index=True).drop_duplicates("ts", keep="last")
                   .sort_values("ts").reset_index(drop=True))
            added = len(m) - len(old)
            m.to_parquet(p, index=False)
        f2 = load_bars(sym, tf)
        out[sym] = {"added": int(added), "last": (str(f2["ts"].max()) if not f2.empty else None)}
        if progress:
            progress((k + 1) / max(len(symbols), 1), sym)
    return out


# ── frames and their clocks ──────────────────────────────────────────────────────────
def _intraday_ends(ts: pd.Series) -> np.ndarray:
    """A bar ends where the next bar of the SAME session starts, else at 15:30. The day's last
    1h bar (14:15, stub-merged) therefore ends at 15:30, as does the 13:15 4h bar."""
    ts = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)
    nxt = ts.shift(-1)
    same = nxt.dt.normalize() == ts.dt.normalize()
    return nxt.where(same, ts.dt.normalize() + SESSION_END).to_numpy()


def _period_frame(d: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Daily -> weekly ('W-FRI') or monthly ('M') bars, GROUPED (a holiday week is absent, not
    an empty NaN row) and stamped with the PERIOD's end -- Friday / month-end 15:30 -- so the
    running period is never visible before it is over."""
    x = d.copy()
    x["p"] = x["ts"].dt.to_period(freq)
    g = x.groupby("p", sort=True)
    out = pd.DataFrame({"ts": g["ts"].min(), "open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last()}).reset_index()
    out["end"] = out["p"].apply(lambda p: p.end_time.normalize() + SESSION_END)
    return out.drop(columns="p").reset_index(drop=True)


def build_frames(hz: str, daily_d: pd.DataFrame | None, fbase: pd.DataFrame | None) -> dict:
    """{tf: bars with an `end` column} for the horizon's trigger, confirm and one-up frames.

    `fbase` is the horizon's intraday source (SOURCE[hz]: 5m or 15m); every coarser intraday
    frame is resampled from it with the board's session-aligned, stub-merged resample."""
    from . import live
    need = {HORIZONS[hz]["ltf"], HORIZONS[hz]["htf"], HORIZONS[hz]["up"]}
    src = SOURCE.get(hz, "15m")
    fr = {}
    if need & set(_BAR_MIN):
        if fbase is None or fbase.empty:
            return {}
        f = indicators.adjust_corporate_actions(fbase)
        for tf in sorted(need & set(_BAR_MIN), key=_BAR_MIN.get):
            freq = None if tf == src else f"{_BAR_MIN[tf]}min"
            b = live._resample_ohlcv(f, freq).reset_index(drop=True)
            b["end"] = _intraday_ends(b["ts"])
            fr[tf] = b
    if need & {"1D", "1W", "1M"}:
        if daily_d is None or daily_d.empty:
            return {}
        if "1D" in need:
            b = daily_d.copy()
            b["end"] = b["ts"] + SESSION_END
            fr["1D"] = b
        if "1W" in need:
            fr["1W"] = _period_frame(daily_d, "W-FRI")
        if "1M" in need:
            fr["1M"] = _period_frame(daily_d, "M")
    return fr


# ── signals ──────────────────────────────────────────────────────────────────────────
class _Col:
    __slots__ = ("a",)

    def __init__(self, a):
        self.a = a

    def to_numpy(self, dtype=None):
        return self.a


class _View:
    """A zero-copy window over a frame's OHLC arrays, shaped like the DataFrame that
    indicators.struct_full reads (`.columns`, `[col].to_numpy()`, `len`).

    It exists for speed only -- DataFrame.iloc was 70% of the runtime at 12k 15m bars a name --
    and it lets the simulator keep calling the REAL struct_full rather than a vectorised copy
    that could drift from the board. test_paper_hz pins that both give identical output."""
    columns = ("open", "high", "low", "close")
    __slots__ = ("arr", "lo", "hi")

    def __init__(self, arr: dict, lo: int, hi: int):
        self.arr, self.lo, self.hi = arr, max(0, lo), hi

    def __getitem__(self, k):
        return _Col(self.arr[k][self.lo:self.hi])

    def __len__(self):
        return self.hi - self.lo


def _arrays(b: pd.DataFrame) -> dict:
    return {k: b[k].to_numpy(float) for k in ("open", "high", "low", "close")}


def _sr_match(kl, kh, spot: float, a: float, tol_bps: float, near_atr: float):
    """The board's confluence rule (live.add_setup): a trigger-frame level within near_atr
    ATR of price, matched by a SAME-SIDE confirm-frame level within tol_bps. Nearest wins.
    Returns (side, level) or None."""
    tol, near, best = tol_bps / 1e4 * spot, near_atr * a, None
    for dn in (True, False):
        for x, _t, _nl, _nh in kl:
            gap = (spot - x) if dn else (x - spot)
            if gap < 0 or gap > near:
                continue
            same = [z for z in kh if ((z[0] <= spot) if dn else (z[0] >= spot))]
            if not same:
                continue
            m = min(same, key=lambda z: abs(z[0] - x))
            if abs(m[0] - x) > tol:
                continue
            if best is None or gap < best[0]:
                best = (gap, x, "LONG" if dn else "SHORT")
    return None if best is None else (best[2], best[1])


def room_bucket(gap: float) -> str:
    """The board's Upper-TF S/R thresholds (dashboard _quality_filter), in trigger-frame ATR."""
    if gap is None or gap != gap:
        return "n/a"
    if np.isinf(gap) or gap >= 1.0:
        return ROOM_BUCKETS[0]
    return ROOM_BUCKETS[1] if gap >= 0.5 else ROOM_BUCKETS[2]


def signals(fr: dict, hz: str, lens: str = "STRUCTURE", start=None,
            conf_tol_bps: float | None = None, near_atr: float | None = None) -> list[dict]:
    """Every trigger bar (from `start`) at which the lens takes a side, with the Upper-TF wall.

    Includes the LAST bar: a signal there cannot fill yet, and is reported as PENDING -- the
    trade the page would tell you to take at the next open."""
    conf_tol_bps = float(config.SR_CONF_TOL_BPS if conf_tol_bps is None else conf_tol_bps)
    near_atr = float(config.SR_CONF_NEAR_ATR if near_atr is None else near_atr)
    h = HORIZONS[hz]
    L, H, U = fr.get(h["ltf"]), fr.get(h["htf"]), fr.get(h["up"])
    if L is None or H is None or len(L) < _STRUCT_TAIL + 2:
        return []
    le = L["end"].to_numpy()
    nh = np.searchsorted(H["end"].to_numpy(), le, side="right")
    nu = (np.searchsorted(U["end"].to_numpy(), le, side="right") if U is not None
          else np.zeros(len(L), dtype=int))
    lb_l, lb_h = _SR_LB.get(h["ltf"], config.SR_LOOKBACK), _SR_LB.get(h["htf"], config.SR_LOOKBACK)
    lb_u = _UP_LB.get(h["up"], config.SR_LOOKBACK)
    i0 = _STRUCT_TAIL
    if start is not None:
        i0 = max(i0, int(np.searchsorted(L["ts"].to_numpy(), np.datetime64(pd.Timestamp(start)))))
    close = L["close"].to_numpy(float)
    ts = L["ts"].to_numpy()
    out = []
    la, ha = _arrays(L), _arrays(H)
    pf_h, pf_l = indicators.pivot_flags(la["high"], la["low"]) if lens == "SR" else (None, None)
    hc, uc, hk = (None, None), (None, None), (None, None)
    for i in range(i0, len(L)):
        n_h = int(nh[i])
        if n_h < 8:
            continue
        if hc[0] != n_h:
            hc = (n_h, indicators.struct_full(_View(ha, n_h - _STRUCT_TAIL, n_h)))
        ls = indicators.struct_full(_View(la, i + 1 - _STRUCT_TAIL, i + 1))
        a = float(ls.get("atr") or 0)
        spot = float(close[i])
        if a <= 0 or spot <= 0:
            continue
        if lens == "STRUCTURE":
            s = mtf.synthesize(hc[1], ls, spot)
            side = mtf.side_of(s["tag"], s.get("dir", "NONE"))
            if side not in ("LONG", "SHORT"):
                continue
            tag, lvl = s["tag"], float("nan")
        else:
            # THE TRIGGER FRAME'S LEVELS, WITHOUT sr_levels' DISPLAY EXTRAS. sr_levels builds
            # levels_kind as walls_kind(last lookback highs/lows, SR_TOL_ATR x ATR) on the same
            # 14-bar ATR struct_full just returned (frames are already split-adjusted); the
            # zone-visit and headroom work around it cost ~3ms a bar and are never read here.
            # test_paper_hz pins the two as identical.
            _lo = max(0, i + 1 - lb_l)
            if i + 1 - _lo >= 12:
                _a, _b = _lo + 2, i + 1 - 2           # bars whose +/-2 neighbours sit in the window
                kl = indicators.walls_kind_from(la["high"][_a:_b][pf_h[_a:_b]],
                                                la["low"][_a:_b][pf_l[_a:_b]],
                                                config.SR_TOL_ATR * a)
            else:
                kl = []
            if hk[0] != n_h:
                hk = (n_h, indicators.sr_levels(H.iloc[max(0, n_h - lb_h):n_h], lookback=lb_h))
            kh = hk[1].get("levels_kind") or []
            if not kl or not kh:
                continue
            m = _sr_match(kl, kh, spot, a, conf_tol_bps, near_atr)
            if m is None:
                continue
            side, lvl = m
            tag = "SR " + ("SUP" if side == "LONG" else "RES")
        # ── UPPER-TF S/R: the one frame above the pair, CLOSED bars only ──
        n_u = int(nu[i])
        if U is not None and n_u >= 12:
            if uc[0] != n_u:
                uc = (n_u, indicators.sr_levels(U.iloc[max(0, n_u - lb_u):n_u], lookback=lb_u))
            sr = uc[1]
            if sr:
                w, g, _x = indicators.big_wall(
                    spot, a, side != "SHORT",
                    [(h["up"], sr.get("levels_ext") or sr.get("levels"), sr.get("blind"))])
            else:
                w, g = "", float("nan")
        else:
            w, g = "", float("nan")
        out.append({"i": i, "ts": ts[i], "side": side, "tag": tag, "level": lvl, "atr": a,
                    "spot": spot, "up_wall": w, "up_gap": g, "room": room_bucket(g)})
    return out


# ── trade management ─────────────────────────────────────────────────────────────────
def walk(o, hi, lo, cl, i: int, side: str, stop_atr: float, rr: float, max_bars: int,
         atr: float, last_i: int | None = None) -> dict | None:
    """Fill at bar i+1's OPEN; walk to a gap, the stop, the target, the time stop or last_i.

    `last_i` is the hard exit bar (the session's last bar for an Intraday square-off). Running
    off the end of the data returns why='open' -- the trade is not over."""
    n = len(o)
    j = i + 1
    if j >= n or atr <= 0:
        return None
    entry = float(o[j])
    risk = stop_atr * atr
    if entry <= 0 or risk <= 0:
        return None
    long = side == "LONG"
    stop = entry - risk if long else entry + risk
    targ = entry + rr * risk if long else entry - rr * risk
    sgn = 1.0 if long else -1.0
    end = j + max_bars - 1
    forced = last_i is not None and last_i < end
    if forced:
        end = last_i
    for k in range(j, min(end, n - 1) + 1):
        ok = float(o[k])
        if k > j:
            # A GAP THROUGH THE STOP FILLS AT THE OPEN. A stop is a market order once touched,
            # and an open beyond it is the first price anyone could trade.
            if (ok <= stop) if long else (ok >= stop):
                return {"exit_i": k, "exit_px": ok, "R": round(sgn * (ok - entry) / risk, 3),
                        "why": "gap-stop", "entry": entry, "stop": stop, "target": targ}
            if (ok >= targ) if long else (ok <= targ):
                return {"exit_i": k, "exit_px": ok, "R": round(sgn * (ok - entry) / risk, 3),
                        "why": "gap-target", "entry": entry, "stop": stop, "target": targ}
        # BOTH IN ONE BAR -> THE STOP. No intrabar path; the optimistic resolution is how a
        # backtest invents an edge.
        if (float(lo[k]) <= stop) if long else (float(hi[k]) >= stop):
            return {"exit_i": k, "exit_px": stop, "R": -1.0, "why": "stop",
                    "entry": entry, "stop": stop, "target": targ}
        if (float(hi[k]) >= targ) if long else (float(lo[k]) <= targ):
            return {"exit_i": k, "exit_px": targ, "R": float(rr), "why": "target",
                    "entry": entry, "stop": stop, "target": targ}
    k = min(end, n - 1)
    px = float(cl[k])
    why = "open" if end > n - 1 else ("session" if forced else "time")
    return {"exit_i": k, "exit_px": px, "R": round(sgn * (px - entry) / risk, 3), "why": why,
            "entry": entry, "stop": stop, "target": targ}


def _session_last(ts) -> np.ndarray:
    d = pd.to_datetime(pd.Series(ts)).dt.normalize().to_numpy()
    idx = np.arange(len(d))
    last = pd.Series(idx).groupby(d).transform("max").to_numpy()
    return last


# ── the run ──────────────────────────────────────────────────────────────────────────
def build_bundle(hz: str, lens: str = "STRUCTURE", years: float = 2.0, n: int = N_NAMES,
                 universe_mode: str = "pit", end=None, progress=None) -> dict:
    """Generate every signal once for the horizon (exit rules are applied later, by walk_bundle).

    universe_mode 'pit' = point-in-time monthly top-n; 'today' = today's top-n throughout."""
    syms = nifty100()
    try:
        daily = daily_adjusted(syms, start="2016-01-01", end=None if end is None else str(end))
    except Exception as e:                                     # noqa: BLE001
        # DuckDB allows many readers OR one writer. A DCM sync or its dashboard holding the
        # file must cost one run with a readable reason, not a traceback on the page.
        return {"sig": {}, "meta": {"error": f"EOD archive unavailable: {str(e)[:200]}"}}
    if not daily:
        return {"sig": {}, "meta": {"error": "EOD archive returned nothing (DuckDB lock?)"}}
    last_daily = max(d["ts"].max() for d in daily.values())
    today_top = vol_rank(daily, as_of=last_daily + pd.Timedelta(days=1), n=n)
    h = HORIZONS[hz]
    src = SOURCE.get(hz, "15m")
    intraday_src = src != "1D"
    # WHERE THE TAPE ENDS decides the window. For the 15m horizons it is the 15m history, not the
    # archive -- asking for 3 years of 1h bars when 2 exist would silently return 2.
    if universe_mode == "today":
        names = list(today_top["symbol"])
    else:
        names = None                                  # decided after the window is known
    f15s, ends = {}, []
    if intraday_src:
        cand = names if names is not None else list(daily)
        for s in cand:
            f = load_bars(s, src)
            if not f.empty:
                f15s[s] = f
                ends.append(f["ts"].max())
        tape_end = max(ends) if ends else None
        tape_start = min(f["ts"].min() for f in f15s.values()) if f15s else None
    else:
        tape_end, tape_start = last_daily, min(d["ts"].min() for d in daily.values())
    if tape_end is None:
        return {"sig": {}, "meta": {"error": f"no {src} history found", "hz": hz}}
    start = max(pd.Timestamp(tape_end) - pd.Timedelta(days=int(365.25 * float(years))) if years else tape_start,
                pd.Timestamp(tape_start))
    if universe_mode == "today":
        members = None
    else:
        members = pit_members(daily, start, tape_end, n=n)
        names = sorted(set().union(*members.values())) if members else []
    sig = {}
    missing = []
    for k, s in enumerate(names):
        fr = build_frames(hz, daily.get(s), f15s.get(s) if intraday_src else None)
        if not fr:
            missing.append(s)
            continue
        got = signals(fr, hz, lens=lens, start=start)
        L = fr[h["ltf"]]
        sig[s] = {"o": L["open"].to_numpy(float), "h": L["high"].to_numpy(float),
                  "l": L["low"].to_numpy(float), "c": L["close"].to_numpy(float),
                  "ts": L["ts"].to_numpy(), "sess_last": _session_last(L["ts"]),
                  "sig": got}
        if progress:
            progress((k + 1) / max(len(names), 1), s)
    return {"sig": sig, "members": members, "meta": {
        "hz": hz, "lens": lens, "ltf": h["ltf"], "htf": h["htf"], "up": h["up"],
        "universe_mode": universe_mode, "n": n, "names": names, "missing": missing,
        "today_top": today_top, "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(tape_end)), "archive_end": str(pd.Timestamp(last_daily).date()),
        "universe_asof": universe_asof()}}


def walk_bundle(bundle: dict, rr: float = 2.0, stop_atr: float = 1.0, max_bars: int | None = None,
                cost_bps: float = config.COST_BPS, sides=("LONG", "SHORT"),
                room: str = "All") -> pd.DataFrame:
    """Signals -> trades under one exit rule and one Upper-TF filter.

    The room filter is applied HERE, before the one-position guard, because it decides which
    signals are TAKEN -- and a skipped signal leaves the name free for the next one."""
    if not bundle.get("sig"):
        return pd.DataFrame()
    hz = bundle["meta"]["hz"]
    max_bars = int(max_bars or DEFAULT_HOLD[hz])
    members = bundle.get("members")
    rows = []
    for sym, b in bundle["sig"].items():
        free = -1
        n = len(b["o"])
        for s in b["sig"]:
            i = s["i"]
            if s["side"] not in sides or i <= free:
                continue
            if room != "All" and s["room"] != room:
                continue
            if members is not None:
                m = pd.Timestamp(s["ts"]).to_period("M")
                if sym not in members.get(m, ()):
                    continue
            base = {"symbol": sym, "side": s["side"], "setup": s["tag"],
                    "signal_ts": pd.Timestamp(s["ts"]), "upper_tf": s["up_wall"] or "∞ clear",
                    "up_gap_atr": s["up_gap"], "room": s["room"]}
            if SQUARE_OFF[hz] and int(b["sess_last"][i]) == i:
                continue                      # an Intraday signal on the day's last bar has no session left
            if i >= n - 1:
                # THE SIGNAL THAT HAS NOT FILLED YET. The one row a trader acts on.
                risk = stop_atr * s["atr"]
                px = s["spot"]
                rows.append({**base, "status": "PENDING", "entry_ts": pd.NaT,
                             "entry": round(px, 2),
                             "stop": round(px - risk if s["side"] == "LONG" else px + risk, 2),
                             "target": round(px + rr * risk if s["side"] == "LONG"
                                             else px - rr * risk, 2),
                             "risk%": round(100 * risk / px, 2), "why": "fills next open"})
                free = n
                continue
            last_i = int(b["sess_last"][i + 1]) if SQUARE_OFF[hz] else None
            tr = walk(b["o"], b["h"], b["l"], b["c"], i, s["side"], stop_atr, rr, max_bars,
                      s["atr"], last_i=last_i)
            if tr is None:
                continue
            free = tr["exit_i"]
            risk = stop_atr * s["atr"]
            cost_r = (cost_bps / 1e4 * tr["entry"]) / risk
            rows.append({**base, "status": "OPEN" if tr["why"] == "open" else "CLOSED",
                         "entry_ts": pd.Timestamp(b["ts"][i + 1]), "entry": round(tr["entry"], 2),
                         "stop": round(tr["stop"], 2), "target": round(tr["target"], 2),
                         "risk%": round(100 * risk / tr["entry"], 2),
                         "exit_ts": pd.Timestamp(b["ts"][tr["exit_i"]]),
                         "exit": round(tr["exit_px"], 2), "bars": int(tr["exit_i"] - i),
                         "why": tr["why"], "R_gross": tr["R"], "cost_R": round(cost_r, 3),
                         "R": round(tr["R"] - cost_r, 3)})
    t = pd.DataFrame(rows)
    if not t.empty:
        t = t.sort_values("signal_ts", ascending=False).reset_index(drop=True)
    return t


EXIT_WORDS = {
    "target": "🎯 Target hit", "stop": "🛑 Stop hit", "gap-stop": "⚡ Gapped past stop",
    "gap-target": "🚀 Gapped past target", "time": "⏱ Time limit",
    "session": "🔔 Squared off at close", "open": "⏳ Still open",
}


def trade_view(t: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Plain-language columns for the trade log: WIN/LOSS, how it ended, P&L in % and rupees.

    WIN means the trade made money AFTER cost (net R > 0), not "the target was hit". The two
    differ: a time-limit exit a few paise up is a gross gain and a net LOSS once the round trip
    is paid, and a trader reading the log needs the net answer."""
    if t.empty:
        return t.copy()
    v = t.copy()
    sgn = np.where(v["side"] == "LONG", 1.0, -1.0)
    v["pnl_rs"] = (sgn * (v["exit"] - v["entry"])).round(2)          # per share, gross
    v["pnl_pct"] = (100 * sgn * (v["exit"] - v["entry"]) / v["entry"]
                    - cost_bps / 100.0).round(2)                    # after cost
    v["result"] = np.where(v["R"] > 0, "✅ WIN", "❌ LOSS")
    v["exit_how"] = v["why"].map(EXIT_WORDS).fillna(v["why"])
    return v


def day_scoreboard(v: pd.DataFrame) -> pd.DataFrame:
    """One row per entry session: how many trades, how many won, the net result."""
    if v.empty:
        return pd.DataFrame()
    d = v.assign(day=pd.to_datetime(v["entry_ts"]).dt.date, win=v["R"] > 0)
    g = d.groupby("day")
    out = pd.DataFrame({"trades": g.size(), "wins": g["win"].sum().astype(int),
                        "net_R": g["R"].sum().round(2), "avg_pnl_pct": g["pnl_pct"].mean().round(2)})
    out["losses"] = out["trades"] - out["wins"]
    out["day_result"] = np.where(out["net_R"] > 0, "✅ Profit day", "❌ Loss day")
    return (out.reset_index()[["day", "day_result", "trades", "wins", "losses", "net_R",
                               "avg_pnl_pct"]].sort_values("day", ascending=False))


def clustered_t(closed: pd.DataFrame) -> float:
    """t-stat of the mean R with trades CLUSTERED BY ENTRY SESSION.

    Twenty high-volatility names are one market: on a gap-down morning half of them stop out
    together. Those are not independent draws, and the per-trade t counts them as if they were.
    Summing R per session and testing the session series is the honest version."""
    if closed.empty:
        return float("nan")
    g = closed.groupby(pd.to_datetime(closed["entry_ts"]).dt.normalize())["R"]
    s, k = g.sum(), g.count()
    if len(s) < 3:
        return float("nan")
    # mean R per trade = sum(R)/sum(n); its SE from the session totals (ratio estimator)
    mu = s.sum() / k.sum()
    resid = s - mu * k
    se = np.sqrt((resid ** 2).sum() * len(s) / (len(s) - 1)) / k.sum()
    return round(float(mu / se), 2) if se > 0 else float("nan")


def summarize(t: pd.DataFrame) -> dict:
    from . import paper
    closed = t[t["status"] == "CLOSED"] if not t.empty else t
    if closed.empty:
        return {}
    e = paper.expectancy(closed["R"])
    e["t_clustered"] = clustered_t(closed)
    e["sessions"] = int(pd.to_datetime(closed["entry_ts"]).dt.normalize().nunique())
    return e


def rr_sweep(bundle: dict, grid=(1.0, 1.5, 2.0, 3.0, 5.0), **kw) -> pd.DataFrame:
    rows = []
    for rr in grid:
        e = summarize(walk_bundle(bundle, rr=rr, **kw))
        if e:
            rows.append({"R:R": f"1:{rr:g}", "n": e["n"], "win%": e["win%"],
                         "avg_win_R": e["avg_win_R"], "avg_loss_R": e["avg_loss_R"],
                         "expectancy_R": e["expectancy_R"], "total_R": e["total_R"],
                         "t": e["t"], "t_clustered": e["t_clustered"]})
    return pd.DataFrame(rows)


def by_room(bundle: dict, **kw) -> pd.DataFrame:
    """E_R per Upper-TF bucket on the SAME signals -- whether the filter earns its place is a
    measurement, not an assumption."""
    kw = {k: v for k, v in kw.items() if k != "room"}
    rows = []
    for rb in ROOM_BUCKETS:
        e = summarize(walk_bundle(bundle, room=rb, **kw))
        if e:
            rows.append({"Upper-TF S/R": rb, "n": e["n"], "win%": e["win%"],
                         "expectancy_R": e["expectancy_R"], "t_clustered": e["t_clustered"]})
    return pd.DataFrame(rows)
