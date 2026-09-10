"""
paper.py — R-MULTIPLE paper simulator for the board's two lenses.

WHAT QUESTION THIS ANSWERS. Van Tharp's expectancy algebra says a 40% win rate at 1:3 beats
a 60% win rate at 1:1:

    E = (0.40 x 3R) - (0.60 x 1R) = +0.60R      vs      (0.60 x 1R) - (0.40 x 1R) = +0.20R

That arithmetic is correct and it is also the most over-quoted line in retail trading, because
it silently assumes the two numbers are INDEPENDENT. They are not. You do not get to pick a
3R target and keep the 40% hit rate -- widening the target is exactly what lowers it. The only
honest way to use the formula is to MEASURE the win rate that each target actually produces on
your own signal, then compute expectancy from the measured pair. That is what this module does:
it runs the same trade at several reward multiples and prints the win rate each one earned, so
the R:R choice is read off evidence instead of assumed.

WHAT IT SIMULATES. Two lenses, the same two the board renders:

  STRUCTURE  -- the higher-frame box x lower-frame structure read (mtf.synthesize). A trade is
                taken when the tag is DIRECTIONAL and takes a side (LONG/SHORT).
  SR         -- the confluence read: both frames marking a level at the same price, with price
                standing on it. Long at a support, short at a resistance -- the same rule the
                board's tabs use.

NO LOOKAHEAD, and the rules that guarantee it:
  * the signal at bar t reads bars <= t only;
  * the ENTRY fills at bar t+1's OPEN, never at t's close;
  * the weekly (confirmation) frame uses only weeks that have CLOSED before bar t, so the
    partial current week can never inform a decision taken inside it;
  * when a bar's range contains BOTH the stop and the target, the STOP is assumed to have hit
    first. Daily bars carry no intrabar path, and the optimistic assumption is how a simulator
    manufactures an edge that evaporates live.

ONE POSITION PER NAME AT A TIME. A structure tag persists for many bars, so an unguarded loop
re-enters the same trade daily -- measured on a smoke run, 1,467 "trades" out of 8 names in
under four years, with the same AXISBANK short opened on six consecutive sessions. Those are
not 1,467 observations; they are a few dozen positions counted many times, they overlap so
they are not independent, and the t-stat inherits both errors. Nor is it tradeable: six
overlapping shorts in one name are not six 1R risks. A signal arriving while the previous
position is still open is skipped.

COST IS CHARGED IN R, NOT IN BPS, and this matters more than it looks. A fixed 22bps round trip
is a DIFFERENT fraction of risk depending on how wide the stop is: on a 1-ATR stop worth 2% of
price it is 0.11R, on a tight 0.5% stop it is 0.44R. Quoting expectancy gross of cost, or
subtracting a flat "0.1R", would flatter tight stops exactly where they are least survivable.

WHY DAILY BARS. The frames are 1D (trigger) + 1W (confirm) -- the Positional horizon -- because
that is the only pair the 8-year EOD archive can reproduce leak-free and for free. The intraday
pairs (15m/1h, 1h/4h, 4h/1D) need broker history, which serves ~60 days and is rate-limited, so
a "backtest" on them would be a handful of months of one regime dressed up as evidence. The
page says so rather than quietly running a worse study.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data, indicators, mtf

# The pair the archive can reproduce. Kept as data so the page can name it and so adding an
# intraday pair later is a table entry plus a candle source, not a rewrite.
FRAMES = {"positional": ("1D", "1W")}

# Reward multiples the sweep reports. Spans the retail folklore range (1:1 through 1:5) so the
# win-rate-vs-target trade-off is visible rather than asserted.
RR_GRID = (1.0, 1.5, 2.0, 3.0, 5.0)


# ── frame construction ───────────────────────────────────────────────────────────────
def _weekly(d: pd.DataFrame) -> pd.DataFrame:
    """Daily OHLC -> CLOSED weekly bars, indexed by the date each week ENDED.

    The index is the week's last trading date, so a lookup for "the newest weekly bar strictly
    before today" is a plain searchsorted and cannot accidentally include the running week.
    """
    x = d[["trade_date", "open_price", "high_price", "low_price", "close_price"]].copy()
    # GROUP BY THE WEEK PERIOD, do not resample. resample() manufactures EMPTY buckets for
    # holiday weeks, and an empty weekly bar is not a quiet week -- it is a NaN row that
    # struct_full would then count toward its 20-bar lookback.
    x["wk"] = x["trade_date"].dt.to_period("W-FRI")
    g = x.groupby("wk", sort=True)
    return pd.DataFrame({
        "open": g["open_price"].first(), "high": g["high_price"].max(),
        "low": g["low_price"].min(), "close": g["close_price"].last(),
        "end": g["trade_date"].max(),
    }).reset_index(drop=True)


def _bars(d: pd.DataFrame) -> pd.DataFrame:
    """The daily frame in the column names indicators.* expects."""
    return pd.DataFrame({
        "ts": d["trade_date"].to_numpy(),
        "open": d["open_price"].to_numpy(float),
        "high": d["high_price"].to_numpy(float),
        "low": d["low_price"].to_numpy(float),
        "close": d["close_price"].to_numpy(float),
    })


# ── signal generation, one symbol ────────────────────────────────────────────────────
def _signals_for_symbol(d: pd.DataFrame, lens: str, i0: int,
                        conf_tol_bps: float, near_atr: float) -> list[tuple]:
    """[(bar_index, side, level_or_nan, atr)] for every bar from i0 onward that fires.

    The ATR travels WITH the signal because struct_full already computed it on exactly the
    bars the decision saw. Recomputing it outside the loop risks a different window, and a
    risk unit that disagrees with the read it sizes is how R stops meaning anything.

    Calls the REAL indicators.struct_full / mtf.synthesize / indicators.walls_kind rather than a
    vectorised copy of them. A faster re-implementation that drifts from the board would answer
    a question about a system nobody is running, which is the one outcome a simulator must not
    produce.
    """
    day = _bars(d)
    wk = _weekly(d)
    if len(wk) < 8 or len(day) < i0 + 2:
        return []
    wk_bars = pd.DataFrame({"ts": wk["end"].to_numpy(), "open": wk["open"].to_numpy(float),
                            "high": wk["high"].to_numpy(float), "low": wk["low"].to_numpy(float),
                            "close": wk["close"].to_numpy(float)})
    wk_end = pd.to_datetime(wk["end"]).to_numpy()
    lb = config.STRUCT_LOOKBACK
    sr_win = max(lb * 3, 60)                      # levels persist longer than a trend regime
    out: list[tuple] = []
    _wk_cache: dict[int, dict] = {}
    for i in range(i0, len(day) - 1):             # -1: the last bar has no next open to fill on
        t = day["ts"].iloc[i]
        # THE WEEKLY FRAME MUST BE CLOSED. searchsorted on the week-END dates with side="left"
        # gives the count of weeks that finished STRICTLY BEFORE today -- so the week today sits
        # inside is excluded by construction, not by a fragile date comparison.
        nw = int(np.searchsorted(wk_end, np.datetime64(pd.Timestamp(t)), side="left"))
        if nw < 8:
            continue
        hs = _wk_cache.get(nw)
        if hs is None:
            hs = indicators.struct_full(wk_bars.iloc[:nw])
            _wk_cache = {nw: hs}                  # weeks advance monotonically; keep one
        ls = indicators.struct_full(day.iloc[:i + 1])
        spot = float(day["close"].iloc[i])
        if lens == "STRUCTURE":
            s = mtf.synthesize(hs, ls, spot)
            side = mtf.side_of(s["tag"], s.get("dir", "NONE"))
            if side in ("LONG", "SHORT"):
                out.append((i, side, float("nan"), float(ls.get("atr") or 0)))
            continue
        # ── SR lens: both frames marking a level at the same price, price standing on it ──
        a = float(ls.get("atr") or 0)
        if a <= 0:
            continue
        seg_d = day.iloc[max(0, i + 1 - sr_win):i + 1]
        seg_w = wk_bars.iloc[max(0, nw - sr_win):nw]
        kl = indicators.walls_kind(seg_d["high"], seg_d["low"], config.SR_TOL_ATR * a)
        kh = indicators.walls_kind(seg_w["high"], seg_w["low"], config.SR_TOL_ATR * a)
        if not kl or not kh:
            continue
        tol = conf_tol_bps / 1e4 * spot
        near = near_atr * a
        best = None
        for look_dn in (True, False):             # both sides; the NEARER level wins
            for x, _t, _nl, _nh in kl:
                gap = (spot - x) if look_dn else (x - spot)
                if gap < 0 or gap > near:
                    continue
                same = [z for z in kh if ((z[0] <= spot) if look_dn else (z[0] >= spot))]
                if not same:
                    continue
                m = min(same, key=lambda z: abs(z[0] - x))
                if abs(m[0] - x) > tol:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, x, "LONG" if look_dn else "SHORT")
        if best is not None:
            out.append((i, best[2], best[1], a))
    return out


# ── trade management ─────────────────────────────────────────────────────────────────
def _walk(day: pd.DataFrame, i: int, side: str, stop_atr: float, rr: float,
          max_bars: int, atr: float) -> dict | None:
    """Fill at bar i+1's OPEN, then walk forward to the stop, the target or the time stop."""
    j = i + 1
    if j >= len(day) or atr <= 0:
        return None
    entry = float(day["open"].iloc[j])
    if entry <= 0:
        return None
    risk = stop_atr * atr
    if risk <= 0:
        return None
    long = side == "LONG"
    stop = entry - risk if long else entry + risk
    targ = entry + rr * risk if long else entry - rr * risk
    for k in range(j, min(j + max_bars, len(day))):
        hi, lo = float(day["high"].iloc[k]), float(day["low"].iloc[k])
        hit_stop = (lo <= stop) if long else (hi >= stop)
        hit_targ = (hi >= targ) if long else (lo <= targ)
        # BOTH IN ONE BAR -> ASSUME THE STOP. A daily bar carries no intrabar path, and
        # resolving the ambiguity in the trade's favour is precisely how a backtest invents an
        # edge that does not survive contact with a real tape.
        if hit_stop:
            return {"exit_i": k, "exit_px": stop, "R": -1.0, "why": "stop"}
        if hit_targ:
            return {"exit_i": k, "exit_px": targ, "R": rr, "why": "target"}
    k = min(j + max_bars, len(day)) - 1
    px = float(day["close"].iloc[k])
    r = (px - entry) / risk if long else (entry - px) / risk
    return {"exit_i": k, "exit_px": px, "R": round(r, 3), "why": "time"}


def build_signals(lens: str = "STRUCTURE", start: str | None = None, end: str | None = None,
                  symbols: list[str] | None = None, max_names: int | None = None,
                  conf_tol_bps: float | None = None, near_atr: float | None = None) -> dict:
    """Replay the tape ONCE and keep every signal, independent of any exit rule.

    Signals are a function of the bars and the lens alone -- the reward multiple, the stop
    width and the time stop all belong to trade MANAGEMENT. Regenerating them per setting (the
    first version of the sweep did) costs five full replays through struct_full / synthesize /
    walls_kind to answer a question that changes only the exit, and it also lets the five sweep
    rows drift apart: any difference between regenerations would show up as a target effect
    while actually being a signal difference. One generation makes the rows comparable by
    construction.
    """
    conf_tol_bps = float(config.SR_CONF_TOL_BPS if conf_tol_bps is None else conf_tol_bps)
    near_atr = float(config.SR_CONF_NEAR_ATR if near_atr is None else near_atr)
    df = data.load_eod(start=start, end=end)
    if symbols:
        df = df[df["symbol"].isin(symbols)]
    if max_names:
        # RANK BY TURNOVER, NOT ALPHABETICALLY. A truncated universe has to be a TRADEABLE
        # subset or the study quietly becomes "how does this do on names beginning with A".
        keep = (df.groupby("symbol")["turnover_lacs"].median()
                  .sort_values(ascending=False).head(max_names).index)
        df = df[df["symbol"].isin(keep)]
    i0 = config.STRUCT_LOOKBACK + 5
    sig = {}
    for sym, d in df.groupby("symbol", sort=False):
        d = d.sort_values("trade_date").reset_index(drop=True)
        if len(d) < i0 + 30:
            continue
        got = _signals_for_symbol(d, lens, i0, conf_tol_bps, near_atr)
        if got:
            sig[sym] = (_bars(d), d["trade_date"].to_numpy(), got)
    return {"sig": sig, "meta": {
        "lens": lens, "conf_tol_bps": conf_tol_bps, "near_atr": near_atr,
        "ltf": FRAMES["positional"][0], "htf": FRAMES["positional"][1],
        "n_names": int(df["symbol"].nunique()),
        "start": str(df["trade_date"].min().date()) if len(df) else None,
        "end": str(df["trade_date"].max().date()) if len(df) else None}}


def walk_signals(bundle: dict, rr: float = 3.0, stop_atr: float = 1.0, max_bars: int = 20,
                 cost_bps: float | None = None,
                 sides: tuple[str, ...] = ("LONG", "SHORT")) -> pd.DataFrame:
    """Turn one signal bundle into trades under one exit rule."""
    cost_bps = float(config.COST_BPS if cost_bps is None else cost_bps)
    rows = []
    for sym, (day, dates, got) in bundle["sig"].items():
        # ONE POSITION PER NAME AT A TIME. A structure tag PERSISTS for many bars, so a naive
        # loop re-enters the same trade every session: the smoke run showed AXISBANK firing
        # SHORT on six consecutive days, 1,467 "trades" from 8 names in under four years --
        # one entry per name per day. Those are not 1,467 observations, they are a few dozen
        # positions counted many times, and every statistic built on them is wrong in the
        # flattering direction: n is inflated, the trades overlap so they are not independent,
        # and the t-stat inherits both errors. It is also not a thing anyone can trade -- you
        # cannot hold six overlapping shorts in one name and call each a 1R risk.
        #
        # NOTE THIS LIVES HERE, NOT IN build_signals: which signals are BLOCKED depends on when
        # the previous position EXITED, which is a function of the exit rule. Filtering at
        # generation time would freeze one rule's blocking pattern into every other.
        _free_at = -1
        for i, side, lvl, a in got:
            if side not in sides or i <= _free_at:
                continue
            tr = _walk(day, i, side, stop_atr, rr, max_bars, a)
            if tr is None:
                continue
            _free_at = tr["exit_i"]
            entry = float(day["open"].iloc[i + 1])
            risk = stop_atr * a
            # COST IN R. A flat bps charge is a different fraction of risk for every stop
            # width, so converting it here is the only way the R numbers stay comparable
            # across names and across stop settings.
            cost_r = (cost_bps / 1e4 * entry) / risk if risk > 0 else np.nan
            rows.append({
                "symbol": sym, "signal_date": dates[i], "entry_date": dates[i + 1],
                "side": side, "entry": round(entry, 2), "risk_pts": round(risk, 2),
                "risk%": round(100 * risk / entry, 2), "level": lvl,
                "exit_date": dates[tr["exit_i"]], "exit": round(tr["exit_px"], 2),
                "bars": int(tr["exit_i"] - i), "why": tr["why"], "R_gross": tr["R"],
                "R": round(tr["R"] - cost_r, 3), "cost_R": round(cost_r, 3),
            })
    return pd.DataFrame(rows)


def simulate(lens: str = "STRUCTURE", rr: float = 3.0, stop_atr: float = 1.0,
             max_bars: int = 20, start: str | None = None, end: str | None = None,
             symbols: list[str] | None = None, max_names: int | None = None,
             cost_bps: float | None = None, sides: tuple[str, ...] = ("LONG", "SHORT"),
             conf_tol_bps: float | None = None, near_atr: float | None = None,
             rr_grid=None, bundle: dict | None = None) -> dict:
    """Run the simulator. Returns {"trades", "summary", "by_year", "rr_curve", "meta"}."""
    cost_bps = float(config.COST_BPS if cost_bps is None else cost_bps)
    if bundle is None:
        bundle = build_signals(lens=lens, start=start, end=end, symbols=symbols,
                               max_names=max_names, conf_tol_bps=conf_tol_bps,
                               near_atr=near_atr)
    trades = walk_signals(bundle, rr=rr, stop_atr=stop_atr, max_bars=max_bars,
                          cost_bps=cost_bps, sides=sides)
    meta = {**bundle["meta"], "rr": rr, "stop_atr": stop_atr, "max_bars": max_bars,
            "cost_bps": cost_bps}
    if trades.empty:
        return {"trades": trades, "summary": {}, "by_year": pd.DataFrame(),
                "rr_curve": pd.DataFrame(), "meta": meta}
    out = {"trades": trades, "summary": expectancy(trades["R"]), "meta": meta,
           "by_year": _by_year(trades)}
    if rr_grid is not False:
        out["rr_curve"] = rr_sweep(bundle=bundle, stop_atr=stop_atr, max_bars=max_bars,
                                   cost_bps=cost_bps, sides=sides,
                                   grid=tuple(rr_grid) if rr_grid else RR_GRID)
    return out


# ── scoring ──────────────────────────────────────────────────────────────────────────
def expectancy(R: pd.Series) -> dict:
    """Van Tharp's numbers, computed from the trades rather than assumed.

    `expectancy` is simply mean(R) -- the win-rate/avg-win decomposition below is the same
    quantity rearranged, and it is reported alongside so the two can be checked against each
    other. If they ever disagree, the decomposition is wrong, not the mean.
    """
    R = pd.to_numeric(R, errors="coerce").dropna()
    n = len(R)
    if not n:
        return {}
    w, l = R[R > 0], R[R <= 0]
    pw = len(w) / n
    aw = float(w.mean()) if len(w) else 0.0
    al = float(-l.mean()) if len(l) else 0.0
    return {
        "n": n, "win%": round(100 * pw, 1),
        "avg_win_R": round(aw, 2), "avg_loss_R": round(al, 2),
        "expectancy_R": round(float(R.mean()), 3),
        "expectancy_check": round(pw * aw - (1 - pw) * al, 3),
        "total_R": round(float(R.sum()), 1),
        "profit_factor": (round(float(w.sum() / -l.sum()), 2)
                          if len(l) and l.sum() < 0 else float("inf")),
        # STANDARD ERROR, BECAUSE A MEAN WITHOUT ONE IS A GUESS. R-multiples are heavy-tailed;
        # with a 3R target the distribution is two spikes, so the t here is the only thing
        # separating "a real edge" from "40 trades went well".
        "se_R": round(float(R.std(ddof=1) / np.sqrt(n)), 3) if n > 1 else np.nan,
        "t": (round(float(R.mean() / (R.std(ddof=1) / np.sqrt(n))), 2)
              if n > 1 and R.std(ddof=1) > 0 else np.nan),
        "max_dd_R": round(float((R.cumsum().cummax() - R.cumsum()).max()), 1),
    }


def _by_year(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t["year"] = pd.to_datetime(t["entry_date"]).dt.year
    rows = []
    for y, s in t.groupby("year"):
        e = expectancy(s["R"])
        rows.append({"year": int(y), **{k: e.get(k) for k in
                                        ("n", "win%", "avg_win_R", "avg_loss_R",
                                         "expectancy_R", "total_R")}})
    return pd.DataFrame(rows)


def rr_sweep(bundle: dict | None = None, grid=RR_GRID, stop_atr: float = 1.0,
             max_bars: int = 20, cost_bps: float | None = None,
             sides: tuple[str, ...] = ("LONG", "SHORT"), **kw) -> pd.DataFrame:
    """THE POINT OF THE WHOLE PAGE: win rate is not independent of the target.

    Walks ONE signal bundle at each reward multiple and reports the win rate each one actually
    earned. Expectancy is then computed from the MEASURED pair, which is the only version of
    Tharp's formula that is not circular. Because every row comes from the same generation,
    any difference between rows is the TARGET and nothing else.
    """
    if bundle is None:
        bundle = build_signals(**kw)
    rows = []
    for rr in grid:
        t = walk_signals(bundle, rr=rr, stop_atr=stop_atr, max_bars=max_bars,
                         cost_bps=cost_bps, sides=sides)
        e = expectancy(t["R"]) if len(t) else {}
        if not e:
            continue
        rows.append({"R:R": f"1:{rr:g}", "n": e["n"], "win%": e["win%"],
                     "avg_win_R": e["avg_win_R"], "avg_loss_R": e["avg_loss_R"],
                     "expectancy_R": e["expectancy_R"], "total_R": e["total_R"],
                     "t": e["t"]})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════════════
# SCALP LANE — 1m trigger / 5m confirm
# ═════════════════════════════════════════════════════════════════════════════════════
SCALP_FRAMES = ("1m", "5m")
# Hold horizons in MINUTES (= 1-minute bars). Capped at one hour because past that it stops
# being a scalp and starts being an intraday swing with a scalp's cost structure.
SCALP_HOLDS = (5, 15, 30, 60)


def _sess_end(ts) -> "np.ndarray":
    """For each bar, the index of the LAST bar of its own session (the square-off point)."""
    d = pd.to_datetime(pd.Series(ts)).dt.date.to_numpy()
    out = np.zeros(len(d), dtype=int)
    idx = np.arange(len(d))
    for day in pd.unique(d):
        m = d == day
        out[m] = idx[m].max()
    return out


def scalp_signals(ltf: pd.DataFrame, htf: pd.DataFrame, lens: str, i0: int,
                  conf_tol_bps: float | None = None,
                  near_atr: float | None = None) -> list[tuple]:
    """1m x 5m signals. The confirm frame is counted CLOSED-ONLY.

    scalp.py records what the alternative costs: its first offline replay printed WITH-TREND
    CONTINUATION at +16.6bps with t = 33 and 100% of sessions positive, because each 1-minute
    bar was indexed to the 5-minute bar CONTAINING it instead of the last one to have CLOSED --
    handing the structure window up to four minutes of the future against a five-minute
    forward return. searchsorted(htf_starts[1:], t, "right") counts confirm bars whose
    SUCCESSOR has already begun, so only closed bars can ever be seen.
    """
    conf_tol_bps = float(config.SR_CONF_TOL_BPS if conf_tol_bps is None else conf_tol_bps)
    near_atr = float(config.SR_CONF_NEAR_ATR if near_atr is None else near_atr)
    lt, ht = ltf["ts"].to_numpy(), htf["ts"].to_numpy()
    closed = np.searchsorted(ht[1:], lt, side="right")
    sr_win = max(config.STRUCT_LOOKBACK * 3, 60)
    out, cache = [], (None, None)
    for i in range(i0, len(ltf) - 1):
        nh = int(closed[i])
        if nh < 8:
            continue
        if cache[0] != nh:
            cache = (nh, indicators.struct_full(htf.iloc[:nh]))
        hs = cache[1]
        ls = indicators.struct_full(ltf.iloc[:i + 1])
        spot = float(ltf["close"].iloc[i])
        a = float(ls.get("atr") or 0)
        if a <= 0:
            continue
        if lens == "STRUCTURE":
            sy = mtf.synthesize(hs, ls, spot)
            sd = mtf.side_of(sy["tag"], sy.get("dir", "NONE"))
            if sd in ("LONG", "SHORT"):
                out.append((i, sd, sy["tag"], a))
            continue
        segl = ltf.iloc[max(0, i + 1 - sr_win):i + 1]
        segh = htf.iloc[max(0, nh - sr_win):nh]
        kl = indicators.walls_kind(segl["high"], segl["low"], config.SR_TOL_ATR * a)
        kh = indicators.walls_kind(segh["high"], segh["low"], config.SR_TOL_ATR * a)
        if not kl or not kh:
            continue
        tol, near, best = conf_tol_bps / 1e4 * spot, near_atr * a, None
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
        if best is not None:
            out.append((i, best[2], "SR", a))
    return out


def _walk_scalp(bars, sess_end, i, side, stop_atr, rr, max_bars, atr, maker=False):
    """Fill, then walk to the stop / target / time stop / SESSION END, whichever comes first.

    MAKER FILLS ARE NOT A DISCOUNT. A resting limit is filled only when price comes TO it,
    which is disproportionately when the market is about to continue against you -- adverse
    selection, and the reason "just use limit orders" mostly fails in practice. So the maker
    path rests at the signal bar's close and is filled ONLY if the next bar actually trades
    through that price; an unfilled signal is not a trade at all. The gap between the taker
    and maker win rates is then a MEASUREMENT of adverse selection rather than an assumption.
    """
    j = i + 1
    if j >= len(bars) or atr <= 0:
        return None
    if maker:
        limit = float(bars["close"].iloc[i])
        lo_j, hi_j = float(bars["low"].iloc[j]), float(bars["high"].iloc[j])
        filled = (lo_j <= limit) if side == "LONG" else (hi_j >= limit)
        if not filled:
            return {"unfilled": True}
        entry = limit
    else:
        entry = float(bars["open"].iloc[j])
    risk = stop_atr * atr
    if entry <= 0 or risk <= 0:
        return None
    long = side == "LONG"
    stop = entry - risk if long else entry + risk
    targ = entry + rr * risk if long else entry - rr * risk
    last = min(j + max_bars, int(sess_end[j]), len(bars) - 1)
    for k in range(j, last + 1):
        hi, lo = float(bars["high"].iloc[k]), float(bars["low"].iloc[k])
        # Ambiguous bar -> the STOP. A 1-minute candle carries no intrabar path either, and
        # resolving it in the trade's favour is how a scalp backtest invents an edge.
        # EVERY branch returns `entry`. It used to be returned only by the time-stop path,
        # so a MAKER trade that hit its stop or target fell back to the taker fill price
        # (the next bar's OPEN) when the caller computed cost_R -- pricing the maker's cost
        # off a fill it never got. Small in rupees, wrong in kind: the lane exists to compare
        # those two fills.
        if (lo <= stop) if long else (hi >= stop):
            return {"exit_i": k, "exit_px": stop, "R": -1.0, "why": "stop", "entry": entry}
        if (hi >= targ) if long else (lo <= targ):
            return {"exit_i": k, "exit_px": targ, "R": rr, "why": "target", "entry": entry}
    px = float(bars["close"].iloc[last])
    r = (px - entry) / risk if long else (entry - px) / risk
    why = "session" if last == int(sess_end[j]) and last < j + max_bars else "time"
    return {"exit_i": last, "exit_px": px, "R": round(r, 3), "why": why, "entry": entry}


def breakeven_win_rate(cost_r: float, rr: float) -> float:
    """p such that p*(rr - cost) - (1-p)*(1 + cost) = 0, i.e. p = (1+cost_R)/(1+R:R).

    Printed beside the measured win rate because it is the whole argument of this lane: the
    requirement is a pure function of the cost fraction, and cost_R = cost / stop, so a
    TIGHTER scalp stop RAISES the bar. Below roughly a 10-tick stop the requirement passes
    100% and no hit rate whatsoever can pay -- the round trip exceeds the risk it is taken on.
    """
    return (1.0 + float(cost_r)) / (1.0 + float(rr))


def walk_scalp_bundle(bundle: dict, rr: float = 2.0, stop_atr: float = 1.0,
                      max_bars: int = 15, maker: bool = False,
                      sides: tuple[str, ...] = ("LONG", "SHORT")) -> pd.DataFrame:
    rows = []
    for sym, (bars, se, cost_bps, sig) in bundle["sig"].items():
        free, seen, fills = -1, 0, 0
        for i, side, tag, a in sig:
            if side not in sides or i <= free:
                continue
            seen += 1
            tr = _walk_scalp(bars, se, i, side, stop_atr, rr, max_bars, a, maker=maker)
            if tr is None:
                continue
            if tr.get("unfilled"):
                # A resting order that never traded is NOT a loss and NOT a win -- it is not a
                # trade. Counting it either way would corrupt the win rate; it is reported
                # separately as the fill rate.
                continue
            fills += 1
            free = tr["exit_i"]
            entry = tr.get("entry", float(bars["open"].iloc[i + 1]))
            # A MAKER DOES NOT PAY THE SPREAD LINE. Everything else on the sheet still applies
            # -- STT, brokerage, exchange, GST, stamp are indifferent to how you got filled.
            c = cost_bps["all"] - (cost_bps["spread"] if maker else 0.0)
            cost_r = (c / 1e4 * entry) / (stop_atr * a)
            rows.append({"symbol": sym, "side": side, "tag": tag, "why": tr["why"],
                         "R_gross": tr["R"], "R": tr["R"] - cost_r, "cost_R": cost_r,
                         "risk_bps": 1e4 * (stop_atr * a) / entry,
                         "mins": int(tr["exit_i"] - i),
                         "ts": pd.Timestamp(bars["ts"].iloc[i + 1])})
        if sig:
            bundle.setdefault("_fill", {})[sym] = (fills, seen)
    return pd.DataFrame(rows)


def build_scalp_bundle(names: list[str] | None = None, n_names: int = 20, days: int = 30,
                       position: float | None = None, spread_ticks: float | None = None,
                       lens: str = "STRUCTURE") -> dict:
    """Fetch 1-minute bars, build the 5-minute confirm frame, and generate every signal once.

    The universe defaults to scalp.scalp_universe, which ranks by ATR% rather than turnover --
    Spearman +0.902 against realised mean 5-minute move, where turnover ranks -0.10. That is
    not a detail: 1R on a 1-minute bar is the ONLY term in cost_R that the trader controls, so
    the study must run on the names where it is largest, otherwise a negative result could be
    dismissed as having tested the wrong list. These are the best case, not the average.
    """
    from . import scalp as _scalp
    position = float(_scalp.DEFAULT_POSITION if position is None else position)
    names = list(names) if names else list(_scalp.scalp_universe(n=n_names))
    sig, meta_names = {}, []
    for sym in names:
        f = _scalp.fetch_1m(sym, days=days)
        if f is None or len(f) < 800:
            continue
        f = f.sort_values("ts").reset_index(drop=True)
        h5 = _scalp._resample(f, 5)
        if h5 is None or len(h5) < 60:
            continue
        px = float(f["close"].iloc[-1])
        parts = (_scalp.cost_parts(position, px) if spread_ticks is None
                 else _scalp.cost_parts(position, px, spread_ticks))
        cost = {**parts, "all": float(sum(parts.values()))}
        g = scalp_signals(f, h5, lens, config.STRUCT_LOOKBACK + 5)
        if g:
            sig[sym] = (f, _sess_end(f["ts"]), cost, g)
            meta_names.append(sym)
    return {"sig": sig, "meta": {
        "lens": lens, "ltf": SCALP_FRAMES[0], "htf": SCALP_FRAMES[1], "days": days,
        "position": position, "n_names": len(sig), "names": meta_names,
        "n_signals": int(sum(len(v[3]) for v in sig.values())),
        "med_cost_bps": (round(float(np.median([v[2]["all"] for v in sig.values()])), 2)
                         if sig else None)}}


def simulate_scalp(bundle: dict | None = None, rr: float = 2.0, stop_atr: float = 1.0,
                   max_bars: int = 15, maker: bool = False,
                   sides: tuple[str, ...] = ("LONG", "SHORT"),
                   rr_grid=None, **kw) -> dict:
    """Scalp equivalent of simulate(). Adds the breakeven-vs-measured comparison."""
    if bundle is None:
        bundle = build_scalp_bundle(**kw)
    t = walk_scalp_bundle(bundle, rr=rr, stop_atr=stop_atr, max_bars=max_bars,
                          maker=maker, sides=sides)
    meta = {**bundle["meta"], "rr": rr, "stop_atr": stop_atr, "max_bars": max_bars,
            "maker": maker}
    if t.empty:
        return {"trades": t, "summary": {}, "rr_curve": pd.DataFrame(), "meta": meta}
    e = expectancy(t["R"])
    med_cost_r = float(t["cost_R"].median())
    rows = []
    for _rr in (tuple(rr_grid) if rr_grid else RR_GRID):
        tt = walk_scalp_bundle(bundle, rr=_rr, stop_atr=stop_atr, max_bars=max_bars,
                               maker=maker, sides=sides)
        if tt.empty:
            continue
        ee = expectancy(tt["R"])
        cr = float(tt["cost_R"].median())
        need = 100 * breakeven_win_rate(cr, _rr)
        w = tt[tt["R"] > 0]
        rows.append({"R:R": f"1:{_rr:g}", "n": ee["n"], "win%": ee["win%"],
                     "need%": (">100 (impossible)" if need > 100 else round(need, 1)),
                     "gap": (np.nan if need > 100 else round(ee["win%"] - need, 1)),
                     "tgtHits%": (round(float((w["why"] == "target").mean() * 100), 1)
                                  if len(w) else np.nan),
                     "E_R": ee["expectancy_R"], "t": ee["t"],
                     "1R_bps": round(float(tt["risk_bps"].median()), 1),
                     "cost_R": round(cr, 3)})
    fills = bundle.get("_fill", {})
    got = sum(a for a, _b in fills.values())
    tried = sum(b for _a, b in fills.values())
    meta["fill_rate"] = round(100 * got / tried, 1) if tried else None
    meta["med_cost_R"] = round(med_cost_r, 3)
    return {"trades": t, "summary": e, "rr_curve": pd.DataFrame(rows), "meta": meta}
