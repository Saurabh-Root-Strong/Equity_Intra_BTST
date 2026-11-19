"""Can option / futures positioning tell EARLY that an intraday trend is about to turn?

Data: Tradebot's live archive (data/intraday/live), Jun-Nov 2026, 4 indices.
  chain_snapshots  per strike CE/PE oi, volume, iv, ltp every ~30 s (nearest expiry, 51 strikes)
  candles 1min     index spot
  futures_oi       NSE oi-spurts, index futures OI (all months), ~1 min (live estimate, noisy)

Design (fixed before looking at any result):
  Grid: every 5 minutes 09:45-14:30 IST (features need 30 min of history, targets need 60 min).
  Trend: causal zigzag on 1-min spot with threshold THETA (per index = median |60-min move|
         over the sample). At time t the CURRENT LEG direction d (+1 up / -1 down) is known
         from pivots already confirmed. Only grid points inside a leg at least THETA long count.
  Target RACE: from t, does price move THETA AGAINST the leg before it moves THETA WITH it,
         within 90 min? 1 = trend turned first, 0 = trend continued first, NaN = neither.
  Target FWD30: 30-min return x d, in units of THETA.
  Features (all from data <= t, signed so that + = "against the leg" = should mean TURN):
    wall_defense   writers ADDING at the max-OI strike just ahead of price (CE wall above in an
                   up-leg, PE wall below in a down-leg), 15-min OI change / that strike's OI
    band_flow      ATM+-5 strikes, 15-min (dCE_oi - dPE_oi) / band OI, x d (call writing in an
                   up-leg = against)
    cover_fuel     share of the leg's band OI change that is covering on the side that was
                   squeezed (up-leg: CE OI falling; down-leg: PE OI falling)
    vol_climax     band option volume in last 15 min / the day's average 15 min so far
    iv_against     ATM IV change over 15 min x d (IV rising into a rally = against)
    pcr_div        band PCR change over 30 min x -d (PCR falling in an up-leg = against)
    fut_cover      futures OI change over 15 min, negative while the leg runs = covering
                   (x -1 so that + = covering = against)
    wall_shift     max-OI CE/PE strike moving AGAINST the leg over 30 min (walls stepping back)
  Price-only controls: leg_len (leg size in THETA), leg_age (minutes), extension (distance from
  the 30-min mean in THETA), tod (minutes since open).
  Level events: first touch per leg of the max-OI wall ahead (within 0.1 THETA); outcome
  rejected (THETA back first) vs broke (THETA through first); split by writers adding vs
  cutting at that strike in the prior 15 min.
Inference: per-day rank IC averaged over days (t = mean / se across days), per index, and for
  two halves (Jun-Aug, Sep-Nov). Benjamini-Hochberg over every feature x index x target.
  Combined: logistic regression, leave-one-month-out, AUC of positioning+price vs price-only.
"""
from __future__ import annotations

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
LIVE = r"d:\Python Projects\Tradebot\data\intraday\live"
CACHE = os.environ.get("TT_CACHE", os.path.join(os.environ.get("TEMP", "."), "tt_panel.parquet"))
SYMS = {"NSE:NIFTY50-INDEX": ("NIFTY", 50), "NSE:NIFTYBANK-INDEX": ("BANKNIFTY", 100),
        "NSE:FINNIFTY-INDEX": ("FINNIFTY", 50), "NSE:MIDCPNIFTY-INDEX": ("MIDCPNIFTY", 25)}
IST = "Asia/Kolkata"


def _ist(s):
    s = pd.to_datetime(s)
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    return s.dt.tz_convert(IST).dt.tz_localize(None)


def load_day(day: str) -> dict:
    out = {}
    try:
        ch = pd.read_parquet(f"{LIVE}/{day}_chain_snapshots.parquet",
                             columns=["ts", "symbol", "strike", "side", "oi", "volume", "iv", "ltp"])
        cd = pd.read_parquet(f"{LIVE}/{day}_candles.parquet")
    except Exception:
        return out
    if ch.empty or cd.empty:
        return out
    try:
        fo = pd.read_parquet(f"{LIVE}/{day}_futures_oi.parquet")
        fo["ts"] = pd.to_datetime(fo["ts"]).dt.tz_convert(IST).dt.tz_localize(None)
    except Exception:
        fo = pd.DataFrame(columns=["ts", "symbol", "oi"])
    ch["ts"] = _ist(ch["ts"])
    cd = cd[cd["resolution"] == "1min"].copy()
    cd["ts"] = _ist(cd["ts"])
    d0 = pd.Timestamp(day)
    lo, hi = d0 + pd.Timedelta("09:15:00"), d0 + pd.Timedelta("15:30:00")
    grid = pd.date_range(d0 + pd.Timedelta("09:16:00"), d0 + pd.Timedelta("15:29:00"), freq="1min")
    for sym, (code, _step) in SYMS.items():
        px = cd[(cd["symbol"] == sym) & (cd["ts"] >= lo) & (cd["ts"] < hi)]
        px = px.drop_duplicates("ts", keep="last").set_index("ts")["close"].sort_index()
        c = ch[(ch["symbol"] == sym) & (ch["ts"] >= lo) & (ch["ts"] < hi)]
        if len(px) < 300 or c["ts"].nunique() < 200:
            continue
        # candle stamped at its START: the close of the 09:15 bar is known at 09:16
        px.index = px.index + pd.Timedelta("1min")
        spot = px.reindex(grid, method="ffill")
        c = c.copy()
        c["m"] = c["ts"].dt.ceil("1min")
        c = c.sort_values("ts").drop_duplicates(["m", "strike", "side"], keep="last")
        wide = {}
        for f in ("oi", "volume", "iv"):
            p = c.pivot_table(index="m", columns=["side", "strike"], values=f, aggfunc="last")
            wide[f] = p.reindex(grid, method="ffill")
        fut = fo[fo["symbol"] == code].drop_duplicates("ts", keep="last").set_index("ts")["oi"]
        fut = fut[(fut.index >= lo) & (fut.index < hi)].sort_index()
        fut.index = fut.index.ceil("1min")
        fut = fut[~fut.index.duplicated(keep="last")].reindex(grid, method="ffill")
        out[code] = dict(spot=spot, oi=wide["oi"], vol=wide["volume"], iv=wide["iv"], fut=fut)
    return out


def zigzag_state(px: np.ndarray, theta: float):
    """Causal: at each minute the direction of the current leg, its start price / index,
    using only prices up to that minute. A leg flips when price retraces theta from the
    running extreme."""
    n = len(px)
    d = np.zeros(n)
    st_px = np.full(n, np.nan)
    st_ix = np.zeros(n, dtype=int)
    if n == 0:
        return d, st_px, st_ix
    direc, ext, ext_i, start, start_i = 0, px[0], 0, px[0], 0
    lo_, lo_i, hi_, hi_i = px[0], 0, px[0], 0
    for i, p in enumerate(px):
        if np.isnan(p):
            d[i], st_px[i], st_ix[i] = direc, start, start_i
            continue
        if direc == 0:
            if p > hi_:
                hi_, hi_i = p, i
            if p < lo_:
                lo_, lo_i = p, i
            if p - lo_ >= theta:
                direc, start, start_i, ext, ext_i = 1, lo_, lo_i, p, i
            elif hi_ - p >= theta:
                direc, start, start_i, ext, ext_i = -1, hi_, hi_i, p, i
        elif direc == 1:
            if p > ext:
                ext, ext_i = p, i
            elif ext - p >= theta:
                direc, start, start_i, ext, ext_i = -1, ext, ext_i, p, i
        else:
            if p < ext:
                ext, ext_i = p, i
            elif p - ext >= theta:
                direc, start, start_i, ext, ext_i = 1, ext, ext_i, p, i
        d[i], st_px[i], st_ix[i] = direc, start, start_i
    return d, st_px, st_ix


def race(px: np.ndarray, i: int, d: float, theta: float, horizon: int = 90) -> float:
    p0 = px[i]
    seg = px[i + 1:i + 1 + horizon]
    for p in seg:
        if np.isnan(p):
            continue
        m = (p - p0) * d
        if m <= -theta:
            return 1.0
        if m >= theta:
            return 0.0
    return np.nan


def features_for(code: str, day: str, D: dict, theta: float) -> list:
    spot, oi, vol, iv, fut = D["spot"], D["oi"], D["vol"], D["iv"], D["fut"]
    px = spot.values.astype(float)
    dirn, st_px, st_ix = zigzag_state(px, theta)
    t_index = spot.index
    strikes = np.array(sorted({k for _, k in oi.columns}))
    if len(strikes) < 11 or "CE" not in oi.columns.get_level_values(0) \
            or "PE" not in oi.columns.get_level_values(0):
        return []
    ce_oi, pe_oi = oi["CE"].reindex(columns=strikes), oi["PE"].reindex(columns=strikes)
    ce_v, pe_v = vol["CE"].reindex(columns=strikes), vol["PE"].reindex(columns=strikes)
    ce_iv, pe_iv = iv["CE"].reindex(columns=strikes), iv["PE"].reindex(columns=strikes)
    rows = []
    open_i = 0
    for i, t in enumerate(t_index):
        hm = t.hour * 60 + t.minute
        if hm < 9 * 60 + 45 or hm > 14 * 60 + 30 or t.minute % 5:
            continue
        if np.isnan(px[i]) or dirn[i] == 0:
            continue
        d = dirn[i]
        leg = (px[i] - st_px[i]) * d
        if leg < theta:
            continue
        s = px[i]
        atm = int(np.argmin(np.abs(strikes - s)))
        band = slice(max(atm - 5, 0), atm + 6)
        i15, i30 = i - 15, i - 30
        if i30 < 0:
            continue
        cb, pb = ce_oi.iloc[i, band], pe_oi.iloc[i, band]
        cb15, pb15 = ce_oi.iloc[i15, band], pe_oi.iloc[i15, band]
        boi = float(cb.sum() + pb.sum())
        if not boi or np.isnan(boi):
            continue
        dce, dpe = float((cb - cb15).sum()), float((pb - pb15).sum())
        # wall ahead: max OI strike within 1.5 theta ahead of price, on the side that defends
        if d > 0:
            ahead = (strikes > s) & (strikes <= s + 1.5 * theta)
            wall_ser = ce_oi
        else:
            ahead = (strikes < s) & (strikes >= s - 1.5 * theta)
            wall_ser = pe_oi
        wall_def, wall_k = np.nan, np.nan
        if ahead.any():
            cols = np.where(ahead)[0]
            j = cols[int(np.nanargmax(wall_ser.iloc[i, cols].values))] if \
                np.isfinite(wall_ser.iloc[i, cols].values).any() else None
            if j is not None:
                now_, then_ = wall_ser.iloc[i, j], wall_ser.iloc[i15, j]
                if now_ and np.isfinite(now_) and np.isfinite(then_):
                    wall_def = (now_ - then_) / now_
                    wall_k = strikes[j]
        # leg fuel since leg start (band at the leg start's ATM is close enough: same band)
        j0 = max(int(st_ix[i]), 0)
        dce_leg = float((ce_oi.iloc[i, band] - ce_oi.iloc[j0, band]).sum())
        dpe_leg = float((pe_oi.iloc[i, band] - pe_oi.iloc[j0, band]).sum())
        squeezed = dce_leg if d > 0 else dpe_leg
        tot = abs(dce_leg) + abs(dpe_leg)
        cover = (max(-squeezed, 0.0) / tot) if tot else np.nan
        # volume climax: band volume in the last 15 min vs the day's 15-min average so far
        bv = (ce_v.iloc[:, band].sum(axis=1) + pe_v.iloc[:, band].sum(axis=1)).values
        v15 = bv[i] - bv[i15]
        avg15 = (bv[i] - bv[min(open_i + 5, i15)]) / max((i - min(open_i + 5, i15)) / 15, 1)
        climax = v15 / avg15 if avg15 and avg15 > 0 and v15 >= 0 else np.nan
        iv_now = np.nanmean([ce_iv.iloc[i, atm], pe_iv.iloc[i, atm]])
        iv_then = np.nanmean([ce_iv.iloc[i15, atm], pe_iv.iloc[i15, atm]])
        pcr_now = pb.sum() / cb.sum() if cb.sum() else np.nan
        cb30, pb30 = ce_oi.iloc[i30, band], pe_oi.iloc[i30, band]
        pcr_then = pb30.sum() / cb30.sum() if cb30.sum() else np.nan
        f_now, f_then = fut.iloc[i], fut.iloc[i15]
        fut_c = -(f_now - f_then) / f_then if f_then and np.isfinite(f_then) and np.isfinite(f_now) else np.nan
        cw = lambda k: strikes[int(np.nanargmax(ce_oi.iloc[k].values))] if np.isfinite(ce_oi.iloc[k].values).any() else np.nan
        pw = lambda k: strikes[int(np.nanargmax(pe_oi.iloc[k].values))] if np.isfinite(pe_oi.iloc[k].values).any() else np.nan
        shift = ((cw(i) - cw(i30)) + (pw(i) - pw(i30))) / 2.0 * -d / theta
        mean30 = np.nanmean(px[i30:i + 1])
        rows.append(dict(
            code=code, day=day, t=t, d=d, theta=theta,
            wall_defense=wall_def, wall_k=wall_k,
            band_flow=(dce - dpe) / boi * d,
            cover_fuel=cover, vol_climax=climax,
            iv_against=(iv_now - iv_then) * d if np.isfinite(iv_now) and np.isfinite(iv_then) else np.nan,
            pcr_div=(pcr_now - pcr_then) * -d if np.isfinite(pcr_now) and np.isfinite(pcr_then) else np.nan,
            fut_cover=fut_c, wall_shift=shift,
            leg_len=leg / theta, leg_age=i - j0, extension=(s - mean30) * d / theta, tod=hm - 555,
            race=race(px, i, d, theta),
            fwd30=((px[i + 30] - s) * d / theta) if i + 30 < len(px) and np.isfinite(px[i + 30]) else np.nan,
        ))
    return rows


def level_events(code: str, day: str, D: dict, theta: float) -> list:
    """First touch per leg of the biggest OI wall ahead; rejected vs broke; writers adding or
    cutting at that strike in the 15 min before the touch."""
    spot, oi = D["spot"], D["oi"]
    px = spot.values.astype(float)
    dirn, st_px, st_ix = zigzag_state(px, theta)
    strikes = np.array(sorted({k for _, k in oi.columns}))
    ce, pe = oi["CE"].reindex(columns=strikes), oi["PE"].reindex(columns=strikes)
    out, seen = [], set()
    for i in range(30, len(px) - 1):
        t = spot.index[i]
        hm = t.hour * 60 + t.minute
        if hm < 9 * 60 + 30 or hm > 14 * 60 + 45 or dirn[i] == 0 or np.isnan(px[i]):
            continue
        d = dirn[i]
        leg_id = (int(st_ix[i]), d)
        s = px[i]
        if d > 0:
            m = (strikes > s - 0.1 * theta) & (strikes <= s + 2 * theta)
            ser = ce
        else:
            m = (strikes < s + 0.1 * theta) & (strikes >= s - 2 * theta)
            ser = pe
        if not m.any():
            continue
        cols = np.where(m)[0]
        vals = ser.iloc[i - 1, cols].values
        if not np.isfinite(vals).any():
            continue
        j = cols[int(np.nanargmax(vals))]
        k = strikes[j]
        # the wall must be big: top-3 OI on its side of the whole chain
        side_rank = (ser.iloc[i - 1].values > vals[np.nanargmax(vals)]).sum()
        if side_rank > 2:
            continue
        if abs(s - k) > 0.1 * theta or (leg_id, k) in seen:
            continue
        seen.add((leg_id, k))
        now_, then_ = ser.iloc[i, j], ser.iloc[i - 15, j]
        if not (np.isfinite(now_) and np.isfinite(then_) and now_):
            continue
        res = race(px, i, d, theta / 2, 60)          # half-theta: a level test is a small move
        out.append(dict(code=code, day=day, t=t, d=d, k=k, writers=(now_ - then_) / now_,
                        rejected=res, leg_len=(s - st_px[i]) * d / theta))
    return out


def build():
    days = sorted({os.path.basename(f)[:10] for f in glob.glob(f"{LIVE}/*_chain_snapshots.parquet")
                   if os.path.getsize(f) > 500_000})
    raw = {}
    for dday in days:
        D = load_day(dday)
        for code, v in D.items():
            raw.setdefault(code, {})[dday] = v
        print(f"loaded {dday}: {sorted(D)}", flush=True)
    thetas = {}
    for code, dd in raw.items():
        mv = []
        for v in dd.values():
            s = v["spot"].dropna()
            mv.extend(np.abs(s.values[60:] - s.values[:-60]))
        thetas[code] = float(np.median(mv))
    print("THETA (median |60-min move|):", {k: round(v, 1) for k, v in thetas.items()})
    rows, ev = [], []
    for code, dd in raw.items():
        for dday, v in dd.items():
            rows += features_for(code, dday, v, thetas[code])
            ev += level_events(code, dday, v, thetas[code])
    return pd.DataFrame(rows), pd.DataFrame(ev), thetas


FEATS = ["wall_defense", "band_flow", "cover_fuel", "vol_climax", "iv_against", "pcr_div",
         "fut_cover", "wall_shift"]
CTRL = ["leg_len", "leg_age", "extension", "tod"]


def day_ic(df, f, y):
    ics = []
    for _, g in df.groupby("day"):
        g = g[[f, y]].dropna()
        if len(g) >= 8 and g[f].nunique() > 2:
            ics.append(g[f].rank().corr(g[y].rank()))
    ics = np.array([x for x in ics if np.isfinite(x)])
    if len(ics) < 10:
        return np.nan, np.nan, len(ics)
    return ics.mean(), ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))), len(ics)


def bh(p, q=0.10):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), bool)
    if not ok.any():
        return out
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m = ok.sum()
    thr = q * np.arange(1, m + 1) / m
    passed = np.where(p[idx] <= thr)[0]
    if len(passed):
        out[idx[:passed.max() + 1]] = True
    return out


def main():
    import math

    def p_norm(t):                        # two-sided; ~100 days, normal is close to t
        return math.erfc(abs(t) / math.sqrt(2))

    def welch(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        t = (a.mean() - b.mean()) / se if se else float("nan")
        return t, p_norm(t)
    if os.path.exists(CACHE) and "--rebuild" not in sys.argv:
        P = pd.read_parquet(CACHE)
        E = pd.read_parquet(CACHE.replace(".parquet", "_events.parquet"))
    else:
        P, E, th = build()
        P.to_parquet(CACHE)
        E.to_parquet(CACHE.replace(".parquet", "_events.parquet"))
    P["month"] = pd.to_datetime(P["day"]).dt.month
    P["half"] = np.where(P["month"] <= 8, "Jun-Aug", "Sep-Nov")
    print(f"\npanel: {len(P):,} grid points in a leg, {P['day'].nunique()} days, "
          f"race known {P['race'].notna().mean():.0%}, base turn-first rate {P['race'].mean():.1%}")
    print(P.groupby("code").agg(days=("day", "nunique"), n=("race", "size"),
                                turn_first=("race", "mean")).round(3).to_string())

    res = []
    for code in ["ALL"] + sorted(P["code"].unique()):
        g = P if code == "ALL" else P[P["code"] == code]
        for f in FEATS + CTRL:
            for y in ("race", "fwd30"):
                ic, t, n = day_ic(g, f, y)
                h1 = day_ic(g[g["half"] == "Jun-Aug"], f, y)[0]
                h2 = day_ic(g[g["half"] == "Sep-Nov"], f, y)[0]
                p = p_norm(t) if np.isfinite(t) else np.nan
                res.append(dict(index=code, feature=f, target=y, ic=ic, t=t, days=n, p=p,
                                ic_JunAug=h1, ic_SepNov=h2))
    R = pd.DataFrame(res)
    # race: + IC = turn more likely (features signed 'against'); fwd30 is signed WITH the leg,
    # so a working 'against' feature has NEGATIVE fwd30 IC
    R["fdr10"] = bh(R["p"].values)
    pd.set_option("display.width", 200)
    print("\n== per-day rank IC (feature signed so + = against the leg) ==")
    print(R[R["index"] == "ALL"].round(3).to_string(index=False))
    print("\n== survivors of Benjamini-Hochberg 10% over", len(R), "tests ==")
    S = R[R["fdr10"]].sort_values("p")
    print(S.round(3).to_string(index=False) if len(S) else "none")

    # buckets for the readable story: top vs bottom fifth of each feature, turn-first rate
    print("\n== turn-first rate by feature fifth (ALL indices; base %.1f%%) ==" % (100 * P["race"].mean()))
    lines = []
    for f in FEATS + CTRL:
        g = P[[f, "race", "day"]].dropna()
        if g[f].nunique() < 5:
            continue
        q = pd.qcut(g[f].rank(method="first"), 5, labels=False)
        r = g.groupby(q)["race"].mean()
        lines.append(dict(feature=f, Q1_low=r.iloc[0], Q2=r.iloc[1], Q3=r.iloc[2], Q4=r.iloc[3],
                          Q5_high=r.iloc[-1], n=len(g)))
    print((pd.DataFrame(lines).set_index("feature") * [100, 100, 100, 100, 100, 1]).round(1).to_string())

    # combined, leave-one-month-out
    def fit_lr(X, y, lam=1.0, it=50):
        X = np.c_[np.ones(len(X)), X]
        w = np.zeros(X.shape[1])
        Rg = lam * np.eye(X.shape[1])
        Rg[0, 0] = 0
        for _ in range(it):
            p_ = 1 / (1 + np.exp(-X @ w))
            g = X.T @ (p_ - y) + Rg @ w
            H = (X * (p_ * (1 - p_))[:, None]).T @ X + Rg
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        return w

    def predict(w, X):
        return 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ w)))

    def roc_auc_score(y, s_):
        y = np.asarray(y)
        r = pd.Series(np.asarray(s_)).rank().values
        n1 = (y == 1).sum()
        n0 = len(y) - n1
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    G = P.dropna(subset=["race"]).copy()
    for c in FEATS + CTRL:
        G[c] = G[c].fillna(G[c].median())
        lo, hi = G[c].quantile([0.01, 0.99])
        G[c] = G[c].clip(lo, hi)
    out = {}
    for name, cols in (("price only", CTRL), ("price + positioning", CTRL + FEATS),
                       ("positioning only", FEATS)):
        pr = np.full(len(G), np.nan)
        for m in sorted(G["month"].unique()):
            tr, te = G["month"] != m, G["month"] == m
            X = G[cols]
            mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)
            w = fit_lr(((X[tr] - mu) / sd).values, G.loc[tr, "race"].values)
            pr[te.values] = predict(w, ((X[te] - mu) / sd).values)
        G["_pr"] = pr
        aucs = [roc_auc_score(g["race"], g["_pr"]) for _, g in G.groupby("month")
                if g["race"].nunique() == 2 and len(g) > 50]
        out[name] = (roc_auc_score(G["race"], pr), np.mean(aucs), np.min(aucs))
    print("\n== combined model, leave-one-month-out AUC (0.50 = coin) ==")
    for k, (a, am, amin) in out.items():
        print(f"  {k:22s} AUC {a:.3f}  (month avg {am:.3f}, worst month {amin:.3f})")

    # level events
    if len(E):
        E = E.dropna(subset=["rejected"])
        E["writers_side"] = np.where(E["writers"] > 0.01, "writers ADDING (defend)",
                                     np.where(E["writers"] < -0.01, "writers CUTTING (give up)",
                                              "flat"))
        print(f"\n== level test: price reaches the biggest OI wall ahead ({len(E)} touches, "
              f"{E['day'].nunique()} days) — rejected = half-theta back before half-theta through ==")
        t_ = E.groupby(["writers_side"]).agg(touches=("rejected", "size"), rejected=("rejected", "mean"),
                                             days=("day", "nunique"))
        print(t_.round(3).to_string())
        print(E.groupby(["code", "writers_side"])["rejected"].agg(["size", "mean"]).round(3).to_string())
        a = E[E["writers_side"].str.startswith("writers ADD")]["rejected"]
        b = E[E["writers_side"].str.startswith("writers CUT")]["rejected"]
        if len(a) > 5 and len(b) > 5:
            # day-clustered: compare per-day means
            da = E[E["writers_side"].str.startswith("writers ADD")].groupby("day")["rejected"].mean()
            db = E[E["writers_side"].str.startswith("writers CUT")].groupby("day")["rejected"].mean()
            tt = welch(da, db)
            print(f"  adding vs cutting: {a.mean():.1%} vs {b.mean():.1%} rejected; "
                  f"day-level Welch t {tt[0]:.2f} p {tt[1]:.3f}")
    R.to_csv(CACHE.replace(".parquet", "_ic.csv"), index=False)


if __name__ == "__main__":
    main()
