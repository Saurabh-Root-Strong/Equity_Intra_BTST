"""Strike anomaly study (user, 19 Nov 2026).

Claim: a strike ABOVE the index normally shows calls heavier (call OI added > put OI added,
the call is out of the money). When it shows PUTS heavier — put writing at an in-the-money
put, or call writers covering — big players expect the index to get there: the index will
reach / cross that strike. Mirror below the index: calls heavier at a strike under the price
-> the index will fall to it.

Data: Tradebot's archive (data/intraday/live: chain_snapshots ~30 s, nearest expiry, 1-min
candles), all four indices. Sampled every 15 min 09:45-14:30.

Guards against the traps:
  * the strike was NEVER reached earlier in the day (else since-open OI holds positions
    taken when it sat on the other side of the price: path, not intent)
  * compared at the same distance, time of day and move so far (logistic regression with
    those as covariates); the move so far is ITSELF the competing explanation (call writers
    covering as price rises)
  * inference over DAYS: day-cluster bootstrap; all strikes / indices of a day share one
    market path
Outcomes: index touches the strike before the close; within 60 min; signed % move toward the
strike over 60 min.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from study_trend_turns import LIVE, load_day  # noqa: E402

OUT = os.environ.get("SA_OUT", os.path.join(os.environ.get("TEMP", "."), "sa_panel.parquet"))
TIMES = [f"{h:02d}:{m:02d}" for h in range(9, 15) for m in (0, 15, 30, 45)
         if "09:45" <= f"{h:02d}:{m:02d}" <= "14:30"]
DMIN, DMAX = 0.0020, 0.0120          # strikes 0.2% - 1.2% from the index


def rows_for(code: str, day: str, D: dict) -> list:
    spot, oi = D["spot"], D["oi"]
    if "CE" not in oi.columns.get_level_values(0) or "PE" not in oi.columns.get_level_values(0):
        return []
    ks = sorted({k for _, k in oi.columns})
    ce, pe = oi["CE"].reindex(columns=ks), oi["PE"].reindex(columns=ks)
    base_ce, base_pe = ce.bfill().iloc[0], pe.bfill().iloc[0]     # first reading of the day
    s = spot.values
    idx = spot.index
    open_ = s[~np.isnan(s)][0]
    out = []
    for hm in TIMES:
        t = pd.Timestamp(f"{day} {hm}")
        if t not in idx:
            continue
        i = idx.get_loc(t)
        u = s[i]
        if np.isnan(u):
            continue
        hi_sf, lo_sf = np.nanmax(s[:i + 1]), np.nanmin(s[:i + 1])
        fut = s[i + 1:]
        fut60 = s[i + 1:i + 61]
        u60 = s[min(i + 60, len(s) - 1)]
        d_ce = ce.iloc[i] - base_ce
        d_pe = pe.iloc[i] - base_pe
        # last 30 min flow
        j = max(i - 30, 0)
        f_ce, f_pe = ce.iloc[i] - ce.iloc[j], pe.iloc[i] - pe.iloc[j]
        net_all = (d_ce - d_pe).abs()
        typ = float(np.nanmedian(net_all.values)) or 1.0
        for k in ks:
            dist = (k - u) / u
            if not (DMIN <= abs(dist) <= DMAX):
                continue
            up = dist > 0
            # never reached before t (the strike stayed on the same side all day)
            if (up and hi_sf >= k) or ((not up) and lo_sf <= k):
                continue
            a, b = d_ce.get(k, np.nan), d_pe.get(k, np.nan)
            if np.isnan(a) or np.isnan(b):
                continue
            net = a - b
            gross = abs(a) + abs(b)
            sgn = 1 if up else -1
            # 'normal' above the price = calls heavier (net > 0); anomaly = the other way
            wrong = -sgn * net                     # > 0 when the strike shows the anomaly
            tilt = wrong / gross if gross else 0.0
            fn = f_ce.get(k, np.nan) - f_pe.get(k, np.nan)
            touch = bool(np.nanmax(fut) >= k) if up else bool(np.nanmin(fut) <= k)
            touch60 = (bool(np.nanmax(fut60) >= k) if up else bool(np.nanmin(fut60) <= k)) \
                if len(fut60) else False
            out.append(dict(
                code=code, day=day, t=hm, K=k, up=up, dist=abs(dist),
                mv=sgn * (u / open_ - 1),          # move so far TOWARD the strike, %
                wrong=wrong, tilt=tilt, size=wrong / typ,
                pe_add=(b if up else -b), ce_cut=(-a if up else a),   # the two components
                flow=-sgn * fn / typ,
                touch=touch, touch60=touch60,
                r60=sgn * (u60 / u - 1)))
    return out


def build():
    days = sorted({os.path.basename(f)[:10]
                   for f in glob.glob(f"{LIVE}/*_chain_snapshots.parquet")})
    rows = []
    for n, day in enumerate(days, 1):
        D = load_day(day)
        for code, d in D.items():
            rows += rows_for(code, day, d)
        del D
        print(f"\r{n}/{len(days)} {day} rows {len(rows):,}", end="", flush=True)
    print()
    P = pd.DataFrame(rows)
    P.to_parquet(OUT)
    return P


def fit_lr(X, y, lam=1.0, it=60):
    X = np.c_[np.ones(len(X)), X]
    w = np.zeros(X.shape[1])
    R = lam * np.eye(X.shape[1])
    R[0, 0] = 0
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + R @ w
        H = (X * (p * (1 - p))[:, None]).T @ X + R
        st = np.linalg.solve(H, g)
        w -= st
        if np.abs(st).max() < 1e-9:
            break
    return w


def main():
    P = pd.read_parquet(OUT) if os.path.exists(OUT) and "--rebuild" not in sys.argv else build()
    P["tb"] = pd.to_datetime(P["t"], format="%H:%M").dt.hour
    P["db"] = pd.cut(P["dist"], [0, .004, .007, .012], labels=["0.2-0.4%", "0.4-0.7%", "0.7-1.2%"])
    P["anom"] = (P["tilt"] >= 0.2) & (P["size"] >= 1.0)          # clear, sizeable anomaly
    days = P["day"].nunique()
    print(f"rows {len(P):,} | days {days} | indices {sorted(P['code'].unique())} | "
          f"anomaly rows {P['anom'].sum():,} ({P['anom'].mean():.1%})")

    # 1. raw: touch rates, anomaly vs normal, same distance bucket
    print("\n1) Raw touch rate before the close (same distance bucket)")
    for db, g in P.groupby("db", observed=True):
        a, b = g[g["anom"]], g[~g["anom"]]
        print(f"  {db}: anomaly {a['touch'].mean():.1%} (n {len(a):,}, {a['day'].nunique()} days) "
              f"vs normal {b['touch'].mean():.1%} (n {len(b):,})")

    # 2. per-day paired difference within (code, distance, hour) cells
    def cell_diff(g, y):
        out = []
        for _, c in g.groupby(["code", "db", "tb"], observed=True):
            if c["anom"].any() and (~c["anom"]).any():
                out.append(c.loc[c["anom"], y].mean() - c.loc[~c["anom"], y].mean())
        return np.mean(out) if out else np.nan
    print("\n2) Matched difference (anomaly − normal, within index × distance × hour), per DAY")
    for y in ("touch", "touch60", "r60"):
        dd = P.groupby("day").apply(lambda g: cell_diff(g, y), include_groups=False).dropna()
        t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
        unit = "pp" if y != "r60" else "%"
        sc = 100
        print(f"  {y:8s}: mean {dd.mean() * sc:+.2f}{unit}  t {t:+.2f}  days {len(dd)}  "
              f"positive on {(dd > 0).mean():.0%} of days")

    # 3. logistic: does the anomaly add anything over distance, time, move so far?
    print("\n3) Logistic regression, touch before close ~ distance + hours left + move so far "
          "+ anomaly (day-cluster bootstrap)")
    P["hl"] = (pd.Timestamp("2000-01-01 15:30") - pd.to_datetime("2000-01-01 " + P["t"])) \
        .dt.total_seconds() / 3600
    feats = {"tilt (continuous)": "tilt", "flow last 30 min": "flow",
             "put OI added at the strike": "pe_add", "call OI cut at the strike": "ce_cut"}
    rng = np.random.default_rng(7)
    dlist = P["day"].unique()
    P = P[np.isfinite(P[list(feats.values())]).all(axis=1)].reset_index(drop=True)
    dlist = P["day"].unique()
    for lab, f in feats.items():
        X0 = P[["dist", "hl", "mv"]].copy()
        X0[f] = np.sign(P[f]) * np.log1p(np.abs(P[f]) / (P[f].abs().median() or 1.0)) \
            if f in ("pe_add", "ce_cut") else P[f]
        X0 = (X0 - X0.mean()) / X0.std()
        y = P["touch"].astype(float).values
        w = fit_lr(X0.values, y)
        bs = []
        groups = {d: np.where(P["day"].values == d)[0] for d in dlist}
        for _ in range(200):
            pick = np.concatenate([groups[d] for d in rng.choice(dlist, len(dlist))])
            bs.append(fit_lr(X0.values[pick], y[pick])[-1])
        se = np.std(bs)
        print(f"  {lab:28s}: coef {w[-1]:+.3f}  boot se {se:.3f}  z {w[-1] / se:+.2f}   "
              f"(move-so-far coef {w[3]:+.3f})")

    # 4. by index and by direction
    print("\n4) Matched touch difference by index / side (per-day t)")
    for key in ("code", "up"):
        for v, g in P.groupby(key):
            dd = g.groupby("day").apply(lambda x: cell_diff(x, "touch"), include_groups=False).dropna()
            if len(dd) > 3:
                print(f"  {key}={v}: {dd.mean() * 100:+.2f}pp  t {dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))):+.2f}  days {len(dd)}")

    # 5. the user's exact case: Nifty, strike 100-200 pts above, never reached, puts heavier
    n = P[(P["code"] == "NIFTY") & P["up"]].copy()
    n["pts"] = n["dist"] * 25300
    m = n[(n["pts"] >= 90) & (n["pts"] <= 210)]
    a, b = m[m["anom"]], m[~m["anom"]]
    print(f"\n5) Nifty, strike 100-200 pts ABOVE, never reached: puts heavier -> touched before "
          f"close {a['touch'].mean():.1%} (n {len(a)}, {a['day'].nunique()} days) vs calls "
          f"heavier {b['touch'].mean():.1%} (n {len(b)}, {b['day'].nunique()} days)")
    dd = m.groupby("day").apply(lambda x: x.loc[x["anom"], "touch"].mean() -
                                x.loc[~x["anom"], "touch"].mean(), include_groups=False).dropna()
    if len(dd) > 3:
        print(f"   per day (both kinds that day): {dd.mean() * 100:+.1f}pp  "
              f"t {dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))):+.2f}  days {len(dd)}")


if __name__ == "__main__":
    main()
