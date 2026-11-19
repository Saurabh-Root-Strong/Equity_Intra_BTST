"""Cross-check of study_strike_anomaly.py on the Index F&O lane's own logs, with the SCREEN's
definition: calls - puts OI change SINCE YESTERDAY'S CLOSE (NSE's change-in-OI field), the
same 'never reached' guard, every reading-minute sampled at 15-min marks 09:45-14:30."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"d:\Python Projects\Equity_Intra_BTST")
from eqbtst import index_fno as f  # noqa: E402

TIMES = [f"{h:02d}:{m:02d}" for h in range(9, 15) for m in (0, 15, 30, 45)
         if "09:45" <= f"{h:02d}:{m:02d}" <= "14:30"]
rows = []
for code in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"):
    for day in sorted(f.logged_days(code)):
        if day == pd.Timestamp.now().date():
            continue                                      # today is not finished
        exps = [e for e in f.logged_expiries(code, day) if e >= day.isoformat()]
        if not exps:
            continue
        df = f.frame(f.chain_log(code, day, exps[0]))
        if df.empty:
            continue
        u = df.groupby("t")["u"].first().dropna().sort_index()
        piv = df.pivot_table(index="t", columns=["side", "K"], values="doi", aggfunc="last")
        piv = piv.reindex(u.index).ffill()
        for hm in TIMES:
            t = pd.Timestamp(f"{day} {hm}")
            pos = u.index.searchsorted(t, side="right") - 1
            if pos < 0 or (t - u.index[pos]) > pd.Timedelta(minutes=5):
                continue                                  # no fresh reading at that mark
            spot = u.iloc[pos]
            hi_sf, lo_sf = u.iloc[:pos + 1].max(), u.iloc[:pos + 1].min()
            fut = u.iloc[pos + 1:]
            if fut.empty:
                continue
            rowp = piv.iloc[pos]
            typ = np.nanmedian(np.abs(rowp.get("CE", pd.Series(dtype=float)).values
                                      - rowp.get("PE", pd.Series(dtype=float)).reindex(
                                          rowp.get("CE", pd.Series(dtype=float)).index).values)) or 1.0
            for k in sorted(df["K"].unique()):
                dist = (k - spot) / spot
                if not (0.002 <= abs(dist) <= 0.012):
                    continue
                up = dist > 0
                if (up and hi_sf >= k) or ((not up) and lo_sf <= k):
                    continue
                a, b = rowp.get(("CE", k), np.nan), rowp.get(("PE", k), np.nan)
                if np.isnan(a) or np.isnan(b) or (abs(a) + abs(b)) == 0:
                    continue
                sgn = 1 if up else -1
                wrong = -sgn * (a - b)
                rows.append(dict(code=code, day=str(day), t=hm, K=k, up=up, dist=abs(dist),
                                 mv=sgn * (spot / u.iloc[0] - 1),
                                 tilt=wrong / (abs(a) + abs(b)), size=wrong / typ,
                                 touch=bool(fut.max() >= k) if up else bool(fut.min() <= k)))
P = pd.DataFrame(rows)
P["anom"] = (P["tilt"] >= 0.2) & (P["size"] >= 1.0)
P["db"] = pd.cut(P["dist"], [0, .004, .007, .012], labels=["0.2-0.4%", "0.4-0.7%", "0.7-1.2%"])
P["tb"] = P["t"].str[:2]
print(f"lane logs: rows {len(P):,} | days {P['day'].nunique()} | anomaly {P['anom'].mean():.1%}")
for db, g in P.groupby("db", observed=True):
    a, b = g[g["anom"]], g[~g["anom"]]
    print(f"  {db}: anomaly touched {a['touch'].mean():.1%} (n {len(a)}) vs normal "
          f"{b['touch'].mean():.1%} (n {len(b)})")
out = []
for d, g in P.groupby("day"):
    diffs = [c.loc[c["anom"], "touch"].mean() - c.loc[~c["anom"], "touch"].mean()
             for _, c in g.groupby(["code", "db", "tb"], observed=True)
             if c["anom"].any() and (~c["anom"]).any()]
    if diffs:
        out.append((d, np.mean(diffs)))
dd = pd.Series(dict(out))
print("matched per day (anomaly − normal touch):",
      " ".join(f"{d[5:]} {v * 100:+.0f}pp" for d, v in dd.items()),
      f"| mean {dd.mean() * 100:+.1f}pp, t {dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))):+.2f}")
