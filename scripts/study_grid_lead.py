"""Does the CALLS/PUTS grid's colour balance in one block call the NEXT move?

Per index, per captured day: rows of the most-read expiry -> strike_states (the grid's own
code, 15-min blocks, premium judge) on strikes within +-1.2% of the first reading.
score_t = mean sign of 'lean' over all cells of block t (+1 all green, -1 all red).
Index level at each block's last reading = ub. Tests:
  coincident: score_t vs move in block t (expected strong: the cells are partly the move)
  lead:       score_t vs ub[t+h] - ub[t], h = 1, 2, 4 blocks (15 / 30 / 60 min)
  the user's setup: strong score (|s| >= 0.5) while block t's own move was SMALL -> next 30 min
Entry is at the block's last reading; live data runs 1-2 min late, so the real entry is later
and this is an upper bound."""
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, r"d:\Python Projects\Equity_Intra_BTST")
from eqbtst import index_fno as f  # noqa: E402

D = Path(r"d:\Python Projects\Equity_Intra_BTST\data\index_fno")
out = []
for p in sorted(D.glob("chain_*_2026*.jsonl*")):
    code, day = p.name.split("_")[1], p.name.split("_")[2][:8]
    rows = f._rows(Path(str(p).replace(".gz", "")))
    ex = Counter(r.get("e") for r in rows if r.get("e") and r.get("e") != "fut").most_common(1)
    if not ex:
        continue
    rows = [r for r in rows if r.get("e") == ex[0][0]]
    df = f.frame(rows)
    if df.empty:
        continue
    u0 = df["u"].dropna().iloc[0]
    ks = sorted(k for k in df["K"].unique() if abs(k / u0 - 1) <= 0.012)
    stt = f.strike_states(df, ks, 15)
    if stt.empty:
        continue
    sc = stt.groupby("block")["lean"].apply(lambda s: np.sign(s).mean())
    u = df.groupby("t")["u"].first().dropna()
    ub = u.groupby(f.block_start(pd.Series(u.index), 15).values).last()
    g = pd.DataFrame({"score": sc}).join(ub.rename("u"), how="inner").sort_index()
    # only consecutive blocks (a capture gap must not count as one block)
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="15min"))
    g["mv"] = g["u"].diff()
    for h in (1, 2, 4):
        g[f"f{h}"] = g["u"].shift(-h) - g["u"]
    g["code"], g["day"] = code, day
    out.append(g.iloc[1:].dropna(subset=["score", "mv"]))   # block 1 holds the overnight gap

G = pd.concat(out)
print(f"blocks: {len(G)} | days: {G['day'].nunique()} | indices: {sorted(G['code'].unique())}")


def ic(a, b):
    m = a.notna() & b.notna()
    return a[m].rank().corr(b[m].rank()), int(m.sum())


for code, g in list(G.groupby("code")) + [("ALL (scaled)", G)]:
    if code == "ALL (scaled)":
        # put moves on one scale: divide by each index's median |15-min move|
        g = g.copy()
        s = g.groupby("code")["mv"].transform(lambda x: x.abs().median())
        for c in ("mv", "f1", "f2", "f4"):
            g[c] = g[c] / s
    r0, n0 = ic(g["score"], g["mv"])
    r = [ic(g["score"], g[f"f{h}"]) for h in (1, 2, 4)]
    print(f"\n{code}: n={n0}  same block IC {r0:+.2f} | NEXT 15m {r[0][0]:+.2f} "
          f"30m {r[1][0]:+.2f} 60m {r[2][0]:+.2f}  (|IC| needs > {2/np.sqrt(n0):.2f} for ~2se)")
    # the user's setup: big positioning, price not moved yet
    small = g["mv"].abs() <= g["mv"].abs().median()
    for lab, m in (("strong GREEN, price flat", (g["score"] >= 0.5) & small),
                   ("strong RED,   price flat", (g["score"] <= -0.5) & small),
                   ("strong GREEN, any", g["score"] >= 0.5),
                   ("strong RED,   any", g["score"] <= -0.5)):
        x = g.loc[m, "f2"].dropna()
        sgn = 1 if "GREEN" in lab else -1
        if len(x):
            print(f"   {lab}: n={len(x):3d}  next 30m went the called way "
                  f"{(np.sign(x) == sgn).mean():.0%}  mean {sgn * x.mean():+.2f}")

print("\nBASE RATE: next 30m UP, all blocks:", f"{(G['f2'].dropna() > 0).mean():.0%}")
print("per day (all 4 indices): next-30m IC, share of blocks with next 30m up, day's net move sign")
for d, g in G.groupby("day"):
    r, n = ic(g["score"], g["f2"])
    print(f"  {d}: IC {r:+.2f} (n {n})  up-share {(g['f2'].dropna() > 0).mean():.0%}")
