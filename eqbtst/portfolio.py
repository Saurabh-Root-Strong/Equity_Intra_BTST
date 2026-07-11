"""
portfolio.py — turn ranked candidates into a deployable overnight book.

Applies the risk cases that protect the validated long edge (none of which the raw
signal handles): per-sector concentration cap (N longs in one sector = a single
overnight macro bet), a hard position count, and equal-weight sizing. Long-only.

This is deliberately simple and mechanical — the edge is thin, so the job here is
to NOT give it back through concentration or oversizing, not to add cleverness.
"""
from __future__ import annotations

import pandas as pd

from . import config, data


def select(cand: pd.DataFrame) -> pd.DataFrame:
    """Rank-respecting greedy selection with a per-sector cap, then top-N, then
    equal weights. `cand` must be sorted best-first (highest score) and carry a
    `symbol` column. Returns the book with `sector` and `weight` columns added."""
    if cand.empty:
        return cand.assign(sector=[], weight=[])
    sectors = data.load_sectors()
    cand = cand.copy()
    cand["sector"] = cand["symbol"].map(lambda s: sectors.get(s, f"_{s}"))

    picked, per_sec = [], {}
    for r in cand.itertuples():
        if len(picked) >= config.TOP_N:
            break
        n = per_sec.get(r.sector, 0)
        if n >= config.MAX_PER_SECTOR:
            continue                       # sector full — skip, keep scanning down the rank
        per_sec[r.sector] = n + 1
        picked.append(r.Index)

    book = cand.loc[picked].copy()
    book["weight"] = round(1.0 / len(book), 4) if len(book) else 0.0   # equal-weight
    return book
