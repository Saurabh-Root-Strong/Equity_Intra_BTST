"""
index_fno_fyers.py — TEST DATA ONLY: turn Tradebot's Fyers option-chain capture into Index F&O
session logs, so the page can be exercised on a real session while the market is shut.

    python -m eqbtst.index_fno_fyers import 2026-11-05 2026-11-06
    python -m eqbtst.index_fno_fyers remove            # delete every imported log

Live capture never uses this: from the next session the page logs NSE's own chain as usual.

SOURCE  Tradebot/data/intraday/live/<day>_chain_snapshots.parquet (the VM's Fyers capture,
        pulled home by sync_from_vm / eod_sync) + <day>_candles.parquet (index level).

WHAT IS CONVERTED, AND HOW IT DIFFERS FROM NSE'S CHAIN (checked 6 Nov 2026)
  * OI, change in OI and volume are SHARES in Fyers; divided by the lot here (every value is a
    whole multiple of it — checked per import). Fyers' first snapshot of 6 Nov (09:14) equals
    the 5 Nov bhavcopy exactly (Nifty 25000 CE 11,960,400 vs 11,960,390 shares), so the
    untagged "nearest" series is the nearest listed expiry; it is matched from the bhavcopy.
  * Timestamps are UTC; converted to IST. Snapshots before 09:15 are dropped: the 09:14 one
    still carries the previous day's volume and no change in OI.
  * One IV per strike (call = put in 100% of rows) — Fyers' model, kept as the 'NSE IV'
    field; the page's own IV is recomputed from last prices (no bid/ask in Fyers' chain).
  * 51 strikes per snapshot (ATM +-25 on the listed gap). Max pain and whole-chain PCR are
    therefore over those 51 strikes, not the whole chain.
  * No index futures: Tradebot's futures_oi rows for indices do not match NSE's futures
    (6 Nov Nifty 11.09M vs 12.09M official shares, volume 20x off).
Every row carries "src": "fyers" and the page says so in a banner.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import index_fno as f

TRADEBOT_LIVE = Path(r"D:\Python Projects\Tradebot\data\intraday\live")
CODE = {"NSE:NIFTY50-INDEX": "NIFTY", "NSE:NIFTYBANK-INDEX": "BANKNIFTY",
        "NSE:FINNIFTY-INDEX": "FINNIFTY", "NSE:MIDCPNIFTY-INDEX": "MIDCPNIFTY"}
LOTS = dict(f.LOT_FALLBACK)


def _nearest_expiry(code: str, day: dt.date):
    from . import data
    with data._connect() as c:
        r = c.execute("""select min(expiry_date) from fno_bhavcopy where trade_date = ?
                         and symbol = ? and instrument = 'OPTIDX' and expiry_date >= ?""",
                      [day, code, day]).fetchone()
    return pd.Timestamp(r[0]).date() if r and r[0] else None


def _official(code: str, day: dt.date, expiry: dt.date) -> pd.DataFrame:
    from . import data
    with data._connect() as c:
        return c.execute("""select strike_price k, option_type s, open_interest oi, chg_in_oi doi
                            from fno_bhavcopy where trade_date = ? and symbol = ?
                            and instrument = 'OPTIDX' and expiry_date = ?""",
                         [day, code, expiry]).df()


def convert_day(day: dt.date, src: Path = TRADEBOT_LIVE) -> dict:
    """{code: [rows]} in index_fno's log format, plus a per-code check report."""
    ch = pd.read_parquet(src / f"{day}_chain_snapshots.parquet")
    cd = pd.read_parquet(src / f"{day}_candles.parquet")
    ch["t"] = ch["ts"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.floor("s")
    cd = cd[cd["resolution"] == "1sec"].copy()
    cd["t"] = pd.to_datetime(cd["ts"]).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    out, report = {}, {}
    for sym, g in ch.groupby("symbol"):
        code = CODE.get(sym)
        if not code:
            continue
        lot = LOTS[code]
        bad = int(((g["oi"] % lot) != 0).sum())
        expiry = _nearest_expiry(code, day)
        spot = cd[cd["symbol"] == sym].sort_values("t")[["t", "close"]]
        snaps = sorted(g["t"].unique())
        rows, last_fp = [], None
        for t in snaps:
            t = pd.Timestamp(t).to_pydatetime()
            if not f.in_session(t):
                continue
            s = spot[spot["t"] <= t]
            u = float(s["close"].iloc[-1]) if len(s) else None
            if not u:
                continue
            x = g[g["t"] == t]
            ce, pe = {}, {}
            for r in x.itertuples():
                v = (r.ltp if r.ltp > 0 else None, r.ltpch, r.oi / lot, r.oich / lot,
                     r.volume / lot, r.iv if r.iv > 0 else None, None, None)
                (ce if r.side == "CE" else pe)[float(r.strike)] = v
            both = sorted(set(ce) & set(pe))
            if len(both) < f.MIN_STRIKES:
                continue
            tot = [sum(v[2] for v in ce.values()), sum(v[4] for v in ce.values()),
                   sum(v[2] for v in pe.values()), sum(v[4] for v in pe.values())]
            chain = dict(ts=t, spot=u, expiry=expiry, ce=ce, pe=pe, strikes=both, tot=tot,
                         mp=f.max_pain({k: v[2] for k, v in ce.items()},
                                       {k: v[2] for k, v in pe.items()}), fetched=t)
            row = f.compact(chain)
            row.update(f.model_iv(row, expiry, t) or {"v": []})
            fp = f._fp(row)
            if fp == last_fp:
                continue
            last_fp = fp
            row["src"] = "fyers"
            rows.append(row)
        out[code] = rows
        # check: the last snapshot's change-on-the-day vs the official bhavcopy that evening
        rep = dict(expiry=str(expiry), rows=len(rows), not_whole_lots=bad,
                   first=rows[0]["t"][11:19] if rows else None,
                   last=rows[-1]["t"][11:19] if rows else None)
        if rows:
            off = _official(code, day, expiry)
            lce, lpe = f.row_sides(rows[-1])
            ks = f.window(sorted(set(lce) & set(lpe)), rows[-1]["a"], f.BAND)
            agree, bias_l, bias_o = [], 0.0, 0.0
            for o in off.itertuples():
                d = (lce if o.s == "CE" else lpe).get(float(o.k))
                if not d or float(o.k) not in ks:
                    continue
                bias_l += d["oi"]
                bias_o += o.oi / lot
                if d["doi"] and o.doi:
                    agree.append(np.sign(d["doi"]) == np.sign(o.doi))
            rep.update(sign_vs_official=round(float(np.mean(agree)) * 100, 1) if agree else None,
                       oi_vs_official=round((bias_l / bias_o - 1) * 100, 2) if bias_o else None)
        report[code] = rep
    return {"rows": out, "report": report}


def import_days(days: list, force: bool = False) -> dict:
    rep = {}
    for day in days:
        res = convert_day(day)
        for code, rows in res["rows"].items():
            p = f._path("chain", code, day)
            existing = f.im._load_jsonl(p) if (p.exists() or p.with_suffix(".jsonl.gz").exists()) else []
            if any(r.get("src") != "fyers" for r in existing) and not force:
                res["report"][code]["skipped"] = "a real NSE log exists for this day"
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        rep[str(day)] = res["report"]
    f._READ.clear()
    return rep


def remove() -> int:
    """Delete every log made only of imported rows. A log with any NSE row is left alone."""
    n = 0
    for p in list(f.DATA_DIR.glob("chain_*_????????.jsonl")) + list(
            f.DATA_DIR.glob("chain_*_????????.jsonl.gz")):
        rows = f.im._load_jsonl(p.with_suffix("") if p.suffix == ".gz" else p)
        if rows and all(r.get("src") == "fyers" for r in rows):
            p.unlink()
            n += 1
    f._READ.clear()
    return n


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "import":
        days = [dt.date.fromisoformat(d) for d in sys.argv[2:] if not d.startswith("--")]
        print(json.dumps(import_days(days, force="--force" in sys.argv), indent=1))
    elif len(sys.argv) > 1 and sys.argv[1] == "remove":
        print(f"removed {remove()} imported log file(s)")
    else:
        print(__doc__)
