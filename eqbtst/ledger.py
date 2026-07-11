"""
ledger.py — PAPER-FIRST ledger for the accumulation-overnight edge.

Clones the discipline proven in Tradebot's btst_signal.py: emit tonight's screened
longs, reconcile the next-morning exits, score paper-vs-backtest, and — critically —
guard against SILENTLY DROPPED positions (a stale OPEN that never reconciled inflates
the record by hiding a trade) and watch a LOCKED rule for DECAY rather than tuning it.

Nothing auto-executes. Exit fill = next trading day's open (the honest overnight-gap
proxy; the live layer will improve this to the morning strength peak).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import config, data, screen

_COLS = ["date", "symbol", "entry_px", "clr", "deliv_per", "vol_ratio", "score",
         "exit_date", "exit_px", "gross_bps", "net_bps", "status"]


def _load(path=None) -> pd.DataFrame:
    path = path or config.LEDGER
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=_COLS)


def _save(df: pd.DataFrame, path=None) -> None:
    path = path or config.LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _as_date(x) -> dt.date:
    if isinstance(x, dt.date) and not isinstance(x, dt.datetime):
        return x
    return pd.Timestamp(x).date()


# --- emit ---------------------------------------------------------------------
def emit(date: pd.Timestamp | None = None, path=None) -> pd.DataFrame:
    out = screen.screen(date)
    date = out.attrs["date"]
    led = _load(path)
    print(screen.format_screen(out))
    if out.empty or not out.attrs.get("risk_on", False):
        print("\n  (nothing logged — no candidates or regime off.)")
        return led
    for r in out.itertuples():
        key = (led["date"].astype(str) == str(_as_date(date))) & (led["symbol"] == r.symbol)
        if key.any():                          # re-emit same day -> refresh the OPEN row
            m = key & (led["status"] == "OPEN")
            led.loc[m, ["entry_px", "clr", "deliv_per", "vol_ratio", "score"]] = \
                [r.close_price, r.clr, r.deliv_per, r.vol_ratio, r.score]
        else:
            led = pd.concat([led, pd.DataFrame([{
                "date": _as_date(date), "symbol": r.symbol, "entry_px": r.close_price,
                "clr": r.clr, "deliv_per": r.deliv_per, "vol_ratio": r.vol_ratio,
                "score": r.score, "status": "OPEN"}])], ignore_index=True)
    _save(led, path)
    print(f"\n  paper-logged {len(out)} long(s). Run --reconcile next morning.")
    return led


# --- reconcile ----------------------------------------------------------------
def _next_open_fill(symbol: str, after: dt.date, eod: pd.DataFrame) -> tuple[dt.date, float] | None:
    """First available session strictly after `after` for `symbol`: fill at its open.
    The honest overnight-gap exit proxy (close_t -> next_open)."""
    rows = eod[(eod["symbol"] == symbol) & (eod["trade_date"] > pd.Timestamp(after))]
    if rows.empty:
        return None
    nxt = rows.sort_values("trade_date").iloc[0]
    return nxt["trade_date"].date(), float(nxt["open_price"])


def reconcile(path=None) -> None:
    led = _load(path)
    if led.empty:
        print("  ledger empty."); return
    opens = led[led["status"] == "OPEN"]
    if opens.empty:
        print("  no open positions."); return
    lo = (opens["date"].map(_as_date).min())
    eod = data.load_eod(start=(pd.Timestamp(lo)).strftime("%Y-%m-%d"))
    filled = 0
    for i, r in opens.iterrows():
        fill = _next_open_fill(r["symbol"], _as_date(r["date"]), eod)
        if fill is None:
            continue                            # exit day not in archive yet
        xd, xp = fill
        gross = (xp / r["entry_px"] - 1.0) * 1e4
        led.at[i, "exit_date"] = xd
        led.at[i, "exit_px"] = xp
        led.at[i, "gross_bps"] = round(gross, 1)
        led.at[i, "net_bps"] = round(gross - config.COST_BPS, 1)
        led.at[i, "status"] = "CLOSED"
        filled += 1
    _save(led, path)
    print(f"  reconciled {filled} position(s).")


# --- integrity + scorecard ----------------------------------------------------
def _report_open(led: pd.DataFrame) -> int:
    """Flag STALE opens (exit day passed, reconcile never ran). They are excluded
    from the P&L and therefore OVERSTATE the edge — never let them be invisible."""
    o = led[led["status"] == "OPEN"]
    if o.empty:
        return 0
    today = dt.date.today()
    eod_max = data.last_trading_date().date()
    stale = o[o["date"].map(lambda d: _as_date(d) < eod_max)]
    if len(stale):
        print(f"\n  ⚠ {len(stale)} STALE-OPEN position(s) — exit day passed, --reconcile "
              "never ran. They are EXCLUDED below, so the P&L OVERSTATES the edge.")
        for r in stale.itertuples():
            print(f"      {_as_date(r.date)}  {r.symbol}")
        print("  FIX: run  python -m eqbtst.cli reconcile")
    elif len(o):
        print(f"\n  {len(o)} open position(s) awaiting next-morning exit (normal).")
    return len(stale)


def state(path=None) -> dict:
    """Ledger snapshot for the dashboard: open positions (HOLD, exit next morning),
    closed positions, and P&L summary. No printing."""
    led = _load(path)
    openp = led[led["status"] == "OPEN"].copy()
    closed = led[led["status"] == "CLOSED"].copy()
    summ = {}
    if not closed.empty:
        p = closed["net_bps"].to_numpy(float)
        summ = {"trades": len(p), "win": float((p > 0).mean()),
                "mean_bps": float(p.mean()), "total_bps": float(p.sum()),
                "worst_bps": float(p.min())}
    return {"open": openp, "closed": closed, "summary": summ}


def scorecard(path=None) -> None:
    led = _load(path)
    _report_open(led)
    c = led[led["status"] == "CLOSED"]
    print("\n  ── PAPER SCORECARD (closed overnight longs, net of "
          f"{config.COST_BPS:.0f}bps) ──")
    if c.empty:
        print("    no closed positions yet."); return
    p = c["net_bps"].to_numpy(float)
    eq = np.cumsum(p); dd = (eq - np.maximum.accumulate(eq)).min()
    sh = p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else 0.0
    print(f"    trades {len(p)}   win {100*(p>0).mean():.0f}%   mean {p.mean():+.1f}bps   "
          f"total {p.sum():+.0f}bps   maxDD {dd:+.0f}   Sharpe {sh:+.2f}   worst {p.min():+.0f}")
    print("    NOTE: exit = next-open proxy. The live layer (VWAP/RSI morning exit) "
          "targets better fills; treat this as the FLOOR.")
    _health(c)


def _health(c: pd.DataFrame, window: int = 20) -> None:
    """Watch the LOCKED rule for DECAY (don't tune it). The edge rides risk-on
    overnight drift; a risk-off regime flips it. Compare trailing window vs full."""
    p = c.sort_values("date")["net_bps"].to_numpy(float)
    if len(p) < window + 5:
        print(f"\n    edge-health: n={len(p)} (<{window+5}) — accruing, no decay read yet")
        return
    full_m, rec = p.mean(), p[-window:]
    rec_m, rec_wr = rec.mean(), 100 * (rec > 0).mean()
    decayed = rec_m <= 0 or (full_m > 0 and rec_m < 0.4 * full_m)
    flag = "⚠ DECAY — stand aside / re-validate" if decayed else "✓ edge intact"
    print(f"\n    edge-health (last {window} vs full): trailing {rec_m:+.1f}bps "
          f"(full {full_m:+.1f})  win {rec_wr:.0f}%   {flag}")
