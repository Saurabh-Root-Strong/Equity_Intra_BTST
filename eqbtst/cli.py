"""
cli.py — command entrypoint.

    python -m eqbtst.cli screen                 tonight's ranked longs (latest close)
    python -m eqbtst.cli screen --date 2026-07-10
    python -m eqbtst.cli emit                   screen + paper-log the longs
    python -m eqbtst.cli reconcile              fill yesterday's exits (next-open)
    python -m eqbtst.cli scorecard              paper P&L + edge-health
    python -m eqbtst.cli backtest               walk-forward validation (gated, net)
    python -m eqbtst.cli backtest --ungated     show why the gate is mandatory
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from . import backtest, ledger, screen


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eqbtst")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("screen"); s.add_argument("--date", default=None); s.add_argument("--top", type=int, default=None)
    e = sub.add_parser("emit"); e.add_argument("--date", default=None)
    sub.add_parser("reconcile")
    sub.add_parser("scorecard")
    b = sub.add_parser("backtest")
    b.add_argument("--ungated", action="store_true"); b.add_argument("--top", type=int, default=None)
    b.add_argument("--start", default="2018-01-01")

    a = ap.parse_args(argv)
    if a.cmd == "screen":
        d = pd.Timestamp(a.date) if a.date else None
        print(screen.format_screen(screen.screen(d, top_n=a.top)))
    elif a.cmd == "emit":
        ledger.emit(pd.Timestamp(a.date) if a.date else None)
    elif a.cmd == "reconcile":
        ledger.reconcile()
    elif a.cmd == "scorecard":
        ledger.scorecard()
    elif a.cmd == "backtest":
        gated = not a.ungated
        tbl = backtest.run(start=a.start, gated=gated, top_n=a.top)
        title = ("REGIME-GATED (Nifty>50MA)" if gated else "UNGATED (why the gate is mandatory)")
        if a.top:
            title += f", top-{a.top}/day"
        title += " — net of cost, LOCKED signal"
        print(backtest.format_run(tbl, title))
        if gated:
            print("\n  D2net (day-2 hold, MIS cost) is negative every year -> NEVER hold into "
                  "day 2. Capture the overnight gap only.")


if __name__ == "__main__":
    main()
