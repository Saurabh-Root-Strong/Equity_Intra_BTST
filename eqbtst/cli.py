"""
cli.py — command entrypoint.

    python -m eqbtst.cli screen                 tonight's ranked longs (latest close)
    python -m eqbtst.cli screen --date 2026-07-10
    python -m eqbtst.cli emit                   screen + paper-log the longs
    python -m eqbtst.cli reconcile              fill yesterday's exits (next-open)
    python -m eqbtst.cli scorecard              paper P&L + edge-health
    python -m eqbtst.cli calibrate              self-calibrate net-edge/cost/size (learns knob)
    python -m eqbtst.cli backtest               walk-forward validation (gated, net)
    python -m eqbtst.cli backtest --ungated     show why the gate is mandatory
    python -m eqbtst.cli tilt                   DCM 1-2wk sector tilt as of a close
    python -m eqbtst.cli tilt-history           backfill the tilt history -> parquet
    python -m eqbtst.cli tilt-measure           DOES the sector tilt condition the
                                                overnight payoff? (pre-registered rule)
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from . import backtest, calibrate, ledger, screen, sector_tilt


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eqbtst")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("screen"); s.add_argument("--date", default=None); s.add_argument("--top", type=int, default=None)
    e = sub.add_parser("emit"); e.add_argument("--date", default=None)
    sub.add_parser("reconcile")
    sub.add_parser("scorecard")
    sub.add_parser("calibrate")
    b = sub.add_parser("backtest")
    b.add_argument("--ungated", action="store_true"); b.add_argument("--top", type=int, default=None)
    b.add_argument("--start", default="2018-01-01")
    t = sub.add_parser("tilt"); t.add_argument("--date", default=None)
    th = sub.add_parser("tilt-history")
    th.add_argument("--start", default="2017-06-01")
    tm = sub.add_parser("tilt-measure")
    tm.add_argument("--start", default="2018-01-01")
    tm.add_argument("--ungated", action="store_true")
    lv = sub.add_parser("levels"); lv.add_argument("symbol")
    lv.add_argument("--frame", default="1D", choices=["1D", "1W"])
    lv.add_argument("--date", default=None)

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
    elif a.cmd == "calibrate":
        print(calibrate.format_calibration(calibrate.calibrate()))
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
    elif a.cmd == "tilt":
        from . import data as _d
        d = pd.Timestamp(a.date) if a.date else _d.last_trading_date()
        df, meta = sector_tilt.sector_tilt(d)
        if df.empty:
            print(f"  No sector tilt available as of {pd.Timestamp(d).date()}.")
            return
        print(f"\n  DCM 1-2wk FORWARD SECTOR TILT — as of close {meta['as_of']}")
        print(f"  regime {meta['state']} · {meta['verdict']} · size {meta['size_hint']:.2f} · "
              f"{meta['divergence']} · dispersion {meta['dispersion']:.2f} · "
              f"{meta['n_ow']} OW / {meta['n_uw']} UW of {meta['n_sectors']}")
        print(f"  {'sector':<30}{'tilt':<13}{'#':>4}{'rank':>7}{'rs2w%':>8}{'breadth':>9}"
              f"{'nliq':>6}{'relbps':>8}{'conf':>6}")
        for sec, r in df.iterrows():
            print(f"  {str(sec)[:29]:<30}{str(r['tilt']):<13}{int(r['rank_pos']):>4}"
                  f"{r['rank']:>7.2f}{r['rs_2w']:>+8.1f}{r['accum_breadth']:>9.2f}"
                  f"{int(r['n_liq']):>6}{r['est_rel_bps']:>+8.0f}{r['confidence']:>6.2f}")
        print("\n  CONTEXT ONLY — a 10-day RELATIVE sector call. Nothing in the engine reads "
              "it.\n  Run `tilt-measure` to see whether it touches the OVERNIGHT payoff.")
    elif a.cmd == "tilt-history":
        print(f"  Building sector-tilt history from {a.start} "
              f"-> {sector_tilt.TILT_HISTORY}")
        h = sector_tilt.build_history(start=a.start)
        print(f"  wrote {len(h)} sector-days, "
              f"{h['trade_date'].nunique() if len(h) else 0} dates")
    elif a.cmd == "levels":
        from . import data as _d, indicators as _i, live as _l, config as _c
        end = pd.Timestamp(a.date) if a.date else _d.last_trading_date()
        raw = _d.load_eod(start=(end - pd.Timedelta(days=900)).strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"))
        g = raw[raw["symbol"] == a.symbol.upper()].sort_values("trade_date")
        if g.empty:
            print(f"  no EOD rows for {a.symbol.upper()}"); return
        cd = _i.adjust_corporate_actions(g.rename(columns={
            "trade_date": "ts", "open_price": "open", "high_price": "high",
            "low_price": "low", "close_price": "close"})[["ts", "open", "high", "low", "close"]]
            .reset_index(drop=True))
        lb = _c.SR_DAILY_LOOKBACK if a.frame == "1D" else _c.SR_LOOKBACK
        if a.frame == "1W":
            cd = _l.weekly_frame(cd)
        cd = cd.tail(lb).reset_index(drop=True)
        sr = _i.sr_levels(cd, lookback=lb)
        if not sr:
            print("  not enough bars for a level read"); return
        px, atr = float(cd["close"].iloc[-1]), sr["atr"]
        print(f"\n  {a.symbol.upper()}  {a.frame}  as-of {cd['ts'].iloc[-1].date()}   "
              f"close {px:.2f}   ATR {atr:.2f} ({100*atr/px:.1f}%)   "
              f"zone +/-{_c.SR_TOL_ATR*atr:.2f}   window {len(cd)} bars")
        print(f"  {'level':>10}{'dist%':>8}{'pivots':>8}{'visits':>8}{'bars':>7}"
              f"{'closes':>8}{'time%':>7}   side")
        for x, t in sorted(sr["levels"], key=lambda z: -z[0]):
            z = _i.zone_visits(cd, x, atr)
            side = "RES" if x > px else "SUP"
            print(f"  {x:>10.2f}{100*(x-px)/px:>+8.1f}{t:>8}{z['visits']:>8}{z['bars']:>7}"
                  f"{z['closes']:>8}{z['time_pct']:>7.1f}   {side}")
        print(f"\n  nearest support {sr['support']} x{sr['sup_touches']} pivots / "
              f"{sr['sup_visits']} visits ({sr['sup_time_pct']}% of the window)")
        print(f"  nearest resist  {sr['resistance']} x{sr['res_touches']} pivots / "
              f"{sr['res_visits']} visits ({sr['res_time_pct']}% of the window)")
        print("\n  PIVOTS = 5-bar swing extremes (what sup×/res× on the board count).")
        print("  VISITS = separate approaches into the zone — closer to what the eye counts.")
        print("  Both are DESCRIPTIVE: measured on 8yr, neither ranks levels by what they pay.\n")
    elif a.cmd == "tilt-measure":
        m = sector_tilt.measure_overnight(start=a.start, gated=not a.ungated)
        print(sector_tilt.format_measurement(m))
        if not a.ungated and a.start == "2018-01-01":
            sector_tilt.save_measurement(m)      # only the canonical run feeds the board
            print(f"  persisted -> {sector_tilt.TILT_MEASUREMENT}")


if __name__ == "__main__":
    main()
