"""
calibrate.py — the SELF-IMPROVING layer. Learns KNOBS, never the arrow.

The signal is LOCKED (config.py, validated on 8yr). This module does NOT re-tune it.
It reads the paper/live ledger each night and answers the one question the audit left
open: *does the realized net-edge (and the real fill cost) match the backtest, or is it
drifting?* — and shrinks the answer toward the backtest prior so a handful of trades can
never flip the model, but a few hundred can.

Two calibrated knobs, both Bayesian-shrunk to the backtest prior:
  1. NET-EDGE  — posterior expected net bps/trade (risk-on regime, the deployable one).
  2. COST      — effective round-trip cost. In PAPER the fills ARE the close/open prints
                 so realized slippage is 0 and cost stays at the assumption; once LIVE
                 fills diverge from the prints, the divergence is measured and folded in.

Output → a SIZE MULTIPLIER in [0, 1] that only CUTS size when live underperforms the
prior (never levers above backtest), plus a trust level driven by sample size and the
edge-health decay flag. Persisted to data/validation/calibration.json for the dashboard
and the (future) sizer to read. Nothing auto-executes.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import backtest, config, ledger

_STATE = config.LEDGER.parent / "calibration.json"

# how much we let live evidence pull away from the backtest prior. τ = the between-year
# net-edge dispersion (regime variability): the prior is "this good on average across
# regimes", and a run of live data is allowed to move the estimate by about that much.
_MIN_TRADES_READ = 10          # below this: prior only, no calibration claim
_MIN_TRADES_TRUST = 40         # above this: the posterior is trusted enough to size up
_SIZE_FLOOR = 0.30             # never size below 30% while the edge is not proven dead
_SIZE_CAP = 1.00               # never lever above backtest-implied size


def _backtest_prior() -> dict:
    """Prior from the LOCKED engine (gated, top-5): a ROBUST net-edge anchor + the
    between-year dispersion (τ). ROBUST on purpose — the audit showed 2020-21 (COVID
    bull) roughly DOUBLES the n-weighted 8yr mean (+20.3) vs the honest deployable
    regime (~+13, matches 2024-26 = +12.3). Anchoring size to the inflated mean would
    over-size; the YEAR-MEDIAN is outlier-robust and lines up with the recent regime."""
    tbl = backtest.run(gated=True, top_n=5)
    if tbl.empty:
        return {"mu0": 0.0, "mu0_mean8yr": 0.0, "tau": 30.0, "n0": 0}
    mu0 = float(tbl["net_ON"].median())                       # robust anchor (COVID-outlier-safe)
    mu0_mean = float(np.average(tbl["net_ON"], weights=tbl["n"]))   # shown for reference only
    # τ: how much the net-edge itself swings year-to-year (regime variability), floored
    tau = float(max(tbl["net_ON"].std(ddof=0), 8.0))
    return {"mu0": round(mu0, 2), "mu0_mean8yr": round(mu0_mean, 2),
            "tau": round(tau, 2), "n0": int(tbl["n"].sum())}


def _posterior(mu0: float, tau: float, m: float, s: float, n: int) -> tuple[float, float]:
    """Normal-normal conjugate update of the MEAN net-edge.
      prior:   μ ~ N(mu0, tau²)
      data:    sample mean m of n trades, per-trade std s → mean var = s²/n
    Returns (posterior_mean, posterior_sd). n=0 → prior unchanged."""
    if n <= 0 or s <= 0:
        return mu0, tau
    prior_prec = 1.0 / (tau * tau)
    data_prec = n / (s * s)
    post_var = 1.0 / (prior_prec + data_prec)
    post_mean = post_var * (mu0 * prior_prec + m * data_prec)
    return post_mean, post_var ** 0.5


def _realized_cost(closed: pd.DataFrame) -> dict:
    """Effective cost from the ledger. In paper, entry_px = close print and exit_px =
    next-open print, so realized cost == the assumption (no incremental slippage). If
    live rows carry ref_close / ref_open (the prints the backtest assumed), measure the
    slippage the live fill actually paid and report the effective cost."""
    base = config.COST_BPS
    if closed.empty or not {"ref_close", "ref_open"}.issubset(closed.columns):
        return {"effective_cost_bps": base, "slippage_bps": 0.0, "n_live_fills": 0,
                "note": "paper fills = close/open prints; cost at assumption"}
    live = closed.dropna(subset=["ref_close", "ref_open"])
    live = live[(live["ref_close"] != live["entry_px"]) | (live["ref_open"] != live["exit_px"])]
    if live.empty:
        return {"effective_cost_bps": base, "slippage_bps": 0.0, "n_live_fills": 0,
                "note": "no divergent live fills yet; cost at assumption"}
    # slippage = worse entry (paid above close) + worse exit (sold below open), in bps
    slip = ((live["entry_px"] / live["ref_close"] - 1.0)
            + (live["ref_open"] / live["exit_px"] - 1.0)) * 1e4
    slip_bps = float(slip.mean())
    return {"effective_cost_bps": round(base + slip_bps, 1), "slippage_bps": round(slip_bps, 1),
            "n_live_fills": int(len(live)), "note": "effective cost = assumption + measured live slippage"}


def calibrate(path=None, persist: bool = True) -> dict:
    """Read the ledger, shrink realized net-edge to the backtest prior, and emit the
    calibrated knobs + size multiplier. Pure read (optionally writes the state file)."""
    prior = _backtest_prior()
    st = ledger.state(path)
    closed = st["closed"]
    if len(closed):                                  # a CLOSED row must have a net_bps
        closed = closed.dropna(subset=["net_bps"]).copy()
    n = len(closed)
    p = closed["net_bps"].to_numpy(float) if n else np.array([])

    if n >= _MIN_TRADES_READ and p.std() > 0:
        m, s = float(p.mean()), float(p.std(ddof=1))
        post_mean, post_sd = _posterior(prior["mu0"], prior["tau"], m, s, n)
        realized = {"mean_bps": round(m, 1), "std_bps": round(s, 0), "win": round(float((p > 0).mean()), 3)}
    else:
        post_mean, post_sd = prior["mu0"], prior["tau"]
        m = float(p.mean()) if n else None
        realized = {"mean_bps": round(m, 1) if n else None, "std_bps": None,
                    "win": round(float((p > 0).mean()), 3) if n else None}

    cost = _realized_cost(closed)
    # net of the effective (possibly live-adjusted) cost, relative to the prior's 22bps
    post_net = post_mean - (cost["effective_cost_bps"] - config.COST_BPS)

    # RECENT-DECAY alarm — a fast regime-flip detector, made NOISE-AWARE: a 20-trade
    # window at ~115bps std has SE ~26bps, so a raw "trailing mean <= 0" trips on noise.
    # Require the recent run to be CONFIDENTLY negative (mean + 1 SE still < 0) before
    # it can force stand-aside — otherwise a good run's noisy tail would zero the size.
    decayed = False
    if n >= 25:
        rec = p[np.argsort(closed["date"].values)][-20:]
        rec_se = rec.std(ddof=1) / np.sqrt(len(rec)) if rec.std() > 0 else 0.0
        decayed = (rec.mean() + rec_se) < 0

    # SIZE MULTIPLIER — only CUTS below 1.0; never levers above backtest.
    #  * posterior confidently negative (mean + 1 SD < 0) -> 0 (edge looks dead)
    #  * recent confident decay                            -> 0 (regime flip)
    #  * < read threshold                                  -> floor (start cautious)
    #  * forming (read..trust)  -> ratio, but HAIRCUT-capped (a lucky small sample
    #                              must not earn full size before enough evidence)
    #  * trusted                -> posterior_net / prior_mu0, clamped [floor, cap]
    _FORMING_CAP = 0.75
    if post_net + post_sd < 0:
        mult, trust = 0.0, "posterior negative — stand aside"
    elif decayed:
        mult, trust = 0.0, "recent DECAY — stand aside"
    elif n < _MIN_TRADES_READ:
        mult, trust = _SIZE_FLOOR, f"prior-only (n={n}<{_MIN_TRADES_READ}); start small"
    else:
        ratio = post_net / prior["mu0"] if prior["mu0"] > 0 else 0.0
        mult = float(np.clip(ratio, _SIZE_FLOOR, _SIZE_CAP))
        if n < _MIN_TRADES_TRUST:                    # confidence haircut until trusted
            mult = min(mult, _FORMING_CAP)
            trust = f"forming (n={n}<{_MIN_TRADES_TRUST}; capped {_FORMING_CAP})"
        else:
            trust = "trusted"

    out = {
        "as_of": dt.date.today().isoformat(),
        "prior": prior,
        "realized": realized,
        "posterior_net_bps": round(post_net, 1),
        "posterior_sd_bps": round(post_sd, 1),
        "cost": cost,
        "decayed": bool(decayed),
        "size_multiplier": round(mult, 2),
        "trust": trust,
        "n_closed": n,
    }
    if persist:
        _STATE.parent.mkdir(parents=True, exist_ok=True)
        _STATE.write_text(json.dumps(out, indent=2))
    return out


def load_state() -> dict | None:
    """Last persisted calibration for the dashboard/sizer. None if never run."""
    if _STATE.exists():
        try:
            return json.loads(_STATE.read_text())
        except Exception:
            return None
    return None


def format_calibration(c: dict) -> str:
    pr, re_ = c["prior"], c["realized"]
    L = ["\n  ── SELF-CALIBRATION (learns the knob, signal stays LOCKED) ──",
         f"    prior (backtest gated top5): {pr['mu0']:+.1f}bps ROBUST year-median  τ={pr['tau']:.0f}  "
         f"(8yr-mean {pr.get('mu0_mean8yr', pr['mu0']):+.1f} = COVID-inflated, not used)  n={pr['n0']}",
         f"    realized (paper/live ledger): "
         + (f"{re_['mean_bps']:+.1f}bps  win {100*re_['win']:.0f}%  n={c['n_closed']}"
            if re_['mean_bps'] is not None else f"none yet (n={c['n_closed']})"),
         f"    → posterior net-edge:        {c['posterior_net_bps']:+.1f}bps  (±{c['posterior_sd_bps']:.0f})",
         f"    → effective cost:            {c['cost']['effective_cost_bps']:.0f}bps "
         f"({c['cost']['note']})",
         f"    → SIZE MULTIPLIER:           {c['size_multiplier']:.2f}   [{c['trust']}]"]
    if c["decayed"]:
        L.append("    ⚠ DECAY flagged — edge underperforming; multiplier forced to 0.")
    return "\n".join(L)
