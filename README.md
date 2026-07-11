# Equity_Intra_BTST

Institutional-grade **delivery-conviction overnight equity** system for the
Nifty-200 F&O cash universe (~270 single-stock names). No options, **no theta**.

The whole system is built on one honest, data-proven premise and refuses to
pretend the edge is bigger than the 8-year archive says it is.

---

## The edge (proven on 8 years of Daily_Cash_Market data, 2018–2026)

A **smart-money accumulation footprint** at the close predicts an **overnight
drift up** into the next morning:

| Feature | Threshold | Meaning |
|---|---|---|
| `clr` = (close−low)/(high−low) | ≥ 0.70 | buyers held into the bell |
| `deliv_per` | ≥ 60% | shares actually **taken**, not churned |
| `deliv_spike` = deliv% − own 20d median | ≥ +10pp | **fresh, abnormal** accumulation |
| `vol_ratio` = vol / own 20d median | ≥ 2× | real participation surge |
| `ret` (day return) | ≥ +1% | demand in control |

**Long only.** A weak close is *not* a short — shorts fight the overnight drift.

### What the data actually says (do not skip this)

1. **The tradeable payoff is the OVERNIGHT GAP** (close → next-morning). Conviction
   names drift up overnight; random names mean-revert.
2. **Holding into DAY 2 is DEAD.** Buy-next-open / sell-next-close is negative
   *every* year (median −7 to −27bps). The gap reverts intraday on day 2.
   → capture the overnight gap **only**.
3. **It is COST-BOUND.** At a 22bps retail BTST round-trip the naive signal is
   net ~0-to-negative in 2024–26. It carried the 8yr sample almost entirely on
   **2020–21** (COVID bull, +70bps/night net).
4. **The regime gate is mandatory.** Restricting to a Nifty uptrend (close > 50d MA)
   revives net edge to positive in **4 of 5** recent years. Ungated, 2024–26 is a
   loss. Top-N cross-sectional selection improves it further (2024 +35, 2026 +7
   net; only 2025 flat).

Reproduce it yourself:

```
python -m eqbtst.cli backtest --top 5      # regime-gated, net of cost, per year
python -m eqbtst.cli backtest --ungated    # proves why the gate is mandatory
```

### Why this is worth building even though net edge is thin

The **gross** directional edge is alive (win 58–66% overnight, into 2025–26). The
22bps retail cost is what eats it. **Live tick data is the lever** that converts
gross → net: better entry than the blind closing print, exit at the morning
strength peak (VWAP/RSI) instead of the raw open, lower-cost execution. That is
Phase 2. Phase 1 proves the signal on paper first.

---

## Usage (Phase 1 — EOD screen + paper ledger, nothing auto-executes)

```
python -m eqbtst.cli screen              # tonight's ranked longs (latest close)
python -m eqbtst.cli emit                # screen + paper-log them
python -m eqbtst.cli reconcile           # next morning: fill exits (next-open proxy)
python -m eqbtst.cli scorecard           # paper P&L + edge-health decay monitor
python -m eqbtst.cli backtest --top 5    # walk-forward validation
```

Reads the **Daily_Cash_Market** DuckDB EOD archive read-only
(`data/market_data.duckdb`). Configure the path in `eqbtst/config.py`.

## Discipline (borrowed from the proven Tradebot BTST loop)

- **Rule is LOCKED** — thresholds fixed by the 8yr validation, no in-sample tuning.
  A tripwire test fails if they change silently.
- **Paper-first** — nothing auto-executes.
- **Stale-open guard** — an unreconciled position is never silently dropped from
  the P&L (a dropped losing trade flatters the record).
- **Edge-health monitor** — watches the LOCKED rule for *decay* (trailing vs full);
  the edge rides risk-on drift and dies when the regime flips.

## Architecture

```
eqbtst/
  config.py      LOCKED thresholds, cost, paths, universe
  data.py        read-only DCM DuckDB loader (EOD spine + Nifty index)
  features.py    accumulation footprint (causal medians, no lookahead) + score
  regime.py      mandatory Nifty 50d-MA gate
  screen.py      nightly ranked top-N candidates
  ledger.py      paper emit / reconcile / scorecard / edge-health
  backtest.py    walk-forward validation (gated, net, per-year)
  cli.py         entrypoint
tests/           offline unit tests (no DB/network)
```

## Roadmap

- **Phase 1 (done)** — EOD screen + regime gate + paper ledger + validation.
- **Phase 2** — live Fyers tick layer (reuse Tradebot's connector + tick mirror +
  market calendar): VWAP + proactive RSI for entry/exit timing on the screened
  names → the execution alpha that converts gross edge to net.
- **Phase 3** — dashboard (ranked candidates, regime badge, live board, honest
  expectancy vs cost floor).
- **Phase 4** — decay-aware sizing + go-live only after paper tracks backtest.
