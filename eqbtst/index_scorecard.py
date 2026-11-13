"""Scorecard for the Index F&O lane: does anything on that page forecast the index?

WHAT IT IS. Every finished session, for each index and each captured expiry, the page's
readings are REBUILT point-in-time at fixed stamps (09:45, 10:15, 11:00, 12:00, 13:00,
14:00, 14:30, 15:00) from the day's log, using only readings at or before the stamp and
the previous session's corrected close as the base. Each reading is stored next to what
the index did afterwards (30 min, 60 min, to the 15:30 close). One row per stamp.

WHY REBUILT, NOT FROZEN LIVE. (1) The five sessions already logged can be graded today
instead of starting from zero. (2) A fix to a reading re-grades every day the same way
(`VERSION` bumps, the file is rebuilt) — a frozen live record would mix old and new
definitions. (3) Nothing live can fail or be forgotten.

PRE-REGISTERED SIGNALS (sign + = the reading that the page's own words call bullish).
Decided before any grade was seen; adding a signal after looking at the grades is how a
scorecard lies to its owner.
  net_bb        bull − bear contracts since the open (15-min blocks, premium judge, the
                17-strike ladder); the 'Day in one picture' bars added up.
  best_says     Σ over the Best bucket (centre ± 3) of call lean + put lean from the day
                states since yesterday's close — the 'Strike says' column.
  checked_says  the same with only IV-confirmed sides — 'Strike says (checked)'. None on
                expiry day (not checked there).
  cp_doi        puts' ΔOI − calls' ΔOI since yesterday over the ladder, as % of its OI.
                Lore: puts written below = support (+), calls written above = lid (−).
  chain_iv      the chain-wide IV move since yesterday's close with the index effect
                removed (vol points). Stored with its own sign; lore reads a rise as fear,
                so the direction test uses −chain_iv.
  mp_gap        (max pain − index) / index, %. Lore: the index is pulled toward max pain.
  pcr_ladder    put OI / call OI over the ladder, minus 1. Lore: above 1 = supported.
Context stored with every row: index, open, previous close, sessions to expiry, expiry-day
flag, the data source (NSE or the Fyers test import), readings used.

TARGETS. fwd_30, fwd_60, fwd_close: the index's % move from the stamp's reading to the
last reading at or before stamp + 30 / 60 min / 15:30. A target whose reading is more than
`MAX_LAG` minutes late (laptop asleep) is left empty.

JUDGING. The panel stays locked until `UNLOCK_DAYS` sessions are scored for an index. Eight
stamps a day are not eight observations: each cell is one stamp × one horizon, so a day
counts once in it. Shown per cell: n, hit rate of the sign, mean move when the signal was
+ and when −, Spearman IC. Expiry days are graded apart. Nov–Jan is one regime: a number
that holds at 40 sessions holds in that regime.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import index_fno as f
from . import index_mgmt as im

VERSION = 1
FILE = "scorecard.jsonl"
STAMPS = ("09:45", "10:15", "11:00", "12:00", "13:00", "14:00", "14:30", "15:00")
HORIZONS = {"fwd_30": 30, "fwd_60": 60, "fwd_close": None}
SESSION_CLOSE = dt.time(15, 30)
MAX_LAG = 6               # minutes a reading may trail its stamp before the stamp is a hole
MIN_END = dt.time(15, 20)  # a log ending earlier has no 'to close' target
UNLOCK_DAYS = 40          # sessions per index before the panel shows grades
BLOCK_MIN = 15
SIGNALS = ("net_bb", "best_says", "checked_says", "cp_doi", "chain_iv", "mp_gap", "pcr_ladder")
# the sign under which each signal's lore calls the index UP
LORE_SIGN = {"net_bb": 1, "best_says": 1, "checked_says": 1, "cp_doi": 1, "chain_iv": -1,
             "mp_gap": 1, "pcr_ladder": 1}
SIGNAL_WORD = {
    "net_bb": "Bull − bear contracts since open (the day's bars)",
    "best_says": "'Strike says' over the Best bucket",
    "checked_says": "'Strike says (checked)' over the Best bucket",
    "cp_doi": "Puts − calls added since yesterday (ladder, % of OI)",
    "chain_iv": "Chain-wide IV move since yesterday's close (− = fear fading)",
    "mp_gap": "Max pain above (+) / below (−) the index",
    "pcr_ladder": "PCR of the ladder − 1",
}


def _path() -> Path:
    return f.DATA_DIR / FILE


def _stamp_dt(day: dt.date, hhmm: str) -> dt.datetime:
    h, m = hhmm.split(":")
    return dt.datetime.combine(day, dt.time(int(h), int(m)))


def _at(rows: list, when: dt.datetime, max_lag: int = MAX_LAG) -> dict | None:
    """The last reading stamped at or before `when`, or None when it trails by more than
    `max_lag` minutes (a hole in the log — the laptop slept, NSE was down)."""
    best = None
    w = when.isoformat(timespec="seconds")
    for r in rows:
        if r.get("t", "") <= w:
            best = r
        else:
            break
    if best is None:
        return None
    t = dt.datetime.fromisoformat(best["t"][:19])
    return best if (when - t) <= dt.timedelta(minutes=max_lag) else None


def ladder_for(rows: list, code: str, day: dt.date) -> tuple:
    """(strikes, centre, step) the page fixes at 09:15: the auto gap from the first straddle,
    the centre at NSE's official open. None when the open is unknown (capture started late)."""
    if not rows:
        return [], None, None
    step, _, _ = f.auto_gap(rows, rows[-1])
    o_lvl, _ = f.day_open(code, day, rows)
    ce, pe = f.row_sides(rows[-1])
    all_ks = f.grid(set(ce) & set(pe), step)
    if o_lvl is None or not all_ks:
        return [], None, step
    centre = f.pick_atm(all_ks, o_lvl)
    return f.window(all_ks, centre, f.BAND), centre, step


def signals_at(rows_upto: list, latest: dict, ks: list, centre, base_row: dict | None,
               expiry_day: bool) -> dict:
    """The seven readings from the log up to one stamp. Pure: no clock, no disk."""
    out = {s: None for s in SIGNALS}
    if not ks or not latest:
        return out
    df = f.frame(rows_upto)
    st_ = f.strike_states(df, ks, BLOCK_MIN, "premium")
    bb = f.bull_bear(st_)
    out["net_bb"] = float(bb["net"].sum()) if not bb.empty else None
    chk = None if expiry_day else f.day_check(df, latest, base_row)
    bmap = f.bucket_map(ks, centre)
    best = [k for k, b in bmap.items() if b == "Best"]
    tab = f.drill_table(latest, df, best, centre, "premium", latest.get("u"), chk, expiry_day)
    if not tab.empty:
        out["best_says"] = int(sum(_says_sign(v) for v in tab["Strike says"]))
        if "Strike says (checked)" in tab.columns and not expiry_day:
            out["checked_says"] = int(sum(_says_sign(v) for v in tab["Strike says (checked)"]))
    ce, pe = f.row_sides(latest)
    c_oi = sum((ce.get(k) or {}).get("oi") or 0 for k in ks)
    p_oi = sum((pe.get(k) or {}).get("oi") or 0 for k in ks)
    c_d = sum((ce.get(k) or {}).get("doi") or 0 for k in ks)
    p_d = sum((pe.get(k) or {}).get("doi") or 0 for k in ks)
    if c_oi + p_oi:
        out["cp_doi"] = float((p_d - c_d) / (c_oi + p_oi) * 100)
    if c_oi:
        out["pcr_ladder"] = float(p_oi / c_oi - 1)
    if chk and chk.get("common") is not None:
        out["chain_iv"] = float(chk["common"])
    u, mp = latest.get("u"), latest.get("mp")
    if u and mp:
        out["mp_gap"] = float((mp - u) / u * 100)
    return out


def _says_sign(v) -> int:
    s = str(v)
    return 1 if s.startswith("🟢") else -1 if s.startswith("🔴") else 0


def score_day(code: str, day: dt.date, expiry: str) -> list:
    """Rows for one index, day and expiry: one per stamp with a reading."""
    rows = f.chain_log(code, day, expiry)
    if not rows:
        return []
    ks, centre, step = ladder_for(rows, code, day)
    base_row = f.close_base(code, day, expiry)
    expiry_day = expiry == day.isoformat()
    try:
        dte = f.sessions_left(day, dt.date.fromisoformat(expiry))
    except Exception:                                        # noqa: BLE001
        dte = None
    prev = f.prev_close(code, day)
    o_lvl, _ = f.day_open(code, day, rows)
    close_row = _at(rows, dt.datetime.combine(day, SESSION_CLOSE), max_lag=10)
    if close_row and close_row["t"][11:16] < MIN_END.strftime("%H:%M"):
        close_row = None
    out = []
    for hhmm in STAMPS:
        when = _stamp_dt(day, hhmm)
        latest = _at(rows, when)
        if latest is None or not latest.get("u"):
            continue
        upto = [r for r in rows if r["t"] <= latest["t"]]
        sig = signals_at(upto, latest, ks, centre, base_row, expiry_day)
        u0 = float(latest["u"])
        rec = dict(v=VERSION, day=day.isoformat(), code=code, e=expiry, stamp=hhmm,
                   t=latest["t"], u=u0, open=o_lvl, prev_close=prev, dte=dte,
                   expiry_day=expiry_day, src=rows[0].get("src") or "nse",
                   n_read=len(upto), ladder=len(ks), centre=centre, step=step,
                   base=("close" if base_row else "first"), **sig)
        for name, mins in HORIZONS.items():
            if mins is None:
                r1 = close_row
            else:
                r1 = _at(rows, when + dt.timedelta(minutes=mins))
                if r1 is not None and r1["t"] <= latest["t"]:
                    r1 = None
            rec[name] = (float(r1["u"]) / u0 - 1) * 100 if r1 and r1.get("u") else None
        out.append(rec)
    return out


def load(code: str | None = None) -> pd.DataFrame:
    rows = [r for r in im._load_jsonl(_path()) if r.get("v") == VERSION
            and (code is None or r.get("code") == code)]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(["day", "code", "e", "stamp"], keep="last")


def scored_days(code: str) -> int:
    df = load(code)
    return int(df["day"].nunique()) if not df.empty else 0


def score_all(today: dt.date | None = None, days_back: int = 400) -> int:
    """Score every finished, not yet scored (at this VERSION) day with a log. An older
    version's rows are dropped so the file holds one definition. Returns days scored."""
    today = today or f._now().date()
    p = _path()
    old = im._load_jsonl(p)
    keep = [r for r in old if r.get("v") == VERSION]
    if len(keep) != len(old):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
    done = {(r.get("day"), r.get("code"), r.get("e")) for r in keep}
    n = 0
    for code in f.CODES:
        for day in f.logged_days(code):
            if day >= today and f._now().time() < dt.time(16, 0):
                continue
            if (today - day).days > days_back:
                continue
            for e in f.logged_expiries(code, day):
                if (day.isoformat(), code, e) in done:
                    continue
                try:
                    recs = score_day(code, day, e)
                except Exception as exc:                     # noqa: BLE001
                    f._log_error("scorecard", f"{code} {day} {e}: {type(exc).__name__}: {exc}")
                    continue
                for r in recs or [dict(v=VERSION, day=day.isoformat(), code=code, e=e,
                                       stamp=None)]:
                    im._append(p, r)
                done.add((day.isoformat(), code, e))
                n += 1
    return n


# ── grading ───────────────────────────────────────────────────────────────────────────
def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 5:
        return None
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def grade(df: pd.DataFrame, signal: str, stamp: str, horizon: str,
          expiry_day: bool | None = False) -> dict:
    """One cell: this signal read at this stamp vs this horizon's move, one row per day
    (nearest captured expiry of that day). expiry_day None = both kinds together."""
    if df.empty or signal not in df.columns or horizon not in df.columns:
        return dict(n=0)
    d = df[df["stamp"].eq(stamp)].dropna(subset=[signal, horizon])
    if expiry_day is not None:
        d = d[d["expiry_day"].eq(expiry_day)]
    if d.empty:
        return dict(n=0)
    d = d.sort_values("e").drop_duplicates("day", keep="first")
    x = d[signal].to_numpy(dtype=float) * LORE_SIGN[signal]
    y = d[horizon].to_numpy(dtype=float)
    live = x != 0
    hit = float(np.mean(np.sign(x[live]) == np.sign(y[live]))) if live.any() else None
    return dict(n=int(len(d)), n_called=int(live.sum()), hit=hit,
                up=float(y[x > 0].mean()) if (x > 0).any() else None,
                down=float(y[x < 0].mean()) if (x < 0).any() else None,
                ic=_spearman(x, y))


def grade_table(df: pd.DataFrame, horizon: str = "fwd_close",
                expiry_day: bool | None = False) -> pd.DataFrame:
    recs = []
    for s in SIGNALS:
        for stamp in STAMPS:
            g = grade(df, s, stamp, horizon, expiry_day)
            recs.append(dict(signal=s, stamp=stamp, **g))
    return pd.DataFrame(recs)


# ── panel ─────────────────────────────────────────────────────────────────────────────
def render(st, code: str) -> None:
    df = load(code)
    n_days = int(df["day"].nunique()) if not df.empty else 0
    with st.expander(f"🔒 Scorecard — does anything here forecast? ({n_days} of "
                     f"{UNLOCK_DAYS} sessions scored)"):
        st.markdown(
            "Every finished session is re-read at eight fixed times from its own log and "
            "stored next to what the index did in the next 30 and 60 minutes and to the "
            "close. Seven readings, **named before any grade was looked at** (see the "
            "module's note): the day's bull − bear bars, 'Strike says' over the Best bucket, "
            "the same IV-checked, puts − calls added, the chain-wide IV move, max pain's "
            "distance, the ladder's PCR. The panel opens at "
            f"**{UNLOCK_DAYS} sessions** per index; before that a hit rate is a coin's "
            "recent streak.")
        st.progress(min(1.0, n_days / UNLOCK_DAYS),
                    text=f"{DISPLAY_OF(code)}: {n_days} session(s) scored — "
                         + (f"about {UNLOCK_DAYS - n_days} trading days to go"
                            if n_days < UNLOCK_DAYS else "unlocked"))
        if df.empty:
            return
        peek = n_days >= UNLOCK_DAYS or st.checkbox(
            f"Peek anyway (n = {n_days}: these numbers are noise, shown grey on purpose)",
            key=f"if_score_peek_{code}")
        if not peek:
            return
        c1, c2, c3 = st.columns(3)
        hz = c1.segmented_control("Horizon", list(HORIZONS), default="fwd_close",
                                  key=f"if_score_hz_{code}",
                                  format_func=lambda h: {"fwd_30": "next 30 min",
                                                         "fwd_60": "next 60 min",
                                                         "fwd_close": "to the close"}[h]) \
            or "fwd_close"
        kind = c2.segmented_control("Days", ["ordinary", "expiry day", "all"],
                                    default="ordinary", key=f"if_score_kind_{code}") \
            or "ordinary"
        ed = {"ordinary": False, "expiry day": True, "all": None}[kind]
        stamp = c3.segmented_control("Read at", list(STAMPS), default="12:00",
                                     key=f"if_score_stamp_{code}") or "12:00"
        recs = []
        for s in SIGNALS:
            g = grade(df, s, stamp, hz, ed)
            recs.append({"Reading": SIGNAL_WORD[s], "days": g.get("n", 0),
                         "called": g.get("n_called"),
                         "hit %": None if g.get("hit") is None else g["hit"] * 100,
                         "move when +": g.get("up"), "move when −": g.get("down"),
                         "IC": g.get("ic")})
        tab = pd.DataFrame(recs)
        cc = st.column_config
        st.dataframe(tab, hide_index=True, width="stretch", column_config={
            "days": cc.NumberColumn("days", format="%d"),
            "called": cc.NumberColumn("called", format="%d",
                                      help="days the reading was not zero"),
            "hit %": cc.NumberColumn("hit %", format="%.0f",
                                     help="share of called days where the index moved the "
                                          "way the reading's lore says"),
            "move when +": cc.NumberColumn("move when + (%)", format="%+.2f"),
            "move when −": cc.NumberColumn("move when − (%)", format="%+.2f"),
            "IC": cc.NumberColumn("IC", format="%+.2f",
                                  help="Spearman rank correlation between the reading and "
                                       "the move; 0 = no relation, ±0.1 is a lot for a "
                                       "daily signal")})
        if n_days < UNLOCK_DAYS:
            st.caption(f":grey[n = {n_days}. A fair coin shows 60–70% on 10 days as often "
                       "as not. Nothing here is a finding yet.]")
        else:
            st.caption("Each cell is one reading at one time of day against one horizon: a "
                       "day counts once. Nov–Jan is one regime; a number that holds here "
                       "holds in that regime until another is seen.")


def DISPLAY_OF(code: str) -> str:
    return f.DISPLAY.get(code, code)


if __name__ == "__main__":
    # python -m eqbtst.index_scorecard [score|show CODE]
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "score":
        print(f"scored {score_all()} day-expiry log(s)")
    elif cmd == "show":
        code = sys.argv[2] if len(sys.argv) > 2 else "NIFTY"
        d = load(code)
        print(f"{code}: {scored_days(code)} session(s), {len(d)} rows")
        if not d.empty:
            pd.set_option("display.width", 200)
            print(d[["day", "e", "stamp", "u", *SIGNALS, *HORIZONS]].to_string(index=False))
            print()
            print(grade_table(d, "fwd_close", None).to_string(index=False))
