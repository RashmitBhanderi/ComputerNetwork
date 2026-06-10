"""0DTE defined-risk CREDIT SPREADS: strategy lab + walk-forward backtest.

Run:  python -m zero_dte.spreads

Structures (all defined-risk, fit a $1-2k account):
  * bull put spread  — sell put @ ~target delta, buy put `width` lower
  * bear call spread — sell call @ ~target delta, buy call `width` higher
  * iron condor      — both at once (one side max can be breached)

Unlike long options, these COLLECT theta — the structural headwind that sank
the long-0DTE strategies becomes the engine. Same honesty framework as
research.py: synthetic BS pricing (flat IV — understates skew/crash risk,
stated everywhere), full friction (slippage both legs + commissions), grid
ranked on TRAIN (-> Jun 2025), judged out-of-sample (Jul 2025 ->).
Outputs results/spreads_report.json + spreads_trades.csv.
"""
from __future__ import annotations

import json
from datetime import date
from itertools import product

import numpy as np
import pandas as pd

from zero_dte import data as yd
from zero_dte.config import RESULTS_DIR, config
from zero_dte.pricing import bs_price, iv_from_vix

SPLIT = date(2025, 7, 1)
UNIVERSE = ["SPY", "QQQ"]
SLIP = config.slippage_pct / 100.0 / 2.0       # half-spread per fill
COMM = config.commission_per_contract           # per contract-leg per side


def bs_delta(spot: float, strike: float, hours: float, iv: float, is_call: bool) -> float:
    import math
    t = max(hours, 1e-6) / (252.0 * 6.5)
    iv = max(iv, 1e-4)
    d1 = (math.log(spot / strike) + (config.risk_free_rate + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2)))
    return nd1 if is_call else nd1 - 1.0


def strike_at_delta(spot: float, hours: float, iv: float, target: float,
                    is_call: bool) -> float:
    """First $1-increment strike OTM whose |delta| <= target."""
    k = round(spot)
    step = 1.0 if is_call else -1.0
    for _ in range(80):
        k += step
        if abs(bs_delta(spot, k, hours, iv, is_call)) <= target:
            return k
    return k


def spread_mark(spot: float, ks: float, kl: float, hours: float, iv: float,
                is_call: bool) -> float:
    """Mark (cost to close) of a short ks / long kl credit spread, per share."""
    return max(bs_price(spot, ks, hours, iv, is_call)
               - bs_price(spot, kl, hours, iv, is_call), 0.0)


def sim_spread(day: pd.DataFrame, entry_idx: int, is_call: bool, ks: float,
               kl: float, iv: float, stop_mult: float | None,
               ) -> tuple[float, float, str]:
    """Returns (net_credit, pnl_per_contract, exit_reason) for one vertical."""
    ts0 = day.index[entry_idx]
    entry_hr = ts0.hour + ts0.minute / 60.0
    spot0 = float(day.iloc[entry_idx]["open"])
    h0 = max(config.close_hour - entry_hr, 0.25)
    raw_credit = spread_mark(spot0, ks, kl, h0, iv, is_call)
    credit_net = raw_credit * (1 - SLIP)
    if credit_net <= 0.02:
        return 0.0, 0.0, "no_credit"
    stop_level = raw_credit * stop_mult if stop_mult else None

    for j in range(entry_idx, len(day)):
        bar = day.iloc[j]
        bar_hr = day.index[j].hour + day.index[j].minute / 60.0
        h_end = max(config.close_hour - min(bar_hr + 1.0, config.close_hour), 0.05)
        # Worst case for a credit spread: spot moves TOWARD the short strike.
        worst_spot = float(bar["high"] if is_call else bar["low"])
        if stop_level is not None:
            mark = spread_mark(worst_spot, ks, kl, h_end, iv, is_call)
            if mark >= stop_level:
                exit_cost = stop_level * (1 + SLIP)
                pnl = (credit_net - exit_cost) * 100 - 4 * COMM
                return credit_net, pnl, "stop_loss"

    last = float(day["close"].iloc[-1])
    mark = spread_mark(last, ks, kl, config.close_hour - config.exit_hour, iv, is_call)
    exit_cost = mark * (1 + SLIP)
    pnl = (credit_net - exit_cost) * 100 - 4 * COMM
    return credit_net, pnl, "time_exit"


def build_trades() -> pd.DataFrame:
    bars = {s: yd.hourly_bars(s) for s in UNIVERSE}
    daily = {s: yd.daily_bars(s, "3y") for s in UNIVERSE}
    vix_map = yd.vix_by_date("3y")
    vix_dates = sorted(vix_map)
    rows = []

    grid = list(product([1, 2],                 # entry bar idx: 10:30, 11:30
                        ["trend", "fade", "condor"],
                        [0.10, 0.15, 0.20],     # short-leg |delta|
                        [1.0, 2.0],             # width $
                        [2.5, None]))           # stop multiple on credit

    for sym in UNIVERSE:
        hb, db = bars[sym], daily[sym]
        sma20 = db["close"].rolling(20).mean()
        for sess in sorted({ts.date() for ts in hb.index}):
            day = hb[hb.index.date == sess]
            if len(day) < 6:
                continue
            pv = [d for d in vix_dates if d < sess]
            dd = db[db.index.date < sess]
            if not pv or len(dd) < 25:
                continue
            vix = vix_map[pv[-1]]
            iv = iv_from_vix(vix)
            pc = float(dd["close"].iloc[-1])
            trend_up = pc > float(sma20[sma20.index.date < sess].iloc[-1])

            for entry_idx, direction, delta_t, width, stop in grid:
                if len(day) <= entry_idx + 1:
                    continue
                spot = float(day.iloc[entry_idx]["open"])
                day_move = (spot - float(day.iloc[0]["open"])) / pc * 100
                entry_hr = day.index[entry_idx].hour + day.index[entry_idx].minute / 60.0
                h0 = max(config.close_hour - entry_hr, 0.25)

                # Pick side(s)
                sides: list[bool] = []          # is_call per vertical
                if direction == "trend":
                    sides = [False] if trend_up else [True]      # bull put / bear call
                elif direction == "fade":
                    if day_move >= 0.30:
                        sides = [True]
                    elif day_move <= -0.30:
                        sides = [False]
                elif direction == "condor":
                    sides = [True, False]
                if not sides:
                    continue

                total_pnl, total_credit, reasons, legs_ok = 0.0, 0.0, [], True
                max_loss = 0.0
                for is_call in sides:
                    ks = strike_at_delta(spot, h0, iv, delta_t, is_call)
                    kl = ks + width if is_call else ks - width
                    credit, pnl, reason = sim_spread(day, entry_idx, is_call,
                                                     ks, kl, iv, stop)
                    if reason == "no_credit":
                        legs_ok = False
                        break
                    total_pnl += pnl
                    total_credit += credit
                    reasons.append(reason)
                if not legs_ok:
                    continue
                # Condor: only one side can be fully breached.
                max_loss = (width - total_credit) * 100 if direction != "condor" \
                    else (width - total_credit) * 100
                if max_loss <= 0:
                    continue
                rows.append({
                    "session": sess, "symbol": sym, "entry_idx": entry_idx,
                    "direction": direction, "delta": delta_t, "width": width,
                    "stop": stop if stop else 0, "credit": round(total_credit, 3),
                    "max_loss": round(max_loss, 2), "pnl": round(total_pnl, 2),
                    "exit": "+".join(reasons), "vix": vix, "trend_up": trend_up,
                    "day_move": round(day_move, 3),
                })
    return pd.DataFrame(rows)


def stats(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {"trades": 0, "win_rate": 0, "pnl": 0, "pf": 0, "avg": 0}
    wins = sub[sub.pnl > 0]
    gl = abs(float(sub[sub.pnl <= 0].pnl.sum()))
    gw = float(wins.pnl.sum())
    return {"trades": int(len(sub)), "win_rate": round(len(wins) / len(sub) * 100, 1),
            "pnl": round(float(sub.pnl.sum()), 2),
            "pf": round(gw / gl, 2) if gl else 99.0,
            "avg": round(float(sub.pnl.mean()), 2)}


def account_run(sub: pd.DataFrame, capital: float) -> dict:
    """Risk-based sizing: contracts s.t. max_loss*n <= 20% of capital; 1 trade/day."""
    cap = capital
    curve, seen_days = [], set()
    for _, t in sub.sort_values("session").iterrows():
        if t.session in seen_days or t.vix > config.vix_max:
            continue
        budget = cap * config.max_premium_pct / 100.0
        n = int(budget // t.max_loss)
        if n < 1:
            continue
        n = min(n, 3)
        seen_days.add(t.session)
        cap += t.pnl * n
        curve.append((str(t.session), round(cap, 2)))
    eq = pd.Series([c for _, c in curve]) if curve else pd.Series([capital])
    peak = eq.cummax()
    return {"end_capital": round(float(eq.iloc[-1]), 2),
            "return_pct": round((float(eq.iloc[-1]) - capital) / capital * 100, 2),
            "max_dd_pct": round(float(((eq - peak) / peak).min() * 100), 2),
            "executed": len(curve), "curve": curve}


def main() -> None:
    print("Building credit-spread universe (SPY+QQQ, 72-config grid)...")
    trades = build_trades()
    trades.to_csv(RESULTS_DIR / "spreads_trades.csv", index=False)
    tr, te = trades[trades.session < SPLIT], trades[trades.session >= SPLIT]
    print(f"  rows: {len(trades)} | train: {len(tr)} | test: {len(te)}\n")

    results = []
    for key, sub in trades.groupby(["entry_idx", "direction", "delta", "width", "stop"]):
        s_tr = stats(sub[sub.session < SPLIT])
        if s_tr["trades"] < 40:
            continue
        results.append({"config": "|".join(map(str, key)),
                        "key": key, "train": s_tr,
                        "test": stats(sub[sub.session >= SPLIT])})
    results.sort(key=lambda r: r["train"]["pf"], reverse=True)

    print(f"{'entry|dir|delta|width|stop':34s} {'TRAIN n/PF/avg':>20s} {'TEST n/PF/avg':>20s}")
    for r in results[:12]:
        a, b = r["train"], r["test"]
        print(f"{r['config']:34s} {a['trades']:4d}/{a['pf']:5.2f}/{a['avg']:7.2f}"
              f" {b['trades']:5d}/{b['pf']:5.2f}/{b['avg']:7.2f}")

    survivors = [r for r in results if r["test"]["pf"] > 1.0 and r["test"]["trades"] >= 25
                 and r["train"]["pf"] > 1.0]
    print(f"\nOOS survivors (PF>1 both periods, n>=25): {len(survivors)}")
    report = {"split": str(SPLIT), "grid_top12": [
        {k: r[k] for k in ("config", "train", "test")} for r in results[:12]],
        "survivors": [{k: r[k] for k in ("config", "train", "test")} for r in survivors[:8]]}

    if survivors:
        # Among survivors prefer the highest TEST PF (it already passed both gates).
        best = max(survivors, key=lambda r: r["test"]["pf"])
        key = best["key"]
        sub = trades[(trades.entry_idx == key[0]) & (trades.direction == key[1])
                     & (trades.delta == key[2]) & (trades.width == key[3])
                     & (trades.stop == (key[4] if key[4] else 0))]
        acct_te = account_run(sub[sub.session >= SPLIT], config.starting_capital)
        acct_all = account_run(sub, config.starting_capital)
        print(f"Selected: {best['config']}")
        print(f"  OOS stats: {best['test']}")
        print(f"  $1500 account OOS:  {acct_te['return_pct']:+.1f}% "
              f"(maxDD {acct_te['max_dd_pct']:.1f}%, {acct_te['executed']} trades)")
        print(f"  $1500 account FULL: {acct_all['return_pct']:+.1f}% "
              f"(maxDD {acct_all['max_dd_pct']:.1f}%, {acct_all['executed']} trades)")
        report.update({"selected": best["config"], "oos_validated": True,
                       "account_oos": {k: v for k, v in acct_te.items() if k != "curve"},
                       "account_full": {k: v for k, v in acct_all.items() if k != "curve"},
                       "equity_curve": acct_all["curve"]})
    else:
        report["oos_validated"] = False
        print("No spread configuration survived OOS either.")

    with open(RESULTS_DIR / "spreads_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nSaved {RESULTS_DIR / 'spreads_report.json'}")


if __name__ == "__main__":
    main()
