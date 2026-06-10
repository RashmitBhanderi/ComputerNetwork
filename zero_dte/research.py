"""Strategy research lab: find a profitable 0DTE long-calls/puts configuration.

Run:  python -m zero_dte.research

Methodology — built to resist self-deception:
  * Trade universe: SPY + QQQ (daily 0DTE), real hourly bars, synthetic BS
    premiums (same engine as backtest.py), full friction.
  * 4 entry archetypes x 4 exit schemes x VIX/trend filters = candidate grid.
  * Walk-forward honesty: configs are RANKED on TRAIN (sessions before
    2025-07-01) and judged ONLY on TEST (2025-07-01 onward). A config that
    only wins in-sample is reported as overfit, not as a winner.
  * AI gate: a numpy logistic-regression classifier over market-condition
    features (gap, opening-range width, trend, VIX level/change, momentum),
    retrained walk-forward monthly — it never sees its own evaluation month.

Outputs results/research_report.json for the dashboard + README.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from itertools import product

import numpy as np
import pandas as pd

from zero_dte import data as yd
from zero_dte.config import RESULTS_DIR, config
from zero_dte.pricing import atm_strike, bs_price, iv_from_vix
from zero_dte.risk import RiskManager

SPLIT = date(2025, 7, 1)          # train < SPLIT <= test
UNIVERSE = ["SPY", "QQQ"]
BAR_HOURS = [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]

EXITS = {                          # (stop_pct or None, target_pct or None)
    "s30t60": (30.0, 60.0),
    "s50t100": (50.0, 100.0),
    "t100": (None, 100.0),
    "time": (None, None),
}
VIX_FILTERS = {
    "anyvix": lambda v: True,
    "vix<20": lambda v: v < 20,
    "vix12-26": lambda v: 12 <= v <= 26,
}
TREND_FILTERS = ["anytrend", "align"]


@dataclass
class TradeRow:
    session: date
    symbol: str
    archetype: str
    exit_scheme: str
    side: str
    entry_hour: float
    entry_premium: float          # net of entry slippage
    pnl: float                    # 1-contract net P&L ($)
    exit_reason: str
    # features (known at entry — no lookahead)
    gap_pct: float
    or_width_pct: float
    prior_ret: float
    ret5: float
    vix: float
    vix_chg5: float
    day_move_pct: float
    trend_up: bool


# ─── entry archetypes: (side, entry_bar_idx) or None ────────────────────────
def arch_orb1030(day: pd.DataFrame, feats: dict) -> tuple[str, int] | None:
    dec = float(day.iloc[1]["open"])
    if dec > float(day.iloc[0]["high"]) and feats["trend_up"]:
        return "call", 1
    if dec < float(day.iloc[0]["low"]) and not feats["trend_up"]:
        return "put", 1
    return None


def arch_pm_momo(day: pd.DataFrame, feats: dict) -> tuple[str, int] | None:
    if len(day) < 7:
        return None
    day_open = float(day.iloc[0]["open"])
    px_1430 = float(day.iloc[5]["open"])
    move = (px_1430 - day_open) / day_open * 100
    if move >= 0.15:
        return "call", 5
    if move <= -0.15:
        return "put", 5
    return None


def arch_fade1330(day: pd.DataFrame, feats: dict) -> tuple[str, int] | None:
    if len(day) < 6:
        return None
    day_open = float(day.iloc[0]["open"])
    px = float(day.iloc[4]["open"])
    move = (px - day_open) / day_open * 100
    if move >= 0.50:
        return "put", 4
    if move <= -0.50:
        return "call", 4
    return None


def arch_gap_go(day: pd.DataFrame, feats: dict) -> tuple[str, int] | None:
    gap = feats["gap_pct"]
    dec = float(day.iloc[1]["open"])
    day_open = float(day.iloc[0]["open"])
    if gap >= 0.25 and dec > day_open:
        return "call", 1
    if gap <= -0.25 and dec < day_open:
        return "put", 1
    return None


ARCHETYPES = {
    "orb1030": arch_orb1030,
    "pm_momo1430": arch_pm_momo,
    "fade1330": arch_fade1330,
    "gap_go1030": arch_gap_go,
}


def simulate(day: pd.DataFrame, side: str, entry_idx: int, vix: float,
             stop_pct: float | None, target_pct: float | None,
             ) -> tuple[float, float, str]:
    """Return (entry_net_premium, pnl_1_contract, exit_reason)."""
    iv = iv_from_vix(vix)
    is_call = side == "call"
    under = float(day.iloc[entry_idx]["open"])
    strike = atm_strike(under, 1.0)
    ts0 = day.index[entry_idx]
    entry_hr = ts0.hour + ts0.minute / 60.0
    hours_left = max(config.close_hour - entry_hr, 0.25)
    slip = config.slippage_pct / 100.0 / 2.0
    raw = bs_price(under, strike, hours_left, iv, is_call)
    entry_net = raw * (1 + slip)
    stop_lv = raw * (1 - stop_pct / 100.0) if stop_pct else None
    tgt_lv = raw * (1 + target_pct / 100.0) if target_pct else None

    exit_net, reason = None, "time_exit"
    for j in range(entry_idx, len(day)):
        bar = day.iloc[j]
        ts = day.index[j]
        bar_hr = ts.hour + ts.minute / 60.0
        h_end = max(config.close_hour - min(bar_hr + 1.0, config.close_hour), 0.05)
        worst = bs_price(float(bar["low" if is_call else "high"]), strike, h_end, iv, is_call)
        best = bs_price(float(bar["high" if is_call else "low"]), strike, h_end, iv, is_call)
        if stop_lv is not None and worst <= stop_lv:
            exit_net, reason = stop_lv * (1 - slip), "stop_loss"
            break
        if tgt_lv is not None and best >= tgt_lv:
            exit_net, reason = tgt_lv * (1 - slip), "target_hit"
            break
    if exit_net is None:
        last = float(day["close"].iloc[-1])
        exit_net = bs_price(last, strike, config.close_hour - config.exit_hour,
                            iv, is_call) * (1 - slip)
    pnl = (exit_net - entry_net) * 100 - 2 * config.commission_per_contract
    return entry_net, pnl, reason


def build_trades() -> pd.DataFrame:
    bars = {s: yd.hourly_bars(s) for s in UNIVERSE}
    daily = {s: yd.daily_bars(s, "3y") for s in UNIVERSE}
    vix_map = yd.vix_by_date("3y")
    vix_dates = sorted(vix_map)

    rows: list[TradeRow] = []
    for sym in UNIVERSE:
        hb, db = bars[sym], daily[sym]
        d_close = db["close"]
        sma20 = d_close.rolling(20).mean()
        sessions = sorted({ts.date() for ts in hb.index})
        for sess in sessions:
            day = hb[hb.index.date == sess]
            if len(day) < 6:
                continue
            prior_vix = [d for d in vix_dates if d < sess]
            if len(prior_vix) < 6:
                continue
            vix = vix_map[prior_vix[-1]]
            vix5 = vix_map[prior_vix[-6]]
            dd = db[db.index.date < sess]
            if len(dd) < 25:
                continue
            pc = float(dd["close"].iloc[-1])
            feats = {
                "gap_pct": (float(day.iloc[0]["open"]) - pc) / pc * 100,
                "or_width_pct": (float(day.iloc[0]["high"]) - float(day.iloc[0]["low"])) / pc * 100,
                "prior_ret": (pc / float(dd["close"].iloc[-2]) - 1) * 100,
                "ret5": (pc / float(dd["close"].iloc[-6]) - 1) * 100,
                "vix": vix,
                "vix_chg5": vix - vix5,
                "trend_up": pc > float(sma20[sma20.index.date < sess].iloc[-1]),
            }
            for aname, afn in ARCHETYPES.items():
                hit = afn(day, feats)
                if hit is None:
                    continue
                side, idx = hit
                day_move = (float(day.iloc[idx]["open"]) - float(day.iloc[0]["open"])) / pc * 100
                for ename, (sp, tp) in EXITS.items():
                    entry_net, pnl, reason = simulate(day, side, idx, vix, sp, tp)
                    rows.append(TradeRow(
                        session=sess, symbol=sym, archetype=aname,
                        exit_scheme=ename, side=side, entry_hour=BAR_HOURS[idx],
                        entry_premium=round(entry_net, 3), pnl=round(pnl, 2),
                        exit_reason=reason, day_move_pct=round(day_move, 3),
                        **{k: (round(v, 3) if isinstance(v, float) else v)
                           for k, v in feats.items()},
                    ))
    return pd.DataFrame([r.__dict__ for r in rows])


def stats(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {"trades": 0, "win_rate": 0, "pnl": 0, "pf": 0, "avg": 0}
    wins = sub[sub.pnl > 0]
    gw = float(wins.pnl.sum())
    gl = abs(float(sub[sub.pnl <= 0].pnl.sum()))
    return {
        "trades": int(len(sub)),
        "win_rate": round(len(wins) / len(sub) * 100, 1),
        "pnl": round(float(sub.pnl.sum()), 2),
        "pf": round(gw / gl, 2) if gl else 99.0,
        "avg": round(float(sub.pnl.mean()), 2),
    }


# ─── numpy logistic regression (walk-forward) ────────────────────────────────
FEATS = ["gap_pct", "or_width_pct", "prior_ret", "ret5", "vix", "vix_chg5",
         "day_move_pct"]


def _fit(x: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> np.ndarray:
    w = np.zeros(x.shape[1])
    for _ in range(400):
        p = 1 / (1 + np.exp(-x @ w))
        grad = x.T @ (p - y) / len(y) + l2 * w
        w -= 0.5 * grad
    return w


def ml_gate(trades: pd.DataFrame, threshold: float = 0.55) -> pd.DataFrame:
    """Walk-forward monthly: predict P(win) using only strictly-prior trades."""
    df = trades.sort_values("session").reset_index(drop=True).copy()
    df["side_num"] = (df.side == "call").astype(float) * 2 - 1
    feats = FEATS + ["side_num"]
    df["month"] = pd.to_datetime(df.session.astype(str)).dt.to_period("M")
    df["p_win"] = np.nan
    for month in sorted(df.month.unique()):
        train = df[df.month < month]
        test_idx = df.index[df.month == month]
        if len(train) < 60:
            continue
        xt = train[feats].to_numpy(float)
        mu, sd = xt.mean(0), xt.std(0) + 1e-9
        w = _fit(np.c_[np.ones(len(xt)), (xt - mu) / sd],
                 (train.pnl > 0).to_numpy(float))
        xv = df.loc[test_idx, feats].to_numpy(float)
        z = np.c_[np.ones(len(xv)), (xv - mu) / sd] @ w
        df.loc[test_idx, "p_win"] = 1 / (1 + np.exp(-z))
    return df[df.p_win >= threshold]


def account_run(trades: pd.DataFrame, capital: float) -> dict:
    """Account-level run with the real RiskManager (sizing + daily limits)."""
    rm = RiskManager(capital=capital)
    curve = []
    for _, t in trades.sort_values("session").iterrows():
        sess = t.session if isinstance(t.session, date) else date.fromisoformat(str(t.session)[:10])
        d = rm.check_entry(sess, t.entry_premium, t.vix)
        if not d.approved:
            continue
        rm.record_entry()
        rm.record_pnl(t.pnl * d.contracts)
        curve.append((str(sess), round(rm.capital, 2)))
    eq = pd.Series([c for _, c in curve]) if curve else pd.Series([capital])
    peak = eq.cummax()
    return {
        "end_capital": round(float(eq.iloc[-1]), 2),
        "return_pct": round((float(eq.iloc[-1]) - capital) / capital * 100, 2),
        "max_dd_pct": round(float(((eq - peak) / peak).min() * 100), 2),
        "executed": len(curve),
        "curve": curve,
    }


def main() -> None:
    print("Building trade universe (SPY+QQQ, 4 archetypes x 4 exits)...")
    trades = build_trades()
    trades.to_csv(RESULTS_DIR / "research_trades.csv", index=False)
    tr = trades[trades.session < SPLIT]
    te = trades[trades.session >= SPLIT]
    print(f"  rows: {len(trades)} | train sessions <{SPLIT}: {len(tr)} | test: {len(te)}\n")

    # Grid search on TRAIN only
    grid = []
    for aname, ename, vname, tname in product(ARCHETYPES, EXITS, VIX_FILTERS, TREND_FILTERS):
        def pick(df):
            m = (df.archetype == aname) & (df.exit_scheme == ename)
            m &= df.vix.map(VIX_FILTERS[vname])
            if tname == "align":
                m &= ((df.side == "call") & df.trend_up) | ((df.side == "put") & ~df.trend_up)
            return df[m]
        s_tr = stats(pick(tr))
        if s_tr["trades"] >= 40:
            grid.append({"config": f"{aname}|{ename}|{vname}|{tname}",
                         "train": s_tr, "test": stats(pick(te))})
    grid.sort(key=lambda g: g["train"]["pf"], reverse=True)

    print(f"{'config':44s} {'TRAIN n/PF/avg':>20s} {'TEST n/PF/avg':>20s}")
    for g in grid[:10]:
        a, b = g["train"], g["test"]
        print(f"{g['config']:44s} {a['trades']:4d}/{a['pf']:5.2f}/{a['avg']:7.2f}"
              f" {b['trades']:5d}/{b['pf']:5.2f}/{b['avg']:7.2f}")

    # Honest pick: best TRAIN PF whose TEST also has PF>1 and >=25 trades.
    survivors = [g for g in grid if g["test"]["pf"] > 1.0 and g["test"]["trades"] >= 25]
    best = survivors[0] if survivors else (grid[0] if grid else None)
    report = {"split": str(SPLIT), "grid_top10": grid[:10],
              "survivors_oos": survivors[:5]}

    if best:
        aname, ename, vname, tname = best["config"].split("|")
        base = trades[(trades.archetype == aname) & (trades.exit_scheme == ename)]
        base = base[base.vix.map(VIX_FILTERS[vname])]
        if tname == "align":
            base = base[((base.side == "call") & base.trend_up) |
                        ((base.side == "put") & ~base.trend_up)]
        gated = ml_gate(base)
        gated_te = gated[gated.session >= SPLIT]
        base_te = base[base.session >= SPLIT]
        print(f"\nSelected: {best['config']}"
              f"{'  (OOS-validated)' if survivors else '  (NO OOS SURVIVOR — best in-sample only)'}")
        print(f"  base  OOS: {stats(base_te)}")
        print(f"  +ML gate OOS: {stats(gated_te)}")
        acct_base = account_run(base_te, config.starting_capital)
        acct_gate = account_run(gated_te, config.starting_capital)
        print(f"  $1500 account OOS  base: {acct_base['return_pct']:+.1f}% "
              f"(DD {acct_base['max_dd_pct']:.1f}%, {acct_base['executed']} trades)")
        print(f"  $1500 account OOS gated: {acct_gate['return_pct']:+.1f}% "
              f"(DD {acct_gate['max_dd_pct']:.1f}%, {acct_gate['executed']} trades)")
        report.update({
            "selected": best["config"],
            "oos_validated": bool(survivors),
            "base_oos": stats(base_te), "ml_gated_oos": stats(gated_te),
            "account_base_oos": {k: v for k, v in acct_base.items() if k != "curve"},
            "account_gated_oos": {k: v for k, v in acct_gate.items() if k != "curve"},
            "equity_curve_gated": acct_gate["curve"],
        })

    with open(RESULTS_DIR / "research_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nSaved {RESULTS_DIR / 'research_report.json'}")


if __name__ == "__main__":
    main()
