"""Backtest the 0DTE call/put strategy over 2025-2026 on free Yahoo data.

Run:  python -m zero_dte.backtest  [--start 2025-01-01] [--capital 1500]

Methodology (and its limits — read this):
  * Underlying paths are REAL hourly SPY/QQQ/etc. bars from Yahoo.
  * Option premiums are SYNTHETIC: ATM same-day Black-Scholes seeded with the
    session's VIX close (no free historical 0DTE quote source exists). Real
    fills will differ — treat results as strategy-shape validation, not a
    P&L promise.
  * Friction is modelled: configurable slippage on both sides + commissions.
  * Exits are conservative: within each hourly bar the stop is checked before
    the target, and remaining time value is taken at the bar's END.
  * No lookahead: signals use only data before the 10:30 ET decision bar.

Outputs ``results/backtest_trades.csv`` and ``results/backtest_summary.json``
which the dashboard renders.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date

import pandas as pd

from zero_dte import data as yd
from zero_dte.config import DAILY_0DTE, FRIDAY_0DTE, RESULTS_DIR, WATCHLIST, config
from zero_dte.pricing import atm_strike, bs_price, iv_from_vix, strike_increment
from zero_dte.risk import RiskManager
from zero_dte.signals import Signal, evaluate_session


@dataclass
class BTrade:
    session: str
    symbol: str
    side: str
    score: int
    strike: float
    entry_underlying: float
    entry_premium: float      # net of entry slippage, per share
    exit_premium: float       # net of exit slippage, per share
    contracts: int
    pnl: float                # dollars, net of all friction
    exit_reason: str
    reasons: str


def _entry_hours_left() -> float:
    return config.close_hour - config.entry_hour


def simulate_trade(
    sig: Signal, day_bars: pd.DataFrame, vix: float, contracts: int
) -> tuple[float, float, str, float]:
    """Walk the post-entry hourly bars; return (entry_net, exit_net, reason, strike)."""
    iv = iv_from_vix(vix)
    is_call = sig.side == "call"
    inc = strike_increment(sig.symbol, sig.entry_underlying)
    strike = atm_strike(sig.entry_underlying, inc)
    slip = config.slippage_pct / 100.0 / 2.0   # half-spread each side

    raw_entry = bs_price(sig.entry_underlying, strike, _entry_hours_left(), iv, is_call)
    entry_net = raw_entry * (1 + slip)
    stop_level = raw_entry * (1 - config.stop_loss_pct / 100.0)
    target_level = raw_entry * (1 + config.take_profit_pct / 100.0)

    # Bars strictly after the decision bar (11:30 onward; index 2+ of the day).
    path = day_bars.iloc[2:]
    for ts, bar in path.iterrows():
        hour_end = ts.hour + ts.minute / 60.0 + 1.0
        hours_left = max(config.close_hour - min(hour_end, config.close_hour), 0.05)
        worst_under = float(bar["low"]) if is_call else float(bar["high"])
        best_under = float(bar["high"]) if is_call else float(bar["low"])
        worst_prem = bs_price(worst_under, strike, hours_left, iv, is_call)
        best_prem = bs_price(best_under, strike, hours_left, iv, is_call)
        if worst_prem <= stop_level:                       # stop first: conservative
            return entry_net, stop_level * (1 - slip), "stop_loss", strike
        if best_prem >= target_level:
            return entry_net, target_level * (1 - slip), "target_hit", strike

    # Force-flat near the close at the last bar's close price.
    last_under = float(day_bars["close"].iloc[-1])
    minutes_left = max((config.close_hour - config.exit_hour), 0.05)
    flat_prem = bs_price(last_under, strike, minutes_left, iv, is_call)
    return entry_net, flat_prem * (1 - slip), "time_exit", strike


def run_backtest(start: date, capital: float) -> dict:
    bars = {s: yd.hourly_bars(s) for s in WATCHLIST}
    vix = yd.vix_by_date("3y")
    vix_series = sorted(vix.items())

    def vix_for(session: date) -> float | None:
        prior = [v for d, v in vix_series if d <= session]
        return prior[-1] if prior else None

    sessions = sorted(
        {ts.date() for ts in bars["SPY"].index if ts.date() >= start}
    )
    rm = RiskManager(capital=capital)
    trades: list[BTrade] = []
    equity_curve: list[tuple[str, float]] = []

    for session in sessions:
        rm.start_session(session)
        tradeable = list(DAILY_0DTE)
        if session.weekday() == 4:                # Friday: weeklies expire
            tradeable += FRIDAY_0DTE
        day_vix = vix_for(session)

        for symbol in tradeable:
            sym_bars = bars[symbol]
            day = sym_bars[sym_bars.index.date == session]
            if len(day) < 3 or day_vix is None:
                continue
            sig = evaluate_session(symbol, sym_bars, session)
            if sig is None:
                continue

            # Price first so sizing uses the actual entry cost.
            probe_entry, *_ = simulate_trade(sig, day, day_vix, 1)
            decision = rm.check_entry(session, probe_entry, day_vix)
            if not decision.approved:
                continue

            entry_net, exit_net, reason, strike = simulate_trade(
                sig, day, day_vix, decision.contracts
            )
            n = decision.contracts
            pnl = (exit_net - entry_net) * n * 100 - 2 * config.commission_per_contract * n
            rm.record_entry()
            rm.record_pnl(pnl)
            trades.append(
                BTrade(
                    session=session.isoformat(), symbol=symbol, side=sig.side,
                    score=sig.score, strike=strike,
                    entry_underlying=round(sig.entry_underlying, 2),
                    entry_premium=round(entry_net, 3),
                    exit_premium=round(exit_net, 3),
                    contracts=n, pnl=round(pnl, 2), exit_reason=reason,
                    reasons="; ".join(sig.reasons),
                )
            )
        equity_curve.append((session.isoformat(), round(rm.capital, 2)))

    return _summarise(trades, equity_curve, capital, rm.capital)


def _summarise(
    trades: list[BTrade], curve: list[tuple[str, float]],
    start_capital: float, end_capital: float,
) -> dict:
    df = pd.DataFrame([asdict(t) for t in trades])
    eq = pd.DataFrame(curve, columns=["session", "equity"])

    def block(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {"trades": 0, "win_rate": 0.0, "pnl": 0.0, "profit_factor": 0.0}
        wins = sub[sub.pnl > 0]
        losses = sub[sub.pnl <= 0]
        gross_win = float(wins.pnl.sum())
        gross_loss = abs(float(losses.pnl.sum()))
        return {
            "trades": int(len(sub)),
            "win_rate": round(len(wins) / len(sub) * 100, 1),
            "pnl": round(float(sub.pnl.sum()), 2),
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else float("inf"),
            "avg_win": round(float(wins.pnl.mean()), 2) if len(wins) else 0.0,
            "avg_loss": round(float(losses.pnl.mean()), 2) if len(losses) else 0.0,
        }

    peak = eq.equity.cummax()
    max_dd = round(float(((eq.equity - peak) / peak).min() * 100), 2) if not eq.empty else 0.0

    summary = {
        "start_capital": start_capital,
        "end_capital": round(end_capital, 2),
        "total_return_pct": round((end_capital - start_capital) / start_capital * 100, 2),
        "max_drawdown_pct": max_dd,
        "overall": block(df),
        "by_year": {},
        "by_symbol": {},
        "by_exit": {},
        "note": (
            "Synthetic Black-Scholes 0DTE premiums seeded from VIX over real "
            "hourly underlying data; includes slippage + commissions. "
            "Validation of strategy shape, not a fill-accurate P&L."
        ),
    }
    if not df.empty:
        df["year"] = df.session.str.slice(0, 4)
        for year, sub in df.groupby("year"):
            summary["by_year"][year] = block(sub)
        for sym, sub in df.groupby("symbol"):
            summary["by_symbol"][sym] = block(sub)
        for reason, sub in df.groupby("exit_reason"):
            summary["by_exit"][reason] = block(sub)

    df.to_csv(RESULTS_DIR / "backtest_trades.csv", index=False)
    eq.to_csv(RESULTS_DIR / "backtest_equity.csv", index=False)
    with open(RESULTS_DIR / "backtest_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="0DTE strategy backtest")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--capital", type=float, default=config.starting_capital)
    args = parser.parse_args()
    summary = run_backtest(date.fromisoformat(args.start), args.capital)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
