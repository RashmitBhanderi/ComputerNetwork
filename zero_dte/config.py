"""Configuration for the 0DTE signal tool.

Strategy scope (deliberately narrow): long CALLS and long PUTS only — no
spreads, no selling. Small account ($1k-$2k) on moomoo via manual copy-trade.

0DTE availability: SPY and QQQ list options expiring every trading day. The
single-name tickers (AAPL, TSLA, NVDA, AMZN) only have weekly Friday expiries,
so they are tradeable by this tool on Fridays only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Mirrors trading_bot's watchlist (kept in sync manually so this package is
# standalone). SPY/QQQ have daily expiries; the rest are Friday-only.
DAILY_0DTE = ["SPY", "QQQ"]
FRIDAY_0DTE = ["AAPL", "TSLA", "NVDA", "AMZN"]
WATCHLIST = DAILY_0DTE + FRIDAY_0DTE


def _f(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class ZConfig:
    # ── account ────────────────────────────────────────────────────────────
    starting_capital: float = field(default_factory=lambda: _f("ZDTE_CAPITAL", 1500.0))

    # ── risk ───────────────────────────────────────────────────────────────
    # Max premium outlay on one trade, as % of current capital.
    max_premium_pct: float = 20.0
    # Hard stop on the option premium (sell if premium falls this % below entry).
    stop_loss_pct: float = 30.0
    # Take profit on the option premium.
    take_profit_pct: float = 60.0
    # Stop trading for the day after losing this % of capital intraday.
    daily_loss_limit_pct: float = 15.0
    max_trades_per_day: int = 2
    # Skip days when VIX is above this (0DTE gamma risk too hot for small acct).
    vix_max: float = 35.0

    # ── signal engine ──────────────────────────────────────────────────────
    # Decision time: after the first full hour (opening range = 9:30-10:30 ET).
    # Minimum confluence score (0-3) required to fire a signal.
    min_score: int = 2
    ema_fast: int = 9       # in hourly bars
    ema_slow: int = 21
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # ── synthetic option pricing (backtest + premium estimates) ───────────
    risk_free_rate: float = 0.045
    # 0DTE ATM IV ~ VIX level; tune via this multiplier if calibrating.
    iv_multiplier: float = 1.0
    # Round-trip friction: bid/ask slippage as % of premium, plus commissions.
    slippage_pct: float = 4.0
    commission_per_contract: float = 0.65  # per side (moomoo US options approx)

    # ── session times (US/Eastern, decimal hours) ──────────────────────────
    entry_hour: float = 10.5    # 10:30 ET, after opening range completes
    exit_hour: float = 15.75    # force-flat 15:45 ET, before close auction
    close_hour: float = 16.0


config = ZConfig()
