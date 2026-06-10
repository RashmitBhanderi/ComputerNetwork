"""Risk management for a $1k-$2k 0DTE account.

Enforced rules:
  * per-trade premium outlay capped at ``max_premium_pct`` of current capital,
  * hard daily halt after losing ``daily_loss_limit_pct`` of start-of-day capital,
  * at most ``max_trades_per_day`` entries,
  * no trading when VIX exceeds ``vix_max``.

Sizing is whole contracts (premium × 100 per contract); 0 contracts means the
premium is too rich for the account — skip rather than oversize.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from zero_dte.config import config


@dataclass
class RiskDecision:
    approved: bool
    contracts: int
    reason: str


@dataclass
class DayState:
    session: Optional[date] = None
    start_capital: float = 0.0
    realized: float = 0.0
    trades: int = 0


@dataclass
class RiskManager:
    capital: float = field(default_factory=lambda: config.starting_capital)
    day: DayState = field(default_factory=DayState)

    def start_session(self, session: date) -> None:
        if self.day.session != session:
            self.day = DayState(session=session, start_capital=self.capital)

    def size(self, premium: float) -> int:
        """Whole contracts whose total premium fits the per-trade cap."""
        if premium <= 0:
            return 0
        budget = self.capital * config.max_premium_pct / 100.0
        return max(0, math.floor(budget / (premium * 100.0)))

    def check_entry(
        self, session: date, premium: float, vix: Optional[float]
    ) -> RiskDecision:
        self.start_session(session)
        if vix is not None and vix > config.vix_max:
            return RiskDecision(False, 0, f"VIX {vix:.1f} > {config.vix_max:.0f} halt")
        if self.day.trades >= config.max_trades_per_day:
            return RiskDecision(False, 0, f"max {config.max_trades_per_day} trades/day")
        loss_limit = self.day.start_capital * config.daily_loss_limit_pct / 100.0
        if self.day.realized <= -loss_limit:
            return RiskDecision(False, 0, "daily loss limit hit")
        contracts = self.size(premium)
        if contracts < 1:
            return RiskDecision(
                False, 0,
                f"premium ${premium:.2f} exceeds "
                f"{config.max_premium_pct:.0f}% per-trade cap",
            )
        return RiskDecision(True, contracts, "approved")

    def record_entry(self) -> None:
        self.day.trades += 1

    def record_pnl(self, pnl: float) -> None:
        self.capital += pnl
        self.day.realized += pnl
