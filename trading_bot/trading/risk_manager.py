"""Risk management: position sizing, exposure limits, and volatility halts.

Enforces the account-level guardrails before any paper trade is placed:
  * max 5% of portfolio risked per trade (position sizing),
  * max 3 simultaneous open positions,
  * no trading within 30 minutes of an earnings event,
  * skip when VIX > 35, unless the strategy is explicitly volatility-based.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from config import settings
from logging_config import get_logger

log = get_logger(__name__)

# Strategy names that are intended to harvest/trade volatility and are therefore
# allowed even when VIX is elevated.
_VOLATILITY_STRATEGIES = (
    "iron condor",
    "straddle",
    "strangle",
    "calendar",
    "butterfly",
    "ratio",
)

_EARNINGS_BLACKOUT = timedelta(minutes=30)


@dataclass
class RiskDecision:
    approved: bool
    contracts: int
    reason: str


class RiskManager:
    def __init__(self) -> None:
        self.max_positions = settings.max_open_positions
        self.max_risk_pct = settings.max_risk_per_trade_pct
        self.stop_loss_pct = settings.stop_loss_pct
        self.vix_halt = settings.vix_halt_threshold

    @staticmethod
    def is_volatility_strategy(strategy: str) -> bool:
        s = strategy.lower()
        return any(v in s for v in _VOLATILITY_STRATEGIES)

    def vix_allows(self, vix: Optional[float], strategy: str) -> bool:
        if vix is None or vix <= self.vix_halt:
            return True
        return self.is_volatility_strategy(strategy)

    def within_earnings_blackout(
        self, now: datetime, earnings_at: Optional[datetime]
    ) -> bool:
        """True if `now` is within 30 minutes before/after the earnings time."""
        if earnings_at is None:
            return False
        if earnings_at.tzinfo is None:
            earnings_at = earnings_at.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return abs(now - earnings_at) <= _EARNINGS_BLACKOUT

    def position_size(self, equity: float, premium_per_contract: float) -> int:
        """Number of contracts whose total cost stays within the per-trade cap."""
        if premium_per_contract <= 0:
            return 0
        max_dollars = equity * (self.max_risk_pct / 100.0)
        cost_per_contract = premium_per_contract * 100  # 100 shares per contract
        contracts = math.floor(max_dollars / cost_per_contract)
        return max(0, contracts)

    def stop_premium(self, entry_premium: float) -> float:
        """Stop-loss premium: entry minus stop_loss_pct of the premium."""
        return round(entry_premium * (1 - self.stop_loss_pct / 100.0), 4)

    def evaluate(
        self,
        *,
        strategy: str,
        equity: float,
        open_positions: int,
        premium_per_contract: float,
        vix: Optional[float],
        now: Optional[datetime] = None,
        earnings_at: Optional[datetime] = None,
    ) -> RiskDecision:
        """Run every guardrail and return a sizing decision."""
        now = now or datetime.now(timezone.utc)

        if open_positions >= self.max_positions:
            return RiskDecision(False, 0, f"max {self.max_positions} positions open")

        if not self.vix_allows(vix, strategy):
            return RiskDecision(
                False, 0, f"VIX {vix} > {self.vix_halt} (non-volatility strategy)"
            )

        if self.within_earnings_blackout(now, earnings_at):
            return RiskDecision(False, 0, "within 30-min earnings blackout")

        if premium_per_contract <= 0:
            return RiskDecision(False, 0, "no valid premium to price the trade")

        contracts = self.position_size(equity, premium_per_contract)
        if contracts < 1:
            return RiskDecision(
                False, 0,
                f"premium ${premium_per_contract:.2f} too large for "
                f"{self.max_risk_pct:.0f}% risk cap",
            )

        return RiskDecision(True, contracts, "approved")
