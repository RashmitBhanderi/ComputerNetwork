"""Paper-trading execution engine: simulates fills, tracks positions and P&L.

Starts with $100,000 of virtual capital. Cash is reconstructed from the
database on startup (starting capital + realized P&L - open cost basis) so the
account survives restarts. Positions are marked to market from re-quoted option
premiums; the engine auto-closes on a -40% stop, a +60% take-profit, or expiry.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from config import settings
from database.db import Database
from logging_config import get_logger
from models import Trade, TradeSignal, TradeStatus
from trading.risk_manager import RiskManager
from trading.strategy import OrderPlan

log = get_logger(__name__)

# Profit target as a percentage gain on the entry premium (midday/EOD checks).
TAKE_PROFIT_PCT = 60.0


class PaperTrader:
    def __init__(self, db: Database, risk: RiskManager) -> None:
        self.db = db
        self.risk = risk
        self.cash = self._recompute_cash()

    def _recompute_cash(self) -> float:
        realized = self.db.realized_pnl_total()
        open_cost = sum(t.cost_basis for t in self.db.get_open_trades())
        cash = settings.starting_capital + realized - open_cost
        log.info("Cash reconstructed: $%s", f"{cash:,.2f}")
        return cash

    # ─── opening ─────────────────────────────────────────────────────────────
    def open_trade(
        self, signal: TradeSignal, plan: OrderPlan, contracts: int
    ) -> Optional[Trade]:
        cost = plan.entry_premium * contracts * 100
        if cost > self.cash:
            log.warning(
                "Insufficient cash for %s: need $%s have $%s",
                plan.ticker, f"{cost:,.0f}", f"{self.cash:,.0f}",
            )
            return None

        trade = Trade(
            ticker=plan.ticker,
            strategy=plan.strategy,
            direction=plan.direction,
            option_symbol=plan.option_symbol,
            contracts=contracts,
            entry_price=plan.entry_premium,
            confidence=signal.confidence,
            sentiment_score=signal.sentiment_score,
            delta=plan.delta,
            theta=plan.theta,
            iv=plan.iv,
            stop_price=self.risk.stop_premium(plan.entry_premium),
            expiration=plan.expiration,
            reasoning=signal.reasoning,
            status=TradeStatus.OPEN,
        )
        trade.id = self.db.insert_trade(trade)
        self.cash -= cost
        log.info(
            "OPENED %s %s x%d @ $%.2f (cost $%s, cash $%s)",
            plan.ticker, plan.strategy, contracts, plan.entry_premium,
            f"{cost:,.0f}", f"{self.cash:,.0f}",
        )
        return trade

    # ─── closing ─────────────────────────────────────────────────────────────
    def close_trade(self, trade: Trade, current_premium: float, reason: str) -> float:
        proceeds = max(0.0, current_premium) * trade.contracts * trade.quantity_multiplier
        pnl = proceeds - trade.cost_basis
        self.cash += proceeds
        if trade.id is not None:
            self.db.close_trade(trade.id, round(current_premium, 4), round(pnl, 2), reason)
        log.info(
            "CLOSED %s %s pnl $%s (%s), cash $%s",
            trade.ticker, trade.strategy, f"{pnl:,.2f}", reason,
            f"{self.cash:,.0f}",
        )
        return pnl

    # ─── monitoring ──────────────────────────────────────────────────────────
    def evaluate_open_positions(
        self,
        premiums: dict[str, Optional[float]],
        today: Optional[date] = None,
        close_expiring: bool = True,
    ) -> list[tuple[Trade, float, str]]:
        """Mark positions to market and auto-close on stop/target/expiry.

        `premiums` maps option_symbol -> current premium (None if unknown).
        Returns a list of (trade, pnl, reason) for everything closed.
        """
        today = today or datetime.now(timezone.utc).date()
        closed: list[tuple[Trade, float, str]] = []

        for trade in self.db.get_open_trades():
            premium = premiums.get(trade.option_symbol)
            expired = self._is_expired(trade, today)

            # Expiry: settle at last known premium (intrinsic ~ market premium).
            if close_expiring and expired:
                settle = premium if premium is not None else 0.0
                pnl = self.close_trade(trade, settle, "expiry")
                closed.append((trade, pnl, "expiry"))
                continue

            if premium is None:
                continue

            # Stop-loss: premium fell to/below the -40% stop level.
            if trade.stop_price is not None and premium <= trade.stop_price:
                pnl = self.close_trade(trade, premium, "stop_loss")
                closed.append((trade, pnl, "stop_loss"))
                continue

            # Take-profit: premium gained TAKE_PROFIT_PCT over entry.
            target = trade.entry_price * (1 + TAKE_PROFIT_PCT / 100.0)
            if premium >= target:
                pnl = self.close_trade(trade, premium, "target_hit")
                closed.append((trade, pnl, "target_hit"))

        return closed

    @staticmethod
    def _is_expired(trade: Trade, today: date) -> bool:
        if not trade.expiration:
            return False
        try:
            exp = date.fromisoformat(trade.expiration)
        except ValueError:
            return False
        return exp <= today

    # ─── accounting ──────────────────────────────────────────────────────────
    def equity(self, premiums: dict[str, Optional[float]]) -> float:
        """Total account equity = cash + marked-to-market open positions."""
        market_value = 0.0
        for trade in self.db.get_open_trades():
            premium = premiums.get(trade.option_symbol)
            px = premium if premium is not None else trade.entry_price
            market_value += trade.market_value(px)
        return round(self.cash + market_value, 2)

    def snapshot_equity(self, premiums: dict[str, Optional[float]]) -> float:
        equity = self.equity(premiums)
        self.db.record_equity(
            equity=equity,
            cash=round(self.cash, 2),
            open_positions=self.db.count_open_positions(),
            realized_pnl=round(self.db.realized_pnl_total(), 2),
        )
        return equity
