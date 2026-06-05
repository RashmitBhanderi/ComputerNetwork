"""In-context learning memory: load past trades to inject into new prompts.

This is the heart of the learning loop. Before each analysis we pull the most
recent closed trades for the ticker (optionally narrowed to a strategy) and
format them as a compact, human-readable block. Claude reads its own track
record and adjusts — e.g. "your last 5 bull call spreads on TSLA were 1-4, the
losers all entered into high IV rank".
"""
from __future__ import annotations

from typing import Optional

from config import settings
from database.db import Database
from logging_config import get_logger
from models import Trade

log = get_logger(__name__)


class TradeMemory:
    def __init__(self, db: Database) -> None:
        self.db = db

    def recent_trades(
        self, ticker: str, strategy: Optional[str] = None, limit: Optional[int] = None
    ) -> list[Trade]:
        limit = limit or settings.history_lookback
        return self.db.get_recent_trades(ticker, strategy, limit)

    def format_learning_context(self, ticker: str) -> str:
        """Build the 'last N trades with outcomes' block for the prompt."""
        trades = self.recent_trades(ticker, strategy=None, limit=settings.history_lookback)
        if not trades:
            return (
                f"No prior closed trades on {ticker} yet. This is an early "
                "decision — be conservative."
            )

        lines: list[str] = [
            f"Your last {len(trades)} closed trades on {ticker} "
            "(most recent first):"
        ]
        for t in trades:
            outcome = "WIN" if (t.pnl or 0) > 0 else "LOSS"
            pnl = t.pnl if t.pnl is not None else 0.0
            reason = t.close_reason or "n/a"
            lines.append(
                f"  - [{outcome}] {t.strategy} ({t.direction.value}) "
                f"P&L ${pnl:,.0f} | entry ${t.entry_price:.2f} "
                f"exit ${t.exit_price:.2f} | confidence {t.confidence:.0f}% "
                f"| closed via {reason}"
            )
            if t.reasoning:
                snippet = t.reasoning.strip().replace("\n", " ")[:160]
                lines.append(f"      rationale: {snippet}")
        return "\n".join(lines)

    def format_strategy_stats(self) -> str:
        """Win rate and P&L per strategy across all closed trades."""
        stats = self.db.strategy_stats()
        if not stats:
            return "No strategy performance history yet."
        lines = ["Strategy performance to date (all tickers):"]
        for s in stats:
            lines.append(
                f"  - {s['strategy']}: {s['trades']} trades, "
                f"{s['win_rate']:.0f}% win rate, "
                f"${s['total_pnl']:,.0f} total P&L "
                f"(${s['avg_pnl']:,.0f} avg)"
            )
        return "\n".join(lines)
