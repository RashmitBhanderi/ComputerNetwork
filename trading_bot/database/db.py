"""SQLite handler for trades, signals, sentiment, and the equity curve.

A thin, dependency-free wrapper around the stdlib `sqlite3` module. All rows are
returned as dicts (via `sqlite3.Row`) so callers can hydrate pydantic models.
The connection uses `check_same_thread=False` because APScheduler jobs and the
Streamlit dashboard may touch the DB from different threads; writes are guarded
by a lock.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from config import DB_PATH
from logging_config import get_logger
from models import Trade, TradeSignal, SentimentBreakdown, TradeStatus

log = get_logger(__name__)


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


class Database:
    def __init__(self, path: Path | str = DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._create_schema()

    # ─── schema ──────────────────────────────────────────────────────────────
    def _create_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    option_symbol TEXT NOT NULL,
                    contracts INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity_multiplier INTEGER NOT NULL DEFAULT 100,
                    pnl REAL,
                    confidence REAL NOT NULL DEFAULT 0,
                    sentiment_score REAL NOT NULL DEFAULT 0,
                    delta REAL, theta REAL, iv REAL,
                    stop_price REAL,
                    expiration TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    reasoning TEXT DEFAULT '',
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    close_reason TEXT
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT,
                    sentiment_score REAL,
                    executed INTEGER NOT NULL DEFAULT 0,
                    raw_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    unified_score REAL NOT NULL,
                    bias TEXT NOT NULL,
                    news_score REAL, iv_rank_score REAL, put_call_score REAL,
                    vix_score REAL, options_flow_score REAL,
                    vix_level REAL, fear_greed REAL, headline_count INTEGER,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS equity_curve (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    open_positions INTEGER NOT NULL,
                    realized_pnl REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
                CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
                CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
                """
            )

    # ─── trades ──────────────────────────────────────────────────────────────
    def insert_trade(self, trade: Trade) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO trades (
                    ticker, strategy, direction, option_symbol, contracts,
                    entry_price, exit_price, quantity_multiplier, pnl, confidence,
                    sentiment_score, delta, theta, iv, stop_price, expiration,
                    status, reasoning, opened_at, closed_at, close_reason
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    trade.ticker, trade.strategy, trade.direction.value,
                    trade.option_symbol, trade.contracts, trade.entry_price,
                    trade.exit_price, trade.quantity_multiplier, trade.pnl,
                    trade.confidence, trade.sentiment_score, trade.delta,
                    trade.theta, trade.iv, trade.stop_price, trade.expiration,
                    trade.status.value, trade.reasoning, _iso(trade.opened_at),
                    _iso(trade.closed_at), trade.close_reason,
                ),
            )
            trade_id = int(cur.lastrowid)
            log.info("Recorded trade #%s %s %s", trade_id, trade.ticker, trade.strategy)
            return trade_id

    def close_trade(
        self, trade_id: int, exit_price: float, pnl: float, reason: str
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE trades
                   SET status='closed', exit_price=?, pnl=?, closed_at=?,
                       close_reason=?
                 WHERE id=?
                """,
                (exit_price, pnl, _iso(datetime.now(timezone.utc)), reason, trade_id),
            )
        log.info("Closed trade #%s pnl=%.2f (%s)", trade_id, pnl, reason)

    def get_open_trades(self) -> list[Trade]:
        rows = self._conn.execute(
            "SELECT * FROM trades WHERE status='open' ORDER BY opened_at"
        ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def get_trades(self, limit: int = 200) -> list[Trade]:
        rows = self._conn.execute(
            "SELECT * FROM trades ORDER BY opened_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def get_recent_trades(
        self, ticker: str, strategy: Optional[str] = None, limit: int = 10
    ) -> list[Trade]:
        """Closed trades for the learning loop, newest first."""
        if strategy:
            rows = self._conn.execute(
                """
                SELECT * FROM trades
                 WHERE ticker=? AND strategy=? AND status='closed'
                 ORDER BY closed_at DESC LIMIT ?
                """,
                (ticker, strategy, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM trades
                 WHERE ticker=? AND status='closed'
                 ORDER BY closed_at DESC LIMIT ?
                """,
                (ticker, limit),
            ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def count_open_positions(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM trades WHERE status='open'"
        ).fetchone()
        return int(row["n"])

    def has_open_position(self, ticker: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM trades WHERE ticker=? AND status='open' LIMIT 1", (ticker,)
        ).fetchone()
        return row is not None

    # ─── signals ─────────────────────────────────────────────────────────────
    def insert_signal(self, signal: TradeSignal, executed: bool) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO signals (
                    ticker, direction, strategy, confidence, reasoning,
                    sentiment_score, executed, raw_json, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    signal.ticker, signal.direction.value, signal.strategy,
                    signal.confidence, signal.reasoning, signal.sentiment_score,
                    int(executed), signal.model_dump_json(),
                    _iso(datetime.now(timezone.utc)),
                ),
            )
            return int(cur.lastrowid)

    def get_latest_signal(self, ticker: Optional[str] = None) -> Optional[dict[str, Any]]:
        if ticker:
            row = self._conn.execute(
                "SELECT * FROM signals WHERE ticker=? ORDER BY created_at DESC LIMIT 1",
                (ticker,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM signals ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    # ─── sentiment ───────────────────────────────────────────────────────────
    def insert_sentiment(self, s: SentimentBreakdown) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO sentiment (
                    ticker, unified_score, bias, news_score, iv_rank_score,
                    put_call_score, vix_score, options_flow_score, vix_level,
                    fear_greed, headline_count, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    s.ticker, s.unified_score, s.bias.value, s.news_score,
                    s.iv_rank_score, s.put_call_score, s.vix_score,
                    s.options_flow_score, s.vix_level, s.fear_greed,
                    s.headline_count, _iso(s.as_of),
                ),
            )
            return int(cur.lastrowid)

    def get_latest_sentiment(self, ticker: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM sentiment WHERE ticker=? ORDER BY created_at DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        return dict(row) if row else None

    # ─── equity curve ────────────────────────────────────────────────────────
    def record_equity(
        self, equity: float, cash: float, open_positions: int, realized_pnl: float
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO equity_curve (
                    equity, cash, open_positions, realized_pnl, created_at
                ) VALUES (?,?,?,?,?)
                """,
                (equity, cash, open_positions, realized_pnl,
                 _iso(datetime.now(timezone.utc))),
            )

    def get_equity_curve(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM equity_curve ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]

    # ─── analytics ───────────────────────────────────────────────────────────
    def strategy_stats(self) -> list[dict[str, Any]]:
        """Win rate and P&L aggregated per strategy (closed trades only)."""
        rows = self._conn.execute(
            """
            SELECT strategy,
                   COUNT(*)                                   AS trades,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)   AS wins,
                   COALESCE(SUM(pnl), 0)                       AS total_pnl,
                   COALESCE(AVG(pnl), 0)                       AS avg_pnl
              FROM trades
             WHERE status='closed'
             GROUP BY strategy
             ORDER BY total_pnl DESC
            """
        ).fetchall()
        stats: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["win_rate"] = (d["wins"] / d["trades"] * 100) if d["trades"] else 0.0
            stats.append(d)
        return stats

    def realized_pnl_total(self) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) AS total FROM trades WHERE status='closed'"
        ).fetchone()
        return float(row["total"])

    # ─── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _row_to_trade(row: sqlite3.Row) -> Trade:
        data = dict(row)
        data["status"] = TradeStatus(data["status"])
        return Trade.model_validate(data)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _demo(rows: Iterable[dict[str, Any]]) -> str:  # pragma: no cover - utility
    return json.dumps(list(rows), indent=2, default=str)
