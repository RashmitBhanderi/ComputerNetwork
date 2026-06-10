"""Entry point + scheduler for the AI options paper trading bot.

Wires together the data pipeline, sentiment engine, Claude analyst, risk
manager, and paper-trading engine, and drives them on an APScheduler cron
schedule (US/Eastern):

    08:30  pre-market   — fetch news, build sentiment scores
    09:45  market scan  — analyse watchlist, place paper trades
    12:30  midday check — mark positions, take profits / stops
    15:45  end of day   — close expiring positions, log daily summary
    17:00  post-market  — generate daily performance report

Run continuously:   python main.py
Run a single full cycle then exit (handy for testing):   python main.py --once
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from typing import Optional

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from ai.analyst import Analyst
from ai.memory import TradeMemory
from config import settings
from data.market_data import MarketDataClient
from data.news_data import NewsDataClient
from data.options_data import OptionsDataClient
from database.db import Database
from logging_config import get_logger
from models import OptionsSnapshot, SentimentBreakdown
from sentiment.market_sentiment import MarketSentimentAggregator
from sentiment.sentiment_engine import SentimentEngine
from trading.paper_trader import PaperTrader
from trading.risk_manager import RiskManager
from trading.strategy import StrategyEngine

log = get_logger("trading_bot.main")

EASTERN = pytz.timezone("US/Eastern")


class TradingBot:
    def __init__(self) -> None:
        self.db = Database()
        self.risk = RiskManager()
        self.paper_trader = PaperTrader(self.db, self.risk)
        self.memory = TradeMemory(self.db)
        self.analyst = Analyst(self.memory)
        self.strategy = StrategyEngine()
        self.sentiment_engine = SentimentEngine()

    # ─── shared helpers ──────────────────────────────────────────────────────
    async def _market_context(
        self, market: MarketDataClient, agg: MarketSentimentAggregator
    ) -> tuple[Optional[float], Optional[float]]:
        vix = await market.get_vix()
        fear_greed = await agg.get_fear_greed()
        log.info("Market context: VIX=%s Fear&Greed=%s", vix, fear_greed)
        return vix, fear_greed

    async def _build_sentiment(
        self,
        ticker: str,
        market: MarketDataClient,
        options_client: OptionsDataClient,
        news: NewsDataClient,
        vix: Optional[float],
        fear_greed: Optional[float],
    ) -> tuple[SentimentBreakdown, OptionsSnapshot, Optional[float]]:
        price = await market.get_price(ticker)
        underlying = price.price if price else None
        options = await options_client.get_snapshot(ticker, underlying)
        headlines = await news.get_headlines(ticker)
        breakdown = await self.sentiment_engine.build(
            ticker, headlines, options, vix, fear_greed
        )
        self.db.insert_sentiment(breakdown)
        return breakdown, options, underlying

    # ─── scheduled jobs ──────────────────────────────────────────────────────
    async def run_premarket(self) -> None:
        log.info("=== PRE-MARKET: building sentiment for watchlist ===")
        async with MarketDataClient() as market, OptionsDataClient() as opts, \
                NewsDataClient() as news, MarketSentimentAggregator() as agg:
            vix, fear_greed = await self._market_context(market, agg)
            for ticker in settings.watchlist:
                try:
                    await self._build_sentiment(
                        ticker, market, opts, news, vix, fear_greed
                    )
                except Exception as exc:  # noqa: BLE001
                    log.exception("Pre-market failed for %s: %s", ticker, exc)
        log.info("=== PRE-MARKET complete ===")

    async def run_market_scan(self) -> None:
        log.info("=== MARKET SCAN: analysing watchlist ===")
        async with MarketDataClient() as market, OptionsDataClient() as opts, \
                NewsDataClient() as news, MarketSentimentAggregator() as agg:
            vix, fear_greed = await self._market_context(market, agg)

            for ticker in settings.watchlist:
                try:
                    await self._scan_ticker(
                        ticker, market, opts, news, vix, fear_greed
                    )
                except Exception as exc:  # noqa: BLE001
                    log.exception("Scan failed for %s: %s", ticker, exc)

            # Mark to market after any new entries.
            await self._mark_and_record(opts)
        log.info("=== MARKET SCAN complete ===")

    async def _scan_ticker(
        self,
        ticker: str,
        market: MarketDataClient,
        opts: OptionsDataClient,
        news: NewsDataClient,
        vix: Optional[float],
        fear_greed: Optional[float],
    ) -> None:
        sentiment, options, _ = await self._build_sentiment(
            ticker, market, opts, news, vix, fear_greed
        )
        price = await market.get_price(ticker)
        signal = await self.analyst.analyze(ticker, price, options, sentiment)

        enter, gate_reason = self.strategy.should_enter(signal)
        executed = False

        if not enter:
            log.info("Skip %s: %s", ticker, gate_reason)
        elif self.db.has_open_position(ticker):
            log.info("Skip %s: already holding an open position", ticker)
        else:
            executed = await self._attempt_trade(signal, options, vix)

        self.db.insert_signal(signal, executed)

    async def _attempt_trade(
        self, signal, options: OptionsSnapshot, vix: Optional[float]
    ) -> bool:
        plan = self.strategy.build_order(signal, options)
        if plan is None:
            log.info("No priceable order for %s", signal.ticker)
            return False

        equity = self.paper_trader.equity({})
        decision = self.risk.evaluate(
            strategy=plan.strategy,
            equity=equity,
            open_positions=self.db.count_open_positions(),
            premium_per_contract=plan.entry_premium,
            vix=vix,
        )
        if not decision.approved:
            log.info("Risk blocked %s: %s", signal.ticker, decision.reason)
            return False

        trade = self.paper_trader.open_trade(signal, plan, decision.contracts)
        return trade is not None

    async def run_midday_check(self) -> None:
        log.info("=== MIDDAY CHECK: reviewing open positions ===")
        async with OptionsDataClient() as opts:
            closed = await self._evaluate_positions(opts, close_expiring=False)
            await self._mark_and_record(opts)
        log.info("=== MIDDAY CHECK complete (%d closed) ===", len(closed))

    async def run_eod(self) -> None:
        log.info("=== END OF DAY: closing expiring positions ===")
        async with OptionsDataClient() as opts:
            closed = await self._evaluate_positions(opts, close_expiring=True)
            equity = await self._mark_and_record(opts)
        log.info(
            "=== END OF DAY complete: %d closed, equity $%s ===",
            len(closed), f"{equity:,.2f}",
        )

    async def run_postmarket(self) -> None:
        log.info("=== POST-MARKET: daily performance report ===")
        report = self.daily_report()
        for line in report.splitlines():
            log.info(line)
        log.info("=== POST-MARKET complete ===")

    # ─── position maintenance ────────────────────────────────────────────────
    async def _open_position_premiums(
        self, opts: OptionsDataClient
    ) -> dict[str, Optional[float]]:
        symbols = [t.option_symbol for t in self.db.get_open_trades()]
        if not symbols:
            return {}
        quotes = await opts.get_option_quotes(symbols)
        return {s: quotes.get(s) for s in symbols}

    async def _evaluate_positions(
        self, opts: OptionsDataClient, close_expiring: bool
    ) -> list:
        premiums = await self._open_position_premiums(opts)
        return self.paper_trader.evaluate_open_positions(
            premiums, close_expiring=close_expiring
        )

    async def _mark_and_record(self, opts: OptionsDataClient) -> float:
        premiums = await self._open_position_premiums(opts)
        return self.paper_trader.snapshot_equity(premiums)

    # ─── reporting ───────────────────────────────────────────────────────────
    def daily_report(self) -> str:
        equity_curve = self.db.get_equity_curve()
        equity = equity_curve[-1]["equity"] if equity_curve else settings.starting_capital
        realized = self.db.realized_pnl_total()
        open_trades = self.db.get_open_trades()
        stats = self.db.strategy_stats()

        lines = [
            f"Daily report — {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
            f"  Equity:        ${equity:,.2f}",
            f"  Cash:          ${self.paper_trader.cash:,.2f}",
            f"  Realized P&L:  ${realized:,.2f}",
            f"  Open positions: {len(open_trades)}",
        ]
        for t in open_trades:
            lines.append(
                f"    - {t.ticker} {t.strategy} x{t.contracts} "
                f"@ ${t.entry_price:.2f} ({t.confidence:.0f}% conf)"
            )
        if stats:
            lines.append("  Strategy win rates:")
            for s in stats:
                lines.append(
                    f"    - {s['strategy']}: {s['win_rate']:.0f}% over "
                    f"{s['trades']} trades, ${s['total_pnl']:,.0f} P&L"
                )
        return "\n".join(lines)

    async def run_full_cycle(self) -> None:
        """Run every stage once, in order (used by --once)."""
        await self.run_premarket()
        await self.run_market_scan()
        await self.run_midday_check()
        await self.run_eod()
        await self.run_postmarket()


def build_scheduler(bot: TradingBot) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=EASTERN)
    jobs = [
        ("premarket", bot.run_premarket, 8, 30),
        ("market_scan", bot.run_market_scan, 9, 45),
        ("midday_check", bot.run_midday_check, 12, 30),
        ("eod", bot.run_eod, 15, 45),
        ("postmarket", bot.run_postmarket, 17, 0),
    ]
    for job_id, func, hour, minute in jobs:
        scheduler.add_job(
            func,
            CronTrigger(day_of_week="mon-fri", hour=hour, minute=minute,
                        timezone=EASTERN),
            id=job_id,
            misfire_grace_time=600,
            max_instances=1,
        )
    return scheduler


def _check_keys() -> None:
    missing = settings.missing_keys()
    if missing:
        log.warning(
            "Missing API keys: %s. The bot will run but those data sources / "
            "Claude analysis will be skipped (neutral fallbacks).",
            ", ".join(missing),
        )
    else:
        log.info("All required API keys configured.")


async def _run_forever(bot: TradingBot) -> None:
    scheduler = build_scheduler(bot)
    scheduler.start()
    log.info("Scheduler started. Watchlist: %s", ", ".join(settings.watchlist))
    log.info("Jobs (US/Eastern): premarket 08:30, scan 09:45, midday 12:30, "
             "eod 15:45, postmarket 17:00")
    try:
        # Keep the event loop alive indefinitely.
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):  # pragma: no cover
        log.info("Shutting down scheduler...")
    finally:
        scheduler.shutdown(wait=False)
        bot.db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="AI options paper trading bot")
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single full cycle (premarket -> postmarket) and exit.",
    )
    args = parser.parse_args()

    _check_keys()
    bot = TradingBot()

    if args.once:
        log.info("Running a single full cycle (--once)...")
        asyncio.run(bot.run_full_cycle())
        bot.db.close()
    else:
        try:
            asyncio.run(_run_forever(bot))
        except KeyboardInterrupt:  # pragma: no cover
            log.info("Interrupted.")


if __name__ == "__main__":
    main()
