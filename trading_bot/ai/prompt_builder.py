"""Construct the analysis prompt sent to Claude.

The prompt assembles five blocks the model needs to make (and learn from) a
decision: price action, the options-chain snapshot, the unified sentiment
breakdown, the recent trade history for this ticker, and strategy-level
performance stats. The system prompt pins the model to a disciplined options
trader persona and to the JSON contract the analyst expects back.
"""
from __future__ import annotations

from typing import Optional

from ai.memory import TradeMemory
from config import settings
from models import OptionsSnapshot, PriceSnapshot, SentimentBreakdown

SYSTEM_PROMPT = (
    "You are a disciplined short-term options trader running a paper-trading "
    "account. You specialise in 0-21 DTE options on liquid large-cap names and "
    "index ETFs. You weigh sentiment, implied volatility, and options flow, and "
    "you learn from your own track record provided in each prompt.\n\n"
    "Rules you must follow:\n"
    f"- Only recommend a trade with confidence >= {settings.min_confidence}% "
    "when the setup is genuinely strong; otherwise return 'neutral' with low "
    "confidence and explain why you are standing aside.\n"
    "- Prefer defined-risk structures (spreads) when IV rank is elevated; favour "
    "long single options only when IV rank is low and conviction is high.\n"
    "- Be concrete: name a specific strategy (e.g. 'bull call spread', "
    "'buy put', 'iron condor', 'bear put spread'), entry and exit levels on the "
    "underlying, and a stop level.\n"
    "- Keep reasoning to 2-4 sentences. Reference the sentiment breakdown and "
    "your trade history explicitly when relevant.\n"
)


class PromptBuilder:
    def __init__(self, memory: TradeMemory) -> None:
        self.memory = memory

    def _price_block(self, price: Optional[PriceSnapshot]) -> str:
        if price is None:
            return "Price action: unavailable."
        parts = [f"Price action for {price.ticker}:", f"  - Last: ${price.price:.2f}"]
        if price.change_pct is not None:
            parts.append(f"  - Day change: {price.change_pct:+.2f}%")
        if price.day_high is not None and price.day_low is not None:
            parts.append(f"  - Day range: ${price.day_low:.2f} - ${price.day_high:.2f}")
        if price.prev_close is not None:
            parts.append(f"  - Prev close: ${price.prev_close:.2f}")
        if price.volume is not None:
            parts.append(f"  - Volume: {price.volume:,.0f}")
        return "\n".join(parts)

    def _options_block(self, options: OptionsSnapshot) -> str:
        lines = [
            f"Options snapshot (expiry {options.expiration or 'n/a'}):",
            f"  - Underlying: ${options.underlying_price:.2f}"
            if options.underlying_price else "  - Underlying: n/a",
            f"  - ATM strike: {options.atm_strike}",
            f"  - IV rank (est.): {options.iv_rank}",
            f"  - Put/Call ratio: {options.put_call_ratio}",
        ]
        if options.contracts:
            lines.append("  - ATM-region contracts (strike/type: delta, theta, IV, mid):")
            for c in sorted(options.contracts, key=lambda x: (x.strike, x.option_type)):
                lines.append(
                    f"      {c.strike:.1f} {c.option_type}: "
                    f"delta={c.delta if c.delta is not None else 'n/a'}, "
                    f"theta={c.theta if c.theta is not None else 'n/a'}, "
                    f"iv={c.iv if c.iv is not None else 'n/a'}, "
                    f"mid={c.mid if c.mid is not None else 'n/a'} "
                    f"[{c.symbol}]"
                )
        return "\n".join(lines)

    def _sentiment_block(self, s: SentimentBreakdown) -> str:
        return (
            f"Unified sentiment for {s.ticker}: {s.unified_score:+.2f} "
            f"(bias: {s.bias.value})\n"
            f"  - News: {s.news_score:+.2f} (weight {settings.weight_news:.0%}, "
            f"{s.headline_count} headlines)\n"
            f"  - IV rank: {s.iv_rank_score:+.2f} (weight {settings.weight_iv_rank:.0%})\n"
            f"  - Put/Call: {s.put_call_score:+.2f} (weight {settings.weight_put_call:.0%})\n"
            f"  - VIX: {s.vix_score:+.2f} (weight {settings.weight_vix:.0%}, "
            f"level {s.vix_level})\n"
            f"  - Options flow: {s.options_flow_score:+.2f} "
            f"(weight {settings.weight_options_flow:.0%})\n"
            f"  - Fear & Greed: {s.fear_greed}"
        )

    def build(
        self,
        ticker: str,
        price: Optional[PriceSnapshot],
        options: OptionsSnapshot,
        sentiment: SentimentBreakdown,
    ) -> str:
        """Assemble the full user prompt for one ticker."""
        return "\n\n".join(
            [
                f"## Analysis request: {ticker}",
                self._price_block(price),
                self._options_block(options),
                self._sentiment_block(sentiment),
                self.memory.format_learning_context(ticker),
                self.memory.format_strategy_stats(),
                (
                    "Decide whether to open a short-dated options paper trade now. "
                    "Return your decision as structured output with: direction "
                    "(bullish/bearish/neutral), strategy, confidence (0-100), "
                    "reasoning, and entry/exit/stop levels on the underlying. If "
                    "you reference a specific contract, put its option symbol in "
                    "target_contract."
                ),
            ]
        )
