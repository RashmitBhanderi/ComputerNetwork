"""Send the assembled market context to Claude and parse a structured decision.

Uses the most capable model (Opus) with adaptive thinking and high effort,
because this is the correctness-sensitive step. Structured outputs guarantee a
schema-valid response that we convert into a `TradeSignal`. On any failure we
return a neutral, zero-confidence signal so the strategy layer simply skips the
ticker.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ai.memory import TradeMemory
from ai.prompt_builder import PromptBuilder, SYSTEM_PROMPT
from claude_client import get_async_client, is_available
from config import ANALYSIS_MODEL
from logging_config import get_logger
from models import (
    Direction,
    OptionsSnapshot,
    PriceSnapshot,
    SentimentBreakdown,
    TradeSignal,
)

log = get_logger(__name__)


class _AnalysisResponse(BaseModel):
    """Schema Claude must fill — kept free of defaults for clean JSON schema."""

    direction: str = Field(description="bullish, bearish, or neutral")
    strategy: str = Field(description="e.g. 'bull call spread', 'buy put', 'iron condor'")
    confidence: float = Field(ge=0, le=100, description="0-100 conviction")
    reasoning: str = Field(description="2-4 sentence rationale")
    entry_level: Optional[float] = Field(default=None, description="underlying entry")
    exit_level: Optional[float] = Field(default=None, description="underlying target")
    stop_level: Optional[float] = Field(default=None, description="underlying stop")
    target_contract: Optional[str] = Field(
        default=None, description="specific option symbol if applicable"
    )


class Analyst:
    def __init__(self, memory: TradeMemory) -> None:
        self.memory = memory
        self.prompt_builder = PromptBuilder(memory)

    async def analyze(
        self,
        ticker: str,
        price: Optional[PriceSnapshot],
        options: OptionsSnapshot,
        sentiment: SentimentBreakdown,
    ) -> TradeSignal:
        prompt = self.prompt_builder.build(ticker, price, options, sentiment)

        if not is_available():
            log.info("Claude unavailable — neutral signal for %s", ticker)
            return self._neutral(ticker, sentiment, "Claude API key not configured")

        try:
            client = get_async_client()
            response = await client.messages.parse(
                model=ANALYSIS_MODEL,
                max_tokens=4000,
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_format=_AnalysisResponse,
            )
            parsed: Optional[_AnalysisResponse] = response.parsed_output
        except Exception as exc:  # noqa: BLE001
            log.warning("Analysis failed for %s: %s", ticker, exc)
            return self._neutral(ticker, sentiment, f"analysis error: {exc}")

        if parsed is None:
            return self._neutral(ticker, sentiment, "empty analysis response")

        try:
            signal = TradeSignal(
                ticker=ticker,
                direction=parsed.direction,
                strategy=parsed.strategy,
                confidence=parsed.confidence,
                reasoning=parsed.reasoning,
                entry_level=parsed.entry_level,
                exit_level=parsed.exit_level,
                stop_level=parsed.stop_level,
                target_contract=parsed.target_contract,
                sentiment_score=sentiment.unified_score,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Signal validation failed for %s: %s", ticker, exc)
            return self._neutral(ticker, sentiment, f"signal validation error: {exc}")

        log.info(
            "Analysis %s: %s %s @ %.0f%% confidence",
            ticker, signal.direction.value, signal.strategy, signal.confidence,
        )
        return signal

    @staticmethod
    def _neutral(
        ticker: str, sentiment: SentimentBreakdown, reason: str
    ) -> TradeSignal:
        return TradeSignal(
            ticker=ticker,
            direction=Direction.NEUTRAL,
            strategy="no trade",
            confidence=0.0,
            reasoning=reason,
            sentiment_score=sentiment.unified_score,
        )
