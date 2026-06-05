"""Score news headlines from -1.0 (very bearish) to +1.0 (very bullish) via Claude.

All headlines for a ticker are scored in a single request using structured
outputs, so we get a validated JSON object back rather than parsing free text.
If Claude is unavailable or the response can't be aligned to the headlines, we
fall back to a neutral 0.0 score so the pipeline keeps moving.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from claude_client import get_async_client, is_available
from config import SENTIMENT_MODEL
from logging_config import get_logger
from models import NewsHeadline

log = get_logger(__name__)


class _ScoredHeadline(BaseModel):
    index: int = Field(description="0-based index of the headline in the input list")
    score: float = Field(ge=-1.0, le=1.0, description="-1 bearish .. +1 bullish")


class _NewsSentimentResult(BaseModel):
    overall: float = Field(ge=-1.0, le=1.0)
    headlines: list[_ScoredHeadline] = Field(default_factory=list)


_SYSTEM = (
    "You are a financial news sentiment analyst. You read stock-market headlines "
    "and rate how bullish or bearish each is for the specific ticker's share "
    "price over the next 1-3 weeks. Use -1.0 for very bearish, 0.0 for neutral, "
    "and +1.0 for very bullish. Be calibrated: most routine headlines are near 0."
)


class NewsSentimentScorer:
    async def score(
        self, ticker: str, headlines: list[NewsHeadline]
    ) -> tuple[list[NewsHeadline], float]:
        """Return headlines with `.sentiment` filled in, plus the average score."""
        if not headlines:
            return headlines, 0.0
        if not is_available():
            log.info("Claude unavailable — neutral news sentiment for %s", ticker)
            return headlines, 0.0

        numbered = "\n".join(f"[{i}] {h.title}" for i, h in enumerate(headlines))
        prompt = (
            f"Ticker: {ticker}\n\n"
            f"Score each of the following {len(headlines)} headlines for {ticker}. "
            "Return a score for every index, plus an overall average.\n\n"
            f"{numbered}"
        )

        try:
            client = get_async_client()
            response = await client.messages.parse(
                model=SENTIMENT_MODEL,
                max_tokens=2000,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_format=_NewsSentimentResult,
            )
            result: Optional[_NewsSentimentResult] = response.parsed_output
        except Exception as exc:  # noqa: BLE001
            log.warning("News sentiment scoring failed for %s: %s", ticker, exc)
            return headlines, 0.0

        if result is None:
            return headlines, 0.0

        # Map scores back onto headlines by index.
        by_index = {s.index: s.score for s in result.headlines}
        for i, h in enumerate(headlines):
            if i in by_index:
                h.sentiment = by_index[i]

        scored = [h.sentiment for h in headlines if h.sentiment is not None]
        average = result.overall if not scored else sum(scored) / len(scored)
        log.info("Scored %d headlines for %s -> %.2f", len(headlines), ticker, average)
        return headlines, round(average, 3)
