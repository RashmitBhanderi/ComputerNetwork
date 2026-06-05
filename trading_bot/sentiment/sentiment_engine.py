"""Combine every signal into one unified sentiment score and directional bias.

Weights (from config):
    news 30% + IV rank 25% + put/call 20% + VIX 15% + options flow 10%.

Each component is already normalised to -1..+1 (positive = bullish) by the
news scorer and the market aggregator's helpers, so the unified score is a
straight weighted sum that also lands in -1..+1.
"""
from __future__ import annotations

from typing import Optional

from config import settings
from logging_config import get_logger
from models import (
    Direction,
    NewsHeadline,
    OptionsSnapshot,
    SentimentBreakdown,
)
from sentiment.market_sentiment import MarketSentimentAggregator
from sentiment.news_sentiment import NewsSentimentScorer

log = get_logger(__name__)

# Above/below these thresholds the unified score is bullish/bearish.
_BULL_THRESHOLD = 0.15
_BEAR_THRESHOLD = -0.15


class SentimentEngine:
    def __init__(
        self,
        news_scorer: Optional[NewsSentimentScorer] = None,
        market: Optional[MarketSentimentAggregator] = None,
    ) -> None:
        self.news_scorer = news_scorer or NewsSentimentScorer()
        self.market = market

    async def build(
        self,
        ticker: str,
        headlines: list[NewsHeadline],
        options: OptionsSnapshot,
        vix: Optional[float],
        fear_greed: Optional[float],
    ) -> SentimentBreakdown:
        """Score headlines, normalise each signal, and blend into one score."""
        _, news_avg = await self.news_scorer.score(ticker, headlines)

        agg = MarketSentimentAggregator
        news_score = news_avg
        iv_score = agg.iv_rank_to_score(options.iv_rank)
        pc_score = agg.put_call_to_score(options.put_call_ratio)
        vix_score = agg.vix_to_score(vix)
        flow_score = agg.options_flow_score(options)

        unified = (
            settings.weight_news * news_score
            + settings.weight_iv_rank * iv_score
            + settings.weight_put_call * pc_score
            + settings.weight_vix * vix_score
            + settings.weight_options_flow * flow_score
        )
        unified = round(max(-1.0, min(1.0, unified)), 3)

        if unified >= _BULL_THRESHOLD:
            bias = Direction.BULLISH
        elif unified <= _BEAR_THRESHOLD:
            bias = Direction.BEARISH
        else:
            bias = Direction.NEUTRAL

        breakdown = SentimentBreakdown(
            ticker=ticker,
            news_score=round(news_score, 3),
            iv_rank_score=iv_score,
            put_call_score=pc_score,
            vix_score=vix_score,
            options_flow_score=flow_score,
            unified_score=unified,
            bias=bias,
            vix_level=vix,
            fear_greed=fear_greed,
            headline_count=len(headlines),
        )
        log.info(
            "Sentiment %s: unified=%.2f bias=%s (news=%.2f iv=%.2f pc=%.2f "
            "vix=%.2f flow=%.2f)",
            ticker, unified, bias.value, news_score, iv_score, pc_score,
            vix_score, flow_score,
        )
        return breakdown
