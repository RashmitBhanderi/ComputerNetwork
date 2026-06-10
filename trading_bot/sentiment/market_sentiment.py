"""Market-wide sentiment signals: VIX, CNN Fear & Greed, and options flow.

This module also owns the normalisation helpers that map each raw signal onto a
common -1..+1 scale (positive = bullish) so the engine can blend them. The CNN
Fear & Greed index is fetched from its public dataviz endpoint; if that fails we
simply omit the component.
"""
from __future__ import annotations

from typing import Optional

import httpx

from data.http_client import request_json
from logging_config import get_logger
from models import OptionsSnapshot

log = get_logger(__name__)

_CNN_FNG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


class MarketSentimentAggregator:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient()

    async def __aenter__(self) -> "MarketSentimentAggregator":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def get_fear_greed(self) -> Optional[float]:
        """CNN Fear & Greed Index (0 = extreme fear, 100 = extreme greed)."""
        try:
            data = await request_json(
                self._client, "GET", _CNN_FNG_URL, headers=_BROWSER_HEADERS
            )
        except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001
            log.warning("Fear & Greed fetch failed: %s", exc)
            return None
        score = (data.get("fear_and_greed") or {}).get("score")
        return float(score) if score is not None else None

    # ─── normalisation helpers (raw signal -> -1..+1, positive = bullish) ────
    @staticmethod
    def vix_to_score(vix: Optional[float]) -> float:
        """Low VIX (~12) is bullish (+), high VIX (~40) is bearish (-)."""
        if vix is None:
            return 0.0
        # Map 12 -> +1, 26 -> 0, 40 -> -1, clamped.
        score = (26.0 - vix) / 14.0
        return round(max(-1.0, min(1.0, score)), 3)

    @staticmethod
    def fear_greed_to_score(fg: Optional[float]) -> float:
        """0..100 greed index -> -1..+1 (greed is bullish)."""
        if fg is None:
            return 0.0
        return round((fg - 50.0) / 50.0, 3)

    @staticmethod
    def iv_rank_to_score(iv_rank: Optional[float]) -> float:
        """High IV rank means rich premium / elevated fear -> mildly bearish.

        Returned in -1..+1: IV rank 0 -> +1, 50 -> 0, 100 -> -1.
        """
        if iv_rank is None:
            return 0.0
        return round((50.0 - iv_rank) / 50.0, 3)

    @staticmethod
    def put_call_to_score(put_call: Optional[float]) -> float:
        """Put/call ratio -> -1..+1. <0.7 bullish, ~1.0 neutral, >1.0 bearish."""
        if put_call is None:
            return 0.0
        # Centre on 1.0; a ratio of 0.7 -> +0.6, 1.3 -> -0.6 (clamped).
        score = (1.0 - put_call) / 0.5
        return round(max(-1.0, min(1.0, score)), 3)

    @staticmethod
    def options_flow_score(snapshot: OptionsSnapshot) -> float:
        """Delta-weighted call-vs-put volume skew from the focused chain."""
        call_flow = 0.0
        put_flow = 0.0
        for c in snapshot.contracts:
            vol = c.volume or 0
            weight = abs(c.delta) if c.delta is not None else 0.5
            if c.option_type == "call":
                call_flow += vol * weight
            elif c.option_type == "put":
                put_flow += vol * weight
        total = call_flow + put_flow
        if total == 0:
            return 0.0
        score = (call_flow - put_flow) / total
        return round(max(-1.0, min(1.0, score)), 3)
