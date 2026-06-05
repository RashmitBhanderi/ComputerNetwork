"""News fetcher: NewsAPI (primary) + Benzinga (optional secondary).

Returns up to ``settings.headlines_per_ticker`` recent headlines per ticker as
``NewsHeadline`` objects (sentiment is filled in later by the sentiment module).
Both sources degrade gracefully when their key is missing or a call fails.
"""
from __future__ import annotations

import httpx

from config import settings
from data.http_client import request_json
from logging_config import get_logger
from models import NewsHeadline

log = get_logger(__name__)

# Map tickers to company keywords so NewsAPI returns relevant business news.
_TICKER_QUERY = {
    "SPY": "S&P 500 OR SPY ETF",
    "QQQ": "Nasdaq 100 OR QQQ ETF",
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "AMZN": "Amazon",
}


class NewsDataClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient()

    async def __aenter__(self) -> "NewsDataClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def get_headlines(self, ticker: str) -> list[NewsHeadline]:
        """Combine NewsAPI + Benzinga, de-duplicated, newest first."""
        headlines: list[NewsHeadline] = []
        headlines.extend(await self._newsapi(ticker))
        headlines.extend(await self._benzinga(ticker))

        seen: set[str] = set()
        unique: list[NewsHeadline] = []
        for h in headlines:
            key = h.title.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(h)
        return unique[: settings.headlines_per_ticker]

    async def _newsapi(self, ticker: str) -> list[NewsHeadline]:
        if not settings.news_api_key:
            return []
        query = _TICKER_QUERY.get(ticker, ticker)
        try:
            data = await request_json(
                self._client, "GET", "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": str(settings.headlines_per_ticker),
                    "apiKey": settings.news_api_key,
                },
            )
        except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001
            log.warning("NewsAPI failed for %s: %s", ticker, exc)
            return []

        out: list[NewsHeadline] = []
        for article in data.get("articles", []):
            title = (article.get("title") or "").strip()
            if not title:
                continue
            out.append(
                NewsHeadline(
                    title=title,
                    source=(article.get("source") or {}).get("name", "NewsAPI"),
                    url=article.get("url", ""),
                    published_at=article.get("publishedAt"),
                )
            )
        return out

    async def _benzinga(self, ticker: str) -> list[NewsHeadline]:
        if not settings.benzinga_api_key:
            return []
        try:
            data = await request_json(
                self._client, "GET", "https://api.benzinga.com/api/v2/news",
                params={
                    "token": settings.benzinga_api_key,
                    "tickers": ticker,
                    "pageSize": str(settings.headlines_per_ticker),
                    "displayOutput": "abstract",
                },
                headers={"Accept": "application/json"},
            )
        except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001
            log.warning("Benzinga failed for %s: %s", ticker, exc)
            return []

        # Benzinga may return a bare list or a wrapped object.
        items = data if isinstance(data, list) else data.get("news", [])
        out: list[NewsHeadline] = []
        for item in items or []:
            title = (item.get("title") or "").strip()
            if not title:
                continue
            out.append(
                NewsHeadline(
                    title=title,
                    source="Benzinga",
                    url=item.get("url", ""),
                    published_at=item.get("created"),
                )
            )
        return out
