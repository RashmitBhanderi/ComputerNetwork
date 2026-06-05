"""News fetcher: NewsAPI (primary) + Alpaca News (free secondary).

Returns up to ``settings.headlines_per_ticker`` recent headlines per ticker as
``NewsHeadline`` objects (sentiment is filled in later by the sentiment module).
Alpaca's News API is free with the same Alpaca data key the bot already uses,
so the secondary source costs nothing. Both sources degrade gracefully when
their key is missing or a call fails.
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
        """Combine NewsAPI + Alpaca News, de-duplicated, newest first."""
        headlines: list[NewsHeadline] = []
        headlines.extend(await self._newsapi(ticker))
        headlines.extend(await self._alpaca_news(ticker))

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

    async def _alpaca_news(self, ticker: str) -> list[NewsHeadline]:
        """Alpaca News API — free with the Alpaca data key the bot already uses."""
        if not (settings.alpaca_api_key and settings.alpaca_secret_key):
            return []
        try:
            data = await request_json(
                self._client, "GET",
                f"{settings.alpaca_data_url}/v1beta1/news",
                params={
                    "symbols": ticker,
                    "limit": str(min(settings.headlines_per_ticker, 50)),
                    "sort": "desc",
                },
                headers={
                    "APCA-API-KEY-ID": settings.alpaca_api_key,
                    "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
                },
            )
        except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001
            log.warning("Alpaca News failed for %s: %s", ticker, exc)
            return []

        out: list[NewsHeadline] = []
        for item in data.get("news", []) or []:
            title = (item.get("headline") or "").strip()
            if not title:
                continue
            out.append(
                NewsHeadline(
                    title=title,
                    source=item.get("source", "Alpaca"),
                    url=item.get("url", ""),
                    published_at=item.get("created_at"),
                )
            )
        return out
