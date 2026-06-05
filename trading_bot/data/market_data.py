"""Price / quote fetcher.

Primary source is Alpaca (paper account, latest trade + daily bar). Tradier is
used as a fallback quote source and also exposes the VIX level used by the
sentiment engine. Every method degrades gracefully: if a key is missing or a
request fails after retries, it returns ``None`` rather than raising, so the
scheduler keeps running.
"""
from __future__ import annotations

from typing import Optional

import httpx

from config import settings
from data.http_client import request_json
from logging_config import get_logger
from models import PriceSnapshot

log = get_logger(__name__)


class MarketDataClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient()

    async def __aenter__(self) -> "MarketDataClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    # ─── Alpaca ──────────────────────────────────────────────────────────────
    @property
    def _alpaca_headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        }

    async def get_price(self, ticker: str) -> Optional[PriceSnapshot]:
        """Latest price for a ticker, Alpaca first then Tradier fallback."""
        snap = await self._alpaca_snapshot(ticker)
        if snap is not None:
            return snap
        return await self._tradier_quote(ticker)

    async def _alpaca_snapshot(self, ticker: str) -> Optional[PriceSnapshot]:
        if not (settings.alpaca_api_key and settings.alpaca_secret_key):
            return None
        url = f"{settings.alpaca_data_url}/v2/stocks/{ticker}/snapshot"
        try:
            data = await request_json(
                self._client, "GET", url, headers=self._alpaca_headers
            )
        except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001
            log.warning("Alpaca snapshot failed for %s: %s", ticker, exc)
            return None

        latest = data.get("latestTrade") or {}
        daily = data.get("dailyBar") or {}
        prev = data.get("prevDailyBar") or {}
        price = latest.get("p") or daily.get("c")
        if price is None:
            return None
        prev_close = prev.get("c")
        change_pct = None
        if prev_close:
            change_pct = round((price - prev_close) / prev_close * 100, 3)
        return PriceSnapshot(
            ticker=ticker,
            price=float(price),
            prev_close=float(prev_close) if prev_close else None,
            change_pct=change_pct,
            day_high=daily.get("h"),
            day_low=daily.get("l"),
            volume=daily.get("v"),
        )

    # ─── Tradier ─────────────────────────────────────────────────────────────
    @property
    def _tradier_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {settings.tradier_api_key}",
            "Accept": "application/json",
        }

    async def _tradier_quote(self, ticker: str) -> Optional[PriceSnapshot]:
        if not settings.tradier_api_key:
            return None
        url = f"{settings.tradier_base_url}/v1/markets/quotes"
        try:
            data = await request_json(
                self._client, "GET", url,
                params={"symbols": ticker}, headers=self._tradier_headers,
            )
        except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001
            log.warning("Tradier quote failed for %s: %s", ticker, exc)
            return None

        quote = (data.get("quotes") or {}).get("quote")
        if isinstance(quote, list):
            quote = quote[0] if quote else None
        if not quote or quote.get("last") is None:
            return None
        price = float(quote["last"])
        prev_close = quote.get("prevclose")
        change_pct = quote.get("change_percentage")
        return PriceSnapshot(
            ticker=ticker,
            price=price,
            prev_close=float(prev_close) if prev_close else None,
            change_pct=float(change_pct) if change_pct is not None else None,
            day_high=quote.get("high"),
            day_low=quote.get("low"),
            volume=quote.get("volume"),
        )

    async def get_vix(self) -> Optional[float]:
        """Current VIX level via Tradier (symbol ``VIX``)."""
        if not settings.tradier_api_key:
            return None
        url = f"{settings.tradier_base_url}/v1/markets/quotes"
        try:
            data = await request_json(
                self._client, "GET", url,
                params={"symbols": "VIX"}, headers=self._tradier_headers,
            )
        except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001
            log.warning("VIX fetch failed: %s", exc)
            return None
        quote = (data.get("quotes") or {}).get("quote")
        if isinstance(quote, list):
            quote = quote[0] if quote else None
        if not quote or quote.get("last") is None:
            return None
        return float(quote["last"])
