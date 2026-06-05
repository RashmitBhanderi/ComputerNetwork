"""Price / quote fetcher.

Primary source is Alpaca (paper account, latest trade + daily bar). marketdata.app
is the fallback quote source and supplies the VIX level used by the sentiment
engine. Every method degrades gracefully: if a key is missing or a request fails
after retries it returns ``None`` rather than raising, so the scheduler keeps
running.
"""
from __future__ import annotations

from typing import Any, Optional

import httpx

from config import MARKETDATA_BASE_URL, settings
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
        """Latest price for a ticker, Alpaca first then marketdata.app fallback."""
        snap = await self._alpaca_snapshot(ticker)
        if snap is not None:
            return snap
        return await self._marketdata_quote(ticker)

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

    # ─── marketdata.app ──────────────────────────────────────────────────────
    @property
    def _marketdata_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {settings.marketdata_api_key}",
            "Accept": "application/json",
        }

    @staticmethod
    def _first(data: dict, key: str) -> Optional[Any]:
        """Pull the first element of a columnar marketdata field."""
        val = data.get(key)
        if isinstance(val, list) and val:
            return val[0]
        return None

    async def _marketdata_quote(self, ticker: str) -> Optional[PriceSnapshot]:
        if not settings.marketdata_api_key:
            return None
        url = f"{MARKETDATA_BASE_URL}/v1/stocks/quotes/{ticker}/"
        try:
            data = await request_json(
                self._client, "GET", url, headers=self._marketdata_headers
            )
        except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001
            log.warning("marketdata.app quote failed for %s: %s", ticker, exc)
            return None

        if data.get("s") != "ok":
            return None
        last = self._first(data, "last")
        if last is None:
            return None
        change = self._first(data, "change")
        change_pct = self._first(data, "changepct")
        prev_close = last - change if change is not None else None
        # marketdata returns changepct as a fraction (e.g. 0.012 = 1.2%).
        change_pct_val = round(change_pct * 100, 3) if change_pct is not None else None
        return PriceSnapshot(
            ticker=ticker,
            price=float(last),
            prev_close=float(prev_close) if prev_close is not None else None,
            change_pct=change_pct_val,
            volume=self._first(data, "volume"),
        )

    async def get_vix(self) -> Optional[float]:
        """Current VIX level via Yahoo Finance (free, no key, ``^VIX``).

        marketdata.app's index endpoint requires a separate indices entitlement,
        so we use Yahoo's public chart endpoint instead — it's free and reliable.
        """
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
        try:
            data = await request_json(
                self._client, "GET", url,
                params={"interval": "1d", "range": "1d"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
        except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001
            log.warning("VIX fetch failed: %s", exc)
            return None
        try:
            result = (data.get("chart") or {}).get("result") or []
            price = result[0]["meta"].get("regularMarketPrice")
            return float(price) if price is not None else None
        except (KeyError, IndexError, TypeError):
            return None
