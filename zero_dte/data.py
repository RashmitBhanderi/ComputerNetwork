"""Free market data via Yahoo Finance (no API key required).

Provides hourly intraday bars (~730 days of history — enough for a 2025-2026
backtest), daily bars, and the daily VIX series used to seed synthetic 0DTE
implied volatility.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, date
from typing import Optional

import pandas as pd
import pytz

EASTERN = pytz.timezone("US/Eastern")
_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def _fetch(symbol: str, interval: str, range_: str, retries: int = 3) -> dict:
    url = f"{_BASE}/{urllib.request.quote(symbol)}?interval={interval}&range={range_}"
    last: Exception = RuntimeError("unreachable")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    raise last


def _to_frame(payload: dict, tz=EASTERN) -> pd.DataFrame:
    result = payload["chart"]["result"][0]
    ts = result.get("timestamp") or []
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz),
    )
    return df.dropna(subset=["open", "high", "low", "close"])


def hourly_bars(symbol: str, range_: str = "730d") -> pd.DataFrame:
    """Hourly RTH bars (Eastern-indexed): 09:30, 10:30, ... 15:30."""
    return _to_frame(_fetch(symbol, "60m", range_))


def daily_bars(symbol: str, range_: str = "2y") -> pd.DataFrame:
    return _to_frame(_fetch(symbol, "1d", range_))


def vix_by_date(range_: str = "2y") -> dict[date, float]:
    """Map of trading date -> VIX close (used as the synthetic IV seed)."""
    df = daily_bars("^VIX", range_)
    return {idx.date(): float(row["close"]) for idx, row in df.iterrows()}


def latest_price(symbol: str) -> Optional[float]:
    try:
        payload = _fetch(symbol, "1d", "1d")
        meta = payload["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice")
        return float(price) if price is not None else None
    except Exception:  # noqa: BLE001
        return None


def latest_vix() -> Optional[float]:
    return latest_price("^VIX")


def now_eastern() -> datetime:
    return datetime.now(EASTERN)
