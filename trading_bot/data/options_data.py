"""Options chain + Greeks fetcher (marketdata.app), with put/call ratio + IV rank.

marketdata.app returns per-contract Greeks (delta, gamma, theta, vega) and IV in
a single chain call, plus a quotes-by-OCC-symbol endpoint for mark-to-market.
It's a data vendor (not a brokerage), so it has no account-residency restriction
— a working replacement for Tradier outside the US.

From a near-dated chain we derive:
  * the nearest expiry inside the configured DTE window,
  * the ATM strike and a window of strikes around it,
  * the put/call ratio (put volume / call volume, falling back to open interest),
  * an IV-rank proxy from the at-the-money implied volatility.

IV rank normally needs a trailing 52-week IV series, which the chain endpoint
doesn't expose; we approximate it by mapping ATM IV onto a typical 10-80% band
and label it an estimate everywhere it surfaces.

Responses are *columnar*: each field is a parallel array, one entry per
contract. We zip them back into `OptionContract` rows.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Optional

import httpx
import pytz

from config import MARKETDATA_BASE_URL, settings
from data.http_client import request_json
from logging_config import get_logger
from models import OptionContract, OptionsSnapshot

log = get_logger(__name__)

_EASTERN = pytz.timezone("US/Eastern")

# Typical equity-index ATM IV band used for the IV-rank proxy.
_IV_FLOOR = 0.10
_IV_CEIL = 0.80


def _ts_to_iso_date(value: Any) -> str:
    """Convert a marketdata expiration value to an ISO date string.

    The chain returns expirations as Unix timestamps (seconds, at the exchange
    close). We interpret them in US/Eastern so the calendar date is correct.
    Already-ISO strings are passed through.
    """
    if isinstance(value, str):
        if value.isdigit():
            value = int(value)
        else:
            return value[:10]
    try:
        return datetime.fromtimestamp(int(value), _EASTERN).date().isoformat()
    except (ValueError, OSError, OverflowError, TypeError):
        return ""


class OptionsDataClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient()

    async def __aenter__(self) -> "OptionsDataClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {settings.marketdata_api_key}",
            "Accept": "application/json",
        }

    async def _get(self, path: str, params: Optional[dict[str, str]] = None) -> dict:
        url = f"{MARKETDATA_BASE_URL}{path}"
        return await request_json(
            self._client, "GET", url, params=params, headers=self._headers
        )

    @staticmethod
    def _rows(data: dict) -> list[dict[str, Any]]:
        """Turn a columnar marketdata response into a list of per-row dicts."""
        if data.get("s") != "ok":
            return []
        length = 0
        for v in data.values():
            if isinstance(v, list):
                length = len(v)
                break
        rows: list[dict[str, Any]] = []
        for i in range(length):
            row = {}
            for key, val in data.items():
                if isinstance(val, list) and i < len(val):
                    row[key] = val[i]
            rows.append(row)
        return rows

    async def _fetch_chain(
        self, ticker: str, underlying_price: Optional[float]
    ) -> list[OptionContract]:
        if not settings.marketdata_api_key:
            return []

        today = date.today()
        params: dict[str, str] = {
            "from": today.isoformat(),
            "to": (today + timedelta(days=settings.max_dte)).isoformat(),
        }
        # Bound the response to a sensible band around the underlying.
        if underlying_price:
            lo = max(1, int(underlying_price * 0.85))
            hi = int(underlying_price * 1.15) + 1
            params["strike"] = f"{lo}-{hi}"

        try:
            data = await self._get(f"/v1/options/chain/{ticker}/", params)
        except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001
            log.warning("Chain fetch failed for %s: %s", ticker, exc)
            return []

        contracts: list[OptionContract] = []
        for row in self._rows(data):
            contracts.append(
                OptionContract(
                    symbol=row.get("optionSymbol", ""),
                    underlying=ticker,
                    expiration=_ts_to_iso_date(row.get("expiration")),
                    strike=float(row.get("strike", 0) or 0),
                    option_type=row.get("side", ""),
                    bid=row.get("bid"),
                    ask=row.get("ask"),
                    last=row.get("last"),
                    volume=row.get("volume"),
                    open_interest=row.get("openInterest"),
                    delta=row.get("delta"),
                    gamma=row.get("gamma"),
                    theta=row.get("theta"),
                    vega=row.get("vega"),
                    iv=row.get("iv"),
                )
            )
        return contracts

    async def get_option_quotes(self, symbols: list[str]) -> dict[str, float]:
        """Current mid price (per share) for a list of OCC option symbols."""
        if not symbols or not settings.marketdata_api_key:
            return {}

        result: dict[str, float] = {}
        for sym in symbols:
            try:
                data = await self._get(f"/v1/options/quotes/{sym}/")
            except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001
                log.warning("Option quote fetch failed for %s: %s", sym, exc)
                continue
            rows = self._rows(data)
            if not rows:
                continue
            q = rows[0]
            bid, ask, mid, last = q.get("bid"), q.get("ask"), q.get("mid"), q.get("last")
            if mid is not None:
                result[sym] = round(float(mid), 4)
            elif bid is not None and ask is not None and ask > 0:
                result[sym] = round((bid + ask) / 2, 4)
            elif last is not None:
                result[sym] = float(last)
        return result

    def _select_expiration(self, expirations: list[str]) -> Optional[str]:
        """Pick the nearest expiry that falls inside the configured DTE window."""
        today = date.today()
        candidates: list[tuple[int, str]] = []
        for exp in expirations:
            try:
                exp_date = date.fromisoformat(exp)
            except ValueError:
                continue
            dte = (exp_date - today).days
            if settings.min_dte <= dte <= settings.max_dte:
                candidates.append((dte, exp))
        if not candidates:
            valid = sorted(e for e in expirations if e)
            return valid[0] if valid else None
        candidates.sort()
        return candidates[0][1]

    async def get_snapshot(
        self, ticker: str, underlying_price: Optional[float]
    ) -> OptionsSnapshot:
        """Full short-dated options snapshot around the ATM strike."""
        contracts = await self._fetch_chain(ticker, underlying_price)
        if not contracts:
            return OptionsSnapshot(ticker=ticker, underlying_price=underlying_price)

        expirations = sorted({c.expiration for c in contracts if c.expiration})
        expiration = self._select_expiration(expirations)
        if not expiration:
            return OptionsSnapshot(ticker=ticker, underlying_price=underlying_price)

        exp_contracts = [c for c in contracts if c.expiration == expiration]
        strikes = sorted({c.strike for c in exp_contracts})
        atm_strike = self._nearest_strike(strikes, underlying_price)

        window = self._strike_window(strikes, atm_strike)
        focused = [c for c in exp_contracts if c.strike in window]

        return OptionsSnapshot(
            ticker=ticker,
            expiration=expiration,
            underlying_price=underlying_price,
            atm_strike=atm_strike,
            iv_rank=self._iv_rank(exp_contracts, atm_strike),
            put_call_ratio=self._put_call_ratio(exp_contracts),
            contracts=focused,
        )

    # ─── provider-agnostic helpers ───────────────────────────────────────────
    @staticmethod
    def _nearest_strike(
        strikes: list[float], price: Optional[float]
    ) -> Optional[float]:
        if not strikes:
            return None
        if price is None:
            return strikes[len(strikes) // 2]
        return min(strikes, key=lambda s: abs(s - price))

    def _strike_window(
        self, strikes: list[float], atm: Optional[float]
    ) -> set[float]:
        if atm is None or atm not in strikes:
            return set(strikes)
        idx = strikes.index(atm)
        n = settings.strikes_around_atm
        lo = max(0, idx - n)
        hi = min(len(strikes), idx + n + 1)
        return set(strikes[lo:hi])

    @staticmethod
    def _put_call_ratio(contracts: list[OptionContract]) -> Optional[float]:
        put_vol = sum((c.volume or 0) for c in contracts if c.option_type == "put")
        call_vol = sum((c.volume or 0) for c in contracts if c.option_type == "call")
        if call_vol == 0:
            put_oi = sum((c.open_interest or 0) for c in contracts if c.option_type == "put")
            call_oi = sum((c.open_interest or 0) for c in contracts if c.option_type == "call")
            if call_oi == 0:
                return None
            return round(put_oi / call_oi, 3)
        return round(put_vol / call_vol, 3)

    @staticmethod
    def _iv_rank(
        contracts: list[OptionContract], atm_strike: Optional[float]
    ) -> Optional[float]:
        ivs = [
            c.iv for c in contracts
            if c.iv is not None and (atm_strike is None or c.strike == atm_strike)
        ]
        if not ivs:
            ivs = [c.iv for c in contracts if c.iv is not None]
        if not ivs:
            return None
        atm_iv = sum(ivs) / len(ivs)
        rank = (atm_iv - _IV_FLOOR) / (_IV_CEIL - _IV_FLOOR) * 100
        return round(max(0.0, min(100.0, rank)), 1)
