"""Options chain + Greeks fetcher (Tradier), with put/call ratio and IV rank.

Tradier returns per-contract Greeks (delta, gamma, theta, vega, mid IV) when
``greeks=true``. From a single expiry chain we derive:

* the ATM strike and a window of strikes around it (config: strikes_around_atm),
* the put/call ratio (put volume / call volume, falling back to open interest),
* an IV-rank proxy from the at-the-money implied volatility.

IV rank normally needs a trailing 52-week IV series, which Tradier doesn't
expose directly; we approximate it by mapping ATM IV onto a typical 10–80% band.
The value is clearly labelled as an estimate everywhere it surfaces.
"""
from __future__ import annotations

from typing import Optional

import httpx

from config import settings
from data.http_client import request_json
from logging_config import get_logger
from models import OptionContract, OptionsSnapshot

log = get_logger(__name__)

# Typical equity-index ATM IV band used for the IV-rank proxy.
_IV_FLOOR = 0.10
_IV_CEIL = 0.80


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
            "Authorization": f"Bearer {settings.tradier_api_key}",
            "Accept": "application/json",
        }

    async def _get(self, path: str, params: dict[str, str]) -> dict:
        url = f"{settings.tradier_base_url}{path}"
        return await request_json(self._client, "GET", url,
                                  params=params, headers=self._headers)

    async def get_expirations(self, ticker: str) -> list[str]:
        if not settings.tradier_api_key:
            return []
        try:
            data = await self._get(
                "/v1/markets/options/expirations",
                {"symbol": ticker, "includeAllRoots": "true"},
            )
        except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001
            log.warning("Expirations fetch failed for %s: %s", ticker, exc)
            return []
        dates = (data.get("expirations") or {}).get("date")
        if dates is None:
            return []
        return dates if isinstance(dates, list) else [dates]

    async def get_chain(self, ticker: str, expiration: str) -> list[OptionContract]:
        if not settings.tradier_api_key:
            return []
        try:
            data = await self._get(
                "/v1/markets/options/chains",
                {"symbol": ticker, "expiration": expiration, "greeks": "true"},
            )
        except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001
            log.warning("Chain fetch failed for %s %s: %s", ticker, expiration, exc)
            return []

        options = (data.get("options") or {}).get("option")
        if options is None:
            return []
        if not isinstance(options, list):
            options = [options]

        contracts: list[OptionContract] = []
        for o in options:
            greeks = o.get("greeks") or {}
            contracts.append(
                OptionContract(
                    symbol=o.get("symbol", ""),
                    underlying=ticker,
                    expiration=expiration,
                    strike=float(o.get("strike", 0) or 0),
                    option_type=o.get("option_type", ""),
                    bid=o.get("bid"),
                    ask=o.get("ask"),
                    last=o.get("last"),
                    volume=o.get("volume"),
                    open_interest=o.get("open_interest"),
                    delta=greeks.get("delta"),
                    gamma=greeks.get("gamma"),
                    theta=greeks.get("theta"),
                    vega=greeks.get("vega"),
                    iv=greeks.get("mid_iv") or greeks.get("smv_vol"),
                )
            )
        return contracts

    async def get_option_quotes(self, symbols: list[str]) -> dict[str, float]:
        """Current mid price (per share) for a list of OCC option symbols."""
        if not symbols or not settings.tradier_api_key:
            return {}
        try:
            data = await self._get(
                "/v1/markets/quotes",
                {"symbols": ",".join(symbols), "greeks": "false"},
            )
        except (httpx.HTTPError, Exception) as exc:  # noqa: BLE001
            log.warning("Option quote fetch failed: %s", exc)
            return {}

        quotes = (data.get("quotes") or {}).get("quote")
        if quotes is None:
            return {}
        if not isinstance(quotes, list):
            quotes = [quotes]

        result: dict[str, float] = {}
        for q in quotes:
            symbol = q.get("symbol")
            if not symbol:
                continue
            bid, ask, last = q.get("bid"), q.get("ask"), q.get("last")
            if bid is not None and ask is not None and ask > 0:
                result[symbol] = round((bid + ask) / 2, 4)
            elif last is not None:
                result[symbol] = float(last)
        return result

    def _select_expiration(self, expirations: list[str]) -> Optional[str]:
        """Pick the nearest expiry that falls inside the configured DTE window."""
        from datetime import date

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
            return expirations[0] if expirations else None
        candidates.sort()
        return candidates[0][1]

    async def get_snapshot(
        self, ticker: str, underlying_price: Optional[float]
    ) -> OptionsSnapshot:
        """Full short-dated options snapshot around the ATM strike."""
        expirations = await self.get_expirations(ticker)
        expiration = self._select_expiration(expirations)
        if not expiration:
            return OptionsSnapshot(ticker=ticker, underlying_price=underlying_price)

        contracts = await self.get_chain(ticker, expiration)
        if not contracts:
            return OptionsSnapshot(
                ticker=ticker, expiration=expiration,
                underlying_price=underlying_price,
            )

        strikes = sorted({c.strike for c in contracts})
        atm_strike = self._nearest_strike(strikes, underlying_price)

        window = self._strike_window(strikes, atm_strike)
        focused = [c for c in contracts if c.strike in window]

        return OptionsSnapshot(
            ticker=ticker,
            expiration=expiration,
            underlying_price=underlying_price,
            atm_strike=atm_strike,
            iv_rank=self._iv_rank(contracts, atm_strike),
            put_call_ratio=self._put_call_ratio(contracts),
            contracts=focused,
        )

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
            # Fall back to open interest when intraday volume is empty.
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
