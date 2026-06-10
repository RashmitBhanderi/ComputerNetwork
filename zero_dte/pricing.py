"""Synthetic 0DTE option pricing via Black-Scholes.

The free data pipeline has no historical options quotes, so the backtest and
the live premium *estimates* price ATM same-day options with Black-Scholes,
seeding IV from the day's VIX level. This is an approximation — real 0DTE
premiums differ with the intraday vol surface — and every surface that shows
these numbers labels them as estimates. Friction (slippage + commissions) is
applied separately by the backtester so the pricing stays pure.
"""
from __future__ import annotations

import math

from zero_dte.config import config

SQRT_2 = math.sqrt(2.0)
TRADING_DAY_HOURS = 6.5
YEAR_HOURS = 252.0 * TRADING_DAY_HOURS


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


def bs_price(
    spot: float,
    strike: float,
    hours_to_expiry: float,
    iv: float,
    is_call: bool,
    rate: float | None = None,
) -> float:
    """Black-Scholes price of a same-day option, time measured in market hours."""
    rate = config.risk_free_rate if rate is None else rate
    t = max(hours_to_expiry, 1e-6) / YEAR_HOURS
    iv = max(iv, 1e-4)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * t) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    if is_call:
        price = spot * _norm_cdf(d1) - strike * math.exp(-rate * t) * _norm_cdf(d2)
    else:
        price = strike * math.exp(-rate * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
    return max(price, intrinsic)


def iv_from_vix(vix: float) -> float:
    """Annualised IV seed for synthetic 0DTE pricing from the VIX level."""
    return max(vix, 5.0) / 100.0 * config.iv_multiplier


def atm_strike(spot: float, increment: float = 1.0) -> float:
    """Nearest listed strike. SPY/QQQ list $1-wide strikes near the money."""
    return round(spot / increment) * increment


def strike_increment(symbol: str, spot: float) -> float:
    """Approximate near-the-money strike spacing per symbol."""
    if symbol in ("SPY", "QQQ"):
        return 1.0
    if spot >= 500:
        return 5.0
    return 2.5 if spot >= 200 else 1.0
