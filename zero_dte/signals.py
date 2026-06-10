"""0DTE signal engine: opening-range breakout + trend confluence.

Decision is made once per session at 10:30 ET, after the opening range
(09:30-10:30) completes. Three independent votes:

  1. Opening-range breakout — 10:30 price above the OR high (call) or below
     the OR low (put).
  2. Trend — fast EMA above/below slow EMA on hourly closes.
  3. Prior-session momentum — yesterday's close-over-open direction agrees.

An RSI extreme veto blocks chasing: no calls when hourly RSI > overbought,
no puts when RSI < oversold. A signal fires when at least ``min_score`` votes
agree on one side. Pure function of the bar history — deterministic and
backtestable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from zero_dte.config import config


@dataclass(frozen=True)
class Signal:
    symbol: str
    session: date
    side: str               # "call" | "put"
    score: int              # confluence votes (min_score..3)
    entry_underlying: float
    or_high: float
    or_low: float
    reasons: tuple[str, ...]


def _rsi(closes: pd.Series, period: int) -> float:
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    last_gain = float(gain.iloc[-1])
    last_loss = float(loss.iloc[-1])
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return 100.0 - 100.0 / (1.0 + rs)


def evaluate_session(
    symbol: str, bars: pd.DataFrame, session: date
) -> Optional[Signal]:
    """Score one session at the 10:30 ET decision point. None = stand aside.

    ``bars`` is the full hourly history (Eastern-indexed); only data up to and
    including the session's 10:30 bar open is used — no lookahead.
    """
    day = bars[bars.index.date == session]
    if len(day) < 2:
        return None

    or_bar = day.iloc[0]                  # 09:30-10:30 opening range
    or_high, or_low = float(or_bar["high"]), float(or_bar["low"])
    decision_price = float(day.iloc[1]["open"])   # 10:30 bar open

    history = bars[bars.index < day.index[1]]     # strictly before 10:30
    closes = history["close"]
    if len(closes) < config.ema_slow + 2:
        return None

    call_votes: list[str] = []
    put_votes: list[str] = []

    # 1. Opening-range breakout
    if decision_price > or_high:
        call_votes.append(f"ORB breakout above {or_high:.2f}")
    elif decision_price < or_low:
        put_votes.append(f"ORB breakdown below {or_low:.2f}")

    # 2. EMA trend
    ema_fast = float(closes.ewm(span=config.ema_fast).mean().iloc[-1])
    ema_slow = float(closes.ewm(span=config.ema_slow).mean().iloc[-1])
    if ema_fast > ema_slow:
        call_votes.append(f"EMA{config.ema_fast}>{config.ema_slow} uptrend")
    elif ema_fast < ema_slow:
        put_votes.append(f"EMA{config.ema_fast}<{config.ema_slow} downtrend")

    # 3. Prior-session momentum
    prev = bars[bars.index.date < session]
    if not prev.empty:
        prev_day = prev[prev.index.date == prev.index.date[-1]]
        prev_move = float(prev_day["close"].iloc[-1]) - float(prev_day["open"].iloc[0])
        if prev_move > 0:
            call_votes.append("prior session closed up")
        elif prev_move < 0:
            put_votes.append("prior session closed down")

    rsi = _rsi(closes, config.rsi_period)
    side: Optional[str] = None
    reasons: list[str] = []
    if len(call_votes) >= config.min_score and len(call_votes) > len(put_votes):
        if rsi <= config.rsi_overbought:
            side, reasons = "call", call_votes
    elif len(put_votes) >= config.min_score and len(put_votes) > len(call_votes):
        if rsi >= config.rsi_oversold:
            side, reasons = "put", put_votes

    if side is None:
        return None
    return Signal(
        symbol=symbol,
        session=session,
        side=side,
        score=len(reasons),
        entry_underlying=decision_price,
        or_high=or_high,
        or_low=or_low,
        reasons=tuple(reasons),
    )
