"""Pydantic models shared across the bot: signals, sentiment, AI responses, trades.

Using pydantic gives us validation at the boundaries (especially parsing the
Claude JSON response) and a single typed vocabulary for the whole pipeline.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─── Market / options snapshots ──────────────────────────────────────────────
class PriceSnapshot(BaseModel):
    ticker: str
    price: float
    prev_close: Optional[float] = None
    change_pct: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    volume: Optional[float] = None
    as_of: datetime = Field(default_factory=_utcnow)


class OptionContract(BaseModel):
    symbol: str
    underlying: str
    expiration: str
    strike: float
    option_type: str  # "call" | "put"
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    iv: Optional[float] = None

    @property
    def mid(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None and self.ask > 0:
            return round((self.bid + self.ask) / 2, 4)
        return self.last


class OptionsSnapshot(BaseModel):
    ticker: str
    expiration: Optional[str] = None
    underlying_price: Optional[float] = None
    atm_strike: Optional[float] = None
    iv_rank: Optional[float] = None  # 0-100
    put_call_ratio: Optional[float] = None
    contracts: list[OptionContract] = Field(default_factory=list)
    as_of: datetime = Field(default_factory=_utcnow)


# ─── Sentiment ───────────────────────────────────────────────────────────────
class NewsHeadline(BaseModel):
    title: str
    source: str = "unknown"
    url: str = ""
    published_at: Optional[str] = None
    sentiment: Optional[float] = None  # -1.0 .. +1.0, filled by Claude


class SentimentBreakdown(BaseModel):
    """Component scores (each normalised to -1..+1) plus the weighted result."""

    ticker: str
    news_score: float = 0.0
    iv_rank_score: float = 0.0
    put_call_score: float = 0.0
    vix_score: float = 0.0
    options_flow_score: float = 0.0
    unified_score: float = 0.0  # -1..+1
    bias: Direction = Direction.NEUTRAL
    vix_level: Optional[float] = None
    fear_greed: Optional[float] = None
    headline_count: int = 0
    as_of: datetime = Field(default_factory=_utcnow)


# ─── AI trade decision ───────────────────────────────────────────────────────
class TradeSignal(BaseModel):
    """Structured output parsed from Claude's analysis response."""

    ticker: str
    direction: Direction
    strategy: str
    confidence: float = Field(ge=0, le=100)
    reasoning: str
    entry_level: Optional[float] = None
    exit_level: Optional[float] = None
    stop_level: Optional[float] = None
    target_contract: Optional[str] = None
    sentiment_score: float = 0.0
    as_of: datetime = Field(default_factory=_utcnow)

    @field_validator("direction", mode="before")
    @classmethod
    def _coerce_direction(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v: object) -> object:
        # Claude may return "75%" or 0.75 — normalise to a 0-100 float.
        if isinstance(v, str):
            v = v.replace("%", "").strip()
        num = float(v)  # type: ignore[arg-type]
        if 0 < num <= 1:
            num *= 100
        return max(0.0, min(100.0, num))


# ─── Positions & trades ──────────────────────────────────────────────────────
class Trade(BaseModel):
    id: Optional[int] = None
    ticker: str
    strategy: str
    direction: Direction
    option_symbol: str
    contracts: int
    entry_price: float  # premium per contract (per share)
    exit_price: Optional[float] = None
    quantity_multiplier: int = 100  # shares per option contract
    pnl: Optional[float] = None
    confidence: float = 0.0
    sentiment_score: float = 0.0
    delta: Optional[float] = None
    theta: Optional[float] = None
    iv: Optional[float] = None
    stop_price: Optional[float] = None
    expiration: Optional[str] = None
    status: TradeStatus = TradeStatus.OPEN
    reasoning: str = ""
    opened_at: datetime = Field(default_factory=_utcnow)
    closed_at: Optional[datetime] = None
    close_reason: Optional[str] = None

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.contracts * self.quantity_multiplier

    def market_value(self, current_premium: float) -> float:
        return current_premium * self.contracts * self.quantity_multiplier

    def unrealized_pnl(self, current_premium: float) -> float:
        return self.market_value(current_premium) - self.cost_basis
