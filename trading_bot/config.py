"""Central configuration: loads API keys and settings from the environment.

All other modules import from here so there is a single source of truth. Values
come from a `.env` file (see `.env.example`) loaded via python-dotenv, with
hard-coded defaults for everything that is not a secret.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Resolve paths relative to this file so the bot runs from any working dir.
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
DB_PATH = BASE_DIR / "database" / "trading_bot.db"

# Load .env from the project root if present (no error if it's missing).
load_dotenv(BASE_DIR / ".env")

LOG_DIR.mkdir(exist_ok=True)


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _get_int(key: str, default: int) -> int:
    raw = _get(key)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    raw = _get(key)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _get_list(key: str, default: list[str]) -> list[str]:
    raw = _get(key)
    if not raw:
        return default
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


# ─── Claude model IDs ────────────────────────────────────────────────────────
# Sentiment scoring is high-volume + simple -> use the cheaper Sonnet tier.
# Deep trade analysis is correctness-sensitive -> use the most capable Opus.
SENTIMENT_MODEL = "claude-sonnet-4-6"
ANALYSIS_MODEL = "claude-opus-4-8"

# marketdata.app is the options/Greeks + VIX provider (Canada-accessible; it's a
# data vendor, not a brokerage, so there's no account-residency restriction).
MARKETDATA_BASE_URL = "https://api.marketdata.app"


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings, populated from the environment."""

    # ─── API credentials ────────────────────────────────────────────────────
    alpaca_api_key: str = field(default_factory=lambda: _get("ALPACA_API_KEY"))
    alpaca_secret_key: str = field(default_factory=lambda: _get("ALPACA_SECRET_KEY"))
    alpaca_base_url: str = field(
        default_factory=lambda: _get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    )
    alpaca_data_url: str = field(
        default_factory=lambda: _get("ALPACA_DATA_URL", "https://data.alpaca.markets")
    )
    marketdata_api_key: str = field(default_factory=lambda: _get("MARKETDATA_API_KEY"))
    news_api_key: str = field(default_factory=lambda: _get("NEWS_API_KEY"))
    benzinga_api_key: str = field(default_factory=lambda: _get("BENZINGA_API_KEY"))
    anthropic_api_key: str = field(default_factory=lambda: _get("ANTHROPIC_API_KEY"))

    # ─── Trading parameters ──────────────────────────────────────────────────
    watchlist: list[str] = field(
        default_factory=lambda: _get_list(
            "WATCHLIST", ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMZN"]
        )
    )
    starting_capital: float = field(
        default_factory=lambda: _get_float("STARTING_CAPITAL", 100_000.0)
    )
    min_confidence: int = field(default_factory=lambda: _get_int("MIN_CONFIDENCE", 65))
    max_open_positions: int = field(
        default_factory=lambda: _get_int("MAX_OPEN_POSITIONS", 3)
    )
    max_risk_per_trade_pct: float = field(
        default_factory=lambda: _get_float("MAX_RISK_PER_TRADE_PCT", 5.0)
    )
    stop_loss_pct: float = field(default_factory=lambda: _get_float("STOP_LOSS_PCT", 40.0))
    vix_halt_threshold: float = field(
        default_factory=lambda: _get_float("VIX_HALT_THRESHOLD", 35.0)
    )

    # ─── Options scan focus ──────────────────────────────────────────────────
    min_dte: int = 0
    max_dte: int = 21
    strikes_around_atm: int = 2

    # ─── Sentiment engine weights (must sum to 1.0) ──────────────────────────
    weight_news: float = 0.30
    weight_iv_rank: float = 0.25
    weight_put_call: float = 0.20
    weight_vix: float = 0.15
    weight_options_flow: float = 0.10

    # ─── Learning loop ───────────────────────────────────────────────────────
    history_lookback: int = 10  # trades injected into the prompt as context

    @property
    def headlines_per_ticker(self) -> int:
        return 20

    def missing_keys(self) -> list[str]:
        """Return the names of required credentials that are not configured."""
        required = {
            "ALPACA_API_KEY": self.alpaca_api_key,
            "ALPACA_SECRET_KEY": self.alpaca_secret_key,
            "MARKETDATA_API_KEY": self.marketdata_api_key,
            "NEWS_API_KEY": self.news_api_key,
            "ANTHROPIC_API_KEY": self.anthropic_api_key,
        }
        return [name for name, value in required.items() if not value]


# Singleton imported across the project.
settings = Settings()
