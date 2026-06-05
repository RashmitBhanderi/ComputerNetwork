"""Shared Anthropic client factory used by the sentiment and AI modules.

Centralising construction means the API key check and retry configuration live
in one place. The SDK already retries 429/5xx with exponential backoff; we bump
`max_retries` a little for resilience inside the scheduled jobs.
"""
from __future__ import annotations

from functools import lru_cache

import anthropic

from config import settings
from logging_config import get_logger

log = get_logger(__name__)


class ClaudeUnavailable(RuntimeError):
    """Raised when the Anthropic API key is not configured."""


@lru_cache(maxsize=1)
def get_async_client() -> anthropic.AsyncAnthropic:
    if not settings.anthropic_api_key:
        raise ClaudeUnavailable("ANTHROPIC_API_KEY is not set")
    return anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key, max_retries=4
    )


def is_available() -> bool:
    return bool(settings.anthropic_api_key)
