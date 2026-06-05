"""Shared Anthropic client factory used by the sentiment and AI modules.

Centralising construction means the API key check and retry configuration live
in one place. The SDK already retries 429/5xx with exponential backoff; we bump
`max_retries` a little for resilience inside the scheduled jobs.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Optional

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


def text_of(message: anthropic.types.Message) -> str:
    """Concatenate all text blocks of a Messages API response."""
    return "".join(b.text for b in message.content if getattr(b, "type", None) == "text")


def extract_json(text: str) -> Optional[dict]:
    """Best-effort parse of a JSON object from a model response.

    Handles plain JSON, JSON wrapped in ```json fences, or JSON embedded in
    surrounding prose. Returns None if nothing parseable is found. This keeps us
    independent of the SDK's structured-output helper (absent in older versions).
    """
    if not text:
        return None
    text = text.strip()
    # Strip code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced {...} span.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None
