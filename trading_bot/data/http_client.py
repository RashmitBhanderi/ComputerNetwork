"""Async HTTP helpers with retry/backoff shared by every data fetcher.

`request_json` wraps httpx with tenacity so transient network errors and 5xx /
429 responses are retried with exponential backoff. Permanent client errors
(4xx other than 429) raise immediately so we don't hammer a misconfigured key.
"""
from __future__ import annotations

from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from logging_config import get_logger

log = get_logger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


class RetryableHTTPError(Exception):
    """Raised for 429/5xx so tenacity retries; 4xx are not wrapped."""


@retry(
    retry=retry_if_exception_type((httpx.TransportError, RetryableHTTPError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=16),
    reraise=True,
)
async def request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    json_body: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Perform a request and return parsed JSON, retrying transient failures."""
    resp = await client.request(
        method, url, params=params, headers=headers, json=json_body,
        timeout=DEFAULT_TIMEOUT,
    )
    if resp.status_code == 429 or resp.status_code >= 500:
        log.warning("Retryable %s from %s", resp.status_code, url)
        raise RetryableHTTPError(f"{resp.status_code} from {url}")
    resp.raise_for_status()
    if not resp.content:
        return {}
    return resp.json()
