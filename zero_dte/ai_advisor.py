"""Claude-powered live trade advisor (runtime layer, uses ANTHROPIC_API_KEY).

The backtestable AI core is the walk-forward ML gate in research.py — its
2025-2026 verdict (no setup clears 55% win probability for long ATM 0DTE)
is injected into this advisor's prompt so Claude reasons WITH the evidence,
not against it. The advisor returns a structured go/no-go + sizing
recommendation for a proposed ticket. No key → conservative NO-GO fallback.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

ADVISOR_MODEL = "claude-opus-4-8"

_SYSTEM = (
    "You are a ruthless risk manager for a $1-2k retail account trading long "
    "0DTE calls/puts on moomoo. Walk-forward research on 2025-2026 (96 configs, "
    "real SPY/QQQ data) found NO profitable configuration: best out-of-sample "
    "profit factor 0.43, and an ML gate trained on market-condition features "
    "filtered out every trade (no setup exceeded 55% win probability). Theta "
    "decay plus spread friction structurally exceed the directional edge at "
    "this resolution. Your default answer is NO-GO; approve only a truly "
    "exceptional confluence, and never above 1 contract / minimum size.\n\n"
    'Respond ONLY with JSON: {"go": true|false, "size_contracts": <int>, '
    '"confidence": <0-100>, "reasoning": "<2-3 sentences>"}'
)


@dataclass
class Advice:
    go: bool
    size_contracts: int
    confidence: float
    reasoning: str
    source: str  # "claude" | "fallback"


def advise(
    symbol: str, side: str, strike: float, est_premium: float,
    vix: Optional[float], signal_reasons: tuple[str, ...] | list[str],
    capital: float,
) -> Advice:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    fallback = Advice(
        go=False, size_contracts=0, confidence=0.0,
        reasoning=("Research verdict: no profitable long-0DTE configuration "
                   "found 2025-2026; ML gate filters all setups. Standing aside."),
        source="fallback",
    )
    if not api_key:
        return fallback
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, max_retries=2)
        prompt = (
            f"Proposed ticket: {symbol} 0DTE {side.upper()} strike {strike:g}, "
            f"est premium ${est_premium:.2f} (${est_premium * 100:.0f}/contract), "
            f"account capital ${capital:,.0f}, VIX {vix if vix is not None else 'n/a'}.\n"
            f"Rule-engine reasons: {'; '.join(signal_reasons) or 'none'}.\n"
            "Decision?"
        )
        msg = client.messages.create(
            model=ADVISOR_MODEL, max_tokens=500, system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return fallback
        data = json.loads(m.group(0))
        return Advice(
            go=bool(data.get("go", False)),
            size_contracts=max(0, min(int(data.get("size_contracts", 0)), 1)),
            confidence=float(data.get("confidence", 0)),
            reasoning=str(data.get("reasoning", ""))[:500],
            source="claude",
        )
    except Exception as exc:  # noqa: BLE001
        fallback.reasoning += f" (advisor error: {exc})"
        return fallback
