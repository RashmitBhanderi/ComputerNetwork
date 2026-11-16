"""Claude review of candidate trades, via the local Claude Code CLI.

Runs `claude -p` headless, so it uses whatever account the CLI is logged into
(your Claude subscription) instead of a metered API key. Fails closed: if the
call fails or the answer can't be parsed, nothing is approved."""
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile

log = logging.getLogger("ai")

SYSTEM = (
    "You are the risk-review step of an intraday long-options trading bot for a small "
    "account. You are given candidate trades that already passed mechanical momentum and "
    "liquidity filters. Be sceptical: reject chasing extended moves, entries into nearby "
    "support/resistance (opening range, day high/low, VWAP), weak or fading volume, signals "
    "that conflict with the broad market (SPY/QQQ), and contracts whose theta or IV make the "
    "+50% target unrealistic against the -30% stop. Approving nothing is a fine answer. "
    "Reply with JSON only, no prose and no code fences."
)


def _call(prompt: str, cfg: dict) -> str | None:
    exe = shutil.which("claude")
    if not exe:
        log.error("claude CLI not found on PATH")
        return None
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}  # force subscription login
    cmd = [exe, "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
           "--system-prompt", SYSTEM]
    if cfg.get("ai_model"):
        cmd += ["--model", str(cfg["ai_model"])]
    try:
        out = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=env,
                             cwd=tempfile.gettempdir(), timeout=cfg.get("ai_timeout_seconds", 120))
        data = json.loads(out.stdout)
        if data.get("is_error"):
            log.error("claude error: %s", str(data.get("result"))[:300])
            return None
        return data.get("result")
    except Exception as e:  # timeout, bad JSON, non-zero exit
        log.error("claude call failed: %s", e)
        return None


def _json(text: str | None, opener: str):
    if not text:
        return None
    closer = "]" if opener == "[" else "}"
    m = re.search(re.escape(opener) + r".*" + re.escape(closer), text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


def review(candidates: list[dict], context: dict, cfg: dict) -> dict:
    """Returns {symbol: {"take": bool, "confidence": float, "reason": str}}."""
    prompt = (
        "Market context:\n" + json.dumps(context, indent=1) +
        "\n\nCandidates:\n" + json.dumps(candidates, indent=1) +
        '\n\nReturn a JSON array with one object per candidate: '
        '{"symbol": str, "decision": "take" | "skip", "confidence": 0..1, "reason": "<= 25 words"}'
    )
    rows = _json(_call(prompt, cfg), "[")
    out = {}
    for r in rows if isinstance(rows, list) else []:
        try:
            conf = float(r.get("confidence", 0))
            out[str(r["symbol"])] = {
                "take": r.get("decision") == "take" and conf >= cfg["ai_min_confidence"],
                "confidence": conf, "reason": str(r.get("reason", ""))[:200]}
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return out


def morning_brief(market: dict, cfg: dict) -> str:
    prompt = (
        "Pre-market data for today's watchlist (last daily closes, gaps):\n" + json.dumps(market, indent=1) +
        '\n\nReturn a JSON object: {"bias": "bullish" | "bearish" | "mixed", '
        '"brief": "<= 80 words on what to favour and avoid today"}'
    )
    obj = _json(_call(prompt, cfg), "{")
    if isinstance(obj, dict) and "brief" in obj:
        return f"{obj.get('bias', 'mixed')}: {obj['brief']}"[:700]
    return ""
