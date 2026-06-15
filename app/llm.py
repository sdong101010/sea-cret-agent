"""Shared Claude client used by all LLM-calling modules.

Talks to the Salesforce AI Model Gateway, which fronts Bedrock with a
bearer-token auth and Bedrock-style routing (/model/{model}/invoke)
but returns Anthropic-shaped JSON.
"""

import logging
import re

import httpx

import config

logger = logging.getLogger(__name__)


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Claude sometimes wraps JSON in ```json fences despite instructions."""
    m = _FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text


async def generate(
    prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 300,
    json_mode: bool = False,
    thinking_budget: int | None = None,
    timeout: float = 30.0,
) -> str:
    """Call Claude and return the text response. Returns '' on failure.

    Args:
        thinking_budget: Token budget for extended thinking. 0 or None disables.
                         When enabled, temperature is forced to 1 (API requirement)
                         and max_tokens is bumped above the budget.
        json_mode: Strips ```json code fences from the response (Claude has no
                   native JSON mode; prompts must instruct JSON output).
        timeout: HTTP request timeout in seconds.
    """
    base = config.ANTHROPIC_BASE_URL.rstrip("/")
    url = f"{base}/model/{config.ANTHROPIC_MODEL}/invoke"

    # Opus 4.7 on the gateway has deprecated `temperature` -- the parameter
    # is accepted in the public function signature for caller compatibility
    # but never forwarded. Same for `thinking` (extended thinking on Opus 4.7
    # is not exposed through this gateway).
    body: dict = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max(max_tokens, (thinking_budget or 0) + 1024)
                      if thinking_budget else max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    headers = {
        "Authorization": f"Bearer {config.ANTHROPIC_AUTH_TOKEN}",
        "content-type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code >= 400:
                logger.error("Claude gateway %d: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
            data = resp.json()
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text = block.get("text", "").strip()
                    return _strip_code_fences(text) if json_mode else text
            return ""
    except Exception:
        logger.exception("Claude API call failed")
        return ""
