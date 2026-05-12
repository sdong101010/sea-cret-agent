"""Shared Gemini API client used by all LLM-calling modules."""

import logging

import httpx

import config

logger = logging.getLogger(__name__)

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


async def generate(
    prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 300,
    json_mode: bool = False,
    thinking_budget: int | None = None,
    timeout: float = 30.0,
) -> str:
    """Call Gemini and return the text response. Returns '' on failure.

    Args:
        thinking_budget: Token budget for Gemini 2.5's internal reasoning.
                         0 disables thinking (good for simple classification).
                         None uses the model default.
        timeout: HTTP request timeout in seconds.
    """
    url = GEMINI_ENDPOINT.format(model=config.GEMINI_MODEL)

    generation_config: dict = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
    }
    if json_mode:
        generation_config["responseMimeType"] = "application/json"
    if thinking_budget is not None:
        generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                url,
                params={"key": config.GEMINI_API_KEY},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": generation_config,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            parts = data["candidates"][0]["content"]["parts"]
            for part in reversed(parts):
                if not part.get("thought") and part.get("text"):
                    return part["text"].strip()
            return parts[-1].get("text", "").strip()
    except Exception:
        logger.exception("Gemini API call failed")
        return ""
