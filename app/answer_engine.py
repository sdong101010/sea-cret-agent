"""Generate answers from live web research.

Pipeline:
  1. Web search finds relevant pages (Salesforce-scoped + broad)
  2. Page content is fetched and extracted
  3. Combined web context is sent to Gemini for a thorough answer
"""

import logging

from app.question_detector import DetectedQuestion
from app.web_search import research_web
from app import llm

logger = logging.getLogger(__name__)

ANSWER_PROMPT = """\
You are coaching a Salesforce Solution Architect who is on a live call RIGHT NOW. \
They need thorough, accurate talking points they can glance at and say naturally.

Below is WEB RESEARCH fetched just now for the customer's question.

Rules:
- 3-6 bullet points. Each bullet is 1-2 sentences with a specific fact or capability.
- Use actual Salesforce product names (Data Cloud, Agentforce, MuleSoft, Flow, Shield, etc.).
- Lead with the strongest, most directly relevant point.
- Frame architecturally: lead with what the platform does. Never say "workaround."
- Include specific numbers, feature names, certifications, or limits where available.
- If a capability requires configuration, present it positively ("natively supported \
when configured" not "requires manual setup").
- If the sources don't cover this topic well, say so and suggest offering to follow up.
- Do NOT fabricate capabilities. Only state what the sources support.
- Start IMMEDIATELY with the first bullet. NO preamble, intro, or summary line.

CUSTOMER QUESTION:
{question}

WEB RESEARCH:
{context}

Output ONLY the bullets, one per line, starting each with "•":\
"""

FALLBACK_MSG = (
    "Web research didn't surface enough information on this topic. "
    "Consider addressing from your own expertise or offering to follow up with details."
)


class AnswerEngine:
    def __init__(self):
        logger.info("Answer engine initialized (web-first mode)")

    async def get_answer(
        self, question: DetectedQuestion
    ) -> tuple[str, list[str], list[dict], bool]:
        """Research the web and generate an answer.

        Returns (answer, source_refs, links, is_confident).
        """
        query = question.search_query or question.text

        web_results, web_context = await research_web(query, num_results=5)

        web_links = [
            {"title": r["title"], "url": r["url"]}
            for r in web_results if r.get("url")
        ][:3]

        if not web_context:
            return FALLBACK_MSG, [], web_links, False

        prompt = ANSWER_PROMPT.format(question=query, context=web_context)
        answer = await llm.generate(
            prompt, max_tokens=2048, thinking_budget=0, timeout=60.0
        )

        if not answer:
            return FALLBACK_MSG, [], web_links, False

        sources = [r["title"] for r in web_results if r.get("title")][:3]

        question.answer = answer
        question.source_refs = sources
        logger.info(
            "Generated answer for: %s (%d web pages)",
            question.text[:60], len(web_results),
        )
        return answer, sources, web_links, True
