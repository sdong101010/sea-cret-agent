"""Answer engine -- shells out to `claude -p` with the slackbot research skill,
now grounded in the live meeting context.

The answer engine is told:
  1. Who is in the meeting and what they're discussing
  2. The exact question that just came up
  3. Why the speaker likely asked it (intent inferred by the detector)

So the answer comes back specific to the meeting, not a generic primer.
"""

import asyncio
import json
import logging
import re

from app.question_detector import DetectedQuestion
from app.meeting_state import MeetingState

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """\
You are an invisible silent expert sitting in on a live customer meeting. \
A question just came up that someone in the room (or on the call) needs an \
answer to RIGHT NOW. Your job: research it via Slack and the public web, \
then return a concise, MEETING-GROUNDED answer the live person can glance \
at and say naturally.

You are an expert in Agentforce, Data Cloud, MuleSoft, Informatica, and the \
Salesforce core platform.

RESEARCH METHODOLOGY:

1. Search Slack (via the Slack MCP tools) for internal announcements, \
enablement content, product updates, and prior discussions on this topic. \
Focus on messages from product, engineering, and enablement teams.
2. Search the public web for authoritative documentation: \
help.salesforce.com, Trailhead, Salesforce release notes, salesforce.com/blog, \
docs.mulesoft.com, Informatica docs, official Agentforce/Data Cloud product pages.

SOURCE QUALITY HIERARCHY (apply strictly, prefer higher tiers):
- Tier 1: Official Salesforce/MuleSoft/Informatica product documentation
- Tier 2: Official release notes and roadmap announcements
- Tier 3: Internal Slack announcements from product or enablement teams
- Tier 4: Trailhead and official learning content
- Tier 5: Official Salesforce blog posts
- Tier 6 (use only as a pointer, never as primary source): Community posts, \
unofficial blogs, third-party content

VALIDATION RULES:
- Only state something as fact if it is confirmed by at least one Tier 1-5 source.
- If the only matching Slack result is a message the user themselves posted, \
explicitly flag this as not independently validated.
- If sources conflict, surface the disagreement -- never silently pick one.
- If you cannot find an authoritative answer, say so directly. Do not guess.
- Every factual statement in your answer must trace back to a cited source.

GROUNDING RULES (THIS IS CRITICAL):
- The answer must be specific to THIS meeting's context. Don't give a generic \
explainer of the topic; tailor it to what the customer actually needs.
- Reference specifics from the meeting state when relevant (e.g. "given your \
10M-customer KYC use case..."). Do NOT invent specifics that aren't in the state.
- Use the speaker's intent ("why_now") to shape your framing -- if they're \
skeptical, address the skepticism head-on; if they're scoping, give scope-relevant \
detail.
- Frame architecturally and confidently. Lead with what the platform does. \
Never say "workaround". If a capability requires configuration, present it \
positively ("natively supported when configured" not "requires manual setup").

MEETING STATE:
{meeting_state}

WHO ASKED & WHY:
- Speaker: {speaker}
- Why they likely asked this (inferred): {why_now}

THE QUESTION:
{question}

OUTPUT FORMAT (return strictly valid JSON, nothing else):
{{
  "answer": "<3-6 short bullet points, each starting with • and 1-2 sentences. \
Use real Salesforce product names. Lead with the strongest, most directly \
relevant point for THIS customer's situation. If sources do not cover the \
topic well, say so and suggest follow-up.>",
  "sources": ["<short source title>", ...],
  "urls": [{{"title": "<source title>", "url": "<full URL>"}}, ...],
  "is_confident": true | false,
  "confidence_note": "<empty string if confident; otherwise explain why \
confidence is limited>"
}}
"""


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


FALLBACK_MSG = (
    "Couldn't reach the research agent in time. "
    "Consider addressing from your own expertise or offering to follow up."
)

CLAUDE_TIMEOUT_SECONDS = 180.0


class AnswerEngine:
    def __init__(self):
        logger.info("Answer engine initialized (claude -p, grounded in meeting context)")

    async def get_answer(
        self,
        question: DetectedQuestion,
        meeting_state: MeetingState,
        speaker_label: str = "someone in the meeting",
    ) -> tuple[str, list[str], list[dict], bool]:
        """Research the question and return a meeting-grounded answer."""
        query = question.text or question.search_query

        prompt = PROMPT_TEMPLATE.format(
            meeting_state=meeting_state.context_block(),
            speaker=speaker_label,
            why_now=question.why_now or "(not stated)",
            question=query,
        )

        cmd = [
            "claude",
            "--print",
            "--output-format", "json",
            "--no-session-persistence",
            "--permission-mode", "bypassPermissions",
            prompt,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=CLAUDE_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.warning("claude -p timed out after %.0fs for: %s",
                           CLAUDE_TIMEOUT_SECONDS, query[:80])
            return FALLBACK_MSG, [], [], False
        except FileNotFoundError:
            logger.error("`claude` CLI not found on PATH; cannot answer questions")
            return FALLBACK_MSG, [], [], False
        except Exception:
            logger.exception("claude -p invocation failed")
            return FALLBACK_MSG, [], [], False

        if proc.returncode != 0:
            logger.error("claude -p exited %d. stderr: %s",
                         proc.returncode, stderr.decode("utf-8", "replace")[:500])
            return FALLBACK_MSG, [], [], False

        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError:
            logger.error("claude -p envelope not JSON: %s", stdout[:300])
            return FALLBACK_MSG, [], [], False

        inner = envelope.get("result", "")
        data = _extract_json_object(inner) if isinstance(inner, str) else None
        if data is None:
            logger.info("claude returned prose, using as answer; question=%s", query[:60])
            prose = (inner or "").strip()
            return (prose or FALLBACK_MSG), [], [], bool(prose)

        answer = (data.get("answer") or "").strip()
        sources = data.get("sources") or []
        urls = data.get("urls") or []
        is_confident = bool(data.get("is_confident", False))
        confidence_note = (data.get("confidence_note") or "").strip()

        if confidence_note and not is_confident:
            answer = f"{answer}\n\n_Confidence note: {confidence_note}_"

        question.answer = answer
        question.source_refs = sources
        logger.info(
            "Answer ready for: %s (confident=%s, %d sources)",
            query[:60], is_confident, len(sources),
        )
        return answer, sources, urls, is_confident
