"""Streaming question detection.

Different from the old periodic-scan model: we look at NEW segments only,
one detection pass per "thought boundary" (a pause or a question mark in
the latest segment), and use the running meeting state to decide whether
the speaker just asked a real question that needs an answer.

Public API kept compatible with main.py:
  - detect(new_lines, meeting_state, last_segment_speaker) -> list[(q, thread)]
  - update_thread_answer / get_all_threads / get_all_questions / clear
  - DetectedQuestion / QuestionThread dataclasses

Key behaviors:
  - We pass the FULL meeting context (state + the new segments) to the LLM,
    not just a 60s rolling window. The model judges "is this a real question
    that needs research?" with the whole picture.
  - We do NOT dedup against keyword overlap any more. Dedup is now handled
    by treating new questions on the same topic as follow-ups within an
    existing thread (the LLM decides this).
  - "Why now": every detected question carries a one-line rationale we feed
    into the answer engine so it knows the speaker's intent.
"""

import json
import logging
import time
from dataclasses import dataclass, field

import config
from app import llm
from app.meeting_state import MeetingState

logger = logging.getLogger(__name__)


DETECTION_PROMPT = """\
You are a silent expert listening to a live meeting. Your job: spot when \
someone has just asked a real question that needs a substantive technical \
answer, so research can start in the background.

You receive:
1. A snapshot of the meeting state (what's been established so far).
2. The newest transcript segments since you last looked.

Decide whether the newest segments contain one or more questions that warrant \
researching an answer. Be discerning but not stingy -- if someone asks a \
genuine question, surface it. Ignore filler ("right?", "you know?"), \
acknowledgements, and rhetorical asides.

A QUESTION qualifies if:
- It's a complete, coherent thought (not cut off mid-sentence)
- It's asking about a product capability, technical detail, requirement, \
limitation, pricing/commercial detail, or implementation concern
- It's NOT already answered by something said earlier in the meeting

Use the meeting state to RESOLVE references. If the speaker says "Will that \
work?" and the meeting just discussed sending KYC documents to Data Cloud, \
the resolved question is about Data Cloud handling KYC document images.

For each genuine question found, also produce:
- raw_text: the exact words spoken
- resolved_text: the question rewritten to be self-contained (someone reading \
ONLY this should understand exactly what's being asked, given the meeting \
context)
- search_query: a query that would find authoritative information about it
- topic: 2-4 word topic label
- why_now: ONE sentence on why the speaker likely asked this -- their intent \
or motivation, given the meeting so far. (E.g. "Speaker is probing whether \
encryption is sufficient for their regulated KYC workflow.")
- followup_thread_id: if this is a follow-up to one of the existing threads \
listed below, set to that thread's id; otherwise null.
- confidence: 0.0-1.0

EXISTING QUESTION THREADS (use followup_thread_id to link follow-ups):
{threads}

MEETING STATE:
{meeting_state}

NEWEST TRANSCRIPT SEGMENTS (chronological, speaker-labeled):
{new_segments}

Return ONLY valid JSON:
{{"questions": [
  {{
    "raw_text": "...", "resolved_text": "...", "search_query": "...",
    "topic": "...", "why_now": "...",
    "followup_thread_id": <id or null>, "confidence": <0.0-1.0>
  }}
]}}

If there are no genuine questions in the newest segments, return: {{"questions": []}}
"""


@dataclass
class DetectedQuestion:
    text: str
    raw_text: str
    topic: str
    confidence: float
    search_query: str = ""
    why_now: str = ""
    thread_id: int = 0
    is_followup: bool = False
    timestamp: float = field(default_factory=time.time)
    answer: str | None = None
    source_refs: list[str] = field(default_factory=list)


@dataclass
class QuestionThread:
    id: int
    topic: str
    primary_question: str
    questions: list[DetectedQuestion] = field(default_factory=list)
    answer: str | None = None
    source_refs: list[str] = field(default_factory=list)
    urls: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    suppressed: bool = False
    timestamp: float = field(default_factory=time.time)


class QuestionDetector:
    def __init__(self):
        self._threads: list[QuestionThread] = []
        self._all_questions: list[DetectedQuestion] = []
        self._next_thread_id: int = 0

    def _format_threads(self) -> str:
        if not self._threads:
            return "  (none yet)"
        lines = []
        for t in self._threads[-8:]:
            lines.append(f'  Thread {t.id} [{t.topic}]: "{t.primary_question}"')
        return "\n".join(lines)

    async def detect(
        self,
        new_segments_text: str,
        meeting_state: MeetingState,
    ) -> list[tuple[DetectedQuestion, QuestionThread]]:
        """Look for questions in the newest segments, given the meeting state."""
        if not new_segments_text.strip():
            return []

        prompt = DETECTION_PROMPT.format(
            threads=self._format_threads(),
            meeting_state=meeting_state.context_block(),
            new_segments=new_segments_text,
        )

        raw = await llm.generate(
            prompt,
            max_tokens=2048,
            json_mode=True,
            timeout=45.0,
        )
        if not raw:
            return []

        try:
            parsed = json.loads(raw)
            questions_data = parsed.get("questions", [])
        except json.JSONDecodeError:
            logger.warning("Failed to parse detection response: %r", raw[:200])
            return []

        results: list[tuple[DetectedQuestion, QuestionThread]] = []
        for qd in questions_data:
            raw_text = (qd.get("raw_text") or "").strip()
            resolved_text = (qd.get("resolved_text") or raw_text).strip()
            confidence = float(qd.get("confidence", 0.0))
            topic = qd.get("topic") or "general"
            why_now = (qd.get("why_now") or "").strip()
            followup_tid = qd.get("followup_thread_id")
            search_query = (qd.get("search_query") or "").strip() or resolved_text

            if not resolved_text:
                continue
            if confidence < config.QUESTION_CONFIDENCE_THRESHOLD:
                logger.info(
                    "Rejected (confidence %.2f < %.2f): %s",
                    confidence,
                    config.QUESTION_CONFIDENCE_THRESHOLD,
                    resolved_text[:80],
                )
                continue

            is_followup = False
            thread = None
            if followup_tid is not None:
                thread = next((t for t in self._threads if t.id == followup_tid), None)
                if thread:
                    is_followup = True

            if thread is None:
                thread_id = self._next_thread_id
                self._next_thread_id += 1
                thread = QuestionThread(
                    id=thread_id,
                    topic=topic,
                    primary_question=resolved_text,
                    confidence=confidence,
                )
                self._threads.append(thread)
            else:
                if confidence > thread.confidence:
                    thread.confidence = confidence
                    thread.topic = topic

            q = DetectedQuestion(
                text=resolved_text,
                raw_text=raw_text,
                topic=topic,
                confidence=confidence,
                search_query=search_query,
                why_now=why_now,
                thread_id=thread.id,
                is_followup=is_followup,
            )
            thread.questions.append(q)
            self._all_questions.append(q)
            results.append((q, thread))

            logger.info(
                'Detected %s (%.0f%%): "%s" [thread %d] -- why: %s',
                "follow-up" if is_followup else "new",
                confidence * 100,
                resolved_text[:80],
                thread.id,
                why_now[:120],
            )

        return results

    def get_all_questions(self) -> list[DetectedQuestion]:
        return list(self._all_questions)

    def get_all_threads(self) -> list[QuestionThread]:
        return list(self._threads)

    def get_thread(self, thread_id: int) -> QuestionThread | None:
        return next((t for t in self._threads if t.id == thread_id), None)

    def update_thread_answer(
        self, thread_id: int, answer: str, sources: list[str],
        urls: list[dict], suppressed: bool = False,
    ):
        thread = self.get_thread(thread_id)
        if thread:
            thread.answer = answer
            thread.source_refs = sources
            thread.urls = urls
            thread.suppressed = suppressed
            for q in thread.questions:
                q.answer = answer
                q.source_refs = sources

    def clear(self):
        self._threads.clear()
        self._all_questions.clear()
        self._next_thread_id = 0
