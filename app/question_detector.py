"""Detect and resolve customer questions from transcript using Gemini.

Two-stage pipeline:
  1. Detect: find questions in the transcript window
  2. Resolve: rewrite vague/follow-up questions into self-contained queries
     using full conversation context and existing question threads
"""

import json
import logging
import time
from dataclasses import dataclass, field

import config
from app import llm

logger = logging.getLogger(__name__)

DETECTION_PROMPT = """\
You are analyzing a live meeting transcript to find real customer questions that \
need a technical answer. Be VERY selective. Most transcript windows contain ZERO questions.

A REAL question:
- Has clear interrogative structure (who/what/when/where/why/how, or can/do/does/is/are/will)
- Asks about a product capability, technical detail, or implementation concern
- Is a complete, coherent thought (NOT cut off mid-sentence)

REJECT all of these (return empty list):
- Incomplete or cut-off sentences ("Can it process that and", "So if we were to")
- Statements, even if they mention a topic ("We would have encryption keys")
- Thinking aloud or rhetorical remarks ("So think about it as the self-service...")
- Social chat, filler, or greetings
- Anything where the speaker is explaining, not asking

CRITICAL: Follow-up questions like "Will that work?", "Can it do that?", "Is that possible?" \
are ONLY valid if you can fully resolve what "that"/"it"/"this" refers to from the conversation \
context. If you cannot determine the specific subject, REJECT the question.
{threads_context}
For each real question found:
1. raw_text: the exact words as spoken
2. resolved_text: rewrite into a FULLY self-contained question — replace ALL pronouns \
and vague references ("that", "it", "this", "those") with the specific subject from context. \
Someone reading ONLY the resolved_text must understand exactly what is being asked.
3. search_query: an optimized search query for web research
4. topic: 2-3 word topic label
5. confidence: 0.0-1.0
6. followup_thread_id: if this is a follow-up to an existing thread, set to that thread's id; otherwise null

GOOD examples:
- Raw: "Can you decrypt the data if there are utilities available?"
  Resolved: same (already self-contained)
- Raw: "Will that work?" (after discussion about sending scanned driver's license to Data Cloud)
  Resolved: "Can Salesforce Data Cloud process and extract fields from scanned document images like driver's licenses?"
- Raw: "How does it handle field-level encryption?"
  Resolved: "How does Salesforce handle field-level encryption?"

BAD examples (REJECT these):
- "Can it process that and" → incomplete sentence, cut off mid-thought
- "We would have the platform encryption keys" → statement, not a question
- "Will that work?" → REJECT if you cannot determine what "that" refers to

Return JSON: {{"questions": [{{"raw_text": "<exact words>", "resolved_text": "<self-contained rewrite>", "search_query": "<optimized query>", "topic": "<2-3 words>", "confidence": <0.0-1.0>, "followup_thread_id": <id or null>}}]}}
If nothing qualifies, return: {{"questions": []}}

TRANSCRIPT (recent conversation):
---
{transcript}
---

Return ONLY valid JSON."""

THREADS_CONTEXT = """
EXISTING QUESTION THREADS (use followup_thread_id to link follow-ups):
{threads}
"""


@dataclass
class DetectedQuestion:
    text: str
    raw_text: str
    topic: str
    confidence: float
    search_query: str = ""
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
        self._last_detection_time: float = 0
        self._min_interval: float = 3.0
        self._next_thread_id: int = 0

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        """Structural check: reject text that isn't actually a question."""
        t = text.strip()
        if t.endswith("?"):
            return True
        t_lower = t.lower()
        interrogatives = (
            "who ", "what ", "when ", "where ", "why ", "how ",
            "can ", "could ", "do ", "does ", "did ", "is ", "are ",
            "will ", "would ", "should ", "has ", "have ", "was ", "were ",
        )
        return t_lower.startswith(interrogatives)

    def _is_duplicate(self, resolved_text: str) -> bool:
        """Check if a question is too similar to a recently detected one."""
        resolved_lower = resolved_text.lower().strip("? ")
        for q in self._all_questions[-10:]:
            existing = q.text.lower().strip("? ")
            if resolved_lower in existing or existing in resolved_lower:
                return True
            words_new = set(resolved_lower.split())
            words_existing = set(existing.split())
            if len(words_new) > 3 and len(words_existing) > 3:
                overlap = len(words_new & words_existing) / max(len(words_new), len(words_existing))
                if overlap > 0.7:
                    return True
        return False

    def _build_threads_context(self) -> str:
        if not self._threads:
            return ""
        lines = []
        for t in self._threads[-5:]:
            lines.append(f'  Thread {t.id}: "{t.primary_question}" (topic: {t.topic})')
        return THREADS_CONTEXT.format(threads="\n".join(lines))

    async def detect(self, transcript_window: str) -> list[tuple[DetectedQuestion, QuestionThread]]:
        """Detect and resolve questions from a transcript window.

        Returns list of (question, thread) tuples. Each question has been
        resolved into a self-contained query. Follow-ups are linked to
        their parent thread.
        """
        now = time.time()
        if now - self._last_detection_time < self._min_interval:
            return []
        self._last_detection_time = now

        if len(transcript_window.strip()) < 20:
            return []

        threads_context = self._build_threads_context()
        prompt = DETECTION_PROMPT.format(
            transcript=transcript_window,
            threads_context=threads_context,
        )

        raw = await llm.generate(
            prompt, temperature=0.1, max_tokens=1024, json_mode=True,
            thinking_budget=0,
        )
        if not raw:
            return []

        try:
            parsed = json.loads(raw)
            questions_data = parsed.get("questions", [])
        except json.JSONDecodeError:
            logger.warning("Failed to parse question detection response: %s", raw[:200])
            return []

        results = []
        for qd in questions_data:
            raw_text = qd.get("raw_text", qd.get("text", "")).strip()
            resolved_text = qd.get("resolved_text", raw_text).strip()
            confidence = float(qd.get("confidence", 0.0))
            topic = qd.get("topic", "general")
            followup_tid = qd.get("followup_thread_id")

            if not resolved_text or confidence < config.QUESTION_CONFIDENCE_THRESHOLD:
                continue
            if not self._looks_like_question(raw_text) and not self._looks_like_question(resolved_text):
                logger.debug("Rejected (not interrogative): %s", raw_text[:80])
                continue
            if self._is_duplicate(resolved_text):
                continue

            search_query = qd.get("search_query", "").strip() or resolved_text

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
                thread_id=thread.id,
                is_followup=is_followup,
            )
            thread.questions.append(q)
            self._all_questions.append(q)
            results.append((q, thread))

            logger.info(
                'Detected %s (%.0f%%): "%s" -> "%s" [thread %d]',
                "follow-up" if is_followup else "question",
                confidence * 100, raw_text[:40], resolved_text[:60], thread.id,
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
