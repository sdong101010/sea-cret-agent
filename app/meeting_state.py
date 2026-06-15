"""Live meeting state -- a continuously evolving understanding of the meeting.

This module owns the "what's the meeting about so far?" view that gets passed
to both the question detector (so it can resolve pronouns and judge whether
something is genuinely a new question) and the answer engine (so its answers
are grounded in the actual meeting context, not generic).

Design:

  - We don't try to keep a perfect verbatim transcript in the LLM call --
    that gets expensive and dilutes attention. Instead we maintain a SHORT
    structured summary that gets updated as the meeting progresses.
  - Updates are debounced: we only refresh the state every UPDATE_INTERVAL
    seconds OR after N new segments accumulate, whichever comes first.
  - Updates run in the background so they never block question detection.
  - When a question fires, the detector and answer engine both consume the
    last-known state -- so an answer can lag behind reality by up to
    UPDATE_INTERVAL, which is fine.

The state is a small dict the LLM produces, NOT free text. Structured so we
can show it in the UI, drop into prompts, and persist with the summary.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from app import llm

logger = logging.getLogger(__name__)


# Update cadence. The state is a "diff in our heads" of what's been said --
# 30s feels right for normal meeting pace.
UPDATE_INTERVAL_SECONDS = 30.0
# Or after this many new segments, refresh sooner (rapid back-and-forth).
UPDATE_AFTER_N_SEGMENTS = 12


UPDATE_PROMPT = """\
You are a silent observer in a live meeting, building a running understanding \
of what's happening. Update the meeting state below using the new transcript \
segments. Be concise; this is your working memory, not a transcript.

CURRENT STATE:
{current_state}

NEW TRANSCRIPT SEGMENTS (chronological, speaker-labeled):
{new_segments}

Update the state. Rules:
- "topic": one sentence. What is this meeting about? (E.g. "Customer evaluating \
Data Cloud for KYC document processing on 10M user base.")
- "participants": dict of speaker label to a short note about them \
(e.g. {{"Me": "the SE/seller", "Speaker 1": "customer architect, asks technical \
questions"}}). Add new speakers as they appear; don't drop existing ones.
- "key_facts": short bullet list of things established so far -- numbers, \
products in use, requirements, constraints. Cap at 12 most relevant; drop \
older items if list grows.
- "open_threads": short bullet list of topics the meeting is currently \
working through but hasn't concluded. Drop ones that have been resolved.
- "recent_focus": one sentence about what the conversation is on RIGHT NOW \
(based on the newest segments).

If the meeting just started and there's not enough context yet, say so in \
the relevant fields ("Not yet established") rather than guessing.

Return ONLY valid JSON in this exact shape:
{{
  "topic": "<one sentence>",
  "participants": {{"<speaker label>": "<short note>"}},
  "key_facts": ["<fact>", ...],
  "open_threads": ["<thread>", ...],
  "recent_focus": "<one sentence>"
}}
"""


INITIAL_STATE = {
    "topic": "Not yet established -- meeting just started.",
    "participants": {},
    "key_facts": [],
    "open_threads": [],
    "recent_focus": "Listening for the first speaker.",
}


@dataclass
class MeetingState:
    """Owns the running state and the update lock."""

    state: dict = field(default_factory=lambda: dict(INITIAL_STATE))
    _last_update_time: float = 0.0
    _segments_since_update: int = 0
    _updating: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def reset(self):
        self.state = dict(INITIAL_STATE)
        self._last_update_time = 0.0
        self._segments_since_update = 0

    def note_new_segments(self, n: int):
        self._segments_since_update += n

    def needs_update(self) -> bool:
        if self._updating:
            return False
        if self._segments_since_update == 0:
            return False
        if self._segments_since_update >= UPDATE_AFTER_N_SEGMENTS:
            return True
        elapsed = time.time() - self._last_update_time
        return elapsed >= UPDATE_INTERVAL_SECONDS

    async def update(self, new_segments_text: str):
        """Refresh state given new transcript text. Safe to await concurrently
        with detection -- we hold a lock so only one update runs at a time."""
        async with self._lock:
            if self._updating:
                return
            self._updating = True

        try:
            current_state_str = json.dumps(self.state, indent=2)
            prompt = UPDATE_PROMPT.format(
                current_state=current_state_str,
                new_segments=new_segments_text,
            )
            raw = await llm.generate(
                prompt,
                max_tokens=1500,
                json_mode=True,
                timeout=45.0,
            )
            if not raw:
                logger.warning("Meeting state update: empty response")
                return

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Meeting state update: bad JSON %r", raw[:200])
                return

            # Merge: keep existing participants if the model omits them.
            new_participants = parsed.get("participants") or {}
            merged_participants = dict(self.state.get("participants") or {})
            merged_participants.update(new_participants)

            self.state = {
                "topic": parsed.get("topic") or self.state["topic"],
                "participants": merged_participants,
                "key_facts": parsed.get("key_facts") or [],
                "open_threads": parsed.get("open_threads") or [],
                "recent_focus": parsed.get("recent_focus") or self.state["recent_focus"],
            }
            self._last_update_time = time.time()
            self._segments_since_update = 0
            logger.info(
                "Meeting state updated: topic=%r facts=%d threads=%d",
                self.state["topic"][:80],
                len(self.state["key_facts"]),
                len(self.state["open_threads"]),
            )
        finally:
            self._updating = False

    def context_block(self) -> str:
        """Render the state as a block of text suitable for inclusion in a prompt."""
        s = self.state
        parts = [
            f"Meeting topic: {s.get('topic', '')}",
        ]
        participants = s.get("participants") or {}
        if participants:
            lines = [f"  - {label}: {note}" for label, note in participants.items()]
            parts.append("Participants:\n" + "\n".join(lines))
        facts = s.get("key_facts") or []
        if facts:
            parts.append("Key facts established so far:\n" + "\n".join(f"  - {f}" for f in facts))
        threads = s.get("open_threads") or []
        if threads:
            parts.append("Open threads (currently being worked through):\n" + "\n".join(f"  - {t}" for t in threads))
        focus = s.get("recent_focus") or ""
        if focus:
            parts.append(f"Right now the conversation is focused on: {focus}")
        return "\n\n".join(parts)
