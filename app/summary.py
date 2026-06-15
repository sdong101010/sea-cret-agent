"""Generate post-meeting summary with Q&A pairs and action items."""

import logging
from datetime import datetime

import config
from app.question_detector import DetectedQuestion
from app.transcriber import TranscriptSegment
from app import llm

logger = logging.getLogger(__name__)

ACTION_ITEM_PROMPT = """\
Extract action items from this meeting transcript. Look for phrases like \
"we'll follow up", "let me get back to you", "I'll send", "action item", \
"next steps", "to-do", "we need to", etc.

Return a simple numbered list of action items. If none found, return "None identified."

TRANSCRIPT:
{transcript}

ACTION ITEMS:"""


class SummaryGenerator:
    async def generate(
        self,
        segments: list[TranscriptSegment],
        questions: list[DetectedQuestion],
        meeting_title: str | None = None,
    ) -> str:
        """Build a Markdown summary of the meeting."""
        now = datetime.now()
        title = meeting_title or f"Meeting {now.strftime('%Y-%m-%d %H:%M')}"
        full_transcript = " ".join(s.text for s in segments)

        action_items = await self._extract_action_items(full_transcript)

        answered = [q for q in questions if q.answer and "didn't surface enough" not in q.answer]
        unanswered = [q for q in questions if not q.answer or "didn't surface enough" in q.answer]

        lines = [
            f"# {title}",
            f"**Date:** {now.strftime('%A, %B %d, %Y at %H:%M')}",
            f"**Duration:** {self._format_duration(segments)}",
            f"**Questions detected:** {len(questions)} ({len(answered)} answered, {len(unanswered)} gaps)",
            "",
            "---",
            "",
        ]

        if answered:
            lines.append("## Questions & Answers")
            lines.append("")
            for i, q in enumerate(answered, 1):
                lines.append(f"### Q{i}: {q.text}")
                lines.append(f"**Topic:** {q.topic}  ")
                lines.append(f"**Confidence:** {q.confidence:.0%}")
                lines.append("")
                lines.append(f"> {q.answer}")
                lines.append("")
                if q.source_refs:
                    lines.append("**Sources:** " + ", ".join(q.source_refs))
                lines.append("")

        if unanswered:
            lines.append("## Unanswered / Low-Confidence Questions")
            lines.append("")
            for q in unanswered:
                lines.append(f"- **{q.text}** (topic: {q.topic})")
            lines.append("")

        lines.append("## Action Items")
        lines.append("")
        lines.append(action_items)
        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## Full Transcript")
        lines.append("")
        for seg in segments:
            minutes = int(seg.start_time // 60)
            seconds = int(seg.start_time % 60)
            speaker = seg.speaker or "Unknown"
            lines.append(f"**[{minutes:02d}:{seconds:02d}]** [{speaker}] {seg.text}")
        lines.append("")

        content = "\n".join(lines)

        filename = f"{now.strftime('%Y%m%d_%H%M')}_{title.replace(' ', '_')[:40]}.md"
        filepath = config.SUMMARIES_DIR / filename
        filepath.write_text(content, encoding="utf-8")
        logger.info("Summary saved to %s", filepath)

        return content

    async def _extract_action_items(self, transcript: str) -> str:
        if len(transcript.strip()) < 50:
            return "None identified."

        trimmed = transcript[:4000]
        prompt = ACTION_ITEM_PROMPT.format(transcript=trimmed)

        result = await llm.generate(prompt, temperature=0.2, max_tokens=300)
        return result or "Action item extraction failed -- review transcript manually."

    def _format_duration(self, segments: list[TranscriptSegment]) -> str:
        if not segments:
            return "Unknown"
        total_seconds = segments[-1].end_time - segments[0].start_time
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"
