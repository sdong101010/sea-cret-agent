"""FastAPI backend -- WebSocket endpoints, session management, wires everything together."""

import asyncio
import json
import logging
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from app.audio import AudioCapture
from app.transcriber import Transcriber, TranscriptSegment
from app.question_detector import QuestionDetector, DetectedQuestion, QuestionThread
from app.answer_engine import AnswerEngine
from app.summary import SummaryGenerator
from app.speaker import SpeakerIdentifier
from app.meeting_state import MeetingState
from app import google_doc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

transcriber = Transcriber()
detector = QuestionDetector()
answer_engine = AnswerEngine()
summary_gen = SummaryGenerator()
speaker_id = SpeakerIdentifier()
meeting_state = MeetingState()

session_active = False
connected_clients: set[WebSocket] = set()
transcribe_task: asyncio.Task | None = None
state_update_task: asyncio.Task | None = None
# Pending segments not yet seen by the detector. Drained on each thought
# boundary (pause or question mark in the latest segment).
unseen_segments: list[TranscriptSegment] = []
# The most recent summary text and title, kept around so the user can
# trigger a Google Doc export after they see it on screen.
last_summary_md: str = ""
last_meeting_title: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    speaker_id.load_model()
    logger.info("Startup complete. Open http://localhost:8765 in your browser.")
    yield
    logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent.parent / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent.parent / "static" / "index.html")


SUMMARIES_DIR = Path(__file__).parent.parent / "data" / "summaries"


@app.get("/api/summaries")
async def list_summaries():
    """Return a list of past meeting summaries, newest first."""
    if not SUMMARIES_DIR.exists():
        return {"summaries": []}
    items = []
    for p in sorted(SUMMARIES_DIR.glob("*.md"), reverse=True):
        # Filename pattern: YYYYMMDD_HHMM_<title-with-underscores>.md
        stem = p.stem
        try:
            datestamp, timestamp, *title_parts = stem.split("_")
            display_title = " ".join(title_parts).replace(":", "")
            display_when = f"{datestamp[:4]}-{datestamp[4:6]}-{datestamp[6:8]} {timestamp[:2]}:{timestamp[2:]}"
        except ValueError:
            display_title = stem
            display_when = ""
        items.append({
            "file": p.name,
            "title": display_title or stem,
            "when": display_when,
        })
    return {"summaries": items}


@app.get("/api/summaries/{filename}")
async def read_summary(filename: str):
    """Return the raw markdown of a past summary. Filename must be a basename
    inside data/summaries/ — no path traversal."""
    safe = Path(filename).name
    p = SUMMARIES_DIR / safe
    if not p.exists() or not p.is_file() or p.suffix != ".md":
        return {"error": "Not found"}
    return {"file": safe, "markdown": p.read_text()}


@app.get("/api/summaries/{filename}/structured")
async def read_summary_structured(filename: str):
    """Parse a saved summary back into the live-session shape: ordered Q&A
    threads plus timestamped transcript lines. Lets the UI rehydrate a past
    meeting into the same two-panel layout the user saw when it ended."""
    safe = Path(filename).name
    p = SUMMARIES_DIR / safe
    if not p.exists() or not p.is_file() or p.suffix != ".md":
        return {"error": "Not found"}
    md = p.read_text()
    return {"file": safe, **_parse_summary_markdown(md)}


_Q_HEADER_RE = re.compile(r"^### Q(\d+):\s*(.+?)\s*$")
_TOPIC_RE = re.compile(r"^\*\*Topic:\*\*\s*(.+?)\s*$")
_CONF_RE = re.compile(r"^\*\*Confidence:\*\*\s*(\d+)%\s*$")
_SOURCES_RE = re.compile(r"^\*\*Sources:\*\*\s*(.+?)\s*$")
_TRANSCRIPT_LINE_RE = re.compile(r"^\*\*\[(\d{2}):(\d{2})\]\*\*\s*(?:\[([^\]]+)\]\s*)?(.*)$")


def _parse_summary_markdown(md: str) -> dict:
    """Walk the markdown produced by SummaryGenerator and pull out the parts
    the UI needs. We split on top-level `## ` sections so each block is small.

    Returns: {title, threads: [...], transcript: [...]} — same field names the
    websocket uses for live messages, so the frontend can reuse renderAnswer /
    appendTranscript without a second code path.
    """
    lines = md.splitlines()
    title = ""
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    sections: dict[str, list[str]] = {}
    current = None
    for line in lines:
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)

    threads = []
    qa_block = sections.get("Questions & Answers", [])
    # Split into per-question chunks at each `### Q` header.
    chunks: list[list[str]] = []
    current_chunk: list[str] | None = None
    for line in qa_block:
        if _Q_HEADER_RE.match(line):
            if current_chunk is not None:
                chunks.append(current_chunk)
            current_chunk = [line]
        elif current_chunk is not None:
            current_chunk.append(line)
    if current_chunk is not None:
        chunks.append(current_chunk)

    for idx, chunk in enumerate(chunks, 1):
        header_match = _Q_HEADER_RE.match(chunk[0])
        question_text = header_match.group(2) if header_match else ""
        topic = ""
        confidence = 0.0
        sources: list[str] = []
        # Answer body is everything between the metadata lines and **Sources:**.
        # The summary writer only prefixes the FIRST bullet with `> `, so we
        # can't use the `>` prefix as a body filter — we'd lose continuation
        # bullets and the trailing `_Confidence note:_` paragraph. Instead,
        # treat any non-metadata line as body and just trim the leading `>`
        # if present.
        body_lines: list[str] = []
        in_body = False
        for line in chunk[1:]:
            if (m := _TOPIC_RE.match(line)):
                topic = m.group(1)
                continue
            if (m := _CONF_RE.match(line)):
                confidence = int(m.group(1)) / 100.0
                continue
            if (m := _SOURCES_RE.match(line)):
                sources = [s.strip() for s in m.group(1).split(",") if s.strip()]
                break
            if line.startswith("> "):
                body_lines.append(line[2:])
                in_body = True
            elif line.startswith(">"):
                body_lines.append(line[1:].lstrip())
                in_body = True
            elif in_body or line.strip():
                # Continuation lines (no `>` prefix) and the confidence-note
                # paragraph belong to the answer body.
                body_lines.append(line)
                in_body = True
        answer = "\n".join(body_lines).strip()
        threads.append({
            "thread_id": f"past-{idx}",
            "text": question_text,
            "topic": topic or "general",
            "confidence": confidence,
            "answer": answer,
            "sources": sources,
            "urls": [],
            "suppressed": False,
        })

    transcript = []
    for line in sections.get("Full Transcript", []):
        m = _TRANSCRIPT_LINE_RE.match(line)
        if not m:
            continue
        mins, secs = int(m.group(1)), int(m.group(2))
        speaker = m.group(3) or "Unknown"
        text = m.group(4)
        start = mins * 60 + secs
        transcript.append({
            "text": text,
            "start_time": start,
            "end_time": start,
            "speaker": speaker,
        })

    return {"title": title, "threads": threads, "transcript": transcript}


async def broadcast(message: dict):
    """Send a message to all connected WebSocket clients."""
    payload = json.dumps(message)
    disconnected = set()
    for ws in connected_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            disconnected.add(ws)
    connected_clients.difference_update(disconnected)


def on_audio_chunk(mixed, mic, system):
    """Callback from the audio capture -- feeds data to the transcriber.

    The transcriber sends `mixed` to the speech sidecar and stores the
    separate `mic`/`system` streams in rolling buffers for speaker ID.
    """
    transcriber.add_audio(mixed, mic, system)


def _format_segments_for_prompt(segments: list[TranscriptSegment]) -> str:
    """Render segments as `[Speaker] text` lines, the format the LLM expects."""
    lines = []
    for s in segments:
        label = s.speaker if s.speaker not in ("", "Unknown") else "Speaker"
        lines.append(f"[{label}] {s.text}")
    return "\n".join(lines)


async def transcription_loop():
    """Continuously transcribe audio, label speakers, and stream new segments
    to question detection at thought boundaries.

    Thought boundary = (a pause longer than THOUGHT_PAUSE_SECONDS) OR
    (the latest segment ends with "?"). At that moment, all unseen segments
    are handed to the detector along with the current meeting state.

    Meeting-state updates happen in the background whenever the state's
    needs_update() returns True (every ~30s or every N new segments).
    """
    global unseen_segments
    loop = asyncio.get_running_loop()
    last_speech_time = 0.0
    detection_pending = False

    while session_active:
        try:
            segments, mic_audio, system_audio = await transcriber.transcribe_buffer()

            if segments and (mic_audio is not None or system_audio is not None):
                ref_audio = system_audio if system_audio is not None else mic_audio
                buf_duration = len(ref_audio) / config.AUDIO_SAMPLE_RATE
                offset = transcriber._total_processed - buf_duration
                seg_ranges = [
                    {"start": s.start_time - offset, "end": s.end_time - offset}
                    for s in segments
                ]
                labels = await loop.run_in_executor(
                    None, speaker_id.identify_batch, mic_audio, system_audio, seg_ranges
                )
                for seg, label in zip(segments, labels):
                    seg.speaker = label

            for seg in segments:
                await broadcast({
                    "type": "transcript",
                    "text": seg.text,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "speaker": seg.speaker,
                })

            if segments:
                unseen_segments.extend(segments)
                meeting_state.note_new_segments(len(segments))
                last_speech_time = time.time()
                detection_pending = True

            has_obvious_question = (
                unseen_segments and unseen_segments[-1].text.strip().endswith("?")
            )
            pause_elapsed = time.time() - last_speech_time if last_speech_time > 0 else 0
            should_detect = detection_pending and unseen_segments and (
                has_obvious_question or pause_elapsed >= config.THOUGHT_PAUSE_SECONDS
            )

            if should_detect:
                detection_pending = False
                # Snapshot + clear the unseen buffer atomically.
                batch = unseen_segments
                unseen_segments = []

                new_text = _format_segments_for_prompt(batch)
                primary_speaker = batch[-1].speaker if batch else "Unknown"

                results = await detector.detect(new_text, meeting_state)
                for q, thread in results:
                    await broadcast({
                        "type": "question_update" if q.is_followup else "question",
                        "thread_id": thread.id,
                        "text": q.text,
                        "raw_text": q.raw_text,
                        "topic": thread.topic,
                        "confidence": thread.confidence,
                        "is_followup": q.is_followup,
                        "status": "searching",
                    })
                    asyncio.create_task(answer_question(q, thread, primary_speaker))

            # Background: refresh meeting state when warranted.
            if meeting_state.needs_update():
                window_text = transcriber.get_recent_conversation()
                if window_text.strip():
                    asyncio.create_task(meeting_state.update(window_text))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error in transcription loop")

        await asyncio.sleep(0.25)


async def answer_question(
    question: DetectedQuestion,
    thread: QuestionThread,
    speaker_label: str = "someone in the meeting",
):
    """Retrieve a meeting-grounded answer and broadcast it."""
    answer_text, sources, urls, is_confident = await answer_engine.get_answer(
        question, meeting_state, speaker_label=speaker_label,
    )
    suppressed = not is_confident
    detector.update_thread_answer(thread.id, answer_text, sources, urls, suppressed)
    await broadcast({
        "type": "answer",
        "thread_id": thread.id,
        "question": question.text,
        "answer": answer_text,
        "sources": sources,
        "urls": urls,
        "confidence": question.confidence,
        "topic": thread.topic,
        "suppressed": suppressed,
    })


audio_capture: AudioCapture | None = None


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    logger.info("Client connected (%d total)", len(connected_clients))

    try:
        await ws.send_text(json.dumps({
            "type": "status",
            "session_active": session_active,
            "speakers": speaker_id.get_speakers(),
            "threads": [
                {
                    "thread_id": t.id,
                    "text": t.primary_question,
                    "topic": t.topic,
                    "confidence": t.confidence,
                    "answer": t.answer,
                    "sources": t.source_refs,
                    "urls": t.urls,
                    "suppressed": t.suppressed,
                }
                for t in detector.get_all_threads()
            ],
        }))

        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            await handle_client_message(msg)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        connected_clients.discard(ws)
        logger.info("Client disconnected (%d remaining)", len(connected_clients))


async def handle_client_message(msg: dict):
    global session_active, audio_capture, transcribe_task

    action = msg.get("action")

    if action == "start":
        if session_active:
            return
        logger.info("Starting meeting session...")
        speaker_id.reset()
        meeting_state.reset()
        unseen_segments.clear()
        await transcriber.initialize()
        audio_capture = AudioCapture(on_audio_chunk=on_audio_chunk)
        await audio_capture.start()
        session_active = True
        transcribe_task = asyncio.create_task(transcription_loop())
        await broadcast({"type": "session_started", "mic_enabled": audio_capture.mic_enabled})
        logger.info("Meeting session active")

    elif action == "stop":
        if not session_active:
            return
        global last_summary_md, last_meeting_title
        logger.info("Stopping meeting session...")
        session_active = False
        if transcribe_task:
            transcribe_task.cancel()
            try:
                await transcribe_task
            except asyncio.CancelledError:
                pass
        if audio_capture:
            await audio_capture.stop()

        segments = transcriber.get_all_segments()
        questions = detector.get_all_questions()
        meeting_title = msg.get("title")
        summary = await summary_gen.generate(segments, questions, meeting_title)
        last_summary_md = summary
        last_meeting_title = meeting_title

        await broadcast({"type": "session_stopped", "summary": summary})

        transcriber.clear()
        transcriber.shutdown()
        detector.clear()
        logger.info("Session stopped, summary generated")

    elif action == "create_google_doc":
        # Two paths: live session uses last_summary_md; reopened past sessions
        # pass `filename` so we read the saved markdown straight off disk.
        summary_md = last_summary_md
        title_for_doc = last_meeting_title
        filename = (msg.get("filename") or "").strip()
        if filename:
            safe = Path(filename).name
            p = SUMMARIES_DIR / safe
            if p.exists() and p.is_file() and p.suffix == ".md":
                summary_md = p.read_text()
                # Strip the YYYYMMDD_HHMM_ prefix so the doc title reads cleanly.
                stem_parts = p.stem.split("_", 2)
                title_for_doc = stem_parts[2].replace("_", " ") if len(stem_parts) == 3 else p.stem
        if not summary_md:
            await broadcast({
                "type": "google_doc_result",
                "error": "No summary to export. Run a session first.",
            })
            return
        await broadcast({"type": "google_doc_started"})
        result = await google_doc.create_doc(summary_md, title_for_doc)
        if "error" in result:
            await broadcast({"type": "google_doc_result", "error": result["error"]})
        else:
            await broadcast({
                "type": "google_doc_result",
                "url": result["url"],
                "title": result["title"],
            })

    elif action == "rename_speaker":
        old = (msg.get("old") or "").strip()
        new = (msg.get("new") or "").strip()
        if not old or not new:
            return
        ok = speaker_id.rename_speaker(old, new)
        if ok:
            await broadcast({"type": "speaker_renamed", "old": old, "new": new})



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
