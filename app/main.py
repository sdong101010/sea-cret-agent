"""FastAPI backend -- WebSocket endpoints, session management, wires everything together."""

import asyncio
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from app.audio import AudioCapture
from app.transcriber import Transcriber
from app.question_detector import QuestionDetector, DetectedQuestion, QuestionThread
from app.answer_engine import AnswerEngine
from app.summary import SummaryGenerator
from app.speaker import SpeakerIdentifier

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

session_active = False
calibrating = False
calibration_audio: list = []
connected_clients: set[WebSocket] = set()
transcribe_task: asyncio.Task | None = None


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


def on_audio_chunk(audio_data):
    """Callback from the audio capture -- feeds data to the transcriber."""
    if calibrating:
        calibration_audio.append(audio_data.copy())
    else:
        transcriber.add_audio(audio_data)


async def transcription_loop():
    """Continuously transcribe buffered audio and detect questions.

    Uses thought-completion buffering: question detection only runs after
    a pause in speech, so we don't analyze incomplete sentences.
    Fast-path: if the latest segment ends with "?", skip the pause
    and run detection immediately — the question is already complete.
    """
    loop = asyncio.get_running_loop()
    last_speech_time = 0.0
    detection_pending = False

    while session_active:
        try:
            segments, raw_audio = await transcriber.transcribe_buffer()

            if segments and raw_audio is not None and speaker_id.is_calibrated:
                buf_duration = len(raw_audio) / config.AUDIO_SAMPLE_RATE
                offset = transcriber._total_processed - buf_duration
                seg_ranges = [
                    {"start": s.start_time - offset, "end": s.end_time - offset}
                    for s in segments
                ]
                labels = await loop.run_in_executor(
                    None, speaker_id.identify_batch, raw_audio, seg_ranges
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

            has_obvious_question = (
                segments and segments[-1].text.strip().endswith("?")
            )

            if segments:
                last_speech_time = time.time()
                detection_pending = True

            pause_elapsed = time.time() - last_speech_time if last_speech_time > 0 else 0
            should_detect = detection_pending and (
                has_obvious_question or pause_elapsed >= config.THOUGHT_PAUSE_SECONDS
            )

            if should_detect:
                detection_pending = False
                window = transcriber.get_recent_conversation(exclude_speaker="You")
                if window.strip():
                    results = await detector.detect(window)

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

                        asyncio.create_task(answer_question(q, thread))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error in transcription loop")

        await asyncio.sleep(0.25)


async def answer_question(question: DetectedQuestion, thread: QuestionThread):
    """Retrieve an answer and broadcast it."""
    answer_text, sources, urls, is_confident = await answer_engine.get_answer(question)
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
            "calibrated": speaker_id.is_calibrated,
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
    global session_active, audio_capture, transcribe_task, calibrating

    action = msg.get("action")

    if action == "start":
        if session_active:
            return
        logger.info("Starting meeting session...")
        await transcriber.initialize()
        audio_capture = AudioCapture(on_audio_chunk=on_audio_chunk)
        await audio_capture.start()
        session_active = True
        transcribe_task = asyncio.create_task(transcription_loop())
        await broadcast({"type": "session_started", "calibrated": speaker_id.is_calibrated})
        logger.info("Meeting session active")

    elif action == "stop":
        if not session_active:
            return
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
        summary = await summary_gen.generate(segments, questions, msg.get("title"))

        await broadcast({"type": "session_stopped", "summary": summary})

        transcriber.clear()
        detector.clear()
        logger.info("Session stopped, summary generated")

    elif action == "calibrate_start":
        if not session_active or not audio_capture:
            await broadcast({"type": "calibrate_error", "message": "Start a session first"})
            return
        logger.info("Starting voice calibration -- speak for 3 seconds...")
        calibration_audio.clear()
        calibrating = True
        await broadcast({"type": "calibrating"})

    elif action == "calibrate_stop":
        calibrating = False
        if not calibration_audio:
            await broadcast({"type": "calibrate_error", "message": "No audio captured"})
            return
        audio = np.concatenate(calibration_audio)
        calibration_audio.clear()
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(None, speaker_id.calibrate, audio)
        if success:
            logger.info("Voice calibration succeeded")
            await broadcast({"type": "calibrated"})
        else:
            logger.warning("Voice calibration failed -- audio too short")
            await broadcast({"type": "calibrate_error", "message": "Audio too short, try again"})



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
