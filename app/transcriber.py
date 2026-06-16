"""Speech-to-text via the Apple SpeechAnalyzer sidecar (macOS 26+).

The Swift binary at bin/speech_sidecar reads 16-bit PCM @ 16kHz on stdin and
emits one JSONL line per recognized segment on stdout. We pipe the same
float32 audio buffers we capture from BlackHole into it (converted to Int16)
and parse the streamed results into TranscriptSegment objects.

A rolling 60-second audio buffer is kept so the speaker identifier can slice
the audio range for any segment it gets back.
"""

import asyncio
import json
import logging
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config
from app.whisper_worker import WhisperWorker

logger = logging.getLogger(__name__)


SIDECAR_PATH = Path(__file__).parent.parent / "bin" / "speech_sidecar"
BUILD_SCRIPT = Path(__file__).parent.parent / "bin" / "build_sidecar.sh"
ROLLING_BUFFER_SECONDS = 60


@dataclass
class TranscriptSegment:
    text: str
    start_time: float
    end_time: float
    speaker: str = "Unknown"
    timestamp: float = field(default_factory=time.time)
    is_final: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    whisper_final: bool = False
    apple_text: str = ""
    _whisper_event: asyncio.Event | None = field(default=None, repr=False, compare=False)

    @property
    def whisper_event(self) -> asyncio.Event:
        # Lazily create the event so it binds to whichever loop is running
        # when the worker / waiter first touches it.
        if self._whisper_event is None:
            self._whisper_event = asyncio.Event()
        return self._whisper_event


def _ensure_binary_built():
    """Build the Swift sidecar if it's missing."""
    if SIDECAR_PATH.exists():
        return
    logger.info("Speech sidecar binary missing, building...")
    subprocess.run([str(BUILD_SCRIPT)], check=True)


class Transcriber:
    def __init__(self, broadcast=None):
        # Default: no-op coroutine. Real broadcast is wired in by main.py.
        async def _noop_broadcast(_msg):
            return
        self._broadcast = broadcast or _noop_broadcast

        self._proc: subprocess.Popen | None = None
        self._stdout_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Pending finalized segments waiting to be polled by main.py.
        self._pending: list[TranscriptSegment] = []
        self._pending_lock = threading.Lock()

        # All finalized segments, in order.
        self._segments: list[TranscriptSegment] = []

        # Rolling float32 audio buffers for speaker ID. We keep mic and
        # system separate so we can detect "Me" via mic energy and run
        # voice embeddings on the system stream only (uncontaminated by the
        # user's own voice).
        self._rolling_mic: np.ndarray = np.zeros(0, dtype=np.float32)
        self._rolling_system: np.ndarray = np.zeros(0, dtype=np.float32)
        # The mixed (mic + system) buffer mirrors what we feed to the Apple
        # sidecar — the Whisper worker re-transcribes from this so it sees the
        # same signal the sidecar saw, not just BlackHole audio.
        self._rolling_mixed: np.ndarray = np.zeros(0, dtype=np.float32)
        self._rolling_lock = threading.Lock()

        # Wall-clock origin for converting sidecar segment times into our
        # session timeline.
        self._session_start: float = 0.0
        # Sidecar reports times relative to the start of its audio stream;
        # we keep a counter of how many seconds we've fed it. Used to keep
        # the offset model consistent with the old implementation.
        self._total_processed: float = 0.0

        # Set in initialize() if config.WHISPER_ENABLED. Stays None for the
        # disabled / fallback path so wait_whisper_final and flush_whisper become no-ops.
        self._whisper_worker = None

    async def initialize(self):
        self._loop = asyncio.get_running_loop()
        await asyncio.get_running_loop().run_in_executor(None, _ensure_binary_built)

        if self._proc is not None:
            self.shutdown()

        logger.info("Starting speech sidecar...")
        self._proc = subprocess.Popen(
            [str(SIDECAR_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        self._session_start = time.time()
        self._stdout_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())
        logger.info("Speech sidecar started (pid %d)", self._proc.pid)

        if config.WHISPER_ENABLED:
            try:
                self._whisper_worker = WhisperWorker(self, self._broadcast)
                await self._whisper_worker.initialize()
            except Exception:
                logger.exception("Failed to start WhisperWorker — falling back to Apple-only path")
                self._whisper_worker = None
        else:
            logger.info("WHISPER_ENABLED=0 — running Apple-only transcription")
            self._whisper_worker = None

    async def _read_stdout(self):
        """Drain JSONL from the sidecar's stdout into self._pending."""
        if self._proc is None or self._proc.stdout is None:
            return
        loop = asyncio.get_running_loop()
        proc_stdout = self._proc.stdout
        while True:
            line = await loop.run_in_executor(None, proc_stdout.readline)
            if not line:
                logger.info("Speech sidecar stdout closed")
                return
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Sidecar non-JSON line: %r", line[:200])
                continue

            if obj.get("error"):
                logger.error("Sidecar error: %s", obj)
                continue

            # Only surface finalized segments; partials would spam the UI.
            if not obj.get("is_final"):
                continue

            text = (obj.get("text") or "").strip()
            if not text:
                continue
            seg = TranscriptSegment(
                text=text,
                start_time=float(obj.get("start", 0.0)),
                end_time=float(obj.get("end", 0.0)),
            )
            seg.apple_text = seg.text
            duration = max(0.0, seg.end_time - seg.start_time)
            if self._whisper_worker is not None and duration >= config.WHISPER_MIN_SEGMENT_SECONDS:
                # asyncio.Queue is not threadsafe; _read_stdout runs in an executor
                # thread, so we must hop onto the asyncio loop to enqueue.
                worker = self._whisper_worker  # capture so the closure is stable
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(worker.enqueue, seg)
                else:
                    worker.enqueue(seg)
            else:
                self._mark_segment_final_fastpath(seg)
            with self._pending_lock:
                self._pending.append(seg)
                self._segments.append(seg)

    async def _read_stderr(self):
        """Forward sidecar stderr to our logger."""
        if self._proc is None or self._proc.stderr is None:
            return
        loop = asyncio.get_running_loop()
        proc_stderr = self._proc.stderr
        while True:
            line = await loop.run_in_executor(None, proc_stderr.readline)
            if not line:
                return
            logger.info("sidecar: %s", line.decode("utf-8", "replace").rstrip())

    def add_audio(
        self,
        mixed: np.ndarray,
        mic: np.ndarray | None = None,
        system: np.ndarray | None = None,
    ):
        """Push a chunk to the sidecar and rolling buffers.

        Called from sounddevice's C callback thread.

        Args:
            mixed: combined mic+system audio for transcription (float32 mono).
            mic:   mic-only audio for "Me" detection (zeros if mic disabled).
            system: system-only audio for speaker clustering.
        """
        if self._proc is None or self._proc.stdin is None:
            return

        if mic is None:
            mic = np.zeros_like(mixed)
        if system is None:
            system = mixed

        # Update rolling buffers (trim to ROLLING_BUFFER_SECONDS).
        max_samples = int(ROLLING_BUFFER_SECONDS * config.AUDIO_SAMPLE_RATE)
        with self._rolling_lock:
            self._rolling_mic = np.concatenate([self._rolling_mic, mic])
            if len(self._rolling_mic) > max_samples:
                self._rolling_mic = self._rolling_mic[-max_samples:]
            self._rolling_system = np.concatenate([self._rolling_system, system])
            if len(self._rolling_system) > max_samples:
                self._rolling_system = self._rolling_system[-max_samples:]
            self._rolling_mixed = np.concatenate([self._rolling_mixed, mixed])
            if len(self._rolling_mixed) > max_samples:
                self._rolling_mixed = self._rolling_mixed[-max_samples:]

        self._total_processed += len(mixed) / config.AUDIO_SAMPLE_RATE

        # Convert float32 [-1, 1] to Int16 LE bytes for the sidecar.
        clipped = np.clip(mixed, -1.0, 1.0)
        int16 = (clipped * 32767.0).astype(np.int16, copy=False)
        try:
            self._proc.stdin.write(int16.tobytes())
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            logger.warning("Speech sidecar stdin closed unexpectedly")

    async def transcribe_buffer(
        self,
    ) -> tuple[list[TranscriptSegment], np.ndarray | None, np.ndarray | None]:
        """Drain pending segments. Returns (segments, mic_audio, system_audio).

        The audio buffers are copies of the rolling 60s windows; segment
        start/end_time are absolute timestamps which main.py converts to
        in-buffer offsets via (start_time - (total_processed - len(audio)/sr)).
        """
        with self._pending_lock:
            if not self._pending:
                return [], None, None
            segs = self._pending
            self._pending = []

        with self._rolling_lock:
            mic = self._rolling_mic.copy() if self._rolling_mic.size > 0 else None
            system = self._rolling_system.copy() if self._rolling_system.size > 0 else None

        return segs, mic, system

    def get_recent_text(
        self,
        window_seconds: float | None = None,
        exclude_speaker: str | None = None,
    ) -> str:
        """Get recent transcript text, optionally excluding a speaker."""
        if window_seconds is None:
            window_seconds = config.TRANSCRIPT_WINDOW_SECONDS
        cutoff = time.time() - window_seconds
        recent = [s for s in self._segments if s.timestamp >= cutoff]
        if exclude_speaker:
            recent = [s for s in recent if s.speaker != exclude_speaker]
        return " ".join(s.text for s in recent)

    def get_recent_conversation(
        self,
        window_seconds: float | None = None,
        exclude_speaker: str | None = None,
    ) -> str:
        """Get recent transcript as speaker-labeled lines for context resolution."""
        if window_seconds is None:
            window_seconds = config.TRANSCRIPT_WINDOW_SECONDS
        cutoff = time.time() - window_seconds
        recent = [s for s in self._segments if s.timestamp >= cutoff]
        if exclude_speaker:
            recent = [s for s in recent if s.speaker != exclude_speaker]
        lines = []
        for s in recent:
            label = s.speaker if s.speaker != "Unknown" else "Speaker"
            lines.append(f"[{label}] {s.text}")
        return "\n".join(lines)

    def get_all_segments(self) -> list[TranscriptSegment]:
        return list(self._segments)

    def clear(self):
        with self._pending_lock:
            self._pending.clear()
        self._segments.clear()
        with self._rolling_lock:
            self._rolling_mic = np.zeros(0, dtype=np.float32)
            self._rolling_system = np.zeros(0, dtype=np.float32)
            self._rolling_mixed = np.zeros(0, dtype=np.float32)
        self._total_processed = 0.0

    def _mark_segment_final_fastpath(self, seg: TranscriptSegment):
        """Skip the worker — mark the segment final so the detector never blocks.

        Called when WhisperWorker is disabled OR the segment is too short to be
        worth Whispering (Whisper hallucinates 'Thank you for watching' on near-
        silent <1s clips, so we keep the Apple text instead).
        """
        seg.whisper_final = True
        # Touching .whisper_event here would bind to whichever loop is running.
        # _read_stdout runs inside the asyncio loop's executor thread, so we use
        # call_soon_threadsafe to set the event in the main loop.
        if self._loop is not None:
            self._loop.call_soon_threadsafe(seg.whisper_event.set)
        else:
            seg.whisper_event.set()

    async def wait_whisper_final(self, segs: list["TranscriptSegment"], timeout: float):
        """Wait until every segment in `segs` has whisper_final=True, or `timeout` elapses.

        Returns silently in either case — caller proceeds with whatever text is final.
        """
        deadline = time.monotonic() + timeout
        for s in segs:
            if s.whisper_final:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            try:
                await asyncio.wait_for(s.whisper_event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                logger.info(
                    "wait_whisper_final timed out with %d unfinal segs remaining",
                    sum(1 for x in segs if not x.whisper_final),
                )
                return

    async def flush_whisper(self, timeout: float):
        """Drain the whisper worker queue. Called at end-of-session before summary generation."""
        if self._whisper_worker is None:
            return
        await self._whisper_worker.flush(timeout=timeout)

    def shutdown(self):
        if self._whisper_worker is not None:
            try:
                # We may be called from the stop-session path which is async; if a
                # loop is running, schedule the shutdown coroutine. Otherwise this
                # is best-effort cancellation.
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._whisper_worker.shutdown())
            except RuntimeError:
                pass
            self._whisper_worker = None
        # Cancel the stdout/stderr reader tasks so they don't crash with
        # AssertionError when self._proc goes None below. shutdown is sync —
        # we schedule cancellation on the running loop if available; otherwise
        # the tasks were never started or the loop is gone (process exit).
        for task_attr in ("_stdout_task", "_stderr_task"):
            task = getattr(self, task_attr, None)
            if task is not None and not task.done():
                task.cancel()
            setattr(self, task_attr, None)
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=3)
        except (subprocess.TimeoutExpired, OSError):
            self._proc.kill()
        finally:
            self._proc = None
