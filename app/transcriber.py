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
    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._stdout_task: asyncio.Task | None = None
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
        """Build the sidecar if needed and spawn it."""
        self._loop = asyncio.get_running_loop()
        await asyncio.get_running_loop().run_in_executor(None, _ensure_binary_built)

        # If a previous session left a sidecar running, tear it down first
        # so we don't leak processes or feed audio to a stale instance.
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
        asyncio.create_task(self._read_stderr())
        logger.info("Speech sidecar started (pid %d)", self._proc.pid)

    async def _read_stdout(self):
        """Drain JSONL from the sidecar's stdout into self._pending."""
        assert self._proc and self._proc.stdout
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, self._proc.stdout.readline)
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
            with self._pending_lock:
                self._pending.append(seg)
                self._segments.append(seg)

    async def _read_stderr(self):
        """Forward sidecar stderr to our logger."""
        assert self._proc and self._proc.stderr
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, self._proc.stderr.readline)
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
        self._total_processed = 0.0

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
        """Cleanly stop the sidecar process."""
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
