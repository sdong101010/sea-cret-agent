"""Local speech-to-text using faster-whisper."""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from faster_whisper import WhisperModel

import config

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    text: str
    start_time: float
    end_time: float
    speaker: str = "Unknown"
    timestamp: float = field(default_factory=time.time)
    is_final: bool = True


class Transcriber:
    def __init__(self):
        self._model: WhisperModel | None = None
        self._audio_buffer: list[np.ndarray] = []
        self._buffer_duration: float = 0.0
        self._segments: list[TranscriptSegment] = []
        self._session_start: float = 0.0
        self._total_processed: float = 0.0
        self._buf_lock = threading.Lock()
        self._transcribe_lock = asyncio.Lock()

    async def initialize(self):
        """Load the Whisper model (runs in thread to avoid blocking)."""
        logger.info("Loading Whisper model '%s' (this may take a moment on first run)...", config.WHISPER_MODEL_SIZE)
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(
                config.WHISPER_MODEL_SIZE,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
            ),
        )
        self._session_start = time.time()
        logger.info("Whisper model loaded successfully")

    def add_audio(self, audio_chunk: np.ndarray):
        """Buffer incoming audio (called from sounddevice's C thread)."""
        with self._buf_lock:
            self._audio_buffer.append(audio_chunk)
            self._buffer_duration += len(audio_chunk) / config.AUDIO_SAMPLE_RATE

    async def transcribe_buffer(self) -> tuple[list[TranscriptSegment], np.ndarray | None]:
        """Transcribe the buffered audio and return (new_segments, raw_audio).

        The raw audio is returned so callers can run speaker identification
        on each segment's time range.
        """
        if not self._audio_buffer or self._model is None:
            return [], None

        async with self._transcribe_lock:
            with self._buf_lock:
                if not self._audio_buffer:
                    return [], None
                audio = np.concatenate(self._audio_buffer)
                self._audio_buffer.clear()
                buffer_dur = self._buffer_duration
                self._buffer_duration = 0.0

            if np.max(np.abs(audio)) < 0.01:
                self._total_processed += buffer_dur
                return [], None

            audio_copy = audio.copy()

            loop = asyncio.get_event_loop()

            def _run_whisper():
                segments_iter, info = self._model.transcribe(
                    audio_copy,
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    no_speech_threshold=0.5,
                    condition_on_previous_text=False,
                )
                return list(segments_iter)

            try:
                raw_segments = await loop.run_in_executor(None, _run_whisper)
            except Exception:
                logger.exception("Whisper transcription failed")
                self._total_processed += buffer_dur
                return [], None

            new_segments = []
            for seg in raw_segments:
                text = seg.text.strip()
                if not text:
                    continue
                ts = TranscriptSegment(
                    text=text,
                    start_time=self._total_processed + seg.start,
                    end_time=self._total_processed + seg.end,
                )
                new_segments.append(ts)
                self._segments.append(ts)

            self._total_processed += buffer_dur
            if new_segments:
                logger.info("Transcribed %d segment(s): %s", len(new_segments), new_segments[-1].text[:80])
            return new_segments, audio_copy

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
        self._segments.clear()
        self._audio_buffer.clear()
        self._buffer_duration = 0.0
        self._total_processed = 0.0
