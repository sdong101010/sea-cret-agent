"""Background mlx-whisper worker. Re-transcribes finalized segments from the
Apple sidecar to upgrade their text in place. See
docs/superpowers/specs/2026-06-15-mlx-whisper-hybrid-design.md for the design.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, TYPE_CHECKING

import numpy as np

import config

if TYPE_CHECKING:
    from app.transcriber import Transcriber, TranscriptSegment

logger = logging.getLogger(__name__)


class WhisperWorker:
    def __init__(self, transcriber: "Transcriber", broadcast: Callable):
        self._transcriber = transcriber
        self._broadcast = broadcast
        self._model_name = config.WHISPER_MODEL
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None
        self._mlx_whisper = None

    async def initialize(self):
        """Lazy-import mlx_whisper, warm up, and start the worker loop."""
        # Lazy import — keeps app startup fast and lets the disabled path work
        # cleanly on non-Apple-Silicon machines or when mlx-whisper isn't installed.
        import mlx_whisper
        self._mlx_whisper = mlx_whisper
        await asyncio.get_running_loop().run_in_executor(None, self._warmup)
        self._task = asyncio.create_task(self._loop())
        self._monitor_task = asyncio.create_task(self._monitor_queue())
        logger.info("WhisperWorker started (model=%s)", self._model_name)

    def _warmup(self):
        """Run mlx_whisper on 1s of silence so the first real call is hot."""
        silence = np.zeros(config.AUDIO_SAMPLE_RATE, dtype=np.float32)
        self._mlx_whisper.transcribe(
            silence,
            path_or_hf_repo=self._model_name,
            word_timestamps=True,
            language="en",
        )

    def enqueue(self, seg: "TranscriptSegment"):
        self._queue.put_nowait(seg)

    async def _monitor_queue(self):
        while True:
            await asyncio.sleep(10.0)
            depth = self._queue.qsize()
            if depth >= 10:
                logger.warning("Whisper worker backlog: %d segments queued", depth)

    async def _loop(self):
        while True:
            seg = await self._queue.get()
            try:
                await self._process(seg)
            except Exception:
                logger.exception("Whisper worker failed for segment %s", seg.id)
            finally:
                # CONTRACT: only _loop sets whisper_final / signals the event,
                # so the guarantee holds even when _process raises. Don't
                # move these into _process.
                seg.whisper_final = True
                seg.whisper_event.set()

    async def _process(self, seg: "TranscriptSegment"):
        """Re-transcribe one segment's audio with mlx-whisper. Mutates seg.text on
        success and broadcasts a transcript_update. Does NOT set whisper_final or
        signal the event — that's _loop's finally block's job (see CONTRACT).
        """
        sr = config.AUDIO_SAMPLE_RATE
        # Atomic snapshot under the rolling lock — the audio callback thread
        # mutates the rolling buffers and _total_processed concurrently.
        # Use the MIXED buffer (mic + system), same signal the Apple sidecar
        # gets. Reading _rolling_system alone is BlackHole-only and goes silent
        # when the user is speaking into the mic with nothing playing back,
        # which makes Whisper hallucinate ("you" / "Thank you for watching").
        with self._transcriber._rolling_lock:
            rolling = self._transcriber._rolling_mixed.copy()
            total_processed = self._transcriber._total_processed
        if rolling.size == 0:
            return
        buf_duration = len(rolling) / sr
        offset = total_processed - buf_duration
        seg_start_in_buf = seg.start_time - offset
        seg_end_in_buf = seg.end_time - offset
        if seg_start_in_buf < 0:
            logger.warning("Segment %s aged out of rolling buffer; keeping Apple text", seg.id)
            return
        pad = config.WHISPER_PAD_SECONDS
        slice_start = max(0.0, seg_start_in_buf - pad)
        slice_end = min(buf_duration, seg_end_in_buf + pad)
        audio = rolling[int(slice_start * sr): int(slice_end * sr)]
        if audio.size == 0:
            return
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self._mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self._model_name,
                word_timestamps=True,
                language="en",
            ),
        )
        win_start = seg_start_in_buf - slice_start
        win_end = seg_end_in_buf - slice_start
        words = []
        for s in result.get("segments", []):
            for w in s.get("words", []):
                if w.get("end", 0) >= win_start and w.get("start", 0) <= win_end:
                    words.append(w["word"])
        new_text = "".join(words).strip()
        if not new_text:
            return  # keep Apple text on empty / hallucinated near-silence
        seg.text = new_text
        await self._broadcast({"type": "transcript_update", "id": seg.id, "text": seg.text})

    async def flush(self, timeout: float):
        """Wait for the queue to drain and the worker to process every queued segment."""
        if self._task is None:
            return
        deadline = asyncio.get_running_loop().time() + timeout
        while self._queue.qsize() > 0:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.warning("Whisper flush timeout with %d segments still queued", self._queue.qsize())
                return
            await asyncio.sleep(0.05)

    async def shutdown(self):
        for t in (self._task, self._monitor_task):
            if t is not None:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        self._task = None
        self._monitor_task = None
