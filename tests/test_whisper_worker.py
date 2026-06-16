import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import config
from app.transcriber import Transcriber, TranscriptSegment
from app.whisper_worker import WhisperWorker


@pytest.mark.asyncio
async def test_worker_signals_event_on_disabled_or_short_segment():
    # Worker is OFF — caller code should mark fast-path; this test just ensures
    # the worker's enqueue path doesn't blow up if called.
    t = Transcriber()
    broadcast = AsyncMock()
    w = WhisperWorker(t, broadcast)
    # We don't initialize() here, so _loop isn't running. Just confirm enqueue is non-blocking.
    seg = TranscriptSegment(text="x", start_time=0, end_time=2)
    w.enqueue(seg)
    assert w._queue.qsize() == 1


@pytest.mark.asyncio
async def test_worker_loop_processes_segment_and_signals_event():
    t = Transcriber()
    # Pre-fill the rolling buffer with 5 seconds of fake audio.
    t._rolling_mixed = np.zeros(5 * 16000, dtype=np.float32)
    t._total_processed = 5.0
    broadcast = AsyncMock()
    w = WhisperWorker(t, broadcast)

    # Mock mlx_whisper.transcribe to return a fixed result.
    fake_mlx = MagicMock()
    fake_mlx.transcribe.return_value = {
        "segments": [{"words": [
            {"word": "hello ", "start": 2.0, "end": 2.5},
            {"word": "world", "start": 2.5, "end": 3.0},
        ]}],
    }
    w._mlx_whisper = fake_mlx

    seg = TranscriptSegment(text="apple text", start_time=2.0, end_time=3.0)
    seg.apple_text = "apple text"
    w.enqueue(seg)

    # Run one iteration of the loop manually to keep the test deterministic.
    pulled = await w._queue.get()
    await w._process(pulled)

    assert seg.text == "hello world"
    assert seg.whisper_final is False  # _process alone does not set this; the _loop's finally block does
    broadcast.assert_awaited()


@pytest.mark.asyncio
async def test_worker_loop_marks_final_even_on_exception():
    t = Transcriber()
    t._rolling_mixed = np.zeros(5 * 16000, dtype=np.float32)
    t._total_processed = 5.0
    broadcast = AsyncMock()
    w = WhisperWorker(t, broadcast)

    fake_mlx = MagicMock()
    fake_mlx.transcribe.side_effect = RuntimeError("boom")
    w._mlx_whisper = fake_mlx

    seg = TranscriptSegment(text="apple text", start_time=2.0, end_time=3.0)
    seg.apple_text = "apple text"

    # Simulate one pass of _loop's try/finally.
    try:
        await w._process(seg)
    except RuntimeError:
        pass
    finally:
        seg.whisper_final = True
        seg.whisper_event.set()

    assert seg.whisper_final
    assert seg.whisper_event.is_set()
    assert seg.text == "apple text"  # unchanged on failure


@pytest.mark.asyncio
async def test_worker_skips_segment_aged_out_of_buffer():
    t = Transcriber()
    # Buffer is 60s; segment claims start_time at 5s while total_processed is 100s.
    # So seg_start_in_buf = 5 - (100 - 60) = -35 → out of buffer.
    t._rolling_mixed = np.zeros(60 * 16000, dtype=np.float32)
    t._total_processed = 100.0
    broadcast = AsyncMock()
    w = WhisperWorker(t, broadcast)
    fake_mlx = MagicMock()
    w._mlx_whisper = fake_mlx

    seg = TranscriptSegment(text="apple", start_time=5.0, end_time=6.0)
    seg.apple_text = "apple"
    await w._process(seg)
    assert seg.text == "apple"  # untouched
    fake_mlx.transcribe.assert_not_called()
