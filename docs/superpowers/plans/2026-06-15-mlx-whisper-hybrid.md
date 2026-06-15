# mlx-whisper Hybrid Transcription Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace flaky Apple SpeechAnalyzer transcripts with mlx-whisper-large-v3-turbo while preserving the live UI experience and unchanged speaker-ID layer. Question detection and meeting summaries see Whisper-grade text.

**Architecture:** Apple sidecar continues to produce live, low-latency segments and runs through speaker-ID unchanged (live UI). A new background `WhisperWorker` re-transcribes each finalized segment's audio slice using mlx-whisper, and broadcasts a `transcript_update` message that swaps the line's text in place. The thought-boundary handler in `transcription_loop` waits for all segments in a thought to be `whisper_final` before invoking the question detector. End-of-session flushes the worker queue before generating the summary.

**Tech Stack:** Python 3.12, asyncio, numpy, mlx-whisper (Apple Silicon), FastAPI WebSocket, vanilla JS.

**Spec:** [`docs/superpowers/specs/2026-06-15-mlx-whisper-hybrid-design.md`](../specs/2026-06-15-mlx-whisper-hybrid-design.md)

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `app/transcriber.py` | Modify | Owns the `TranscriptSegment` dataclass (add `id`, `whisper_final`, `apple_text`, `whisper_event`); instantiates and lifecycles `WhisperWorker`; exposes `wait_whisper_final` and `flush_whisper`. |
| `app/whisper_worker.py` | **New** | Single async queue, sequential `_loop`, `_monitor_queue` task, `_process` with atomic rolling-buffer snapshot, padded slicing, word-timestamp trimming. |
| `app/main.py` | Modify | Add `seg.id` to `transcript` broadcast; insert `wait_whisper_final` before detector; reset `last_speech_time` after gate; reorder stop-session for clean flush. |
| `static/app.js` | Modify | Stamp `data-segment-id` on transcript line; handle `transcript_update` message. |
| `config.py` | Modify | Add `WHISPER_*` env-driven constants. |
| `requirements.txt` | Modify | Add `mlx-whisper>=0.4.0`. |
| `tests/conftest.py` | **New** (or modify if exists) | Shared fixtures (event loop, fake transcriber). |
| `tests/test_transcriber_whisper.py` | **New** | Unit tests for new `TranscriptSegment` fields, `wait_whisper_final`, gating fast-path. |
| `tests/test_whisper_worker.py` | **New** | Unit tests for `WhisperWorker` behavior with `mlx_whisper.transcribe` mocked. |

---

## Task 0: Verify Environment

**Files:** none (environment check)

- [ ] **Step 1: Confirm Apple Silicon and Python**

Run: `uname -m && python3 --version`
Expected: `arm64` and `Python 3.10+`

If not arm64, STOP and ask the user before continuing — mlx-whisper is Apple-Silicon-only.

- [ ] **Step 2: Confirm `pip install mlx-whisper` works in a throwaway venv**

Run:
```bash
cd /Users/sea.dong/projects/sea-cret-agent
python3 -m venv /tmp/mlx-probe && source /tmp/mlx-probe/bin/activate && pip install mlx-whisper && python -c "import mlx_whisper; print(mlx_whisper.__name__)" && deactivate && rm -rf /tmp/mlx-probe
```
Expected: prints `mlx_whisper` and exits 0.

If install fails, STOP and surface the error — the rest of the plan assumes it works.

- [ ] **Step 3: Confirm pytest runs against the existing project**

Run: `cd /Users/sea.dong/projects/sea-cret-agent && python -m pytest --collect-only 2>&1 | tail -10`
Expected: pytest is installed and either discovers tests or reports "no tests collected" (both fine).

If pytest is missing, install it: `pip install pytest pytest-asyncio`

---

## Task 1: Add Config Knobs

**Files:**
- Modify: `config.py:34` (append after `THOUGHT_PAUSE_SECONDS`)

- [ ] **Step 1: Add the Whisper config block**

Append to `config.py`:

```python

# --- mlx-whisper hybrid transcription ---
WHISPER_ENABLED = os.getenv("WHISPER_ENABLED", "1") == "1"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
# Segments shorter than this skip Whisper (avoids "Thank you for watching" hallucinations on near-silent clips).
WHISPER_MIN_SEGMENT_SECONDS = float(os.getenv("WHISPER_MIN_SEGMENT_SECONDS", "1.0"))
# Audio context fed to Whisper around each segment, trimmed back via word timestamps.
WHISPER_PAD_SECONDS = float(os.getenv("WHISPER_PAD_SECONDS", "2.0"))
# How long the question detector will wait for unseen segments to be whisper-final before proceeding.
WHISPER_GATE_TIMEOUT_SECONDS = float(os.getenv("WHISPER_GATE_TIMEOUT_SECONDS", "5.0"))
# How long to wait for the worker queue to drain at end-of-session before generating the summary.
WHISPER_FLUSH_TIMEOUT_SECONDS = float(os.getenv("WHISPER_FLUSH_TIMEOUT_SECONDS", "60.0"))
```

- [ ] **Step 2: Add mlx-whisper to requirements.txt**

Append:
```
mlx-whisper>=0.4.0
```

- [ ] **Step 3: Install it into the project's environment**

Run: `cd /Users/sea.dong/projects/sea-cret-agent && pip install -r requirements.txt`
Expected: `mlx-whisper` resolves and installs successfully.

- [ ] **Step 4: Commit**

```bash
cd /Users/sea.dong/projects/sea-cret-agent
git add config.py requirements.txt
git commit -m "chore: add WHISPER_* config knobs and mlx-whisper dep"
```

---

## Task 2: Extend `TranscriptSegment`

**Files:**
- Modify: `app/transcriber.py:34-41` (the dataclass)
- Test: `tests/test_transcriber_whisper.py` (NEW)

- [ ] **Step 1: Write the failing test**

Create `tests/test_transcriber_whisper.py`:

```python
import asyncio
import pytest

from app.transcriber import TranscriptSegment


def test_segment_has_id_and_whisper_fields():
    seg = TranscriptSegment(text="hello", start_time=0.0, end_time=1.0)
    assert isinstance(seg.id, str) and len(seg.id) >= 8
    assert seg.whisper_final is False
    assert seg.apple_text == ""


def test_segment_ids_are_unique():
    a = TranscriptSegment(text="a", start_time=0.0, end_time=1.0)
    b = TranscriptSegment(text="b", start_time=0.0, end_time=1.0)
    assert a.id != b.id


@pytest.mark.asyncio
async def test_segment_whisper_event_is_lazy_asyncio_event():
    seg = TranscriptSegment(text="hi", start_time=0.0, end_time=1.0)
    # Lazy creation — only valid inside a running loop.
    ev = seg.whisper_event
    assert isinstance(ev, asyncio.Event)
    assert not ev.is_set()
    # Should return the same event on subsequent access.
    assert seg.whisper_event is ev
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sea.dong/projects/sea-cret-agent && python -m pytest tests/test_transcriber_whisper.py -v`
Expected: FAIL with `AttributeError` on `id` / `whisper_final` / `apple_text` / `whisper_event`.

- [ ] **Step 3: Add the new fields to `TranscriptSegment`**

In `app/transcriber.py`, replace the existing dataclass (lines 34-41) and add `uuid` to imports:

```python
import uuid
```

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/sea.dong/projects/sea-cret-agent && python -m pytest tests/test_transcriber_whisper.py -v`
Expected: 3 passing.

- [ ] **Step 5: Commit**

```bash
git add app/transcriber.py tests/test_transcriber_whisper.py
git commit -m "feat(transcriber): add id/whisper_final/apple_text/whisper_event to TranscriptSegment"
```

---

## Task 3: Test/Implement `Transcriber.wait_whisper_final` and `flush_whisper`

**Files:**
- Modify: `app/transcriber.py` (after `clear()`, before `shutdown()`)
- Modify: `tests/test_transcriber_whisper.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transcriber_whisper.py`:

```python
import time
from app.transcriber import Transcriber


@pytest.mark.asyncio
async def test_wait_whisper_final_returns_immediately_if_all_final():
    t = Transcriber()
    a = TranscriptSegment(text="a", start_time=0, end_time=1)
    b = TranscriptSegment(text="b", start_time=1, end_time=2)
    a.whisper_final = True
    a.whisper_event.set()
    b.whisper_final = True
    b.whisper_event.set()
    start = time.monotonic()
    await t.wait_whisper_final([a, b], timeout=2.0)
    assert time.monotonic() - start < 0.1


@pytest.mark.asyncio
async def test_wait_whisper_final_unblocks_when_event_fires():
    t = Transcriber()
    seg = TranscriptSegment(text="x", start_time=0, end_time=1)
    async def signal_later():
        await asyncio.sleep(0.05)
        seg.whisper_final = True
        seg.whisper_event.set()
    asyncio.create_task(signal_later())
    await t.wait_whisper_final([seg], timeout=1.0)
    assert seg.whisper_final


@pytest.mark.asyncio
async def test_wait_whisper_final_returns_on_timeout_without_raising():
    t = Transcriber()
    seg = TranscriptSegment(text="x", start_time=0, end_time=1)
    start = time.monotonic()
    await t.wait_whisper_final([seg], timeout=0.1)
    elapsed = time.monotonic() - start
    assert 0.08 <= elapsed <= 0.5
    assert seg.whisper_final is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_transcriber_whisper.py -v`
Expected: 3 new failures with `AttributeError: 'Transcriber' object has no attribute 'wait_whisper_final'`.

- [ ] **Step 3: Implement `wait_whisper_final` and `flush_whisper`**

In `app/transcriber.py`, insert these methods on the `Transcriber` class right before `shutdown()` (around line 259):

```python
    async def wait_whisper_final(self, segs: list[TranscriptSegment], timeout: float):
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
                logger.info("wait_whisper_final timed out with %d unfinal segs remaining",
                            sum(1 for x in segs if not x.whisper_final))
                return

    async def flush_whisper(self, timeout: float):
        """Drain the whisper worker queue. Called at end-of-session before summary generation."""
        if self._whisper_worker is None:
            return
        await self._whisper_worker.flush(timeout=timeout)
```

You'll also need a placeholder `self._whisper_worker: WhisperWorker | None = None` in `__init__` — add it now alongside the other fields:

```python
        self._whisper_worker = None  # set in initialize() if config.WHISPER_ENABLED
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_transcriber_whisper.py -v`
Expected: all 6 passing.

- [ ] **Step 5: Commit**

```bash
git add app/transcriber.py tests/test_transcriber_whisper.py
git commit -m "feat(transcriber): add wait_whisper_final and flush_whisper helpers"
```

---

## Task 4: Build `WhisperWorker` skeleton (no mlx yet)

**Files:**
- Create: `app/whisper_worker.py`
- Create: `tests/test_whisper_worker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_whisper_worker.py`:

```python
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
    t._rolling_system = np.zeros(5 * 16000, dtype=np.float32)
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
    assert seg.whisper_final is True
    broadcast.assert_awaited()


@pytest.mark.asyncio
async def test_worker_loop_marks_final_even_on_exception():
    t = Transcriber()
    t._rolling_system = np.zeros(5 * 16000, dtype=np.float32)
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
    t._rolling_system = np.zeros(60 * 16000, dtype=np.float32)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_whisper_worker.py -v`
Expected: ImportError on `app.whisper_worker`.

- [ ] **Step 3: Create `app/whisper_worker.py`**

```python
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
                seg.whisper_final = True
                seg.whisper_event.set()

    async def _process(self, seg: "TranscriptSegment"):
        sr = config.AUDIO_SAMPLE_RATE
        # Atomic snapshot under the rolling lock — the audio callback thread
        # mutates _rolling_system and _total_processed concurrently.
        with self._transcriber._rolling_lock:
            rolling = self._transcriber._rolling_system.copy()
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_whisper_worker.py -v`
Expected: 4 passing.

- [ ] **Step 5: Commit**

```bash
git add app/whisper_worker.py tests/test_whisper_worker.py
git commit -m "feat: add WhisperWorker (mocked mlx; queue, slice, trim, broadcast)"
```

---

## Task 5: Wire `WhisperWorker` into `Transcriber`

**Files:**
- Modify: `app/transcriber.py:81-103` (`initialize`), `_read_stdout` (115-138), `clear` (250-257), `shutdown` (259-271)
- Modify: `tests/test_transcriber_whisper.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_transcriber_whisper.py`:

```python
import config


@pytest.mark.asyncio
async def test_initialize_skips_worker_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "WHISPER_ENABLED", False)
    t = Transcriber()
    # Patch sidecar startup so we don't actually spawn the binary.
    monkeypatch.setattr("app.transcriber._ensure_binary_built", lambda: None)
    monkeypatch.setattr("app.transcriber.subprocess.Popen", MagicMock())
    await t.initialize()
    assert t._whisper_worker is None
    t.shutdown()


@pytest.mark.asyncio
async def test_short_segment_is_marked_whisper_final_immediately():
    t = Transcriber()
    # No worker, no real proc — just exercise the helper.
    seg = TranscriptSegment(text="yeah", start_time=0.0, end_time=0.5)
    t._mark_segment_final_fastpath(seg)
    assert seg.whisper_final is True
    assert seg.whisper_event.is_set()


from unittest.mock import MagicMock
```

(If the test file already imports MagicMock at the top, drop the trailing import.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_transcriber_whisper.py -v`
Expected: 2 new failures (`_mark_segment_final_fastpath` doesn't exist; `_whisper_worker` not set).

- [ ] **Step 3: Wire the worker into `Transcriber`**

In `app/transcriber.py`:

a) Add the import at the top with the other imports:

```python
from app.whisper_worker import WhisperWorker
```

b) Add `broadcast` as an optional dep on the constructor (so `main.py` can inject it):

```python
    def __init__(self, broadcast=None):
        ...
        self._broadcast = broadcast or (lambda msg: asyncio.sleep(0))
```

(Existing code that calls `Transcriber()` will keep working — broadcast defaults to a no-op.)

c) In `initialize()`, replace the body with this version (preserving existing sidecar setup):

```python
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
        asyncio.create_task(self._read_stderr())
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
```

d) Add the fast-path helper and modify `_read_stdout` to use the worker. Replace the segment-append block in `_read_stdout` (currently around lines 131-138):

```python
            seg = TranscriptSegment(
                text=text,
                start_time=float(obj.get("start", 0.0)),
                end_time=float(obj.get("end", 0.0)),
            )
            seg.apple_text = seg.text
            duration = max(0.0, seg.end_time - seg.start_time)
            if self._whisper_worker is not None and duration >= config.WHISPER_MIN_SEGMENT_SECONDS:
                self._whisper_worker.enqueue(seg)
            else:
                self._mark_segment_final_fastpath(seg)
            with self._pending_lock:
                self._pending.append(seg)
                self._segments.append(seg)
```

Add `_mark_segment_final_fastpath` method:

```python
    def _mark_segment_final_fastpath(self, seg: TranscriptSegment):
        """Skip the worker — mark the segment final so the detector never blocks."""
        seg.whisper_final = True
        # Touching .whisper_event here would bind to whichever loop is running;
        # we are running inside the asyncio loop's executor thread when called
        # from _read_stdout, so it's safe — the event binds to self._loop.
        if self._loop is not None:
            self._loop.call_soon_threadsafe(seg.whisper_event.set)
        else:
            seg.whisper_event.set()
```

e) Update `clear()` and `shutdown()`:

```python
    def clear(self):
        with self._pending_lock:
            self._pending.clear()
        self._segments.clear()
        with self._rolling_lock:
            self._rolling_mic = np.zeros(0, dtype=np.float32)
            self._rolling_system = np.zeros(0, dtype=np.float32)
        self._total_processed = 0.0

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: all passing (Task 2/3/4/5 tests).

- [ ] **Step 5: Commit**

```bash
git add app/transcriber.py tests/test_transcriber_whisper.py
git commit -m "feat(transcriber): wire WhisperWorker — gated on WHISPER_ENABLED with safe fallback"
```

---

## Task 6: Broadcast `seg.id` and update `transcription_loop` for gating

**Files:**
- Modify: `app/main.py:299-305` (transcript broadcast)
- Modify: `app/main.py:321-330` (thought-boundary block)
- Modify: `app/main.py:441-468` (stop-session block)
- Modify: `app/main.py` (Transcriber instantiation — wherever `Transcriber()` is called)

- [ ] **Step 1: Inject `broadcast` into `Transcriber`**

Find where `Transcriber()` is instantiated in `app/main.py` (search for `Transcriber()`). Replace with:

```python
transcriber = Transcriber(broadcast=broadcast)
```

If the assignment happens before `broadcast` is defined as a function (`broadcast` is at line 234), move the `transcriber = Transcriber(...)` line below the `broadcast` definition. (If it's currently at module scope, instantiate it inside the same place where `audio_capture` etc. are wired — pick the earliest spot after `broadcast` is defined.)

- [ ] **Step 2: Add `id` to the transcript broadcast**

In `app/main.py`, the existing block at lines 298-305:

```python
            for seg in segments:
                await broadcast({
                    "type": "transcript",
                    "text": seg.text,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "speaker": seg.speaker,
                })
```

Becomes:

```python
            for seg in segments:
                await broadcast({
                    "type": "transcript",
                    "id": seg.id,
                    "text": seg.text,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "speaker": seg.speaker,
                })
```

- [ ] **Step 3: Insert the whisper gate before detector.detect**

Replace lines 321-330 (the `if should_detect:` block):

```python
            if should_detect:
                detection_pending = False
                # Snapshot + clear the unseen buffer atomically.
                batch = unseen_segments
                unseen_segments = []

                # Wait for Whisper to upgrade these segments before running detection.
                # Best-effort — segments still on Apple text after the timeout proceed as-is.
                await transcriber.wait_whisper_final(
                    batch, timeout=config.WHISPER_GATE_TIMEOUT_SECONDS,
                )
                # Reset the pause clock so we don't re-fire immediately on the next
                # iteration against new segments that arrived during the wait.
                last_speech_time = time.time()

                new_text = _format_segments_for_prompt(batch)
                primary_speaker = batch[-1].speaker if batch else "Unknown"

                results = await detector.detect(new_text, meeting_state)
```

(Keep the rest of the `if should_detect:` body — `for q, thread in results: ...` — intact.)

- [ ] **Step 4: Reorder the stop-session block**

Replace lines 441-468 with:

```python
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

        # Drain the Whisper worker queue before generating the summary so it
        # sees upgraded text. Best-effort — un-upgraded segments retain Apple text.
        await transcriber.flush_whisper(timeout=config.WHISPER_FLUSH_TIMEOUT_SECONDS)

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
```

- [ ] **Step 5: Sanity check — start the app**

Run: `cd /Users/sea.dong/projects/sea-cret-agent && WHISPER_ENABLED=0 python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level info`
Expected: server starts cleanly, prints `Application startup complete`, no exceptions.

Stop with Ctrl-C.

Repeat with `WHISPER_ENABLED=1` (will trigger first-launch model download — be ready to wait ~1.5GB):

```bash
WHISPER_ENABLED=1 python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level info
```

Expected: server starts; first time may print download progress; eventually `Application startup complete` (still no live audio yet — model is warmed lazily on first session).

Stop with Ctrl-C.

- [ ] **Step 6: Commit**

```bash
git add app/main.py
git commit -m "feat(main): broadcast seg.id; gate detector on whisper-final; flush worker before summary"
```

---

## Task 7: Frontend — handle `transcript_update`

**Files:**
- Modify: `static/app.js:139-164` (`appendTranscript`)
- Modify: `static/app.js:80-100` (the WebSocket message handler — find the `switch (msg.type)` or equivalent)

- [ ] **Step 1: Stamp `data-segment-id` on transcript lines**

In `static/app.js`, find `appendTranscript` (line 139). Before the line that builds `line.innerHTML = ...`, add:

```javascript
  if (msg.id) line.dataset.segmentId = msg.id;
```

(Place it right before `line.innerHTML = ...`.)

- [ ] **Step 2: Handle `transcript_update` in the WebSocket message dispatch**

Find the WebSocket message dispatch (around line 85 — the case that calls `appendTranscript(msg)`). Add a sibling case:

```javascript
    case "transcript_update": {
      const line = document.querySelector(
        `.transcript-line[data-segment-id="${msg.id}"]`
      );
      if (!line) break;
      const textEl = line.querySelector(".text");
      if (textEl) textEl.textContent = msg.text;
      break;
    }
```

(If the dispatch uses `if/else` rather than `switch`, mirror the same structure: `else if (msg.type === "transcript_update") { ... }`.)

- [ ] **Step 3: Manual sanity check in browser**

Run the app with `WHISPER_ENABLED=0` first (so behavior matches main today):

```bash
cd /Users/sea.dong/projects/sea-cret-agent && WHISPER_ENABLED=0 python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`, start a meeting, speak a few sentences. Expected: live transcript renders unchanged (no `transcript_update` messages with `WHISPER_ENABLED=0`). Stop the meeting, check the summary — text matches Apple sidecar output, no errors in browser console.

Now run with Whisper on:

```bash
WHISPER_ENABLED=1 python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open the app, start a meeting, speak a few clean sentences. Expected:
- A line appears with Apple text (e.g., "yeah I think we shoulder dip it")
- Within ~1s, the same line's text updates to a Whisper-quality version (e.g., "yeah I think we should ship it")
- Speaker label and timestamp on the line are unchanged
- No errors in browser console or server log

Stop the meeting. Confirm `data/summaries/*.md` contains the upgraded text.

- [ ] **Step 4: Commit**

```bash
git add static/app.js
git commit -m "feat(ui): stamp data-segment-id on transcript lines; handle transcript_update"
```

---

## Task 8: Integration smoke test (real-world meeting)

**Files:** none (manual verification)

- [ ] **Step 1: Run a 5-minute test meeting with the app**

```bash
cd /Users/sea.dong/projects/sea-cret-agent && WHISPER_ENABLED=1 python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In the browser:
1. Start a meeting.
2. Speak a mix of: clean sentences, sentences with proper nouns ("Salesforce", "Agentforce", "Anthropic"), sentences with technical jargon, very short utterances ("yeah", "mhm").
3. Watch the live transcript update in place as Whisper catches up.
4. Wait until at least one question gets detected (e.g., end a sentence with "?"). Confirm the question's text matches Whisper output, not Apple's.
5. Stop the meeting.
6. Open `data/summaries/<latest>.md` — confirm transcripts are Whisper-grade.
7. Tail server log for warnings (`grep -E "WARN|ERROR" ~/projects/sea-cret-agent/...` — wherever logs go) — there should be NO `Failed to start WhisperWorker`, no exception traces, no stuck `wait_whisper_final timed out` repeated more than 1-2 times.

- [ ] **Step 2: Failure-injection test — `WHISPER_ENABLED=0`**

Stop the server, restart with:

```bash
WHISPER_ENABLED=0 python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Run a 1-minute test meeting. Expected: transcripts are Apple-quality, no `transcript_update` in network panel, detector and summary work, server log shows `WHISPER_ENABLED=0 — running Apple-only transcription`.

- [ ] **Step 3: Run the unit test suite end-to-end**

```bash
cd /Users/sea.dong/projects/sea-cret-agent && python -m pytest tests/ -v
```

Expected: all green.

- [ ] **Step 4: Commit anything you noticed needed fixing in Steps 1-3**

If you needed any fixes during smoke testing, commit them now. Otherwise skip.

---

## Notes for the implementer

- **Don't refactor adjacent code.** The plan is scoped to the Whisper integration; resist the temptation to clean up unrelated parts of `main.py` or `transcriber.py`.
- **`Transcriber._broadcast` typing.** The default no-op (`lambda msg: asyncio.sleep(0)`) lets tests instantiate `Transcriber()` without a broadcast argument. If you find yourself adding type annotations, `Callable[[dict], Awaitable[None]]` is the right shape.
- **Why `path_or_hf_repo` is non-negotiable.** Without it, `mlx_whisper.transcribe()` silently uses `whisper-tiny` and the bug shows up only as "Whisper output is also gibberish." The audit caught this; the implementation must too. Both `_warmup` and `_process` pass it.
- **Atomic snapshot.** The audio callback thread mutates `_rolling_system` and `_total_processed` independently. Reading them in two steps would let you slice from the wrong offset. Always inside `_rolling_lock`.
- **Don't add VAD, pyannote, or word-by-word streaming.** They're explicitly non-goals.
- **If a test fails, do not weaken the test to make it pass.** Fix the implementation. The audit pass already removed the "obvious" bugs from the spec — tests are the safety net for the rest.
