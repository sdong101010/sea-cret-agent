# Hybrid Whisper Transcription Upgrade

**Date:** 2026-06-15
**Status:** Proposed
**Owner:** Sea

## Problem

The current speech pipeline uses Apple's on-device `SpeechAnalyzer` via the
Swift sidecar at `bin/speech_sidecar`. It's fast and runs live, but accuracy is
mediocre — proper nouns, technical jargon, and noisy audio routinely come
through as gibberish. Because the question-detection LLM and the meeting
summary both consume the transcript text directly, gibberish in =
gibberish-derived questions and useless summaries.

## Goal

Raise transcript accuracy without losing the live UI feel, then gate the LLM
question detector on the upgraded text so detection sees Whisper-grade
input.

## Approach: Hybrid (Apple → mlx-whisper → detector)

Keep the Apple sidecar as the live, low-latency path. Add a background
mlx-whisper worker that re-transcribes each finalized segment's audio slice
and replaces the segment text in place. The question detector waits until
all segments in a "thought" have been upgraded before running.

```
audio chunk
   ├──► sidecar ──► finalized segment ──► speaker-ID ──► broadcast(transcript) [live UI]
   │                                              └──► whisper queue
   │
   └──► rolling mic / system buffers (60s, unchanged)

whisper worker (background, sequential):
   pop segment ──► slice audio ±2s padding from rolling buffer
              ──► mlx_whisper.transcribe() with word timestamps
              ──► trim words to original [start_time, end_time]
              ──► swap seg.text in place
              ──► broadcast({type: "transcript_update", segment_id, text})
              ──► mark seg.whisper_final = True

transcription_loop (existing thought-boundary logic):
   thought boundary fires (pause >= THOUGHT_PAUSE_SECONDS or trailing "?")
   ──► await all unseen_segments to be whisper_final (or timeout)
   ──► detector.detect(upgraded_text) ──► answer_question()
```

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Model | `mlx-community/whisper-large-v3-turbo` | ~4× faster than large-v3, ~98% as accurate for English. Empirically sub-real-time on M-series (target p95 < 1s on a 5s clip — measured by Test #3, not vendor-attested). MLX-converted weights are ~1.5GB on first download (cached at `~/.cache/huggingface/hub/`). |
| Live UI text | Apple sidecar output (unchanged) | Preserves the cosmetic "streaming" feel during the second of latency before Whisper lands. |
| Re-transcribe scope | Every finalized segment ≥ 1.0s | Whisper hallucinates "Thank you for watching" / "Subtitles by..." on near-silent ≤1s clips. Below threshold: keep Apple text. |
| Audio context | ±2s padding around segment | Whisper accuracy improves materially with surrounding context; word-timestamps let us trim back to the original range. |
| Question detection gating | Block on whisper-final, with 3s timeout | Detector LLM should see best-available text. Worst-case 3s wait if Whisper backs up. |
| Diarization | Unchanged | Apple sidecar still does utterance segmentation; speaker-ID still runs on the segment ranges. Whisper just rewrites the text within those ranges. |
| Summary input | Whisper-final segments only | `summary_gen.generate(segments, ...)` already takes `transcriber.get_all_segments()`; we just ensure end-of-meeting flush waits for the worker queue to drain. |

## Components

### `app/transcriber.py` — additions

`TranscriptSegment` gains:

```python
@dataclass
class TranscriptSegment:
    text: str
    start_time: float
    end_time: float
    speaker: str = "Unknown"
    timestamp: float = field(default_factory=time.time)
    is_final: bool = True
    # NEW:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    whisper_final: bool = False        # True once mlx-whisper has processed it
    apple_text: str = ""                # original sidecar text, retained for debugging / fallback
```

Each `TranscriptSegment` also owns `self._whisper_event: asyncio.Event`
(created lazily on first await — events bind to the running loop). The worker
sets it in BOTH the success path and the failure path. `wait_whisper_final`
awaits each event with a deadline.

`Transcriber` gains:

- `self._whisper_worker: WhisperWorker | None = None` — created in `initialize()` only when `config.WHISPER_ENABLED` is true and the `mlx_whisper` import succeeds. Shut down in `shutdown()`.
- After appending a finalized segment in `_read_stdout`:
  - If `_whisper_worker is None` (disabled or import failed): set `whisper_final = True` and signal its event immediately. Detector never blocks.
  - Else if segment duration < `WHISPER_MIN_SEGMENT_SECONDS`: same — keep Apple text, mark final, signal.
  - Else: enqueue to worker. The worker will set `whisper_final` and signal the event when done (success or failure).
- `await wait_whisper_final(segs, timeout=3.0)` helper used by `main.py`'s
  thought-boundary handler. Implementation:
  ```python
  async def wait_whisper_final(self, segs, timeout):
      deadline = time.monotonic() + timeout
      for s in segs:
          remaining = max(0.0, deadline - time.monotonic())
          try:
              await asyncio.wait_for(s.whisper_event.wait(), timeout=remaining)
          except asyncio.TimeoutError:
              return  # caller proceeds with whatever is final-by-now
  ```
- `clear()` must cancel any pending worker queue items (drain the queue and signal events with `whisper_final=False`-but-released, so any awaiter unblocks). `shutdown()` cancels the worker's `_loop` task and awaits it.

### `app/whisper_worker.py` — new module

Single-file worker. Owns the mlx-whisper model and a serial async queue.

```python
class WhisperWorker:
    def __init__(self, transcriber: "Transcriber", broadcast: Callable):
        self._model_name = "mlx-community/whisper-large-v3-turbo"
        self._queue: asyncio.Queue[TranscriptSegment] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._transcriber = transcriber
        self._broadcast = broadcast

    async def initialize(self):
        # Lazy import — keeps app startup fast and lets the disabled path work
        # cleanly on non-Apple-Silicon machines or when mlx-whisper isn't installed.
        import mlx_whisper
        self._mlx_whisper = mlx_whisper
        # Warm the model with 1s of silence to avoid a cold-start hit on the first real segment.
        # NOTE: must pass path_or_hf_repo here too — defaults to whisper-tiny otherwise.
        await asyncio.get_running_loop().run_in_executor(None, self._warmup)
        self._task = asyncio.create_task(self._loop())
        self._monitor_task = asyncio.create_task(self._monitor_queue())

    def enqueue(self, seg: TranscriptSegment):
        self._queue.put_nowait(seg)

    async def _monitor_queue(self):
        """Log a warning every 10s when queue depth >= 10 — the worker is falling behind."""
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
                # ALWAYS mark final and release waiters, success or failure.
                seg.whisper_final = True
                seg.whisper_event.set()

    async def _process(self, seg: TranscriptSegment):
        # 1. Atomic snapshot under _rolling_lock: copy audio + read _total_processed
        #    in one critical section, since the audio callback thread mutates both.
        with self._transcriber._rolling_lock:
            rolling = self._transcriber._rolling_system.copy()
            total_processed = self._transcriber._total_processed
        sr = config.AUDIO_SAMPLE_RATE
        buf_duration = len(rolling) / sr
        offset = total_processed - buf_duration
        seg_start_in_buf = seg.start_time - offset
        seg_end_in_buf = seg.end_time - offset
        # 2. Skip if the segment is older than the rolling window (partial overlap
        #    is treated the same — we don't run Whisper on truncated audio).
        if seg_start_in_buf < 0:
            logger.warning("Segment %s aged out of rolling buffer; keeping Apple text", seg.id)
            return
        # 3. Pad ±WHISPER_PAD_SECONDS, clipped to buffer extent. Right-pad will be
        #    asymmetric near the head of the buffer because newer audio simply
        #    doesn't exist yet — accepted, not waited-for.
        pad = config.WHISPER_PAD_SECONDS
        slice_start = max(0.0, seg_start_in_buf - pad)
        slice_end   = min(buf_duration, seg_end_in_buf + pad)
        audio = rolling[int(slice_start * sr) : int(slice_end * sr)]
        # 4. Run mlx-whisper. CRITICAL: path_or_hf_repo is required — its default is whisper-tiny.
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self._mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self._model_name,
                word_timestamps=True,
                language="en",
            ),
        )
        # 5. Trim words to the original segment's window inside the slice.
        win_start = seg_start_in_buf - slice_start
        win_end   = seg_end_in_buf - slice_start
        words = []
        for s in result.get("segments", []):
            for w in s.get("words", []):
                if w["end"] >= win_start and w["start"] <= win_end:
                    words.append(w["word"])
        new_text = "".join(words).strip()
        # 6. Empty / hallucinated near-silence: fall back to Apple text.
        if not new_text:
            return
        seg.text = new_text
        await self._broadcast({"type": "transcript_update", "id": seg.id, "text": seg.text})
```

Audio-slice math is the only subtle bit. Replicate the offset model used in
`main.py`'s `transcription_loop`:

```python
buf_duration = len(rolling_system) / SR
offset = transcriber._total_processed - buf_duration
seg_start_in_buf = seg.start_time - offset
# pad ±2s, clipped to [0, len(buf)/SR]
pad = 2.0
slice_start = max(0.0, seg_start_in_buf - pad)
slice_end = min(buf_duration, (seg.end_time - offset) + pad)
audio_slice = rolling_system[int(slice_start*SR) : int(slice_end*SR)]
# After whisper: trim word-timestamps back to [seg_start_in_buf - slice_start, seg_end_in_buf - slice_start]
```

If `seg_start_in_buf < 0` (segment older than the rolling window — possible
under heavy load), skip Whisper and mark `whisper_final = True` with the
original Apple text. Log a warning; this is a signal the worker is falling
behind.

### `app/main.py` — changes

- The existing `transcript` broadcast (currently lines 299–305) must include
  `seg.id` so the frontend can match updates against the original line. Add
  `"id": seg.id` to the broadcast payload.
- `WebSocket` handler adds a `transcript_update` message type. The frontend
  finds the line by `data-segment-id` and updates the `.text` span only —
  speaker label, timestamps, and DOM ordering are untouched.
- `transcription_loop` thought-boundary block becomes:
  ```python
  if should_detect:
      detection_pending = False
      batch = unseen_segments
      unseen_segments = []
      await transcriber.wait_whisper_final(batch, timeout=config.WHISPER_GATE_TIMEOUT_SECONDS)
      # Reset the pause clock so we don't immediately re-fire on the next iteration
      # against new segments that arrived during the wait.
      last_speech_time = time.time()
      new_text = _format_segments_for_prompt(batch)              # now uses upgraded text
      ...
  ```
- End-of-meeting flush ordering (in the stop-session path):
  ```python
  # 1. Stop accepting new audio (cancel audio callbacks).
  audio_capture.stop()
  # 2. Cancel transcribe_task — but NOT the whisper worker.
  transcribe_task.cancel()
  # 3. Drain the whisper queue. The worker keeps running.
  await transcriber.flush_whisper(timeout=config.WHISPER_FLUSH_TIMEOUT_SECONDS)
  # 4. Now generate the summary against upgraded segments.
  segments = transcriber.get_all_segments()
  summary = await summary_gen.generate(segments, questions, meeting_title)
  # 5. Finally, shut down the whisper worker.
  transcriber.shutdown()
  ```
  If `flush_whisper` times out, the summary uses whatever segments finished —
  un-upgraded segments retain their Apple text. Logged at WARNING.

### `static/app.js` — changes

`appendTranscript` (currently builds `<div class="transcript-line">` with
`<span class="text">` inside) must:
- Stamp the line with `line.dataset.segmentId = msg.id` on the initial `transcript` message.
- Add a handler for `transcript_update`:
  ```js
  case "transcript_update": {
      const line = document.querySelector(`.transcript-line[data-segment-id="${msg.id}"]`);
      if (!line) return;
      const textEl = line.querySelector(".text");
      if (textEl) textEl.textContent = msg.text;
      break;
  }
  ```
Speaker label, timestamp, and DOM ordering are not touched.

### `requirements.txt`

```
mlx-whisper>=0.4.0
```

### `config.py`

```python
WHISPER_ENABLED = os.getenv("WHISPER_ENABLED", "1") == "1"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
WHISPER_MIN_SEGMENT_SECONDS = float(os.getenv("WHISPER_MIN_SEGMENT_SECONDS", "1.0"))
WHISPER_PAD_SECONDS = float(os.getenv("WHISPER_PAD_SECONDS", "2.0"))
# Worst-case wait: ~4 segments × 0.5s = 2s on a healthy worker. 5s gives headroom under load.
WHISPER_GATE_TIMEOUT_SECONDS = float(os.getenv("WHISPER_GATE_TIMEOUT_SECONDS", "5.0"))
# Worst-case flush: 100 backed-up segments × 0.5s = 50s. 60s gives headroom; over that we accept partial Apple-text fallback.
WHISPER_FLUSH_TIMEOUT_SECONDS = float(os.getenv("WHISPER_FLUSH_TIMEOUT_SECONDS", "60.0"))
```

## Failure Modes

| Failure | Behavior |
|---|---|
| `mlx-whisper` import fails (not installed, non-Apple-Silicon) OR `WHISPER_ENABLED=0` | `Transcriber.initialize()` logs INFO and skips worker creation. `_whisper_worker` stays `None`. Every segment is marked `whisper_final=True` and its event signaled at append time. Detector never blocks. |
| Whisper worker exception on a segment | Logged. `finally` block in `_loop` always sets `whisper_final=True` and signals the event with Apple text retained. |
| Worker falling behind (queue depth ≥ 10) | `_monitor_queue` logs WARNING every 10s. Segments aged beyond the rolling buffer window skip Whisper (Apple text retained). |
| Detector wait timeout (`WHISPER_GATE_TIMEOUT_SECONDS`) | Proceed with whatever text is available — some segments may still be Apple-text. Logged at INFO. |
| Flush timeout at end-of-session (`WHISPER_FLUSH_TIMEOUT_SECONDS`) | Summary uses whatever finished. Un-upgraded segments retain Apple text. Logged at WARNING. |
| Cold-start latency (first model load + cold cache: model download ~1.5GB on first ever launch, then 2–5s load from disk on subsequent launches) | First-launch download is shown as a startup status. Subsequent launches: silent warmup with 1s of silence in `WhisperWorker.initialize()` before live audio is accepted into the queue. |

## Non-Goals

- VAD-based chunking (Apple sidecar already does utterance segmentation).
- pyannote.audio diarization (existing speaker-ID layer is sufficient).
- Multi-language support (English-only matches current behavior).
- Streaming partial Whisper results (the live partial UX is provided by
  Apple's sidecar; Whisper runs as background final-pass only).
- Cloud STT alternatives (Deepgram, AssemblyAI). Local-only by design.

## Test Plan

1. **Cold-start**: launch app, confirm warmup runs, confirm first real segment lands within ~1s after warmup completes.
2. **Accuracy A/B**: run a 10-minute meeting recording (saved WAV) twice — once with `WHISPER_ENABLED=0`, once `=1`. Diff transcripts. Expect substantial improvement on names/jargon.
3. **Latency**: instrument `_process` with start/end timestamps; confirm p95 < 1s on 5s segments on M2 Pro.
4. **Detector gating**: insert a `time.sleep(2.0)` in `_process` to simulate slow Whisper; confirm thought-boundary correctly waits and that `WHISPER_GATE_TIMEOUT_SECONDS` cap kicks in if exceeded. Confirm `last_speech_time` reset prevents spurious immediate re-fire.
5. **Padding**: assert `_process` calls `mlx_whisper.transcribe` with audio whose duration ≈ `(seg.end - seg.start) + 2 × WHISPER_PAD_SECONDS` (or less near buffer head). Mock `transcribe` and capture call args.
6. **Speaker preservation**: emit a `transcript_update` for a segment with `speaker="Alice"`; assert the line's speaker label is unchanged after the swap.
7. **Summary parity**: end a meeting, inspect the generated `data/summaries/*.md` — segment text should be Whisper output, not Apple.
8. **Failure injection — import**: set `WHISPER_ENABLED=0`; confirm app runs in Apple-only mode and detector never blocks.
9. **Failure injection — runtime**: throw inside `_process`; confirm `whisper_event.set()` still fires (finally block) and detector unblocks.
10. **Short-segment skip**: confirm segments < `WHISPER_MIN_SEGMENT_SECONDS` keep their Apple text and don't get fed to Whisper (check logs and `seg.whisper_event.is_set()` immediately).
11. **Aged-out segment**: synthetically age a segment beyond the 60s rolling buffer; confirm it skips Whisper, retains Apple text, logs the warning.
12. **Queue backlog warning**: stuff 15 segments into the queue with `_process` stubbed to sleep 1s; confirm the monitor logs the backlog warning within 10s.

## Rollout

1. Implement behind a kill switch: `WHISPER_ENABLED = os.getenv("WHISPER_ENABLED", "1") == "1"`. Setting to `0` reverts to current behavior at startup.
2. Land the implementation, dogfood for a few meetings, compare summary quality.
3. Once stable, default `WHISPER_ENABLED=1` and keep the env knob for emergencies.

## Open Questions

None — recommended decisions accepted in brainstorming. Three review passes
flagged blockers around `path_or_hf_repo`, atomic rolling-buffer snapshotting,
worker lifecycle vs. session shutdown, and `wait_whisper_final` synchronization;
all addressed inline above.
