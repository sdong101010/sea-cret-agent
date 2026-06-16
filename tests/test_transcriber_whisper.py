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


@pytest.mark.asyncio
async def test_wait_whisper_final_returns_immediately_if_all_final():
    import time
    from app.transcriber import Transcriber

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
    from app.transcriber import Transcriber

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
    import time
    from app.transcriber import Transcriber

    t = Transcriber()
    seg = TranscriptSegment(text="x", start_time=0, end_time=1)
    start = time.monotonic()
    await t.wait_whisper_final([seg], timeout=0.1)
    elapsed = time.monotonic() - start
    assert 0.08 <= elapsed <= 0.5
    assert seg.whisper_final is False


@pytest.mark.asyncio
async def test_initialize_skips_worker_when_disabled(monkeypatch):
    from unittest.mock import MagicMock
    from app.transcriber import Transcriber
    import config

    monkeypatch.setattr(config, "WHISPER_ENABLED", False)
    t = Transcriber()
    # Patch sidecar startup so we don't actually spawn the binary.
    monkeypatch.setattr("app.transcriber._ensure_binary_built", lambda: None)
    fake_proc = MagicMock()
    fake_proc.pid = 1234
    fake_proc.stdout.readline.return_value = b""
    fake_proc.stderr.readline.return_value = b""
    monkeypatch.setattr("app.transcriber.subprocess.Popen", lambda *a, **kw: fake_proc)
    await t.initialize()
    assert t._whisper_worker is None
    t.shutdown()


@pytest.mark.asyncio
async def test_short_segment_is_marked_whisper_final_immediately():
    from app.transcriber import Transcriber

    t = Transcriber()
    # No worker, no real proc — just exercise the helper.
    seg = TranscriptSegment(text="yeah", start_time=0.0, end_time=0.5)
    t._mark_segment_final_fastpath(seg)
    assert seg.whisper_final is True
    assert seg.whisper_event.is_set()
