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
