from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.tracing_v2 import TraceContext, TraceRecorderSink
from src.tts.response_reader import ResponseReader


@pytest.mark.asyncio
async def test_response_reader_emits_tts_spans():
    """ResponseReader should emit tts spans when a trace is active."""
    # Mock TTS provider and websocket
    provider = MagicMock()
    provider.synthesize = AsyncMock(return_value=b"audio_bytes")

    ws = MagicMock()
    ws.send_json = AsyncMock()

    rr = ResponseReader(provider, ws)

    # Prepare trace recorder context
    recorder = TraceRecorderSink()
    trace = TraceContext(
        session_id="s1",
        trace_id="t1",
        sink=recorder,
        now_ms=lambda: 1234567890,
    )

    # Run under trace
    from src.llm.tracing_v2 import use_trace

    async with use_trace(trace):
        await rr.read_response_chunked("Hello world.")

    # Collect completed spans
    completed = recorder.to_completed_trace(session_id="s1", trace_id="t1")
    span_names = [s.get("name") for s in completed.get("spans", [])]

    assert any(n == "tts.total" for n in span_names), "Missing tts.total span"
    assert any(n == "tts.synthesize" for n in span_names), "Missing tts.synthesize span"
