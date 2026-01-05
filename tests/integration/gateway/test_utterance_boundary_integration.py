import asyncio
import tempfile
import threading
import time

import pytest

from src.asr.base import TranscriptEvent
from src.gateway.audio_session import AudioSession


class FakeWebSocket:
    def __init__(self):
        from starlette.websockets import WebSocketState

        self.client_state = WebSocketState.CONNECTED
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class FakeStreamingASR:
    def __init__(self):
        self.transcription_callback = None
        self.interim_callback = None
        self.emit_events = False

    def initialize(self, **kwargs) -> None:
        return

    def start_streaming(
        self, transcription_callback, interim_callback=None, *, emit_events=False
    ) -> None:
        self.transcription_callback = transcription_callback
        self.interim_callback = interim_callback
        self.emit_events = emit_events

    def stream_audio(self, audio_chunk):
        return

    def stop_streaming(self) -> None:
        return

    def emit_events_in_thread(
        self, events: list[TranscriptEvent], delay_s: float = 0.0
    ) -> threading.Thread:
        def _runner():
            for e in events:
                if delay_s:
                    time.sleep(delay_s)

                if e.speech_event_type and (self.interim_callback is not None):
                    self.interim_callback(e)
                    continue

                if e.is_final:
                    self.transcription_callback(e)
                else:
                    if self.interim_callback is not None:
                        self.interim_callback(e)

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        return t


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_audio_session_finalizes_utterance_with_events(monkeypatch):
    monkeypatch.setenv("UTT_SHORT_TIMEOUT_S", "0.01")
    monkeypatch.setenv("UTT_MEDIUM_TIMEOUT_S", "0.02")
    monkeypatch.setenv("UTT_LONG_TIMEOUT_S", "0.2")

    fake_asr = FakeStreamingASR()

    # Patch the ASRFactory used inside AudioSession to return our fake.
    monkeypatch.setattr(
        "src.gateway.audio_session.ASRFactory.create_provider", lambda *_args, **_kwargs: fake_asr
    )

    ws = FakeWebSocket()
    loop = asyncio.get_running_loop()
    from src.gateway.llm_integration import LLMIntegrationManager

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        llm_mgr = await LLMIntegrationManager.create(database_path=tmp.name)

    session = AudioSession(ws, loop, llm_manager=llm_mgr)

    try:
        now = time.time()
        events = [
            TranscriptEvent(
                text="",
                is_final=False,
                confidence=0.0,
                stability=0.0,
                received_time=now,
                speech_event_type="SPEECH_ACTIVITY_START",
            ),
            TranscriptEvent(
                text="what time is it",
                is_final=False,
                confidence=0.95,
                stability=0.0,
                received_time=now + 0.01,
            ),
            TranscriptEvent(
                text="what time is it",
                is_final=True,
                confidence=0.95,
                stability=0.0,
                received_time=now + 0.02,
            ),
        ]

        t = fake_asr.emit_events_in_thread(events)
        t.join(timeout=1.0)

        await asyncio.sleep(0.08)

        types = [m.get("type") for m in ws.sent]
        assert "partial_transcript" in types
        assert "final_transcript" in types

        final_msgs = [m for m in ws.sent if m.get("type") == "final_transcript"]
        assert len(final_msgs) == 1
        assert final_msgs[0].get("text") == "what time is it"

        trace_events = [m for m in ws.sent if m.get("type") == "trace_event"]
        assert any(
            (m.get("event") or {}).get("kind") == "event"
            and (m.get("event") or {}).get("name") == "utterance.finalized"
            for m in trace_events
        )

    finally:
        session.cleanup()
        # Ensure background cleanup task is canceled to avoid hanging event loop
        await llm_mgr.shutdown()
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_audio_session_finalizes_on_speech_end_event(monkeypatch):
    monkeypatch.setenv("UTT_MEDIUM_TIMEOUT_S", "0.02")
    monkeypatch.setenv("UTT_LONG_TIMEOUT_S", "0.2")

    fake_asr = FakeStreamingASR()
    monkeypatch.setattr(
        "src.gateway.audio_session.ASRFactory.create_provider", lambda *_args, **_kwargs: fake_asr
    )

    ws = FakeWebSocket()
    loop = asyncio.get_running_loop()
    from src.gateway.llm_integration import LLMIntegrationManager

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        llm_mgr = await LLMIntegrationManager.create(database_path=tmp.name)

    session = AudioSession(ws, loop, llm_manager=llm_mgr)

    try:
        now = time.time()
        events = [
            TranscriptEvent(
                text="hello there",
                is_final=False,
                confidence=0.8,
                stability=0.0,
                received_time=now,
            ),
            TranscriptEvent(
                text="",
                is_final=False,
                confidence=0.0,
                stability=0.0,
                received_time=now + 0.01,
                speech_event_type="SPEECH_ACTIVITY_END",
            ),
        ]

        t = fake_asr.emit_events_in_thread(events)
        t.join(timeout=1.0)

        await asyncio.sleep(0.08)

        final_msgs = [m for m in ws.sent if m.get("type") == "final_transcript"]
        assert len(final_msgs) == 1
        assert final_msgs[0].get("text") == "hello there"

        trace_events = [m for m in ws.sent if m.get("type") == "trace_event"]
        assert any(
            (m.get("event") or {}).get("kind") == "event"
            and (m.get("event") or {}).get("name") == "utterance.finalized"
            for m in trace_events
        )

    finally:
        session.cleanup()
        # Ensure background cleanup task is canceled to avoid hanging event loop
        await llm_mgr.shutdown()
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_audio_session_interrupts_in_flight_action(monkeypatch):
    monkeypatch.setenv("UTT_SHORT_TIMEOUT_S", "0.01")
    monkeypatch.setenv("UTT_MEDIUM_TIMEOUT_S", "0.02")
    monkeypatch.setenv("UTT_LONG_TIMEOUT_S", "0.2")

    fake_asr = FakeStreamingASR()
    monkeypatch.setattr(
        "src.gateway.audio_session.ASRFactory.create_provider", lambda *_args, **_kwargs: fake_asr
    )

    ws = FakeWebSocket()
    loop = asyncio.get_running_loop()
    from src.gateway.llm_integration import LLMIntegrationManager

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        llm_mgr = await LLMIntegrationManager.create(database_path=tmp.name)

    session = AudioSession(ws, loop, llm_manager=llm_mgr)

    async def slow_action(_: str, _confidence: float):
        await session._send_asr_state("thinking")
        try:
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            await session._send_asr_state("canceled")
            raise

    # Replace the downstream action so we can interrupt it.
    session._process_final_utterance = slow_action  # type: ignore[method-assign]

    try:
        now = time.time()
        events = [
            TranscriptEvent(
                text="what time is it",
                is_final=True,
                confidence=0.95,
                stability=0.0,
                received_time=now,
            ),
        ]

        t = fake_asr.emit_events_in_thread(events)
        t.join(timeout=1.0)

        # Allow action task to start
        # Allow a bit more time for cancellation to propagate on CI
        await asyncio.sleep(0.1)

        # User starts speaking again -> interruption should cancel in-flight.
        t2 = fake_asr.emit_events_in_thread(
            [
                TranscriptEvent(
                    text="",
                    is_final=False,
                    confidence=0.0,
                    stability=0.0,
                    received_time=time.time(),
                    speech_event_type="SPEECH_ACTIVITY_START",
                )
            ]
        )
        t2.join(timeout=1.0)

        await asyncio.sleep(0.05)

        assert session._in_flight_action_task is not None
        assert session._in_flight_action_task.cancelled() or session._in_flight_action_task.done()

    finally:
        # Cancel any lingering in-flight task defensively
        if session._in_flight_action_task and not session._in_flight_action_task.done():
            session._in_flight_action_task.cancel()
            try:
                await asyncio.wait_for(session._in_flight_action_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        session.cleanup()
        # Ensure background cleanup task is canceled to avoid hanging event loop
        await llm_mgr.shutdown()
        await asyncio.sleep(0)
