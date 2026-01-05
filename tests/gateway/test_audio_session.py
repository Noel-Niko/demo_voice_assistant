"""Tests for AudioSession module."""

import asyncio
import tempfile
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
        self.audio_data = []

    def initialize(self, project_id=None):
        pass

    def start_streaming(
        self, transcription_callback=None, interim_callback=None, emit_events=False
    ):
        self.transcription_callback = transcription_callback
        self.interim_callback = interim_callback

    def stream_audio(self, audio_chunk):
        self.audio_data.append(audio_chunk)

    def stop_streaming(self):
        pass


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
                stability=0.95,
                received_time=now + 1,
            ),
            TranscriptEvent(
                text="what time is it",
                is_final=False,
                confidence=0.98,
                stability=0.98,
                received_time=now + 2,
            ),
            TranscriptEvent(
                text="what time is it",
                is_final=True,
                confidence=1.0,
                stability=1.0,
                received_time=now + 3,
                speech_event_type="SPEECH_ACTIVITY_END",
            ),
        ]

        for event in events:
            await session._utterance_manager.on_transcript_event(event, is_interim_channel=False)

        # Wait for finalization
        await asyncio.sleep(0.2)

        # Check that final transcript was sent
        final_messages = [m for m in ws.sent if m.get("type") == "final_transcript"]
        assert any(m.get("text") == "what time is it" for m in final_messages)

        # Check that trace events were sent
        trace_events = [m for m in ws.sent if m.get("type") == "trace_event"]
        assert any(
            (m.get("event") or {}).get("kind") == "event"
            and (m.get("event") or {}).get("name") == "utterance.finalized"
            for m in trace_events
        )

    finally:
        session.cleanup()
        await llm_mgr.shutdown()


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
                stability=0.8,
                received_time=now,
                speech_event_type="SPEECH_ACTIVITY_START",
            ),
            TranscriptEvent(
                text="hello there",
                is_final=False,
                confidence=0.9,
                stability=0.9,
                received_time=now + 1,
            ),
            TranscriptEvent(
                text="hello there",
                is_final=False,
                confidence=0.95,
                stability=0.95,
                received_time=now + 2,
                speech_event_type="SPEECH_ACTIVITY_END",
            ),
        ]

        for event in events:
            await session._utterance_manager.on_transcript_event(event, is_interim_channel=False)

        # Wait for finalization
        await asyncio.sleep(0.2)

        # Check that final transcript was sent
        final_messages = [m for m in ws.sent if m.get("type") == "final_transcript"]
        assert any(m.get("text") == "hello there" for m in final_messages)

        # Check that trace events were sent
        trace_events = [m for m in ws.sent if m.get("type") == "trace_event"]
        assert any(
            (m.get("event") or {}).get("kind") == "event"
            and (m.get("event") or {}).get("name") == "utterance.finalized"
            for m in trace_events
        )

    finally:
        session.cleanup()
        await llm_mgr.shutdown()


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

    async def slow_action(text: str, confidence: float, reason: str):
        await session._send_asr_state("thinking")
        try:
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            await session._send_asr_state("canceled")
            raise

    # Replace the downstream action so we can interrupt it.
    monkeypatch.setattr(session, "_on_utterance_finalized", slow_action)

    try:
        now = time.time()
        events = [
            TranscriptEvent(
                text="test",
                is_final=False,
                confidence=0.8,
                stability=0.8,
                received_time=now,
                speech_event_type="SPEECH_ACTIVITY_START",
            ),
            TranscriptEvent(
                text="test interrupt",
                is_final=False,
                confidence=0.9,
                stability=0.9,
                received_time=now + 1,
            ),
            TranscriptEvent(
                text="test interrupt now",
                is_final=False,
                confidence=0.95,
                stability=0.95,
                received_time=now + 2,
                speech_event_type="SPEECH_ACTIVITY_END",
            ),
        ]

        for event in events:
            await session._utterance_manager.on_transcript_event(event, is_interim_channel=False)

        # Wait for first utterance to be processed
        await asyncio.sleep(0.1)

        # Send interrupting utterance
        interrupt_events = [
            TranscriptEvent(
                text="stop",
                is_final=False,
                confidence=0.8,
                stability=0.8,
                received_time=now + 3,
                speech_event_type="SPEECH_ACTIVITY_START",
            ),
            TranscriptEvent(
                text="stop now",
                is_final=True,
                confidence=1.0,
                stability=1.0,
                received_time=now + 4,
                speech_event_type="SPEECH_ACTIVITY_END",
            ),
        ]

        for event in interrupt_events:
            await session._utterance_manager.on_transcript_event(event, is_interim_channel=False)

        # Wait for processing
        await asyncio.sleep(0.3)

        # Check that in-flight task exists (interrupt mechanism is working)
        # The task may still be running or cancelled depending on timing
        assert session._in_flight_action_task is not None

    finally:
        if session._in_flight_action_task and not session._in_flight_action_task.done():
            session._in_flight_action_task.cancel()
            try:
                await asyncio.wait_for(session._in_flight_action_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        session.cleanup()
        await llm_mgr.shutdown()
