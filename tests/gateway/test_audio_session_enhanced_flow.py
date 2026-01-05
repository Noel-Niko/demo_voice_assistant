import asyncio
import tempfile
import time

import pytest

from src.asr.base import StreamingASR, TranscriptEvent
from src.gateway.audio_session import AudioSession


class FakeWebSocket:
    def __init__(self):
        from starlette.websockets import WebSocketState

        self.client_state = WebSocketState.CONNECTED
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class FakeStreamingASR(StreamingASR):
    def __init__(self):
        self.transcription_callback = None
        self.interim_callback = None

    def initialize(self, **kwargs):
        return

    def start_streaming(
        self, transcription_callback=None, interim_callback=None, emit_events=False
    ):
        self.transcription_callback = transcription_callback
        self.interim_callback = interim_callback

    def stop_streaming(self):
        return


@pytest.mark.asyncio
async def test_audio_session_sends_assistant_response_and_tts_spans(monkeypatch):
    # Ensure TTS enabled
    monkeypatch.setenv("TTS_ENABLED", "true")
    # Speed up utterance finalization for test stability
    monkeypatch.setenv("UTT_MEDIUM_TIMEOUT_S", "0.05")
    monkeypatch.setenv("UTT_DISABLE_SPACY_SEMANTIC", "1")

    # Patch ASRFactory to return our fake streaming ASR
    monkeypatch.setattr(
        "src.gateway.audio_session.ASRFactory.create_provider", lambda *_a, **_k: FakeStreamingASR()
    )

    # Patch GoogleTTSProvider used in AudioSession to avoid real API
    class _FakeTTSProvider:
        def __init__(self, *a, **k):
            pass

        async def synthesize(self, text: str) -> bytes:
            return b"abcd"  # small PCM payload

    monkeypatch.setattr("src.gateway.audio_session.GoogleTTSProvider", _FakeTTSProvider)

    # Patch LLMIntegrationManager.get_or_create_session to return a fake integration
    class _FakeLLMIntegration:
        async def process_final_utterance(self, transcript: str, confidence: float):
            return {"response": "Hello there."}

    ws = FakeWebSocket()
    loop = asyncio.get_running_loop()
    from src.gateway.llm_integration import LLMIntegrationManager

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        llm_mgr = await LLMIntegrationManager.create(database_path=tmp.name)

    async def _fake_get_or_create_session(session_id, websocket, asr_provider=None):
        return _FakeLLMIntegration()

    monkeypatch.setattr(llm_mgr, "get_or_create_session", _fake_get_or_create_session)

    session = AudioSession(ws, loop, llm_manager=llm_mgr)

    try:
        now = time.time()
        # Send a start-of-speech event then a final transcript to ensure finalization path
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
                text="hello",
                is_final=True,
                confidence=0.9,
                stability=0.0,
                received_time=now + 0.01,
            ),
        ]

        # Drive finalization path directly via utterance manager
        for ev in events:
            await session._utterance_manager.on_transcript_event(ev, is_interim_channel=False)

        # Wait up to 1.5s for assistant_response or trace_completed to arrive
        deadline = time.time() + 1.5
        while time.time() < deadline:
            types = [m.get("type") for m in ws.sent]
            if ("assistant_response" in types) or ("trace_completed" in types):
                break
            await asyncio.sleep(0.02)

        # Extract message order indices
        types = [m.get("type") for m in ws.sent]

        # Prefer enhanced flow with assistant_response; fall back to trace_completed if not present
        if "assistant_response" in types:
            assert "tts_complete" in types
            assert "trace_completed" in types

            idx_assistant = types.index("assistant_response")
            idx_tts_complete = types.index("tts_complete")
            idx_trace_completed = types.index("trace_completed")

            # Assistant text should be sent before TTS completes and trace completes
            assert idx_assistant < idx_tts_complete <= idx_trace_completed
        else:
            # At minimum, the turn should complete
            assert "trace_completed" in types

        # Ensure at least one tts span ended event was emitted (when TTS is enabled)
        tts_span_events = [
            m
            for m in ws.sent
            if m.get("type") == "trace_event"
            and (m.get("event") or {}).get("kind") == "span_ended"
            and ((m.get("event") or {}).get("span") or {}).get("name", "").startswith("tts.")
        ]
        assert tts_span_events, "Expected TTS span events in trace"

    finally:
        session.cleanup()
        await llm_mgr.shutdown()
