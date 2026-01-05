import pytest


@pytest.mark.asyncio
async def test_websocket_sink_emits_trace_event_envelope():
    from unittest.mock import AsyncMock

    from src.gateway.tracing_protocol_v2 import WebSocketTraceSink

    websocket = AsyncMock()
    sink = WebSocketTraceSink(websocket=websocket, session_id="session-1", trace_id="trace-1")

    await sink.emit(
        {
            "event": {
                "kind": "event",
                "ts_ms": 123,
                "name": "utterance.finalized",
                "attrs": {"text_len": 12},
            }
        }
    )

    websocket.send_json.assert_awaited_once()
    payload = websocket.send_json.await_args.args[0]

    assert payload["type"] == "trace_event"
    assert payload["session_id"] == "session-1"
    assert payload["trace_id"] == "trace-1"
    assert payload["event"]["kind"] == "event"
    assert payload["event"]["name"] == "utterance.finalized"


def test_parse_hello_requires_session_id_and_sets_v2_version():
    from src.gateway.tracing_protocol_v2 import parse_hello_message

    msg = {"type": "hello", "session_id": "abc", "client_ts": 1, "client_version": "v2"}
    parsed = parse_hello_message(msg)

    assert parsed.session_id == "abc"
    assert parsed.client_version == "v2"

    with pytest.raises(ValueError):
        parse_hello_message({"type": "hello"})
