from src.llm.tracing_v2 import SessionTraceAggregator


def test_session_trace_aggregator_includes_tts_metrics():
    agg = SessionTraceAggregator(session_id="s1")

    trace = {
        "session_id": "s1",
        "trace_id": "t1",
        "spans": [
            {"name": "asr.transcription", "duration_ms": 100, "status": "ok", "attrs": {}},
            {"name": "llm.chat_completion", "duration_ms": 200, "status": "ok", "attrs": {}},
            {"name": "tts.synthesize", "duration_ms": 30, "status": "ok", "attrs": {}},
            {"name": "tts.synthesize", "duration_ms": 20, "status": "ok", "attrs": {}},
            {"name": "tts.total", "duration_ms": 80, "status": "ok", "attrs": {}},
        ],
    }

    agg.add_completed_trace(trace)
    summary = agg.get_session_summary()

    stages = summary.get("stages", {})

    assert "tts.total_ms" in stages
    assert stages["tts.total_ms"]["count"] == 1
    assert stages["tts.total_ms"]["mean_ms"] == 80

    assert "tts.synthesize_ms" in stages
    assert stages["tts.synthesize_ms"]["mean_ms"] == 50

    # End-to-end with TTS should reflect addition
    assert stages["turn.total_ms"]["mean_ms"] == 300
    assert stages["turn.total_with_tts_ms"]["mean_ms"] == 380
