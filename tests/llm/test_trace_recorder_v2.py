import pytest


@pytest.mark.asyncio
async def test_trace_recorder_collects_completed_spans_for_aggregation():
    from src.llm.tracing_v2 import TraceContext, TraceRecorderSink

    t = 100

    def now_ms():
        return t

    recorder = TraceRecorderSink()
    trace = TraceContext(session_id="s1", trace_id="t1", sink=recorder, now_ms=now_ms)

    async with trace.span("tool_selection.total"):
        t += 5

    async with trace.span("tool.get_product_docs"):
        t += 7

    completed = recorder.to_completed_trace(session_id="s1", trace_id="t1")
    assert completed["session_id"] == "s1"
    assert completed["trace_id"] == "t1"

    spans = {s["name"]: s for s in completed["spans"]}
    assert spans["tool_selection.total"]["duration_ms"] == 5
    assert spans["tool.get_product_docs"]["duration_ms"] == 7


@pytest.mark.asyncio
async def test_composite_sink_fans_out_payloads():
    from src.llm.tracing_v2 import CompositeTraceSink, TraceContext

    events_a = []
    events_b = []

    class _Sink:
        def __init__(self, store):
            self._store = store

        async def emit(self, payload):
            self._store.append(payload)

    sink = CompositeTraceSink([_Sink(events_a), _Sink(events_b)])

    t = 0

    def now_ms():
        return t

    trace = TraceContext(session_id="s", trace_id="tr", sink=sink, now_ms=now_ms)
    await trace.emit_event("utterance.finalized")

    assert len(events_a) == 1
    assert len(events_b) == 1
    assert events_a[0]["event"]["name"] == "utterance.finalized"
