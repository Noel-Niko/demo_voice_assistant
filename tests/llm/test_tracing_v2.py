import pytest


class _FakeClock:
    def __init__(self, start_ms: int = 0):
        self._ms = start_ms

    def now_ms(self) -> int:
        return self._ms

    def advance(self, delta_ms: int) -> None:
        self._ms += delta_ms


class _CollectingSink:
    def __init__(self):
        self.events = []

    async def emit(self, payload):
        self.events.append(payload)


@pytest.mark.asyncio
async def test_trace_emits_span_started_and_ended_with_duration_and_parenting():
    from src.llm.tracing_v2 import TraceContext

    clock = _FakeClock(start_ms=1000)
    sink = _CollectingSink()

    trace = TraceContext(
        session_id="session-1",
        trace_id="trace-1",
        sink=sink,
        now_ms=clock.now_ms,
    )

    await trace.emit_event("utterance.finalized", attrs={"text_len": 12})

    async with trace.span("tool_selection.total", attrs={"model": "gpt-5"}) as parent:
        clock.advance(5)
        async with trace.span("tool_selection.cache_lookup") as child:
            clock.advance(7)

        clock.advance(3)

    assert parent.name == "tool_selection.total"
    assert child.name == "tool_selection.cache_lookup"

    started = [e for e in sink.events if e["event"]["kind"] == "span_started"]
    ended = [e for e in sink.events if e["event"]["kind"] == "span_ended"]
    events = [e for e in sink.events if e["event"]["kind"] == "event"]

    assert len(events) == 1
    assert events[0]["event"]["name"] == "utterance.finalized"

    assert len(started) == 2
    assert len(ended) == 2

    ended_by_name = {e["event"]["span"]["name"]: e for e in ended}

    assert ended_by_name["tool_selection.cache_lookup"]["event"]["span"]["duration_ms"] == 7
    assert ended_by_name["tool_selection.total"]["event"]["span"]["duration_ms"] == 15

    child_span = ended_by_name["tool_selection.cache_lookup"]["event"]["span"]
    parent_span = ended_by_name["tool_selection.total"]["event"]["span"]
    assert child_span["parent_span_id"] == parent_span["span_id"]


@pytest.mark.asyncio
async def test_span_end_status_error_is_reported():
    from src.llm.tracing_v2 import TraceContext

    clock = _FakeClock(start_ms=0)
    sink = _CollectingSink()

    trace = TraceContext(
        session_id="session-1",
        trace_id="trace-1",
        sink=sink,
        now_ms=clock.now_ms,
    )

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        async with trace.span("tool.get_product_docs"):
            clock.advance(4)
            raise _Boom("fail")

    ended = [e for e in sink.events if e["event"]["kind"] == "span_ended"]
    assert len(ended) == 1
    assert ended[0]["event"]["span"]["status"] == "error"
    assert ended[0]["event"]["span"]["duration_ms"] == 4
