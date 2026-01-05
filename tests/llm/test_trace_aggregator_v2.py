def _trace(*, session_id: str, trace_id: str, spans: list[dict]):
    return {"session_id": session_id, "trace_id": trace_id, "spans": spans}


def _span(name: str, duration_ms: int, **attrs):
    return {
        "name": name,
        "duration_ms": duration_ms,
        "status": "ok",
        "attrs": attrs,
    }


def _tool_span(tool: str, duration_ms: int):
    return _span(f"tool.{tool}", duration_ms, tool=tool)


def test_session_aggregator_rolls_up_turns_and_computes_percentiles_and_tool_critical_path():
    from src.llm.tracing_v2 import SessionTraceAggregator

    agg = SessionTraceAggregator(session_id="session-1")

    t1 = _trace(
        session_id="session-1",
        trace_id="t1",
        spans=[
            _span("tool_selection.total", 100, model="gpt-5"),
            _span("tools.parallel_total", 1000),
            _tool_span("get_product_docs", 1000),
            _tool_span("get_raw_docs", 200),
            _span("llm.chat_completion", 200, model="gpt-5"),
        ],
    )
    t2 = _trace(
        session_id="session-1",
        trace_id="t2",
        spans=[
            _span("tool_selection.total", 150, model="gpt-5"),
            _span("tools.parallel_total", 500),
            _tool_span("get_product_docs", 500),
            _tool_span("get_raw_docs", 480),
            _span("llm.chat_completion", 250, model="gpt-5"),
        ],
    )
    t3 = _trace(
        session_id="session-1",
        trace_id="t3",
        spans=[
            _span("tool_selection.total", 90, model="gpt-5"),
            _span("tools.parallel_total", 800),
            _tool_span("get_product_docs", 800),
            _span("llm.chat_completion", 210, model="gpt-5"),
        ],
    )

    agg.add_completed_trace(t1)
    agg.add_completed_trace(t2)
    agg.add_completed_trace(t3)

    summary = agg.get_session_summary()

    assert summary["session_id"] == "session-1"
    assert summary["turn_count"] == 3

    totals = summary["stages"]["turn.total_ms"]
    assert totals["min_ms"] == 900
    assert totals["max_ms"] == 1300
    assert totals["p50_ms"] == 1100
    assert totals["p95_ms"] == 1300

    tools = summary["tools"]
    assert tools["critical_path"][0]["tool"] == "get_product_docs"
    assert tools["critical_path"][0]["count"] == 3


def test_parallel_tool_total_can_be_derived_from_child_tool_spans_when_missing():
    from src.llm.tracing_v2 import SessionTraceAggregator

    agg = SessionTraceAggregator(session_id="session-1")

    t1 = _trace(
        session_id="session-1",
        trace_id="t1",
        spans=[
            _span("tool_selection.total", 100, model="gpt-5"),
            _tool_span("get_product_docs", 1000),
            _tool_span("get_raw_docs", 200),
            _span("llm.chat_completion", 200, model="gpt-5"),
        ],
    )

    agg.add_completed_trace(t1)
    summary = agg.get_session_summary()

    assert summary["stages"]["tools.parallel_total_ms"]["max_ms"] == 1000
    assert summary["stages"]["turn.total_ms"]["max_ms"] == 1300
