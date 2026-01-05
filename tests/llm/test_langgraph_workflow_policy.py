"""
Tests for policy additions in langgraph_workflow:
- validate_no_order_hallucination function
- comprehensive system instruction injection and response validation
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.agent_state import create_initial_state
from src.llm.langgraph_workflow import (
    generate_response_node,
    validate_no_order_hallucination,
)

pytestmark = pytest.mark.asyncio


class TestOrderHallucinationGuard:
    def test_validator_flags_order_terms(self):
        cases = [
            "Your order number is 12345.",
            "Based on recent shipments, you might need...",
            "Tracking info for your order...",
            "Customer number 555 shows last order shipped.",
            "UN 1993 placard and labels for vinyl packs of 2",
        ]
        for text in cases:
            ok, safe = validate_no_order_hallucination(text)
            assert ok is False
            assert "I cannot access order information" in safe

    def test_validator_allows_normal_text(self):
        ok, safe = validate_no_order_hallucination("Here are some general safety tips.")
        assert ok is True
        assert safe == "Here are some general safety tips."


class TestSystemInstructionInjection:
    async def test_injection_and_blocking_direct_llm(self):
        state = create_initial_state("s1", "u1")
        state["current_transcript"] = "Hello"
        state["tool_results"] = []

        # Mock LLM to try to produce order-related content
        mock_openai = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.content = "Your order number is 12345, shipped yesterday."
        mock_openai.chat_completion = AsyncMock(return_value=mock_resp)

        result = await generate_response_node(state, mock_openai)

        # Should use safe fallback
        assert "I cannot access order information" in result.get("response", "")

        # Verify injected system policy is present and replaces prior system messages
        called = mock_openai.chat_completion.call_args
        assert called is not None
        kwargs = called.kwargs
        messages = kwargs.get("messages") or called.args[0]
        assert messages[0]["role"] == "system"
        assert "CRITICAL POLICY - YOU MUST FOLLOW THIS" in messages[0]["content"]

    async def test_blocking_in_clarification_flow(self):
        state = create_initial_state("s2", "u2")
        state["current_transcript"] = "hi"
        state["requires_clarification"] = True

        mock_openai = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.content = "Do you mean your order status?"
        mock_openai.chat_completion = AsyncMock(return_value=mock_resp)

        result = await generate_response_node(state, mock_openai)
        assert "I cannot access order information" in result.get("response", "")

    async def test_blocking_in_tool_result_flow(self):
        state = create_initial_state("s3", "u3")
        state["current_transcript"] = "Find gloves"
        state["tool_results"] = [
            {"tool_name": "get_product_docs", "success": True, "content": "Found 3"}
        ]

        mock_openai = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.content = "Based on your recent order, here are matches."
        mock_openai.chat_completion = AsyncMock(return_value=mock_resp)

        result = await generate_response_node(state, mock_openai)
        assert "I cannot access order information" in result.get("response", "")
