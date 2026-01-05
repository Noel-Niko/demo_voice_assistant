"""
Unit tests for analyze_query_node tool selection behavior.

This test verifies that the analyze_query_node properly calls tool_selector
and populates the selected_tools field in the state.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.agent_state import create_initial_state
from src.llm.langgraph_workflow import analyze_query_node

pytestmark = pytest.mark.asyncio


class TestAnalyzeQueryToolSelection:
    """Test that analyze_query_node properly selects tools."""

    async def test_analyze_query_selects_tools_for_product_query(self):
        """Test that product-related queries trigger tool selection."""
        # Create initial state
        state = create_initial_state("test_session", "test_user")
        state["current_transcript"] = "Recommend a hammer for me"

        # Mock dependencies
        openai_client = AsyncMock()
        tool_selector = AsyncMock()

        # Mock tool selector to return product search tool
        mock_tool = MagicMock()
        mock_tool.name = "get_product_docs"
        tool_selector.select_relevant_tools = AsyncMock(return_value=[mock_tool])

        # Call analyze_query_node
        result = await analyze_query_node(state, openai_client, tool_selector)

        # Verify tool selector was called
        tool_selector.select_relevant_tools.assert_called_once_with("Recommend a hammer for me")

        # Verify selected_tools is populated
        assert "selected_tools" in result
        assert len(result["selected_tools"]) == 1
        assert result["selected_tools"][0] == "get_product_docs"
        assert result["requires_clarification"] is False
        assert result["current_status"] == "analyzed"

    async def test_analyze_query_selects_no_tools_for_general_query(self):
        """Test that general queries don't trigger tool selection."""
        # Create initial state
        state = create_initial_state("test_session", "test_user")
        state["current_transcript"] = "What is the weather like today?"

        # Mock dependencies
        openai_client = AsyncMock()
        tool_selector = AsyncMock()

        # Mock tool selector to return no tools
        tool_selector.select_relevant_tools = AsyncMock(return_value=[])

        # Call analyze_query_node
        result = await analyze_query_node(state, openai_client, tool_selector)

        # Verify tool selector was called
        tool_selector.select_relevant_tools.assert_called_once_with(
            "What is the weather like today?"
        )

        # Verify selected_tools is empty
        assert "selected_tools" in result
        assert len(result["selected_tools"]) == 0
        assert result["requires_clarification"] is False
        assert result["current_status"] == "analyzed"

    async def test_analyze_query_requires_clarification_for_vague_query(self):
        """Test that vague queries trigger clarification without tool selection."""
        # Create initial state
        state = create_initial_state("test_session", "test_user")
        state["current_transcript"] = "hi"

        # Mock dependencies
        openai_client = AsyncMock()
        tool_selector = AsyncMock()

        # Call analyze_query_node
        result = await analyze_query_node(state, openai_client, tool_selector)

        # Verify tool selector was NOT called for vague queries
        tool_selector.select_relevant_tools.assert_not_called()

        # Verify clarification is needed
        assert result["requires_clarification"] is True
        assert "selected_tools" in result
        assert len(result["selected_tools"]) == 0

    async def test_analyze_query_selects_multiple_tools(self):
        """Test that queries can trigger multiple tool selections."""
        # Create initial state
        state = create_initial_state("test_session", "test_user")
        state["current_transcript"] = "Find me safety gloves and a hammer"

        # Mock dependencies
        openai_client = AsyncMock()
        tool_selector = AsyncMock()

        # Mock tool selector to return multiple tools
        mock_tool1 = MagicMock()
        mock_tool1.name = "get_product_docs"
        mock_tool2 = MagicMock()
        mock_tool2.name = "get_raw_docs"
        tool_selector.select_relevant_tools = AsyncMock(return_value=[mock_tool1, mock_tool2])

        # Call analyze_query_node
        result = await analyze_query_node(state, openai_client, tool_selector)

        # Verify selected_tools contains both tools
        assert "selected_tools" in result
        assert len(result["selected_tools"]) == 2
        assert "get_product_docs" in result["selected_tools"]
        assert "get_raw_docs" in result["selected_tools"]
        assert result["requires_clarification"] is False

    async def test_analyze_query_handles_tool_selector_error(self):
        """Test that errors in tool selection are handled gracefully."""
        # Create initial state
        state = create_initial_state("test_session", "test_user")
        state["current_transcript"] = "Recommend a hammer"

        # Mock dependencies
        openai_client = AsyncMock()
        tool_selector = AsyncMock()

        # Mock tool selector to raise an error
        tool_selector.select_relevant_tools = AsyncMock(
            side_effect=Exception("Tool selector error")
        )

        # Call analyze_query_node
        result = await analyze_query_node(state, openai_client, tool_selector)

        # Verify error is handled
        assert result["current_status"] == "error"
        assert "error_message" in result
        assert "Tool selector error" in result["error_message"]

    async def test_analyze_query_with_ladder_query(self):
        """Test specific query from logs: 'Recommend a rolling ladder'."""
        # Create initial state
        state = create_initial_state("test_session", "test_user")
        state["current_transcript"] = "Recommend a rolling ladder"

        # Mock dependencies
        openai_client = AsyncMock()
        tool_selector = AsyncMock()

        # Mock tool selector to return product search tool
        mock_tool = MagicMock()
        mock_tool.name = "get_product_docs"
        tool_selector.select_relevant_tools = AsyncMock(return_value=[mock_tool])

        # Call analyze_query_node
        result = await analyze_query_node(state, openai_client, tool_selector)

        # Verify tool selector was called
        tool_selector.select_relevant_tools.assert_called_once_with("Recommend a rolling ladder")

        # Verify selected_tools is populated (this should NOT be empty!)
        assert "selected_tools" in result
        assert len(result["selected_tools"]) == 1
        assert result["selected_tools"][0] == "get_product_docs"
        assert result["requires_clarification"] is False
