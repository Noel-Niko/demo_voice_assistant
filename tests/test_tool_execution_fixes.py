"""
Unit tests for tool execution orchestrator fixes.
Tests the vector_store parameter fix and tool execution logic.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.llm.agentic_system import ToolExecutionOrchestrator, ToolExecutionPlan
from src.llm.tool_manager import ToolResult


class TestToolExecutionOrchestrator:
    """Test ToolExecutionOrchestrator with vector_store parameter fixes."""

    @pytest.fixture
    def mock_tool_manager(self):
        """Create a mock tool manager."""
        mock_manager = Mock()
        mock_manager.batch_call_tools = AsyncMock()
        return mock_manager

    @pytest.fixture
    def orchestrator(self, mock_tool_manager):
        """Create ToolExecutionOrchestrator instance."""
        return ToolExecutionOrchestrator(mock_tool_manager)

    @pytest.mark.asyncio
    async def test_execute_plan_get_product_docs_with_vector_store(
        self, orchestrator, mock_tool_manager
    ):
        """Test get_product_docs execution includes vector_store parameter."""
        plan = ToolExecutionPlan(
            query="test hammer", tools=["get_product_docs"], reasoning="Need product information"
        )

        # Mock successful tool execution
        mock_tool_manager.batch_call_tools.return_value = [
            ToolResult(True, "Found products", "get_product_docs", {"execution_time": 1.2})
        ]

        results = await orchestrator.execute_plan(plan)  # noqa: F841

        # Verify the tool call was made with correct arguments
        mock_tool_manager.batch_call_tools.assert_called_once()
        tool_calls = mock_tool_manager.batch_call_tools.call_args[0][0]

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_product_docs"
        assert "inputs" in tool_calls[0].arguments
        assert tool_calls[0].arguments["inputs"]["query"] == "test hammer"
        assert tool_calls[0].arguments["inputs"]["vector_store"] == "Product"
        assert tool_calls[0].arguments["inputs"]["skus"] == []
        assert tool_calls[0].arguments["inputs"]["model_nos"] == []
        assert tool_calls[0].arguments["inputs"]["brands"] == []
        assert tool_calls[0].arguments["inputs"]["lns"] == []

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].content == "Found products"

    @pytest.mark.asyncio
    async def test_execute_plan_get_raw_docs_with_vector_store(
        self, orchestrator, mock_tool_manager
    ):
        """Test get_raw_docs execution includes vector_store parameter (the main fix)."""
        plan = ToolExecutionPlan(
            query="test gloves", tools=["get_raw_docs"], reasoning="Need raw document information"
        )

        # Mock successful tool execution
        mock_tool_manager.batch_call_tools.return_value = [
            ToolResult(True, "Found raw documents", "get_raw_docs", {"execution_time": 1.5})
        ]

        results = await orchestrator.execute_plan(plan)  # noqa: F841

        # Verify the tool call was made with correct arguments
        mock_tool_manager.batch_call_tools.assert_called_once()
        tool_calls = mock_tool_manager.batch_call_tools.call_args[0][0]

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_raw_docs"
        assert "inputs" in tool_calls[0].arguments
        assert tool_calls[0].arguments["inputs"]["query"] == "test gloves"
        assert tool_calls[0].arguments["inputs"]["vector_store"] == "Product"
        assert tool_calls[0].arguments["inputs"]["skus"] == []
        assert tool_calls[0].arguments["inputs"]["model_nos"] == []
        assert tool_calls[0].arguments["inputs"]["brands"] == []
        assert tool_calls[0].arguments["inputs"]["lns"] == []

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].content == "Found raw documents"

    @pytest.mark.asyncio
    async def test_execute_plan_parse_customer_query(self, orchestrator, mock_tool_manager):
        """Test parse_customer_query execution with correct arguments."""
        plan = ToolExecutionPlan(
            query="what kind of gloves",
            tools=["parse_customer_query"],
            reasoning="Need to parse customer query",
        )

        # Mock successful tool execution
        mock_tool_manager.batch_call_tools.return_value = [
            ToolResult(True, "Parsed query", "parse_customer_query", {"execution_time": 0.5})
        ]

        results = await orchestrator.execute_plan(plan)  # noqa: F841

        # Verify the tool call was made with correct arguments
        mock_tool_manager.batch_call_tools.assert_called_once()
        tool_calls = mock_tool_manager.batch_call_tools.call_args[0][0]

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "parse_customer_query"
        assert "inputs" in tool_calls[0].arguments
        assert tool_calls[0].arguments["inputs"]["query"] == "what kind of gloves"

        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_execute_plan_multiple_tools(self, orchestrator, mock_tool_manager):
        """Test execution of multiple tools including the fixed get_raw_docs."""
        plan = ToolExecutionPlan(
            query="safety gloves",
            tools=["get_product_docs", "get_raw_docs"],
            reasoning="Need comprehensive product information",
        )

        # Mock successful tool execution
        mock_tool_manager.batch_call_tools.return_value = [
            ToolResult(True, "Product docs found", "get_product_docs", {"execution_time": 1.2}),
            ToolResult(True, "Raw docs found", "get_raw_docs", {"execution_time": 1.5}),
        ]

        results = await orchestrator.execute_plan(plan)  # noqa: F841

        # Verify both tools were called with correct arguments
        mock_tool_manager.batch_call_tools.assert_called_once()
        tool_calls = mock_tool_manager.batch_call_tools.call_args[0][0]

        assert len(tool_calls) == 2

        # Check first tool (get_product_docs)
        assert tool_calls[0].name == "get_product_docs"
        assert tool_calls[0].arguments["inputs"]["vector_store"] == "Product"

        # Check second tool (get_raw_docs) - this was the problematic one
        assert tool_calls[1].name == "get_raw_docs"
        assert tool_calls[1].arguments["inputs"]["vector_store"] == "Product"
        assert tool_calls[1].arguments["inputs"]["query"] == "safety gloves"

        assert len(results) == 2
        assert all(result.success for result in results)

    @pytest.mark.asyncio
    async def test_execute_plan_unknown_tool_fallback(self, orchestrator, mock_tool_manager):
        """Test execution of unknown tool falls back to default arguments."""
        plan = ToolExecutionPlan(
            query="test query", tools=["unknown_tool"], reasoning="Testing unknown tool"
        )

        # Mock successful tool execution
        mock_tool_manager.batch_call_tools.return_value = [
            ToolResult(True, "Unknown tool result", "unknown_tool", {"execution_time": 1.0})
        ]

        results = await orchestrator.execute_plan(plan)

        # Verify the tool call was made with default arguments
        mock_tool_manager.batch_call_tools.assert_called_once()
        tool_calls = mock_tool_manager.batch_call_tools.call_args[0][0]

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "unknown_tool"
        assert tool_calls[0].arguments == {"inputs": {"query": "test query"}}

        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_execute_plan_handles_tool_failure(self, orchestrator, mock_tool_manager):
        """Test execution handles tool manager failures gracefully."""
        plan = ToolExecutionPlan(
            query="test query", tools=["get_raw_docs"], reasoning="Testing failure"
        )

        # Mock tool manager failure
        mock_tool_manager.batch_call_tools.side_effect = Exception("Tool execution failed")

        results = await orchestrator.execute_plan(plan)

        # Should return empty list on failure
        assert results == []

        # Plan should be marked as failed
        assert plan.status == "failed"
        assert "Tool execution failed" in plan.error

    @pytest.mark.asyncio
    async def test_execute_plan_marks_plan_completed(self, orchestrator, mock_tool_manager):
        """Test execution marks plan as completed with results."""
        plan = ToolExecutionPlan(
            query="test query", tools=["get_raw_docs"], reasoning="Testing completion"
        )

        # Mock successful tool execution
        expected_results = [ToolResult(True, "Success", "get_raw_docs", {"execution_time": 1.0})]
        mock_tool_manager.batch_call_tools.return_value = expected_results

        results = await orchestrator.execute_plan(plan)  # noqa: F841

        # Plan should be marked as completed
        assert plan.status == "completed"
        assert plan.results == expected_results
        assert plan.execution_start is not None
        assert plan.execution_end is not None
        assert plan.execution_end > plan.execution_start

    @pytest.mark.asyncio
    async def test_execute_plan_marks_plan_executing(self, orchestrator, mock_tool_manager):
        """Test execution marks plan as executing before completion."""
        plan = ToolExecutionPlan(
            query="test query", tools=["get_raw_docs"], reasoning="Testing executing state"
        )

        # Create a mock that allows us to check the plan state during execution
        async def mock_batch_call(tool_calls):
            # Check that plan is marked as executing during tool call
            assert plan.status == "executing"
            assert plan.execution_start is not None
            assert plan.execution_end is None
            return [ToolResult(True, "Success", "get_raw_docs", {})]

        mock_tool_manager.batch_call_tools = mock_batch_call

        results = await orchestrator.execute_plan(plan)  # noqa: F841

        # Plan should be completed after execution
        assert plan.status == "completed"
        assert plan.execution_end is not None


class TestVectorStoreParameterFix:
    """Specific tests for the vector_store parameter fix that resolved the main issue."""

    def test_get_raw_docs_requires_vector_store_in_schema(self):
        """Test that get_raw_docs tool schema requires vector_store parameter."""
        from src.llm.tool_manager import MCPToolManager

        with patch("src.llm.tool_manager.aiohttp.ClientSession"):
            manager = MCPToolManager(base_url="https://test.example.com", auth_token="test_token")

            tool_def = manager._available_tools["get_raw_docs"]
            params = tool_def["function"]["parameters"]["properties"]["inputs"]

            # Verify vector_store is required in the schema
            assert "vector_store" in params["properties"]
            assert "vector_store" in params["required"]
            assert params["properties"]["vector_store"]["type"] == "string"
            assert "Product" in params["properties"]["vector_store"]["enum"]

    @pytest.mark.asyncio
    async def test_vector_store_fix_prevents_validation_error(self):
        """Test that the vector_store fix prevents the 'vector_store is required' error."""
        from src.llm.agentic_system import ToolExecutionOrchestrator

        # Create mock tool manager that validates input
        mock_tool_manager = Mock()

        async def mock_batch_call(tool_calls):
            # Simulate validation that would fail without vector_store
            for call in tool_calls:
                if call.name == "get_raw_docs":
                    inputs = call.arguments.get("inputs", {})
                    if "vector_store" not in inputs:
                        raise ValueError("'vector_store' is a required property")
            return [ToolResult(True, "Success", call.name, {})]

        mock_tool_manager.batch_call_tools = mock_batch_call

        orchestrator = ToolExecutionOrchestrator(mock_tool_manager)

        # Test with the fixed get_raw_docs tool
        plan = ToolExecutionPlan(
            query="test gloves", tools=["get_raw_docs"], reasoning="Testing vector_store fix"
        )

        # This should not raise a validation error
        results = await orchestrator.execute_plan(plan)

        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_vector_store_fix_with_actual_validation(self):
        """Test the fix against actual validation logic similar to MCP server."""
        from src.llm.agentic_system import ToolExecutionOrchestrator

        # Create mock tool manager with realistic validation
        mock_tool_manager = Mock()

        async def realistic_mock_batch_call(tool_calls):
            results = []
            for call in tool_calls:
                inputs = call.arguments.get("inputs", {})

                # Simulate MCP server validation
                if call.name == "get_raw_docs":
                    required_params = ["query", "vector_store"]
                    for param in required_params:
                        if param not in inputs:
                            raise ValueError(
                                f"Input validation error: '{param}' is a required property"
                            )

                    # Validate vector_store enum
                    if inputs["vector_store"] not in ["Product"]:
                        raise ValueError("Input validation error: 'vector_store' must be 'Product'")

                results.append(ToolResult(True, f"Success for {call.name}", call.name, {}))

            return results

        mock_tool_manager.batch_call_tools = realistic_mock_batch_call

        orchestrator = ToolExecutionOrchestrator(mock_tool_manager)

        # Test the fixed implementation
        plan = ToolExecutionPlan(
            query="test query", tools=["get_raw_docs"], reasoning="Testing realistic validation"
        )

        results = await orchestrator.execute_plan(plan)

        # Should pass validation and return results
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].tool_name == "get_raw_docs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
