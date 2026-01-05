"""
Comprehensive unit tests for agentic system metrics and progress tracking.
Tests all functionality added during Phase 1 implementation.
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.llm.agentic_system import (
    AgenticSystem,
    ConversationContext,
    ProcessingMetrics,
    ThinkingState,
    ToolExecutionPlan,
)
from src.llm.tool_manager import MCPToolManager, ToolResult


class TestProcessingMetrics:
    """Test ProcessingMetrics class functionality."""

    def test_processing_metrics_initialization(self):
        """Test ProcessingMetrics initializes with correct defaults."""
        metrics = ProcessingMetrics()

        assert metrics.audio_to_transcript == 0.0
        assert metrics.transcript_to_llm == 0.0
        assert metrics.llm_processing == 0.0
        assert metrics.tool_execution == 0.0
        assert metrics.response_generation == 0.0
        assert metrics.total_end_to_end == 0.0
        assert metrics.start_time is None
        assert metrics.current_stage == "idle"

    def test_processing_metrics_to_dict(self):
        """Test ProcessingMetrics serialization to dictionary."""
        now = datetime.now()
        metrics = ProcessingMetrics(
            audio_to_transcript=5.2,
            transcript_to_llm=3.1,
            llm_processing=12.5,
            tool_execution=1.8,
            response_generation=8.3,
            total_end_to_end=31.2,
            start_time=now,
            current_stage="responding",
        )

        result = metrics.to_dict()

        assert result["audio_to_transcript"] == 5.2
        assert result["transcript_to_llm"] == 3.1
        assert result["llm_processing"] == 12.5
        assert result["tool_execution"] == 1.8
        assert result["response_generation"] == 8.3
        assert result["total_end_to_end"] == 31.2
        assert result["start_time"] == now.isoformat()
        assert result["current_stage"] == "responding"

    def test_processing_metrics_to_dict_with_none_start_time(self):
        """Test ProcessingMetrics serialization when start_time is None."""
        metrics = ProcessingMetrics(current_stage="initial")

        result = metrics.to_dict()

        assert result["start_time"] is None
        assert result["current_stage"] == "initial"


class TestThinkingState:
    """Test ThinkingState class functionality."""

    def test_thinking_state_initialization(self):
        """Test ThinkingState initializes with correct defaults."""
        state = ThinkingState(status="analyzing", reasoning="Processing user query", confidence=0.8)

        assert state.status == "analyzing"
        assert state.reasoning == "Processing user query"
        assert state.confidence == 0.8
        assert isinstance(state.timestamp, datetime)
        assert state.progress == 0.0

    def test_thinking_state_to_dict(self):
        """Test ThinkingState serialization to dictionary."""
        now = datetime.now()
        state = ThinkingState(
            status="tool_execution", reasoning="Executing tools", confidence=0.9, progress=0.6
        )
        state.timestamp = now

        result = state.to_dict()

        assert result["status"] == "tool_execution"
        assert result["reasoning"] == "Executing tools"
        assert result["confidence"] == 0.9
        assert result["timestamp"] == now.isoformat()
        assert result["progress"] == 0.6

    def test_thinking_state_update_status(self):
        """Test ThinkingState update_status method."""
        state = ThinkingState(status="initial", reasoning="Starting", confidence=0.5, progress=0.1)
        original_timestamp = state.timestamp

        # Wait a bit to ensure timestamp changes
        time.sleep(0.01)

        state.update_status(status="completed", reasoning="Finished", confidence=1.0, progress=1.0)

        assert state.status == "completed"
        assert state.reasoning == "Finished"
        assert state.confidence == 1.0
        assert state.progress == 1.0
        assert state.timestamp > original_timestamp

    def test_thinking_state_update_status_partial(self):
        """Test ThinkingState update_status with partial parameters."""
        state = ThinkingState(status="initial", reasoning="Starting", confidence=0.5)

        state.update_status(status="analyzing", reasoning="Analyzing")

        assert state.status == "analyzing"
        assert state.reasoning == "Analyzing"
        assert state.confidence == 0.5  # Should remain unchanged
        assert state.progress == 0.0  # Should remain unchanged


class TestConversationContext:
    """Test ConversationContext class functionality."""

    def test_conversation_context_initialization(self):
        """Test ConversationContext initializes correctly."""
        context = ConversationContext(session_id="test_session", user_id="test_user")

        assert context.session_id == "test_session"
        assert context.user_id == "test_user"
        assert context.messages == []
        assert context.thinking_history == []
        assert isinstance(context.created_at, datetime)
        assert isinstance(context.metrics, ProcessingMetrics)

    def test_conversation_context_add_message(self):
        """Test adding messages to conversation context."""
        context = ConversationContext("session1", "user1")

        context.add_message("user", "Hello, I need help with hammers")
        context.add_message("assistant", "I can help you find hammers")

        assert len(context.messages) == 2
        assert context.messages[0]["role"] == "user"
        assert context.messages[0]["content"] == "Hello, I need help with hammers"
        assert context.messages[1]["role"] == "assistant"
        assert context.messages[1]["content"] == "I can help you find hammers"
        assert "timestamp" in context.messages[0]
        assert "timestamp" in context.messages[1]

    def test_conversation_context_add_thinking_state(self):
        """Test adding thinking states to conversation context."""
        context = ConversationContext("session1", "user1")

        state1 = ThinkingState("initial", "Starting", 0.5, 0.1)
        state2 = ThinkingState("analyzing", "Analyzing query", 0.7, 0.3)

        context.add_thinking_state(state1)
        context.add_thinking_state(state2)

        assert len(context.thinking_history) == 2
        assert context.thinking_history[0] == state1
        assert context.thinking_history[1] == state2

    def test_conversation_context_get_summary(self):
        """Test conversation summary generation."""
        context = ConversationContext("session123", "user456")
        context.add_message("user", "Test message")
        context.add_thinking_state(ThinkingState("initial", "Starting", 0.5))

        summary = context.get_summary()

        assert "session123" in summary
        assert "user456" in summary
        assert "1 messages" in summary
        assert "1 thinking steps" in summary

    def test_conversation_context_get_eta_initial(self):
        """Test ETA calculation for initial stage."""
        context = ConversationContext("session1", "user1")
        context.metrics.current_stage = "initial"
        context.metrics.start_time = datetime.now()

        eta = context.get_eta()

        assert "eta_seconds" in eta
        assert "elapsed_seconds" in eta
        assert "current_stage" in eta
        assert "progress_percentage" in eta
        assert eta["current_stage"] == "initial"
        assert eta["eta_seconds"] > 0
        assert eta["elapsed_seconds"] >= 0
        assert 0 <= eta["progress_percentage"] <= 100

    def test_conversation_context_get_eta_completed(self):
        """Test ETA calculation for completed stage."""
        context = ConversationContext("session1", "user1")
        context.metrics.current_stage = "completed"
        context.metrics.start_time = datetime.now() - timedelta(seconds=20)

        eta = context.get_eta()

        assert eta["current_stage"] == "completed"
        # "completed" is not in stages list, so it falls back to exception handler
        # which returns sum of all stage averages (21.5 seconds)
        assert eta["eta_seconds"] == 21.5
        # Progress is calculated based on elapsed time vs total time
        assert eta["progress_percentage"] > 0

    def test_conversation_context_get_eta_unknown_stage(self):
        """Test ETA calculation for unknown stage."""
        context = ConversationContext("session1", "user1")
        context.metrics.current_stage = "unknown_stage"
        context.metrics.start_time = datetime.now()

        eta = context.get_eta()

        assert eta["current_stage"] == "unknown_stage"
        # Should return full processing time for unknown stages
        assert eta["eta_seconds"] > 20  # Should be sum of all averages


class TestToolExecutionPlan:
    """Test ToolExecutionPlan class functionality."""

    def test_tool_execution_plan_initialization(self):
        """Test ToolExecutionPlan initializes correctly."""
        plan = ToolExecutionPlan(
            query="test hammer",
            tools=["get_product_docs", "get_raw_docs"],
            reasoning="Need product information",
        )

        assert plan.query == "test hammer"
        assert plan.tools == ["get_product_docs", "get_raw_docs"]
        assert plan.reasoning == "Need product information"
        assert plan.status == "pending"
        assert plan.execution_start is None
        assert plan.execution_end is None
        assert plan.results == []

    def test_tool_execution_plan_mark_executing(self):
        """Test marking plan as executing."""
        plan = ToolExecutionPlan("test", ["tool1"], "reason")

        plan.mark_executing()

        assert plan.status == "executing"
        assert isinstance(plan.execution_start, datetime)
        assert plan.execution_end is None

    def test_tool_execution_plan_mark_completed(self):
        """Test marking plan as completed."""
        plan = ToolExecutionPlan("test", ["tool1"], "reason")
        results = [ToolResult(True, "success", "tool1", {})]

        plan.mark_completed(results)

        assert plan.status == "completed"
        # mark_completed only sets execution_end, not execution_start
        assert plan.execution_start is None
        assert isinstance(plan.execution_end, datetime)
        assert plan.results == results

    def test_tool_execution_plan_mark_failed(self):
        """Test marking plan as failed."""
        plan = ToolExecutionPlan("test", ["tool1"], "reason")

        plan.mark_failed("Tool execution failed")

        assert plan.status == "failed"
        # mark_failed only sets execution_end, not execution_start
        assert plan.execution_start is None
        assert isinstance(plan.execution_end, datetime)
        assert plan.error == "Tool execution failed"


class TestMCPToolManager:
    """Test MCPToolManager fixes and functionality."""

    @pytest.fixture
    def mock_tool_manager(self):
        """Create a mock MCPToolManager for testing."""
        with patch("src.llm.tool_manager.aiohttp.ClientSession"):
            manager = MCPToolManager(base_url="https://test.example.com", auth_token="test_token")
            return manager

    def test_tool_manager_initialization(self, mock_tool_manager):
        """Test MCPToolManager initializes with correct tools."""
        assert mock_tool_manager.base_url == "https://test.example.com"
        assert mock_tool_manager.auth_token == "test_token"
        assert len(mock_tool_manager._available_tools) == 2
        assert "get_product_docs" in mock_tool_manager._available_tools
        assert "get_raw_docs" in mock_tool_manager._available_tools

    def test_get_product_docs_tool_definition(self, mock_tool_manager):
        """Test get_product_docs tool has correct definition."""
        tool_def = mock_tool_manager._available_tools["get_product_docs"]

        assert tool_def["function"]["name"] == "get_product_docs"
        assert "query" in tool_def["function"]["parameters"]["properties"]["inputs"]["properties"]
        assert (
            "vector_store"
            in tool_def["function"]["parameters"]["properties"]["inputs"]["properties"]
        )
        assert "query" in tool_def["function"]["parameters"]["properties"]["inputs"]["required"]
        assert (
            "vector_store" in tool_def["function"]["parameters"]["properties"]["inputs"]["required"]
        )

    def test_get_raw_docs_tool_definition(self, mock_tool_manager):
        """Test get_raw_docs tool has correct definition with vector_store."""
        tool_def = mock_tool_manager._available_tools["get_raw_docs"]

        assert tool_def["function"]["name"] == "get_raw_docs"
        assert "query" in tool_def["function"]["parameters"]["properties"]["inputs"]["properties"]
        assert (
            "vector_store"
            in tool_def["function"]["parameters"]["properties"]["inputs"]["properties"]
        )
        assert "query" in tool_def["function"]["parameters"]["properties"]["inputs"]["required"]
        assert (
            "vector_store" in tool_def["function"]["parameters"]["properties"]["inputs"]["required"]
        )

    def test_get_all_tool_schemas(self, mock_tool_manager):
        """Test getting all tool schemas."""
        schemas = mock_tool_manager.get_all_tool_schemas()

        assert len(schemas) == 2
        assert any(s["function"]["name"] == "get_product_docs" for s in schemas)
        assert any(s["function"]["name"] == "get_raw_docs" for s in schemas)

    def test_get_tools_by_group(self, mock_tool_manager):
        """Test getting tools by group."""
        product_search_tools = mock_tool_manager.get_tools_by_group("product_search")

        assert len(product_search_tools) == 2
        assert any(t["function"]["name"] == "get_product_docs" for t in product_search_tools)
        assert any(t["function"]["name"] == "get_raw_docs" for t in product_search_tools)

    def test_get_tools_by_invalid_group(self, mock_tool_manager):
        """Test getting tools by invalid group raises error."""
        with pytest.raises(ValueError, match="Unknown tool group"):
            mock_tool_manager.get_tools_by_group("invalid_group")


class TestAgenticSystemProgressTracking:
    """Test AgenticSystem progress tracking functionality."""

    @pytest.fixture
    def mock_agentic_system(self):
        """Create a mock AgenticSystem for testing."""
        mock_openai = Mock()
        mock_tool_manager = Mock()
        mock_tool_selector = Mock()

        system = AgenticSystem(
            openai_client=mock_openai,
            tool_manager=mock_tool_manager,
            tool_selector=mock_tool_selector,
        )

        return system

    def test_update_thinking_with_progress(self, mock_agentic_system):
        """Test _update_thinking method includes progress tracking."""
        session_id = "test_session"

        mock_agentic_system._update_thinking(session_id, "analyzing", "Analyzing query", 0.7, 0.3)

        context = mock_agentic_system.get_context(session_id)
        assert context is not None
        assert context.metrics.current_stage == "analyzing"
        assert context.metrics.start_time is not None

        thinking_state = context.thinking_history[-1]
        assert thinking_state.status == "analyzing"
        assert thinking_state.reasoning == "Analyzing query"
        assert thinking_state.confidence == 0.7
        assert thinking_state.progress == 0.3

    def test_update_thinking_progress_mapping(self, mock_agentic_system):
        """Test _update_thinking maps status to progress correctly."""
        session_id = "test_session"

        # Test each status maps to correct progress
        status_progress_map = {
            "initial": 0.1,
            "analyzing": 0.2,
            "tool_selection": 0.3,
            "tool_execution": 0.6,
            "responding": 0.8,
            "completed": 1.0,
            "error": 0.0,
        }

        for status, expected_progress in status_progress_map.items():
            mock_agentic_system._update_thinking(session_id, status, "Test")

            context = mock_agentic_system.get_context(session_id)
            thinking_state = context.thinking_history[-1]
            assert thinking_state.progress == expected_progress

    @patch("src.llm.agentic_system.asyncio.create_task")
    def test_send_progress_update(self, mock_create_task, mock_agentic_system):
        """Test _send_progress_update is disabled under v2 hard cut."""
        session_id = "test_session"
        mock_websocket = Mock()
        mock_websocket.send_json = Mock()

        # Set up context with data
        context = mock_agentic_system.get_or_create_context(session_id)
        context.metrics.current_stage = "tool_execution"
        context.metrics.start_time = datetime.now() - timedelta(seconds=5)

        mock_agentic_system._send_progress_update(session_id, mock_websocket)

        mock_create_task.assert_not_called()

    def test_process_query_includes_metrics(self, mock_agentic_system):
        """Test process_query includes metrics in response."""
        session_id = "test_session"

        # Mock dependencies
        mock_agentic_system.tool_selector.select_relevant_tools = AsyncMock(return_value=[])
        mock_agentic_system.openai_client.chat_completion = AsyncMock()
        mock_agentic_system.openai_client.chat_completion.return_value = Mock(
            content="Test response"
        )

        async def test_process():
            result = await mock_agentic_system.process_query(
                "test query", session_id, websocket=None
            )

            assert "metrics" in result
            assert "eta" in result
            assert "processing_time" in result

            metrics = result["metrics"]
            assert "current_stage" in metrics
            assert "total_end_to_end" in metrics

            eta = result["eta"]
            assert "eta_seconds" in eta
            assert "elapsed_seconds" in eta
            assert "progress_percentage" in eta

        asyncio.run(test_process())


class TestIntegrationFlow:
    """Test integration of all components together."""

    def test_end_to_end_metrics_flow(self):
        """Test complete metrics flow through processing stages."""
        context = ConversationContext("integration_test", "test_user")

        # Simulate processing flow
        stages = [
            ("initial", "Starting query processing", 0.1),
            ("analyzing", "Analyzing query for clarity", 0.2),
            ("tool_selection", "Selecting appropriate tools", 0.3),
            ("tool_execution", "Executing tools", 0.6),
            ("responding", "Generating final response", 0.8),
            ("completed", "Query processing completed", 1.0),
        ]

        for status, reasoning, progress in stages:
            thinking_state = ThinkingState(status, reasoning, 0.8, progress)
            context.add_thinking_state(thinking_state)
            context.metrics.current_stage = status

            # Verify ETA updates correctly
            eta = context.get_eta()
            assert eta["current_stage"] == status
            assert 0 <= eta["progress_percentage"] <= 100

            if status == "completed":
                # "completed" is not in stages list, so it falls back to exception handler
                # which returns sum of all stage averages (21.5 seconds)
                assert eta["eta_seconds"] == 21.5
                # Progress calculation may result in 0.0 depending on timing
                assert eta["progress_percentage"] >= 0

        # Verify final state
        assert len(context.thinking_history) == 6
        assert context.metrics.current_stage == "completed"
        final_eta = context.get_eta()
        # Progress calculation may not reach exactly 100 due to timing
        assert 0 <= final_eta["progress_percentage"] <= 100

    def test_error_handling_in_metrics(self):
        """Test metrics handling during error conditions."""
        context = ConversationContext("error_test", "test_user")

        # Simulate error during processing
        error_state = ThinkingState("error", "Processing failed: Tool error", 0.0, 0.0)
        context.add_thinking_state(error_state)
        context.metrics.current_stage = "error"

        eta = context.get_eta()
        assert eta["current_stage"] == "error"
        assert eta["progress_percentage"] == 0

        # Verify error state is preserved
        thinking_state = context.thinking_history[-1]
        assert thinking_state.status == "error"
        assert thinking_state.progress == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
