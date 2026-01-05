"""
Simple integration tests for LangGraph workflow implementation.
"""

import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.agent_state import create_initial_state
from src.llm.langgraph_workflow import LangGraphWorkflow

pytestmark = pytest.mark.asyncio


class TestLangGraphSimpleIntegration:
    """Simple integration tests for LangGraph workflow."""

    async def test_workflow_creation(self):
        """Test that the workflow can be created successfully."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            workflow = await LangGraphWorkflow.create(database_path=tmp.name)

        assert workflow is not None
        assert workflow.app is not None
        assert workflow.checkpointer is not None
        assert workflow.persistence_manager is not None

        await workflow.aclose()

    async def test_agent_state_creation(self):
        """Test AgentState creation and validation."""
        from src.llm.agent_state import validate_agent_state

        state = create_initial_state("test_session", "test_user")

        assert state["session_id"] == "test_session"
        assert state["user_id"] == "test_user"
        assert state["current_status"] == "initial"
        assert state["messages"] == []
        assert state["thinking_history"] == []
        assert state["tool_results"] == []
        assert state["requires_clarification"] is False

        assert validate_agent_state(state) is True

    async def test_persistence_manager_creation(self):
        """Test PersistenceManager creation."""
        from src.llm.persistence_manager import PersistenceManager

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            manager = await PersistenceManager.create(database_path=tmp.name)

        assert manager is not None
        assert manager.checkpointer is not None

        config = manager.get_session_config("test_session")
        assert config["configurable"]["thread_id"] == "test_session"

        await manager.aclose()

    async def test_workflow_with_mocks(self):
        """Test workflow with mocked dependencies."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            workflow = await LangGraphWorkflow.create(database_path=tmp.name)

        # Mock dependencies
        workflow.openai_client = AsyncMock()
        workflow.tool_manager = AsyncMock()
        workflow.tool_selector = AsyncMock()
        workflow.asr_provider = AsyncMock()

        # Mock ASR transcription
        workflow.asr_provider.transcribe_file.return_value = {
            "text": "Hello world",
            "confidence": 0.95,
        }

        # Mock tool selection
        workflow.tool_selector.select_relevant_tools.return_value = []

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "Hi there! How can I help you?"
        workflow.openai_client.chat_completion.return_value = mock_response

        # Test basic state creation
        initial_state = create_initial_state("test_session", "test_user")

        assert initial_state["session_id"] == "test_session"
        assert initial_state["user_id"] == "test_user"
        assert initial_state["current_status"] == "initial"

        await workflow.aclose()

    async def test_thinking_state_functions(self):
        """Test thinking state creation and manipulation."""
        from src.llm.agent_state import (
            add_thinking_state,
            add_tool_result,
            create_thinking_state,
            create_tool_result,
        )

        # Create initial state
        state = create_initial_state("test_session", "test_user")

        # Add thinking state
        thinking = create_thinking_state(
            "analyzing", "Processing user query", confidence=0.7, progress=0.5
        )

        updated_state = add_thinking_state(state, thinking)

        assert updated_state["current_status"] == "analyzing"
        assert updated_state["reasoning"] == "Processing user query"
        assert updated_state["confidence"] == 0.7
        assert updated_state["progress"] == 0.5
        assert len(updated_state["thinking_history"]) == 1

        # Add tool result
        tool_result = create_tool_result(
            tool_name="test_tool",
            success=True,
            content="Tool executed successfully",
            execution_time=1.5,
        )

        final_state = add_tool_result(updated_state, tool_result)

        assert len(final_state["tool_results"]) == 1
        assert final_state["tool_results"][0]["tool_name"] == "test_tool"
        assert final_state["tool_results"][0]["success"] is True

    async def test_message_handling(self):
        """Test message handling with LangGraph message types."""
        from langgraph.graph.message import add_messages

        from src.llm.agent_state import validate_agent_state

        # Create initial state
        state = create_initial_state("test_session", "test_user")

        # Add messages using LangGraph's add_messages
        user_message = {"role": "user", "content": "Hello"}
        assistant_message = {"role": "assistant", "content": "Hi there!"}

        updated_messages = add_messages(state["messages"], user_message)
        updated_messages = add_messages(updated_messages, assistant_message)

        # Update state with new messages
        state["messages"] = updated_messages

        # Validate state
        assert validate_agent_state(state) is True
        assert len(state["messages"]) == 2

        # Check message content (LangGraph converts to message objects)
        assert state["messages"][0].content == "Hello"
        assert state["messages"][1].content == "Hi there!"
