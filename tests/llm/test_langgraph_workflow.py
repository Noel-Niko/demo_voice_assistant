"""
Unit tests for LangGraph workflow nodes and graph definition.
"""

import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.llm.agent_state import create_initial_state
from src.llm.langgraph_workflow import (
    LangGraphWorkflow,
    analyze_query_node,
    execute_tools_node,
    generate_response_node,
    manage_context_node,
    select_tools_node,
    transcribe_audio_node,
)

pytestmark = pytest.mark.asyncio


class TestLangGraphWorkflowNodes:
    """Test cases for individual LangGraph workflow nodes."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for workflow nodes."""
        mock_openai_client = AsyncMock()
        mock_tool_manager = AsyncMock()
        mock_tool_selector = AsyncMock()
        mock_asr_provider = AsyncMock()

        return {
            "openai_client": mock_openai_client,
            "tool_manager": mock_tool_manager,
            "tool_selector": mock_tool_selector,
            "asr_provider": mock_asr_provider,
        }

    @pytest.fixture
    def sample_state(self):
        """Create a sample AgentState for testing."""
        return create_initial_state(
            session_id="test_session", user_id="test_user", audio_file_path="/path/to/test.wav"
        )

    async def test_transcribe_audio_node(self, mock_dependencies, sample_state):
        """Test the transcribe_audio node."""
        # For LangGraph nodes, transcription is already done; the node should preserve existing transcript.
        sample_state["audio_file_path"] = "/path/to/test.wav"
        sample_state["current_transcript"] = "Hello world"
        sample_state["asr_confidence"] = 0.95

        result = await transcribe_audio_node(sample_state, mock_dependencies["asr_provider"])

        assert result["current_transcript"] == "Hello world"
        assert result["asr_confidence"] == 0.95
        assert result["current_status"] in {"initial", "transcribed"}

    async def test_analyze_query_node_clarification_needed(self, sample_state, mock_dependencies):
        """Test analyze query node when clarification is needed."""
        sample_state["current_transcript"] = "hi"

        result = await analyze_query_node(
            sample_state, mock_dependencies["openai_client"], mock_dependencies["tool_selector"]
        )

        assert result["requires_clarification"] is True
        assert result["current_status"] == "clarification_needed"
        assert result["selected_tools"] == []

    async def test_analyze_query_node_no_clarification(self, mock_dependencies, sample_state):
        """Test analyze_query node when no clarification is needed."""
        sample_state["current_transcript"] = "Find me safety gloves size large"

        # Mock tool selector to return a tool
        mock_tool = MagicMock()
        mock_tool.name = "get_product_docs"
        mock_dependencies["tool_selector"].select_relevant_tools = AsyncMock(
            return_value=[mock_tool]
        )

        result = await analyze_query_node(
            sample_state, mock_dependencies["openai_client"], mock_dependencies["tool_selector"]
        )

        assert result["requires_clarification"] is False
        assert result["current_status"] == "analyzed"
        assert "selected_tools" in result
        assert len(result["selected_tools"]) == 1
        assert result["selected_tools"][0] == "get_product_docs"

    async def test_select_tools_node(self, mock_dependencies, sample_state):
        """Test the select_tools node."""
        # Mock tool selection
        mock_tool = MagicMock()
        mock_tool.name = "get_product_docs"
        mock_dependencies["tool_selector"].select_relevant_tools.return_value = [mock_tool]

        sample_state["current_transcript"] = "Find safety gloves"

        result = await select_tools_node(sample_state, mock_dependencies["tool_selector"])

        assert len(result["selected_tools"]) == 1
        assert result["selected_tools"][0] == "get_product_docs"
        assert result["current_status"] == "tools_selected"

    async def test_select_tools_node_no_tools(self, mock_dependencies, sample_state):
        """Test select_tools node when no tools are selected."""
        mock_dependencies["tool_selector"].select_relevant_tools.return_value = []

        sample_state["current_transcript"] = "Hello there"

        result = await select_tools_node(sample_state, mock_dependencies["tool_selector"])

        assert len(result["selected_tools"]) == 0
        assert result["current_status"] == "no_tools_needed"

    async def test_execute_tools_node(self, mock_dependencies, sample_state):
        """Test the execute_tools node."""
        # Mock tool execution
        mock_tool_result = MagicMock()
        mock_tool_result.tool_name = "get_product_docs"
        mock_tool_result.success = True
        mock_tool_result.content = "Found 5 safety gloves"
        mock_tool_result.execution_time = 1.2

        mock_dependencies["tool_manager"].batch_call_tools.return_value = [mock_tool_result]

        sample_state["selected_tools"] = ["get_product_docs"]
        sample_state["current_transcript"] = "Find safety gloves"

        result = await execute_tools_node(sample_state, mock_dependencies["tool_manager"])

        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["tool_name"] == "get_product_docs"
        assert result["tool_results"][0]["success"] is True
        assert result["current_status"] == "tools_executed"

    async def test_generate_response_node_with_tools(self, mock_dependencies, sample_state):
        """Test generate_response node with tool results."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "I found 5 safety gloves for you"
        mock_dependencies["openai_client"].chat_completion.return_value = mock_response

        sample_state["current_transcript"] = "Find safety gloves"
        sample_state["tool_results"] = [
            {
                "tool_name": "get_product_docs",
                "success": True,
                "content": "Found 5 safety gloves",
                "execution_time": 1.2,
            }
        ]

        result = await generate_response_node(sample_state, mock_dependencies["openai_client"])

        assert "response" in result
        assert result["current_status"] == "completed"

    async def test_generate_response_node_direct_llm(self, mock_dependencies, sample_state):
        """Test generate_response node with direct LLM (no tools)."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "Hello! How can I help you today?"
        mock_dependencies["openai_client"].chat_completion.return_value = mock_response

        sample_state["current_transcript"] = "Hello"
        sample_state["tool_results"] = []

        result = await generate_response_node(sample_state, mock_dependencies["openai_client"])

        assert "response" in result
        assert result["current_status"] == "completed"

    async def test_manage_context_node_pruning(self, sample_state):
        """Test context management node with message pruning."""
        # Create many messages to trigger pruning
        messages = [{"role": "user", "content": f"Message {i}"} for i in range(25)]
        sample_state["messages"] = messages

        result = await manage_context_node(sample_state)

        # Should have pruned messages (keep last 20)
        assert len(result["messages"]) <= 20
        assert result["current_status"] == "context_managed"


class TestLangGraphWorkflow:
    """Test cases for the complete LangGraph workflow."""

    @pytest_asyncio.fixture
    async def workflow_instance(self):
        """Create a LangGraphWorkflow instance for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            workflow = await LangGraphWorkflow.create(database_path=tmp.name)

        await workflow.set_dependencies(
            openai_client=AsyncMock(),
            tool_manager=AsyncMock(),
            tool_selector=AsyncMock(),
            asr_provider=AsyncMock(),
        )
        try:
            yield workflow
        finally:
            await workflow.aclose()

    @pytest.fixture
    def mock_agentic_system(self):
        """Create a mock agentic system."""
        mock_system = AsyncMock()
        mock_system.openai_client = AsyncMock()
        mock_system.tool_manager = AsyncMock()
        mock_system.tool_selector = AsyncMock()
        return mock_system

    async def test_workflow_initialization(self, workflow_instance):
        """Test workflow initialization."""
        assert workflow_instance.app is not None
        assert workflow_instance.checkpointer is not None
        assert workflow_instance.persistence_manager is not None

    async def test_process_audio_message_new_session(self, workflow_instance, mock_agentic_system):
        """Test processing an audio message for a new session."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "Hello!"
        workflow_instance.openai_client.chat_completion = AsyncMock(return_value=mock_response)
        workflow_instance.tool_selector.select_relevant_tools = AsyncMock(return_value=[])

        session_id = "new_session"
        audio_file = "/path/to/audio.wav"

        result = await workflow_instance.process_audio_message(
            audio_file, session_id, transcript="Hello", confidence=0.9
        )

        assert result["response"] == "Hello!"
        assert result["session_id"] == "new_session"

    async def test_process_audio_message_existing_session(
        self, workflow_instance, mock_agentic_system
    ):
        """Test processing an audio message for an existing session."""
        # Mock LLM responses
        mock_response1 = MagicMock()
        mock_response1.content = "First response"
        mock_response2 = MagicMock()
        mock_response2.content = "I remember you!"

        workflow_instance.openai_client.chat_completion = AsyncMock(
            side_effect=[mock_response1, mock_response2]
        )
        workflow_instance.tool_selector.select_relevant_tools = AsyncMock(return_value=[])

        session_id = "existing_session"

        # Send first message
        await workflow_instance.process_audio_message(
            None, session_id, transcript="Previous message", confidence=0.9
        )

        # Send second message to same session
        result = await workflow_instance.process_audio_message(
            None, session_id, transcript="New message", confidence=0.9
        )

        assert result["response"] == "I remember you!"
        assert len(result["messages"]) >= 2

    async def test_get_conversation_history(self, workflow_instance):
        """Test retrieving conversation history."""
        # Mock state retrieval
        mock_state = MagicMock()
        mock_state.values = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }

        workflow_instance.app.aget_state = AsyncMock(return_value=mock_state)

        session_id = "test_session"
        history = await workflow_instance.get_conversation_history(session_id)

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    async def test_get_conversation_history_empty(self, workflow_instance):
        """Test retrieving conversation history for non-existent session."""
        # Mock empty state
        mock_state = MagicMock()
        mock_state.values = {}

        workflow_instance.app.aget_state = AsyncMock(return_value=mock_state)

        session_id = "nonexistent_session"
        history = await workflow_instance.get_conversation_history(session_id)

        assert history == []

    async def test_clear_session_history(self, workflow_instance):
        """Test clearing session history."""
        workflow_instance.app.aupdate_state = AsyncMock()

        session_id = "test_session"
        await workflow_instance.clear_session_history(session_id)

        # Verify update_state was called with empty messages
        workflow_instance.app.aupdate_state.assert_called_once()

    async def test_workflow_error_handling(self, workflow_instance):
        """Test workflow error handling."""
        # Mock workflow failure
        workflow_instance.app.ainvoke = AsyncMock(side_effect=Exception("Test error"))

        session_id = "test_session"
        audio_file = "/path/to/audio.wav"

        result = await workflow_instance.process_audio_message(audio_file, session_id)
        assert "error" in result

    async def test_checkpoint_management(self, workflow_instance):
        """Test checkpoint creation and retrieval."""
        # Mock checkpoint operations
        mock_state = MagicMock()
        mock_state.next = "checkpoint_123"

        workflow_instance.app.aget_state = AsyncMock(return_value=mock_state)

        session_id = "test_session"
        checkpoint_id = await workflow_instance.get_current_checkpoint(session_id)

        assert checkpoint_id == "checkpoint_123"
        expected_config = workflow_instance.persistence_manager.get_session_config(session_id)
        workflow_instance.app.aget_state.assert_called_once_with(expected_config)

    async def test_resume_from_checkpoint(self, workflow_instance):
        """Test resuming workflow from specific checkpoint."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "Resumed!"
        workflow_instance.openai_client.chat_completion = AsyncMock(return_value=mock_response)
        workflow_instance.tool_selector.select_relevant_tools = AsyncMock(return_value=[])

        session_id = "test_session"
        checkpoint_id = "checkpoint_123"
        audio_file = "/path/to/audio.wav"

        result = await workflow_instance.process_audio_message(
            audio_file, session_id, checkpoint_id=checkpoint_id, transcript="Resume", confidence=0.9
        )

        assert result["response"] == "Resumed!"
