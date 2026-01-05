"""
Unit tests for AgentState schema and related functionality.
"""

from datetime import datetime

from src.llm.agent_state import AgentState, create_initial_state, validate_agent_state


class TestAgentState:
    """Test cases for AgentState schema."""

    def test_agent_state_creation(self):
        """Test creating a valid AgentState."""
        state = AgentState(
            messages=[],
            session_id="test_session_123",
            user_id="test_user",
            audio_file_path=None,
            current_transcript=None,
            asr_confidence=0.0,
            thinking_history=[],
            current_status="initial",
            reasoning="Starting processing",
            confidence=0.5,
            progress=0.0,
            tool_results=[],
            selected_tools=[],
            execution_plan=None,
            start_time=None,
            processing_metrics={},
            user_context={},
        )

        assert state["session_id"] == "test_session_123"
        assert state["user_id"] == "test_user"
        assert state["messages"] == []
        assert state["current_status"] == "initial"

    def test_agent_state_with_data(self):
        """Test AgentState with populated data."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        thinking_history = [
            {
                "status": "initial",
                "reasoning": "Starting",
                "confidence": 0.5,
                "timestamp": datetime.now().isoformat(),
                "progress": 0.0,
            }
        ]

        state = AgentState(
            messages=messages,
            session_id="session_456",
            user_id="user_456",
            current_transcript="Hello world",
            asr_confidence=0.95,
            thinking_history=thinking_history,
            current_status="analyzing",
            reasoning="Analyzing query",
            confidence=0.7,
            progress=0.3,
            tool_results=[],
            selected_tools=["get_product_docs"],
            execution_plan=None,
            start_time=datetime.now(),
            processing_metrics={"audio_to_transcript": 1.5},
            user_context={"preferences": {"language": "en"}},
        )

        assert len(state["messages"]) == 2
        assert state["current_transcript"] == "Hello world"
        assert state["asr_confidence"] == 0.95
        assert state["current_status"] == "analyzing"
        assert state["selected_tools"] == ["get_product_docs"]

    def test_validate_agent_state_valid(self):
        """Test validation of valid AgentState."""
        state = AgentState(
            messages=[],
            session_id="valid_session",
            user_id="valid_user",
            audio_file_path=None,
            current_transcript=None,
            asr_confidence=0.0,
            thinking_history=[],
            current_status="initial",
            reasoning="Valid state",
            confidence=0.5,
            progress=0.0,
            tool_results=[],
            selected_tools=[],
            execution_plan=None,
            start_time=None,
            processing_metrics={},
            user_context={},
            requires_clarification=False,
        )

        assert validate_agent_state(state) is True

    def test_validate_agent_state_missing_required(self):
        """Test validation fails with missing required fields."""
        # Missing session_id
        invalid_state = {"messages": [], "user_id": "test_user", "current_status": "initial"}

        assert validate_agent_state(invalid_state) is False

    def test_validate_agent_state_invalid_confidence(self):
        """Test validation fails with invalid confidence values."""
        state = AgentState(
            messages=[],
            session_id="test_session",
            user_id="test_user",
            audio_file_path=None,
            current_transcript=None,
            asr_confidence=1.5,  # Invalid: > 1.0
            thinking_history=[],
            current_status="initial",
            reasoning="Test",
            confidence=-0.1,  # Invalid: < 0.0
            progress=0.0,
            tool_results=[],
            selected_tools=[],
            execution_plan=None,
            start_time=None,
            processing_metrics={},
            user_context={},
        )

        assert validate_agent_state(state) is False

    def test_create_initial_state(self):
        """Test creating initial state for new session."""
        session_id = "new_session_123"
        user_id = "new_user"
        audio_path = "/path/to/audio.wav"

        state = create_initial_state(session_id, user_id, audio_path)

        assert state["session_id"] == session_id
        assert state["user_id"] == user_id
        assert state["audio_file_path"] == audio_path
        assert state["messages"] == []
        assert state["thinking_history"] == []
        assert state["current_status"] == "initial"
        assert state["confidence"] == 0.0
        assert state["progress"] == 0.0
        assert state["tool_results"] == []
        assert state["selected_tools"] == []
        assert state["start_time"] is not None

    def test_create_initial_state_without_audio(self):
        """Test creating initial state without audio file."""
        session_id = "session_no_audio"
        user_id = "user_no_audio"

        state = create_initial_state(session_id, user_id)

        assert state["session_id"] == session_id
        assert state["user_id"] == user_id
        assert state["audio_file_path"] is None
        assert state["current_transcript"] is None

    def test_agent_state_message_addition(self):
        """Test that messages can be properly added to state."""
        from langgraph.graph.message import add_messages

        initial_messages = [{"role": "user", "content": "Hello"}]
        new_message = {"role": "assistant", "content": "Hi there!"}

        updated_messages = add_messages(initial_messages, new_message)

        assert len(updated_messages) == 2
        # LangGraph's add_messages returns message objects, not dictionaries
        # Check that the content is preserved
        assert updated_messages[1].content == "Hi there!"

    def test_thinking_state_serialization(self):
        """Test that thinking states can be serialized/deserialized."""
        thinking_state = {
            "status": "analyzing",
            "reasoning": "Processing user query",
            "confidence": 0.8,
            "timestamp": datetime.now().isoformat(),
            "progress": 0.4,
        }

        state = AgentState(
            messages=[],
            session_id="test_session",
            user_id="test_user",
            audio_file_path=None,
            current_transcript=None,
            asr_confidence=0.0,
            thinking_history=[thinking_state],
            current_status="analyzing",
            reasoning="Processing",
            confidence=0.8,
            progress=0.4,
            tool_results=[],
            selected_tools=[],
            execution_plan=None,
            start_time=None,
            processing_metrics={},
            user_context={},
        )

        assert len(state["thinking_history"]) == 1
        assert state["thinking_history"][0]["status"] == "analyzing"
        assert state["thinking_history"][0]["confidence"] == 0.8

    def test_tool_results_structure(self):
        """Test tool results structure in AgentState."""
        tool_result = {
            "tool_name": "get_product_docs",
            "success": True,
            "content": "Product documentation found",
            "execution_time": 1.2,
        }

        state = AgentState(
            messages=[],
            session_id="test_session",
            user_id="test_user",
            audio_file_path=None,
            current_transcript=None,
            asr_confidence=0.0,
            thinking_history=[],
            current_status="completed",
            reasoning="Completed successfully",
            confidence=1.0,
            progress=1.0,
            tool_results=[tool_result],
            selected_tools=["get_product_docs"],
            execution_plan=None,
            start_time=None,
            processing_metrics={"tool_execution": 1.2},
            user_context={},
        )

        assert len(state["tool_results"]) == 1
        assert state["tool_results"][0]["tool_name"] == "get_product_docs"
        assert state["tool_results"][0]["success"] is True
        assert state["processing_metrics"]["tool_execution"] == 1.2
