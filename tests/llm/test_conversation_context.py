"""
Test conversation context handling in LangGraph workflow.

This test verifies that the LLM properly receives and utilizes conversation history
to maintain context across multiple turns.
"""

import asyncio
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.llm.langgraph_workflow import LangGraphWorkflow

pytestmark = pytest.mark.asyncio


class TestConversationContext:
    """Test conversation context preservation and usage."""

    @pytest_asyncio.fixture
    async def workflow(self):
        """Create a LangGraph workflow for testing."""
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

    async def test_conversation_context_preserved(self):
        """Test that conversation context is preserved across turns."""

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            workflow = await LangGraphWorkflow.create(database_path=tmp.name)

        # Mock dependencies
        mock_openai_client = AsyncMock()
        mock_tool_manager = AsyncMock()
        mock_tool_selector = AsyncMock()
        mock_asr_provider = AsyncMock()

        # Set dependencies
        await workflow.set_dependencies(
            openai_client=mock_openai_client,
            tool_manager=mock_tool_manager,
            tool_selector=mock_tool_selector,
            asr_provider=mock_asr_provider,
        )

        # Mock LLM responses that show context awareness
        responses = [
            "What type of hammer are you looking for—claw hammer, sledgehammer, or something else?",
            "Great! A sledgehammer is perfect for heavy demolition work. What specific projects will you be working on?",
        ]

        mock_openai_client.chat_completion = AsyncMock(
            side_effect=[MagicMock(content=responses[0]), MagicMock(content=responses[1])]
        )

        try:
            # First conversation turn
            result1 = await workflow.process_audio_message(
                audio_file_path=None,
                session_id="test_session",
                user_id="test_user",
                transcript="I need a hammer",
                confidence=0.95,
            )

            # Verify first response
            assert result1["response"] == responses[0]
            assert result1["current_status"] == "context_managed"

            # Second conversation turn (should have context)
            result2 = await workflow.process_audio_message(
                audio_file_path=None,
                session_id="test_session",
                user_id="test_user",
                transcript="Sledge",
                confidence=0.95,
            )

            # Verify second response shows context awareness
            assert result2["response"] == responses[1]
            assert result2["current_status"] == "context_managed"

            # Verify conversation history was passed to LLM
            calls = mock_openai_client.chat_completion.call_args_list

            # First call should include the initial user query
            # (an optional system prompt may be prepended by the workflow)
            first_call_messages = calls[0][1]["messages"]
            user_only = [m for m in first_call_messages if m.get("role") == "user"]
            assert len(user_only) == 1
            assert user_only[0]["content"] == "I need a hammer"

            # Second call should have conversation history (previous exchange + current query)
            second_call_messages = calls[1][1]["messages"]
            # Allow an optional system prompt at the beginning
            non_system = [m for m in second_call_messages if m.get("role") != "system"]
            assert len(non_system) >= 3  # At least: user1, assistant1, user2

            # Verify the conversation flow is correct (allowing for duplicates)
            user_messages = [msg for msg in second_call_messages if msg["role"] == "user"]
            assistant_messages = [msg for msg in second_call_messages if msg["role"] == "assistant"]

            # Should have at least the conversation flow
            assert len(user_messages) >= 2, "Should have at least 2 user messages"
            assert len(assistant_messages) >= 1, "Should have at least 1 assistant message"

            # Check the conversation contains the right content
            user_contents = [msg["content"] for msg in user_messages]
            assistant_contents = [msg["content"] for msg in assistant_messages]

            assert "I need a hammer" in user_contents, "Should contain first user message"
            assert "Sledge" in user_contents, "Should contain second user message"
            assert responses[0] in assistant_contents, "Should contain first assistant response"

            # Verify all messages are proper dictionaries
            for msg in second_call_messages:
                assert isinstance(msg, dict)
                assert "role" in msg
                assert "content" in msg

        finally:
            await workflow.aclose()

    async def test_responses_api_path_includes_history_in_second_turn(self):
        """Verify second turn includes first-turn context for Responses API models."""

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            workflow = await LangGraphWorkflow.create(database_path=tmp.name)

        mock_openai_client = AsyncMock()
        mock_tool_manager = AsyncMock()
        mock_tool_selector = AsyncMock()
        mock_asr_provider = AsyncMock()

        await workflow.set_dependencies(
            openai_client=mock_openai_client,
            tool_manager=mock_tool_manager,
            tool_selector=mock_tool_selector,
            asr_provider=mock_asr_provider,
        )

        mock_tool_selector.select_relevant_tools = AsyncMock(return_value=[])
        mock_openai_client.chat_completion = AsyncMock(
            side_effect=[
                MagicMock(content="Sure. What will you use it for?"),
                MagicMock(content="For carpentry, a 16oz claw hammer is a solid choice."),
            ]
        )

        try:
            await workflow.process_audio_message(
                audio_file_path=None,
                session_id="s1",
                user_id="u1",
                transcript="Recommend a hammer.",
                confidence=0.9,
            )

            await workflow.process_audio_message(
                audio_file_path=None,
                session_id="s1",
                user_id="u1",
                transcript="Carpentry.",
                confidence=0.9,
            )

            calls = mock_openai_client.chat_completion.call_args_list
            assert len(calls) == 2

            second_messages = calls[1].kwargs["messages"]
            contents = [m["content"] for m in second_messages if m.get("role") == "user"]
            assert any("Recommend a hammer" in c for c in contents)
            assert any("Carpentry" in c for c in contents)
        finally:
            await workflow.aclose()

    async def test_llm_ignores_context_issue(self, workflow):
        """Test that reproduces the issue where LLM ignores conversation context."""

        # Mock LLM responses that show context ignorance (the actual problem)
        responses = [
            "What type of hammer are you looking for—claw hammer, sledgehammer, or something else?",
            "Could you please clarify what you mean by 'Sledge'? Are you referring to the tool, a game character, a type of sled, or something else?",
        ]

        workflow.openai_client.chat_completion = AsyncMock(
            side_effect=[MagicMock(content=responses[0]), MagicMock(content=responses[1])]
        )

        # Simulate the problematic conversation
        result1 = await workflow.process_audio_message(  # noqa: F841
            audio_file_path=None,
            session_id="test_session",
            user_id="test_user",
            transcript="I need a hammer",
            confidence=0.95,
        )

        result2 = await workflow.process_audio_message(
            audio_file_path=None,
            session_id="test_session",
            user_id="test_user",
            transcript="Sledge",
            confidence=0.95,
        )

        # Get the actual messages sent to LLM
        calls = workflow.openai_client.chat_completion.call_args_list
        second_call_messages = calls[1][1]["messages"]

        # This test documents the current behavior
        # If the LLM had proper context, it would know "Sledge" refers to "sledgehammer"
        # But the second response shows it's asking for clarification

        print("\\nMessages sent to LLM on second call:")
        for i, msg in enumerate(second_call_messages):
            print(f"  {i + 1}. Role: {msg['role']}, Content: {msg['content']}")

        print(f"\\nLLM response: {result2['response']}")

        # This test will help us debug what's actually being sent vs what should be sent
        assert len(second_call_messages) >= 3, "Conversation history should be preserved"

        # The issue might be in the API type or message format
        # We'll investigate this in the fix phase

    async def test_message_format_verification(self, workflow):
        """Test that messages are properly formatted for OpenAI API."""

        workflow.openai_client.chat_completion = AsyncMock(
            return_value=MagicMock(content="Test response")
        )

        await workflow.process_audio_message(
            audio_file_path=None,
            session_id="test_session",
            user_id="test_user",
            transcript="Test message",
            confidence=0.95,
        )

        # Get the messages sent to OpenAI
        call = workflow.openai_client.chat_completion.call_args_list[0]
        messages = call[1]["messages"]

        # Verify message format
        for msg in messages:
            assert isinstance(msg, dict), f"Message should be dict, got {type(msg)}"
            assert "role" in msg, "Message should have 'role' field"
            assert "content" in msg, "Message should have 'content' field"
            assert msg["role"] in ["user", "assistant", "system"], f"Invalid role: {msg['role']}"
            assert isinstance(msg["content"], str), "Content should be string"


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_conversation_context_flow())  # noqa: F821
