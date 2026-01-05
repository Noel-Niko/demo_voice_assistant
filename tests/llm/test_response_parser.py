"""
Unit tests for response parser utility.
Tests parsing of both Chat Completions and Responses API responses.
"""

from unittest.mock import Mock

from src.llm.response_parser import ResponseParser
from src.llm.types import ChatResponse


class TestResponseParser:
    """Test response parsing functionality."""

    def test_parse_chat_completions_response_basic(self):
        """Test parsing basic Chat Completions response."""
        # Mock Chat Completions response
        mock_response = Mock()
        mock_response.model = "gpt-4"

        # Mock choices
        mock_choice = Mock()
        mock_choice.message.content = "Hello, how can I help you?"
        mock_choice.message.tool_calls = None
        mock_response.choices = [mock_choice]

        # Mock usage
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_response.usage = mock_usage

        # Parse response
        result = ResponseParser.parse_chat_completions_response(mock_response)

        # Verify result
        assert isinstance(result, ChatResponse)
        assert result.content == "Hello, how can I help you?"
        assert result.tool_calls == []
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        assert result.model == "gpt-4"

    def test_parse_chat_completions_response_with_tools(self):
        """Test parsing Chat Completions response with tool calls."""
        # Mock tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function.name = "test_function"
        mock_tool_call.function.arguments = '{"arg1": "value1"}'

        # Mock response
        mock_response = Mock()
        mock_response.model = "gpt-4"
        mock_response.usage = None

        mock_choice = Mock()
        mock_choice.message.content = "I'll call the tool for you."
        mock_choice.message.tool_calls = [mock_tool_call]
        mock_response.choices = [mock_choice]

        # Parse response
        result = ResponseParser.parse_chat_completions_response(mock_response)

        # Verify result
        assert result.content == "I'll call the tool for you."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0] == {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_function", "arguments": '{"arg1": "value1"}'},
        }
        assert result.usage is None

    def test_parse_responses_response_basic(self):
        """Test parsing basic Responses API response."""
        # Mock Responses response
        mock_response = Mock()
        mock_response.model = "gpt-4.1-mini"
        mock_response.output_text = "This is a response from the Responses API."
        mock_response.output = None  # No output items, just output_text
        mock_response.usage = None

        # Parse response
        result = ResponseParser.parse_responses_response(mock_response)

        # Verify result
        assert isinstance(result, ChatResponse)
        assert result.content == "This is a response from the Responses API."
        assert result.tool_calls == []
        assert result.usage is None
        assert result.model == "gpt-4.1-mini"

    def test_parse_responses_response_with_output_items(self):
        """Test parsing Responses API response with output items."""
        # Mock output items
        mock_text_item = Mock()
        mock_text_item.type = "text"
        mock_text_item.text = "Hello from output items."

        mock_tool_item = Mock()
        mock_tool_item.type = "tool_call"
        mock_tool_item.id = "tool_456"
        mock_tool_item.name = "another_function"
        mock_tool_item.arguments = '{"param": "value"}'

        # Mock response
        mock_response = Mock()
        mock_response.model = "gpt-4o"
        mock_response.output = [mock_text_item, mock_tool_item]
        # Explicitly set output_text to None so parser uses output items instead
        mock_response.output_text = None

        # Mock usage
        mock_usage = Mock()
        mock_usage.input_tokens = 15
        mock_usage.output_tokens = 25
        mock_usage.total_tokens = 40
        mock_response.usage = mock_usage

        # Parse response
        result = ResponseParser.parse_responses_response(mock_response)

        # Verify result
        assert result.content == "Hello from output items."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0] == {
            "id": "tool_456",
            "type": "function",
            "function": {"name": "another_function", "arguments": '{"param": "value"}'},
        }
        assert result.usage == {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}
        assert result.model == "gpt-4o"

    def test_parse_responses_response_empty_output(self):
        """Test parsing Responses API response with empty output."""
        mock_response = Mock()
        mock_response.model = "gpt-4.1-mini"
        mock_response.output = []
        mock_response.output_text = None

        result = ResponseParser.parse_responses_response(mock_response)

        assert result.content == ""
        assert result.tool_calls == []

    def test_parse_streaming_chunk_chat_completions_content(self):
        """Test parsing Chat Completions streaming chunk with content."""
        # Mock chunk
        mock_delta = Mock()
        mock_delta.content = "Hello "

        mock_choice = Mock()
        mock_choice.delta = mock_delta

        mock_chunk = Mock()
        mock_chunk.choices = [mock_choice]

        # Parse chunk
        result = ResponseParser.parse_streaming_chunk(mock_chunk, "chat_completions")

        assert result == {"type": "content", "content": "Hello "}

    def test_parse_streaming_chunk_chat_completions_tool_call(self):
        """Test parsing Chat Completions streaming chunk with tool call."""
        # Mock tool call delta
        mock_function = Mock()
        mock_function.name = "test_tool"
        mock_function.arguments = '{"arg": "value"}'

        mock_tool_call = Mock()
        mock_tool_call.id = "call_789"
        mock_tool_call.function = mock_function

        mock_delta = Mock()
        mock_delta.tool_calls = [mock_tool_call]
        mock_delta.content = None

        mock_choice = Mock()
        mock_choice.delta = mock_delta

        mock_chunk = Mock()
        mock_chunk.choices = [mock_choice]

        # Parse chunk
        result = ResponseParser.parse_streaming_chunk(mock_chunk, "chat_completions")

        assert result == {
            "type": "tool_call",
            "id": "call_789",
            "name": "test_tool",
            "arguments": '{"arg": "value"}',
        }

    def test_parse_streaming_chunk_responses_text(self):
        """Test parsing Responses streaming chunk with text."""
        mock_chunk = Mock()
        mock_chunk.type = "text"
        mock_chunk.text = "Streaming text"

        result = ResponseParser.parse_streaming_chunk(mock_chunk, "responses")

        assert result == {"type": "content", "content": "Streaming text"}

    def test_parse_streaming_chunk_responses_tool_call(self):
        """Test parsing Responses streaming chunk with tool call."""
        mock_chunk = Mock()
        mock_chunk.type = "tool_call"
        mock_chunk.id = "stream_tool_123"
        mock_chunk.name = "stream_function"
        mock_chunk.arguments = '{"stream": "true"}'

        result = ResponseParser.parse_streaming_chunk(mock_chunk, "responses")

        assert result == {
            "type": "tool_call",
            "id": "stream_tool_123",
            "name": "stream_function",
            "arguments": '{"stream": "true"}',
        }

    def test_parse_streaming_chunk_empty(self):
        """Test parsing empty streaming chunk."""
        mock_chunk = Mock()
        mock_chunk.choices = []

        result = ResponseParser.parse_streaming_chunk(mock_chunk, "chat_completions")

        assert result is None

    def test_parse_streaming_chunk_unknown_api_type(self):
        """Test parsing chunk with unknown API type."""
        mock_chunk = Mock()

        result = ResponseParser.parse_streaming_chunk(mock_chunk, "unknown")

        assert result is None

    def test_parse_chat_completions_empty_content(self):
        """Test parsing Chat Completions response with empty content."""
        mock_response = Mock()
        mock_response.model = "gpt-4"
        mock_response.usage = None

        mock_choice = Mock()
        mock_choice.message.content = None
        mock_choice.message.tool_calls = None
        mock_response.choices = [mock_choice]

        result = ResponseParser.parse_chat_completions_response(mock_response)

        assert result.content == ""
        assert result.tool_calls == []

    def test_parse_responses_missing_attributes(self):
        """Test parsing Responses response with missing attributes."""
        mock_response = Mock()
        mock_response.model = "unknown"
        # Don't set output_text or output - set output to None to avoid iteration error
        mock_response.output = None
        mock_response.output_text = None

        result = ResponseParser.parse_responses_response(mock_response)

        assert result.content == ""
        assert result.tool_calls == []
        assert result.model == "unknown"
