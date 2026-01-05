"""
Unit tests for OpenAI client implementation.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.constants.common_constants import OpenAIDefaults
from src.llm.openai_client import OpenAIClient, OpenAIConfig


class TestOpenAIConfig:
    """Test OpenAI configuration."""

    def test_config_from_env_with_api_key(self, monkeypatch):
        """Test configuration loads API key from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        # Ensure model expectation matches environment-configured value
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5")
        monkeypatch.delenv("OPENAI_MAX_TOKENS", raising=False)
        monkeypatch.delenv("OPENAI_TEMPERATURE", raising=False)
        monkeypatch.delenv("OPENAI_TIMEOUT", raising=False)

        config = OpenAIConfig.from_env()

        assert config.api_key == "test-key-123"
        assert config.model_name == "gpt-5"
        # Defaults should match current OpenAIDefaults when env vars are unset
        assert config.max_tokens == OpenAIDefaults.MAX_TOKENS
        assert config.temperature == OpenAIDefaults.TEMPERATURE

    def test_config_from_env_missing_key(self, monkeypatch):
        """Test configuration raises error when API key is missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is required"):
            OpenAIConfig.from_env()

    def test_config_custom_values(self, monkeypatch):
        """Test configuration with custom environment values."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "2000")
        monkeypatch.setenv("OPENAI_TEMPERATURE", "0.5")

        config = OpenAIConfig.from_env()

        assert config.model_name == "gpt-4"
        assert config.max_tokens == 2000
        assert config.temperature == 0.5


class TestOpenAIClient:
    """Test OpenAI client implementation."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock OpenAI configuration."""
        return OpenAIConfig(
            api_key="test-key-123", model_name="gpt-5", max_tokens=4000, temperature=0.7
        )

    @pytest.fixture
    def client(self, mock_config):
        """Create OpenAI client with mock configuration."""
        with patch("src.llm.openai_client.OpenAI"):
            return OpenAIClient(mock_config)

    def test_client_initialization(self, mock_config):
        """Test client initializes correctly."""
        with patch("src.llm.openai_client.OpenAI") as mock_openai:
            client = OpenAIClient(mock_config)

            mock_openai.assert_called_once_with(api_key="test-key-123", timeout=30)
            assert client.config == mock_config

    @pytest.mark.asyncio
    async def test_chat_completion_success_gpt5(self, client):
        """Test successful chat completion with GPT-5 model."""
        # Mock the OpenAI client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Hello"}], tools=None
        )

        assert response.content == "Test response"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20

        # Verify Chat Completions API was called with max_completion_tokens
        client._client.chat.completions.create.assert_called_once_with(
            model="gpt-5",
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=4000,
        )

    @pytest.mark.asyncio
    async def test_chat_completion_success_responses_api(self):
        """Test successful chat completion with Responses API model."""
        # Create client with Responses API model
        config = OpenAIConfig(
            api_key="test-key-123", model_name="gpt-4.1-mini", max_tokens=4000, temperature=0.7
        )

        with patch("src.llm.openai_client.OpenAI"):
            client = OpenAIClient(config)

        # Mock the Responses API response with proper structure
        mock_response = Mock()
        mock_response.output_text = "Response from Responses API"
        mock_response.model = "gpt-4.1-mini"
        mock_response.usage = None
        # Ensure output is empty or not iterable to avoid errors
        mock_response.output = []

        # Add the responses attribute to the mock client
        client._client.responses = Mock()
        client._client.responses.create = AsyncMock(return_value=mock_response)

        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Hello"}], tools=None
        )

        assert response.content == "Response from Responses API"
        assert response.model == "gpt-4.1-mini"

        # Verify Responses API was called with input and max_output_tokens
        client._client.responses.create.assert_called_once_with(
            model="gpt-4.1-mini", input="Hello", max_output_tokens=4000, temperature=0.7
        )

    @pytest.mark.asyncio
    async def test_chat_completion_responses_api_includes_history(self):
        """Test Responses API input includes full conversation when multiple messages are provided."""
        config = OpenAIConfig(
            api_key="test-key-123", model_name="gpt-4.1-mini", max_tokens=4000, temperature=0.7
        )

        with patch("src.llm.openai_client.OpenAI"):
            client = OpenAIClient(config)

        mock_response = Mock()
        mock_response.output_text = "ok"
        mock_response.model = "gpt-4.1-mini"
        mock_response.usage = None
        mock_response.output = []

        client._client.responses = Mock()
        client._client.responses.create = AsyncMock(return_value=mock_response)

        messages = [
            {"role": "user", "content": "Recommend a hammer."},
            {"role": "assistant", "content": "Sure. What will you use it for?"},
            {"role": "user", "content": "Carpentry."},
        ]

        await client.chat_completion(messages=messages, tools=None)

        client._client.responses.create.assert_called_once()
        called_kwargs = client._client.responses.create.call_args.kwargs
        assert called_kwargs["model"] == "gpt-4.1-mini"
        assert "USER: Recommend a hammer." in called_kwargs["input"]
        assert "ASSISTANT: Sure. What will you use it for?" in called_kwargs["input"]
        assert "USER: Carpentry." in called_kwargs["input"]

    @pytest.mark.asyncio
    async def test_chat_completion_with_tools_gpt5(self, client):
        """Test chat completion with tools using GPT-5."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Tool response"
        mock_response.choices[0].message.tool_calls = []
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 25

        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Use tool"}], tools=tools
        )

        assert response.content == "Tool response"

        client._client.chat.completions.create.assert_called_once_with(
            model="gpt-5",
            messages=[{"role": "user", "content": "Use tool"}],
            max_completion_tokens=4000,
            tools=tools,
        )

    @pytest.mark.asyncio
    async def test_chat_completion_with_tools_responses_api(self):
        """Test chat completion with tools using Responses API."""
        # Create client with Responses API model
        config = OpenAIConfig(
            api_key="test-key-123", model_name="gpt-4o", max_tokens=4000, temperature=0.7
        )

        with patch("src.llm.openai_client.OpenAI"):
            client = OpenAIClient(config)

        mock_response = Mock()
        mock_response.output_text = "Tool response from Responses API"
        mock_response.model = "gpt-4o"
        mock_response.usage = None
        # Ensure output is empty or not iterable to avoid errors
        mock_response.output = []

        # Add the responses attribute to the mock client
        client._client.responses = Mock()
        client._client.responses.create = AsyncMock(return_value=mock_response)

        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Use tool"}], tools=tools
        )

        assert response.content == "Tool response from Responses API"

        client._client.responses.create.assert_called_once_with(
            model="gpt-4o", input="Use tool", max_output_tokens=4000, temperature=0.7, tools=tools
        )

    @pytest.mark.asyncio
    async def test_chat_completion_api_error(self, client):
        """Test handling of API errors."""
        import openai

        mock_request = Mock()
        mock_body = {"error": "API Error"}
        client._client.chat.completions.create = AsyncMock(
            side_effect=openai.APIError("API Error", request=mock_request, body=mock_body)
        )

        with pytest.raises(Exception, match="API Error"):
            await client.chat_completion(
                messages=[{"role": "user", "content": "Hello"}], tools=None
            )

    @pytest.mark.asyncio
    async def test_streaming_chat_completion_gpt5(self, client):
        """Test streaming chat completion with GPT-5."""
        # Mock streaming response
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" world"))]),
            Mock(choices=[Mock(delta=Mock(content="!"))]),
        ]

        async def mock_stream(*args, **kwargs):
            for chunk in mock_chunks:
                yield chunk

        client._client.chat.completions.create = AsyncMock()
        client._client.chat.completions.create.return_value = mock_stream()

        chunks = []
        async for chunk in client.stream_chat_completion(
            messages=[{"role": "user", "content": "Hello"}], tools=None
        ):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " world"
        assert chunks[2].choices[0].delta.content == "!"

    @pytest.mark.asyncio
    async def test_streaming_chat_completion_responses_api(self):
        """Test streaming chat completion with Responses API."""
        # Create client with Responses API model
        config = OpenAIConfig(
            api_key="test-key-123", model_name="gpt-4.1-mini", max_tokens=4000, temperature=0.7
        )

        with patch("src.llm.openai_client.OpenAI"):
            client = OpenAIClient(config)

        # Mock streaming response for Responses API
        mock_chunks = [
            Mock(type="text", text="Streamed "),
            Mock(type="text", text="response"),
            Mock(type="text", text="!"),
        ]

        async def mock_stream(*args, **kwargs):
            for chunk in mock_chunks:
                yield chunk

        # Add the responses attribute to the mock client
        client._client.responses = Mock()
        client._client.responses.create = AsyncMock()
        client._client.responses.create.return_value = mock_stream()

        chunks = []
        async for chunk in client.stream_chat_completion(
            messages=[{"role": "user", "content": "Hello"}], tools=None
        ):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].type == "text"
        assert chunks[0].text == "Streamed "
        assert chunks[1].text == "response"
        assert chunks[2].text == "!"

    def test_get_model_info_gpt5(self, client):
        """Test getting model info for GPT-5 model."""
        info = client.get_model_info()

        assert info["model_name"] == "gpt-5"
        assert info["api_type"] == "chat.completions"
        assert info["token_parameter"] == "max_completion_tokens"
        assert info["max_tokens"] == 4000
        assert info["supports_temperature"] is False
        assert info["temperature"] is None
        assert info["timeout"] == 30

    def test_get_model_info_responses_api(self):
        """Test getting model info for Responses API model."""
        # Create client with Responses API model
        config = OpenAIConfig(
            api_key="test-key-123",
            model_name="gpt-4.1-mini",
            max_tokens=2000,
            temperature=0.5,
            timeout=60,
        )

        with patch("src.llm.openai_client.OpenAI"):
            client = OpenAIClient(config)

        info = client.get_model_info()

        assert info["model_name"] == "gpt-4.1-mini"
        assert info["api_type"] == "responses"
        assert info["token_parameter"] == "max_output_tokens"
        assert info["max_tokens"] == 2000
        assert info["supports_temperature"] is True
        assert info["temperature"] == 0.5
        assert info["timeout"] == 60

    def test_get_model_info_gpt4(self):
        """Test getting model info for GPT-4 model."""
        # Create client with GPT-4 model
        config = OpenAIConfig(
            api_key="test-key-123", model_name="gpt-4", max_tokens=3000, temperature=0.8, timeout=45
        )

        with patch("src.llm.openai_client.OpenAI"):
            client = OpenAIClient(config)

        info = client.get_model_info()

        assert info["model_name"] == "gpt-4"
        assert info["api_type"] == "chat.completions"
        assert info["token_parameter"] == "max_tokens"
        assert info["max_tokens"] == 3000
        assert info["supports_temperature"] is True
        assert info["temperature"] == 0.8
        assert info["timeout"] == 45
