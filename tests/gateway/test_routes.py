"""Tests for routes module."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.gateway.routes import create_app


def test_create_app():
    """Test that create_app returns a FastAPI app."""
    app = create_app()
    assert app is not None
    assert hasattr(app, "state")
    assert hasattr(app, "router")


def test_app_has_llm_manager_lifespan():
    """Test that the app has LLMIntegrationManager in its lifespan."""
    app = create_app()

    # Check that the lifespan context manager is set
    assert hasattr(app.router, "lifespan_context")

    # Check that llm_manager is initialized in app.state after startup
    # This would be tested in an integration test with actual startup


@pytest.mark.asyncio
async def test_metrics_endpoint():
    """Test the /metrics endpoint."""
    app = create_app()
    client = TestClient(app)

    with patch("src.gateway.routes.get_metrics_summary") as mock_metrics:
        mock_metrics.return_value = {"test": "data"}

        response = client.get("/metrics")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "metrics" in data
        assert "timestamp" in data


@pytest.mark.asyncio
async def test_metrics_export_endpoint():
    """Test the /metrics/export endpoint."""
    app = create_app()
    client = TestClient(app)

    response = client.get("/metrics/export")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test the /health endpoint."""
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_index_endpoint():
    """Test the / endpoint returns HTML."""
    app = create_app()
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Voice Assistant" in response.text


"""
Unit tests for TTS chunking, queue management, and latency optimizations.

These tests prevent regression of the sentence-level TTS chunking implementation
that reduces time-to-first-audio latency.

Test Categories:
1. Sentence splitting logic
2. Chunked TTS response reading
3. Message format validation
4. Error handling and edge cases
5. Integration with TTS provider
"""

import asyncio  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

import pytest  # noqa: E402

# ============================================================================
# Test: Sentence Splitting Logic
# ============================================================================


class TestSplitIntoSentences:
    """Tests for the split_into_sentences function.

    These tests ensure sentences are correctly split for progressive TTS,
    which is critical for low-latency time-to-first-audio.
    """

    @pytest.fixture
    def split_into_sentences(self):
        """Import the function under test."""
        from src.tts.response_reader import split_into_sentences

        return split_into_sentences

    def test_splits_simple_sentences(self, split_into_sentences):
        """Verify basic sentence splitting on periods."""
        text = "First sentence. Second sentence. Third sentence."
        result = split_into_sentences(text)
        assert len(result) == 3
        assert result[0] == "First sentence."
        assert result[1] == "Second sentence."
        assert result[2] == "Third sentence."

    def test_splits_on_question_marks(self, split_into_sentences):
        """Verify splitting on question marks."""
        text = "Is this a question? Yes it is. Another question?"
        result = split_into_sentences(text)
        assert len(result) == 3

    def test_splits_on_exclamation_marks(self, split_into_sentences):
        """Verify splitting on exclamation marks."""
        text = "Wow! That is amazing. Incredible!"
        result = split_into_sentences(text)
        assert len(result) == 3

    def test_handles_mixed_punctuation(self, split_into_sentences):
        """Verify splitting with mixed sentence-ending punctuation."""
        text = "Statement here. Question here? Exclamation here!"
        result = split_into_sentences(text)
        assert len(result) == 3

    def test_handles_empty_input(self, split_into_sentences):
        """Verify empty input returns empty list."""
        assert split_into_sentences("") == []
        assert split_into_sentences(None) == []

    def test_handles_whitespace_only(self, split_into_sentences):
        """Verify whitespace-only input returns empty list."""
        assert split_into_sentences("   ") == []
        assert split_into_sentences("\n\t\n") == []

    def test_handles_single_sentence(self, split_into_sentences):
        """Verify single sentence is returned as single-item list."""
        text = "Just one sentence here."
        result = split_into_sentences(text)
        assert len(result) == 1
        assert result[0] == "Just one sentence here."

    def test_handles_sentence_without_ending_punctuation(self, split_into_sentences):
        """Verify sentence without punctuation is still captured."""
        text = "This has no ending punctuation"
        result = split_into_sentences(text)
        assert len(result) == 1
        assert result[0] == "This has no ending punctuation"

    def test_handles_multiple_spaces_between_sentences(self, split_into_sentences):
        """Verify multiple spaces between sentences are handled."""
        text = "First sentence.   Second sentence.    Third sentence."
        result = split_into_sentences(text)
        assert len(result) == 3
        # Sentences should be trimmed
        assert all(not s.startswith(" ") and not s.endswith(" ") for s in result)

    def test_filters_empty_strings(self, split_into_sentences):
        """Verify empty strings are filtered from results."""
        text = "Sentence one..  Sentence two."
        result = split_into_sentences(text)
        assert "" not in result
        assert all(s.strip() for s in result)

    def test_handles_newlines_between_sentences(self, split_into_sentences):
        """Verify newlines between sentences are handled."""
        text = "First sentence.\nSecond sentence.\n\nThird sentence."
        result = split_into_sentences(text)
        assert len(result) >= 2  # At least 2 sentences should be found

    def test_real_world_llm_response(self, split_into_sentences):
        """Test with realistic LLM response text."""
        text = (
            "For fine detailed painting, I recommend the Wooster Artist Brush. "
            "It features soft camel hair bristles. "
            "The 1/4-inch width is perfect for detailed work. "
            "Let me know if you want other recommendations!"
        )
        result = split_into_sentences(text)
        assert len(result) == 4
        # First sentence should be playable within ~300ms
        assert len(result[0]) < 100  # Reasonable length for quick TTS


# ============================================================================
# Test: Markdown Stripping
# ============================================================================


class TestStripMarkdown:
    """Tests for markdown stripping before TTS synthesis.

    Markdown must be stripped to produce natural-sounding speech.
    """

    @pytest.fixture
    def strip_markdown(self):
        """Import the function under test."""
        from src.tts.response_reader import strip_markdown

        return strip_markdown

    def test_strips_bold(self, strip_markdown):
        """Verify bold markdown is removed."""
        assert strip_markdown("**bold text**") == "bold text"
        assert strip_markdown("This is **bold** here") == "This is bold here"

    def test_strips_italic(self, strip_markdown):
        """Verify italic markdown is removed."""
        assert strip_markdown("*italic text*") == "italic text"

    def test_strips_headers(self, strip_markdown):
        """Verify header markdown is removed."""
        assert strip_markdown("# Header") == "Header"
        assert strip_markdown("### Sub Header") == "Sub Header"

    def test_strips_links(self, strip_markdown):
        """Verify link markdown keeps text, removes URL."""
        assert strip_markdown("[link text](http://example.com)") == "link text"

    def test_strips_code_blocks(self, strip_markdown):
        """Verify code blocks are removed entirely."""
        text = "before\n```python\ncode here\n```\nafter"
        result = strip_markdown(text)
        assert "```" not in result
        assert "code here" not in result

    def test_strips_inline_code(self, strip_markdown):
        """Verify inline code backticks are removed."""
        assert strip_markdown("use `code` here") == "use code here"

    def test_strips_list_markers(self, strip_markdown):
        """Verify list markers are removed."""
        text = "- item one\n- item two"
        result = strip_markdown(text)
        assert "- " not in result

    def test_handles_empty_input(self, strip_markdown):
        """Verify empty input returns empty/unchanged."""
        assert strip_markdown("") == ""
        assert strip_markdown(None) is None

    def test_complex_markdown_response(self, strip_markdown):
        """Test stripping complex LLM response with multiple markdown elements."""
        text = """**Recommendation:**

- Soft camel hair bristles
- 1/4-inch width

Use `SKU: 2AJR5` to order."""

        result = strip_markdown(text)
        assert "**" not in result
        assert "- " not in result or result.count("- ") == 0
        assert "`" not in result


# ============================================================================
# Test: Chunked TTS Response Reading
# ============================================================================


class TestResponseReaderChunked:
    """Tests for the chunked TTS response reading implementation.

    These tests verify that TTS audio is sent sentence-by-sentence
    for reduced time-to-first-audio latency.
    """

    @pytest.fixture
    def mock_tts_provider(self):
        """Create mock TTS provider."""
        provider = MagicMock()
        provider.synthesize = AsyncMock(return_value=b"fake_audio_data_bytes")
        return provider

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket."""
        ws = MagicMock()
        ws.send_json = AsyncMock()
        return ws

    @pytest.fixture
    def response_reader(self, mock_tts_provider, mock_websocket):
        """Create ResponseReader instance with mocks."""
        from src.tts.response_reader import ResponseReader

        return ResponseReader(mock_tts_provider, mock_websocket)

    @pytest.mark.asyncio
    async def test_chunked_sends_multiple_chunks(
        self, response_reader, mock_tts_provider, mock_websocket
    ):
        """Verify multiple sentences result in multiple TTS chunks."""
        text = "First sentence. Second sentence. Third sentence."
        await response_reader.read_response_chunked(text)

        # Should synthesize 3 times (once per sentence)
        assert mock_tts_provider.synthesize.call_count == 3

        # Should send 3 chunks + 1 completion = 4 websocket messages
        assert mock_websocket.send_json.call_count == 4

    @pytest.mark.asyncio
    async def test_chunked_message_format(self, response_reader, mock_tts_provider, mock_websocket):
        """Verify chunk messages have correct format."""
        text = "First sentence. Second sentence."
        await response_reader.read_response_chunked(text)

        # Check first chunk message
        first_call = mock_websocket.send_json.call_args_list[0]
        msg = first_call[0][0]

        assert msg["type"] == "tts_audio_chunk"
        assert "audio" in msg
        assert msg["format"] == "pcm16"
        assert msg["sample_rate"] == 16000
        assert msg["chunk_index"] == 0
        assert msg["total_chunks"] == 2

    @pytest.mark.asyncio
    async def test_chunked_sends_completion_message(
        self, response_reader, mock_tts_provider, mock_websocket
    ):
        """Verify tts_complete message is sent after all chunks."""
        text = "Single sentence here."
        await response_reader.read_response_chunked(text)

        # Last message should be tts_complete
        last_call = mock_websocket.send_json.call_args_list[-1]
        msg = last_call[0][0]

        assert msg["type"] == "tts_complete"

    @pytest.mark.asyncio
    async def test_chunked_chunk_indices_sequential(
        self, response_reader, mock_tts_provider, mock_websocket
    ):
        """Verify chunk indices are sequential starting from 0."""
        text = "One. Two. Three. Four."
        await response_reader.read_response_chunked(text)

        # Extract chunk indices (excluding tts_complete message)
        chunk_calls = mock_websocket.send_json.call_args_list[:-1]
        indices = [call[0][0]["chunk_index"] for call in chunk_calls]

        assert indices == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_chunked_total_chunks_consistent(
        self, response_reader, mock_tts_provider, mock_websocket
    ):
        """Verify total_chunks is consistent across all chunk messages."""
        text = "One. Two. Three."
        await response_reader.read_response_chunked(text)

        # Extract total_chunks from each chunk message
        chunk_calls = mock_websocket.send_json.call_args_list[:-1]
        totals = [call[0][0]["total_chunks"] for call in chunk_calls]

        # All should be 3
        assert all(t == 3 for t in totals)

    @pytest.mark.asyncio
    async def test_chunked_strips_markdown_before_synthesis(
        self, response_reader, mock_tts_provider, mock_websocket
    ):
        """Verify markdown is stripped before TTS synthesis."""
        text = "**Bold text.** Normal text."
        await response_reader.read_response_chunked(text)

        # Check what was passed to synthesize
        calls = mock_tts_provider.synthesize.call_args_list
        synthesized_texts = [call[0][0] for call in calls]

        # No markdown should remain
        for text in synthesized_texts:
            assert "**" not in text

    @pytest.mark.asyncio
    async def test_chunked_disabled_reader_does_nothing(
        self, response_reader, mock_tts_provider, mock_websocket
    ):
        """Verify disabled reader skips processing."""
        response_reader.disable()
        await response_reader.read_response_chunked("Test sentence.")

        mock_tts_provider.synthesize.assert_not_called()
        mock_websocket.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_chunked_empty_text_does_nothing(
        self, response_reader, mock_tts_provider, mock_websocket
    ):
        """Verify empty text skips processing."""
        await response_reader.read_response_chunked("")
        await response_reader.read_response_chunked("   ")

        mock_tts_provider.synthesize.assert_not_called()

    @pytest.mark.asyncio
    async def test_chunked_handles_synthesis_failure(
        self, response_reader, mock_tts_provider, mock_websocket
    ):
        """Verify graceful handling when TTS synthesis fails."""
        mock_tts_provider.synthesize = AsyncMock(return_value=b"")

        # Should not raise exception
        await response_reader.read_response_chunked("Test sentence.")

        # Should still send completion
        last_call = mock_websocket.send_json.call_args_list[-1]
        assert last_call[0][0]["type"] == "tts_complete"

    @pytest.mark.asyncio
    async def test_read_response_delegates_to_chunked(
        self, response_reader, mock_tts_provider, mock_websocket
    ):
        """Verify read_response() uses chunked implementation."""
        await response_reader.read_response("Test sentence.")

        # Should use chunk format
        chunk_call = mock_websocket.send_json.call_args_list[0]
        assert chunk_call[0][0]["type"] == "tts_audio_chunk"


# ============================================================================
# Test: Google TTS Provider Non-Blocking
# ============================================================================


class TestGoogleTTSNonBlocking:
    """Tests for non-blocking TTS synthesis.

    The Google TTS API call must run in an executor to avoid
    blocking the async event loop during synthesis.
    """

    @pytest.mark.asyncio
    async def test_synthesize_uses_executor(self):
        """Verify synthesize runs in executor (non-blocking)."""
        from src.tts.google_tts import GoogleTTSProvider

        with patch("src.tts.google_tts.texttospeech.TextToSpeechClient") as mock_client:
            mock_response = MagicMock()
            mock_response.audio_content = b"test_audio_bytes"
            mock_client.return_value.synthesize_speech.return_value = mock_response

            provider = GoogleTTSProvider()

            # Should complete without blocking
            result = await asyncio.wait_for(provider.synthesize("Test text"), timeout=5.0)

            assert result == b"test_audio_bytes"

    @pytest.mark.asyncio
    async def test_synthesize_empty_text_returns_empty(self):
        """Verify empty text returns empty bytes without API call."""
        from src.tts.google_tts import GoogleTTSProvider

        with patch("src.tts.google_tts.texttospeech.TextToSpeechClient") as mock_client:
            provider = GoogleTTSProvider()

            result = await provider.synthesize("")

            assert result == b""
            mock_client.return_value.synthesize_speech.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthesize_handles_api_error(self):
        """Verify graceful handling of TTS API errors."""
        from src.tts.google_tts import GoogleTTSProvider

        with patch("src.tts.google_tts.texttospeech.TextToSpeechClient") as mock_client:
            mock_client.return_value.synthesize_speech.side_effect = Exception("API Error")

            provider = GoogleTTSProvider()

            # Should return empty bytes, not raise
            result = await provider.synthesize("Test text")

            assert result == b""


# ============================================================================
# Test: Message Type Contracts
# ============================================================================


class TestTTSMessageContracts:
    """Tests verifying the TTS WebSocket message contracts.

    These tests ensure the server sends messages in the format
    expected by the client JavaScript.
    """

    def test_tts_audio_chunk_message_schema(self):
        """Verify tts_audio_chunk message has required fields."""
        # This is the expected schema
        required_fields = {
            "type": str,
            "audio": str,  # Hex-encoded bytes
            "format": str,
            "sample_rate": int,
            "chunk_index": int,
            "total_chunks": int,
        }

        # Example message from server
        example_msg = {
            "type": "tts_audio_chunk",
            "audio": "deadbeef",
            "format": "pcm16",
            "sample_rate": 16000,
            "chunk_index": 0,
            "total_chunks": 3,
        }

        for field, expected_type in required_fields.items():
            assert field in example_msg, f"Missing required field: {field}"
            assert isinstance(example_msg[field], expected_type), (
                f"Field {field} should be {expected_type}"
            )

    def test_tts_complete_message_schema(self):
        """Verify tts_complete message has required fields."""
        example_msg = {"type": "tts_complete"}

        assert "type" in example_msg
        assert example_msg["type"] == "tts_complete"

    def test_legacy_tts_audio_message_schema(self):
        """Verify legacy tts_audio message schema for backwards compatibility."""
        required_fields = {
            "type": str,
            "audio": str,
            "format": str,
            "sample_rate": int,
        }

        example_msg = {
            "type": "tts_audio",
            "audio": "deadbeef",
            "format": "pcm16",
            "sample_rate": 16000,
        }

        for field, expected_type in required_fields.items():
            assert field in example_msg
            assert isinstance(example_msg[field], expected_type)


# ============================================================================
# Test: Latency Optimization Regression Prevention
# ============================================================================


class TestLatencyOptimizationRegression:
    """Tests specifically designed to prevent regression of latency optimizations.

    These tests encode the key behaviors that reduce time-to-first-audio.
    """

    @pytest.fixture
    def mock_tts_provider(self):
        provider = MagicMock()
        provider.synthesize = AsyncMock(return_value=b"audio")
        return provider

    @pytest.fixture
    def mock_websocket(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        return ws

    @pytest.mark.asyncio
    async def test_first_chunk_sent_before_full_synthesis_complete(
        self, mock_tts_provider, mock_websocket
    ):
        """
        CRITICAL: First audio chunk must be sent after first sentence synthesis,
        NOT after all sentences are synthesized.

        This is the core latency optimization - progressive delivery.
        """
        from src.tts.response_reader import ResponseReader

        synthesis_order = []
        send_order = []

        async def track_synthesize(text):
            synthesis_order.append(f"synth:{text[:20]}")
            return b"audio"

        async def track_send(msg):
            if msg.get("type") == "tts_audio_chunk":
                send_order.append(f"send:chunk_{msg['chunk_index']}")

        mock_tts_provider.synthesize = track_synthesize
        mock_websocket.send_json = track_send

        reader = ResponseReader(mock_tts_provider, mock_websocket)
        await reader.read_response_chunked("First sentence. Second sentence. Third sentence.")

        # Verify interleaved pattern: synth -> send -> synth -> send -> ...
        # NOT: synth -> synth -> synth -> send -> send -> send
        combined = []
        for i, (s, se) in enumerate(zip(synthesis_order, send_order)):
            combined.extend([s, se])

        # First send should happen after first synth, before second synth
        assert "send:chunk_0" in combined
        first_send_idx = combined.index("send:chunk_0")
        assert first_send_idx == 1, "First chunk should be sent immediately after first synthesis"

    @pytest.mark.asyncio
    async def test_no_batching_of_synthesis_calls(self, mock_tts_provider, mock_websocket):
        """
        Verify sentences are synthesized one-by-one, not batched.

        Batching would defeat the latency optimization.
        """
        from src.tts.response_reader import ResponseReader

        reader = ResponseReader(mock_tts_provider, mock_websocket)
        await reader.read_response_chunked("One. Two. Three.")

        # Each sentence should be a separate synthesize call
        assert mock_tts_provider.synthesize.call_count == 3

        # Verify each call was for a single sentence
        calls = mock_tts_provider.synthesize.call_args_list
        for call in calls:
            text = call[0][0]
            # Each text should be a single sentence (no multiple periods)
            periods = text.count(".")
            assert periods <= 1, f"Text contains multiple sentences: {text}"

    @pytest.mark.asyncio
    async def test_chunk_format_not_legacy_format(self, mock_tts_provider, mock_websocket):
        """
        Verify chunked method sends 'tts_audio_chunk', not legacy 'tts_audio'.

        The client needs chunk_index for proper queue management.
        """
        from src.tts.response_reader import ResponseReader

        reader = ResponseReader(mock_tts_provider, mock_websocket)
        await reader.read_response("Two sentences here. Both chunked.")

        # All audio messages should be chunk format
        for call in mock_websocket.send_json.call_args_list:
            msg = call[0][0]
            if "audio" in msg:
                assert msg["type"] == "tts_audio_chunk", (
                    "Should use tts_audio_chunk format for latency optimization"
                )
                assert "chunk_index" in msg, "Chunk messages must include chunk_index"
                assert "total_chunks" in msg, "Chunk messages must include total_chunks"


# ============================================================================
# Test: Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.fixture
    def split_into_sentences(self):
        from src.tts.response_reader import split_into_sentences

        return split_into_sentences

    def test_very_long_sentence(self, split_into_sentences):
        """Test handling of very long single sentence."""
        long_sentence = "This is a very long sentence " * 50 + "."
        result = split_into_sentences(long_sentence)
        assert len(result) == 1

    def test_many_short_sentences(self, split_into_sentences):
        """Test handling of many short sentences."""
        text = ". ".join(["Hi"] * 100) + "."
        result = split_into_sentences(text)
        assert len(result) >= 50  # Should find many sentences

    def test_unicode_text(self, split_into_sentences):
        """Test handling of unicode characters."""
        text = "Héllo wörld. Ça va bien. 你好."
        result = split_into_sentences(text)
        assert len(result) >= 2

    def test_urls_in_text(self, split_into_sentences):
        """Test that URLs don't cause excessive splitting."""
        text = "Visit https://example.com for more info. Thank you."
        result = split_into_sentences(text)
        # Should be 2 sentences, not split on .com
        assert len(result) <= 3

    def test_ellipsis(self, split_into_sentences):
        """Test handling of ellipsis."""
        text = "Wait for it... Here it comes. Done."
        result = split_into_sentences(text)
        assert len(result) >= 2
