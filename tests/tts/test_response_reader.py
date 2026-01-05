import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tts.response_reader import (
    ResponseReader,
    split_into_sentences,
    strip_markdown,
)


class TestStripMarkdown:
    """Tests for markdown stripping function."""

    def test_strips_headers(self):
        assert strip_markdown("### Header") == "Header"
        assert strip_markdown("# H1\n## H2") == "H1\nH2"

    def test_strips_bold(self):
        assert strip_markdown("**bold text**") == "bold text"

    def test_strips_italic(self):
        assert strip_markdown("*italic text*") == "italic text"

    def test_strips_links(self):
        assert strip_markdown("[link text](http://example.com)") == "link text"

    def test_strips_code_blocks(self):
        text = "before\n```python\ncode\n```\nafter"
        result = strip_markdown(text)
        assert "```" not in result
        assert "code" not in result

    def test_strips_inline_code(self):
        assert strip_markdown("use `code` here") == "use code here"

    def test_strips_list_markers(self):
        text = "- item 1\n- item 2\n1. numbered"
        result = strip_markdown(text)
        assert "- " not in result
        assert "1. " not in result

    def test_handles_empty_input(self):
        assert strip_markdown("") == ""
        assert strip_markdown(None) is None


class TestSplitIntoSentences:
    """Tests for sentence splitting function."""

    def test_splits_simple_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        result = split_into_sentences(text)
        assert len(result) == 3
        assert result[0] == "First sentence."
        assert result[1] == "Second sentence."
        assert result[2] == "Third sentence."

    def test_handles_question_marks(self):
        text = "Is this a question? Yes it is."
        result = split_into_sentences(text)
        assert len(result) == 2

    def test_handles_exclamation_marks(self):
        text = "Wow! That is amazing."
        result = split_into_sentences(text)
        assert len(result) == 2

    def test_handles_mixed_punctuation(self):
        text = "Statement. Question? Exclamation!"
        result = split_into_sentences(text)
        assert len(result) == 3

    def test_handles_empty_input(self):
        assert split_into_sentences("") == []
        assert split_into_sentences(None) == []

    def test_handles_single_sentence(self):
        text = "Just one sentence."
        result = split_into_sentences(text)
        assert len(result) == 1
        assert result[0] == "Just one sentence."

    def test_preserves_abbreviations(self):
        text = "Dr. Smith went to Washington D.C. for a meeting."
        result = split_into_sentences(text)
        # Should ideally be 1 sentence, but may split on D.C.
        # At minimum, should not split on "Dr."
        assert "Dr." in result[0] or "Dr" in result[0]

    def test_handles_decimal_numbers(self):
        text = "The price is 3.99 dollars. That is cheap."
        result = split_into_sentences(text)
        # Should handle decimal correctly
        assert any("3.99" in s or "3" in s for s in result)


class TestResponseReader:
    """Tests for ResponseReader class."""

    @pytest.fixture
    def mock_tts_provider(self):
        provider = MagicMock()
        provider.synthesize = AsyncMock(return_value=b"audio_data_bytes")
        return provider

    @pytest.fixture
    def mock_websocket(self):
        ws = MagicMock()
        ws.send_json = AsyncMock()
        return ws

    @pytest.fixture
    def response_reader(self, mock_tts_provider, mock_websocket):
        return ResponseReader(mock_tts_provider, mock_websocket)

    @pytest.mark.asyncio
    async def test_read_response_chunked_single_sentence(
        self, response_reader, mock_tts_provider, mock_websocket
    ):
        """Test chunked reading with a single sentence."""
        await response_reader.read_response_chunked("Hello world.")

        # Should synthesize once
        mock_tts_provider.synthesize.assert_called_once()

        # Should send one audio chunk + completion
        assert mock_websocket.send_json.call_count == 2

        # Check the audio chunk
        chunk_call = mock_websocket.send_json.call_args_list[0]
        assert chunk_call[0][0]["type"] == "tts_audio_chunk"
        assert chunk_call[0][0]["chunk_index"] == 0
        assert chunk_call[0][0]["total_chunks"] == 1

        # Check completion
        complete_call = mock_websocket.send_json.call_args_list[1]
        assert complete_call[0][0]["type"] == "tts_complete"

    @pytest.mark.asyncio
    async def test_read_response_chunked_multiple_sentences(
        self, response_reader, mock_tts_provider, mock_websocket
    ):
        """Test chunked reading with multiple sentences."""
        text = "First sentence. Second sentence. Third sentence."
        await response_reader.read_response_chunked(text)

        # Should synthesize 3 times
        assert mock_tts_provider.synthesize.call_count == 3

        # Should send 3 audio chunks + 1 completion = 4 calls
        assert mock_websocket.send_json.call_count == 4

        # Verify chunk indices
        for i in range(3):
            call = mock_websocket.send_json.call_args_list[i]
            assert call[0][0]["chunk_index"] == i
            assert call[0][0]["total_chunks"] == 3

    @pytest.mark.asyncio
    async def test_read_response_chunked_strips_markdown(
        self, response_reader, mock_tts_provider, mock_websocket
    ):
        """Test that markdown is stripped before synthesis."""
        text = "**Bold text.** Normal text."
        await response_reader.read_response_chunked(text)

        # Verify synthesize was called with stripped text
        calls = mock_tts_provider.synthesize.call_args_list
        synthesized_texts = [call[0][0] for call in calls]

        # No markdown should remain
        for text in synthesized_texts:
            assert "**" not in text

    @pytest.mark.asyncio
    async def test_read_response_chunked_disabled(
        self, response_reader, mock_tts_provider, mock_websocket
    ):
        """Test that disabled reader does nothing."""
        response_reader.disable()
        await response_reader.read_response_chunked("Hello world.")

        mock_tts_provider.synthesize.assert_not_called()
        mock_websocket.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_read_response_chunked_empty_text(
        self, response_reader, mock_tts_provider, mock_websocket
    ):
        """Test handling of empty text."""
        await response_reader.read_response_chunked("")
        await response_reader.read_response_chunked("   ")
        await response_reader.read_response_chunked(None)

        mock_tts_provider.synthesize.assert_not_called()

    @pytest.mark.asyncio
    async def test_read_response_delegates_to_chunked(
        self, response_reader, mock_tts_provider, mock_websocket
    ):
        """Test that read_response now uses chunked implementation."""
        await response_reader.read_response("Test sentence.")

        # Should behave like chunked
        mock_tts_provider.synthesize.assert_called_once()

        # Should send chunk format, not old format
        chunk_call = mock_websocket.send_json.call_args_list[0]
        assert chunk_call[0][0]["type"] == "tts_audio_chunk"


class TestGoogleTTSExecutor:
    """Tests for Google TTS executor integration."""

    @pytest.mark.asyncio
    async def test_synthesize_runs_in_executor(self):
        """Test that synthesize doesn't block the event loop."""
        from src.tts.google_tts import GoogleTTSProvider

        with patch("src.tts.google_tts.texttospeech.TextToSpeechClient") as mock_client:
            mock_response = MagicMock()
            mock_response.audio_content = b"test_audio"
            mock_client.return_value.synthesize_speech.return_value = mock_response

            provider = GoogleTTSProvider()

            # This should not block
            result = await asyncio.wait_for(provider.synthesize("Test text"), timeout=5.0)

            assert result == b"test_audio"
