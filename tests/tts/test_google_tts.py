"""
Tests for Google Cloud TTS provider.
"""

from unittest.mock import Mock, patch

import pytest

from src.tts.google_tts import GoogleTTSProvider


@pytest.mark.asyncio
@patch("src.tts.google_tts.texttospeech.TextToSpeechClient")
async def test_google_tts_initialization(mock_client_class):
    """Test TTS provider initialization."""
    provider = GoogleTTSProvider(language_code="en-US", speaking_rate=1.0, pitch=0.0)

    assert provider.language_code == "en-US"
    assert provider.speaking_rate == 1.0
    assert provider.pitch == 0.0


@pytest.mark.asyncio
@patch("src.tts.google_tts.texttospeech.TextToSpeechClient")
async def test_synthesize_empty_text(mock_client_class):
    """Test synthesis with empty text."""
    provider = GoogleTTSProvider()

    result = await provider.synthesize("")
    assert result == b""

    result = await provider.synthesize("   ")
    assert result == b""


@pytest.mark.asyncio
@patch("src.tts.google_tts.texttospeech.TextToSpeechClient")
async def test_synthesize_success(mock_client_class):
    """Test successful synthesis."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.audio_content = b"fake_audio_data"
    mock_client.synthesize_speech.return_value = mock_response
    mock_client_class.return_value = mock_client

    provider = GoogleTTSProvider()
    result = await provider.synthesize("Hello world")

    assert result == b"fake_audio_data"
    assert mock_client.synthesize_speech.called


@pytest.mark.asyncio
@patch("src.tts.google_tts.texttospeech.TextToSpeechClient")
async def test_synthesize_stream(mock_client_class):
    """Test streaming synthesis."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.audio_content = b"x" * 10000
    mock_client.synthesize_speech.return_value = mock_response
    mock_client_class.return_value = mock_client

    provider = GoogleTTSProvider()
    chunks = []
    async for chunk in provider.synthesize_stream("Hello world"):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert b"".join(chunks) == b"x" * 10000
