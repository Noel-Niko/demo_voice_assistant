"""
Test to verify the async callback fix works correctly.
"""

import asyncio
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

# Mock the imports that cause issues
sys.modules["google.cloud.speech_v2"] = Mock()
sys.modules["google.cloud.speech_v2.types"] = Mock()
sys.modules["google.cloud.speech_v2.types.cloud_speech"] = Mock()

from src.asr.google_speech_v2 import GoogleSpeechV2Provider  # noqa: E402


class TestFixVerification(unittest.TestCase):
    """Test that the async callback fix works correctly."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = GoogleSpeechV2Provider()

    def test_safe_call_callback_with_sync_callback(self):
        """Test _safe_call_callback with synchronous callback."""
        # Create a sync callback
        sync_callback = Mock()
        self.provider.transcription_callback = sync_callback

        # Call the safe callback method
        test_transcript = "Hello world"
        self.provider._safe_call_callback(test_transcript)

        # Verify the callback was called
        sync_callback.assert_called_once_with(test_transcript)

    def test_safe_call_callback_with_async_callback(self):
        """Test _safe_call_callback with async callback."""
        # Create an async callback
        async_callback = AsyncMock()
        self.provider.transcription_callback = async_callback

        # Call the safe callback method
        test_transcript = "Hello world"

        async def test_async():
            # This should create a task without error
            self.provider._safe_call_callback(test_transcript)

            # Give the task time to execute
            await asyncio.sleep(0.1)

            # Verify the async callback was called
            async_callback.assert_called_once_with(test_transcript)

        # Run the async test
        asyncio.run(test_async())

    def test_safe_call_callback_handles_exceptions(self):
        """Test that _safe_call_callback handles exceptions gracefully."""

        # Create a callback that raises an exception
        def failing_callback(transcript):
            raise ValueError("Test error")

        self.provider.transcription_callback = failing_callback

        # Call the safe callback method - should not raise exception
        with patch("src.asr.google_speech_v2.logger") as mock_logger:
            self.provider._safe_call_callback("test transcript")

            # Verify the error was logged
            mock_logger.error.assert_called_once()

    def test_inspection_detects_async_correctly(self):
        """Test that async detection works correctly."""
        import inspect

        # Create sync and async functions
        def sync_func(x):
            return x

        async def async_func(x):
            return x

        # Test inspection
        self.assertFalse(inspect.iscoroutinefunction(sync_func))
        self.assertTrue(inspect.iscoroutinefunction(async_func))

    def test_integration_scenario(self):
        """Test the complete integration scenario."""

        async def test_integration():
            # Mock the async callback (like AudioSession._handle_transcript)
            mock_websocket = AsyncMock()

            async def mock_handle_transcript(transcript: str):
                """Mock async _handle_transcript."""
                if "test" in transcript.lower():
                    await mock_websocket.send_json({"type": "final_transcript", "text": transcript})

            # Set up provider with async callback
            self.provider.transcription_callback = mock_handle_transcript

            # Call the safe callback method
            test_transcript = "test transcript"
            self.provider._safe_call_callback(test_transcript)

            # Give the async task time to execute
            await asyncio.sleep(0.1)

            # Verify the websocket was called
            mock_websocket.send_json.assert_called()
            call_args = mock_websocket.send_json.call_args[0][0]
            self.assertEqual(call_args["text"], test_transcript)

        # Run the integration test
        asyncio.run(test_integration())


if __name__ == "__main__":
    unittest.main()
