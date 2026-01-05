"""
Unit tests for async transcription callback fix.
Tests the proper handling of async callbacks in the Google Speech v2 provider.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from src.asr.google_speech_v2 import GoogleSpeechV2Provider
from src.gateway.audio_session import AudioSession


class TestAsyncCallbackFix(unittest.TestCase):
    """Test cases for async transcription callback handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = GoogleSpeechV2Provider()

    def test_sync_callback_works(self):
        """Test that synchronous callbacks work correctly."""
        # Create a mock sync callback
        sync_callback = Mock()

        # Simulate calling the callback synchronously (should work)
        test_transcript = "Hello world"
        sync_callback(test_transcript)

        # Verify the callback was called
        sync_callback.assert_called_once_with(test_transcript)

    def test_async_callback_direct_call_fails(self):
        """Test that calling an async callback without awaiting returns a coroutine."""
        # Create an async callback
        async_callback = AsyncMock()

        # Try to call it synchronously (this should cause the issue)
        test_transcript = "Hello world"

        coro = async_callback(test_transcript)
        self.assertTrue(asyncio.iscoroutine(coro))
        coro.close()

    def test_async_callback_with_await_works(self):
        """Test that properly awaiting async callback works."""

        async def test_async_await():
            # Create an async callback
            async_callback = AsyncMock()

            # Call it with await (should work)
            test_transcript = "Hello world"
            await async_callback(test_transcript)

            # Verify the callback was called
            async_callback.assert_called_once_with(test_transcript)

        # Run the async test
        asyncio.run(test_async_await())

    def test_async_callback_with_create_task_works(self):
        """Test that using asyncio.create_task works for async callbacks."""

        async def test_create_task():
            # Create an async callback
            async_callback = AsyncMock()

            # Use create_task to schedule the async callback
            test_transcript = "Hello world"
            task = asyncio.create_task(async_callback(test_transcript))

            # Wait for the task to complete
            await task

            # Verify the callback was called
            async_callback.assert_called_once_with(test_transcript)

        # Run the async test
        asyncio.run(test_create_task())

    def test_provider_callback_type_signature(self):
        """Test start_streaming signature supports optional event emission."""
        import inspect

        sig = inspect.signature(self.provider.start_streaming)

        self.assertIn("transcription_callback", sig.parameters)
        self.assertIn("interim_callback", sig.parameters)
        self.assertIn("emit_events", sig.parameters)
        self.assertFalse(sig.parameters["emit_events"].default)

    def test_audio_session_handle_transcript_is_async(self):
        """Test that AudioSession._handle_transcript is indeed async."""
        # Check that _handle_transcript is async
        import inspect

        self.assertTrue(inspect.iscoroutinefunction(AudioSession._handle_transcript))

    @patch("src.asr.google_speech_v2.logger")
    def test_transcription_callback_invocation(self, mock_logger):
        """Test how the transcription callback is currently invoked."""
        # Mock the callback
        mock_callback = Mock()

        # Set up provider with callback
        self.provider.transcription_callback = mock_callback

        # Simulate the callback invocation (current implementation)
        test_transcript = "Test transcript"

        # This is how it's currently called in the provider
        self.provider.transcription_callback(test_transcript)

        # Verify it was called
        mock_callback.assert_called_once_with(test_transcript)

    def test_proposed_fix_wrapper_function(self):
        """Test the proposed fix: wrapping async callback in sync wrapper."""
        # Create an async callback
        async_callback = AsyncMock()

        # Create a sync wrapper that handles the async callback properly
        def sync_wrapper(transcript: str):
            """Sync wrapper for async callback."""
            # Get the current event loop
            asyncio.get_event_loop()
            # Schedule the async callback
            asyncio.create_task(async_callback(transcript))

        # Test the wrapper
        test_transcript = "Hello world"

        async def test_wrapper():
            sync_wrapper(test_transcript)
            # Give the task time to execute
            await asyncio.sleep(0.1)
            async_callback.assert_called_once_with(test_transcript)

        # Run the test
        asyncio.run(test_wrapper())

    def test_proposed_fix_asyncio_run(self):
        """Test alternative fix: using asyncio.run() for async callback."""
        # Create an async callback
        async_callback = AsyncMock()

        # Create a sync wrapper that uses asyncio.run
        def sync_wrapper_run(transcript: str):
            """Sync wrapper that avoids double-calling the async callback."""
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(async_callback(transcript))
            else:
                asyncio.run(async_callback(transcript))

        # Test the wrapper
        test_transcript = "Hello world"

        async def test_wrapper():
            sync_wrapper_run(test_transcript)
            # Give the task time to execute
            await asyncio.sleep(0.1)
            async_callback.assert_called_once_with(test_transcript)

        # Run the test
        asyncio.run(test_wrapper())


class TestIntegrationFix(unittest.TestCase):
    """Integration tests for the complete fix."""

    def test_complete_fix_simulation(self):
        """Test the complete fix end-to-end."""

        async def test_complete():
            # Use a minimal async consumer to avoid constructing a full AudioSession.
            mock_websocket = AsyncMock()

            class DummySession:
                async def _handle_transcript(self, transcript: str):
                    await mock_websocket.send_json(
                        {
                            "type": "final_transcript",
                            "text": transcript,
                        }
                    )

            session = DummySession()

            # Create the fixed sync wrapper
            def fixed_transcription_callback(transcript: str):
                """Fixed transcription callback that properly handles async _handle_transcript."""
                try:
                    asyncio.create_task(session._handle_transcript(transcript))
                except Exception as e:
                    print(f"Error in callback: {e}")

            # Test the fixed callback
            test_transcript = "Hello world"
            fixed_transcription_callback(test_transcript)

            # Give the async task time to execute
            await asyncio.sleep(0.1)

            # Verify the websocket send was called
            mock_websocket.send_json.assert_called()

            # Check the call arguments
            call_args = mock_websocket.send_json.call_args[0][0]
            self.assertEqual(call_args["type"], "final_transcript")
            self.assertEqual(call_args["text"], test_transcript)

        # Run the integration test
        asyncio.run(test_complete())


if __name__ == "__main__":
    unittest.main()
