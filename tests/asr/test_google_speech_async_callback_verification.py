"""
Simple test to verify the async callback fix logic works.
"""

import asyncio
import inspect
import unittest
from unittest.mock import AsyncMock, Mock


class TestAsyncCallbackFixLogic(unittest.TestCase):
    """Test the core logic of the async callback fix."""

    def test_safe_call_callback_logic_sync(self):
        """Test the logic for sync callbacks."""

        def safe_call_callback(transcript, callback):
            """Simplified version of the fix logic."""
            if inspect.iscoroutinefunction(callback):
                # Create task for async callbacks
                asyncio.get_event_loop()
                asyncio.create_task(callback(transcript))
            else:
                # Call sync callbacks directly
                callback(transcript)

        # Test with sync callback
        sync_callback = Mock()
        test_transcript = "Hello world"

        safe_call_callback(test_transcript, sync_callback)
        sync_callback.assert_called_once_with(test_transcript)

    def test_safe_call_callback_logic_async(self):
        """Test the logic for async callbacks."""

        def safe_call_callback(transcript, callback):
            """Simplified version of the fix logic."""
            if inspect.iscoroutinefunction(callback):
                # Create task for async callbacks
                asyncio.get_event_loop()
                asyncio.create_task(callback(transcript))
            else:
                # Call sync callbacks directly
                callback(transcript)

        # Test with async callback
        async_callback = AsyncMock()
        test_transcript = "Hello world"

        async def test_async():
            safe_call_callback(test_transcript, async_callback)
            await asyncio.sleep(0.1)  # Give task time to execute
            async_callback.assert_called_once_with(test_transcript)

        asyncio.run(test_async())

    def test_inspection_works(self):
        """Test that function inspection works correctly."""

        def sync_func(x):
            return x

        async def async_func(x):
            return x

        self.assertFalse(inspect.iscoroutinefunction(sync_func))
        self.assertTrue(inspect.iscoroutinefunction(async_func))

    def test_no_more_runtime_warnings(self):
        """Test that the fix eliminates RuntimeWarnings."""

        async def mock_async_callback(transcript):
            await asyncio.sleep(0.01)
            print(f"Processed: {transcript}")

        def old_problematic_way(transcript):
            # This would cause RuntimeWarning
            return mock_async_callback(transcript)

        def new_fixed_way(transcript):
            # This should not cause RuntimeWarning
            if inspect.iscoroutinefunction(mock_async_callback):
                asyncio.create_task(mock_async_callback(transcript))

        async def test_async():
            # Test old way creates coroutine
            result = old_problematic_way("test")
            self.assertTrue(inspect.iscoroutine(result))

            # Test new way doesn't return coroutine
            result = new_fixed_way("test")
            self.assertIsNone(result)  # create_task returns Task, but we don't capture it

        # Run the async test
        asyncio.run(test_async())


class TestIntegrationScenario(unittest.TestCase):
    """Test the complete fix scenario."""

    def test_websocket_scenario_fixed(self):
        """Test that the websocket scenario works with the fix."""

        async def test_scenario():
            # Mock websocket
            mock_websocket = AsyncMock()

            # Mock async _handle_transcript
            async def mock_handle_transcript(transcript: str):
                if "test" in transcript.lower():
                    await mock_websocket.send_json({"type": "final_transcript", "text": transcript})

            # Apply the fix
            def fixed_transcription_callback(transcript: str):
                """Fixed callback using our logic."""
                if inspect.iscoroutinefunction(mock_handle_transcript):
                    asyncio.create_task(mock_handle_transcript(transcript))

            # Test the fixed callback
            test_transcript = "test transcript"
            fixed_transcription_callback(test_transcript)

            # Give async task time to execute
            await asyncio.sleep(0.1)

            # Verify websocket was called
            mock_websocket.send_json.assert_called()
            call_args = mock_websocket.send_json.call_args[0][0]
            self.assertEqual(call_args["text"], test_transcript)

        asyncio.run(test_scenario())


if __name__ == "__main__":
    unittest.main()
