"""
Unit tests for thread-safe async callback handling.
Tests the fix for "no current event loop in thread" issue.
"""

import asyncio
import threading
import unittest


class TestThreadSafeAsyncCallback(unittest.TestCase):
    """Test thread-safe async callback handling."""

    def test_event_loop_issue_in_thread(self):
        """Test the event loop issue that occurs in background threads."""

        async def mock_async_callback(transcript):
            await asyncio.sleep(0.01)
            return f"Processed: {transcript}"

        def problematic_callback_in_thread(transcript):
            """This simulates the current problematic approach."""
            try:
                asyncio.get_event_loop()
                coro = mock_async_callback(transcript)
                asyncio.create_task(coro)
            except RuntimeError as e:
                try:
                    coro.close()
                except Exception:
                    pass
                return f"Error: {e}"

        # Test in main thread (should work but might fail if no event loop)
        result = problematic_callback_in_thread("test")
        # In test environment, there might not be a running event loop
        if result is not None:
            self.assertTrue(
                ("no running event loop" in result) or ("no current event loop" in result),
                f"Unexpected error message: {result}",
            )

        # Test in background thread (should fail)
        result_container = []

        def run_in_thread():
            result = problematic_callback_in_thread("test")
            result_container.append(result)

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()

        # Should contain the error
        self.assertTrue(len(result_container) > 0)
        self.assertIn("no current event loop", result_container[0])

    def test_thread_safe_callback_fix(self):
        """Test the thread-safe fix for async callbacks."""

        async def mock_async_callback(transcript):
            await asyncio.sleep(0.01)
            return f"Processed: {transcript}"

        def thread_safe_callback(transcript):
            """Thread-safe version that handles the event loop issue."""
            try:
                # Try to get current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, create task
                    asyncio.create_task(mock_async_callback(transcript))
                else:
                    # If loop is not running, run directly
                    loop.run_until_complete(mock_async_callback(transcript))
            except RuntimeError:
                # No event loop in current thread, create new one
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    new_loop.run_until_complete(mock_async_callback(transcript))
                    new_loop.close()
                except Exception as e:
                    return f"Error in new loop: {e}"
            except Exception as e:
                return f"Error: {e}"

        # Test in main thread
        result = thread_safe_callback("test")
        self.assertIsNone(result)  # Success returns None

        # Test in background thread
        result_container = []

        def run_in_thread():
            result = thread_safe_callback("test")
            result_container.append(result)

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()

        # Should succeed (no error)
        self.assertTrue(len(result_container) > 0)
        self.assertIsNone(result_container[0])  # Success returns None

    def test_alternative_thread_pool_approach(self):
        """Test alternative approach using thread pool executor."""

        async def mock_async_callback(transcript):
            await asyncio.sleep(0.01)
            return f"Processed: {transcript}"

        def thread_pool_callback(transcript):
            """Alternative approach using thread pool executor."""
            import concurrent.futures

            try:
                # Try normal approach first
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(mock_async_callback(transcript))
                else:
                    loop.run_until_complete(mock_async_callback(transcript))
            except RuntimeError:
                # Fallback to thread pool
                try:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        executor.submit(asyncio.run, mock_async_callback(transcript))
                        # Don't wait to avoid blocking, just submit
                except Exception as e:
                    return f"Thread pool error: {e}"
            except Exception as e:
                return f"Error: {e}"

        # Test in background thread
        result_container = []

        def run_in_thread():
            result = thread_pool_callback("test")
            result_container.append(result)

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()

        # Should succeed (no error)
        self.assertTrue(len(result_container) > 0)
        self.assertIsNone(result_container[0])  # Success returns None

    def test_simplified_sync_fallback(self):
        """Test simplified approach with sync fallback."""

        async def mock_async_callback(transcript):
            await asyncio.sleep(0.01)
            return f"Processed: {transcript}"

        def sync_fallback_callback(transcript):
            """Simplified approach with sync fallback."""
            try:
                # Try async approach
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(mock_async_callback(transcript))
                else:
                    loop.run_until_complete(mock_async_callback(transcript))
            except RuntimeError:
                # Fallback: run in new event loop
                asyncio.run(mock_async_callback(transcript))
            except Exception as e:
                return f"Error: {e}"

        # Test in background thread
        result_container = []

        def run_in_thread():
            result = sync_fallback_callback("test")
            result_container.append(result)

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()

        # Should succeed (no error)
        self.assertTrue(len(result_container) > 0)
        self.assertIsNone(result_container[0])  # Success returns None


if __name__ == "__main__":
    unittest.main()
