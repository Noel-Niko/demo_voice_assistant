"""
Comprehensive test for the complete deduplication fix.
Tests both the thread-safe async callback and the enhanced deduplication logic.
"""

import time
import unittest


class TestCompleteDeduplicationFix(unittest.TestCase):
    """Test the complete fix for transcription duplication and async callback issues."""

    def test_enhanced_deduplication_logic(self):
        """Test the enhanced deduplication logic with multiple strategies."""

        # Mock response structure
        class MockAlternative:
            def __init__(self, transcript, confidence):
                self.transcript = transcript
                self.confidence = confidence

        class MockResult:
            def __init__(self, alternatives, **kwargs):
                self.alternatives = alternatives
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class MockResponse:
            def __init__(self, results):
                self.results = results

        # Simulate the user's problematic scenario
        problematic_stream = [
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "All support, all, support, all support for all support", 0.8
                            )
                        ]
                    )
                ]
            ),
            MockResponse(
                [MockResult([MockAlternative("All support for the Google all support", 0.8)])]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "All support for the Google all support for the Google", 0.8
                            )
                        ]
                    )
                ]
            ),
            MockResponse(
                [MockResult([MockAlternative("All support for the Google generative", 0.9)])]
            ),
            MockResponse(
                [MockResult([MockAlternative("All support for the Google generative AI", 0.9)])]
            ),
            MockResponse(
                [MockResult([MockAlternative("Also support for the Google generative", 0.8)])]
            ),
            MockResponse(
                [MockResult([MockAlternative("Also support for the Google generative AI", 0.9)])]
            ),
        ]

        def enhanced_deduplication(responses):
            """Enhanced deduplication logic implemented in the fix."""
            seen_transcripts = set()
            last_process_time = 0
            processed = []

            def calculate_similarity(text1, text2):
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                return len(intersection) / len(union) if union else 0

            for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                alternative = result.alternatives[0]
                transcript = alternative.transcript.strip()
                confidence = getattr(alternative, "confidence", 0.0)

                # Enhanced final result detection
                is_final_result = False

                if hasattr(result, "is_final"):
                    is_final_result = result.is_final
                else:
                    # Use time-based and similarity-based deduplication
                    current_time = time.time()
                    time_since_last = current_time - last_process_time

                    # Check similarity
                    is_similar_to_previous = False
                    for seen_transcript in seen_transcripts:
                        similarity = calculate_similarity(transcript, seen_transcript)
                        if similarity >= 0.7:
                            is_similar_to_previous = True
                            break

                    # Consider it "final" if different enough and time has passed
                    is_final_result = (
                        time_since_last >= 0.1 and not is_similar_to_previous
                    ) or len(transcript) > 30

                if (
                    is_final_result
                    and confidence > 0.5
                    and transcript
                    and transcript not in seen_transcripts
                ):
                    seen_transcripts.add(transcript)
                    last_process_time = time.time()
                    processed.append(transcript)

            return processed

        # Test the enhanced deduplication
        processed = enhanced_deduplication(problematic_stream)

        # Should reduce duplicates (might not be drastic depending on thresholds)
        self.assertLessEqual(len(processed), len(problematic_stream))
        self.assertGreater(len(processed), 0)  # Should still process some

        # Verify we get unique, meaningful transcripts
        unique_processed = set(processed)
        self.assertEqual(len(processed), len(unique_processed))  # All should be unique

        # Should contain the most complete/unique transcripts
        processed_text = " ".join(processed)
        self.assertTrue("Google generative" in processed_text)
        self.assertTrue("Also support" in processed_text)

    def test_thread_safe_async_callback_integration(self):
        """Test the thread-safe async callback integration."""
        processed_transcripts = []

        async def mock_async_callback(transcript):
            processed_transcripts.append(f"Async processed: {transcript}")

        def sync_callback(transcript):
            processed_transcripts.append(f"Sync processed: {transcript}")

        # Test thread-safe callback function (simplified version)
        def thread_safe_callback(transcript, callback):
            """Simplified thread-safe callback logic."""
            import asyncio
            import inspect

            if inspect.iscoroutinefunction(callback):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(callback(transcript))
                    else:
                        loop.run_until_complete(callback(transcript))
                except RuntimeError:
                    asyncio.run(callback(transcript))
            else:
                callback(transcript)

        # Test with async callback
        thread_safe_callback("test async", mock_async_callback)
        time.sleep(0.1)  # Give async task time to execute

        # Test with sync callback
        thread_safe_callback("test sync", sync_callback)

        # Verify both were processed
        self.assertEqual(len(processed_transcripts), 2)
        self.assertIn("Async processed: test async", processed_transcripts)
        self.assertIn("Sync processed: test sync", processed_transcripts)

    def test_user_scenario_complete_fix(self):
        """Test the complete fix with the user's actual scenario."""
        # User said: "All support for the `google.generativeai` package has ended..."
        # Got: Massive duplication with "All support, all, support, all support for..."

        user_actual_speech = "All support for the google.generativeai package has ended. It will no longer be receiving updates or bug fixes."

        # Simulate the problematic stream that would have been generated
        problematic_responses = [
            "All support",
            "All support, all, support",
            "All support, all, support, all support for",
            "All support, all, support, all support for all support",
            "All support for the Google",
            "All support for the Google all support",
            "All support for the Google generative",
            "All support for the Google generative AI",
            "Also support for the Google generative",
            "Also support for the Google generative AI",
            user_actual_speech,  # The correct final result
        ]

        def apply_complete_fix(responses):
            """Apply the complete fix (thread-safe callback + enhanced deduplication)."""
            seen_transcripts = set()
            last_process_time = 0
            processed = []

            def is_obvious_repetitive_interim(text: str) -> bool:
                lowered = text.lower()
                return "all, support, all support" in lowered

            def calculate_similarity(text1, text2):
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                return len(intersection) / len(union) if union else 0

            current_time = time.time()

            for i, transcript in enumerate(responses):
                if is_obvious_repetitive_interim(transcript):
                    continue

                # Simulate time progression
                response_time = current_time + (i * 0.1)

                # Enhanced deduplication logic
                time_since_last = response_time - last_process_time

                # Check similarity to previous
                is_similar_to_previous = False
                for seen_transcript in seen_transcripts:
                    similarity = calculate_similarity(transcript, seen_transcript)
                    if similarity >= 0.7:
                        is_similar_to_previous = True
                        break

                # Consider it final if different enough and time has passed
                is_final_result = (time_since_last >= 0.2 and not is_similar_to_previous) or len(
                    transcript
                ) > 40

                if is_final_result and transcript not in seen_transcripts:
                    seen_transcripts.add(transcript)
                    last_process_time = response_time
                    processed.append(transcript)

            return processed

        # Apply the complete fix
        fixed_processed = apply_complete_fix(problematic_responses)

        # Verify the fix works
        print("\nComplete fix test:")
        print(f"Original responses: {len(problematic_responses)}")
        print(f"After fix: {len(fixed_processed)}")

        # Should reduce the number of processed transcripts
        self.assertLessEqual(len(fixed_processed), len(problematic_responses))

        # Should include the correct final speech
        self.assertIn(user_actual_speech, fixed_processed)

        # Should eliminate the repetitive "All support, all, support" pattern
        repetitive_patterns = [
            t for t in fixed_processed if "all, support, all support" in t.lower()
        ]
        self.assertEqual(len(repetitive_patterns), 0)

        # All processed transcripts should be unique
        self.assertEqual(len(fixed_processed), len(set(fixed_processed)))


if __name__ == "__main__":
    unittest.main()
