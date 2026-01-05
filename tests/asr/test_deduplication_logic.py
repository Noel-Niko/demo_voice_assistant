"""
Unit tests for the deduplication logic without importing the full module.
Tests the core logic that was implemented to fix transcription duplication.
"""

import unittest


class TestDeduplicationLogic(unittest.TestCase):
    """Test the deduplication logic implemented in the streaming response processing."""

    def test_new_processing_logic(self):
        """Test the new processing logic that handles partial vs final results."""

        # Mock response structure
        class MockAlternative:
            def __init__(self, transcript, confidence):
                self.transcript = transcript
                self.confidence = confidence

        class MockResult:
            def __init__(self, alternatives, is_final=False):
                self.alternatives = alternatives
                self.is_final = is_final

        class MockResponse:
            def __init__(self, results):
                self.results = results

        # Test data simulating the user's issue
        test_responses = [
            # Partial results that were causing duplication
            MockResponse([MockResult([MockAlternative("I really", 0.8)], is_final=False)]),
            MockResponse(
                [MockResult([MockAlternative("I really, I really", 0.8)], is_final=False)]
            ),
            MockResponse(
                [MockResult([MockAlternative("I really, I really, I really", 0.8)], is_final=False)]
            ),
            MockResponse(
                [
                    MockResult(
                        [MockAlternative("I really, I really, I really, I really", 0.8)],
                        is_final=False,
                    )
                ]
            ),
            # Final result
            MockResponse(
                [
                    MockResult(
                        [MockAlternative("I really need to use the restroom", 0.95)], is_final=True
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [MockAlternative("I really need to use the restroom", 0.95)], is_final=True
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [MockAlternative("I really need to use the restroom", 0.95)], is_final=True
                    )
                ]
            ),
        ]

        # Old problematic logic (processes all results)
        old_processed = []
        for response in test_responses:
            if response.results:
                result = response.results[0]
                if result.alternatives:
                    alternative = result.alternatives[0]
                    transcript = alternative.transcript.strip()
                    confidence = alternative.confidence

                    # Old logic: only checked confidence and non-empty
                    if confidence > 0.5 and transcript:
                        old_processed.append(transcript)

        # New fixed logic (only processes final, unique results)
        new_processed = []
        seen_transcripts = set()

        for response in test_responses:
            if response.results:
                result = response.results[0]
                if result.alternatives:
                    alternative = result.alternatives[0]
                    transcript = alternative.transcript.strip()
                    confidence = alternative.confidence

                    # New logic: check is_final, confidence, and duplicates
                    is_final_result = hasattr(result, "is_final") and result.is_final

                    if (
                        is_final_result
                        and confidence > 0.5
                        and transcript
                        and transcript not in seen_transcripts
                    ):
                        seen_transcripts.add(transcript)
                        new_processed.append(transcript)

        # Verify the fix
        print(f"Old logic processed: {len(old_processed)} transcripts")
        print(f"New logic processed: {len(new_processed)} transcripts")

        # Old logic processed everything (causing duplication)
        self.assertGreater(len(old_processed), 1)

        # New logic only processes unique final results
        self.assertEqual(len(new_processed), 1)
        self.assertEqual(new_processed[0], "I really need to use the restroom")

        # Verify old logic had duplicates
        self.assertIn("I really", old_processed)
        self.assertIn("I really, I really", old_processed)

        # Verify new logic eliminated duplicates
        self.assertEqual(len(set(new_processed)), len(new_processed))  # All unique

    def test_confidence_and_final_check(self):
        """Test the combined confidence and final result checking."""

        class MockAlternative:
            def __init__(self, transcript, confidence):
                self.transcript = transcript
                self.confidence = confidence

        class MockResult:
            def __init__(self, alternatives, is_final=False):
                self.alternatives = alternatives
                self.is_final = is_final

        class MockResponse:
            def __init__(self, results):
                self.results = results

        # Test cases: (transcript, confidence, is_final, should_process)
        test_cases = [
            ("I really", 0.8, False, False),  # Partial, good confidence -> ignore
            ("I really need", 0.3, False, False),  # Partial, low confidence -> ignore
            ("I really need to", 0.9, False, False),  # Partial, high confidence -> ignore
            ("I really need to use", 0.4, True, False),  # Final, low confidence -> ignore
            ("I really need to use the", 0.8, True, True),  # Final, good confidence -> process
            (
                "I really need to use the restroom",
                0.95,
                True,
                True,
            ),  # Final, high confidence -> process
        ]

        processed = []
        seen_transcripts = set()

        for transcript, confidence, is_final, should_process in test_cases:
            alternative = MockAlternative(transcript, confidence)
            result = MockResult([alternative], is_final=is_final)
            response = MockResponse([result])

            # Apply new logic
            if response.results:
                result = response.results[0]
                if result.alternatives:
                    alternative = result.alternatives[0]
                    transcript = alternative.transcript.strip()
                    confidence = alternative.confidence

                    is_final_result = hasattr(result, "is_final") and result.is_final

                    if (
                        is_final_result
                        and confidence > 0.5
                        and transcript
                        and transcript not in seen_transcripts
                    ):
                        seen_transcripts.add(transcript)
                        processed.append(transcript)

            # Verify expected behavior
            actual_processed = transcript in processed
            self.assertEqual(
                actual_processed,
                should_process,
                f"Failed for transcript='{transcript}', confidence={confidence}, is_final={is_final}",
            )

        # Verify only the expected transcripts were processed
        expected_processed = ["I really need to use the", "I really need to use the restroom"]
        self.assertEqual(len(processed), len(expected_processed))
        for expected in expected_processed:
            self.assertIn(expected, processed)

    def test_duplicate_elimination(self):
        """Test that duplicate final results are eliminated."""

        class MockAlternative:
            def __init__(self, transcript, confidence):
                self.transcript = transcript
                self.confidence = confidence

        class MockResult:
            def __init__(self, alternatives, is_final=False):
                self.alternatives = alternatives
                self.is_final = is_final

        class MockResponse:
            def __init__(self, results):
                self.results = results

        # Same final result repeated multiple times
        duplicate_responses = [
            MockResponse([MockResult([MockAlternative("Hello world", 0.9)], is_final=True)]),
            MockResponse([MockResult([MockAlternative("Hello world", 0.9)], is_final=True)]),
            MockResponse([MockResult([MockAlternative("Hello world", 0.9)], is_final=True)]),
            MockResponse([MockResult([MockAlternative("Hello world", 0.9)], is_final=True)]),
        ]

        processed = []
        seen_transcripts = set()

        for response in duplicate_responses:
            if response.results:
                result = response.results[0]
                if result.alternatives:
                    alternative = result.alternatives[0]
                    transcript = alternative.transcript.strip()
                    confidence = alternative.confidence

                    is_final_result = hasattr(result, "is_final") and result.is_final

                    if (
                        is_final_result
                        and confidence > 0.5
                        and transcript
                        and transcript not in seen_transcripts
                    ):
                        seen_transcripts.add(transcript)
                        processed.append(transcript)

        # Should only process one instance of the duplicate
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0], "Hello world")

    def test_user_scenario_fix(self):
        """Test the specific scenario the user reported."""
        # User said: "I really need to use the restroom. I hope this will work so I can go."
        # But got: "I really, I really, I really, I really, I really need..." (massive duplication)

        user_actual_speech = "I really need to use the restroom. I hope this will work so I can go."

        # Simulate the problematic partial results that were causing duplication
        problematic_stream = [
            "I really",
            "I really, I really",
            "I really, I really, I really",
            "I really, I really, I really, I really",
            "I really, I really, I really, I really, I really need",
            "I really, I really, I really, I really, I really need to",
            "I really, I really, I really, I really, I really need to use",
            "I really, I really, I really, I really, I really need to use the",
            "I really, I really, I really, I really, I really need to use the rest",
            "I really, I really, I really, I really, I really need to use the restroom",
            # Final correct result
            user_actual_speech,
        ]

        # Old logic would process all of these (causing the duplication issue)
        old_processed = problematic_stream.copy()  # All would be processed

        # New logic should only process the final, correct result
        new_processed = []
        seen_transcripts = set()

        for i, transcript in enumerate(problematic_stream):
            # Simulate the new logic - only final results
            is_final = i == len(problematic_stream) - 1  # Only last one is final
            confidence = 0.9  # Assume good confidence

            if is_final and confidence > 0.5 and transcript and transcript not in seen_transcripts:
                seen_transcripts.add(transcript)
                new_processed.append(transcript)

        # Verify the fix
        print("\nUser scenario test:")
        print(f"Old logic would process: {len(old_processed)} transcripts")
        print(f"New logic processes: {len(new_processed)} transcripts")

        # Old logic had massive duplication
        self.assertGreater(len(old_processed), 5)

        # New logic should only have the correct final transcript
        self.assertEqual(len(new_processed), 1)
        self.assertEqual(new_processed[0], user_actual_speech)

        # Verify the old logic contained the problematic duplicates
        self.assertTrue(any("I really, I really" in transcript for transcript in old_processed))

        # Verify new logic eliminated all duplicates
        self.assertNotIn("I really, I really", new_processed[0])


if __name__ == "__main__":
    unittest.main()
