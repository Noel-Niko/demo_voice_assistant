"""
Unit tests to understand Google Speech v2 response structure.
Tests the is_final attribute and other response properties.
"""

import unittest


class TestGoogleSpeechResponseStructure(unittest.TestCase):
    """Test Google Speech v2 response structure to fix duplication issue."""

    def test_response_structure_analysis(self):
        """Analyze the expected Google Speech v2 response structure."""
        # Based on the logs, we can see that transcripts are still being duplicated
        # This suggests the is_final check is not working as expected

        # Let's examine what the actual response structure might be
        class MockAlternative:
            def __init__(self, transcript, confidence):
                self.transcript = transcript
                self.confidence = confidence

        class MockResult:
            def __init__(self, alternatives, **kwargs):
                self.alternatives = alternatives
                # Add various possible attributes that Google Speech v2 might use
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class MockResponse:
            def __init__(self, results):
                self.results = results

        # Test different possible final result indicators
        test_scenarios = [
            # Scenario 1: is_final attribute
            {
                "result": MockResult([MockAlternative("Hello world", 0.9)], is_final=True),
                "should_process": True,
                "description": "Standard is_final=True",
            },
            {
                "result": MockResult([MockAlternative("Hello", 0.8)], is_final=False),
                "should_process": False,
                "description": "Standard is_final=False",
            },
            # Scenario 2: result_type attribute (possible in v2)
            {
                "result": MockResult([MockAlternative("Hello world", 0.9)], result_type="FINAL"),
                "should_process": True,
                "description": "result_type=FINAL",
            },
            {
                "result": MockResult([MockAlternative("Hello", 0.8)], result_type="PARTIAL"),
                "should_process": False,
                "description": "result_type=PARTIAL",
            },
            # Scenario 3: speech_event_type attribute
            {
                "result": MockResult(
                    [MockAlternative("Hello world", 0.9)], speech_event_type="SPEECH_ACTIVITY_END"
                ),
                "should_process": True,
                "description": "speech_event_type=SPEECH_ACTIVITY_END",
            },
            {
                "result": MockResult(
                    [MockAlternative("Hello", 0.8)], speech_event_type="SPEECH_ACTIVITY_START"
                ),
                "should_process": False,
                "description": "speech_event_type=SPEECH_ACTIVITY_START",
            },
            # Scenario 4: No final indicator (all results considered final)
            {
                "result": MockResult([MockAlternative("Hello world", 0.9)]),
                "should_process": True,
                "description": "No final indicator - treat as final",
            },
        ]

        def should_process_response_v2(response):
            """Enhanced logic to handle various Google Speech v2 response formats."""
            if not response.results:
                return False

            result = response.results[0]
            if not result.alternatives:
                return False

            # Check multiple possible final indicators
            is_final = False

            # Standard is_final attribute
            if hasattr(result, "is_final"):
                is_final = result.is_final

            # result_type attribute
            elif hasattr(result, "result_type"):
                is_final = result.result_type.upper() == "FINAL"

            # speech_event_type attribute
            elif hasattr(result, "speech_event_type"):
                is_final = result.speech_event_type.upper() in [
                    "SPEECH_ACTIVITY_END",
                    "RECOGNITION_COMPLETE",
                ]

            # If no final indicator, assume it's final (v2 might work this way)
            else:
                is_final = True

            alternative = result.alternatives[0]
            confidence = getattr(alternative, "confidence", 0.0)
            transcript = alternative.transcript.strip()

            return is_final and confidence > 0.5 and len(transcript) > 0

        # Test each scenario
        for scenario in test_scenarios:
            response = MockResponse([scenario["result"]])
            should_process = should_process_response_v2(response)

            self.assertEqual(
                should_process, scenario["should_process"], f"Failed for {scenario['description']}"
            )

    def test_deduplication_with_time_based_approach(self):
        """Test time-based deduplication as an alternative to is_final."""
        import time

        class MockAlternative:
            def __init__(self, transcript, confidence):
                self.transcript = transcript
                self.confidence = confidence

        class MockResult:
            def __init__(self, alternatives, timestamp=None):
                self.alternatives = alternatives
                self.timestamp = timestamp or time.time()

        class MockResponse:
            def __init__(self, results):
                self.results = results

        # Simulate rapid partial results followed by final result
        base_time = time.time()
        test_responses = [
            MockResponse([MockResult([MockAlternative("I really", 0.8)], base_time)]),
            MockResponse([MockResult([MockAlternative("I really need", 0.8)], base_time + 0.1)]),
            MockResponse(
                [MockResult([MockAlternative("I really need to use", 0.8)], base_time + 0.2)]
            ),
            MockResponse(
                [
                    MockResult(
                        [MockAlternative("I really need to use the restroom", 0.9)], base_time + 0.3
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [MockAlternative("I really need to use the restroom", 0.9)], base_time + 0.4
                    )
                ]
            ),
        ]

        def time_based_deduplication(responses, min_gap=0.5):
            """Deduplicate based on time gaps between similar transcripts."""
            processed = []
            last_processed_time = 0
            seen_transcripts = set()

            for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                alternative = result.alternatives[0]
                transcript = alternative.transcript.strip()
                confidence = getattr(alternative, "confidence", 0.0)

                # Only process if confidence is good and not seen before
                if confidence > 0.5 and transcript not in seen_transcripts:
                    # Check if enough time has passed since last processing
                    current_time = result.timestamp
                    if current_time - last_processed_time >= min_gap:
                        processed.append(transcript)
                        seen_transcripts.add(transcript)
                        last_processed_time = current_time

            return processed

        # Test time-based deduplication
        processed = time_based_deduplication(test_responses)

        # Should process the first transcript that meets the time gap
        self.assertGreaterEqual(len(processed), 1)
        # The first one should be processed since there's no previous transcript
        self.assertEqual(processed[0], "I really")

    def test_similarity_based_deduplication(self):
        """Test similarity-based deduplication to handle near-duplicates."""

        class MockAlternative:
            def __init__(self, transcript, confidence):
                self.transcript = transcript
                self.confidence = confidence

        class MockResult:
            def __init__(self, alternatives):
                self.alternatives = alternatives

        class MockResponse:
            def __init__(self, results):
                self.results = results

        # Simulate the user's actual problematic output
        problematic_responses = [
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
                [MockResult([MockAlternative("All support for the Google generative", 0.8)])]
            ),
            MockResponse(
                [MockResult([MockAlternative("All support for the Google generative AI", 0.9)])]
            ),
            MockResponse(
                [MockResult([MockAlternative("Also support for the Google generative", 0.8)])]
            ),
        ]

        def similarity_based_deduplication(responses, similarity_threshold=0.7):
            """Deduplicate based on transcript similarity."""
            processed = []

            def calculate_similarity(text1, text2):
                """Simple similarity calculation based on word overlap."""
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                return len(intersection) / len(union) if union else 0

            for response in problematic_responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                alternative = result.alternatives[0]
                transcript = alternative.transcript.strip()
                confidence = getattr(alternative, "confidence", 0.0)

                if confidence > 0.5:
                    # Check if similar to already processed
                    is_similar = False
                    for processed_transcript in processed:
                        similarity = calculate_similarity(transcript, processed_transcript)
                        if similarity >= similarity_threshold:
                            is_similar = True
                            break

                    if not is_similar:
                        processed.append(transcript)

            return processed

        # Test similarity-based deduplication
        processed = similarity_based_deduplication(problematic_responses)

        # Should reduce duplicates significantly
        self.assertLess(len(processed), len(problematic_responses))
        # Should keep unique transcripts
        self.assertTrue(len(processed) >= 1)


if __name__ == "__main__":
    unittest.main()
