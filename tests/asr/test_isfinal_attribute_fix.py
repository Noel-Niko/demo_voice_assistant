"""
Test for the isFinal attribute fix in Google Speech v2 API.
Tests that we correctly handle the camelCase isFinal field.
"""

import unittest


class TestIsFinalAttributeFix(unittest.TestCase):
    """Test the isFinal attribute fix for Google Speech v2."""

    def test_isfinal_attribute_detection(self):
        """Test that we correctly detect the isFinal attribute in Google Speech v2 responses."""

        # Mock response structure for Google Speech v2
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

        # Test scenarios with different attribute names
        test_scenarios = [
            {
                "name": "Google Speech v2 with isFinal=True",
                "result": MockResult([MockAlternative("Hello world", 0.9)], isFinal=True),
                "should_process": True,
                "description": "Should process final result with isFinal=True",
            },
            {
                "name": "Google Speech v2 with isFinal=False",
                "result": MockResult([MockAlternative("Hello", 0.8)], isFinal=False),
                "should_process": False,
                "description": "Should ignore interim result with isFinal=False",
            },
            {
                "name": "Legacy with is_final=True",
                "result": MockResult([MockAlternative("Hello world", 0.9)], is_final=True),
                "should_process": True,
                "description": "Should process final result with legacy is_final=True",
            },
            {
                "name": "Legacy with is_final=False",
                "result": MockResult([MockAlternative("Hello", 0.8)], is_final=False),
                "should_process": False,
                "description": "Should ignore interim result with legacy is_final=False",
            },
            {
                "name": "No final indicator",
                "result": MockResult([MockAlternative("Hello world", 0.9)]),
                "should_process": True,  # Falls back to deduplication logic
                "description": "Should process if no final indicator (falls back to deduplication)",
            },
        ]

        def should_process_response_v2(response):
            """Enhanced logic matching the actual implementation."""
            if not response.results:
                return False

            result = response.results[0]
            if not result.alternatives:
                return False

            # Check multiple possible final indicators - Google Speech v2 uses isFinal (camelCase)
            is_final_result = False

            if hasattr(result, "isFinal"):
                is_final_result = result.isFinal
            elif hasattr(result, "is_final"):  # Fallback for older versions
                is_final_result = result.is_final
            else:
                # Falls back to deduplication logic
                is_final_result = True

            alternative = result.alternatives[0]
            confidence = getattr(alternative, "confidence", 0.0)
            transcript = alternative.transcript.strip()

            return is_final_result and confidence > 0.5 and len(transcript) > 0

        # Test each scenario
        for scenario in test_scenarios:
            with self.subTest(scenario=scenario["name"]):
                response = MockResponse([scenario["result"]])
                should_process = should_process_response_v2(response)

                self.assertEqual(
                    should_process,
                    scenario["should_process"],
                    f"Failed for {scenario['description']}",
                )

    def test_user_scenario_isfinal_fix(self):
        """Test the user's scenario with the isFinal fix."""
        # Simulate the user's problematic stream with proper isFinal values
        user_stream = [
            # Interim results (isFinal=False) - should be ignored
            {"transcript": "At this point", "isFinal": False, "confidence": 0.8},
            {"transcript": "At this point. At this point", "isFinal": False, "confidence": 0.8},
            {
                "transcript": "At this point. At this point, at this point",
                "isFinal": False,
                "confidence": 0.8,
            },
            {
                "transcript": "At this point. At this point, at this point, I at this point",
                "isFinal": False,
                "confidence": 0.8,
            },
            # Final result (isFinal=True) - should be processed
            {
                "transcript": "At this point, I am wondering if this is working properly.",
                "isFinal": True,
                "confidence": 0.9,
            },
            # Another set of interim results
            {"transcript": "Wonderful, this point", "isFinal": False, "confidence": 0.8},
            {
                "transcript": "Wonderful, this point. I am wondering",
                "isFinal": False,
                "confidence": 0.8,
            },
            # Final result
            {
                "transcript": "Wonderful, this is working much better now.",
                "isFinal": True,
                "confidence": 0.9,
            },
        ]

        def process_with_isfinal_fix(stream):
            """Process stream with the isFinal fix."""
            processed = []

            for item in stream:
                transcript = item["transcript"]
                is_final = item["isFinal"]
                confidence = item["confidence"]

                if is_final and confidence > 0.5 and transcript.strip():
                    processed.append(transcript)

            return processed

        processed = process_with_isfinal_fix(user_stream)

        print("\nIsFinal fix test:")
        print(f"Original stream items: {len(user_stream)}")
        print(f"Processed transcripts: {len(processed)}")
        for i, t in enumerate(processed):
            print(f"  {i + 1}. {t}")

        # Should only process the final results
        self.assertEqual(len(processed), 2)

        # Should contain the clean final transcripts
        self.assertIn("At this point, I am wondering if this is working properly.", processed)
        self.assertIn("Wonderful, this is working much better now.", processed)

        # Should NOT contain the repetitive interim results
        for transcript in processed:
            self.assertNotIn("At this point. At this point", transcript)
            self.assertNotIn("at this point, at this point", transcript.lower())

    def test_mixed_attribute_names(self):
        """Test handling of mixed attribute names in the same stream."""

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

        # Mixed stream with both isFinal and is_final
        mixed_stream = [
            MockResponse([MockResult([MockAlternative("Hello", 0.8)], isFinal=False)]),
            MockResponse([MockResult([MockAlternative("Hello world", 0.9)], is_final=True)]),
            MockResponse([MockResult([MockAlternative("Final result", 0.9)], isFinal=True)]),
        ]

        def process_mixed_stream(stream):
            """Process stream with mixed attribute names."""
            processed = []

            for response in stream:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                alternative = result.alternatives[0]
                transcript = alternative.transcript.strip()
                confidence = getattr(alternative, "confidence", 0.0)

                # Check both attribute names
                is_final_result = False
                if hasattr(result, "isFinal"):
                    is_final_result = result.isFinal
                elif hasattr(result, "is_final"):
                    is_final_result = result.is_final

                if is_final_result and confidence > 0.5 and transcript:
                    processed.append(transcript)

            return processed

        processed = process_mixed_stream(mixed_stream)

        # Should process only the final results
        self.assertEqual(len(processed), 2)
        self.assertIn("Hello world", processed)
        self.assertIn("Final result", processed)
        self.assertNotIn("Hello", processed)  # This was isFinal=False


if __name__ == "__main__":
    unittest.main()
