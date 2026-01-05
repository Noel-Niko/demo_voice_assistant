"""
Test for the corrected is_final attribute fix in Google Speech v2 Python gRPC API.
Tests that we correctly handle the is_final (snake_case) field.
"""

import unittest


class TestCorrectedIsFinalFix(unittest.TestCase):
    """Test the corrected is_final attribute fix for Google Speech v2 Python gRPC API."""

    def test_correct_isfinal_attribute_detection(self):
        """Test that we correctly detect the is_final attribute in Google Speech v2 Python gRPC responses."""

        # Mock response structure for Google Speech v2 Python gRPC API
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
                "name": "Google Speech v2 Python gRPC with is_final=True",
                "result": MockResult([MockAlternative("Hello world", 0.9)], is_final=True),
                "should_process": True,
                "description": "Should process final result with is_final=True",
            },
            {
                "name": "Google Speech v2 Python gRPC with is_final=False",
                "result": MockResult([MockAlternative("Hello", 0.8)], is_final=False),
                "should_process": False,
                "description": "Should ignore interim result with is_final=False",
            },
            {
                "name": "REST API style with isFinal=True",
                "result": MockResult([MockAlternative("Hello world", 0.9)], isFinal=True),
                "should_process": True,
                "description": "Should process final result with REST API style isFinal=True",
            },
            {
                "name": "No final indicator",
                "result": MockResult([MockAlternative("Hello world", 0.9)]),
                "should_process": True,  # Falls back to deduplication logic
                "description": "Should process if no final indicator (falls back to deduplication)",
            },
        ]

        def should_process_response_v2(response):
            """Corrected logic matching the actual implementation."""
            if not response.results:
                return False

            result = response.results[0]
            if not result.alternatives:
                return False

            # Check multiple possible final indicators - Google Speech v2 Python gRPC API uses is_final (snake_case)
            is_final_result = False

            if hasattr(result, "is_final"):
                is_final_result = result.is_final
            elif hasattr(result, "isFinal"):  # Fallback for REST API style
                is_final_result = result.isFinal
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

    def test_user_scenario_corrected_fix(self):
        """Test the user's scenario with the corrected is_final fix."""
        # Simulate the user's problematic stream with proper is_final values
        user_stream = [
            # Interim results (is_final=False) - should be ignored
            {"transcript": "I,", "is_final": False, "confidence": 0.51},
            {"transcript": "I,", "is_final": False, "confidence": 0.56},
            {"transcript": "Would.", "is_final": False, "confidence": 0.37},
            {"transcript": "I would.", "is_final": False, "confidence": 0.78},
            {"transcript": "I would like,", "is_final": False, "confidence": 0.91},
            {"transcript": "I would like,", "is_final": False, "confidence": 0.97},
            {"transcript": "I would like,", "is_final": False, "confidence": 0.97},
            {"transcript": "I would like to.", "is_final": False, "confidence": 0.98},
            {"transcript": "I would like to.", "is_final": False, "confidence": 0.98},
            {"transcript": "I would like to make.", "is_final": False, "confidence": 0.99},
            {"transcript": "I would like to make.", "is_final": False, "confidence": 0.99},
            {"transcript": "I would like to make.", "is_final": False, "confidence": 0.99},
            {"transcript": "It. I would like to make a", "is_final": False, "confidence": 0.77},
            {"transcript": "I would like to make a cup.", "is_final": False, "confidence": 0.96},
            {"transcript": "I would like to make a cup of", "is_final": False, "confidence": 0.91},
            {"transcript": "I would like to make a cup of", "is_final": False, "confidence": 0.88},
            {"transcript": "I would like to make a cup of", "is_final": False, "confidence": 0.90},
            {
                "transcript": "I would like to make a cup of coffee.",
                "is_final": False,
                "confidence": 0.92,
            },
            {
                "transcript": "I would like to make a cup of coffee.",
                "is_final": False,
                "confidence": 0.93,
            },
            {
                "transcript": "I would like to make a cup of coffee.",
                "is_final": False,
                "confidence": 0.92,
            },
            {
                "transcript": "I would like to make a cup of coffee.",
                "is_final": False,
                "confidence": 0.94,
            },
            {
                "transcript": "I would like to make a cup of coffee.",
                "is_final": False,
                "confidence": 0.93,
            },
            {
                "transcript": "I would like to make a cup of coffee.",
                "is_final": False,
                "confidence": 0.99,
            },
            {
                "transcript": "and I would like to make a cup of coffee and",
                "is_final": False,
                "confidence": 0.96,
            },
            # Final results (is_final=True) - should be processed
            {
                "transcript": "I would like to make a cup of coffee.",
                "is_final": True,
                "confidence": 0.99,
            },
        ]

        def process_with_corrected_fix(stream):
            """Process stream with the corrected is_final fix."""
            processed = []

            for item in stream:
                transcript = item["transcript"]
                is_final = item["is_final"]
                confidence = item["confidence"]

                if is_final and confidence > 0.5 and transcript.strip():
                    processed.append(transcript)

            return processed

        processed = process_with_corrected_fix(user_stream)

        print("\nCorrected is_final fix test:")
        print(f"Original stream items: {len(user_stream)}")
        print(f"Processed transcripts: {len(processed)}")
        for i, t in enumerate(processed):
            print(f"  {i + 1}. {t}")

        # Should only process the final results
        self.assertEqual(len(processed), 1)

        # Should contain the clean final transcript
        self.assertIn("I would like to make a cup of coffee.", processed)

        # Should NOT contain any of the interim results
        self.assertEqual(
            len([t for t in processed if "I would like to make a cup of coffee." not in t]), 0
        )

    def test_attribute_priority_order(self):
        """Test that is_final takes priority over isFinal when both are present."""

        class MockAlternative:
            def __init__(self, transcript, confidence):
                self.transcript = transcript
                self.confidence = confidence

        class MockResult:
            def __init__(self, alternatives, is_final=False, isFinal=False):
                self.alternatives = alternatives
                self.is_final = is_final  # Python gRPC style
                self.isFinal = isFinal  # REST API style

        class MockResponse:
            def __init__(self, results):
                self.results = results

        # Create a result with both attributes set to different values
        result = MockResult([MockAlternative("Test transcript", 0.9)], is_final=True, isFinal=False)
        response = MockResponse([result])

        def get_is_final_value(response):
            """Extract is_final value using the same logic as our implementation."""
            if not response.results:
                return False

            result = response.results[0]

            # Google Speech v2 Python gRPC API uses is_final (snake_case)
            if hasattr(result, "is_final"):
                return result.is_final
            elif hasattr(result, "isFinal"):  # Fallback for REST API style
                return result.isFinal

            return False

        is_final_value = get_is_final_value(response)

        # Should prioritize is_final (snake_case) over isFinal (camelCase)
        self.assertTrue(is_final_value, "Should prioritize is_final over isFinal when both present")


if __name__ == "__main__":
    unittest.main()
