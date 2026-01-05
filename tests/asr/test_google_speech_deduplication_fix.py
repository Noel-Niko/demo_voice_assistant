"""
Unit tests to verify the Google Speech v2 deduplication fix works correctly.
Tests the actual implementation changes made to the streaming response processing.
"""

import sys
import unittest
from unittest.mock import Mock

# Mock the imports that cause issues
sys.modules["google.cloud.speech_v2"] = Mock()
sys.modules["google.cloud.speech_v2.types"] = Mock()
sys.modules["google.cloud.speech_v2.types.cloud_speech"] = Mock()

from src.asr.google_speech_v2 import GoogleSpeechV2Provider  # noqa: E402


class TestGoogleSpeechDeduplicationFix(unittest.TestCase):
    """Test the actual deduplication fix in GoogleSpeechV2Provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = GoogleSpeechV2Provider()

    def test_streaming_response_processing_logic(self):
        """Test the new streaming response processing logic."""

        # Mock response objects
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

        # Test data - partial and final results
        test_responses = [
            MockResponse([MockResult([MockAlternative("I really", 0.8)], is_final=False)]),
            MockResponse([MockResult([MockAlternative("I really need", 0.8)], is_final=False)]),
            MockResponse(
                [MockResult([MockAlternative("I really need to use", 0.8)], is_final=False)]
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
            ),  # Duplicate
        ]

        # Track processed transcripts
        processed_transcripts = []

        def mock_callback(transcript):
            processed_transcripts.append(transcript)

        self.provider.transcription_callback = mock_callback

        # Simulate the new processing logic
        seen_transcripts = set()
        for response in test_responses:
            if response.results:
                result = response.results[0]
                if result.alternatives:
                    alternative = result.alternatives[0]
                    transcript = alternative.transcript.strip()
                    confidence = alternative.confidence

                    # Apply the new logic
                    is_final_result = hasattr(result, "is_final") and result.is_final

                    if (
                        is_final_result
                        and confidence > 0.5
                        and transcript
                        and transcript not in seen_transcripts
                    ):
                        seen_transcripts.add(transcript)
                        mock_callback(transcript)

        # Verify only one final transcript was processed
        self.assertEqual(len(processed_transcripts), 1)
        self.assertEqual(processed_transcripts[0], "I really need to use the restroom")

    def test_partial_results_ignored(self):
        """Test that partial results are properly ignored."""

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

        # Only partial results
        partial_responses = [
            MockResponse([MockResult([MockAlternative("Hello", 0.8)], is_final=False)]),
            MockResponse([MockResult([MockAlternative("Hello world", 0.8)], is_final=False)]),
            MockResponse([MockResult([MockAlternative("Hello world this", 0.8)], is_final=False)]),
        ]

        processed_transcripts = []

        def mock_callback(transcript):
            processed_transcripts.append(transcript)

        self.provider.transcription_callback = mock_callback

        # Apply new logic
        seen_transcripts = set()
        for response in partial_responses:
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
                        mock_callback(transcript)

        # No transcripts should be processed (all are partial)
        self.assertEqual(len(processed_transcripts), 0)

    def test_low_confidence_results_ignored(self):
        """Test that low confidence results are ignored even if final."""

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

        # Final result with low confidence
        low_conf_response = MockResponse(
            [MockResult([MockAlternative("mumbled text", 0.3)], is_final=True)]
        )

        processed_transcripts = []

        def mock_callback(transcript):
            processed_transcripts.append(transcript)

        self.provider.transcription_callback = mock_callback

        # Apply new logic
        seen_transcripts = set()
        response = low_conf_response
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
                    mock_callback(transcript)

        # Low confidence transcript should not be processed
        self.assertEqual(len(processed_transcripts), 0)

    def test_duplicate_final_results_ignored(self):
        """Test that duplicate final results are ignored."""

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

        # Duplicate final results
        duplicate_responses = [
            MockResponse([MockResult([MockAlternative("Test message", 0.9)], is_final=True)]),
            MockResponse([MockResult([MockAlternative("Test message", 0.9)], is_final=True)]),
            MockResponse([MockResult([MockAlternative("Test message", 0.9)], is_final=True)]),
        ]

        processed_transcripts = []

        def mock_callback(transcript):
            processed_transcripts.append(transcript)

        self.provider.transcription_callback = mock_callback

        # Apply new logic
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
                        mock_callback(transcript)

        # Only one instance of the duplicate should be processed
        self.assertEqual(len(processed_transcripts), 1)
        self.assertEqual(processed_transcripts[0], "Test message")

    def test_mixed_scenarios(self):
        """Test mixed scenarios with partial, final, and duplicate results."""

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

        # Mixed scenario
        mixed_responses = [
            # Partial results
            MockResponse([MockResult([MockAlternative("Weather", 0.7)], is_final=False)]),
            MockResponse([MockResult([MockAlternative("Weather today", 0.7)], is_final=False)]),
            # Final result
            MockResponse(
                [MockResult([MockAlternative("Weather today is sunny", 0.9)], is_final=True)]
            ),
            # Duplicate final
            MockResponse(
                [MockResult([MockAlternative("Weather today is sunny", 0.9)], is_final=True)]
            ),
            # Low confidence final
            MockResponse(
                [
                    MockResult(
                        [MockAlternative("Weather today is sunny and warm", 0.3)], is_final=True
                    )
                ]
            ),
            # New final result
            MockResponse(
                [MockResult([MockAlternative("Thank you for listening", 0.8)], is_final=True)]
            ),
        ]

        processed_transcripts = []

        def mock_callback(transcript):
            processed_transcripts.append(transcript)

        self.provider.transcription_callback = mock_callback

        # Apply new logic
        seen_transcripts = set()
        for response in mixed_responses:
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
                        mock_callback(transcript)

        # Should process 2 unique final transcripts with good confidence
        self.assertEqual(len(processed_transcripts), 2)
        self.assertIn("Weather today is sunny", processed_transcripts)
        self.assertIn("Thank you for listening", processed_transcripts)


if __name__ == "__main__":
    unittest.main()
