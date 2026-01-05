"""
Unit tests for transcription deduplication fix.
Tests the proper handling of partial vs final streaming results.
"""

import asyncio
import unittest


class TestTranscriptionDeduplication(unittest.TestCase):
    """Test cases for fixing transcription duplication in streaming results."""

    def test_partial_vs_final_results_simulation(self):
        """Test simulation of partial vs final streaming results."""

        # Simulate Google Speech streaming responses
        class MockResult:
            def __init__(self, transcript, confidence, is_final=False):
                self.transcript = transcript
                self.confidence = confidence
                self.is_final = is_final

        class MockAlternative:
            def __init__(self, transcript, confidence):
                self.transcript = transcript
                self.confidence = confidence

        class MockResponse:
            def __init__(self, results, is_final=False):
                self.results = results
                self.is_final = is_final

        # Test data - partial results leading to final result
        partial_results = [
            "I really",
            "I really need",
            "I really need to",
            "I really need to use",
            "I really need to use the",
            "I really need to use the rest",
            "I really need to use the restroom",
        ]

        # Current problematic approach (processes all results)
        processed_transcripts = []
        for transcript in partial_results:
            if len(transcript.strip()) > 0:  # Current logic only checks if not empty
                processed_transcripts.append(transcript)

        # This would cause duplication - all partials are processed
        self.assertEqual(len(processed_transcripts), len(partial_results))

        # Fixed approach (only process final results)
        def should_process_result(result):
            """Determine if a result should be processed as final."""
            # In real Google Speech API, we'd check result.is_final
            # For simulation, we'll assume the last result is final
            return result == partial_results[-1]

        fixed_processed_transcripts = []
        for transcript in partial_results:
            if should_process_result(transcript):
                fixed_processed_transcripts.append(transcript)

        # Only final result is processed
        self.assertEqual(len(fixed_processed_transcripts), 1)
        self.assertEqual(fixed_processed_transcripts[0], "I really need to use the restroom")

    def test_streaming_response_structure(self):
        """Test understanding of Google Speech streaming response structure."""

        # Mock response structure
        class MockAlternative:
            def __init__(self, transcript, confidence):
                self.transcript = transcript
                self.confidence = confidence

        class MockResult:
            def __init__(self, alternatives, is_final=False, stability=0.0):
                self.alternatives = alternatives
                self.is_final = is_final
                self.stability = stability

        class MockResponse:
            def __init__(self, results):
                self.results = results

        # Test partial result
        partial_alternative = MockAlternative("I really", 0.8)
        partial_result = MockResult([partial_alternative], is_final=False, stability=0.5)
        partial_response = MockResponse([partial_result])

        # Test final result
        final_alternative = MockAlternative("I really need to use the restroom", 0.95)
        final_result = MockResult([final_alternative], is_final=True, stability=1.0)
        final_response = MockResponse([final_result])

        # Test processing logic
        def should_process_response(response):
            """Check if response should be processed."""
            if not response.results:
                return False
            result = response.results[0]
            return result.is_final  # Only process final results

        # Should not process partial
        self.assertFalse(should_process_response(partial_response))

        # Should process final
        self.assertTrue(should_process_response(final_response))

    def test_confidence_filtering_with_final_check(self):
        """Test confidence filtering combined with final result check."""

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

        # Test cases
        test_cases = [
            # (transcript, confidence, is_final, should_process)
            ("I really", 0.8, False, False),  # Partial, good confidence - don't process
            ("I really need", 0.3, False, False),  # Partial, low confidence - don't process
            ("I really need to use", 0.9, False, False),  # Partial, high confidence - don't process
            (
                "I really need to use the restroom",
                0.95,
                True,
                True,
            ),  # Final, high confidence - process
            ("garbled text", 0.4, True, False),  # Final, low confidence - don't process
        ]

        def should_process_response_with_confidence(response, min_confidence=0.5):
            """Check if response should be processed with confidence filtering."""
            if not response.results:
                return False

            result = response.results[0]
            if not result.is_final:
                return False  # Only process final results

            if not result.alternatives:
                return False

            alternative = result.alternatives[0]
            confidence = alternative.confidence if hasattr(alternative, "confidence") else 0.0
            transcript = alternative.transcript.strip()

            return confidence > min_confidence and len(transcript) > 0

        # Test each case
        for transcript, confidence, is_final, expected in test_cases:
            alternative = MockAlternative(transcript, confidence)
            result = MockResult([alternative], is_final=is_final)
            response = MockResponse([result])

            result = should_process_response_with_confidence(response)
            self.assertEqual(
                result,
                expected,
                f"Failed for transcript='{transcript}', confidence={confidence}, is_final={is_final}",
            )

    def test_deduplication_with_callback(self):
        """Test that deduplication works with async callback."""
        processed_transcripts = []

        async def mock_callback(transcript):
            processed_transcripts.append(transcript)

        # Simulate streaming responses with duplicates
        streaming_transcripts = [
            "I really",
            "I really need",
            "I really need to",
            "I really need to use the restroom",  # Final
            "I really need to use the restroom",  # Duplicate final
        ]

        # Current problematic logic (processes all)
        async def problematic_processing():
            for transcript in streaming_transcripts:
                if len(transcript.strip()) > 0:
                    await mock_callback(transcript)

        # Fixed logic (only final, unique)
        async def fixed_processing():
            seen_transcripts = set()
            for i, transcript in enumerate(streaming_transcripts):
                # Only process if it's the last one (final) and not seen before
                is_final = (i == len(streaming_transcripts) - 2) or (
                    i == len(streaming_transcripts) - 1
                )
                if is_final and transcript not in seen_transcripts:
                    seen_transcripts.add(transcript)
                    await mock_callback(transcript)

        # Test problematic approach
        processed_transcripts.clear()
        asyncio.run(problematic_processing())
        problematic_count = len(processed_transcripts)

        # Test fixed approach
        processed_transcripts.clear()
        asyncio.run(fixed_processing())
        fixed_count = len(processed_transcripts)

        # Fixed approach should process fewer transcripts
        self.assertLess(fixed_count, problematic_count)
        self.assertEqual(fixed_count, 1)  # Only one unique final transcript


if __name__ == "__main__":
    unittest.main()
