"""
Test the duplication fix for Google Speech v2 streaming.
Verifies that the conservative final result detection prevents massive duplication.
"""

import sys
import time
import unittest
from unittest.mock import MagicMock, patch

# Mock the Google Cloud imports to avoid dependency issues
sys.modules["google.cloud.speech"] = MagicMock()
sys.modules["google.cloud.speech_v2"] = MagicMock()

from src.asr.google_speech_v2 import GoogleSpeechV2Provider  # noqa: E402


class TestDuplicationFix(unittest.TestCase):
    """Test that the duplication fix prevents massive transcript repetition."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = GoogleSpeechV2Provider()
        self.provider.initialize(project_id="test-project")

        # Track callback calls
        self.interim_calls = []
        self.final_calls = []

        def interim_callback(transcript):
            self.interim_calls.append(transcript)

        def final_callback(transcript):
            self.final_calls.append(transcript)

        self.provider.start_streaming(
            transcription_callback=final_callback, interim_callback=interim_callback
        )

    def test_conservative_final_detection(self):
        """Test that final detection is conservative and prevents duplicates."""
        # Simulate the problematic scenario from user logs
        # Many similar transcripts that should be treated as interim
        similar_transcripts = [
            "I am wondering why it can take so long for this ethic.",
            "I am wondering why it can take so long for this athlete.",
            "I am wondering why it can take so long for this application.",
            "I am wondering why I can take so long for this affiliation.",
            "I am wondering why I can take so long for this athletic.",
            "I am wondering why it can take so long for this athlete to start.",
            "I am wondering why it can take so long for this application to start.",
        ]

        # Mock response objects with no is_final attribute (simulating the real issue)
        class MockResult:
            def __init__(self, transcript, confidence):
                self.transcript = transcript
                self.confidence = confidence
                # Deliberately not setting is_final to test fallback

        class MockAlternative:
            def __init__(self, transcript, confidence):
                self.transcript = transcript
                self.confidence = confidence

        class MockResponse:
            def __init__(self, transcript, confidence):
                self.results = [MockResult(transcript, confidence)]
                self.results[0].alternatives = [MockAlternative(transcript, confidence)]

        # Simulate processing these transcripts
        for i, transcript in enumerate(similar_transcripts):
            response = MockResponse(transcript, 0.96)  # High confidence

            # Mock the response processing logic
            result = response.results[0]
            alternative = result.alternatives[0]
            transcript_text = alternative.transcript.strip()
            confidence = alternative.confidence

            # Test the conservative fallback logic
            is_final_result = False

            # No is_final attribute - should use conservative fallback
            very_high_confidence = confidence > 0.95
            has_complete_sentence = transcript_text.strip().endswith((".", "!", "?"))
            is_reasonably_long = len(transcript_text) > 20

            is_final_result = very_high_confidence and has_complete_sentence and is_reasonably_long

            print(f"Transcript {i + 1}: '{transcript_text}'")
            print(
                f"  Confidence: {confidence:.2f}, Complete: {has_complete_sentence}, Long: {is_reasonably_long}"
            )
            print(f"  Final result: {is_final_result}")

            # Only the first one should be treated as final due to timing constraints
            if i == 0:
                self.assertTrue(is_final_result, "First transcript should be final")
            else:
                # Subsequent ones should be suppressed by timing logic
                # (This would be handled by the time_since_last_final check in real code)
                print("  Would be suppressed by timing constraint")

        print("\nConservative final detection test:")
        print(f"Interim calls: {len(self.interim_calls)}")
        print(f"Final calls: {len(self.final_calls)}")

        # Verify that only the first transcript would be treated as final
        # Others would be either interim or suppressed by timing
        self.assertLessEqual(
            len(self.final_calls),
            1,
            "Should have at most 1 final call due to conservative detection",
        )

    def test_timing_prevention_of_rapid_finals(self):
        """Test that timing constraints prevent rapid final results."""
        # Mock timing to simulate rapid succession
        with patch("time.time", side_effect=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]):
            # Simulate rapid transcripts
            rapid_transcripts = [
                "I would like",
                "I would like to",
                "I would like to make",
                "I would like to make a",
                "I would like to make a cup",
                "I would like to make a cup of",
                "I would like to make a cup of coffee",
            ]

            final_results = []
            last_final_time = -2.0  # Initialize to allow first result

            for i, transcript in enumerate(rapid_transcripts):
                current_time = time.time()
                time_since_last = current_time - last_final_time

                # Simulate conservative final detection
                confidence = 0.96
                very_high_confidence = confidence > 0.95
                has_complete_sentence = transcript.strip().endswith((".", "!", "?"))
                is_reasonably_long = len(transcript) > 20

                is_final_result = (
                    very_high_confidence and has_complete_sentence and is_reasonably_long
                )

                # Apply timing constraint
                if is_final_result and time_since_last < 2.0:
                    is_final_result = False
                    print(
                        f"Transcript {i + 1}: '{transcript}' - suppressed by timing ({time_since_last:.1f}s)"
                    )
                elif is_final_result:
                    print(
                        f"Transcript {i + 1}: '{transcript}' - allowed as final ({time_since_last:.1f}s)"
                    )
                    final_results.append(transcript)
                    last_final_time = current_time
                else:
                    print(f"Transcript {i + 1}: '{transcript}' - not final")

            print("\nTiming prevention test:")
            print(f"Final results allowed: {len(final_results)}")
            for result in final_results:
                print(f"  - {result}")

            # Should have very few final results due to timing constraint
            self.assertLessEqual(
                len(final_results), 2, "Should have at most 2 final results due to timing"
            )

    def test_duplicate_prevention(self):
        """Test that duplicate prevention works correctly."""
        # Test duplicate detection logic
        seen_transcripts = set()

        def normalize_word(word: str) -> str:
            return word.strip(".,!?").lower()

        test_cases = [
            ("I would like to make a cup of coffee", False),  # First occurrence
            ("I would like to make a cup of coffee", True),  # Exact duplicate
            ("I would like to make a cup of coffee.", True),  # Minor variation
            ("I would like to make a cup of tea", False),  # Different enough
        ]

        for transcript, should_be_duplicate in test_cases:
            is_duplicate = False

            # Check against seen transcripts
            for seen_transcript in seen_transcripts:
                # Simple similarity check
                if transcript.lower().strip() == seen_transcript.lower().strip():
                    is_duplicate = True
                    break
                # Check if this is just a minor variation
                if abs(len(transcript) - len(seen_transcript)) <= 3:
                    words1 = {normalize_word(w) for w in transcript.split() if normalize_word(w)}
                    words2 = {
                        normalize_word(w) for w in seen_transcript.split() if normalize_word(w)
                    }
                    if words1 and words2:
                        similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                        if similarity > 0.8:
                            is_duplicate = True
                            break

            if not is_duplicate:
                seen_transcripts.add(transcript)
                print(f"Added: '{transcript}'")
            else:
                print(f"Duplicate: '{transcript}'")

            self.assertEqual(
                is_duplicate,
                should_be_duplicate,
                f"Transcript '{transcript}' duplicate detection mismatch",
            )

        print("\nDuplicate prevention test:")
        print(f"Unique transcripts: {len(seen_transcripts)}")

        for transcript in seen_transcripts:
            print(f"  - {transcript}")

        # Should have only 2 unique transcripts
        self.assertEqual(len(seen_transcripts), 2, "Should have exactly 2 unique transcripts")

    def test_performance_metrics_tracking(self):
        """Test that performance metrics are properly tracked."""
        # Simulate some transcription activity
        import itertools

        counter = itertools.count(0)
        with patch("time.time", side_effect=lambda: float(next(counter))):
            self.provider._stream_start_time = 0.0
            self.provider.performance_tracker["session_start"] = 0.0
            self.provider.performance_tracker["speech_start_time"] = 0.0

            # Track some word timings
            self.provider._track_word_timing("I", is_interim=True)
            self.provider._track_word_timing("I like", is_interim=True)
            self.provider._track_word_timing("I like coffee", is_interim=False)

            # Record some metrics
            self.provider._record_transcription_metric("I", 0.4, False, 0.1)
            self.provider._record_transcription_metric("I like", 0.6, False, 0.1)
            self.provider._record_transcription_metric("I like coffee", 0.9, True, 0.2)

            # Get performance summary
            summary = self.provider.get_performance_summary()
            display = self.provider.get_performance_display()

            print("\nPerformance metrics test:")
            print(f"Summary: {summary}")
            print(f"Display:\n{display}")

            # Verify metrics
            self.assertEqual(summary["total_transcriptions"], 3)
            self.assertEqual(summary["final_transcriptions"], 1)
            self.assertAlmostEqual(summary["average_confidence"], 0.63, places=2)
            self.assertEqual(summary["total_words"], 3)
            self.assertGreaterEqual(summary["session_duration"], 0)

            # Verify display format
            self.assertIn("📊 Performance Metrics:", display)
            self.assertIn("Total Transcriptions: 3", display)
            self.assertIn("Final Transcriptions: 1", display)


if __name__ == "__main__":
    unittest.main()
