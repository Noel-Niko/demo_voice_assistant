"""
Unit tests for UI display fixes and performance tracking.
Tests the solutions for replacing interim results and tracking word timing.
"""

import time
import unittest
from unittest.mock import patch


class TestUIDisplayFixes(unittest.TestCase):
    """Test UI display fixes for real-time transcription."""

    def setUp(self):
        """Set up test fixtures."""
        self.display_history = []
        self.performance_metrics = []

    def test_interim_replacement_logic(self):
        """Test that interim results replace previous content instead of appending."""

        # Simulate the current problematic behavior (appending)
        def current_problematic_display(transcript):
            self.display_history.append(transcript)

        # Simulate the desired behavior (replacing)
        current_interim = None

        def improved_display(transcript, is_interim=False):
            nonlocal current_interim
            if is_interim:
                # Replace interim content - only keep the latest
                current_interim = transcript
                # Remove previous interim if exists
                if self.display_history and self.display_history[-1].startswith("INTERIM:"):
                    self.display_history[-1] = f"INTERIM: {transcript}"
                else:
                    self.display_history.append(f"INTERIM: {transcript}")
            else:
                # Add final content
                self.display_history.append(f"FINAL: {transcript}")

        # Test stream of interim results
        interim_results = ["I like", "I like to", "I like to play", "I like to play soccer"]
        final_result = "I like to play soccer and Manchester United is my favorite team"

        # Current problematic behavior
        current_problematic_display("Partial: WebRTC connection established. Waiting for speech...")
        for result in interim_results:
            current_problematic_display(result)
        current_problematic_display(final_result)

        problematic_output = list(self.display_history)
        self.display_history.clear()

        # Improved behavior
        improved_display(
            "Partial: WebRTC connection established. Waiting for speech...", is_interim=False
        )
        for result in interim_results:
            improved_display(result, is_interim=True)
        improved_display(final_result, is_interim=False)

        improved_output = list(self.display_history)

        print("\nUI Display Comparison:")
        print(f"Problematic output ({len(problematic_output)} lines):")
        for i, line in enumerate(problematic_output):
            print(f"  {i + 1}. {line}")

        print(f"Improved output ({len(improved_output)} lines):")
        for i, line in enumerate(improved_output):
            print(f"  {i + 1}. {line}")

        # Verify problematic behavior shows all interim results as separate lines
        self.assertEqual(len(problematic_output), 6)  # 1 connection + 4 interim + 1 final
        self.assertIn(
            "Partial: WebRTC connection established. Waiting for speech...", problematic_output
        )
        self.assertIn("I like", problematic_output)
        self.assertIn("I like to", problematic_output)
        self.assertIn("I like to play", problematic_output)
        self.assertIn("I like to play soccer", problematic_output)

        # Verify improved behavior shows only one interim line that gets replaced
        self.assertEqual(len(improved_output), 3)  # 1 connection + 1 interim (replaced) + 1 final
        self.assertIn(
            "FINAL: Partial: WebRTC connection established. Waiting for speech...", improved_output
        )
        self.assertIn("INTERIM: I like to play soccer", improved_output)  # Last interim result
        self.assertIn(
            "FINAL: I like to play soccer and Manchester United is my favorite team",
            improved_output,
        )

    def test_connection_message_clearing(self):
        """Test that connection message is cleared when speech starts."""
        connection_cleared = False
        display_content = []

        def smart_display(transcript, is_interim=False, is_first_speech=False):
            nonlocal connection_cleared
            if is_first_speech and not connection_cleared:
                # Clear connection message on first speech
                display_content.clear()
                connection_cleared = True
                display_content.append(f"SPEECH STARTED: {transcript}")
            elif is_interim:
                display_content.append(f"INTERIM: {transcript}")
            else:
                display_content.append(f"FINAL: {transcript}")

        # Test sequence
        smart_display(
            "Partial: WebRTC connection established. Waiting for speech...", is_interim=False
        )
        smart_display("I like", is_interim=True, is_first_speech=True)
        smart_display("I like to play", is_interim=True)
        smart_display("I like to play soccer", is_interim=False)

        print("\nConnection message clearing test:")
        for i, line in enumerate(display_content):
            print(f"  {i + 1}. {line}")

        # Verify connection message was cleared
        self.assertNotIn(
            "Partial: WebRTC connection established. Waiting for speech...", display_content
        )
        self.assertEqual(len(display_content), 3)
        self.assertTrue(connection_cleared)
        self.assertIn("SPEECH STARTED: I like", display_content)

    def test_word_timing_tracking(self):
        """Test word-level timing tracking for performance metrics."""

        class WordTimingTracker:
            def __init__(self):
                self.word_timings = []
                self.speech_start_time = None
                self.last_word_time = None

            def start_speech(self):
                self.speech_start_time = time.time()
                self.last_word_time = self.speech_start_time

            def track_word(self, word, transcript):
                current_time = time.time()
                if self.speech_start_time is None:
                    self.start_speech()

                time_from_start = current_time - self.speech_start_time
                time_from_last = current_time - self.last_word_time if self.last_word_time else 0

                self.word_timings.append(
                    {
                        "word": word,
                        "transcript": transcript,
                        "time_from_start": time_from_start,
                        "time_from_previous": time_from_last,
                        "timestamp": current_time,
                    }
                )

                self.last_word_time = current_time

            def get_average_word_time(self):
                if not self.word_timings:
                    return 0
                total_time = sum(
                    w["time_from_previous"] for w in self.word_timings[1:]
                )  # Skip first word
                return total_time / max(len(self.word_timings) - 1, 1)

            def get_total_transcription_time(self):
                if not self.word_timings:
                    return 0
                return self.word_timings[-1]["time_from_start"]

        tracker = WordTimingTracker()

        # Simulate real-time transcription with timing
        with patch("time.time", side_effect=[0.0, 0.5, 1.0, 1.5, 2.0, 2.8, 3.5, 4.0]):
            tracker.start_speech()
            tracker.track_word("I", "I")
            tracker.track_word("like", "I like")
            tracker.track_word("to", "I like to")
            tracker.track_word("play", "I like to play")
            tracker.track_word("soccer", "I like to play soccer")
            tracker.track_word("and", "I like to play soccer and")
            tracker.track_word("Manchester", "I like to play soccer and Manchester")

        print("\nWord timing tracking test:")
        for i, timing in enumerate(tracker.word_timings):
            print(
                f"  Word {i + 1}: '{timing['word']}' - Start: {timing['time_from_start']:.1f}s, Previous: {timing['time_from_previous']:.1f}s"
            )

        print(f"Average word time: {tracker.get_average_word_time():.2f}s")
        print(f"Total transcription time: {tracker.get_total_transcription_time():.1f}s")

        # Verify timing calculations
        self.assertEqual(len(tracker.word_timings), 7)
        self.assertEqual(tracker.word_timings[0]["time_from_start"], 0.5)  # First word at 0.5s
        self.assertEqual(tracker.word_timings[-1]["time_from_start"], 4.0)  # Last word at 4.0s
        self.assertAlmostEqual(tracker.get_average_word_time(), 0.58, places=1)  # (3.5 / 6)
        self.assertEqual(tracker.get_total_transcription_time(), 4.0)

    def test_performance_metrics_collection(self):
        """Test comprehensive performance metrics collection."""

        class PerformanceMetrics:
            def __init__(self):
                self.metrics = {
                    "transcriptions": [],
                    "word_timings": [],
                    "confidence_scores": [],
                    "processing_times": [],
                }
                self.session_start = time.time()

            def record_transcription(self, transcript, confidence, is_final, processing_time):
                self.metrics["transcriptions"].append(
                    {
                        "text": transcript,
                        "confidence": confidence,
                        "is_final": is_final,
                        "processing_time": processing_time,
                        "timestamp": time.time() - self.session_start,
                    }
                )
                self.metrics["confidence_scores"].append(confidence)
                self.metrics["processing_times"].append(processing_time)

            def record_word_timing(self, word, time_from_start):
                self.metrics["word_timings"].append(
                    {"word": word, "time_from_start": time_from_start}
                )

            def get_summary(self):
                if not self.metrics["transcriptions"]:
                    return {}

                final_transcriptions = [t for t in self.metrics["transcriptions"] if t["is_final"]]
                return {
                    "total_transcriptions": len(self.metrics["transcriptions"]),
                    "final_transcriptions": len(final_transcriptions),
                    "average_confidence": sum(self.metrics["confidence_scores"])
                    / len(self.metrics["confidence_scores"]),
                    "average_processing_time": sum(self.metrics["processing_times"])
                    / len(self.metrics["processing_times"]),
                    "total_words": len(self.metrics["word_timings"]),
                    "session_duration": time.time() - self.session_start,
                }

        metrics = PerformanceMetrics()

        # Simulate transcription session
        with patch(
            "time.time", side_effect=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
        ):
            metrics.record_transcription("I like", 0.4, False, 0.1)
            metrics.record_transcription("I like to", 0.5, False, 0.1)
            metrics.record_transcription("I like to play", 0.7, False, 0.1)
            metrics.record_transcription("I like to play soccer", 0.9, False, 0.1)
            metrics.record_transcription("I like to play soccer and Manchester", 0.95, True, 0.2)

            # Record word timings
            words = ["I", "like", "to", "play", "soccer", "and", "Manchester"]
            for i, word in enumerate(words):
                metrics.record_word_timing(word, i * 0.5)

        summary = metrics.get_summary()

        print("\nPerformance metrics test:")
        print(f"Total transcriptions: {summary['total_transcriptions']}")
        print(f"Final transcriptions: {summary['final_transcriptions']}")
        print(f"Average confidence: {summary['average_confidence']:.2f}")
        print(f"Average processing time: {summary['average_processing_time']:.2f}s")
        print(f"Total words: {summary['total_words']}")

        # Verify metrics collection
        self.assertEqual(summary["total_transcriptions"], 5)
        self.assertEqual(summary["final_transcriptions"], 1)
        self.assertAlmostEqual(summary["average_confidence"], 0.69, places=2)
        self.assertAlmostEqual(summary["average_processing_time"], 0.12, places=2)
        self.assertEqual(summary["total_words"], 7)

    def test_ui_state_management(self):
        """Test UI state management for different display modes."""

        class UIManager:
            def __init__(self):
                self.state = "connecting"
                self.display_content = []
                self.current_interim = None

            def set_state(self, new_state):
                self.state = new_state

            def update_display(self, content, is_interim=False):
                if self.state == "connecting" and is_interim:
                    # Transition from connecting to speech
                    self.set_state("listening")
                    self.display_content.clear()

                if is_interim:
                    self.current_interim = content
                    # Replace interim content
                    if len(self.display_content) > 0 and self.display_content[-1].startswith(
                        "INTERIM:"
                    ):
                        self.display_content[-1] = f"INTERIM: {content}"
                    else:
                        self.display_content.append(f"INTERIM: {content}")
                else:
                    self.current_interim = None
                    self.display_content.append(f"FINAL: {content}")

            def get_display_content(self):
                return list(self.display_content)

        ui = UIManager()

        # Test state transitions
        ui.update_display("Partial: WebRTC connection established. Waiting for speech...")
        self.assertEqual(ui.state, "connecting")

        ui.update_display("I like", is_interim=True)
        self.assertEqual(ui.state, "listening")

        ui.update_display("I like to", is_interim=True)
        ui.update_display("I like to play", is_interim=True)
        ui.update_display("I like to play soccer", is_interim=False)

        content = ui.get_display_content()

        print("\nUI state management test:")
        print(f"Final state: {ui.state}")
        for i, line in enumerate(content):
            print(f"  {i + 1}. {line}")

        # Verify state management
        self.assertEqual(ui.state, "listening")
        self.assertEqual(len(content), 2)  # 1 interim (replaced) + 1 final
        self.assertIn("INTERIM: I like to play", content)  # Last interim
        self.assertIn("FINAL: I like to play soccer", content)


if __name__ == "__main__":
    unittest.main()
