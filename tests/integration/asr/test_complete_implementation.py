"""
Test the complete implementation of UI display fixes and performance tracking.
Integrates all the fixes for real-time LLM conversation streaming.
"""

import itertools
import unittest
from unittest.mock import patch

from src.asr.google_speech_v2 import GoogleSpeechV2Provider


class TestCompleteImplementation(unittest.TestCase):
    """Test the complete implementation with all fixes integrated."""

    def setUp(self):
        """Set up test fixtures."""
        self.provider = GoogleSpeechV2Provider()
        self.interim_results = []
        self.final_results = []
        self.performance_data = []

    def test_performance_tracking_integration(self):
        """Test that performance tracking is properly integrated."""
        # Initialize the provider
        self.provider.initialize(project_id="test-project")

        # Mock callbacks
        def interim_callback(transcript):
            self.interim_results.append(f"INTERIM: {transcript}")

        def final_callback(transcript):
            self.final_results.append(f"FINAL: {transcript}")

        # Start streaming with dual callbacks
        self.provider.start_streaming(
            transcription_callback=final_callback, interim_callback=interim_callback
        )

        # Simulate performance tracking
        self.provider._track_word_timing("I like", is_interim=True)
        self.provider._track_word_timing("I like to", is_interim=True)
        self.provider._track_word_timing("I like to play soccer", is_interim=False)

        self.provider._record_transcription_metric("I like", 0.4, False, 0.1)
        self.provider._record_transcription_metric("I like to", 0.5, False, 0.1)
        self.provider._record_transcription_metric("I like to play soccer", 0.9, True, 0.2)

        # Get performance summary
        summary = self.provider.get_performance_summary()

        print("\nPerformance tracking integration test:")
        print(f"Word timings recorded: {len(self.provider.performance_tracker['word_timings'])}")
        print(
            "Transcription metrics: "
            f"{len(self.provider.performance_tracker['transcription_metrics'])}"
        )
        print(f"Performance summary: {summary}")

        # Verify performance tracking
        self.assertGreater(len(self.provider.performance_tracker["word_timings"]), 0)
        self.assertGreater(len(self.provider.performance_tracker["transcription_metrics"]), 0)
        self.assertIn("total_transcriptions", summary)
        self.assertIn("average_confidence", summary)
        self.assertIn("total_words", summary)

        # Verify specific metrics
        self.assertEqual(summary["total_transcriptions"], 3)
        self.assertEqual(summary["final_transcriptions"], 1)
        self.assertAlmostEqual(summary["average_confidence"], 0.6, places=1)

    def test_dual_callback_with_performance_tracking(self):
        """Test dual callback system with integrated performance tracking."""
        # Initialize the provider
        self.provider.initialize(project_id="test-project")

        # Track callback calls
        interim_calls = []
        final_calls = []

        def interim_callback(transcript):
            interim_calls.append(transcript)

        def final_callback(transcript):
            final_calls.append(transcript)

        # Start streaming
        self.provider.start_streaming(
            transcription_callback=final_callback, interim_callback=interim_callback
        )

        counter = itertools.count(0)
        with patch("time.time", side_effect=lambda: float(next(counter))):
            self.provider._stream_start_time = 0.0
            self.provider.performance_tracker["session_start"] = 0.0
            self.provider.performance_tracker["speech_start_time"] = 0.0

            # Simulate interim results
            self.provider._track_word_timing("I like", is_interim=True)
            self.provider._record_transcription_metric("I like", 0.4, False, 0.1)
            self.provider._safe_call_callback("I like", is_interim=True)

            self.provider._track_word_timing("I like to", is_interim=True)
            self.provider._record_transcription_metric("I like to", 0.5, False, 0.1)
            self.provider._safe_call_callback("I like to", is_interim=True)

            # Simulate final result
            self.provider._track_word_timing("I like to play soccer", is_interim=False)
            self.provider._record_transcription_metric("I like to play soccer", 0.95, True, 0.2)
            self.provider._safe_call_callback("I like to play soccer", is_interim=False)

        print("\nDual callback with performance tracking test:")
        print(f"Interim calls: {interim_calls}")
        print(f"Final calls: {final_calls}")
        print(f"Performance summary: {self.provider.get_performance_summary()}")

        # Verify callback routing
        self.assertEqual(len(interim_calls), 2)
        self.assertEqual(len(final_calls), 1)
        self.assertIn("I like", interim_calls)
        self.assertIn("I like to", interim_calls)
        self.assertIn("I like to play soccer", final_calls)

        # Verify performance tracking
        summary = self.provider.get_performance_summary()
        self.assertGreater(summary.get("total_words", 0), 0)
        self.assertGreaterEqual(summary.get("session_duration", 0), 0)

    def test_ui_display_replacement_simulation(self):
        """Test simulation of UI display replacement behavior."""
        # This test simulates how the UI should behave with the new implementation
        display_state = []

        def ui_display_manager(transcript, is_interim=False):
            """Simulate improved UI display logic."""
            if is_interim:
                # Replace interim content
                if display_state and display_state[-1].startswith("INTERIM:"):
                    display_state[-1] = f"INTERIM: {transcript}"
                else:
                    display_state.append(f"INTERIM: {transcript}")
            else:
                # Add final content
                display_state.append(f"FINAL: {transcript}")

        # Simulate the improved streaming behavior
        ui_display_manager("WebRTC connection established. Waiting for speech...", is_interim=False)

        # Simulate interim results (should replace)
        ui_display_manager("I like", is_interim=True)
        ui_display_manager("I like to", is_interim=True)
        ui_display_manager("I like to play", is_interim=True)
        ui_display_manager("I like to play soccer", is_interim=True)

        # Simulate final result
        ui_display_manager(
            "I like to play soccer and Manchester United is my favorite team", is_interim=False
        )

        print("\nUI display replacement simulation:")
        for i, line in enumerate(display_state):
            print(f"  {i + 1}. {line}")

        # Verify improved display behavior
        self.assertEqual(len(display_state), 3)  # Connection + 1 interim (replaced) + 1 final
        self.assertIn("WebRTC connection established. Waiting for speech...", display_state[0])
        self.assertIn("INTERIM: I like to play soccer", display_state[1])  # Last interim only
        self.assertIn(
            "FINAL: I like to play soccer and Manchester United is my favorite team",
            display_state[2],
        )

        # Verify no duplicate interim results
        interim_lines = [line for line in display_state if line.startswith("INTERIM:")]
        self.assertEqual(len(interim_lines), 1)

    def test_end_to_end_llm_integration_scenario(self):
        """Test complete end-to-end scenario for LLM integration."""
        # Initialize provider
        self.provider.initialize(project_id="test-project")

        # Track LLM processing
        llm_inputs = []
        ui_updates = []

        def ui_interim_callback(transcript):
            """UI callback for real-time display."""
            ui_updates.append(f"UI: {transcript}")

        def llm_final_callback(transcript):
            """LLM callback for processing."""
            llm_inputs.append(f"LLM: {transcript}")

        # Start streaming
        self.provider.start_streaming(
            transcription_callback=llm_final_callback, interim_callback=ui_interim_callback
        )

        # Simulate complete conversation session
        conversation_stream = [
            ("I like", False, 0.4),
            ("I like to", False, 0.5),
            ("I like to play", False, 0.7),
            ("I like to play soccer", False, 0.9),
            ("I like to play soccer and Manchester United is my favorite team", True, 0.95),
        ]

        counter = itertools.count(0)
        with patch("time.time", side_effect=lambda: float(next(counter))):
            self.provider._stream_start_time = 0.0
            self.provider.performance_tracker["session_start"] = 0.0
            self.provider.performance_tracker["speech_start_time"] = 0.0
            for transcript, is_final, confidence in conversation_stream:
                # Track performance
                self.provider._track_word_timing(transcript, is_interim=not is_final)
                self.provider._record_transcription_metric(transcript, confidence, is_final, 0.1)

                # Route to appropriate callback
                if is_final:
                    self.provider._safe_call_callback(transcript, is_interim=False)
                else:
                    self.provider._safe_call_callback(transcript, is_interim=True)

        print("\nEnd-to-end LLM integration scenario:")
        print(f"UI updates (interim): {len(ui_updates)}")
        for update in ui_updates:
            print(f"  {update}")

        print(f"LLM inputs (final): {len(llm_inputs)}")
        for input_text in llm_inputs:
            print(f"  {input_text}")

        print(f"Performance summary: {self.provider.get_performance_summary()}")

        # Verify proper routing
        self.assertEqual(len(ui_updates), 4)  # 4 interim results
        self.assertEqual(len(llm_inputs), 1)  # 1 final result

        # Verify LLM gets complete, confident sentences
        self.assertIn(
            "LLM: I like to play soccer and Manchester United is my favorite team", llm_inputs
        )

        # Verify UI gets progressive updates
        self.assertIn("UI: I like", ui_updates)
        self.assertIn("UI: I like to play soccer", ui_updates)

        # Verify performance tracking
        summary = self.provider.get_performance_summary()
        self.assertEqual(summary["total_transcriptions"], 5)
        self.assertEqual(summary["final_transcriptions"], 1)
        self.assertGreater(summary["average_confidence"], 0.6)

    def test_performance_metrics_accuracy(self):
        """Test accuracy of performance metrics calculation."""
        # Initialize provider
        self.provider.initialize(project_id="test-project")

        # Create controlled test data
        counter = itertools.count(0)
        with patch("time.time", side_effect=lambda: float(next(counter))):
            self.provider._stream_start_time = 0.0
            self.provider.performance_tracker["session_start"] = 0.0
            self.provider.performance_tracker["speech_start_time"] = 0.0
            # Track word timings
            self.provider._track_word_timing("Hello", is_interim=True)
            self.provider._track_word_timing("Hello there", is_interim=True)
            self.provider._track_word_timing("Hello there how", is_interim=True)
            self.provider._track_word_timing("Hello there how are", is_interim=False)
            self.provider._track_word_timing("Hello there how are you", is_interim=False)

            # Record metrics
            self.provider._record_transcription_metric("Hello", 0.4, False, 0.1)
            self.provider._record_transcription_metric("Hello there", 0.5, False, 0.1)
            self.provider._record_transcription_metric("Hello there how", 0.7, False, 0.1)
            self.provider._record_transcription_metric("Hello there how are", 0.9, True, 0.2)
            self.provider._record_transcription_metric("Hello there how are you", 0.95, True, 0.2)

        summary = self.provider.get_performance_summary()

        print("\nPerformance metrics accuracy test:")
        print(f"Summary: {summary}")

        # Verify metrics accuracy
        self.assertEqual(summary["total_transcriptions"], 5)
        self.assertEqual(summary["final_transcriptions"], 2)
        self.assertAlmostEqual(summary["average_confidence"], 0.69, places=2)
        self.assertAlmostEqual(summary["average_processing_time"], 0.14, places=2)
        self.assertEqual(summary["total_words"], 5)
        self.assertGreater(summary["session_duration"], 0)

    def test_streaming_metrics_non_zero(self):
        self.provider.initialize(project_id="test-project")

        self.provider._stream_start_time = 0.0
        self.provider._last_transcript_event_time = None

        with patch("time.time", return_value=0.5):
            t1 = self.provider._compute_processing_time()
        with patch("time.time", return_value=1.0):
            t2 = self.provider._compute_processing_time()

        self.assertGreater(t1, 0.0)
        self.assertGreater(t2, 0.0)

        self.provider._record_transcription_metric("Hello", 0.9, True, t1)
        self.provider._record_transcription_metric("Hello there", 0.9, True, t2)
        summary = self.provider.get_performance_summary()
        self.assertGreater(summary["average_processing_time"], 0.0)

        with patch("time.time", side_effect=[0.0, 0.5, 1.0, 1.5]):
            self.provider._track_word_timing("Hello", is_interim=True)
            self.provider._track_word_timing("Hello there", is_interim=True)
            self.provider._track_word_timing("Hello there friend", is_interim=False)

        summary = self.provider.get_performance_summary()
        self.assertGreater(summary["average_word_time"], 0.0)


if __name__ == "__main__":
    unittest.main()
