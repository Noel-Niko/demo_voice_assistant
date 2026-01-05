"""
Test the dual callback system for interim and final results.
Tests that interim results go to the UI and final results go to the LLM.
"""

import unittest


class TestDualCallbackSystem(unittest.TestCase):
    """Test the dual callback system for real-time streaming."""

    def test_interim_and_final_callback_routing(self):
        """Test that interim results go to interim_callback and final results go to transcription_callback."""

        # Mock response structure for Google Speech v2
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

        # Simulate streaming responses with interim and final results
        streaming_responses = [
            # Interim results (should go to interim_callback)
            MockResponse([MockResult([MockAlternative("I,", 0.4)], is_final=False)]),
            MockResponse([MockResult([MockAlternative("I like,", 0.5)], is_final=False)]),
            MockResponse([MockResult([MockAlternative("I like to", 0.6)], is_final=False)]),
            MockResponse([MockResult([MockAlternative("I like to drink", 0.7)], is_final=False)]),
            MockResponse([MockResult([MockAlternative("I like to drink a", 0.8)], is_final=False)]),
            # Final result (should go to transcription_callback)
            MockResponse(
                [
                    MockResult(
                        [MockAlternative("I like to drink a glass of wine.", 0.95)], is_final=True
                    )
                ]
            ),
        ]

        # Track which callbacks receive which transcripts
        interim_received = []
        final_received = []

        def interim_callback(transcript):
            interim_received.append(transcript)

        def final_callback(transcript):
            final_received.append(transcript)

        def process_dual_callback_stream(stream, interim_cb, final_cb):
            """Process stream with dual callback logic."""
            seen_transcripts = set()

            for response in stream:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                alternative = result.alternatives[0]
                transcript = alternative.transcript.strip()
                confidence = getattr(alternative, "confidence", 0.0)
                is_final = getattr(result, "is_final", False)

                # Handle interim results for real-time display
                if not is_final and interim_cb and confidence > 0.3:
                    interim_cb(transcript)

                # Handle final results for LLM processing
                if (
                    is_final
                    and confidence > 0.5
                    and transcript
                    and transcript not in seen_transcripts
                ):
                    seen_transcripts.add(transcript)
                    if final_cb:
                        final_cb(transcript)

        # Process the stream
        process_dual_callback_stream(streaming_responses, interim_callback, final_callback)

        print("\nDual callback system test:")
        print(f"Interim transcripts received: {len(interim_received)}")
        for i, t in enumerate(interim_received):
            print(f"  Interim {i + 1}: {t}")

        print(f"Final transcripts received: {len(final_received)}")
        for i, t in enumerate(final_received):
            print(f"  Final {i + 1}: {t}")

        # Verify interim results routing
        self.assertEqual(len(interim_received), 5)
        self.assertIn("I,", interim_received)
        self.assertIn("I like,", interim_received)
        self.assertIn("I like to", interim_received)
        self.assertIn("I like to drink", interim_received)
        self.assertIn("I like to drink a", interim_received)

        # Verify final result routing
        self.assertEqual(len(final_received), 1)
        self.assertIn("I like to drink a glass of wine.", final_received)

        # Verify no crossover
        self.assertNotIn("I like to drink a glass of wine.", interim_received)
        self.assertNotIn("I,", final_received)

    def test_confidence_thresholds(self):
        """Test that confidence thresholds work correctly for both callbacks."""

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

        # Test various confidence levels
        test_responses = [
            # Low confidence interim (should be ignored)
            MockResponse([MockResult([MockAlternative("Low confidence", 0.2)], is_final=False)]),
            # Good confidence interim (should be processed)
            MockResponse([MockResult([MockAlternative("Good confidence", 0.6)], is_final=False)]),
            # Low confidence final (should be ignored)
            MockResponse([MockResult([MockAlternative("Low final", 0.3)], is_final=True)]),
            # Good confidence final (should be processed)
            MockResponse([MockResult([MockAlternative("Good final", 0.8)], is_final=True)]),
        ]

        interim_received = []
        final_received = []

        def interim_callback(transcript):
            interim_received.append(transcript)

        def final_callback(transcript):
            final_received.append(transcript)

        def process_with_thresholds(stream, interim_cb, final_cb):
            """Process with confidence thresholds."""
            seen_transcripts = set()

            for response in stream:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                alternative = result.alternatives[0]
                transcript = alternative.transcript.strip()
                confidence = getattr(alternative, "confidence", 0.0)
                is_final = getattr(result, "is_final", False)

                # Handle interim results (confidence > 0.3)
                if not is_final and interim_cb and confidence > 0.3:
                    interim_cb(transcript)

                # Handle final results (confidence > 0.5)
                if (
                    is_final
                    and confidence > 0.5
                    and transcript
                    and transcript not in seen_transcripts
                ):
                    seen_transcripts.add(transcript)
                    if final_cb:
                        final_cb(transcript)

        process_with_thresholds(test_responses, interim_callback, final_callback)

        print("\nConfidence threshold test:")
        print(f"Interim received: {interim_received}")
        print(f"Final received: {final_received}")

        # Verify confidence filtering
        self.assertEqual(len(interim_received), 1)
        self.assertIn("Good confidence", interim_received)
        self.assertNotIn("Low confidence", interim_received)

        self.assertEqual(len(final_received), 1)
        self.assertIn("Good final", final_received)
        self.assertNotIn("Low final", final_received)

    def test_real_time_conversation_scenario(self):
        """Test a realistic real-time conversation scenario."""

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

        # Simulate a real conversation with progressive updates
        conversation_stream = [
            # First utterance
            MockResponse([MockResult([MockAlternative("Hello", 0.4)], is_final=False)]),
            MockResponse([MockResult([MockAlternative("Hello there", 0.5)], is_final=False)]),
            MockResponse([MockResult([MockAlternative("Hello there,", 0.6)], is_final=False)]),
            MockResponse(
                [MockResult([MockAlternative("Hello there, how are you?", 0.8)], is_final=False)]
            ),
            MockResponse(
                [
                    MockResult(
                        [MockAlternative("Hello there, how are you today?", 0.95)], is_final=True
                    )
                ]
            ),
            # Second utterance
            MockResponse(
                [MockResult([MockAlternative("I'm", 0.35)], is_final=False)]
            ),  # Above 0.3 threshold
            MockResponse([MockResult([MockAlternative("I'm doing", 0.5)], is_final=False)]),
            MockResponse([MockResult([MockAlternative("I'm doing well", 0.7)], is_final=False)]),
            MockResponse(
                [MockResult([MockAlternative("I'm doing well thanks", 0.9)], is_final=False)]
            ),
            MockResponse(
                [
                    MockResult(
                        [MockAlternative("I'm doing well thanks for asking", 0.98)], is_final=True
                    )
                ]
            ),
        ]

        ui_updates = []
        llm_inputs = []

        def ui_display_callback(transcript):
            """Simulates UI display of interim results."""
            ui_updates.append(f"UI: {transcript}")

        def llm_processing_callback(transcript):
            """Simulates LLM processing of final results."""
            llm_inputs.append(f"LLM: {transcript}")

        def process_conversation(stream, ui_cb, llm_cb):
            """Process conversation with dual callbacks."""
            seen_transcripts = set()

            for response in stream:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                alternative = result.alternatives[0]
                transcript = alternative.transcript.strip()
                confidence = getattr(alternative, "confidence", 0.0)
                is_final = getattr(result, "is_final", False)

                # UI gets interim updates for real-time feedback
                if not is_final and ui_cb and confidence > 0.3:
                    ui_cb(transcript)

                # LLM gets final, complete sentences for processing
                if (
                    is_final
                    and confidence > 0.5
                    and transcript
                    and transcript not in seen_transcripts
                ):
                    seen_transcripts.add(transcript)
                    if llm_cb:
                        llm_cb(transcript)

        process_conversation(conversation_stream, ui_display_callback, llm_processing_callback)

        print("\nReal-time conversation scenario:")
        print(f"UI updates (interim): {len(ui_updates)}")
        for update in ui_updates:
            print(f"  {update}")

        print(f"LLM inputs (final): {len(llm_inputs)}")
        for input_text in llm_inputs:
            print(f"  {input_text}")

        # Verify the conversation flow
        self.assertEqual(len(ui_updates), 8)  # 4 interim + 4 interim
        self.assertEqual(len(llm_inputs), 2)  # 2 final sentences

        # Verify UI gets progressive updates
        self.assertIn("UI: Hello", ui_updates)
        self.assertIn("UI: Hello there", ui_updates)
        self.assertIn("UI: I'm", ui_updates)
        self.assertIn("UI: I'm doing well", ui_updates)

        # Verify LLM gets complete sentences
        self.assertIn("LLM: Hello there, how are you today?", llm_inputs)
        self.assertIn("LLM: I'm doing well thanks for asking", llm_inputs)

        # Verify no final results in UI
        for update in ui_updates:
            self.assertFalse(update.endswith("today?"))
            self.assertFalse(update.endswith("asking"))

        # Verify no interim results in LLM (check for exact matches, not substrings)
        interim_texts = [
            "Hello",
            "Hello there",
            "Hello there,",
            "Hello there, how are you?",
            "I'm",
            "I'm doing",
            "I'm doing well",
            "I'm doing well thanks",
        ]
        for input_text in llm_inputs:
            self.assertNotIn(input_text.replace("LLM: ", ""), interim_texts)


if __name__ == "__main__":
    unittest.main()
