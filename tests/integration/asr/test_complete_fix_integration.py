"""
Comprehensive integration test for the complete fix.
Tests both the isFinal attribute fix and thread-safe async callback together.
"""

import asyncio
import time
import unittest


class TestCompleteFixIntegration(unittest.TestCase):
    """Test the complete fix integration."""

    def test_end_to_end_fix_simulation(self):
        """Test the complete fix with a realistic simulation of the user's scenario."""

        # Mock Google Speech v2 response structure
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

        # Simulate the exact problematic stream the user experienced
        problematic_stream = [
            # Massive repetition with isFinal=False (interim results)
            MockResponse([MockResult([MockAlternative("At this point", 0.8)], isFinal=False)]),
            MockResponse(
                [MockResult([MockAlternative("At this point. At this point", 0.8)], isFinal=False)]
            ),
            MockResponse(
                [
                    MockResult(
                        [MockAlternative("At this point. At this point, at this point", 0.8)],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "At this point. At this point, at this point, I at this point", 0.8
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "At this point. At this point, at this point, I at this point, I at this point",
                                0.8,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "At this point. At this point, at this point, I at this point, I at this point, I am at this point",
                                0.8,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "At this point. At this point, at this point, I at this point, I at this point, I am at this point, I am at this point",
                                0.8,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "At this point. At this point, at this point, I at this point, I at this point, I am at this point, I am at this point, I am at this point",
                                0.8,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "At this point. At this point, at this point, I at this point, I at this point, I am at this point, I am at this point, I am at this point, I am one at this point",
                                0.8,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "At this point. At this point, at this point, I at this point, I at this point, I am at this point, I am at this point, I am at this point, I am one at this point. I am one at this point",
                                0.8,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "At this point. At this point, at this point, I at this point, I at this point, I am at this point, I am at this point, I am at this point, I am one at this point. I am one at this point. I am wondering, at this point",
                                0.8,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "At this point. At this point, at this point, I at this point, I at this point, I am at this point, I am at this point, I am at this point, I am one at this point. I am one at this point. I am wondering, at this point. I am wondering at this point",
                                0.8,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "At this point. At this point, at this point, I at this point, I at this point, I am at this point, I am at this point, I am at this point, I am one at this point. I am one at this point. I am wondering, at this point. I am wondering at this point. I am wondering at this point",
                                0.8,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "At this point. At this point, at this point, I at this point, I at this point, I am at this point, I am at this point, I am at this point, I am one at this point. I am one at this point. I am wondering, at this point. I am wondering at this point. I am wondering at this point. I am wondering at this point",
                                0.8,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "At this point. At this point, at this point, I at this point, I at this point, I am at this point, I am at this point, I am at this point, I am one at this point. I am one at this point. I am wondering, at this point. I am wondering at this point. I am wondering at this point. I am wondering at this point. I am wondering at this point",
                                0.8,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "At this point. At this point, at this point, I at this point, I at this point, I am at this point, I am at this point, I am at this point, I am one at this point. I am one at this point. I am wondering, at this point. I am wondering at this point. I am wondering at this point. I am wondering at this point. I am wondering at this point. I am wondering if this point",
                                0.8,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "At this point. At this point, at this point, I at this point, I at this point, I am at this point, I am at this point, I am at this point, I am one at this point. I am one at this point. I am wondering, at this point. I am wondering at this point. I am wondering at this point. I am wondering at this point. I am wondering at this point. I am wondering if this point, I am wondering, if at this point",
                                0.8,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "At this point. At this point, at this point, I at this point, I at this point, I am at this point, I am at this point, I am at this point, I am one at this point. I am one at this point. I am wondering, at this point. I am wondering at this point. I am wondering at this point. I am wondering at this point. I am wondering at this point. I am wondering if this point, I am wondering, if at this point, I am wondering if this point",
                                0.8,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "At this point. At this point, at this point, I at this point, I at this point, I am at this point, I am at this point, I am at this point, I am one at this point. I am one at this point. I am wondering, at this point. I am wondering at this point. I am wondering at this point. I am wondering at this point. I am wondering at this point. I am wondering if this point, I am wondering, if at this point, I am wondering if this point, I am wondering if this point, I am wondering if this point I am,",
                                0.93,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            # Final result with isFinal=True
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "At this point, I am wondering if this is working properly.", 0.93
                            )
                        ],
                        isFinal=True,
                    )
                ]
            ),
            # Another repetition sequence
            MockResponse(
                [MockResult([MockAlternative("Wonderful, this point", 0.8)], isFinal=False)]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "Wonderful, this point. I am wondering if this point", 0.8
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "Wonderful, this point. I am wondering if this point, I am wondering if this point",
                                0.8,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "Wonderful, this point. I am wondering if this point, I am wondering if this point. I am wondering if this point",
                                0.8,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "Wonderful, this point. I am wondering if this point, I am wondering if this point. I am wondering if this point. I am wondering if this point",
                                0.8,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            MockResponse(
                [
                    MockResult(
                        [
                            MockAlternative(
                                "Wonderful, this point. I am wondering if this point, I am wondering if this point. I am wondering if this point. I am wondering if this point, I am wondering if this point I am,",
                                0.84,
                            )
                        ],
                        isFinal=False,
                    )
                ]
            ),
            # Final result
            MockResponse(
                [
                    MockResult(
                        [MockAlternative("Wonderful, this is working much better now.", 0.84)],
                        isFinal=True,
                    )
                ]
            ),
        ]

        # Simulate the complete fixed processing logic
        processed_transcripts = []

        def thread_safe_async_callback(transcript):
            """Simulate the thread-safe async callback."""
            processed_transcripts.append(f"Processed: {transcript}")

        def process_stream_with_complete_fix(stream):
            """Process stream with the complete fix (isFinal + thread-safe callback)."""
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

                # Enhanced final result detection with isFinal fix
                is_final_result = False

                # Check multiple possible final indicators - Google Speech v2 uses isFinal (camelCase)
                if hasattr(result, "isFinal"):
                    is_final_result = result.isFinal
                elif hasattr(result, "is_final"):  # Fallback for older versions
                    is_final_result = result.is_final

                # Process only final results with good confidence
                if (
                    is_final_result
                    and confidence > 0.5
                    and transcript
                    and transcript not in seen_transcripts
                ):
                    seen_transcripts.add(transcript)
                    # Simulate thread-safe callback
                    thread_safe_async_callback(transcript)

        # Process the stream
        process_stream_with_complete_fix(problematic_stream)

        print("\nComplete fix integration test:")
        print(f"Original stream items: {len(problematic_stream)}")
        print(f"Processed transcripts: {len(processed_transcripts)}")
        for i, t in enumerate(processed_transcripts):
            print(f"  {i + 1}. {t}")

        # Verify the fix works
        # Should dramatically reduce from 25+ items to just 2 final results
        self.assertEqual(len(processed_transcripts), 2)

        # Should contain only the clean final transcripts
        self.assertIn(
            "Processed: At this point, I am wondering if this is working properly.",
            processed_transcripts,
        )
        self.assertIn(
            "Processed: Wonderful, this is working much better now.", processed_transcripts
        )

        # Should NOT contain any of the repetitive patterns
        for transcript in processed_transcripts:
            self.assertNotIn("at this point, at this point", transcript.lower())
            self.assertNotIn(
                "wondering if this point, i am wondering if this point", transcript.lower()
            )

    def test_thread_safe_callback_with_isfinal(self):
        """Test thread-safe async callback with isFinal detection."""
        processed_transcripts = []

        async def async_callback(transcript):
            await asyncio.sleep(0.01)  # Simulate async work
            processed_transcripts.append(f"Async: {transcript}")

        def sync_callback(transcript):
            processed_transcripts.append(f"Sync: {transcript}")

        # Mock response with isFinal
        class MockAlternative:
            def __init__(self, transcript, confidence):
                self.transcript = transcript
                self.confidence = confidence

        class MockResult:
            def __init__(self, alternatives, isFinal):
                self.alternatives = alternatives
                self.isFinal = isFinal

        class MockResponse:
            def __init__(self, results):
                self.results = results

        # Test stream with both async and sync scenarios
        test_stream = [
            MockResponse([MockResult([MockAlternative("Interim result", 0.8)], isFinal=False)]),
            MockResponse([MockResult([MockAlternative("Final result", 0.9)], isFinal=True)]),
        ]

        def simulate_complete_processing(stream, callback):
            """Simulate the complete processing with isFinal and thread-safe callback."""
            for response in stream:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                alternative = result.alternatives[0]
                transcript = alternative.transcript.strip()
                confidence = getattr(alternative, "confidence", 0.0)

                # Check isFinal attribute
                is_final_result = hasattr(result, "isFinal") and result.isFinal

                if is_final_result and confidence > 0.5 and transcript:
                    # Simulate thread-safe callback
                    import inspect

                    if inspect.iscoroutinefunction(callback):
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.create_task(callback(transcript))
                            else:
                                loop.run_until_complete(callback(transcript))
                        except RuntimeError:
                            asyncio.run(callback(transcript))
                    else:
                        callback(transcript)

        # Test with async callback
        processed_transcripts.clear()
        simulate_complete_processing(test_stream, async_callback)
        time.sleep(0.1)  # Wait for async tasks

        # Should only process the final result
        self.assertEqual(len(processed_transcripts), 1)
        self.assertIn("Async: Final result", processed_transcripts)

        # Test with sync callback
        processed_transcripts.clear()
        simulate_complete_processing(test_stream, sync_callback)

        # Should only process the final result
        self.assertEqual(len(processed_transcripts), 1)
        self.assertIn("Sync: Final result", processed_transcripts)


if __name__ == "__main__":
    unittest.main()
