"""
Unit tests to ensure assistant responses are not appended to the transcript UI.
This mirrors the intended client behavior: user speech (partial/final) appears
in the transcript box, while LLM output appears in a dedicated response box.
"""

import unittest


class MockUI:
    """Lightweight mock of the browser UI state."""

    def __init__(self):
        self.transcript = []  # list of strings representing lines in transcript box
        self.llm_response = None  # text contents of LLM response box
        self._interim = None

    def handle_message(self, msg):
        """Simulate the JS onmessage handler with only the relevant cases."""
        mtype = msg.get("type")
        if mtype == "partial_transcript":
            # Replace/track interim text; don't permanently add to transcript
            self._interim = msg.get("text", "")
        elif mtype == "final_transcript":
            # Commit the final text to the transcript box
            self._interim = None
            text = msg.get("text", "")
            if self.transcript and self._should_replace_last_final(self.transcript[-1], text):
                self.transcript[-1] = text
            else:
                self.transcript.append(text)
        elif mtype == "assistant_response":
            # Must render into LLM response box, not transcript
            self.llm_response = msg.get("text", "")
        elif mtype == "trace_completed":
            # Later updates can also refresh the llm response box
            if "response" in msg and msg["response"]:
                self.llm_response = msg["response"]

    @staticmethod
    def _should_replace_last_final(previous: str, current: str) -> bool:
        # Heuristic mirrors JS: replace if new final extends previous
        return bool(previous) and current.startswith(previous) and len(current) > len(previous)


class TestAssistantTranscriptSeparation(unittest.TestCase):
    def test_assistant_not_in_transcript(self):
        ui = MockUI()

        # Simulate a turn with interim, final, and assistant response
        ui.handle_message({"type": "partial_transcript", "text": "I like"})
        ui.handle_message({"type": "partial_transcript", "text": "I like to"})
        ui.handle_message({"type": "final_transcript", "text": "I like to play soccer"})

        # Assistant response should be separated
        assistant_text = "Great choice! Soccer is a fantastic sport."
        ui.handle_message({"type": "assistant_response", "text": assistant_text})

        # Optional later trace completion refresh
        ui.handle_message({"type": "trace_completed", "response": assistant_text})

        # Assertions: transcript has only user final text
        self.assertEqual(ui.transcript, ["I like to play soccer"])  # No assistant text
        # Assertions: assistant text is present in its own box
        self.assertEqual(ui.llm_response, assistant_text)


if __name__ == "__main__":
    unittest.main()
