"""
TEST 1: Sentence Splitting for TTS
==================================

CRITICAL FIX: The sentence splitter fails on LLM responses with colons and lists.

EVIDENCE FROM LOGS:
- First tts.synthesize: 10,846 ms (should be ~300ms like others)
- Subsequent chunks: 216-305 ms (correct)

PROBLEM: LLM response starts with:
"For painting the ceiling... I recommend the following interior paint options:"

The colon does NOT trigger a split, so the "first sentence" includes the entire
intro + numbered list until the first period deep in the list.

SOLUTION: Expand sentence boundary detection to include:
- Colons followed by newline (list introductions)
- Numbered list items as separate chunks
- Maximum chunk length enforcement

Run: pytest tests/test_01_sentence_splitting.py -v
"""

import re
from typing import List

import pytest

# ============================================================================
# Current Implementation (for reference - copy from your codebase)
# ============================================================================


def split_into_sentences_current(text: str) -> List[str]:
    """
    CURRENT IMPLEMENTATION - Copy this from src/tts/response_reader.py
    """
    if not text:
        return []

    # Current pattern - only splits on . ! ? followed by space
    sentence_pattern = r"(?<![A-Z][a-z])(?<![A-Z])(?<=\.|\!|\?)\s+"
    sentences = re.split(sentence_pattern, text)
    return [s.strip() for s in sentences if s and s.strip()]


# ============================================================================
# Proposed Implementation
# ============================================================================


def split_into_sentences_improved(text: str, max_chunk_length: int = 200) -> List[str]:
    """
    IMPROVED IMPLEMENTATION - Handles LLM response patterns.

    Improvements:
    1. Split on colon + newline (list introductions)
    2. Split on numbered list items (1. 2. 3.)
    3. Split on bullet points (- *)
    4. Enforce maximum chunk length
    5. Keep original sentence boundary detection
    """
    if not text:
        return []

    chunks = []

    # Step 1: Split on major structural boundaries first
    # - Colon followed by newline (list intro)
    # - Double newline (paragraph break)
    structural_pattern = r"(?<=:)\s*\n|(?<=\.)\s*\n\n"
    major_parts = re.split(structural_pattern, text)

    for part in major_parts:
        if not part or not part.strip():
            continue

        part = part.strip()

        # Step 2: Split on list items within each part
        # Match: "1. ", "2. ", "- ", "* "
        list_pattern = r"(?=\n\s*(?:\d+\.|[-*])\s+)"
        list_items = re.split(list_pattern, part)

        for item in list_items:
            if not item or not item.strip():
                continue

            item = item.strip()

            # Step 3: Split on sentence boundaries within each item
            sentence_pattern = r"(?<=[.!?])\s+"
            sentences = re.split(sentence_pattern, item)

            for sentence in sentences:
                if not sentence or not sentence.strip():
                    continue

                sentence = sentence.strip()

                # Step 4: Enforce maximum chunk length
                if len(sentence) > max_chunk_length:
                    # Split long sentences on comma + space
                    comma_parts = sentence.split(", ")
                    current_chunk = ""

                    for cp in comma_parts:
                        if len(current_chunk) + len(cp) + 2 <= max_chunk_length:
                            if current_chunk:
                                current_chunk += ", " + cp
                            else:
                                current_chunk = cp
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = cp

                    if current_chunk:
                        chunks.append(current_chunk)
                else:
                    chunks.append(sentence)

    return chunks


# ============================================================================
# TESTS: Verify Current Implementation Fails
# ============================================================================


class TestCurrentImplementationProblems:
    """Tests demonstrating problems with current sentence splitter."""

    def test_current_fails_on_colon_intro(self):
        """
        CRITICAL TEST: Current splitter treats colon intro as same sentence.

        This is why first TTS takes 10+ seconds.
        """
        text = """For painting the ceiling of your room with a sky blue color, I recommend the following interior paint options that are suitable for ceilings and provide a nice blue shade reminiscent of the sky:

1. **Rust-Oleum Sure Color Interior Primer and Wall Paint, Sky Blue (SKU: 830Y48)**
   - Color: Sky Blue
   - Base Type: Water-based acrylic"""

        sentences = split_into_sentences_current(text)

        # Current implementation likely returns 1-2 chunks, first one being huge
        first_chunk = sentences[0] if sentences else ""

        print(f"\n[CURRENT] Number of chunks: {len(sentences)}")
        print(f"[CURRENT] First chunk length: {len(first_chunk)} chars")
        print(f"[CURRENT] First chunk preview: {first_chunk[:100]}...")

        # This test DOCUMENTS the problem - it will likely fail
        # showing that first chunk is too large
        assert len(first_chunk) < 200, (
            f"First chunk is {len(first_chunk)} chars - WAY too large for fast TTS. "
            f"Should be <200 chars for ~300ms synthesis time."
        )

    def test_current_fails_on_numbered_list(self):
        """Current splitter doesn't recognize numbered list items."""
        text = """Here are three options:

1. First option here with details.
2. Second option here with more details.
3. Third option here with final details."""

        sentences = split_into_sentences_current(text)

        print(f"\n[CURRENT] Chunks from numbered list: {len(sentences)}")
        for i, s in enumerate(sentences):
            print(f"  [{i}]: {s[:50]}...")

        # Should be at least 4 chunks (intro + 3 list items)
        # Current implementation may only get 3 (splits on periods within list)
        assert len(sentences) >= 4, (
            f"Should split into intro + 3 list items (4+ chunks), got {len(sentences)}"
        )


# ============================================================================
# TESTS: Verify Improved Implementation Works
# ============================================================================


class TestImprovedImplementation:
    """Tests for the improved sentence splitter."""

    def test_splits_colon_intro_from_list(self):
        """
        CRITICAL: Colon + newline should create a split point.
        """
        text = """I recommend the following options:

1. First option here.
2. Second option here."""

        chunks = split_into_sentences_improved(text)

        print(f"\n[IMPROVED] Chunks: {len(chunks)}")
        for i, c in enumerate(chunks):
            print(f"  [{i}]: {c[:60]}...")

        # Should be: intro, item 1, item 2 = at least 3 chunks
        assert len(chunks) >= 3, f"Expected 3+ chunks, got {len(chunks)}"

        # First chunk should be short (just the intro)
        assert len(chunks[0]) < 100, (
            f"First chunk should be intro only (<100 chars), got {len(chunks[0])}: {chunks[0]}"
        )

    def test_splits_numbered_list_items(self):
        """Each numbered list item should be a separate chunk."""
        text = """Here are options:

1. **Product A (SKU: 123)** - Great for indoor use.
2. **Product B (SKU: 456)** - Better for outdoor use.
3. **Product C (SKU: 789)** - Best for both."""

        chunks = split_into_sentences_improved(text)

        # Should have intro + 3 items = 4 chunks minimum
        assert len(chunks) >= 4, f"Expected 4+ chunks, got {len(chunks)}: {chunks}"

    def test_splits_bullet_list_items(self):
        """Bullet points should create splits."""
        text = """Features include:
- Easy to apply
- No primer needed
- Washable finish"""

        chunks = split_into_sentences_improved(text)

        assert len(chunks) >= 3, f"Expected 3+ chunks for bullet list, got {len(chunks)}"

    def test_enforces_max_chunk_length(self):
        """No chunk should exceed max length."""
        text = """This is an extremely long sentence that goes on and on with lots of details about various products including their specifications, features, benefits, use cases, pricing information, availability, and compatibility with other products in the catalog."""

        chunks = split_into_sentences_improved(text, max_chunk_length=100)

        for i, chunk in enumerate(chunks):
            assert len(chunk) <= 150, (  # Allow some flexibility
                f"Chunk {i} is {len(chunk)} chars, exceeds limit: {chunk[:50]}..."
            )

    def test_handles_real_llm_response(self):
        """Test with actual LLM response from logs."""
        text = """For painting the ceiling of your room with a sky blue color, I recommend the following interior paint options that are suitable for ceilings and provide a nice blue shade reminiscent of the sky:

1. **Rust-Oleum Sure Color Interior Primer and Wall Paint, Sky Blue (SKU: 830Y48)**
   - Color: Sky Blue
   - Base Type: Water-based acrylic
   - Sheen: Eggshell (provides a soft, subtle finish)
   - Features: No primer required, scrubbable, washable
   - Suitable for interior walls and ceilings
   - Container Size: 1 gallon

2. **Krylon COLORmaxx Interior and Exterior Paint, Regal Blue (SKU: 833ZT4)**
   - Color: Regal Blue (a rich blue shade)
   - Base Type: Water-based acrylic
   - Sheen: Gloss (more reflective finish)
   - Features: No primer required, chemical-resistant
   - Suitable for interior and exterior use
   - Container Size: 1 quart

If you want a softer, more matte finish typical for ceilings, the Rust-Oleum Sky Blue with eggshell sheen would be a great choice.

Let me know if you want help with tools or other paint colors!"""

        chunks = split_into_sentences_improved(text)

        print(f"\n[REAL LLM RESPONSE] Split into {len(chunks)} chunks:")
        for i, c in enumerate(chunks):
            # Estimate TTS time: ~3ms per character for Google TTS
            est_tts_ms = len(c) * 3
            print(f"  [{i}] ({len(c)} chars, ~{est_tts_ms}ms TTS): {c[:60]}...")

        # First chunk should be speakable in < 500ms (~150 chars)
        assert len(chunks[0]) < 200, (
            f"First chunk is {len(chunks[0])} chars - too large. "
            f"Estimated TTS time: {len(chunks[0]) * 3}ms"
        )

        # Should have many chunks (intro + products + conclusion)
        assert len(chunks) >= 5, f"Expected 5+ chunks from this response, got {len(chunks)}"

    def test_preserves_simple_sentences(self):
        """Simple sentence-based text should still work."""
        text = "First sentence here. Second sentence here. Third sentence here."

        chunks = split_into_sentences_improved(text)

        assert len(chunks) == 3, f"Expected 3 sentences, got {len(chunks)}: {chunks}"

    def test_handles_empty_input(self):
        """Empty input should return empty list."""
        assert split_into_sentences_improved("") == []
        assert split_into_sentences_improved(None) == []
        assert split_into_sentences_improved("   ") == []


# ============================================================================
# TESTS: TTS Timing Estimates
# ============================================================================


class TestTTSTimingEstimates:
    """Tests that verify chunks will synthesize quickly."""

    # Based on observed data: ~300ms for typical sentences
    # Google TTS roughly: 2-4ms per character for Neural2 voices
    MS_PER_CHAR = 3
    TARGET_FIRST_CHUNK_MS = 500  # Target: first audio in <500ms

    def test_first_chunk_synthesizes_quickly(self):
        """First chunk should synthesize in target time."""
        text = """I recommend the following products for your project:

1. Product A - Great quality.
2. Product B - Good value."""

        chunks = split_into_sentences_improved(text)

        first_chunk_chars = len(chunks[0])
        estimated_ms = first_chunk_chars * self.MS_PER_CHAR

        assert estimated_ms < self.TARGET_FIRST_CHUNK_MS, (
            f"First chunk ({first_chunk_chars} chars) estimated at {estimated_ms}ms, "
            f"exceeds target of {self.TARGET_FIRST_CHUNK_MS}ms"
        )

    def test_all_chunks_reasonable_size(self):
        """All chunks should be reasonable size for TTS."""
        text = """For painting the ceiling of your room with a sky blue color, I recommend the following interior paint options:

1. **Rust-Oleum Sure Color Interior Primer** - Great coverage.
2. **Krylon COLORmaxx Paint** - Easy application.

Let me know if you need more options!"""

        chunks = split_into_sentences_improved(text)

        max_reasonable_chars = 300  # ~900ms TTS, still acceptable

        for i, chunk in enumerate(chunks):
            assert len(chunk) <= max_reasonable_chars, (
                f"Chunk {i} is {len(chunk)} chars (~{len(chunk) * self.MS_PER_CHAR}ms), "
                f"too large. Content: {chunk[:50]}..."
            )


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
