import asyncio
import time

import pytest

from src.asr.base import TranscriptEvent
from src.gateway.semantic_checker import SpacySemanticChecker
from src.gateway.utterance_manager import UtteranceManager


@pytest.mark.parametrize(
    "text, expected_complete, expected_reason",
    [
        ("what is the weather", False, "incomplete_noun_phrase"),
        ("what is the weather today", True, "syntactically_complete"),
        ("turn on the", False, "ends_with_determiner"),
        ("turn on the lights", True, "complete_command"),
        ("find me the data for", False, "ends_with_preposition"),
        ("find me the Q3 sales data", True, "complete_command"),
    ],
)
def test_spacy_semantic_checker_cases(text, expected_complete, expected_reason):
    checker = SpacySemanticChecker()
    result = checker.is_complete(text)

    assert result.is_complete is expected_complete
    assert result.reason == expected_reason
    assert result.processing_time_ms >= 0.0


@pytest.mark.asyncio
async def test_utterance_manager_semantic_layer_can_delay_finalization(monkeypatch):
    monkeypatch.setenv("UTT_SHORT_TIMEOUT_S", "0.01")
    monkeypatch.setenv("UTT_MEDIUM_TIMEOUT_S", "0.02")
    monkeypatch.setenv("UTT_LONG_TIMEOUT_S", "0.2")
    monkeypatch.setenv("UTT_INCOMPLETE_TIMEOUT_S", "0.06")

    monkeypatch.delenv("UTT_DISABLE_SPACY_SEMANTIC", raising=False)
    monkeypatch.setenv("UTT_SEMANTIC_CONFIDENCE_THRESHOLD", "0.85")

    finals = []

    async def on_partial(_: str):
        return

    async def on_final(t: str, confidence: float, reason: str):
        finals.append((t, confidence, reason))

    async def on_state(_: str):
        return

    async def on_interrupt():
        return

    mgr = UtteranceManager(
        on_partial=on_partial, on_final=on_final, on_state=on_state, on_interrupt=on_interrupt
    )

    e = TranscriptEvent(
        text="what is the",
        is_final=True,
        confidence=0.95,
        stability=0.0,
        received_time=time.time(),
    )
    await mgr.on_transcript_event(e, is_interim_channel=False)

    # Without semantic layer, this would typically finalize via the high-conf question path.
    # With semantic layer enabled, it should be delayed as an incomplete question.
    await asyncio.sleep(0.03)
    assert finals == []

    await asyncio.sleep(0.05)
    assert len(finals) == 1
    assert finals[0][0] == "what is the"
    assert finals[0][2].startswith("semantic_")
