import pytest

from src.gateway.utterance_boundary_decider import UtteranceBoundaryDecider


def test_decider_semantic_disabled_does_not_apply_semantic_override(monkeypatch):
    monkeypatch.setenv("UTT_DISABLE_SPACY_SEMANTIC", "1")
    monkeypatch.setenv("UTT_MEDIUM_TIMEOUT_S", "3.0")
    monkeypatch.setenv("UTT_LONG_TIMEOUT_S", "6.0")

    decider = UtteranceBoundaryDecider.from_env()
    assert decider.semantic_checker is None

    # Would be marked incomplete by semantic (ends with determiner), but with semantic disabled it stays default.
    decision = decider.decide(text="what is the", confidence=0.95, speech_ended=False)
    assert decision.reason == "default"


def test_decider_semantic_load_failure_falls_back_to_primary(monkeypatch):
    monkeypatch.setenv("UTT_DISABLE_SPACY_SEMANTIC", "0")
    monkeypatch.setenv("UTT_MEDIUM_TIMEOUT_S", "3.0")
    monkeypatch.setenv("UTT_LONG_TIMEOUT_S", "6.0")

    import src.gateway.utterance_boundary_decider as decider_module

    class _Boom:
        def __init__(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(decider_module, "SpacySemanticChecker", _Boom)

    decider = UtteranceBoundaryDecider.from_env()
    assert decider.semantic_checker is None

    decision = decider.decide(text="what is the", confidence=0.95, speech_ended=False)
    assert decision.reason == "default"


def test_decider_semantic_complete_does_not_accelerate_or_change_reason(monkeypatch):
    monkeypatch.delenv("UTT_DISABLE_SPACY_SEMANTIC", raising=False)
    monkeypatch.setenv("UTT_MEDIUM_TIMEOUT_S", "3.0")
    monkeypatch.setenv("UTT_LONG_TIMEOUT_S", "6.0")
    monkeypatch.setenv("UTT_SEMANTIC_CONFIDENCE_THRESHOLD", "0.0")

    decider = UtteranceBoundaryDecider.from_env()

    # A simple statement should remain default; even if semantic says complete, it must not shorten timeout.
    decision = decider.decide(text="hello there", confidence=0.2, speech_ended=False)
    assert decision.reason == "default"
    assert decision.timeout_s == 3.0


@pytest.mark.parametrize(
    "text",
    [
        "i'm looking for a hammer that will be",
        "i'm looking for a hammer that will",
        "i'm looking for a hammer that",
        "i'm looking for a hammer which",
        "i'm looking for a hammer going to",
        "i'm looking for a hammer is",
    ],
)
def test_decider_marks_dangling_endings_as_incomplete(monkeypatch, text):
    monkeypatch.setenv("UTT_DISABLE_SPACY_SEMANTIC", "1")
    monkeypatch.setenv("UTT_MEDIUM_TIMEOUT_S", "3.0")
    monkeypatch.setenv("UTT_INCOMPLETE_TIMEOUT_S", "4.0")
    monkeypatch.setenv("UTT_LONG_TIMEOUT_S", "6.0")

    decider = UtteranceBoundaryDecider.from_env()

    decision = decider.decide(text=text, confidence=0.9, speech_ended=False)
    assert decision.reason == "incomplete_phrase"
    assert decision.timeout_s == 4.0
