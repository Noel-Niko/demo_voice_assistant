import asyncio
import time

import pytest

from src.asr.base import TranscriptEvent
from src.gateway.utterance_manager import UtteranceManager


@pytest.mark.asyncio
async def test_high_conf_question_finalizes_quickly(monkeypatch):
    monkeypatch.setenv("UTT_SHORT_TIMEOUT_S", "0.01")
    monkeypatch.setenv("UTT_MEDIUM_TIMEOUT_S", "0.05")
    monkeypatch.setenv("UTT_LONG_TIMEOUT_S", "0.2")

    partials = []
    finals = []
    states = []
    interrupts = 0

    async def on_partial(t: str):
        partials.append(t)

    async def on_final(t: str, confidence: float, reason: str):
        finals.append((t, confidence, reason))

    async def on_state(s: str):
        states.append(s)

    async def on_interrupt():
        nonlocal interrupts
        interrupts += 1

    mgr = UtteranceManager(
        on_partial=on_partial, on_final=on_final, on_state=on_state, on_interrupt=on_interrupt
    )

    e = TranscriptEvent(
        text="what time is it",
        is_final=True,
        confidence=0.95,
        stability=0.0,
        received_time=time.time(),
    )
    await mgr.on_transcript_event(e, is_interim_channel=False)

    await asyncio.sleep(0.03)

    assert len(finals) == 1
    assert finals[0][0] == "what time is it"
    assert finals[0][2] in {"high_conf_question", "default", "speech_end_good_conf"}


@pytest.mark.asyncio
async def test_incomplete_phrase_waits_longer(monkeypatch):
    monkeypatch.setenv("UTT_SHORT_TIMEOUT_S", "0.01")
    monkeypatch.setenv("UTT_MEDIUM_TIMEOUT_S", "0.02")
    monkeypatch.setenv("UTT_LONG_TIMEOUT_S", "0.2")
    monkeypatch.setenv("UTT_INCOMPLETE_TIMEOUT_S", "0.06")

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
        text="find me the latest sales numbers for",
        is_final=True,
        confidence=0.9,
        stability=0.0,
        received_time=time.time(),
    )
    await mgr.on_transcript_event(e, is_interim_channel=False)

    await asyncio.sleep(0.03)
    assert finals == []

    await asyncio.sleep(0.05)
    assert len(finals) == 1
    assert finals[0][2] == "incomplete_phrase"


@pytest.mark.asyncio
async def test_speech_end_event_schedules_finalization(monkeypatch):
    monkeypatch.setenv("UTT_MEDIUM_TIMEOUT_S", "0.03")
    monkeypatch.setenv("UTT_LONG_TIMEOUT_S", "0.2")

    finals = []
    states = []

    async def on_partial(_: str):
        return

    async def on_final(t: str, confidence: float, reason: str):
        finals.append((t, confidence, reason))

    async def on_state(s: str):
        states.append(s)

    async def on_interrupt():
        return

    mgr = UtteranceManager(
        on_partial=on_partial, on_final=on_final, on_state=on_state, on_interrupt=on_interrupt
    )

    # Receive interim transcript text
    e1 = TranscriptEvent(
        text="hello there",
        is_final=False,
        confidence=0.8,
        stability=0.0,
        received_time=time.time(),
    )
    await mgr.on_transcript_event(e1, is_interim_channel=True)

    # Speech end event with no text
    e2 = TranscriptEvent(
        text="",
        is_final=False,
        confidence=0.0,
        stability=0.0,
        received_time=time.time(),
        speech_event_type="SPEECH_ACTIVITY_END",
    )
    await mgr.on_transcript_event(e2, is_interim_channel=True)

    await asyncio.sleep(0.06)
    assert len(finals) == 1
    assert finals[0][0] == "hello there"
    assert "processing" in states


@pytest.mark.asyncio
async def test_interrupt_cancels_in_flight_action(monkeypatch):
    monkeypatch.setenv("UTT_MEDIUM_TIMEOUT_S", "0.05")
    monkeypatch.setenv("UTT_LONG_TIMEOUT_S", "0.2")

    interrupts = 0

    async def on_partial(_: str):
        return

    async def on_final(_: str, __: float, ___: str):
        return

    async def on_state(_: str):
        return

    async def on_interrupt():
        nonlocal interrupts
        interrupts += 1

    mgr = UtteranceManager(
        on_partial=on_partial, on_final=on_final, on_state=on_state, on_interrupt=on_interrupt
    )

    async def long_task():
        await asyncio.sleep(10)

    task = asyncio.create_task(long_task())
    mgr.set_in_flight_action_task(task)

    # A speech start event while action is in flight should interrupt/cancel.
    e = TranscriptEvent(
        text="",
        is_final=False,
        confidence=0.0,
        stability=0.0,
        received_time=time.time(),
        speech_event_type="SPEECH_ACTIVITY_START",
    )
    await mgr.on_transcript_event(e, is_interim_channel=True)

    await asyncio.sleep(0)

    assert interrupts == 1
    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_speech_start_cancels_scheduled_finalization(monkeypatch):
    monkeypatch.setenv("UTT_MEDIUM_TIMEOUT_S", "0.06")
    monkeypatch.setenv("UTT_LONG_TIMEOUT_S", "0.2")

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

    # Schedule finalization based on a final transcript.
    e_final = TranscriptEvent(
        text="hello there",
        is_final=True,
        confidence=0.8,
        stability=0.0,
        received_time=time.time(),
    )
    await mgr.on_transcript_event(e_final, is_interim_channel=False)

    # Before the timeout fires, user starts speaking again.
    e_start = TranscriptEvent(
        text="",
        is_final=False,
        confidence=0.0,
        stability=0.0,
        received_time=time.time(),
        speech_event_type="SPEECH_ACTIVITY_START",
    )
    await mgr.on_transcript_event(e_start, is_interim_channel=True)

    await asyncio.sleep(0.08)
    assert finals == []


@pytest.mark.asyncio
async def test_manager_accumulates_segmented_final_chunks(monkeypatch):
    monkeypatch.setenv("UTT_MEDIUM_TIMEOUT_S", "0.01")
    monkeypatch.setenv("UTT_LONG_TIMEOUT_S", "0.2")
    monkeypatch.setenv("UTT_DISABLE_SPACY_SEMANTIC", "1")
    # Ensure high-confidence complete statements finalize quickly
    monkeypatch.setenv("UTT_SHORT_TIMEOUT_S", "0.01")

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

    now = time.time()

    await mgr.on_transcript_event(
        TranscriptEvent(
            text="I am looking for a hammer that will work",
            is_final=True,
            confidence=0.9,
            stability=0.0,
            received_time=now,
        ),
        is_interim_channel=False,
    )
    await mgr.on_transcript_event(
        TranscriptEvent(
            text="for wood",
            is_final=True,
            confidence=0.9,
            stability=0.0,
            received_time=now + 0.01,
        ),
        is_interim_channel=False,
    )
    await mgr.on_transcript_event(
        TranscriptEvent(
            text="roofing nails.",
            is_final=True,
            confidence=0.9,
            stability=0.0,
            received_time=now + 0.02,
        ),
        is_interim_channel=False,
    )

    await asyncio.sleep(0.05)

    assert len(finals) == 1
    assert finals[0][0] == "I am looking for a hammer that will work for wood roofing nails."
