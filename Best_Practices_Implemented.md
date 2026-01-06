# Best practices implemented (Google Speech-to-Text v2 + Agentic AI System)

## Table of Contents
- [Implemented STT Best Practices](#implemented-stt-best-practices)
- [Recommended improvements (gaps vs docs)](#recommended-improvements-gaps-vs-docs)
- [Best Practices: Text-to-Speech Latency Optimization](#best-practices-text-to-speech-latency-optimization)
  - [Overview](#overview)
  - [Implemented TTS Best Practices](#implemented-tts-best-practices)
  - [Performance Metrics](#performance-metrics)
  - [Testing Strategy](#testing-strategy)
  - [Configuration](#configuration)
  - [Troubleshooting](#troubleshooting)
  - [References](#references)

This document lists the **Google-recommended best practices** that are already implemented in this repo, with **specific code examples**, and a short list of **recommended improvements**.

Sources referenced:
- https://docs.cloud.google.com/speech-to-text/docs/best-practices
- https://docs.cloud.google.com/apis/docs/client-libraries-explained
- https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/speech/snippets
- OpenAI API best practices
- Model Context Protocol (MCP) specifications

---

## Implemented STT Best Practices 

### 1) Use 16 kHz sampling rate where possible

**Recommendation (docs):** Prefer 16000 Hz audio when possible.

**Source:** https://docs.cloud.google.com/speech-to-text/docs/best-practices

**Applied in this repo:**

- `src/asr/google_speech_v2.py` (Speech v2 recognition config uses 16 kHz)

```python
config = cloud_speech.RecognitionConfig(
    explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
        encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        audio_channel_count=1,
    ),
    language_codes=["en-US"],
    model=self.model_name,
)
```

- `src/gateway/webrtc_server.py` (WebRTC ingestion expects 16 kHz)

```python
self.sample_rate = 16000  # Expected sample rate
```

- `src/gateway/webrtc_server.py` (resamples non-16k input to 16k)

```python
if sample_rate != 16000:
    ratio = 16000 / sample_rate
    new_length = int(len(audio_array) * ratio)
    indices = np.linspace(0, len(audio_array) - 1, new_length)
    audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)
```

---

### 2) Use ~100ms frames for streaming

**Recommendation (docs):** A ~100ms frame size is a good latency/efficiency tradeoff.

**Source:** https://docs.cloud.google.com/speech-to-text/docs/best-practices

**Applied in this repo:**

- `src/asr/google_speech_v2.py` (16 kHz * 100ms = 1600 samples)

```python
FRAME_SAMPLES = 1600
```

- `src/asr/google_speech_v2.py` (request generator splits buffered audio into frames)

```python
while len(audio_buffer) >= self.FRAME_SAMPLES:
    frame = audio_buffer[: self.FRAME_SAMPLES]
    del audio_buffer[: self.FRAME_SAMPLES]
    ...
    yield cloud_speech.StreamingRecognizeRequest(audio=frame_bytes)
```

---

### 3) Accurately describe audio with explicit decoding config

**Recommendation (docs):** Ensure `RecognitionConfig` matches the actual audio encoding, sample rate, channel count.

**Source:** https://docs.cloud.google.com/speech-to-text/docs/best-practices

**Applied in this repo:**

- `src/asr/google_speech_v2.py`

```python
explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    audio_channel_count=1,
)
```

---

### 4) Configure language codes and model explicitly

**Recommendation (docs):** Set `language_codes` and an appropriate `model` for accuracy/billing.

**Source:** https://docs.cloud.google.com/speech-to-text/docs/best-practices

**Applied in this repo:**

- `src/asr/google_speech_v2.py`

```python
language_codes=["en-US"],
model=self.model_name,
```

---

### 5) Use the Cloud Client Library (instead of your own REST client)

**Recommendation (docs):** Use Cloud Client Libraries where available (idiomatic API, auth handling, gRPC benefits).

**Source:** https://docs.cloud.google.com/apis/docs/client-libraries-explained

**Applied in this repo:**

- `src/asr/google_speech_v2.py`

```python
from google.cloud.speech_v2 import SpeechClient
...
self.client = SpeechClient()
```

---

### 6) Use application default credentials / service account via env when available

**Recommendation (docs):** Let the client library handle authentication.

**Source:** https://docs.cloud.google.com/apis/docs/client-libraries-explained

**Applied in this repo:**

- `src/asr/google_speech_v2.py`

```python
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if credentials_path:
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    self.client = SpeechClient(credentials=credentials)
else:
    self.client = SpeechClient()
```

---

### 7) Enable interim results for real-time UX

**Recommendation (common streaming pattern):** Enable interim results for live updates.

**Source:** https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_streaming_v2.py

**Applied in this repo:**

- `src/asr/google_speech_v2.py`

```python
streaming_features = cloud_speech.StreamingRecognitionFeatures(
    interim_results=True,
    enable_voice_activity_events=True,
    voice_activity_timeout=voice_activity_timeout,
)
```

---

### 7b) Ensure complete answers within token budgets

**Recommendation (OpenAI best practices):** Avoid mid‑sentence truncation under strict token caps by prompting the model to self‑budget and finish sentences.

**Applied in this repo:**

- System instruction injection (dynamic): Before each LLM call, we inject a system message that says, “You must return no more than N tokens. If your answer would exceed this, compress and summarize so it fits. Always end with a complete sentence and do not return partial sentences.” The value N is taken from the active `OPENAI_MAX_TOKENS`.
- Low‑latency approach: We intentionally avoid additional rewrite passes to minimize latency. For stricter guarantees, consider setting a slightly smaller cap (e.g., 140 for a 150‑token budget) to reduce the chance of hitting the hard limit.

Code references:
- System instruction injection: `src/llm/langgraph_workflow.py` (generate_response_node)

---

### 8) Use voice activity events/timeouts to segment streaming speech

**Recommendation (docs/pattern):** Enable voice activity events/timeouts to detect speech start/end.

**Source:**
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_streaming_voice_activity_events.py
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_streaming_voice_activity_timeouts.py

**Applied in this repo:**

- `src/asr/google_speech_v2.py`

```python
voice_activity_timeout = cloud_speech.StreamingRecognitionFeatures.VoiceActivityTimeout(
    speech_start_timeout=duration_pb2.Duration(seconds=start_timeout_s),
    speech_end_timeout=duration_pb2.Duration(seconds=end_timeout_s),
)
...
enable_voice_activity_events=True,
voice_activity_timeout=voice_activity_timeout,
```

---

### 9) Keepalive frames to keep the stream healthy during brief stalls

**Recommendation (operational):** Ensure the stream doesn’t stall when audio pauses.

**Source:** https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_streaming_v2.py

**Applied in this repo:**

- `src/asr/google_speech_v2.py`

```python
KEEPALIVE_INTERVAL_S = 1.0
silence_frame = (np.zeros(self.FRAME_SAMPLES, dtype=np.int16)).tobytes()
...
if (time.time() - last_keepalive) >= self.KEEPALIVE_INTERVAL_S:
    yield cloud_speech.StreamingRecognizeRequest(audio=silence_frame)
```

---

### 10) Enable automatic punctuation for cleaner downstream LLM prompts

**Why it matters for STT → LLM → MCP:** Cleaner punctuation reduces ambiguity and improves tool-routing prompts.

**Source:** https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_auto_punctuation_v2.py

**Applied in this repo:**

- `src/asr/google_speech_v2.py`

```python
features=cloud_speech.RecognitionFeatures(
    enable_automatic_punctuation=True,
    enable_spoken_punctuation=True,
    ...
)
```

**Google snippet reference:**
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_auto_punctuation_v2.py

---

### 11) Enable word confidence and word time offsets (for richer telemetry)

**Why it matters for STT → LLM → MCP:**
- Word confidence can gate tool-calling (avoid acting on low-confidence transcripts).
- Word offsets can support highlighting, alignment, and better “finalization” heuristics.

**Source:**
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_word_level_confidence_v2.py
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_word_time_offsets_v2.py

**Applied in this repo (enabled in config):**

- `src/asr/google_speech_v2.py`

```python
features=cloud_speech.RecognitionFeatures(
    enable_word_time_offsets=True,
    enable_word_confidence=True,
    ...
)
```

**Google snippet references:**
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_word_level_confidence_v2.py
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_word_time_offsets_v2.py

---

### 12) Debounce/aggregate “final” candidates before triggering downstream work

**Why it matters for STT → LLM → MCP:** Streaming often emits multiple refined hypotheses; triggering the LLM/tooling on every “final” candidate causes duplication and tool spam.

**Source:** Repo implementation (`src/gateway/webrtc_server.py`, `src/gateway/utterance_manager.py`)

**Applied in this repo:**

- `src/gateway/webrtc_server.py` (routes interim + final-candidate updates into the utterance boundary manager)
- `src/gateway/utterance_manager.py` (aggregates updates and finalizes exactly one utterance at the appropriate time)

```python
# Start streaming with rich events enabled so we can make better utterance-boundary decisions.
self.asr.start_streaming(
    transcription_callback=self._on_final_candidate_transcript,
    interim_callback=self._on_interim_transcript,
    emit_events=True,
)

# Both interim and final-candidate updates are fed into UtteranceManager.
await self._utterance_manager.on_transcript_event(event, is_interim_channel=is_interim_channel)
```

**Final-segment aggregation:** Google Speech-to-Text streaming can emit final transcripts as non-cumulative segments (e.g., only the latest phrase fragment). To prevent losing earlier parts of an utterance, `UtteranceManager` implements intelligent transcript aggregation:

```python
# UtteranceManager maintains an accumulated utterance buffer
self._accumulated_text: str = ""

# Merges incoming final segments using overlap detection
def _merge_transcript_text(existing: str, incoming: str) -> str:
    # Handles prefix/suffix relationships
    if incoming.startswith(existing):
        return incoming
    # Detects token-level overlaps and merges intelligently
    a_tokens = existing.split()
    b_tokens = incoming.split()
    for k in range(max_overlap, 0, -1):
        if a_tokens[-k:] == b_tokens[:k]:
            merged = a_tokens + b_tokens[k:]
            return " ".join(merged).strip()
    # Fallback: concatenate with space
    return f"{existing} {incoming}".strip()
```

This ensures that segmented finals like `["I am looking for a hammer", "for wood", "roofing nails."]` are correctly merged into a single complete utterance: `"I am looking for a hammer for wood roofing nails."` before being sent to the LLM.

---

### 13) Hybrid utterance boundary handling (confidence + adaptive timeouts + voice activity events)

**Why it matters for STT → LLM → MCP:**
- Users pause mid-thought.
- Triggering tool calls too early creates incorrect searches/orders.
- Waiting too long feels unresponsive.

**Source (production pattern):**
- Google voice activity events/timeouts docs: https://docs.cloud.google.com/speech-to-text/docs/voice-activity-events

**Applied in this repo (implementation):**

- `src/asr/base.py` defines the structured transcript payload used for decisioning:

```python
@dataclass(frozen=True)
class TranscriptEvent:
    text: str
    is_final: bool
    confidence: float
    stability: float
    received_time: float
    speech_event_type: Optional[str] = None
```

- `src/asr/google_speech_v2.py` can emit `TranscriptEvent` objects into callbacks when `emit_events=True`. It also forwards voice activity events via `speech_event_type`.

- `src/gateway/utterance_manager.py` implements layered decision-making:
  - orchestrates transcript updates, scheduling/cancellation, and interruption handling

- `src/gateway/utterance_boundary_decider.py` implements layered decision-making:
  - multi-tier timeouts (short/medium/long)
  - confidence gating
  - heuristic completeness detection (questions vs dangling prepositions/conjunctions)
  - optional syntactic completeness analysis (spaCy dependency parsing) for edge cases
  - hard max timeout

- `src/gateway/webrtc_server.py` integrates `UtteranceManager` into the WebRTC session and emits explicit state transitions to the browser UI.

**Applied in this repo (secondary semantic layer):**

- `src/gateway/semantic_checker.py` implements `SpacySemanticChecker`, which runs fast dependency parsing (no training data) to determine if an utterance is syntactically complete.
- `src/gateway/utterance_manager.py` can enable this layer via env vars. When enabled, it can:
  - delay finalization for incomplete questions/phrases even if ASR confidence is high
  - accelerate finalization after speech end when syntactic completeness is high

Semantic layer knobs:
- `UTT_DISABLE_SPACY_SEMANTIC` (default `0`)
- `UTT_SEMANTIC_CONFIDENCE_THRESHOLD` (default `0.85`)

---

## Recommended improvements (gaps vs docs)

### 1) Avoid resampling when possible (or upgrade resampling quality)

**Why:** The best-practices doc recommends matching the native sample rate when possible rather than resampling.

**Source:** https://docs.cloud.google.com/speech-to-text/docs/best-practices

**Current code:** `src/gateway/webrtc_server.py` resamples via `np.interp`.

**Recommended improvement:**
- Prefer having the client send 16k PCM directly, or
- use a higher-quality resampler (e.g., `soxr` or `samplerate`) if server-side conversion is required.

---

### 2) Re-evaluate noise reduction / gating before Google STT

**Why:** The best-practices doc warns that noise reduction processing can reduce recognition accuracy.

**Source:** https://docs.cloud.google.com/speech-to-text/docs/best-practices

**Current code:** `src/asr/audio_processor.py` contains a full preprocessing chain (noise gate, band-pass, de-essing, etc.).

**Recommended improvement:**
- Confirm whether this preprocessing is applied before sending audio to Speech v2.
- If it is, consider disabling noise reduction for Speech v2 (or make it optional via config).

---

### 3) If word offsets are enabled, consider parsing and using them

**Why:** You enable `enable_word_time_offsets=True` but the app currently derives timing from transcript evolution.

**Source:** https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_word_time_offsets_v2.py

**Recommended improvement:**
- Extract per-word offsets from the Speech v2 responses and expose them in metrics/UI if you want true word-level timing.

---

### 4) Centralize “audio format contract” between WebRTC ingestion and Speech v2

**Why:** Accuracy depends on making sure the stream truly matches declared encoding/sample rate.

**Source:** https://docs.cloud.google.com/speech-to-text/docs/best-practices

**Recommended improvement:**
- Add explicit assertions/logging for input audio sample rate/channel count before feeding Speech v2.
- Consider standardizing to a single internal format: mono, 16kHz, int16 PCM.

---

### 5) Prefer AutoDetectDecodingConfig for file/header codecs; keep explicit decoding for raw PCM

**Why:** Google snippets frequently use `auto_decoding_config` for file-based inputs with headers; explicit decoding is appropriate for raw/"headerless" streams.

**Source:**
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/quickstart_v2.py
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_streaming_v2.py

**Google snippet reference (v2 file):**
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/quickstart_v2.py

**Google snippet reference (v2 streaming):**
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_streaming_v2.py

---

### 6) Use Recognizer resources intentionally (create/reuse/override)

**Why it matters:** Recognizers let you store defaults (language/model/features) centrally and reuse them across sessions; you can also override per-request settings via a field mask.

**Source:**
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/create_recognizer.py
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_reuse_recognizer.py
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_override_recognizer.py
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_feature_in_recognizer.py

**Google snippet references:**
- Create recognizer: https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/create_recognizer.py
- Reuse recognizer: https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_reuse_recognizer.py
- Override recognizer settings with `FieldMask`: https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_override_recognizer.py
- Create-if-missing and feature-in-recognizer pattern: https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_feature_in_recognizer.py

**Recommended improvement in this repo:**
- Stop using the default `_` recognizer in production and create a named recognizer per environment (dev/stage/prod) with your canonical defaults.

---

### 7) Add Speech Adaptation for domain terms (tool names, product SKUs, internal jargon)

**Why it matters for STT → LLM → MCP:** If tool names/product identifiers are misrecognized, the LLM will call the wrong MCP tool or fail to call any.

**Source:**
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/adaptation_v2_inline_phrase_set.py
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/adaptation_v2_inline_custom_class.py

**Google snippet references:**
- Inline phrase set: https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/adaptation_v2_inline_phrase_set.py
- Inline custom class: https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/adaptation_v2_inline_custom_class.py

**Recommended improvement in this repo:**
- Maintain a phrase set/custom class for:
  - MCP tool names
  - common command verbs ("create", "update", "search", "open")
  - product/part numbers

---

### 8) Support regional endpoints + location-aware recognizer paths (latency / data residency)

**Why:** Helps meet residency requirements and can reduce latency.

**Source:** https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/change_speech_v2_location.py

**Google snippet reference:**
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/change_speech_v2_location.py

**Recommended improvement in this repo:**
- Make Speech endpoint location configurable (env var), and construct recognizer paths using that location.

---

### 9) Add batch transcription flows for long recordings / async pipelines

**Why:** For non-live workflows (uploads, call recordings), batch recognize is more appropriate and can use dynamic batching.

**Source:**
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_gcs_v2.py
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_batch_dynamic_batching_v2.py

**Google snippet references:**
- Transcribe from GCS URI: https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_gcs_v2.py
- Batch recognize with dynamic batching: https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_batch_dynamic_batching_v2.py

---

### 10) Consider multichannel mode if you need speaker separation by channel

**Why:** If you capture separate audio channels (agent/customer), Speech can run separate recognition per channel.

**Source:** https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_multichannel_v2.py

**Google snippet reference:**
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/transcribe_multichannel_v2.py

---

### 11) Consider CMEK for compliance-sensitive deployments

**Why:** Allows customer-managed encryption keys for Speech configuration.

**Source:** https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/enable_cmek.py

**Google snippet reference:**
- https://raw.githubusercontent.com/GoogleCloudPlatform/python-docs-samples/main/speech/snippets/enable_cmek.py


# Best Practices: Text-to-Speech Latency Optimization

This document describes the TTS latency optimizations implemented in this voice assistant, including the problem analysis, solution architecture, and implementation details.

---

## Overview

**Problem:** Users experienced multi-second delays before hearing any audio response from the voice assistant.

**Solution:** Sentence-level progressive TTS synthesis with client-side audio queue management.

**Result:** Time-to-first-audio reduced from 3-5 seconds to ~300-500ms.

---

## Implemented TTS Best Practices

### 1) Sentence-Level Chunked TTS Synthesis

**Problem (before):**
```
User speaks → LLM generates 1500-char response → TTS synthesizes ALL 1500 chars → Audio plays
                                                 ↑
                                            3-5 second delay here
```

**Solution (after):**
```
User speaks → LLM generates response → Split into sentences → TTS synthesizes sentence 1 → Play
                                                            → TTS synthesizes sentence 2 → Queue
                                                            → TTS synthesizes sentence 3 → Queue
                                                              ↑
                                                         ~300ms to first audio
```

**Implementation in `src/tts/response_reader.py`:**

```python
def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences for progressive TTS synthesis.
    
    Splits on sentence-ending punctuation (.!?) followed by whitespace,
    while preserving abbreviations and decimal numbers.
    """
    if not text:
        return []
    
    sentence_pattern = r'(?<![A-Z][a-z])(?<![A-Z])(?<=\.|\!|\?)\s+'
    sentences = re.split(sentence_pattern, text)
    return [s.strip() for s in sentences if s and s.strip()]


async def read_response_chunked(self, text: str) -> None:
    """
    Convert text to speech sentence-by-sentence for faster time-to-first-audio.
    """
    clean_text = strip_markdown(text)
    sentences = split_into_sentences(clean_text)
    
    for i, sentence in enumerate(sentences):
        audio_data = await self.tts_provider.synthesize(sentence)
        
        if audio_data:
            await self.websocket.send_json({
                "type": "tts_audio_chunk",
                "audio": audio_data.hex(),
                "format": "pcm16",
                "sample_rate": 16000,
                "chunk_index": i,
                "total_chunks": len(sentences),
            })
    
    await self.websocket.send_json({"type": "tts_complete"})
```

**Why this works:**
- First sentence (~50-100 chars) synthesizes in ~200-300ms
- Audio playback begins immediately while remaining sentences synthesize
- User perceives near-instant response

---

### 2) Non-Blocking TTS API Calls

**Problem:** Google Cloud TTS `synthesize_speech()` is a synchronous blocking call that would block the async event loop.

**Solution:** Wrap the synchronous call in `run_in_executor()` to run in a thread pool.

**Implementation in `src/tts/google_tts.py`:**

```python
async def synthesize(self, text: str) -> bytes:
    """
    Synthesize audio from text using thread pool executor.
    """
    if not text or not text.strip():
        return b""

    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(...)
        audio_config = texttospeech.AudioConfig(...)

        # Run synchronous API call in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,  # Use default executor
            partial(
                self.client.synthesize_speech,
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config,
            )
        )

        return response.audio_content

    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        return b""
```

**Why this matters:**
- Without executor: TTS blocks all async operations for entire synthesis duration
- With executor: Other async tasks (ASR, WebSocket) continue while TTS runs in background
- Critical for responsive real-time voice interaction

---

### 3) Client-Side Audio Queue Management

**Problem:** Progressive server-side chunking requires client-side queue to:
1. Buffer incoming chunks
2. Play chunks in sequence
3. Handle user interruption (stop button)
4. Clean up on page navigation

**Solution:** JavaScript queue with proper lifecycle management.

**Implementation in client JavaScript (routes.py HTML):**

```javascript
// TTS Audio Chunk Playback
if (msg.type === 'tts_audio_chunk') {
    // Initialize queue if needed
    if (!window.ttsChunkQueue) {
        window.ttsChunkQueue = [];
        window.ttsIsPlayingChunk = false;
    }
    
    // Convert and buffer audio
    const audioBuffer = convertPCM16ToAudioBuffer(msg.audio, msg.sample_rate);
    window.ttsChunkQueue.push(audioBuffer);
    
    // Start playback if not already playing
    if (!window.ttsIsPlayingChunk) {
        playNextChunk();
    }
    
    function playNextChunk() {
        if (window.ttsChunkQueue.length === 0) {
            window.ttsIsPlayingChunk = false;
            return;
        }
        
        window.ttsIsPlayingChunk = true;
        const buffer = window.ttsChunkQueue.shift();
        
        const source = audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(audioContext.destination);
        source.onended = playNextChunk;  // Chain to next chunk
        source.start(0);
    }
}

// TTS Complete - restore microphone after queue drains
if (msg.type === 'tts_complete') {
    function checkQueueDrained() {
        if (window.ttsChunkQueue.length === 0 && !window.ttsIsPlayingChunk) {
            // Restore microphone, update UI
            restoreMicrophoneState();
        } else {
            setTimeout(checkQueueDrained, 100);
        }
    }
    checkQueueDrained();
}
```

---

### 4) Proper Stop/Interrupt Handling

**Problem:** When user presses "Stop TTS":
- Must stop current audio immediately
- Must clear queued chunks (prevent continued playback)
- Must restore microphone state

**Incorrect implementation (causes two voices):**
```javascript
// BAD: Only stops current source, queue continues
stopTTSButton.onclick = function() {
    currentTTSSource.stop();
    currentTTSSource.onended();  // This triggers playNextChunk!
};
```

**Correct implementation:**
```javascript
stopTTSButton.onclick = function() {
    // 1. Clear queue FIRST to prevent playNextChunk from continuing
    if (window.ttsChunkQueue) {
        window.ttsChunkQueue = [];
    }
    window.ttsIsPlayingChunk = false;
    
    // 2. Disconnect callback BEFORE stopping (prevents onended from firing)
    if (currentTTSSource) {
        currentTTSSource.onended = null;
        currentTTSSource.stop();
        currentTTSSource = null;
    }
    
    // 3. Reset state
    isTTSPlaying = false;
    stopTTSButton.style.display = 'none';
    
    // 4. Restore microphone
    if (window.ttsWasRecording) {
        isRecording = true;
        status.textContent = 'Listening...';
    }
};
```

**Key insight:** The order of operations matters:
1. Clear queue (prevents new chunks from playing)
2. Null the callback (prevents onended from triggering)
3. Stop the source (stops current audio)
4. Reset state (clean UI)

---

### 5) Page Lifecycle Cleanup

**Problem:** Page reload/navigation while TTS playing causes:
- Audio continues from previous session
- Multiple audio sources (two voices)
- Memory leaks

**Solution:** Clean up TTS state on page unload:

```javascript
window.addEventListener('beforeunload', function() {
    // Clear TTS queue
    if (window.ttsChunkQueue) {
        window.ttsChunkQueue = [];
    }
    
    // Stop current audio
    if (currentTTSSource) {
        try {
            currentTTSSource.stop();
        } catch (e) {}
    }
    
    // Close audio context
    if (window.ttsAudioContext) {
        try {
            window.ttsAudioContext.close();
        } catch (e) {}
    }
    
    // ... rest of cleanup
});
```

---

### 6) Microphone Muting During TTS Playback

**Problem:** If microphone stays active during TTS:
- TTS audio is picked up by microphone
- Sent to ASR as user speech
- Creates feedback loop and false transcriptions

**Solution:** Mute microphone during TTS, restore after completion:

```javascript
// On first TTS chunk
if (!isTTSPlaying) {
    window.ttsWasRecording = isRecording;  // Save state
    if (isRecording) {
        isRecording = false;  // Stop sending audio to ASR
        log('Microphone muted during TTS playback');
    }
    isTTSPlaying = true;
}

// On TTS complete (queue drained)
if (window.ttsWasRecording) {
    isRecording = true;  // Restore mic
    log('Microphone unmuted after TTS playback');
}
```

**Future enhancement:** Implement Acoustic Echo Cancellation (AEC) to allow user interruption during TTS without muting.

---

### 7) Message Format Contract

**Server sends three TTS message types:**

| Message Type | When Sent | Purpose |
|-------------|-----------|---------|
| `tts_audio_chunk` | For each sentence | Progressive audio delivery |
| `tts_complete` | After all chunks | Signal end of TTS stream |
| `tts_audio` (legacy) | Single synthesis | Backward compatibility |

**`tts_audio_chunk` schema:**
```json
{
    "type": "tts_audio_chunk",
    "audio": "<hex-encoded PCM16 bytes>",
    "format": "pcm16",
    "sample_rate": 16000,
    "chunk_index": 0,
    "total_chunks": 5
}
```

**`tts_complete` schema:**
```json
{
    "type": "tts_complete"
}
```

---

## Performance Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Time-to-first-audio | 3-5 seconds | 300-500ms | 6-10x faster |
| Perceived responsiveness | Poor | Excellent | Qualitative |
| User interruption | Delayed | Immediate | Critical for UX |

---

## Testing Strategy

Unit tests in `tests/tts/test_response_reader.py` and `tests/tts/test_tts_chunking.py` verify:

1. **Sentence splitting logic** - Correct boundary detection
2. **Chunk message format** - Required fields present
3. **Progressive delivery** - First chunk sent before full synthesis
4. **No batching** - Each sentence synthesized separately
5. **Error handling** - Graceful degradation on failures
6. **Markdown stripping** - Clean text for natural speech

Run tests:
```bash
pytest tests/tts/ -v
```

---

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TTS_PROVIDER` | `google` | TTS provider to use |
| `TTS_LANGUAGE_CODE` | `en-US` | Language for synthesis |
| `TTS_VOICE_NAME` | `en-US-Neural2-D` | Voice model |
| `TTS_SPEAKING_RATE` | `1.0` | Speech rate (0.25-4.0) |

---

## Troubleshooting

### No audio plays
1. Check browser console for JavaScript errors
2. Verify client handles `tts_audio_chunk` message type
3. Check WebSocket connection is open

### Two voices heard
1. Verify stop handler clears `ttsChunkQueue`
2. Check `beforeunload` cleanup is registered
3. Ensure `onended` callback is nulled before `.stop()`

### Audio cuts off early
1. Check `tts_complete` handler waits for queue to drain
2. Verify all chunks are being queued

### Long delay before audio
1. Verify `read_response()` delegates to `read_response_chunked()`
2. Check sentences are being split correctly
3. Verify executor is used for synthesis (non-blocking)

---

## References

- Google Cloud TTS Documentation: https://cloud.google.com/text-to-speech/docs
- Web Audio API: https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API
- Voice Assistant UX Best Practices: Time-to-first-audio should be <500ms
