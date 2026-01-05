"""Tests for AudioTrackProcessor module."""

from unittest.mock import Mock

import numpy as np
import pytest

from src.gateway.audio_track_processor import AudioTrackProcessor


class FakeAudioSession:
    def __init__(self):
        self.processed_audio = []

    async def process_audio(self, audio_chunk):
        self.processed_audio.append(audio_chunk)


class FakeMediaStreamTrack:
    def __init__(self):
        self.kind = "audio"
        self.frame_count = 0

    async def recv(self):
        self.frame_count += 1
        # Create a fake audio frame
        frame = Mock()
        frame.channels = 1
        frame.samples = 1600  # 100ms at 16kHz
        frame.data = np.random.randint(-32768, 32767, 1600, dtype=np.int16).tobytes()
        return frame


@pytest.mark.asyncio
async def test_audio_track_processor_initialization():
    """Test that AudioTrackProcessor initializes correctly."""
    fake_track = FakeMediaStreamTrack()  # noqa: F841
    fake_session = FakeAudioSession()

    processor = AudioTrackProcessor(fake_track, fake_session)

    assert processor.kind == "audio"
    assert processor.track == fake_track
    assert processor.session == fake_session
    assert processor.frame_count == 0
    assert processor.total_samples == 0


@pytest.mark.asyncio
async def test_audio_track_processor_recv():
    """Test that AudioTrackProcessor processes frames correctly."""
    fake_track = FakeMediaStreamTrack()  # noqa: F841
    fake_session = FakeAudioSession()

    processor = AudioTrackProcessor(fake_track, fake_session)

    # Process a few frames
    for _ in range(5):
        frame = await processor.recv()
        assert frame is not None

    assert processor.frame_count == 5
    assert len(fake_session.processed_audio) == 5
    assert processor.total_samples > 0


@pytest.mark.asyncio
async def test_audio_track_processor_silent_frames():
    """Test that AudioTrackProcessor handles silent frames correctly."""
    fake_track = FakeMediaStreamTrack()  # noqa: F841
    fake_session = FakeAudioSession()

    # Create a track that returns silent frames
    class SilentTrack:
        def __init__(self):
            self.kind = "audio"
            self.frame_count = 0

        async def recv(self):
            self.frame_count += 1
            frame = Mock()
            frame.channels = 1
            frame.samples = 1600
            # Silent audio (all zeros)
            frame.data = np.zeros(1600, dtype=np.int16).tobytes()
            return frame

    silent_track = SilentTrack()
    processor = AudioTrackProcessor(silent_track, fake_session)

    # Process a few frames
    for _ in range(10):
        frame = await processor.recv()
        assert frame is not None

    # Silent frames should not be processed
    assert len(fake_session.processed_audio) == 0
    assert processor.frame_count == 10


@pytest.mark.asyncio
async def test_audio_track_processor_ndarray_frames():
    """Test that AudioTrackProcessor handles frames with to_ndarray method."""
    fake_track = FakeMediaStreamTrack()  # noqa: F841
    fake_session = FakeAudioSession()

    # Create a track that returns frames with to_ndarray method
    class NdarrayTrack:
        def __init__(self):
            self.kind = "audio"
            self.frame_count = 0

        async def recv(self):
            self.frame_count += 1
            frame = Mock(spec=["to_ndarray"])  # Only spec to_ndarray, no channels/samples
            # Create actual audio data with sufficient amplitude
            audio_array = np.random.uniform(-1, 1, 1600).astype(np.float32)
            # Ensure some values exceed the threshold
            audio_array[0:100] = 0.5  # Set some values to ensure max > 10 after conversion
            frame.to_ndarray = Mock(return_value=audio_array)
            return frame

    ndarray_track = NdarrayTrack()
    processor = AudioTrackProcessor(ndarray_track, fake_session)

    # Process a frame
    frame = await processor.recv()
    assert frame is not None

    assert processor.frame_count == 1
    assert len(fake_session.processed_audio) == 1
    # Check that audio was converted to int16
    assert fake_session.processed_audio[0].dtype == np.int16


@pytest.mark.asyncio
async def test_audio_track_processor_planes_frames():
    """Test that AudioTrackProcessor handles frames with planes attribute."""
    fake_track = FakeMediaStreamTrack()  # noqa: F841
    fake_session = FakeAudioSession()

    # Create a track that returns frames with planes
    class PlanesTrack:
        def __init__(self):
            self.kind = "audio"
            self.frame_count = 0

        async def recv(self):
            self.frame_count += 1
            frame = Mock()
            frame.channels = 1
            frame.samples = 1600
            frame.data = None  # No data attribute
            frame.planes = [np.random.randint(-32768, 32767, 1600, dtype=np.int16).tobytes()]
            return frame

    planes_track = PlanesTrack()
    processor = AudioTrackProcessor(planes_track, fake_session)

    # Process a frame
    frame = await processor.recv()
    assert frame is not None

    assert processor.frame_count == 1
    assert len(fake_session.processed_audio) == 1
