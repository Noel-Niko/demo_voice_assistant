#!/usr/bin/env python3
# tests/asr/test_vad.py

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.asr.vad import SileroVADProvider, VADModelProvider, VoiceActivityDetector


class MockVADProvider(VADModelProvider):
    """Mock VAD model provider for testing."""

    def __init__(self, mock_model=None):
        self.mock_model = mock_model or MagicMock()
        # Mock the reset_states method
        self.mock_model.reset_states = MagicMock()

    def get_model(self):
        return self.mock_model, {}


class TestVoiceActivityDetector(unittest.TestCase):
    def setUp(self):
        # Create a mock model that returns predictable speech probabilities
        self.mock_model = MagicMock()
        self.mock_model.reset_states = MagicMock()

        # Configure the mock model to return different probabilities
        def side_effect(audio, sample_rate):
            # Return high probability for non-zero audio, low for zeros
            if torch.sum(torch.abs(audio)) > 0:
                return torch.tensor([0.8])  # High probability (speech)
            else:
                return torch.tensor([0.2])  # Low probability (no speech)

        self.mock_model.side_effect = side_effect
        self.mock_provider = MockVADProvider(self.mock_model)

        # Create VAD with mock provider
        self.vad = VoiceActivityDetector(
            model_provider=self.mock_provider,
            threshold=0.5,
            min_speech_duration=0.1,
            min_silence_duration=0.5,
            sample_rate=16000,
        )

        # Override the model's behavior for our tests
        self.vad.model = self.mock_model

    def test_initialization(self):
        """Test that the VAD initializes correctly."""
        self.assertEqual(self.vad.threshold, 0.5)
        self.assertEqual(self.vad.min_speech_samples, 1600)  # 0.1s * 16000Hz
        self.assertEqual(self.vad.min_silence_samples, 8000)  # 0.5s * 16000Hz
        self.assertEqual(self.vad.sample_rate, 16000)
        self.assertEqual(self.vad.speech_samples, 0)
        self.assertEqual(self.vad.silence_samples, 0)
        self.assertFalse(self.vad.is_speaking)
        self.assertEqual(len(self.vad.audio_buffer), 0)

    def test_normalize_audio_int16(self):
        """Test audio normalization with int16 input."""
        # Create int16 audio
        audio = np.array([32767, 0, -32768], dtype=np.int16)
        tensor = self.vad._normalize_audio(audio)

        # Check type and values
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertAlmostEqual(tensor[0].item(), 0.999969482, places=6)  # 32767/32768
        self.assertEqual(tensor[1].item(), 0.0)
        self.assertAlmostEqual(tensor[2].item(), -1.0, places=6)  # -32768/32768

    def test_normalize_audio_float32(self):
        """Test audio normalization with float32 input."""
        # Create float32 audio
        audio = np.array([1.0, 0.0, -0.5], dtype=np.float32)
        tensor = self.vad._normalize_audio(audio)

        # Check type and values
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor[0].item(), 1.0)
        self.assertEqual(tensor[1].item(), 0.0)
        self.assertEqual(tensor[2].item(), -0.5)

    def test_process_chunk_speech(self):
        """Test processing a chunk with speech."""
        # Create audio with non-zero values (will be detected as speech)
        audio = np.ones(1000, dtype=np.float32)

        # Configure mock to return high probability
        self.mock_model.return_value = torch.tensor([0.8])

        # Process the chunk
        result = self.vad.process_chunk(audio)

        # Check result
        self.assertTrue(result["is_speech"])
        self.assertAlmostEqual(result["speech_probability"], 0.8, places=5)
        self.assertFalse(result["utterance_end"])
        self.assertEqual(self.vad.speech_samples, 1000)
        self.assertEqual(self.vad.silence_samples, 0)
        self.assertTrue(self.vad.is_speaking)
        self.assertEqual(len(self.vad.audio_buffer), 1)

    def test_process_chunk_silence(self):
        """Test processing a chunk with silence."""
        # Create audio with zeros (will be detected as silence)
        audio = np.zeros(1000, dtype=np.float32)

        # Configure mock to return low probability
        self.mock_model.return_value = torch.tensor([0.2])

        # Process the chunk
        result = self.vad.process_chunk(audio)

        # Check result
        self.assertFalse(result["is_speech"])
        self.assertAlmostEqual(result["speech_probability"], 0.2, places=5)
        self.assertFalse(result["utterance_end"])
        self.assertEqual(self.vad.speech_samples, 0)
        self.assertEqual(self.vad.silence_samples, 1000)
        self.assertFalse(self.vad.is_speaking)
        self.assertEqual(len(self.vad.audio_buffer), 0)

    def test_utterance_end_detection(self):
        """Test detection of utterance end."""
        # First, simulate some speech
        speech = np.ones(2000, dtype=np.float32)  # More than min_speech_samples
        self.mock_model.return_value = torch.tensor([0.8])
        self.vad.process_chunk(speech)

        # Then, simulate silence
        silence = np.zeros(9000, dtype=np.float32)  # More than min_silence_samples
        self.mock_model.return_value = torch.tensor([0.2])
        result = self.vad.process_chunk(silence)

        # Check that utterance end was detected
        self.assertTrue(result["utterance_end"])
        self.assertIn("audio_buffer", result)
        self.assertEqual(len(result["audio_buffer"]), 11000)  # 2000 + 9000

        # Check that state was reset
        self.assertEqual(self.vad.speech_samples, 0)
        self.assertEqual(self.vad.silence_samples, 0)
        self.assertFalse(self.vad.is_speaking)
        self.assertEqual(len(self.vad.audio_buffer), 0)

        # Verify model state was reset
        self.vad.model.reset_states.assert_called_once()

    def test_empty_audio_chunk(self):
        """Test handling of empty audio chunk."""
        with self.assertRaises(ValueError):
            self.vad.process_chunk(np.array([]))

    def test_mixed_dtype_handling(self):
        """Test handling of mixed dtype audio chunks."""
        # Add chunks with different dtypes
        self.mock_model.return_value = torch.tensor([0.8])

        # First chunk (int16)
        chunk1 = np.array([100, 200, 300], dtype=np.int16)
        self.vad.process_chunk(chunk1)

        # Second chunk (float32)
        chunk2 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.vad.process_chunk(chunk2)

        # Force utterance end
        self.vad.is_speaking = True
        self.vad.speech_samples = 2000
        self.vad.silence_samples = 9000

        # Process a final chunk to trigger utterance end
        self.mock_model.return_value = torch.tensor([0.2])
        chunk3 = np.zeros(1000, dtype=np.float32)
        result = self.vad.process_chunk(chunk3)

        # Check that concatenation worked without errors
        self.assertTrue(result["utterance_end"])
        self.assertIn("audio_buffer", result)

        # All chunks should have been converted to a common dtype
        self.assertEqual(result["audio_buffer"].dtype, chunk1.dtype)

    def test_reset(self):
        """Test reset functionality."""
        # Set some state
        self.vad.speech_samples = 1000
        self.vad.silence_samples = 2000
        self.vad.is_speaking = True
        self.vad.audio_buffer = [np.ones(100)]

        # Reset
        self.vad.reset()

        # Check state was reset
        self.assertEqual(self.vad.speech_samples, 0)
        self.assertEqual(self.vad.silence_samples, 0)
        self.assertFalse(self.vad.is_speaking)
        self.assertEqual(len(self.vad.audio_buffer), 0)

        # Check model state was reset
        self.vad.model.reset_states.assert_called_once()


class TestSileroVADProvider(unittest.TestCase):
    @patch("torch.hub.load")
    def test_get_model(self, mock_torch_hub):
        """Test that the provider loads the model correctly."""
        # Setup mock
        mock_model = MagicMock()
        mock_utils = {}
        mock_torch_hub.return_value = (mock_model, mock_utils)

        # Create provider and get model
        provider = SileroVADProvider()
        model, utils = provider.get_model()

        # Check that torch.hub.load was called correctly
        mock_torch_hub.assert_called_once_with(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            source="github",
            force_reload=False,
            verbose=False,
            cache_dir=None,
        )

        # Check that model was put in eval mode
        mock_model.eval.assert_called_once()

        # Check returned values
        self.assertEqual(model, mock_model)
        self.assertEqual(utils, mock_utils)

    @patch("torch.hub.load")
    def test_get_model_error(self, mock_torch_hub):
        """Test error handling when loading the model fails."""
        # Setup mock to raise an exception
        mock_torch_hub.side_effect = RuntimeError("Network error")

        # Create provider
        provider = SileroVADProvider()

        # Check that exception is propagated
        with self.assertRaises(RuntimeError):
            provider.get_model()


if __name__ == "__main__":
    unittest.main()
