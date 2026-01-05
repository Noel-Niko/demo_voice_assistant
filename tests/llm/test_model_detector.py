"""
Unit tests for model detector utility.
Tests model family detection and API routing logic.
"""

import pytest

from src.llm.model_detector import APType, ModelDetector, ModelFamily


class TestModelDetector:
    """Test model detection functionality."""

    def test_detect_responses_models(self):
        """Test detection of Responses API models."""
        responses_models = [
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-4.1-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-turbo",
        ]

        for model in responses_models:
            family = ModelDetector.detect_family(model)
            assert family == ModelFamily.RESPONSES, f"Failed for {model}"
            assert ModelDetector.is_responses_model(model), f"Failed for {model}"
            assert not ModelDetector.is_chat_completions_model(model), f"Failed for {model}"

    def test_detect_chat_completions_models(self):
        """Test detection of Chat Completions API models."""
        chat_models = [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4",
            "gpt-4-32k",
            "gpt-4-turbo",
            "gpt-5",
            "gpt-5-turbo",
        ]

        for model in chat_models:
            family = ModelDetector.detect_family(model)
            assert family == ModelFamily.CHAT_COMPLETIONS, f"Failed for {model}"
            assert ModelDetector.is_chat_completions_model(model), f"Failed for {model}"
            assert not ModelDetector.is_responses_model(model), f"Failed for {model}"

    def test_detect_unknown_models(self):
        """Test that unknown models default to Chat Completions."""
        unknown_models = ["unknown-model", "custom-model", "gpt-6", "claude-3"]

        for model in unknown_models:
            family = ModelDetector.detect_family(model)
            assert family == ModelFamily.CHAT_COMPLETIONS, f"Failed for {model}"
            assert ModelDetector.is_chat_completions_model(model), f"Failed for {model}"

    def test_get_api_type(self):
        """Test API type detection."""
        assert ModelDetector.get_api_type("gpt-4.1-mini") == APType.RESPONSES
        assert ModelDetector.get_api_type("gpt-4o") == APType.RESPONSES
        assert ModelDetector.get_api_type("gpt-4") == APType.CHAT_COMPLETIONS
        assert ModelDetector.get_api_type("gpt-5") == APType.CHAT_COMPLETIONS
        assert ModelDetector.get_api_type("unknown") == APType.CHAT_COMPLETIONS

    def test_get_token_parameter_name(self):
        """Test token parameter name mapping."""
        # Responses API models
        assert ModelDetector.get_token_parameter_name("gpt-4.1-mini") == "max_output_tokens"
        assert ModelDetector.get_token_parameter_name("gpt-4o") == "max_output_tokens"

        # GPT-5 models
        assert ModelDetector.get_token_parameter_name("gpt-5") == "max_completion_tokens"
        assert ModelDetector.get_token_parameter_name("gpt-5-turbo") == "max_completion_tokens"

        # Legacy models
        assert ModelDetector.get_token_parameter_name("gpt-4") == "max_tokens"
        assert ModelDetector.get_token_parameter_name("gpt-3.5-turbo") == "max_tokens"

        # Unknown models default to max_tokens
        assert ModelDetector.get_token_parameter_name("unknown") == "max_tokens"

    def test_supports_temperature(self):
        """Test temperature support detection."""
        # GPT-5 models don't support temperature
        assert not ModelDetector.supports_temperature("gpt-5")
        assert not ModelDetector.supports_temperature("gpt-5-turbo")

        # All other models support temperature
        assert ModelDetector.supports_temperature("gpt-4.1-mini")
        assert ModelDetector.supports_temperature("gpt-4o")
        assert ModelDetector.supports_temperature("gpt-4")
        assert ModelDetector.supports_temperature("gpt-3.5-turbo")
        assert ModelDetector.supports_temperature("unknown")

    def test_case_insensitive(self):
        """Test that model detection is case insensitive."""
        assert ModelDetector.is_responses_model("GPT-4.1-MINI")
        assert ModelDetector.is_responses_model("GPT-4o")
        assert ModelDetector.is_chat_completions_model("GPT-4")
        assert ModelDetector.is_chat_completions_model("GPT-5")

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        assert ModelDetector.is_responses_model("  gpt-4.1-mini  ")
        assert ModelDetector.is_chat_completions_model("  gpt-4  ")

    def test_invalid_model_names(self):
        """Test handling of invalid model names."""
        with pytest.raises(ValueError, match="Model name must be a non-empty string"):
            ModelDetector.detect_family("")

        with pytest.raises(ValueError, match="Model name must be a non-empty string"):
            ModelDetector.detect_family(None)

        with pytest.raises(ValueError, match="Model name must be a non-empty string"):
            ModelDetector.detect_family("   ")


class TestModelFamilyEnum:
    """Test ModelFamily enum values."""

    def test_enum_values(self):
        """Test that enum values are correct."""
        assert ModelFamily.CHAT_COMPLETIONS.value == "chat_completions"
        assert ModelFamily.RESPONSES.value == "responses"


class TestAPTypeEnum:
    """Test APType enum values."""

    def test_enum_values(self):
        """Test that enum values are correct."""
        assert APType.CHAT_COMPLETIONS.value == "chat.completions"
        assert APType.RESPONSES.value == "responses"
