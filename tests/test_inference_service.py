"""Tests for InferenceService with mocked dependencies."""
from unittest.mock import MagicMock, patch
import pytest


def test_sarvam_provider_configured():
    with patch.dict("sys.modules", {"sarvamai": MagicMock()}):
        from core.services.inference.inference import InferenceService
        service = InferenceService(
            provider="sarvam",
            api_key="key",
            model="sarvam-m",
            max_tokens=123,
            temperature=0.5,
            reasoning_effort="medium",
        )
        assert service.is_configured()


def test_sarvam_unconfigured_when_no_key():
    with patch.dict("sys.modules", {"sarvamai": MagicMock()}):
        from core.services.inference.inference import InferenceService
        service = InferenceService(
            provider="sarvam",
            api_key="",
            model="sarvam-m",
            max_tokens=123,
            temperature=0.5,
        )
        assert not service.is_configured()


def test_ollama_configured_without_key():
    with patch.dict("sys.modules", {"litellm": MagicMock()}):
        from core.services.inference.inference import InferenceService
        service = InferenceService(
            provider="ollama",
            api_key="",
            model="llama3",
            max_tokens=512,
            temperature=0.7,
            base_url="http://localhost:11434",
        )
        assert service.is_configured()


def test_openai_unconfigured_without_key():
    with patch.dict("sys.modules", {"litellm": MagicMock()}):
        from core.services.inference.inference import InferenceService
        service = InferenceService(
            provider="openai",
            api_key="",
            model="gpt-4o",
            max_tokens=1024,
            temperature=0.5,
        )
        assert not service.is_configured()


def test_extract_think_and_final():
    from core.services.inference.utils import extract_think_and_final
    content, thinking = extract_think_and_final("<tool_call>plan</tool_call>\n<final>Answer.</final>")
    assert content == "Answer."
    assert thinking == "plan"


def test_litellm_model_string_ollama():
    with patch.dict("sys.modules", {"litellm": MagicMock()}):
        from core.services.inference.providers.litellm import LiteLLMProvider
        provider = LiteLLMProvider(
            provider_name="ollama",
            api_key="ollama",
            model="llama3",
            max_tokens=512,
            temperature=0.7,
            base_url="http://localhost:11434",
        )
        assert provider._get_model_string() == "ollama_chat/llama3"


def test_litellm_model_string_openai():
    with patch.dict("sys.modules", {"litellm": MagicMock()}):
        from core.services.inference.providers.litellm import LiteLLMProvider
        provider = LiteLLMProvider(
            provider_name="openai",
            api_key="sk-test",
            model="gpt-4o",
            max_tokens=1024,
            temperature=0.5,
        )
        assert provider._get_model_string() == "gpt-4o"


def test_litellm_model_string_gemini():
    with patch.dict("sys.modules", {"litellm": MagicMock()}):
        from core.services.inference.providers.litellm import LiteLLMProvider
        provider = LiteLLMProvider(
            provider_name="gemini",
            api_key="AIzaSy...",
            model="gemini-2.5-flash-lite",
            max_tokens=1024,
            temperature=0.5,
        )
        assert provider._get_model_string() == "gemini/gemini-2.5-flash-lite"


def test_litellm_model_string_openrouter():
    with patch.dict("sys.modules", {"litellm": MagicMock()}):
        from core.services.inference.providers.litellm import LiteLLMProvider
        provider = LiteLLMProvider(
            provider_name="openrouter",
            api_key="sk-or-...",
            model="google/gemini-3.1-flash-lite-preview",
            max_tokens=1024,
            temperature=0.5,
        )
        assert provider._get_model_string() == "openrouter/google/gemini-3.1-flash-lite-preview"


def test_unsupported_provider_raises():
    with patch.dict("sys.modules", {"litellm": MagicMock()}):
        from core.services.inference.inference import InferenceService
        with pytest.raises(ValueError, match="Unknown provider"):
            InferenceService(provider="unsupported", api_key="x", model="m", max_tokens=1, temperature=0.5)
