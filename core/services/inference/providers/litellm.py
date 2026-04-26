from __future__ import annotations
from typing import Any, Dict, List, Optional
from .base import LLMProvider


class LiteLLMProvider(LLMProvider):
    def __init__(self, provider_name: str, api_key: str, model: str, max_tokens: int, temperature: float, base_url: Optional[str] = None):
        try:
            import litellm
        except ImportError:
            raise RuntimeError("litellm not installed. Run: pip install litellm")
        self.litellm = litellm
        self.litellm.drop_params = True
        self.provider_name = provider_name
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = base_url

    def _get_model_string(self) -> str:
        # LiteLLM uses provider/model format for routing
        if self.provider_name == "ollama":
            return f"ollama_chat/{self.model}"
        if self.provider_name in ("gemini", "openrouter"):
            return f"{self.provider_name}/{self.model}"
        return self.model

    def chat_completions(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> Any:
        kwargs = {
            "model": self._get_model_string(),
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
            "api_key": self.api_key,
        }
        if self.base_url:
            kwargs["api_base"] = self.base_url
        return self.litellm.completion(**kwargs)
