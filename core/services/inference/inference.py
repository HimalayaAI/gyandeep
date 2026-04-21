from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from .providers import SarvamProvider, LiteLLMProvider


@dataclass
class InferenceService:
    provider: str  # required: "sarvam", "openai", "anthropic", "ollama", "openrouter", "gemini"
    api_key: str
    model: str
    max_tokens: int
    temperature: float
    base_url: str = ""
    reasoning_effort: Optional[str] = None

    def __post_init__(self):
        self._provider = self._create_provider()

    def _create_provider(self):
        if self.provider == "sarvam":
            if not self.api_key:
                return None
            return SarvamProvider(
                api_key=self.api_key,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
            )
        elif self.provider in ("openai", "anthropic", "ollama", "openrouter", "gemini"):
            if not self.api_key and self.provider != "ollama":
                return None
            return LiteLLMProvider(
                provider_name=self.provider,
                api_key=self.api_key or "ollama",
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                base_url=self.base_url or None,
            )
        raise ValueError(f"Unknown provider: {self.provider}")

    def is_configured(self) -> bool:
        return self._provider is not None

    def chat_completions(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> Any:
        if not self._provider:
            raise RuntimeError(f"{self.provider} not configured")
        return self._provider.chat_completions(messages, max_tokens)

    def extract_response_payload(self, response: Any) -> tuple[str, str]:
        if not self._provider:
            raise RuntimeError(f"{self.provider} not configured")
        return self._provider.extract_response_payload(response)
