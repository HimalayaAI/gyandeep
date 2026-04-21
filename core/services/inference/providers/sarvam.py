from __future__ import annotations
from typing import Any, Dict, List, Optional
from .base import LLMProvider


class SarvamProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float, reasoning_effort: Optional[str] = None):
        try:
            from sarvamai import SarvamAI
        except ImportError:
            raise RuntimeError("sarvamai not installed. Run: pip install sarvamai")
        self.client = SarvamAI(api_subscription_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    def chat_completions(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> Any:
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
        }
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        return self.client.chat.completions(**params)
