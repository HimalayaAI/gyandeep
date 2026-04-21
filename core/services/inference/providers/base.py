from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from ..utils import extract_think_and_final


class LLMProvider(ABC):
    @abstractmethod
    def chat_completions(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> Any:
        pass

    def extract_response_payload(self, response: Any) -> Tuple[str, str]:
        msg = response.choices[0].message
        content = (msg.content or "").strip()
        reasoning = getattr(msg, "reasoning_content", None)
        reasoning = reasoning.strip() if reasoning else ""
        content, think_text = extract_think_and_final(content)
        if think_text:
            reasoning = reasoning or think_text
        return content, reasoning
