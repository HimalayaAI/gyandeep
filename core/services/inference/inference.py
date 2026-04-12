from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferenceService:
    api_key: str
    api_key_placeholder: str
    model: str
    max_tokens: int
    temperature: float
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        self._client = None
        try:
            from sarvamai import SarvamAI  # type: ignore

            if self.api_key and self.api_key != self.api_key_placeholder:
                self._client = SarvamAI(api_subscription_key=self.api_key)
        except ImportError:
            self._client = None

    @property
    def client(self):
        return self._client

    def is_configured(self) -> bool:
        return self._client is not None

    def build_params(self, messages: list[dict], max_tokens: int | None = None) -> dict:
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": self.temperature,
        }
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        return params

    def chat_completions(self, messages: list[dict], max_tokens: int | None = None):
        if not self._client:
            raise RuntimeError("Sarvam API not configured")
        return self._client.chat.completions(**self.build_params(messages, max_tokens=max_tokens))

    @staticmethod
    def extract_think_and_final(text: str) -> tuple[str, str]:
        """Extract optional <think> and final content."""
        if not text:
            return "", ""

        import re

        lower = text.lower()
        think_text = ""
        final_text = text

        if "<think>" in lower:
            if "</think>" in lower:
                think_texts = re.findall(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
                think_text = "\n\n".join(t.strip() for t in think_texts if t.strip())
                final_text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
            else:
                # Fallback: treat first paragraph after <think> as reasoning if present.
                after = re.split(r"<think>", text, flags=re.IGNORECASE, maxsplit=1)[-1]
                parts = re.split(r"\n\s*\n", after, maxsplit=1)
                if len(parts) > 1:
                    think_text = parts[0].strip()
                    final_text = parts[1].strip()
                else:
                    final_text = after.strip()
                    think_text = ""

        final_match = re.search(r"<final>(.*?)</final>", final_text, flags=re.IGNORECASE | re.DOTALL)
        if not final_match:
            final_match = re.search(r"<answer>(.*?)</answer>", final_text, flags=re.IGNORECASE | re.DOTALL)

        if final_match:
            final_text = final_match.group(1).strip()
        else:
            final_text = re.sub(r"</?(final|answer)>", "", final_text, flags=re.IGNORECASE).strip()

        final_text = final_text.replace("<think>", "").replace("</think>", "").strip()
        return final_text, think_text

    def extract_response_payload(self, response) -> tuple[str, str]:
        """Extract answer content and any reasoning content without exposing it."""
        msg = response.choices[0].message
        content = (msg.content or "").strip()
        reasoning = getattr(msg, "reasoning_content", None)
        reasoning = reasoning.strip() if reasoning else ""

        content, think_text = self.extract_think_and_final(content)
        if think_text:
            reasoning = reasoning or think_text

        return content, reasoning
