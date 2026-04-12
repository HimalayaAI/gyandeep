from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from core.agents.prompt_manager import PromptManager


class ContextManager:
    """Builds structured context blocks for tutoring."""

    def __init__(
        self,
        inference_service,
        model_context_window: int,
        safety_tokens: int,
        token_char_ratio: float,
        summary_max_tokens: int,
    ):
        self._inference = inference_service
        self._extract_response = inference_service.extract_response_payload
        self._context_window = model_context_window
        self._safety_tokens = safety_tokens
        self._token_char_ratio = token_char_ratio
        self._summary_max_tokens = summary_max_tokens

    def _truncate_raw_text(self, raw_text: str) -> str:
        available_tokens = self._context_window - self._inference.max_tokens - self._safety_tokens
        available_tokens = max(256, available_tokens)
        max_chars = int(available_tokens * self._token_char_ratio)
        max_chars = min(max_chars, 12000)
        if len(raw_text) > max_chars:
            return raw_text[:max_chars]
        return raw_text

    async def build_structured_context(self, raw_text: str) -> str:
        raw_text = raw_text.strip()
        if not raw_text:
            return ""
        raw_text = self._truncate_raw_text(raw_text)
        prompt = PromptManager.env_summary_prompt(raw_text)
        response = await asyncio.to_thread(
            self._inference.chat_completions,
            [{'role': 'user', 'content': prompt}],
            self._summary_max_tokens,
        )
        extracted, _ = self._extract_response(response)
        return extracted or ""

    async def build_global_chunk_summary(
        self,
        raw_text: str,
        page_start: int,
        page_end: int,
    ) -> str:
        raw_text = raw_text.strip()
        if not raw_text:
            return ""
        raw_text = self._truncate_raw_text(raw_text)
        prompt = PromptManager.global_summary_prompt(raw_text, page_start, page_end)
        response = await asyncio.to_thread(
            self._inference.chat_completions,
            [{'role': 'user', 'content': prompt}],
            self._summary_max_tokens,
        )
        extracted, _ = self._extract_response(response)
        return extracted or ""
