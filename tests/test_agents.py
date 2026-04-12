import pytest

from core.agents.context_manager import ContextManager
from core.agents.prompt_manager import PromptManager


class _FakeInference:
    def __init__(self):
        self.messages = None

    def chat_completions(self, messages):
        self.messages = messages

        class _Msg:
            content = "<final>Structured summary.</final>"
            reasoning_content = None

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()

    @staticmethod
    def extract_response_payload(response):
        msg = response.choices[0].message
        return msg.content.replace("<final>", "").replace("</final>", "").strip(), ""


def test_prompt_manager_formats_current_page():
    prompt = PromptManager.current_page_prompt("Context here.")
    assert "Structured Current-Page Context" in prompt
    assert "Context here." in prompt


def test_prompt_manager_formats_whole_book():
    prompt = PromptManager.whole_book_prompt("Chunk A")
    assert "Retrieved Context" in prompt
    assert "Chunk A" in prompt


@pytest.mark.asyncio
async def test_context_manager_builds_structured_context():
    manager = ContextManager(_FakeInference())
    structured = await manager.build_structured_context("raw text")
    assert "Structured summary" in structured
