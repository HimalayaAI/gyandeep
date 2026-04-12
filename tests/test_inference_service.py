from core.services.inference.inference import InferenceService


def test_build_params_includes_reasoning_effort():
    service = InferenceService(
        api_key="key",
        api_key_placeholder="placeholder",
        model="sarvam-m",
        max_tokens=123,
        temperature=0.5,
        reasoning_effort="medium",
    )
    params = service.build_params([{"role": "user", "content": "hi"}])
    assert params["model"] == "sarvam-m"
    assert params["max_tokens"] == 123
    assert params["temperature"] == 0.5
    assert params["reasoning_effort"] == "medium"


def test_extract_think_and_final_parses_tags():
    text = "<think>plan</think>\n<final>Answer.</final>"
    content, thinking = InferenceService.extract_think_and_final(text)
    assert content == "Answer."
    assert thinking == "plan"


def test_extract_think_and_final_unclosed_think():
    text = "<think>Reasoning line 1.\nReasoning line 2.\n\nFinal answer."
    content, thinking = InferenceService.extract_think_and_final(text)
    assert thinking.startswith("Reasoning line 1")
    assert content == "Final answer."
