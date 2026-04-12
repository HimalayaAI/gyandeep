import pytest

pytest.importorskip("jinja2")

from dashboard.backend import app as dashboard_app


def test_dashboard_routes_exist():
    paths = {route.path for route in dashboard_app.app.routes}
    assert "/" in paths
    assert "/api/upload" in paths
    assert "/api/analyze_env" in paths
    assert "/api/analyze_global" in paths
    assert "/api/ask" in paths
    assert "/static" in paths
    assert "/uploads" in paths


def test_extract_response_content_fallback():
    class Message:
        def __init__(self, content, reasoning_content):
            self.content = content
            self.reasoning_content = reasoning_content

    class Choice:
        def __init__(self, message):
            self.message = message

    class Response:
        def __init__(self, message):
            self.choices = [Choice(message)]

    msg = Message(content="", reasoning_content="fallback")
    resp = Response(msg)

    content, reasoning = dashboard_app._extract_response_payload(resp)
    assert content == ""
    assert reasoning == "fallback"


def test_strip_think_tags():
    text = "<think>secret</think>\n<final>Answer.</final>"
    content, reasoning = dashboard_app._extract_response_payload(
        type("Resp", (), {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": text, "reasoning_content": None})()})()]})
    )
    assert content == "Answer."
    assert reasoning == "secret"


def test_unclosed_think_block():
    text = "<think>Reasoning line 1.\nReasoning line 2.\n\nFinal answer."
    content, reasoning = dashboard_app._extract_response_payload(
        type("Resp", (), {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": text, "reasoning_content": None})()})()]})
    )
    assert reasoning.startswith("Reasoning line 1")
    assert content == "Final answer."


def test_heuristic_reasoning_split():
    text = "Okay, the user is asking about X.\n\nThe answer is Y."
    content, reasoning = dashboard_app._extract_response_payload(
        type("Resp", (), {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": text, "reasoning_content": None})()})()]})
    )
    assert reasoning == ""
    assert content == text
