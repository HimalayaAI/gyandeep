import pytest

pytest.importorskip("jinja2")
from fastapi.testclient import TestClient

from fastapi.testclient import TestClient

from dashboard.backend import app as dashboard_app


def test_dashboard_routes_exist():
    paths = {route.path for route in dashboard_app.app.routes}
    assert "/" in paths
    assert "/api/upload" in paths
    assert "/api/analyze_env" in paths
    assert "/api/analyze_global" in paths
    assert "/api/ask" in paths
    assert "/api/plugins/jobs" in paths
    assert "/api/plugins/jobs/{job_id}" in paths
    assert "/api/plugins/jobs/{job_id}/artifacts/{artifact_type}" in paths
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


def test_create_plugin_job_requires_query():
    client = TestClient(dashboard_app.app)
    response = client.post("/api/plugins/jobs", json={"plugin_id": "manim_video", "query": ""})
    assert response.status_code == 400
    assert "query is required" in response.text


def test_create_plugin_job_rejects_unknown_plugin():
    client = TestClient(dashboard_app.app)
    response = client.post("/api/plugins/jobs", json={"plugin_id": "does_not_exist", "query": "animate this"})
    assert response.status_code == 400
    assert "Unknown plugin" in response.text
def test_ask_returns_config_error_before_env_context_build(monkeypatch):
    class DummyInference:
        def is_configured(self) -> bool:
            return False

    def _should_not_run(_page_index: int):
        raise AssertionError("Environment context should not be built when API key is missing.")

    old_state = dashboard_app.global_pdf_data.copy()
    dashboard_app.global_pdf_data.update(
        {
            "filename": "test.pdf",
            "filepath": "dashboard/uploads/test.pdf",
            "total_pages": 10,
            "pages": {},
            "book_id": None,
        }
    )

    monkeypatch.setattr(dashboard_app, "inference_service", DummyInference())
    monkeypatch.setattr(dashboard_app, "_build_env_context", _should_not_run)

    try:
        client = TestClient(dashboard_app.app)
        response = client.post(
            "/api/ask",
            json={"query": "hello", "mode": "environment", "current_page": 1},
        )
    finally:
        dashboard_app.global_pdf_data.clear()
        dashboard_app.global_pdf_data.update(old_state)

    assert response.status_code == 200
    assert dashboard_app.ERR_LLM_NOT_CONFIGURED in response.text
