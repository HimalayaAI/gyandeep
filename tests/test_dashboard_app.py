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


def test_create_plugin_job_falls_back_to_memory_when_db_unavailable(monkeypatch):
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

    async def fake_build_animation_context(query, mode, current_page):
        return "context"

    created_jobs = []

    def fake_create_task(coro):
        created_jobs.append(coro)
        coro.close()
        return None

    monkeypatch.setattr(dashboard_app, "_db_connect", lambda: (_ for _ in ()).throw(ConnectionError("db down")))
    monkeypatch.setattr(dashboard_app, "_build_animation_context", fake_build_animation_context)
    monkeypatch.setattr(dashboard_app.asyncio, "create_task", fake_create_task)

    try:
        client = TestClient(dashboard_app.app)
        response = client.post(
            "/api/plugins/jobs",
            json={"plugin_id": "manim_video", "query": "animate this", "current_page": 1},
        )
    finally:
        dashboard_app.global_pdf_data.clear()
        dashboard_app.global_pdf_data.update(old_state)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["job_id"]
    assert payload["job_id"] in dashboard_app._memory_plugin_jobs
    assert created_jobs


def test_memory_plugin_job_store_supports_update_and_events(monkeypatch):
    monkeypatch.setattr(dashboard_app, "_db_connect", lambda: (_ for _ in ()).throw(ConnectionError("db down")))

    job_id = dashboard_app._create_plugin_job(
        plugin_id="manim_video",
        query="animate this",
        mode="environment",
        current_page=1,
        book_id=None,
        context_text="context",
    )

    dashboard_app._append_plugin_job_event(job_id, "queued", "accepted")
    dashboard_app._update_plugin_job(job_id, status="succeeded", started_at="now", finished_at="now", plan_text="plan")

    job = dashboard_app._fetch_plugin_job(job_id)
    events = dashboard_app._fetch_plugin_job_events(job_id)

    assert job is not None
    assert job["status"] == "succeeded"
    assert job["plan_text"] == "plan"
    assert job["started_at"] is not None
    assert job["finished_at"] is not None
    assert events and events[0]["phase"] == "queued"


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
    assert dashboard_app.ERR_SARVAM_NOT_CONFIGURED in response.text


def test_upload_starts_precompute_when_embeddings_enabled_without_ocr(monkeypatch):
    class DummyDoc:
        def get_toc(self):
            return []

        def __len__(self):
            return 3

        def close(self):
            return None

    created_jobs = []

    def fake_create_task(coro):
        created_jobs.append(coro)
        coro.close()
        return None

    monkeypatch.setattr(dashboard_app, "PRECOMPUTE_OCR_ON_UPLOAD", False)
    monkeypatch.setattr(dashboard_app, "PRECOMPUTE_EMBEDDINGS_ON_UPLOAD", True)
    monkeypatch.setattr(dashboard_app.fitz, "open", lambda _path: DummyDoc())
    monkeypatch.setattr(dashboard_app, "_upsert_book", lambda _name, _hash, _pages: "book-1")
    monkeypatch.setattr(dashboard_app.asyncio, "create_task", fake_create_task)

    client = TestClient(dashboard_app.app)
    response = client.post(
        "/api/upload",
        files={"file": ("demo.pdf", b"%PDF-1.4\n%dummy\n", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert created_jobs


def test_ask_analyze_falls_back_to_env_context_when_embeddings_missing(monkeypatch):
    class DummyInference:
        def __init__(self):
            self.last_messages = []

        def is_configured(self) -> bool:
            return True

        def chat_completions(self, messages, max_tokens=None):
            self.last_messages = messages
            return {"content": "Fallback answer"}

        def extract_response_payload(self, response):
            return response.get("content", ""), ""

    async def fake_retrieve(_query: str, _top_k: int):
        return []

    async def fake_keyword_empty(_query: str, _top_k: int):
        return []

    async def fake_env_context(_current_page: int):
        return "Structured fallback context", "Raw fallback context"

    old_state = dashboard_app.global_pdf_data.copy()
    dashboard_app.global_pdf_data.update(
        {
            "filename": "demo.pdf",
            "filepath": "dashboard/uploads/demo.pdf",
            "total_pages": 10,
            "pages": {},
            "book_id": "book-1",
        }
    )

    dummy = DummyInference()
    monkeypatch.setattr(dashboard_app, "inference_service", dummy)
    monkeypatch.setattr(dashboard_app, "_retrieve_relevant_chunks", fake_retrieve)
    monkeypatch.setattr(dashboard_app, "_retrieve_keyword_chunks", fake_keyword_empty)
    monkeypatch.setattr(dashboard_app, "_build_env_context", fake_env_context)
    monkeypatch.setattr(dashboard_app.PromptManager, "whole_book_prompt", lambda ctx: f"WHOLE::{ctx}")

    try:
        client = TestClient(dashboard_app.app)
        response = client.post(
            "/api/ask",
            json={"query": "Explain example 4", "mode": "analyze", "current_page": 1},
        )
    finally:
        dashboard_app.global_pdf_data.clear()
        dashboard_app.global_pdf_data.update(old_state)

    assert response.status_code == 200
    assert "Fallback answer" in response.text
    assert dummy.last_messages
    assert dummy.last_messages[0]["content"] == "WHOLE::Structured fallback context"


def test_ask_analyze_uses_keyword_fallback_before_env_context(monkeypatch):
    class DummyInference:
        def __init__(self):
            self.last_messages = []

        def is_configured(self) -> bool:
            return True

        def chat_completions(self, messages, max_tokens=None):
            self.last_messages = messages
            return {"content": "Keyword fallback answer"}

        def extract_response_payload(self, response):
            return response.get("content", ""), ""

    async def fake_retrieve(_query: str, _top_k: int):
        return []

    async def fake_keyword(_query: str, _top_k: int):
        return ["--- Retrieved Page 5 (keyword fallback) ---\nExample 4: ..."]

    async def fake_env_context(_current_page: int):
        raise AssertionError("Env context fallback should not run when keyword fallback succeeds")

    old_state = dashboard_app.global_pdf_data.copy()
    dashboard_app.global_pdf_data.update(
        {
            "filename": "demo.pdf",
            "filepath": "dashboard/uploads/demo.pdf",
            "total_pages": 10,
            "pages": {},
            "book_id": "book-1",
        }
    )

    dummy = DummyInference()
    monkeypatch.setattr(dashboard_app, "inference_service", dummy)
    monkeypatch.setattr(dashboard_app, "_retrieve_relevant_chunks", fake_retrieve)
    monkeypatch.setattr(dashboard_app, "_retrieve_keyword_chunks", fake_keyword)
    monkeypatch.setattr(dashboard_app, "_build_env_context", fake_env_context)
    monkeypatch.setattr(dashboard_app.PromptManager, "whole_book_prompt", lambda ctx: f"WHOLE::{ctx}")

    try:
        client = TestClient(dashboard_app.app)
        response = client.post(
            "/api/ask",
            json={"query": "Explain example 4", "mode": "analyze", "current_page": 1},
        )
    finally:
        dashboard_app.global_pdf_data.clear()
        dashboard_app.global_pdf_data.update(old_state)

    assert response.status_code == 200
    assert "Keyword fallback answer" in response.text
    assert dummy.last_messages
    assert "keyword fallback" in dummy.last_messages[0]["content"].lower()
