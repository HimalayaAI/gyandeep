import sys
import types

import pytest

from core.services.storage.embedding_service import index_embeddings
from core.services.storage.embedding_service import EmbeddingService


def _install_fake_psycopg2(monkeypatch):
    state = {
        "connect_calls": [],
        "execute_calls": [],
        "conn": None,
    }

    class FakeCursor:
        def __init__(self):
            self.executed = []

        def execute(self, sql):
            self.executed.append(sql)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConn:
        def __init__(self):
            self.cursor_obj = FakeCursor()

        def cursor(self):
            return self.cursor_obj

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def connect(**kwargs):
        state["connect_calls"].append(kwargs)
        state["conn"] = FakeConn()
        return state["conn"]

    def execute_values(cur, query, records, template=None):
        state["execute_calls"].append(
            {"cur": cur, "query": query, "records": records, "template": template}
        )

    fake_psycopg2 = types.SimpleNamespace(connect=connect)
    fake_extras = types.SimpleNamespace(execute_values=execute_values)

    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", fake_extras)

    return state


def test_index_embeddings_writes_records(monkeypatch):
    state = _install_fake_psycopg2(monkeypatch)

    chunks = ["hello"]
    embeddings = [[1.23456789, 2.0]]

    index_embeddings(chunks, embeddings, source="book-1", ensure_schema=True)

    assert len(state["connect_calls"]) == 1
    assert state["conn"].cursor_obj.executed, "schema should be applied when ensure_schema=True"
    assert len(state["execute_calls"]) == 1

    record = state["execute_calls"][0]["records"][0]
    assert record[0] == "book-1"
    assert record[1] == 0
    assert record[2] == "hello"
    assert record[3] == "[1.234568,2.000000]"


def test_index_embeddings_length_mismatch(monkeypatch):
    state = _install_fake_psycopg2(monkeypatch)

    with pytest.raises(ValueError):
        index_embeddings(["a"], [[0.1], [0.2]])

    assert len(state["connect_calls"]) == 0


@pytest.mark.asyncio
async def test_get_relevant_chunks(monkeypatch):
    state = _install_fake_psycopg2(monkeypatch)

    def connect(**kwargs):
        state["connect_calls"].append(kwargs)

        class FakeCursor:
            def execute(self, query, params):
                self.query = query
                self.params = params

            def fetchall(self):
                return [("chunk-1",), ("chunk-2",)]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeConn:
            def cursor(self):
                return FakeCursor()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def close(self):
                return None

        return FakeConn()

    monkeypatch.setitem(sys.modules["psycopg2"].__dict__, "connect", connect)

    embedder = EmbeddingService()

    async def fake_get_embeddings(texts):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(embedder, "get_embeddings", fake_get_embeddings)

    chunks = await embedder.get_relevant_chunks("what is ai?", top_k=2, source="book-1")
    assert chunks == ["chunk-1", "chunk-2"]
