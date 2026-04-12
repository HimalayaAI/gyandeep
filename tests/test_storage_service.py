from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pytest

from core.services.storage import env_storage


class FakeAsyncDatabase:
    def __init__(self, config=None):
        self.config = config
        self.initialize_called = False
        self.executed_sql = []
        self.executemany_calls = []
        self.fetch_one_calls = []

    async def initialize(self):
        self.initialize_called = True

    async def close(self):
        return None

    async def fetch_one(self, query, *args):
        self.fetch_one_calls.append((query, args))
        return {"id": uuid4()}

    async def executemany(self, query, records):
        self.executemany_calls.append((query, records))

    @asynccontextmanager
    async def safe_transaction(self):
        yield self

    async def execute(self, sql):
        self.executed_sql.append(sql)


@pytest.mark.asyncio
async def test_initialize_applies_schema(tmp_path, monkeypatch):
    schema_path = tmp_path / "schema.sql"
    schema_path.write_text("SELECT 1;")

    monkeypatch.setattr(env_storage, "HAS_ASYNCPG", True)
    monkeypatch.setattr(env_storage, "AsyncDatabase", FakeAsyncDatabase)

    service = env_storage.SQLStorageService(schema_path=str(schema_path))
    await service.initialize()

    assert service._db is not None
    assert service._db.initialize_called is True
    assert service._db.executed_sql == ["SELECT 1;"]


@pytest.mark.asyncio
async def test_store_learning_event_requires_student_id():
    service = env_storage.SQLStorageService()
    service._db = FakeAsyncDatabase()
    session = env_storage.SQLEnvStorageSession(service=service)

    with pytest.raises(ValueError):
        await session.store_learning_event(
            event_type="quiz",
            prompt="Q1",
            response="A1",
        )


@pytest.mark.asyncio
async def test_store_text_chunks_writes_records():
    service = env_storage.SQLStorageService()
    fake_db = FakeAsyncDatabase()
    service._db = fake_db
    session = env_storage.SQLEnvStorageSession(service=service)

    await session.store_text_chunks(
        source="book-1",
        chunks=["hello", "world"],
        embeddings=[[1.0, 2.0], [3.0, 4.0]],
    )

    assert len(fake_db.executemany_calls) == 1
    records = fake_db.executemany_calls[0][1]
    assert records[0][3] == "[1.000000,2.000000]"
    assert records[1][3] == "[3.000000,4.000000]"
