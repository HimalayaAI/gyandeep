"""Minimal SQL storage service for Gyandeep.

Focuses on:
- Text chunks + embeddings (pgvector)
- Simple student learning events
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID

from core.services.storage.db import AsyncDatabase, HAS_ASYNCPG

logger = logging.getLogger("gyandeep.storage")


@dataclass
class DbConfig:
    host: str
    port: int
    user: str
    password: str
    db_name: str
    pool_min: int = 2
    pool_max: int = 10
    retry_delay: float = 1.0
    retry_max_delay: float = 10.0
    retry_backoff_factor: float = 2.0
    retry_jitter: float = 0.1


class SQLStorageService:
    """Minimal async storage service for Gyandeep."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "postgres",
        password: str = "",
        db_name: str = "gyandeep",
        pool_min: int = 2,
        pool_max: int = 10,
        schema_path: Optional[str] = None,
        auto_provision: bool = False,
        docker_compose_path: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db_name = db_name
        self.pool_min = pool_min
        self.pool_max = pool_max
        self.schema_path = schema_path
        self.auto_provision = auto_provision
        self.docker_compose_path = docker_compose_path

        if self.schema_path is None:
            self.schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")

        self._db: Optional[AsyncDatabase] = None
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize connection pool and apply schema."""
        if self._is_initialized:
            return

        if not HAS_ASYNCPG:
            raise RuntimeError("Database storage unavailable: asyncpg not installed.")

        if self._db is None:
            self._db = AsyncDatabase(
                DbConfig(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    db_name=self.db_name,
                    pool_min=self.pool_min,
                    pool_max=self.pool_max,
                )
            )

        if not self.docker_compose_path:
            self.docker_compose_path = os.path.join(
                os.path.dirname(__file__), "docker", "docker-compose.yaml"
            )

        try:
            await self._db.initialize()
        except Exception as exc:
            if self.auto_provision and self._provision_via_docker():
                time.sleep(2)
                await self._db.initialize()
            else:
                raise exc

        if self.schema_path and os.path.exists(self.schema_path):
            with open(self.schema_path, "r") as f:
                schema_sql = f.read()
            try:
                async with self._db.safe_transaction() as conn:
                    await conn.execute(schema_sql)
            except Exception as exc:
                logger.debug("Schema application skipped (may already exist): %s", exc)

        self._is_initialized = True

    async def close(self) -> None:
        if self._db:
            await self._db.close()
        self._is_initialized = False

    def create_session(self, student_id: Optional[UUID] = None) -> "SQLEnvStorageSession":
        return SQLEnvStorageSession(service=self, student_id=student_id)

    def _provision_via_docker(self) -> bool:
        """Attempt to run docker compose up -d for the database."""
        if not self.docker_compose_path or not os.path.exists(self.docker_compose_path):
            logger.error("Docker compose file not found: %s", self.docker_compose_path)
            return False

        commands = [
            ["docker", "compose", "-f", self.docker_compose_path, "up", "-d"],
            ["docker-compose", "-f", self.docker_compose_path, "up", "-d"],
        ]

        for cmd in commands:
            try:
                subprocess.run(cmd, check=True, capture_output=True, env=os.environ.copy())
                logger.info("Database provisioned via Docker.")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        logger.error("Docker not found or failed to start containers.")
        return False


class SQLEnvStorageSession:
    """Scoped session for student learning events and text chunks."""

    def __init__(self, service: SQLStorageService, student_id: Optional[UUID] = None) -> None:
        self.service = service
        self.student_id = student_id

    async def store_learning_event(
        self,
        event_type: str,
        prompt: str,
        response: str,
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        student_id: Optional[UUID] = None,
    ) -> UUID:
        """Store a single learning interaction."""
        if self.service._db is None:
            raise RuntimeError("Database unavailable")

        sid = student_id or self.student_id
        if sid is None:
            raise ValueError("student_id is required")

        query = """
        INSERT INTO learning_events (student_id, event_type, prompt, response, score, metadata)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """
        row = await self.service._db.fetch_one(
            query,
            sid,
            event_type,
            prompt,
            response,
            score,
            metadata or {},
        )
        return row["id"]

    async def store_text_chunks(
        self,
        source: str,
        chunks: List[str],
        embeddings: List[List[float]],
    ) -> None:
        """Upsert text chunks with embeddings."""
        if self.service._db is None:
            raise RuntimeError("Database unavailable")

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        records: List[tuple] = []
        for idx, chunk in enumerate(chunks):
            vector_literal = "[" + ",".join(f"{v:.6f}" for v in embeddings[idx]) + "]"
            records.append((source, idx, chunk, vector_literal))

        async with self.service._db.safe_transaction() as conn:
            await conn.executemany(
                """
                INSERT INTO text_chunks (source, chunk_index, content, embedding)
                VALUES ($1, $2, $3, $4::vector)
                ON CONFLICT (source, chunk_index)
                DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
                """,
                records,
            )


class EnvStorageService(SQLStorageService):
    """Alias for compatibility with earlier naming."""


STORAGE_AVAILABLE = HAS_ASYNCPG

__all__ = [
    "SQLStorageService",
    "EnvStorageService",
    "SQLEnvStorageSession",
    "STORAGE_AVAILABLE",
]
