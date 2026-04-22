"""Lightweight embedding storage helpers for Gyandeep."""
from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass
from typing import Iterable, List, Sequence

try:
    from gyandeep_rs import vectors_to_pg_literals as _rs_vec_literals
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


@dataclass
class EmbeddingConfig:
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    max_chars: int = 8000
    batch_size: int = 64
    max_retries: int = 3
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "postgres"
    db_password: str = "postgres"
    db_name: str = "gyandeep"

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        _load_dotenv_if_available()
        return cls(
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "sentence-transformers"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            max_chars=int(os.getenv("EMBEDDING_MAX_CHARS", "8000")),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
            max_retries=int(os.getenv("EMBEDDING_MAX_RETRIES", "3")),
            db_host=os.getenv("DB_HOST", "localhost"),
            db_port=int(os.getenv("DB_PORT", "5432")),
            db_user=os.getenv("DB_USER", "postgres"),
            db_password=os.getenv("DB_PASSWORD", "postgres"),
            db_name=os.getenv("DB_NAME", "gyandeep"),
        )


class EmbeddingService:
    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig.from_env()
        self._st_model = None

    def _truncate_text(self, text: str) -> str:
        if len(text) <= self.config.max_chars:
            return text
        return text[: self.config.max_chars]

    async def get_embeddings(self, texts):
        """Get embeddings with retry logic and batch processing."""
        single_input = isinstance(texts, str)
        texts = [texts] if single_input else list(texts)

        texts = [self._truncate_text(t) for t in texts]

        if self.config.embedding_provider == "openai":
            all_embeddings = await self._get_openai_embeddings(texts)
        elif self.config.embedding_provider == "sentence-transformers":
            all_embeddings = await self._get_sentence_transformer_embeddings(texts)
        else:
            raise NotImplementedError(
                f"Unknown embedding provider: {self.config.embedding_provider}"
            )

        return all_embeddings[0] if single_input else all_embeddings

    async def _get_sentence_transformer_embeddings(self, texts: list[str]) -> list[list[float]]:
        from sentence_transformers import SentenceTransformer

        if self._st_model is None:
            self._st_model = SentenceTransformer(self.config.embedding_model)

        def _encode(batch: list[str]):
            return self._st_model.encode(batch).tolist()

        embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            batch_embeddings = await asyncio.to_thread(_encode, batch)
            embeddings.extend(batch_embeddings)
        return embeddings

    async def _get_openai_embeddings(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("openai is required for OpenAI embeddings.") from exc

        if not self.config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")

        client = AsyncOpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
        )

        embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            attempt = 0
            while True:
                try:
                    resp = await client.embeddings.create(
                        model=self.config.embedding_model,
                        input=batch,
                    )
                    embeddings.extend([item.embedding for item in resp.data])
                    break
                except Exception as exc:
                    attempt += 1
                    if attempt >= self.config.max_retries:
                        raise exc
                    await asyncio.sleep(1.5 * attempt)

        return embeddings

    async def get_relevant_chunks(
        self,
        query: str,
        top_k: int = 4,
        source: str | None = None,
    ) -> list[str]:
        """Retrieve top-k relevant chunks from pgvector."""
        try:
            import psycopg2
        except ImportError as exc:
            raise ImportError("psycopg2-binary is required for retrieval.") from exc

        embedding = await self.get_embeddings(query)
        vector_literal = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"

        def _query_db():
            conn = psycopg2.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                user=self.config.db_user,
                password=self.config.db_password,
                dbname=self.config.db_name,
            )
            try:
                with conn:
                    with conn.cursor() as cur:
                        if source:
                            cur.execute(
                                """
                                SELECT content
                                FROM text_chunks
                                WHERE source = %s
                                ORDER BY embedding <=> %s::vector
                                LIMIT %s
                                """,
                                (source, vector_literal, top_k),
                            )
                        else:
                            cur.execute(
                                """
                                SELECT content
                                FROM text_chunks
                                ORDER BY embedding <=> %s::vector
                                LIMIT %s
                                """,
                                (vector_literal, top_k),
                            )
                        return [row[0] for row in cur.fetchall()]
            finally:
                conn.close()

        return await asyncio.to_thread(_query_db)


def index_embeddings(
    chunks: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    source: str = "grade-5-science-and-technology",
    ensure_schema: bool = False,
) -> None:
    """Insert or update text chunks and their embeddings in Postgres."""
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError as exc:
        raise ImportError("psycopg2-binary is required to index embeddings into Postgres.") from exc

    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()

    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")

    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "gyandeep")

    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        dbname=db_name,
    )

    create_schema_sql = """
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS text_chunks (
        id BIGSERIAL PRIMARY KEY,
        source TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        embedding vector(384),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE (source, chunk_index)
    );
    """

    with conn:
        with conn.cursor() as cur:
            if ensure_schema:
                cur.execute(create_schema_sql)

            if _HAS_RUST:
                pg_literals = _rs_vec_literals([list(embeddings[i]) for i in range(len(chunks))])
            else:
                pg_literals = [
                    "[" + ",".join(f"{v:.6f}" for v in embeddings[i]) + "]"
                    for i in range(len(chunks))
                ]

            records = [
                (source, idx, chunk, pg_literals[idx])
                for idx, chunk in enumerate(chunks)
            ]

            execute_values(
                cur,
                """
                INSERT INTO text_chunks (source, chunk_index, content, embedding)
                VALUES %s
                ON CONFLICT (source, chunk_index)
                DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
                """,
                records,
                template="(%s, %s, %s, %s::vector)",
            )

    conn.close()
