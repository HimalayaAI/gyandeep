"""Utilities for Gyandeep database management."""
from __future__ import annotations

import asyncio
import logging
import os

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

from core.services.storage.env_storage import EnvStorageService, STORAGE_AVAILABLE


async def setup_gyandeep_db(
    host: str = "localhost",
    port: int = 5432,
    user: str = "postgres",
    password: str = "",
    db_name: str = "gyandeep",
) -> bool:
    """Initialize the Gyandeep database and tables."""
    if not STORAGE_AVAILABLE:
        print("Error: storage dependencies not available. Cannot setup database.")
        return False

    storage = EnvStorageService(
        host=host,
        port=port,
        user=user,
        password=password,
        db_name=db_name,
    )

    try:
        await storage.initialize()
        print(f"Successfully initialized Gyandeep database: {db_name}")
        await storage.close()
        return True
    except Exception as exc:
        print(f"Failed to setup database: {exc}")
        return False


async def check_database_status(
    host: str = "localhost",
    port: int = 5432,
    user: str = "postgres",
    password: str = "",
    db_name: str = "gyandeep",
) -> None:
    """Print status and counts for Gyandeep database tables."""
    if not STORAGE_AVAILABLE:
        print("Error: Storage dependencies not available.")
        return

    storage = EnvStorageService(
        host=host,
        port=port,
        user=user,
        password=password,
        db_name=db_name,
    )

    try:
        await storage.initialize()
        db = storage._db
        if db is None:
            print("Database not initialized.")
            return

        tables = ["students", "learning_events", "text_chunks"]
        print(f"\n--- Gyandeep Database Table Counts ({db_name}) ---")
        for table in tables:
            try:
                res = await db.fetch_one(f"SELECT COUNT(*) as count FROM {table}")
                count = res["count"] if res else 0
                print(f"{table:20}: {count}")
            except Exception:
                print(f"{table:20}: TABLE NOT FOUND")

        await storage.close()
    except Exception as exc:
        print(f"Database check failed: {exc}")


async def test_pg_vector_support(
    host: str = "localhost",
    port: int = 5432,
    user: str = "postgres",
    password: str = "",
    db_name: str = "gyandeep",
) -> bool:
    """Test if pgvector extension is available and working."""
    if not HAS_ASYNCPG:
        print("Error: asyncpg not installed.")
        return False

    try:
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=db_name,
        )

        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        res = await conn.fetchval("SELECT '[1,2,3]'::vector <=> '[3,2,1]'::vector")
        print(f"pgvector check successful. Sample distance: {res}")

        await conn.close()
        return True
    except Exception as exc:
        print(f"pgvector test failed: {exc}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    host = os.environ.get("DB_HOST", "localhost")
    port = int(os.environ.get("DB_PORT", 5432))
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "")
    db_name = os.environ.get("DB_NAME", "gyandeep")

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        asyncio.run(check_database_status(host, port, user, password, db_name))
    else:
        print(f"Setting up database at {host}:{port}...")
        asyncio.run(setup_gyandeep_db(host, port, user, password, db_name))
