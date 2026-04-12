"""Robust asynchronous database wrapper for Gyandeep."""

from __future__ import annotations

import asyncio
import random
import logging
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union, AsyncIterator

try:
    import asyncpg
    from asyncpg import Pool, create_pool
    from asyncpg.exceptions import DuplicateDatabaseError
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

logger = logging.getLogger("gyandeep.storage.db")

class AsyncDatabase:
    """Asynchronous database wrapper using asyncpg with auto-creation support."""

    def __init__(self, config: Any):
        self.config = config
        self.pool: Optional[Pool] = None
        self._lock = asyncio.Lock()
        self._is_initialized = False
        
        # Default retry config if not present in config
        self.retry_config = {
            'initial_delay': getattr(self.config, 'retry_delay', 1.0),
            'max_delay': getattr(self.config, 'retry_max_delay', 10.0),
            'backoff_factor': getattr(self.config, 'retry_backoff_factor', 2.0),
            'jitter': getattr(self.config, 'retry_jitter', 0.1)
        }

    async def initialize(self) -> None:
        """Initialize connection pool and verify database setup."""
        if not HAS_ASYNCPG:
            raise ImportError("asyncpg is required for AsyncDatabase")
        
        if self._is_initialized:
            return

        async with self._lock:
            if self._is_initialized:
                return

            await self._ensure_database_exists()
            
            try:
                self.pool = await create_pool(
                    host=self.config.host,
                    port=self.config.port,
                    user=self.config.user,
                    password=self.config.password,
                    database=self.config.db_name,
                    min_size=getattr(self.config, 'pool_min', 2),
                    max_size=getattr(self.config, 'pool_max', 10),
                    timeout=30,
                    command_timeout=60,
                )
                await self._verify_extensions()
                self._is_initialized = True
                logger.info(f"Database pool initialized for {self.config.db_name}")
            except Exception as e:
                logger.error(f"Failed to initialize database pool: {e}")
                raise

    async def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
        self._is_initialized = False

    async def is_connected(self) -> bool:
        """Check if database connection is alive."""
        try:
            if not self.pool:
                return False
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False

    async def _ensure_database_exists(self) -> None:
        """Create database if it doesn't exist."""
        try:
            # Connect to maintenance database to check/create target DB
            temp_conn = await asyncpg.connect(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database='postgres'
            )

            try:
                await temp_conn.execute(f"CREATE DATABASE {self.config.db_name}")
                logger.info(f"Created database {self.config.db_name}")
            except DuplicateDatabaseError:
                pass
            finally:
                await temp_conn.close()
        except Exception as e:
            logger.error(f"Database creation failed: {e}")
            # Don't re-raise if it's just a permission issue on 'postgres' db
            # as the target DB might already exist.

    async def _verify_extensions(self) -> None:
        """Ensure required PostgreSQL extensions are installed."""
        if self.pool:
            async with self.pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a SQL command and return status."""
        if not self.pool:
            await self.initialize()
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> List[asyncpg.Record]:
        """Execute a query and return results."""
        if not self.pool:
            await self.initialize()
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetch_one(self, query: str, *args: Any) -> Optional[asyncpg.Record]:
        """Execute a query and return a single result."""
        if not self.pool:
            await self.initialize()
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    @asynccontextmanager
    async def transaction(self, isolation: str = "repeatable_read") -> AsyncIterator[asyncpg.Connection]:
        """Transaction context manager."""
        if not self.pool:
            await self.initialize()
        async with self.pool.acquire() as conn:
            async with conn.transaction(isolation=isolation):
                yield conn

    @asynccontextmanager
    async def safe_transaction(self, max_retries: int = 3) -> AsyncIterator[asyncpg.Connection]:
        """Retryable transaction."""
        last_error = None
        for attempt in range(max_retries):
            try:
                async with self.transaction() as conn:
                    yield conn
                    return
            except Exception as e:
                last_error = e
                logger.error(f"Transaction failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(self._calculate_backoff(attempt))
                else:
                    raise last_error

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        current_delay = self.retry_config['initial_delay'] * \
                        (self.retry_config['backoff_factor'] ** attempt)
        jitter_amount = current_delay * self.retry_config['jitter'] * random.uniform(-1, 1)
        next_delay = current_delay + jitter_amount
        return max(0, min(next_delay, self.retry_config['max_delay']))
