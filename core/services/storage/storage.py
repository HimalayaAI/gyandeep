"""Storage service abstractions for Gyandeep."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import UUID

class StorageService(ABC):
    """Abstract base class for storage services."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend (connections, schema, etc.)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend."""
        ...

    @abstractmethod
    def create_session(self, student_id: Optional[UUID] = None) -> "StorageSession":
        """Create a scoped session for recording related entities."""
        ...


class StorageSession(ABC):
    """Abstract base class for storage sessions."""
    student_id: Optional[UUID] = None

    @abstractmethod
    async def store_learning_event(
        self,
        event_type: str,
        prompt: str,
        response: str,
        score: Optional[float] = None,
        metadata: Optional[dict] = None,
        student_id: Optional[UUID] = None,
    ) -> UUID:
        """Store a learning interaction."""
        ...

    @abstractmethod
    async def store_text_chunks(
        self,
        source: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Store or update text chunks with embeddings."""
        ...
