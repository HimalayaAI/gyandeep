"""Pydantic models for GyanDeep domain entities."""

from core.models.plugins import PluginJobRequest, PluginJobResult
from core.models.storage import Book, LearningEvent, OCRPage, Student, TextChunk

__all__ = [
    "Book",
    "LearningEvent",
    "OCRPage",
    "PluginJobRequest",
    "PluginJobResult",
    "Student",
    "TextChunk",
]
