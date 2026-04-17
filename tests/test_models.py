from uuid import UUID

import pytest

from pathlib import Path

from core.models.plugins import PluginJobRequest
from core.models.storage import Book, LearningEvent, OCRPage, Student, TextChunk


def test_book_trims_filename():
    book = Book(filename="  sample.pdf  ")
    assert book.filename == "sample.pdf"


def test_student_grade_range():
    student = Student(name="Sita", grade=9)
    assert student.grade == 9

    with pytest.raises(ValueError):
        Student(name="Ram", grade=0)


def test_ocr_page_requires_content():
    with pytest.raises(ValueError):
        OCRPage(book_id=UUID("12345678-1234-5678-1234-567812345678"), page_index=0, content="  ")


def test_learning_event_score_must_be_finite():
    event = LearningEvent(event_type="quiz", score=0.9)
    assert event.score == 0.9

    with pytest.raises(ValueError):
        LearningEvent(event_type="quiz", score=float("nan"))


def test_text_chunk_embedding_dim():
    chunk = TextChunk(source="book:1", chunk_index=0, content="hello", embedding=[0.0] * 384)
    assert len(chunk.embedding) == 384

    with pytest.raises(ValueError):
        TextChunk(source="book:1", chunk_index=0, content="hello", embedding=[0.0] * 10)


def test_plugin_job_request_trims_and_validates():
    job = PluginJobRequest(
        job_id="  job-1 ",
        plugin_id=" manim_video ",
        query=" explain area ",
        context_text="ctx",
        mode="Environment",
        current_page=3,
        output_dir=Path("/tmp/artifacts"),
    )
    assert job.job_id == "job-1"
    assert job.plugin_id == "manim_video"
    assert job.query == "explain area"
    assert job.mode == "environment"


def test_plugin_job_request_rejects_invalid_mode():
    with pytest.raises(ValueError):
        PluginJobRequest(
            job_id="job-2",
            plugin_id="manim_video",
            query="q",
            context_text="ctx",
            mode="invalid",
            current_page=1,
            output_dir=Path("/tmp/artifacts"),
        )
