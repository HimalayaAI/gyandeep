from __future__ import annotations

from pathlib import Path

import pytest

from core.services.plugins import ManimVideoPlugin, PluginJobRequest, PluginRuntime
from core.services.plugins.runtime import PluginJobResult


class _DummyInference:
    max_tokens = 1200

    @staticmethod
    def is_configured() -> bool:
        return False

    @staticmethod
    def chat_completions(messages, max_tokens=None):
        raise RuntimeError("should not be called when inference is not configured")

    @staticmethod
    def extract_response_payload(response):
        return "", ""


class _SimplePlugin:
    plugin_id = "simple"

    async def run(self, request: PluginJobRequest, emit):
        await emit("planning", "ok")
        return PluginJobResult(plan_text="done")


@pytest.mark.asyncio
async def test_plugin_runtime_routes_to_registered_handler(tmp_path: Path):
    runtime = PluginRuntime(artifact_root=tmp_path / "artifacts")
    runtime.register(_SimplePlugin())
    events: list[tuple[str, str]] = []

    async def emit(phase: str, message: str):
        events.append((phase, message))

    req = PluginJobRequest(
        job_id="job-1",
        plugin_id="simple",
        query="q",
        context_text="ctx",
        mode="environment",
        current_page=1,
        book_id=None,
        output_dir=runtime.create_job_dir("job-1"),
    )
    result = await runtime.run_job(req, emit)
    assert result.plan_text == "done"
    assert events == [("planning", "ok")]


@pytest.mark.asyncio
async def test_plugin_runtime_rejects_unknown_plugin(tmp_path: Path):
    runtime = PluginRuntime(artifact_root=tmp_path / "artifacts")
    req = PluginJobRequest(
        job_id="job-unknown",
        plugin_id="missing",
        query="q",
        context_text="ctx",
        mode="environment",
        current_page=1,
        book_id=None,
        output_dir=runtime.create_job_dir("job-unknown"),
    )

    async def emit(_phase: str, _message: str):
        return

    with pytest.raises(ValueError):
        await runtime.run_job(req, emit)


@pytest.mark.asyncio
async def test_manim_video_plugin_fallback_generates_script_and_video(tmp_path: Path):
    plugin = ManimVideoPlugin(_DummyInference(), skill_root=tmp_path / "missing-skill")
    output_dir = tmp_path / "job"
    output_dir.mkdir(parents=True, exist_ok=True)

    fake_video = output_dir / "media" / "lesson.mp4"
    fake_video.parent.mkdir(parents=True, exist_ok=True)
    fake_video.write_bytes(b"video")

    def _fake_render(script_path: Path, media_dir: Path) -> Path:
        assert script_path.exists()
        assert media_dir.exists()
        return fake_video

    plugin._render = _fake_render  # type: ignore[attr-defined]

    events = []

    async def emit(phase: str, message: str):
        events.append((phase, message))

    request = PluginJobRequest(
        job_id="job-2",
        plugin_id="manim_video",
        query="Explain quadratic roots",
        context_text="Quadratic equations and graph roots.",
        mode="environment",
        current_page=4,
        book_id=None,
        output_dir=output_dir,
    )
    result = await plugin.run(request, emit)

    assert result.script_path is not None
    assert result.video_path is not None
    script_text = Path(result.script_path).read_text(encoding="utf-8")
    assert "class LessonScene(MovingCameraScene)" in script_text
    assert Path(result.video_path).exists()
    assert events[0][0] == "planning"


def test_manim_plugin_resolves_cli_from_path(monkeypatch, tmp_path: Path):
    fake_manim = tmp_path / "manim"
    fake_manim.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_manim.chmod(0o755)

    monkeypatch.setattr("core.services.plugins.manim_video_plugin.shutil.which", lambda _name: str(fake_manim))

    resolved = ManimVideoPlugin._resolve_manim_cli()
    assert resolved == str(fake_manim)


def test_extract_python_block_handles_unclosed_fence():
    text = "```python\nfrom manim import *\nclass LessonScene(Scene):\n    pass\n"
    extracted = ManimVideoPlugin._extract_python_block(text)
    assert extracted.startswith("from manim import *")
    assert "```" not in extracted


def test_script_validation_rejects_invalid_python():
    invalid = "from manim import *\nclass LessonScene(Scene):\n    def construct(self):\n        x ="
    assert not ManimVideoPlugin._script_looks_valid(invalid, "LessonScene")


def test_fallback_script_is_valid_with_multiline_query():
    plugin = ManimVideoPlugin(_DummyInference())
    query = "could you show step-by-step\nformula for volume of sphere"
    plan = plugin._fallback_plan(query, "formula and geometry context")
    script = plugin._template_script_from_plan(query, plan)
    assert "class LessonScene(MovingCameraScene):" in script
    assert "GyanDeep" in script
    assert "camera.frame.animate" in script
    assert "Key Ideas" in script
    assert "Solution Steps" in script
    assert "clear_slide(" in script
    assert "def card(" in script
    assert plugin._script_looks_valid(script, "LessonScene")


def test_real_numbers_fallback_prefers_numberline_visuals():
    plugin = ManimVideoPlugin(_DummyInference())
    plan = plugin._fallback_plan("Explain real numbers", "Real numbers include rational and irrational values.")
    script = plugin._template_script_from_plan("Explain real numbers", plan)
    assert plan["visual_focus"] == "numberline"
    assert "NumberLine" in script
    assert "GyanDeep" in script


def test_plan_generation_returns_steps():
    plugin = ManimVideoPlugin(_DummyInference())
    plan, mode = plugin._generate_plan(
        "show area of scalene triangle with steps",
        "Area of triangle uses half times base times height.",
    )
    assert mode == "inference_unavailable_fallback"
    assert isinstance(plan.get("key_ideas"), list)
    assert 2 <= len(plan["key_ideas"]) <= 3
    assert isinstance(plan.get("steps"), list)
    assert len(plan["steps"]) == 3


def test_timing_profile_caps_wait_budget_under_one_minute():
    plugin = ManimVideoPlugin(_DummyInference())
    plan = plugin._fallback_plan("Explain simple interest", "Interest uses principal, rate, and time.")
    narration_segments = plugin._build_narration_segments("Explain simple interest", plan)
    timings = plugin._build_timing_profile(narration_segments)

    assert narration_segments
    assert sum(timings.values()) <= 40.5
    assert all(value >= 0.6 for value in timings.values())
