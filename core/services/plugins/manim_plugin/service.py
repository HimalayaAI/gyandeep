from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from ..runtime import EmitFn, PluginJobRequest, PluginJobResult
from .planning import build_plan_prompt, fallback_plan, normalize_plan, plan_to_markdown
from .rendering import render_manim
from .scripting import build_script_prompt, script_looks_valid, template_script_from_plan
from .text_utils import extract_json_object, extract_python_block


class ManimVideoService:
    plugin_id = "manim_video"

    def __init__(
        self,
        inference_service,
        skill_root: str | Path = "manim-video",
        quality: str = "l",
        render_timeout_seconds: int = 180,
        scene_name: str = "LessonScene",
    ):
        self._inference = inference_service
        self._skill_root = Path(skill_root)
        self._quality = quality
        self._render_timeout_seconds = render_timeout_seconds
        self._scene_name = scene_name

    def _load_skill_context(self) -> str:
        skill_doc = self._skill_root / "SKILL.md"
        scene_doc = self._skill_root / "references" / "scene-planning.md"

        sections: list[str] = []
        if skill_doc.exists():
            sections.append(skill_doc.read_text(encoding="utf-8")[:5000])
        if scene_doc.exists():
            sections.append(scene_doc.read_text(encoding="utf-8")[:2500])
        return "\n\n".join(sections).strip()

    def _generate_plan(self, query: str, context_text: str) -> tuple[dict[str, object], str]:
        if not self._inference.is_configured():
            return fallback_plan(query, context_text, scene_name=self._scene_name), "inference_unavailable_fallback"

        style_context = self._load_skill_context()
        prompt = build_plan_prompt(query=query, context_text=context_text, style_context=style_context)
        try:
            response = self._inference.chat_completions(
                [{"role": "user", "content": prompt}],
                max_tokens=min(900, max(500, self._inference.max_tokens)),
            )
            content, _reasoning = self._inference.extract_response_payload(response)
            parsed = extract_json_object(content)
            return normalize_plan(parsed, query, context_text, scene_name=self._scene_name), "llm_plan"
        except Exception:
            return fallback_plan(query, context_text, scene_name=self._scene_name), "plan_fallback"

    def _plan_to_markdown(self, request: PluginJobRequest, plan: dict[str, object], plan_mode: str) -> str:
        return plan_to_markdown(request, plan, plan_mode)

    def _template_script_from_plan(self, query: str, plan: dict[str, object]) -> str:
        return template_script_from_plan(query, plan, scene_name=self._scene_name)

    def _script_looks_valid(self, script: str, scene_name: str | None = None) -> bool:
        return script_looks_valid(script, scene_name or self._scene_name)

    def _generate_script(self, query: str, context_text: str, plan: dict[str, object]) -> tuple[str, str]:
        if not self._inference.is_configured():
            return self._template_script_from_plan(query, plan), "template_from_plan"

        style_context = self._load_skill_context()
        prompt = build_script_prompt(
            query=query,
            context_text=context_text,
            style_context=style_context,
            plan=plan,
            scene_name=self._scene_name,
        )
        try:
            response = self._inference.chat_completions(
                [{"role": "user", "content": prompt}],
                max_tokens=min(1700, max(900, self._inference.max_tokens)),
            )
            content, _reasoning = self._inference.extract_response_payload(response)
            script = extract_python_block(content)
            if self._script_looks_valid(script, self._scene_name):
                return script, "llm_script_from_plan"
        except Exception:
            pass
        return self._template_script_from_plan(query, plan), "template_script_fallback"

    def _render(self, script_path: Path, media_dir: Path) -> Path:
        # Render in a path-safe temporary directory first. Manim/ffmpeg can be
        # brittle when the project lives in a directory with spaces or quotes.
        with tempfile.TemporaryDirectory(prefix="manim_render_") as tmp_dir:
            temp_media_dir = Path(tmp_dir)
            rendered_video = render_manim(
                script_path=script_path,
                media_dir=temp_media_dir,
                scene_name=self._scene_name,
                quality=self._quality,
                timeout_seconds=self._render_timeout_seconds,
            )

            media_dir.mkdir(parents=True, exist_ok=True)
            final_video = media_dir / rendered_video.name
            shutil.copy2(rendered_video, final_video)
            return final_video

    async def run(self, request: PluginJobRequest, emit: EmitFn) -> PluginJobResult:
        await emit("planning", "Building solution blueprint from textbook context...")
        plan, plan_mode = await asyncio.to_thread(
            self._generate_plan,
            request.query,
            request.context_text,
        )
        plan_text = self._plan_to_markdown(request, plan, plan_mode)
        (request.output_dir / "plan.md").write_text(plan_text, encoding="utf-8")
        await emit("planning", f"Blueprint ready ({plan_mode}).")

        await emit("scripting", "Generating Manim scene from blueprint...")
        script, generation_mode = await asyncio.to_thread(
            self._generate_script,
            request.query,
            request.context_text,
            plan,
        )
        script_path = request.output_dir / "script.py"
        script_path.write_text(script, encoding="utf-8")
        await emit("scripting", f"Script ready ({generation_mode}).")

        await emit("rendering", "Rendering draft animation (quality=low)...")
        video_path = await asyncio.to_thread(
            self._render,
            script_path,
            request.output_dir / "media",
        )
        await emit("rendering", "Render finished.")

        return PluginJobResult(
            plan_text=plan_text,
            script_path=str(script_path.resolve()),
            video_path=str(video_path.resolve()),
        )
