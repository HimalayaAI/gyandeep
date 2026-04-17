from __future__ import annotations

import asyncio
import ast
import json
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from textwrap import dedent

from .runtime import EmitFn, PluginJobRequest, PluginJobResult


class ManimVideoPlugin:
    plugin_id = "manim_video"

    def __init__(
        self,
        inference_service,
        skill_root: str | Path = "manim-video",
        quality: str = "l",
        render_timeout_seconds: int = 180,
    ):
        self._inference = inference_service
        self._skill_root = Path(skill_root)
        self._quality = quality
        self._render_timeout_seconds = render_timeout_seconds
        self._scene_name = "LessonScene"

    @staticmethod
    def _clip(text: str, limit: int = 220) -> str:
        text = re.sub(r"\s+", " ", str(text or "").strip())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    def _load_skill_context(self) -> str:
        skill_doc = self._skill_root / "SKILL.md"
        scene_doc = self._skill_root / "references" / "scene-planning.md"

        sections: list[str] = []
        if skill_doc.exists():
            sections.append(skill_doc.read_text(encoding="utf-8")[:5000])
        if scene_doc.exists():
            sections.append(scene_doc.read_text(encoding="utf-8")[:2500])
        return "\n\n".join(sections).strip()

    def _plan_prompt(self, query: str, context_text: str, style_context: str) -> str:
        return dedent(
            f"""
            You are a lesson planner for a Nepali high-school tutoring app.
            Use this style guide context:
            {style_context}

            Student question:
            {query}

            Textbook context:
            {context_text}

            Build a concrete teaching blueprint focused on solving the question, not meta commentary.
            Return ONLY a JSON object (no markdown) with keys:
            - "title": short lesson title
            - "learning_goal": one sentence
            - "formula_latex": core formula in latex-like text (or plain formula if unsure)
            - "steps": array of 4 concise actionable steps
            - "worked_example": array of 3 to 5 short solution lines
            - "visual_focus": one of "triangle", "circle", "algebra", "generic"
            - "answer_line": one sentence with the direct answer idea
            """
        ).strip()

    def _script_prompt_from_plan(
        self,
        query: str,
        context_text: str,
        style_context: str,
        plan: dict[str, object],
    ) -> str:
        plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
        return dedent(
            f"""
            You are generating a Manim Community Edition script for a high-school tutoring app.
            Use this style guide context:
            {style_context}

            Student query:
            {query}

            Structured teaching blueprint:
            {plan_json}

            Source context:
            {context_text}

            Requirements:
            - Return only valid Python code in one ```python fenced block.
            - Code must include `from manim import *`.
            - Define class `{self._scene_name}(Scene)`.
            - The animation must teach the actual solution flow (formula + worked example), not just planning text.
            - Keep script robust for low-quality render (`-ql`) and avoid fragile APIs.
            - Use readable text sizes (title >= 42, body >= 30).
            - Include at least 5 explicit `self.wait(...)` pauses.
            - Keep total duration around 18 to 45 seconds.
            - Prefer Text/MathTex, FadeIn/Write/Transform/Create only.
            """
        ).strip()

    @staticmethod
    def _extract_python_block(text: str) -> str:
        text = (text or "").strip()
        match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()

        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(line for line in lines if line.strip() not in {"```", "```python"}).strip()
        return cleaned or text.strip()

    @staticmethod
    def _extract_json_object(text: str) -> dict | None:
        text = (text or "").strip()
        if not text:
            return None

        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]

        try:
            payload = json.loads(text)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _wrap_text(text: str, width: int = 34, max_lines: int = 3) -> str:
        text = re.sub(r"\s+", " ", text.strip())
        lines = textwrap.wrap(text, width=width)
        if not lines:
            return ""
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            if not lines[-1].endswith("..."):
                lines[-1] = lines[-1].rstrip(".") + "..."
        return "\n".join(lines)

    @staticmethod
    def _latex_to_text(expr: str) -> str:
        expr = (expr or "").strip().strip("$")
        expr = expr.replace("\\times", "×")
        expr = expr.replace("\\cdot", "·")
        expr = expr.replace("\\pi", "π")
        expr = expr.replace("\\sqrt", "sqrt")
        expr = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", expr)
        expr = expr.replace("{", "").replace("}", "")
        expr = expr.replace("\\", "")
        expr = re.sub(r"\s+", " ", expr).strip()
        return expr or "Use the core formula from the lesson."

    def _fallback_formula(self, query: str, context_text: str) -> str:
        scope = f"{query} {context_text}".lower()
        if "scalene" in scope and "area" in scope:
            return r"A = \frac{1}{2} b h"
        if "area" in scope and "triangle" in scope:
            return r"A = \frac{1}{2} b h"
        if "volume" in scope and "sphere" in scope:
            return r"V = \frac{4}{3}\pi r^3"
        if "pythag" in scope or "right triangle" in scope:
            return r"c^2 = a^2 + b^2"
        if "simple interest" in scope or "interest" in scope:
            return r"I = P r t"
        return "Write the key formula first, then substitute values step by step."

    def _fallback_plan(self, query: str, context_text: str) -> dict[str, object]:
        scope = f"{query} {context_text}".lower()
        visual_focus = "generic"
        if "triangle" in scope:
            visual_focus = "triangle"
        elif "circle" in scope or "sphere" in scope:
            visual_focus = "circle"
        elif "equation" in scope or "algebra" in scope:
            visual_focus = "algebra"

        formula_latex = self._fallback_formula(query, context_text)
        query_line = self._clip(query, 78)
        return {
            "title": "Step-by-step concept walkthrough",
            "learning_goal": f"Solve: {query_line}",
            "formula_latex": formula_latex,
            "steps": [
                "Identify the known values and what must be found.",
                "Write the core formula clearly before calculation.",
                "Substitute values carefully and simplify line by line.",
                "Check units and verify the final answer is reasonable.",
            ],
            "worked_example": [
                "Given values from the question.",
                f"Use formula: {self._latex_to_text(formula_latex)}",
                "Compute each step and present the final result clearly.",
            ],
            "visual_focus": visual_focus,
            "answer_line": "The answer follows from applying the formula step by step.",
        }

    def _normalize_plan(
        self,
        raw_plan: dict | None,
        query: str,
        context_text: str,
    ) -> dict[str, object]:
        base = self._fallback_plan(query, context_text)
        if not raw_plan:
            return base

        plan = dict(base)
        for field in ("title", "learning_goal", "formula_latex", "visual_focus", "answer_line"):
            value = raw_plan.get(field)
            if isinstance(value, str) and value.strip():
                plan[field] = self._clip(value, 220)

        for field in ("steps", "worked_example"):
            value = raw_plan.get(field)
            cleaned: list[str] = []
            if isinstance(value, list):
                cleaned = [self._clip(str(item), 180) for item in value if str(item).strip()]
            elif isinstance(value, str) and value.strip():
                pieces = re.split(r"(?:\n+|•|- )", value.strip())
                cleaned = [self._clip(piece, 180) for piece in pieces if piece.strip()]
            if cleaned:
                plan[field] = cleaned[:6] if field == "steps" else cleaned[:5]

        visual = str(plan.get("visual_focus", "generic")).strip().lower()
        if visual not in {"triangle", "circle", "algebra", "generic"}:
            visual = "generic"
        plan["visual_focus"] = visual

        formula = str(plan.get("formula_latex", "")).strip()
        if not formula:
            plan["formula_latex"] = self._fallback_formula(query, context_text)

        if not plan["steps"]:
            plan["steps"] = base["steps"]
        if not plan["worked_example"]:
            plan["worked_example"] = base["worked_example"]
        return plan

    def _generate_plan(self, query: str, context_text: str) -> tuple[dict[str, object], str]:
        if not self._inference.is_configured():
            return self._fallback_plan(query, context_text), "inference_unavailable_fallback"

        style_context = self._load_skill_context()
        prompt = self._plan_prompt(query=query, context_text=context_text, style_context=style_context)
        try:
            response = self._inference.chat_completions(
                [{"role": "user", "content": prompt}],
                max_tokens=min(900, max(500, self._inference.max_tokens)),
            )
            content, _reasoning = self._inference.extract_response_payload(response)
            parsed = self._extract_json_object(content)
            return self._normalize_plan(parsed, query, context_text), "llm_plan"
        except Exception:
            return self._fallback_plan(query, context_text), "plan_fallback"

    def _plan_to_markdown(
        self,
        request: PluginJobRequest,
        plan: dict[str, object],
        plan_mode: str,
    ) -> str:
        steps = plan.get("steps") or []
        worked = plan.get("worked_example") or []
        step_lines = "\n".join([f"{idx + 1}. {item}" for idx, item in enumerate(steps)])
        worked_lines = "\n".join([f"- {item}" for item in worked])
        return dedent(
            f"""
            # DeepGyan Animation Plan

            - Plugin: `{request.plugin_id}`
            - Mode: `{request.mode}`
            - Focus page: `{request.current_page}`
            - Plan source: `{plan_mode}`
            - Query: {request.query}

            ## Title
            {plan.get("title", "")}

            ## Learning Goal
            {plan.get("learning_goal", "")}

            ## Formula
            {plan.get("formula_latex", "")}

            ## Steps
            {step_lines}

            ## Worked Example
            {worked_lines}

            ## Answer Line
            {plan.get("answer_line", "")}
            """
        ).strip()

    def _template_script_from_plan(self, query: str, plan: dict[str, object]) -> str:
        title = self._clip(str(plan.get("title", "") or "DeepGyan Animation"), 70)
        learning_goal = self._wrap_text(self._clip(str(plan.get("learning_goal", "") or query), 95), 40, 3)
        formula_text = self._wrap_text(
            self._latex_to_text(str(plan.get("formula_latex", "") or "")),
            width=34,
            max_lines=3,
        )

        steps = [self._clip(str(item), 95) for item in (plan.get("steps") or []) if str(item).strip()]
        if len(steps) < 4:
            steps = [self._clip(str(item), 95) for item in self._fallback_plan(query, "").get("steps", [])]
        steps = steps[:4]

        worked = [self._clip(str(item), 95) for item in (plan.get("worked_example") or []) if str(item).strip()]
        if len(worked) < 3:
            worked = [self._clip(str(item), 95) for item in self._fallback_plan(query, "").get("worked_example", [])]
        worked = worked[:3]

        visual_focus = str(plan.get("visual_focus", "generic"))

        lines = [
            "from manim import *",
            "",
            f"class {self._scene_name}(Scene):",
            "    def construct(self):",
            "        self.camera.background_color = '#0D1326'",
            "",
            f"        title = Text({repr(title)}, font_size=46, color=BLUE_B)",
            f"        goal = Text({repr(learning_goal)}, font_size=30).scale_to_fit_width(12.5)",
            f"        formula = Text({repr(formula_text)}, font_size=34, color=GREEN_B).scale_to_fit_width(12.5)",
            "        title.to_edge(UP, buff=0.45)",
            "        goal.next_to(title, DOWN, buff=0.35)",
            "",
            "        self.play(FadeIn(title, shift=UP * 0.25), run_time=1.0)",
            "        self.play(FadeIn(goal, shift=UP * 0.2), run_time=1.2)",
            "        self.wait(0.9)",
            "",
            "        visual_group = VGroup()",
        ]

        if visual_focus == "triangle":
            lines.extend(
                [
                    "        tri = Polygon(LEFT * 2.8 + DOWN * 1.5, RIGHT * 2.8 + DOWN * 1.5, UP * 1.8, color=YELLOW)",
                    "        base_label = Text('base b', font_size=24, color=YELLOW).next_to(tri, DOWN, buff=0.2)",
                    "        h_line = DashedLine(UP * 1.8, UP * 1.8 + DOWN * 3.3, color=BLUE_B)",
                    "        h_label = Text('height h', font_size=24, color=BLUE_B).next_to(h_line, RIGHT, buff=0.2)",
                    "        visual_group = VGroup(tri, base_label, h_line, h_label).scale(0.65)",
                ]
            )
        elif visual_focus == "circle":
            lines.extend(
                [
                    "        circle = Circle(radius=2.0, color=YELLOW)",
                    "        radius = Line(ORIGIN, RIGHT * 2.0, color=BLUE_B)",
                    "        r_label = Text('r', font_size=28, color=BLUE_B).next_to(radius, UP, buff=0.2)",
                    "        visual_group = VGroup(circle, radius, r_label).scale(0.85)",
                ]
            )
        elif visual_focus == "algebra":
            lines.extend(
                [
                    "        x_box = SurroundingRectangle(Text('x', font_size=30), color=YELLOW, buff=0.25)",
                    "        eq_hint = Text('Solve for unknown', font_size=26, color=YELLOW).next_to(x_box, DOWN, buff=0.3)",
                    "        visual_group = VGroup(x_box, eq_hint)",
                ]
            )
        else:
            lines.extend(
                [
                    "        helper = RoundedRectangle(corner_radius=0.2, width=5.8, height=2.2, color=YELLOW)",
                    "        helper_text = Text('Visual model', font_size=28, color=YELLOW).move_to(helper)",
                    "        visual_group = VGroup(helper, helper_text)",
                ]
            )

        lines.extend(
            [
                "        visual_group.to_edge(RIGHT, buff=0.8).shift(DOWN * 0.35)",
                "        self.play(Create(visual_group), run_time=1.4)",
                "        self.wait(0.9)",
                "",
                "        formula_header = Text('Key Formula', font_size=30, color=YELLOW).next_to(goal, DOWN, buff=0.55).to_edge(LEFT, buff=0.8)",
                "        formula.next_to(formula_header, DOWN, buff=0.3).to_edge(LEFT, buff=0.8)",
                "        self.play(Write(formula_header), run_time=0.8)",
                "        self.play(FadeIn(formula, shift=UP * 0.2), run_time=1.0)",
                "        self.wait(1.2)",
                "",
                "        steps_header = Text('Solution Steps', font_size=30, color=TEAL_B).next_to(formula, DOWN, buff=0.5).to_edge(LEFT, buff=0.8)",
                "        self.play(Write(steps_header), run_time=0.7)",
                "",
                f"        step_text = Text({repr(self._wrap_text(steps[0], 44, 3))}, font_size=30).scale_to_fit_width(8.5)",
                "        step_text.next_to(steps_header, DOWN, buff=0.28).to_edge(LEFT, buff=0.8)",
                "        self.play(FadeIn(step_text, shift=UP * 0.12), run_time=0.9)",
                "        self.wait(1.3)",
            ]
        )

        for idx, step in enumerate(steps[1:], start=2):
            lines.extend(
                [
                    f"        step_{idx} = Text({repr(self._wrap_text(step, 44, 3))}, font_size=30).scale_to_fit_width(8.5).move_to(step_text)",
                    f"        self.play(Transform(step_text, step_{idx}), run_time=1.0)",
                    "        self.wait(1.3)",
                ]
            )

        lines.extend(
            [
                "",
                "        worked_header = Text('Worked Example', font_size=30, color=ORANGE).next_to(step_text, DOWN, buff=0.5).to_edge(LEFT, buff=0.8)",
                "        self.play(Write(worked_header), run_time=0.8)",
                "",
                f"        w1 = Text({repr(self._wrap_text(worked[0], 44, 2))}, font_size=28).scale_to_fit_width(8.5)",
                "        w1.next_to(worked_header, DOWN, buff=0.25).to_edge(LEFT, buff=0.8)",
                f"        w2 = Text({repr(self._wrap_text(worked[1], 44, 2))}, font_size=28).scale_to_fit_width(8.5).next_to(w1, DOWN, buff=0.2).to_edge(LEFT, buff=0.8)",
                f"        w3 = Text({repr(self._wrap_text(worked[2], 44, 2))}, font_size=28, color=GREEN_B).scale_to_fit_width(8.5).next_to(w2, DOWN, buff=0.2).to_edge(LEFT, buff=0.8)",
                "        self.play(FadeIn(w1, shift=UP * 0.08), run_time=0.8)",
                "        self.wait(0.8)",
                "        self.play(FadeIn(w2, shift=UP * 0.08), run_time=0.8)",
                "        self.wait(0.8)",
                "        self.play(FadeIn(w3, shift=UP * 0.08), run_time=0.8)",
                "        self.wait(1.8)",
            ]
        )
        return "\n".join(lines).strip()

    @staticmethod
    def _script_looks_valid(script: str, scene_name: str) -> bool:
        if "from manim import" not in script:
            return False
        if f"class {scene_name}(Scene)" not in script:
            return False
        if script.count("self.wait(") < 2:
            return False
        try:
            ast.parse(script)
        except SyntaxError:
            return False
        return True

    def _generate_script(self, query: str, context_text: str, plan: dict[str, object]) -> tuple[str, str]:
        if not self._inference.is_configured():
            return self._template_script_from_plan(query, plan), "template_from_plan"

        style_context = self._load_skill_context()
        prompt = self._script_prompt_from_plan(
            query=query,
            context_text=context_text,
            style_context=style_context,
            plan=plan,
        )
        try:
            response = self._inference.chat_completions(
                [{"role": "user", "content": prompt}],
                max_tokens=min(1700, max(900, self._inference.max_tokens)),
            )
            content, _reasoning = self._inference.extract_response_payload(response)
            script = self._extract_python_block(content)
            if self._script_looks_valid(script, self._scene_name):
                return script, "llm_script_from_plan"
        except Exception:
            pass
        return self._template_script_from_plan(query, plan), "template_script_fallback"

    def _render(self, script_path: Path, media_dir: Path) -> Path:
        media_dir.mkdir(parents=True, exist_ok=True)
        manim_cli = self._resolve_manim_cli()
        command = [
            manim_cli,
            f"-q{self._quality}",
            str(script_path),
            self._scene_name,
            "-o",
            "lesson.mp4",
            "--media_dir",
            str(media_dir),
        ]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self._render_timeout_seconds,
            check=False,
        )
        if process.returncode != 0:
            stderr = (process.stderr or "").strip()
            stdout = (process.stdout or "").strip()
            details = stderr or stdout or "Unknown manim render failure."
            raise RuntimeError(details[:1200])

        candidates = sorted(media_dir.rglob("lesson.mp4"))
        if not candidates:
            candidates = sorted(media_dir.rglob("*.mp4"))
        if not candidates:
            raise RuntimeError("Render completed but no output video was found.")
        return candidates[-1]

    @staticmethod
    def _resolve_manim_cli() -> str:
        direct = shutil.which("manim")
        if direct:
            return direct

        sibling = Path(sys.executable).resolve().parent / "manim"
        if sibling.exists():
            return str(sibling)

        raise RuntimeError(
            "Manim CLI not found. Install manim in this environment and ensure `manim` is available in PATH."
        )

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
