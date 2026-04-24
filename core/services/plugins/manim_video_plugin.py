from __future__ import annotations

import asyncio
import ast
import importlib.util
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
            - "key_ideas": array of 2 or 3 short anchor ideas
            - "steps": array of exactly 3 concise actionable steps
            - "worked_example": array of 3 or 4 short solution lines
            - "visual_focus": one of "triangle", "circle", "algebra", "numberline", "generic"
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
            - Define class `{self._scene_name}(MovingCameraScene)`.
            - Keep all essential content inside safe margins; do not place important text at the bottom edge.
            - Use `self.camera.frame.animate...` to follow the active writing area instead of stacking one tall page of text.
            - Replace any placeholder "Visual Model" box with a small top-right badge that says `GyanDeep`.
            - Use at least two non-text visuals such as a diagram, number line, dots, arrows, shaded regions, or geometric shapes.
            - Break long explanations into short beats; avoid giant text walls and avoid overlap between text and visuals.
            - The animation must teach the actual solution flow with `Step 1`, `Step 2`, `Step 3`, a separate `Key Ideas` area, and a worked example shown as line-by-line math text instead of stacked boxes.
            - Keep script robust for low-quality render (`-ql`) and avoid fragile APIs.
            - Use readable text sizes (title >= 40, body >= 26).
            - Include at least 5 explicit `self.wait(...)` pauses.
            - Keep total duration under 60 seconds.
            - Prefer MovingCameraScene, Text/MathTex, NumberLine, Dot, Line, Polygon, Circle, RoundedRectangle, VGroup, FadeIn, Write, Transform, Create, and `.animate`.
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
        if "real number" in scope or "real numbers" in scope:
            return "Real numbers = rational numbers + irrational numbers"
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
        if "real number" in scope or "real numbers" in scope:
            return {
                "title": "Understanding real numbers",
                "learning_goal": "See how rational and irrational numbers together make the real number system.",
                "formula_latex": "Real numbers = rational numbers + irrational numbers",
                "key_ideas": [
                    "Every real number can be placed on the number line.",
                    "Rational and irrational numbers are both real numbers.",
                    "The real-number system combines both groups into one set.",
                ],
                "steps": [
                    "Mark a few points on the number line to show where real numbers live.",
                    "Classify sample values into rational numbers and irrational numbers.",
                    "Conclude that both groups together form the complete set of real numbers.",
                ],
                "worked_example": [
                    "-2 and 5 are rational because each can be written as a fraction.",
                    "1/2 is rational, while sqrt(2) and pi are irrational values.",
                    "All four values are real because each one has a point on the number line.",
                ],
                "visual_focus": "numberline",
                "answer_line": "Real numbers are all numbers on the number line, including both rational and irrational numbers.",
            }

        visual_focus = "generic"
        if "triangle" in scope:
            visual_focus = "triangle"
        elif "circle" in scope or "sphere" in scope:
            visual_focus = "circle"
        elif "equation" in scope or "algebra" in scope:
            visual_focus = "algebra"
        elif "number line" in scope or "fraction" in scope or "integer" in scope:
            visual_focus = "numberline"

        formula_latex = self._fallback_formula(query, context_text)
        query_line = self._clip(query, 78)
        return {
            "title": "Step-by-step concept walkthrough",
            "learning_goal": f"Solve: {query_line}",
            "formula_latex": formula_latex,
            "key_ideas": [
                "Read the givens before choosing a method.",
                "Use one core rule or formula to drive the solution.",
                "End with a checked final answer.",
            ],
            "steps": [
                "Identify the known values and what must be found.",
                "Write the core formula clearly before calculation.",
                "Substitute values carefully and simplify to the final answer.",
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

        for field in ("key_ideas", "steps", "worked_example"):
            value = raw_plan.get(field)
            cleaned: list[str] = []
            if isinstance(value, list):
                cleaned = [self._clip(str(item), 180) for item in value if str(item).strip()]
            elif isinstance(value, str) and value.strip():
                pieces = re.split(r"(?:\n+|•|- )", value.strip())
                cleaned = [self._clip(piece, 180) for piece in pieces if piece.strip()]
            if cleaned:
                if field == "key_ideas":
                    plan[field] = cleaned[:3]
                elif field == "steps":
                    plan[field] = cleaned[:3]
                else:
                    plan[field] = cleaned[:4]

        visual = str(plan.get("visual_focus", "generic")).strip().lower()
        if visual not in {"triangle", "circle", "algebra", "numberline", "generic"}:
            visual = "generic"
        plan["visual_focus"] = visual

        formula = str(plan.get("formula_latex", "")).strip()
        if not formula:
            plan["formula_latex"] = self._fallback_formula(query, context_text)

        if not plan["key_ideas"]:
            plan["key_ideas"] = base["key_ideas"]
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
        key_ideas = plan.get("key_ideas") or []
        steps = plan.get("steps") or []
        worked = plan.get("worked_example") or []
        key_idea_lines = "\n".join([f"- {item}" for item in key_ideas])
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

            ## Key Ideas
            {key_idea_lines}

            ## Steps
            {step_lines}

            ## Worked Example
            {worked_lines}

            ## Answer Line
            {plan.get("answer_line", "")}
            """
        ).strip()

    def _build_narration_segments(self, query: str, plan: dict[str, object]) -> list[tuple[str, str]]:
        title = self._clip(str(plan.get("title", "") or "DeepGyan lesson"), 80)
        learning_goal = self._clip(str(plan.get("learning_goal", "") or query), 140)
        formula_text = self._clip(self._latex_to_text(str(plan.get("formula_latex", "") or "")), 140)
        key_ideas = [self._clip(str(item), 120) for item in (plan.get("key_ideas") or []) if str(item).strip()]
        steps = [self._clip(str(item), 120) for item in (plan.get("steps") or []) if str(item).strip()]
        worked = [self._clip(str(item), 120) for item in (plan.get("worked_example") or []) if str(item).strip()]
        answer_line = self._clip(
            str(plan.get("answer_line", "") or "The answer follows from the main rule and the worked steps."),
            140,
        )

        if len(key_ideas) < 2:
            key_ideas = [self._clip(str(item), 120) for item in self._fallback_plan(query, "").get("key_ideas", [])]
        if len(steps) < 3:
            steps = [self._clip(str(item), 120) for item in self._fallback_plan(query, "").get("steps", [])]
        if len(worked) < 3:
            worked = [self._clip(str(item), 120) for item in self._fallback_plan(query, "").get("worked_example", [])]

        segments = [
            ("intro", f"{title}. {learning_goal}"),
            ("formula", f"Core formula: {formula_text}."),
            ("key_ideas", "Key ideas. " + " ".join(f"Idea {idx + 1}: {item}." for idx, item in enumerate(key_ideas[:3]))),
        ]
        segments.extend((f"step_{idx + 1}", f"Step {idx + 1}. {item}.") for idx, item in enumerate(steps[:3]))
        segments.extend((f"example_{idx + 1}", f"Worked example line {idx + 1}. {item}.") for idx, item in enumerate(worked[:3]))
        segments.append(("answer", answer_line))
        return [(name, re.sub(r"\s+", " ", text).strip()) for name, text in segments]

    @staticmethod
    def _estimate_segment_duration(text: str) -> float:
        words = max(1, len(re.findall(r"\w+", text or "")))
        return max(1.0, min(5.4, round(words / 2.7, 2)))

    def _build_timing_profile(self, narration_segments: list[tuple[str, str]]) -> dict[str, float]:
        raw = {
            name: max(1.0, self._estimate_segment_duration(text))
            for name, text in narration_segments
        }
        total = sum(raw.values())

        # Keep total segment pacing close to a 40-second scene.
        target_total = 40.0
        scale = min(1.0, target_total / total) if total else 1.0
        return {name: round(max(0.6, value * scale), 2) for name, value in raw.items()}

    def _template_script_from_plan(
        self,
        query: str,
        plan: dict[str, object],
        timings: dict[str, float] | None = None,
    ) -> str:
        title = self._clip(str(plan.get("title", "") or "DeepGyan Animation"), 70)
        learning_goal = self._wrap_text(self._clip(str(plan.get("learning_goal", "") or query), 95), 40, 3)
        formula_text = self._wrap_text(
            self._latex_to_text(str(plan.get("formula_latex", "") or "")),
            width=34,
            max_lines=2,
        )
        answer_text = self._wrap_text(
            self._clip(
                str(plan.get("answer_line", "") or "The answer follows from the core idea in the lesson."),
                120,
            ),
            width=42,
            max_lines=2,
        )

        key_ideas = [self._clip(str(item), 90) for item in (plan.get("key_ideas") or []) if str(item).strip()]
        if len(key_ideas) < 2:
            key_ideas = [self._clip(str(item), 90) for item in self._fallback_plan(query, "").get("key_ideas", [])]
        key_ideas = key_ideas[:3]

        steps = [self._clip(str(item), 95) for item in (plan.get("steps") or []) if str(item).strip()]
        if len(steps) < 3:
            steps = [self._clip(str(item), 95) for item in self._fallback_plan(query, "").get("steps", [])]
        steps = steps[:3]

        worked = [self._clip(str(item), 95) for item in (plan.get("worked_example") or []) if str(item).strip()]
        if len(worked) < 3:
            worked = [self._clip(str(item), 95) for item in self._fallback_plan(query, "").get("worked_example", [])]
        worked = worked[:3]

        visual_focus = str(plan.get("visual_focus", "generic")).strip().lower()

        # ── Build the scene script ──
        lines = [
            "from manim import *",
            "",
            f"class {self._scene_name}(MovingCameraScene):",
            "    def construct(self):",
            "        self.camera.background_color = '#0B0E1A'",
            "        self.camera.frame.set(width=14.0)",
            "",
            "        MONO = 'Menlo'",
            "        Text.set_default(font=MONO)",
            "",
            "        ACCENT  = '#00D4FF'",
            "        GOLD    = '#FFD166'",
            "        LIME    = '#06D6A0'",
            "        CORAL   = '#EF476F'",
            "        SOFT_W  = '#E0E6ED'",
            "        DIM     = '#3A4260'",
            "        CARD_BG = '#131829'",
            "",
            "        # ── helper: safe text ──",
            "        def st(content, size=22, color=SOFT_W, mw=6.0, bold=False):",
            "            t = Text(content, font_size=size, color=color, weight=BOLD if bold else NORMAL)",
            "            if t.width > mw:",
            "                t.set_width(mw)",
            "            return t",
            "",
            "        # ── helper: rounded card ──",
            "        def card(w, h, border=ACCENT, opacity=0.92):",
            "            r = RoundedRectangle(width=w, height=h, corner_radius=0.22, color=border, stroke_width=2)",
            "            r.set_fill(CARD_BG, opacity=opacity)",
            "            return r",
            "",
            "        # ── helper: clear slide ──",
            "        def clear_slide(mobjects, t=0.45):",
            "            self.play(self.camera.frame.animate.move_to(ORIGIN).set(width=14.0), *[FadeOut(m) for m in mobjects], run_time=t)",
            "",
            "        # ── persistent brand badge (top-right) ──",
            "        brand_bg = card(2.2, 0.6, border=GOLD)",
            "        brand_tx = st('GyanDeep', 20, GOLD, bold=True)",
            "        brand_tx.move_to(brand_bg)",
            "        brand = VGroup(brand_bg, brand_tx).to_corner(UR, buff=0.35)",
            "",
            "        # ── thin progress bar at bottom ──",
            "        bar_bg = Rectangle(width=13.2, height=0.06, color=DIM, stroke_width=0).set_fill(DIM, 0.4)",
            "        bar_bg.to_edge(DOWN, buff=0.18)",
            "        bar_fill = bar_bg.copy().set_fill(ACCENT, 0.9).set_stroke(width=0)",
            "        bar_fill.stretch_to_fit_width(0.01).align_to(bar_bg, LEFT)",
            "",
            "        self.add(brand, bar_bg, bar_fill)",
            "",
            "        def advance_bar(fraction, t=0.4):",
            "            self.play(bar_fill.animate.stretch_to_fit_width(max(0.01, 13.2 * fraction)).align_to(bar_bg, LEFT), run_time=t)",
            "",
            "        # ═══════════════════════════════════════",
            "        # SLIDE 1 — Title + Learning Goal",
            "        # ═══════════════════════════════════════",
            f"        s1_title = st({repr(title)}, 38, ACCENT, mw=10, bold=True)",
            f"        s1_goal  = st({repr(learning_goal)}, 24, SOFT_W, mw=9.5)",
            "        s1 = VGroup(s1_title, s1_goal).arrange(DOWN, buff=0.4)",
            "        s1.move_to(ORIGIN + UP * 0.3)",
            "",
            "        self.play(FadeIn(s1_title, shift=UP * 0.3), run_time=0.9)",
            "        self.play(FadeIn(s1_goal, shift=UP * 0.2), run_time=0.7)",
            "        self.wait(1.2)",
            "        advance_bar(0.12)",
            "        clear_slide([s1])",
            "",
        ]

        # ═══════════════════════════════════════
        # SLIDE 2 — Core Formula + Visual
        # ═══════════════════════════════════════
        lines.extend([
            "        # ═══ SLIDE 2 — Core Formula + Visual ═══",
            f"        s2_label = st('Core Formula', 26, GOLD, bold=True)",
            "        s2_line  = Line(LEFT * 3, RIGHT * 3, color=GOLD, stroke_width=2).set_opacity(0.5)",
            f"        s2_formula = st({repr(formula_text)}, 28, LIME, mw=5.8)",
            "        s2_left = VGroup(s2_label, s2_line, s2_formula).arrange(DOWN, buff=0.25)",
            "        s2_left.move_to(LEFT * 3 + UP * 0.2)",
            "",
        ])

        # Visual panel (right side)
        if visual_focus == "triangle":
            lines.extend([
                "        s2_panel = card(4.5, 3.2, border=ACCENT)",
                "        s2_panel.move_to(RIGHT * 3 + UP * 0.2)",
                "        s2_ptitle = st('Geometry', 18, ACCENT).move_to(s2_panel.get_top() + DOWN * 0.25)",
                "        ctr = s2_panel.get_center() + DOWN * 0.15",
                "        p1 = ctr + LEFT * 1.1 + DOWN * 0.6",
                "        p2 = ctr + RIGHT * 1.1 + DOWN * 0.6",
                "        p3 = ctr + UP * 0.85",
                "        s2_tri = Polygon(p1, p2, p3, color=GOLD, stroke_width=3).set_fill('#1a2040', 0.3)",
                "        foot = p3[0] * RIGHT + p1[1] * UP",
                "        s2_h = DashedLine(p3, foot, color=ACCENT)",
                "        s2_bl = st('b', 16, LIME).next_to(Line(p1, p2), DOWN, buff=0.08)",
                "        s2_hl = st('h', 16, ACCENT).next_to(s2_h, RIGHT, buff=0.08)",
                "        s2_vis = VGroup(s2_panel, s2_ptitle, s2_tri, s2_h, s2_bl, s2_hl)",
            ])
        elif visual_focus == "circle":
            lines.extend([
                "        s2_panel = card(4.5, 3.2, border=ACCENT)",
                "        s2_panel.move_to(RIGHT * 3 + UP * 0.2)",
                "        s2_ptitle = st('Circle', 18, ACCENT).move_to(s2_panel.get_top() + DOWN * 0.25)",
                "        ctr = s2_panel.get_center() + DOWN * 0.1",
                "        s2_circ = Circle(radius=0.9, color=GOLD, stroke_width=3).move_to(ctr).set_fill('#1a2040', 0.2)",
                "        s2_rad = Line(ctr, ctr + RIGHT * 0.9, color=LIME)",
                "        s2_rl = st('r', 16, LIME).next_to(s2_rad, UP, buff=0.06)",
                "        s2_vis = VGroup(s2_panel, s2_ptitle, s2_circ, s2_rad, s2_rl)",
            ])
        elif visual_focus == "numberline":
            lines.extend([
                "        s2_panel = card(4.5, 3.2, border=ACCENT)",
                "        s2_panel.move_to(RIGHT * 3 + UP * 0.2)",
                "        s2_ptitle = st('Number Line', 18, ACCENT).move_to(s2_panel.get_top() + DOWN * 0.25)",
                "        s2_nl = NumberLine(x_range=[-3, 3, 1], length=3.4, include_tip=True, color=ACCENT)",
                "        s2_nl.move_to(s2_panel.get_center() + DOWN * 0.15)",
                "        s2_d1 = Dot(s2_nl.n2p(-2), color=CORAL)",
                "        s2_d2 = Dot(s2_nl.n2p(0.5), color=LIME)",
                "        s2_d3 = Dot(s2_nl.n2p(1.4), color=GOLD)",
                "        s2_vis = VGroup(s2_panel, s2_ptitle, s2_nl, s2_d1, s2_d2, s2_d3)",
            ])
        else:
            lines.extend([
                "        s2_panel = card(4.5, 3.2, border=ACCENT)",
                "        s2_panel.move_to(RIGHT * 3 + UP * 0.2)",
                "        s2_ptitle = st('Concept Map', 18, ACCENT).move_to(s2_panel.get_top() + DOWN * 0.25)",
                "        ctr = s2_panel.get_center() + DOWN * 0.1",
                "        s2_core = Dot(ctr, color=ACCENT, radius=0.12)",
                "        s2_da = Dot(ctr + UL * 0.8, color=GOLD, radius=0.09)",
                "        s2_db = Dot(ctr + UR * 0.8, color=LIME, radius=0.09)",
                "        s2_dc = Dot(ctr + DOWN * 0.85, color=CORAL, radius=0.09)",
                "        s2_la = Line(ctr, s2_da.get_center(), color=DIM)",
                "        s2_lb = Line(ctr, s2_db.get_center(), color=DIM)",
                "        s2_lc = Line(ctr, s2_dc.get_center(), color=DIM)",
                "        s2_vis = VGroup(s2_panel, s2_ptitle, s2_la, s2_lb, s2_lc, s2_core, s2_da, s2_db, s2_dc)",
            ])

        lines.extend([
            "",
            "        self.play(FadeIn(s2_left, shift=RIGHT * 0.2), run_time=0.8)",
            "        self.play(FadeIn(s2_vis, shift=LEFT * 0.2), run_time=0.8)",
            "        self.wait(1.5)",
            "        advance_bar(0.28)",
            "        clear_slide([s2_left, s2_vis])",
            "",
        ])

        # ═══════════════════════════════════════
        # SLIDE 3 — Key Ideas
        # ═══════════════════════════════════════
        lines.append("        # ═══ SLIDE 3 — Key Ideas ═══")
        lines.append("        s3_header = st('Key Ideas', 28, GOLD, bold=True)")
        lines.append("        s3_divider = Line(LEFT * 3.5, RIGHT * 3.5, color=GOLD, stroke_width=2).set_opacity(0.4)")

        for idx, idea in enumerate(key_ideas, start=1):
            wrapped = self._wrap_text(idea, 50, 2)
            color = ["CORAL", "ACCENT", "LIME"][(idx - 1) % 3]
            lines.append(f"        s3_dot_{idx} = Dot(radius=0.08, color={color})")
            lines.append(f"        s3_idea_{idx} = st({repr(wrapped)}, 21, SOFT_W, mw=8)")
            lines.append(f"        s3_row_{idx} = VGroup(s3_dot_{idx}, s3_idea_{idx}).arrange(RIGHT, buff=0.2)")

        idea_rows = ", ".join(f"s3_row_{i}" for i in range(1, len(key_ideas) + 1))
        lines.extend([
            f"        s3_ideas = VGroup({idea_rows}).arrange(DOWN, aligned_edge=LEFT, buff=0.3)",
            "        s3_all = VGroup(s3_header, s3_divider, s3_ideas).arrange(DOWN, buff=0.3)",
            "        s3_all.move_to(ORIGIN + UP * 0.2)",
            "",
            "        self.play(FadeIn(s3_header, shift=UP * 0.2), Create(s3_divider), run_time=0.7)",
        ])
        for idx in range(1, len(key_ideas) + 1):
            lines.append(f"        self.play(FadeIn(s3_row_{idx}, shift=RIGHT * 0.15), run_time=0.5)")
        lines.extend([
            "        self.wait(1.5)",
            "        advance_bar(0.42)",
            "        clear_slide([s3_all])",
            "",
        ])

        # ═══════════════════════════════════════
        # SLIDE 4 — Solution Steps
        # ═══════════════════════════════════════
        lines.append("        # ═══ SLIDE 4 — Solution Steps ═══")
        lines.append("        s4_header = st('Solution Steps', 28, ACCENT, bold=True)")
        lines.append("        s4_divider = Line(LEFT * 3.5, RIGHT * 3.5, color=ACCENT, stroke_width=2).set_opacity(0.4)")

        step_colors = ["GOLD", "LIME", "ACCENT"]
        for idx, step in enumerate(steps, start=1):
            wrapped = self._wrap_text(step, 44, 2)
            sc = step_colors[(idx - 1) % 3]
            lines.extend([
                f"        s4_badge_{idx} = Circle(radius=0.22, color={sc}, stroke_width=2.5).set_fill(CARD_BG, 1)",
                f"        s4_bnum_{idx} = st('{idx}', 18, {sc})",
                f"        s4_bnum_{idx}.move_to(s4_badge_{idx})",
                f"        s4_bg_{idx} = VGroup(s4_badge_{idx}, s4_bnum_{idx})",
                f"        s4_txt_{idx} = st({repr(wrapped)}, 20, SOFT_W, mw=8)",
                f"        s4_row_{idx} = VGroup(s4_bg_{idx}, s4_txt_{idx}).arrange(RIGHT, buff=0.25)",
            ])

        step_rows = ", ".join(f"s4_row_{i}" for i in range(1, len(steps) + 1))
        lines.extend([
            f"        s4_steps = VGroup({step_rows}).arrange(DOWN, aligned_edge=LEFT, buff=0.35)",
            "        s4_all = VGroup(s4_header, s4_divider, s4_steps).arrange(DOWN, buff=0.3)",
            "        s4_all.move_to(ORIGIN + UP * 0.2)",
            "",
            "        self.play(FadeIn(s4_header, shift=UP * 0.2), Create(s4_divider), run_time=0.7)",
        ])
        for idx in range(1, len(steps) + 1):
            lines.append(f"        self.play(FadeIn(s4_row_{idx}, shift=RIGHT * 0.15), run_time=0.55)")
            lines.append(f"        self.wait(0.6)")
        lines.extend([
            "        self.wait(1.0)",
            "        advance_bar(0.62)",
            "        clear_slide([s4_all])",
            "",
        ])

        # ═══════════════════════════════════════
        # SLIDE 5 — Worked Example
        # ═══════════════════════════════════════
        lines.append("        # ═══ SLIDE 5 — Worked Example ═══")
        lines.append("        s5_header = st('Worked Example', 28, LIME, bold=True)")
        lines.append("        s5_divider = Line(LEFT * 3.5, RIGHT * 3.5, color=LIME, stroke_width=2).set_opacity(0.4)")

        for idx, ex in enumerate(worked, start=1):
            wrapped = self._wrap_text(ex, 48, 2)
            lines.extend([
                f"        s5_idx_{idx} = st('{idx}.', 19, GOLD)",
                f"        s5_ex_{idx} = st({repr(wrapped)}, 20, SOFT_W, mw=8.5)",
                f"        s5_row_{idx} = VGroup(s5_idx_{idx}, s5_ex_{idx}).arrange(RIGHT, buff=0.2)",
            ])

        ex_rows = ", ".join(f"s5_row_{i}" for i in range(1, len(worked) + 1))
        lines.extend([
            f"        s5_examples = VGroup({ex_rows}).arrange(DOWN, aligned_edge=LEFT, buff=0.3)",
            "",
            f"        s5_ans_label = st('Answer:', 20, LIME, bold=True)",
            f"        s5_ans_text = st({repr(answer_text)}, 20, SOFT_W, mw=8)",
            "        s5_ans = VGroup(s5_ans_label, s5_ans_text).arrange(RIGHT, buff=0.2)",
            "        s5_ans_box = card(s5_ans.width + 0.6, s5_ans.height + 0.3, border=LIME)",
            "        s5_ans.move_to(s5_ans_box)",
            "        s5_ans_g = VGroup(s5_ans_box, s5_ans)",
            "",
            "        s5_all = VGroup(s5_header, s5_divider, s5_examples, s5_ans_g).arrange(DOWN, buff=0.3)",
            "        s5_all.move_to(ORIGIN + UP * 0.15)",
            "",
            "        self.play(FadeIn(s5_header, shift=UP * 0.2), Create(s5_divider), run_time=0.7)",
        ])
        for idx in range(1, len(worked) + 1):
            lines.append(f"        self.play(FadeIn(s5_row_{idx}, shift=RIGHT * 0.15), run_time=0.5)")
            lines.append(f"        self.wait(0.5)")
        lines.extend([
            "        self.play(FadeIn(s5_ans_g, shift=UP * 0.15), run_time=0.7)",
            "        self.wait(1.5)",
            "        advance_bar(0.85)",
            "        clear_slide([s5_all])",
            "",
        ])

        # ═══════════════════════════════════════
        # SLIDE 6 — Summary (all on screen)
        # ═══════════════════════════════════════
        lines.extend([
            "        # ═══ SLIDE 6 — Summary ═══",
            f"        sum_title = st('Summary', 32, ACCENT, bold=True)",
            "        sum_title.to_edge(UP, buff=0.5)",
            "",
        ])

        # Key ideas column
        idea_sum_items = []
        for idx, idea in enumerate(key_ideas, start=1):
            short = self._wrap_text(idea, 22, 2)
            lines.append(f"        sum_ki_{idx} = st({repr(short)}, 14, SOFT_W, mw=3.6)")
            idea_sum_items.append(f"sum_ki_{idx}")
        ki_group = ", ".join(idea_sum_items)
        lines.extend([
            f"        sum_ki_title = st('Key Ideas', 16, GOLD, bold=True)",
            f"        sum_ki_col = VGroup(sum_ki_title, {ki_group}).arrange(DOWN, aligned_edge=LEFT, buff=0.12)",
            "        sum_ki_bg = card(sum_ki_col.width + 0.5, sum_ki_col.height + 0.4, border=GOLD)",
            "        sum_ki_col.move_to(sum_ki_bg)",
            "        sum_ki = VGroup(sum_ki_bg, sum_ki_col)",
        ])

        # Steps column
        step_sum_items = []
        for idx, step in enumerate(steps, start=1):
            short = self._wrap_text(step, 22, 2)
            lines.append(f"        sum_st_{idx} = st({repr(f'{idx}. ' + short)}, 14, SOFT_W, mw=3.6)")
            step_sum_items.append(f"sum_st_{idx}")
        st_group = ", ".join(step_sum_items)
        lines.extend([
            f"        sum_st_title = st('Steps', 16, ACCENT, bold=True)",
            f"        sum_st_col = VGroup(sum_st_title, {st_group}).arrange(DOWN, aligned_edge=LEFT, buff=0.12)",
            "        sum_st_bg = card(sum_st_col.width + 0.5, sum_st_col.height + 0.4, border=ACCENT)",
            "        sum_st_col.move_to(sum_st_bg)",
            "        sum_st = VGroup(sum_st_bg, sum_st_col)",
        ])

        # Answer column
        lines.extend([
            f"        sum_ans_text = st({repr(self._wrap_text(answer_text, 22, 2))}, 14, SOFT_W, mw=3.6)",
            "        sum_ans_title = st('Answer', 16, LIME, bold=True)",
            "        sum_ans_col = VGroup(sum_ans_title, sum_ans_text).arrange(DOWN, aligned_edge=LEFT, buff=0.12)",
            "        sum_ans_bg = card(sum_ans_col.width + 0.5, sum_ans_col.height + 0.4, border=LIME)",
            "        sum_ans_col.move_to(sum_ans_bg)",
            "        sum_ans = VGroup(sum_ans_bg, sum_ans_col)",
            "",
            "        sum_row = VGroup(sum_ki, sum_st, sum_ans).arrange(RIGHT, buff=0.35)",
            "        sum_row.next_to(sum_title, DOWN, buff=0.45)",
            "",
            "        self.play(FadeIn(sum_title, shift=UP * 0.2), run_time=0.6)",
            "        self.play(LaggedStart(FadeIn(sum_ki, shift=UP*0.1), FadeIn(sum_st, shift=UP*0.1), FadeIn(sum_ans, shift=UP*0.1), lag_ratio=0.15), run_time=1.0)",
            "        self.wait(2.5)",
            "        advance_bar(1.0)",
            "        self.wait(0.5)",
        ])

        return "\n".join(lines).strip()

    @staticmethod
    def _script_looks_valid(script: str, scene_name: str) -> bool:
        if "from manim import" not in script:
            return False
        if "visual model" in script.lower():
            return False
        if "GyanDeep" not in script:
            return False
        if script.count("self.wait(") < 5:
            return False
        try:
            tree = ast.parse(script)
        except SyntaxError:
            return False
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == scene_name:
                base_names: list[str] = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        base_names.append(base.attr)
                return any(name in {"Scene", "MovingCameraScene"} for name in base_names)
        return False

    def _generate_script(
        self,
        query: str,
        context_text: str,
        plan: dict[str, object],
        timings: dict[str, float] | None = None,
    ) -> tuple[str, str]:
        _ = context_text
        return self._template_script_from_plan(query, plan, timings=timings), "template_from_plan"

    def _render(self, script_path: Path, media_dir: Path) -> Path:
        media_dir.mkdir(parents=True, exist_ok=True)
        command = [
            *self._resolve_manim_command(),
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
    def _resolve_manim_command() -> list[str]:
        direct = shutil.which("manim") or shutil.which("manim.exe")
        if direct:
            return [direct]

        scripts_dir = Path(sys.executable).resolve().parent
        for candidate_name in ("manim", "manim.exe"):
            sibling = scripts_dir / candidate_name
            if sibling.exists():
                return [str(sibling)]

        if importlib.util.find_spec("manim") is not None:
            return [sys.executable, "-m", "manim"]

        raise RuntimeError(
            f"Manim CLI not found for Python at {sys.executable}. Install with `python -m pip install manim` in this environment."
        )

    @staticmethod
    def _resolve_manim_cli() -> str:
        return " ".join(ManimVideoPlugin._resolve_manim_command())

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

        narration_segments = self._build_narration_segments(request.query, plan)
        timings = self._build_timing_profile(narration_segments)

        await emit("scripting", "Generating Manim scene from blueprint...")
        script, generation_mode = await asyncio.to_thread(
            self._generate_script,
            request.query,
            request.context_text,
            plan,
            timings,
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
