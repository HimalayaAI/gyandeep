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
            - Use a clean whiteboard-style layout with a light background, dark text, and minimal decoration.
            - Keep all essential content inside safe margins; do not place important text at the bottom edge.
            - Use `self.camera.frame.animate...` to follow the active writing area instead of stacking one tall page of text.
            - Replace any placeholder "Visual Model" box with a small top-right badge that says `GyanDeep`.
            - Use at least two non-text visuals such as a diagram, number line, dots, arrows, shaded regions, or geometric shapes.
            - Break long explanations into short beats; avoid giant text walls and avoid overlap between text and visuals.
            - Structure the lesson as a classroom explanation: hook, key parts, core formula, guided Example 1, insight moment, final answer, Example 2 at a slightly faster pace, optional common mistake, and recap.
            - Reveal equations gradually and prefer `TransformMatchingTex` or `ReplacementTransform` so the learner can track changes from one line to the next.
            - Color code consistently: variables in blue, constants in green, and final answers emphasized in yellow.
            - Use subtle emphasis only, such as `Indicate`, a small scale change, or a thin surrounding shape. Avoid flashy motion and avoid decorative UI cards or progress bars.
            - The animation must teach the actual solution flow with `Step 1`, `Step 2`, `Step 3`, a separate `Key Ideas` area, and a worked example shown as line-by-line math text instead of stacked boxes.
            - Keep script robust for low-quality render (`-ql`) and avoid fragile APIs.
            - Use readable text sizes (title >= 40, body >= 26).
            - Include at least 5 explicit `self.wait(...)` pauses.
            - Keep total duration under 60 seconds.
            - Prefer MovingCameraScene, Text/MathTex, NumberLine, Dot, Line, Polygon, Circle, VGroup, FadeIn, Write, Transform, TransformMatchingTex, ReplacementTransform, Indicate, Create, and `.animate`.
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

        if any(word in scope for word in ("trigonometric", "trigonometry", "sine", "cosine", "tangent", "sin(", "cos(", "tan(")):
            return {
                "title": "Using trigonometric ratios",
                "learning_goal": "Use a right-triangle ratio to connect an angle with side lengths.",
                "formula_latex": r"\sin(\theta)=\frac{\text{opposite}}{\text{hypotenuse}}",
                "key_ideas": [
                    "Choose the ratio that contains the known and unknown sides.",
                    "Substitute values before solving for the missing side.",
                    "Keep the angle and side labels connected to the diagram.",
                ],
                "steps": [
                    "Label the angle, opposite side, adjacent side, and hypotenuse.",
                    "Select sine, cosine, or tangent based on the sides involved.",
                    "Substitute the known values and solve the equation carefully.",
                ],
                "worked_example": [
                    r"\sin(30^\circ)=\frac{x}{10}",
                    r"\frac{1}{2}=\frac{x}{10}",
                    r"x=5",
                ],
                "visual_focus": "triangle",
                "answer_line": "The missing opposite side is found by matching the ratio to the labeled triangle.",
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

    def _classroom_blueprint(self, query: str, plan: dict[str, object]) -> dict[str, object]:
        """Choose a compact worked-example script that can be rendered reliably.

        The LLM plan supplies the lesson topic, but the animation template needs
        predictable equations for smooth TransformMatchingTex transitions.  This
        method maps common school-math topics to small, classroom-ready examples
        and falls back to a simple algebra pattern when the topic is generic.
        """

        title = self._clip(str(plan.get("title", "") or "Step-by-step math lesson"), 64)
        formula_raw = str(plan.get("formula_latex", "") or self._fallback_formula(query, "")).strip()
        scope = f"{query} {title} {formula_raw}".lower()
        visual_focus = str(plan.get("visual_focus", "generic")).strip().lower()

        common = {
            "title": title,
            "hook": "What changes when we replace words with one clear equation?",
            "concept_intro": "A formula is a small machine: label the parts, substitute carefully, then simplify.",
            "formula_latex": formula_raw or r"x+3=7",
            "components": [
                ("x", "variable: the unknown value", "variable"),
                ("3, 7", "constants: fixed numbers", "constant"),
                ("=", "balance: both sides stay equal", "neutral"),
            ],
            "example1_caption": "Example 1: solve one equation slowly",
            "example1_eqs": [r"x+3=7", r"x=7-3", r"x=4"],
            "example1_final": r"x=4",
            "interpretation": "The unknown value is 4 because it keeps the equation balanced.",
            "insight": "Do the same operation to both sides so equality is preserved.",
            "example2_caption": "Example 2: same idea, a little faster",
            "example2_eqs": [r"y+5=12", r"y=12-5", r"y=7"],
            "mistake": r"x+3=7\;\not\Rightarrow\;x=7+3",
            "visual_focus": visual_focus,
        }

        if any(word in scope for word in ("trigonometric", "trigonometry", "sine", "cosine", "tangent", "sin(", "cos(", "tan(")):
            common.update(
                hook="How can one angle tell us a missing side length?",
                concept_intro="In a right triangle, each trig ratio compares two specific sides.",
                formula_latex=r"\sin(\theta)=\frac{\text{opposite}}{\text{hypotenuse}}",
                components=[
                    (r"\theta", "variable: the marked angle", "variable"),
                    ("opposite", "variable side: across from the angle", "variable"),
                    ("hypotenuse", "constant side when given", "constant"),
                ],
                example1_caption="Example 1: find the opposite side",
                example1_eqs=[
                    r"\sin(30^\circ)=\frac{x}{10}",
                    r"\frac{1}{2}=\frac{x}{10}",
                    r"x=5",
                ],
                example1_final=r"x=5",
                interpretation="The side opposite the 30° angle is 5 units.",
                insight="Choose the ratio by asking: which two sides are involved?",
                example2_caption="Example 2: same ratio, faster",
                example2_eqs=[
                    r"\sin(30^\circ)=\frac{x}{14}",
                    r"\frac{1}{2}=\frac{x}{14}",
                    r"x=7",
                ],
                mistake=r"\sin(\theta)\ne\frac{\text{adjacent}}{\text{hypotenuse}}",
                visual_focus="triangle",
            )
        elif "real number" in scope or "real numbers" in scope:
            common.update(
                hook="Where do fractions, decimals, and roots all live?",
                concept_intro="The real-number line contains both rational and irrational values.",
                formula_latex=r"\mathbb{R}=\mathbb{Q}\cup\{\text{irrational numbers}\}",
                components=[
                    (r"\mathbb{Q}", "rational numbers: fractions and repeating decimals", "variable"),
                    (r"\sqrt{2},\pi", "irrational numbers: non-repeating decimals", "constant"),
                    (r"\mathbb{R}", "all points on the number line", "answer"),
                ],
                example1_caption="Example 1: classify a few values",
                example1_eqs=[
                    r"-2,\;\frac{1}{2}\in\mathbb{Q}",
                    r"\sqrt{2},\;\pi\notin\mathbb{Q}",
                    r"-2,\;\frac{1}{2},\;\sqrt{2},\;\pi\in\mathbb{R}",
                ],
                example1_final=r"\text{all are real numbers}",
                interpretation="Every listed value has a place on the number line.",
                insight="Real does not mean rational; irrational values are real too.",
                example2_caption="Example 2: quicker classification",
                example2_eqs=[
                    r"0.75=\frac{3}{4}\in\mathbb{Q}",
                    r"\sqrt{5}\notin\mathbb{Q}",
                    r"0.75,\;\sqrt{5}\in\mathbb{R}",
                ],
                mistake=r"\sqrt{2}\notin\mathbb{Q}\;\text{but}\;\sqrt{2}\in\mathbb{R}",
                visual_focus="numberline",
            )
        elif "sphere" in scope or "volume" in scope and "sphere" in scope or "r^3" in formula_raw.lower():
            common.update(
                hook="How does one radius determine a whole sphere?",
                concept_intro="Sphere volume grows with the cube of the radius, so the radius is the key variable.",
                formula_latex=r"V=\frac{4}{3}\pi r^3",
                components=[
                    ("V", "variable: volume to find", "variable"),
                    ("r", "variable: radius", "variable"),
                    (r"\frac{4}{3},\pi", "constants: fixed multipliers", "constant"),
                ],
                example1_caption="Example 1: radius 3",
                example1_eqs=[r"V=\frac{4}{3}\pi r^3", r"V=\frac{4}{3}\pi(3)^3", r"V=36\pi"],
                example1_final=r"V=36\pi",
                interpretation="A sphere with radius 3 has volume 36π cubic units.",
                insight="Cubing the radius is the turning point: 3 becomes 27.",
                example2_caption="Example 2: radius 6, faster",
                example2_eqs=[r"V=\frac{4}{3}\pi(6)^3", r"V=288\pi"],
                mistake=r"V\ne\frac{4}{3}\pi r^2",
                visual_focus="circle",
            )
        elif "pythag" in scope or "right triangle" in scope and "sin" not in scope:
            common.update(
                hook="If two sides are known, can the third be forced?",
                concept_intro="In a right triangle, the squares of the legs add to the square of the hypotenuse.",
                formula_latex=r"c^2=a^2+b^2",
                components=[
                    ("a, b", "variables: the two legs", "variable"),
                    ("c", "variable: hypotenuse", "variable"),
                    ("2", "constant exponent: square each side", "constant"),
                ],
                example1_caption="Example 1: legs 3 and 4",
                example1_eqs=[r"c^2=a^2+b^2", r"c^2=3^2+4^2", r"c^2=25", r"c=5"],
                example1_final=r"c=5",
                interpretation="The hypotenuse is 5 units long.",
                insight="Add the squares first; take the square root only at the end.",
                example2_caption="Example 2: legs 5 and 12",
                example2_eqs=[r"c^2=5^2+12^2", r"c^2=169", r"c=13"],
                mistake=r"c\ne a+b",
                visual_focus="triangle",
            )
        elif "interest" in scope:
            common.update(
                hook="How do money, rate, and time combine?",
                concept_intro="Simple interest multiplies the principal by the rate and the time.",
                formula_latex=r"I=Prt",
                components=[
                    ("I", "variable: interest earned", "variable"),
                    ("P, r, t", "variables: principal, rate, time", "variable"),
                    ("100", "constant when rate is written as percent", "constant"),
                ],
                example1_caption="Example 1: P=2000, r=5%, t=3",
                example1_eqs=[r"I=Prt", r"I=2000(0.05)(3)", r"I=300"],
                example1_final=r"I=300",
                interpretation="The simple interest is 300 rupees.",
                insight="Convert percent to decimal before multiplying.",
                example2_caption="Example 2: P=1500, r=4%, t=2",
                example2_eqs=[r"I=1500(0.04)(2)", r"I=120"],
                mistake=r"5\%\ne5",
                visual_focus="generic",
            )
        elif "quadratic" in scope or "roots" in scope:
            common.update(
                hook="How can a quadratic reveal where it crosses zero?",
                concept_intro="Roots are the x-values that make the quadratic equal to zero.",
                formula_latex=r"ax^2+bx+c=0",
                components=[
                    ("x", "variable: value we solve for", "variable"),
                    ("a,b,c", "constants: fixed coefficients", "constant"),
                    ("0", "target value for roots", "constant"),
                ],
                example1_caption="Example 1: factor to find roots",
                example1_eqs=[r"x^2-5x+6=0", r"(x-2)(x-3)=0", r"x=2\;\text{or}\;x=3"],
                example1_final=r"x=2\;\text{or}\;x=3",
                interpretation="The graph crosses the x-axis at 2 and 3.",
                insight="A product is zero only when at least one factor is zero.",
                example2_caption="Example 2: same pattern, faster",
                example2_eqs=[r"x^2-7x+10=0", r"(x-5)(x-2)=0", r"x=5\;\text{or}\;x=2"],
                mistake=r"(x-2)(x-3)=0\not\Rightarrow x=-2,-3",
                visual_focus="algebra",
            )
        elif "area" in scope and "triangle" in scope or visual_focus == "triangle":
            common.update(
                hook="Why is a triangle half of a rectangle?",
                concept_intro="Triangle area uses base and height; the factor 1/2 accounts for the matching half.",
                formula_latex=r"A=\frac{1}{2}bh",
                components=[
                    ("A", "variable: area to find", "variable"),
                    ("b, h", "variables: base and height", "variable"),
                    (r"\frac{1}{2}", "constant: take half", "constant"),
                ],
                example1_caption="Example 1: base 8, height 5",
                example1_eqs=[r"A=\frac{1}{2}bh", r"A=\frac{1}{2}(8)(5)", r"A=20"],
                example1_final=r"A=20",
                interpretation="The triangle covers 20 square units.",
                insight="The height must be perpendicular to the base.",
                example2_caption="Example 2: base 10, height 6",
                example2_eqs=[r"A=\frac{1}{2}(10)(6)", r"A=30"],
                mistake=r"A\ne bh",
                visual_focus="triangle",
            )

        return common

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

    @staticmethod
    def _dedupe_strings(values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            cleaned = str(value or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            ordered.append(cleaned)
        return ordered

    @staticmethod
    def _component_label_tokens(label: str) -> list[str]:
        return [part.strip() for part in re.split(r"[,;]", str(label or "")) if part.strip()]

    @staticmethod
    def _numeric_tex_tokens(*expressions: str) -> list[str]:
        tokens: list[str] = []
        for expr in expressions:
            tokens.extend(match.group(0) for match in re.finditer(r"\d+", str(expr or "")))
        return tokens

    def _legacy_template_script_from_plan(
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

    def _template_script_from_plan(
        self,
        query: str,
        plan: dict[str, object],
        timings: dict[str, float] | None = None,
    ) -> str:
        _ = timings
        blueprint = self._classroom_blueprint(query, plan)
        fallback = self._fallback_plan(query, "")

        title = self._clip(str(blueprint.get("title", "") or plan.get("title", "") or "DeepGyan lesson"), 72)
        learning_goal = self._wrap_text(
            self._clip(str(plan.get("learning_goal", "") or query), 110),
            width=48,
            max_lines=2,
        )
        hook = self._wrap_text(
            self._clip(str(blueprint.get("hook", "") or "Let's solve one clear problem."), 110),
            width=48,
            max_lines=2,
        )
        concept_intro = self._wrap_text(
            self._clip(str(blueprint.get("concept_intro", "") or "We will connect the formula to the example."), 140),
            width=52,
            max_lines=3,
        )
        formula_latex = str(blueprint.get("formula_latex", "") or self._fallback_formula(query, "")).strip() or r"x+3=7"
        interpretation = self._wrap_text(
            self._clip(str(blueprint.get("interpretation", "") or plan.get("answer_line", "")), 150),
            width=48,
            max_lines=2,
        )
        insight = self._wrap_text(
            self._clip(str(blueprint.get("insight", "") or "Watch how each change preserves the meaning."), 140),
            width=48,
            max_lines=2,
        )
        mistake = str(blueprint.get("mistake", "") or r"x+3=7\;\not\Rightarrow\;x=7+3")
        example1_caption = self._clip(str(blueprint.get("example1_caption", "") or "Example 1"), 70)
        example2_caption = self._clip(str(blueprint.get("example2_caption", "") or "Example 2"), 70)

        example1_eqs = [str(item).strip() for item in (blueprint.get("example1_eqs") or []) if str(item).strip()]
        example2_eqs = [str(item).strip() for item in (blueprint.get("example2_eqs") or []) if str(item).strip()]
        if not example1_eqs:
            example1_eqs = [formula_latex, r"x=4"]
        if not example2_eqs:
            example2_eqs = [formula_latex, r"y=7"]

        key_ideas = [self._clip(str(item), 95) for item in (plan.get("key_ideas") or []) if str(item).strip()]
        if len(key_ideas) < 2:
            key_ideas = [self._clip(str(item), 95) for item in fallback.get("key_ideas", [])]
        key_ideas = key_ideas[:3]

        steps = [self._clip(str(item), 95) for item in (plan.get("steps") or []) if str(item).strip()]
        if len(steps) < 3:
            steps = [self._clip(str(item), 95) for item in fallback.get("steps", [])]
        steps = steps[:3]

        visual_focus = str(blueprint.get("visual_focus", plan.get("visual_focus", "generic"))).strip().lower()

        component_rows: list[tuple[str, str, str]] = []
        variable_tokens: list[str] = []
        constant_tokens: list[str] = []
        for raw in blueprint.get("components", []) or []:
            if not isinstance(raw, (list, tuple)) or len(raw) < 3:
                continue
            label = str(raw[0]).strip()
            description = self._clip(str(raw[1]).strip(), 90)
            kind = str(raw[2]).strip().lower() or "neutral"
            component_rows.append((self._latex_to_text(label), description, kind))
            tokens = self._component_label_tokens(label)
            if kind == "variable":
                variable_tokens.extend(tokens)
            elif kind in {"constant", "neutral"}:
                constant_tokens.extend(tokens)

        constant_tokens.extend(self._numeric_tex_tokens(formula_latex, *example1_eqs, *example2_eqs, mistake))
        variable_tokens = self._dedupe_strings(variable_tokens)
        constant_tokens = self._dedupe_strings(constant_tokens)

        if not component_rows:
            component_rows = [
                ("unknown", "Track the changing quantity.", "variable"),
                ("given values", "These stay fixed while you solve.", "constant"),
                ("result", "Interpret the final line in context.", "answer"),
            ]

        visual_lines = [
            "        visual_box = Rectangle(width=3.6, height=2.4, color=GREY_C, stroke_width=1.4)",
            "        visual_box.set_fill(WHITE, opacity=0.0)",
        ]
        if visual_focus == "triangle":
            visual_lines.extend(
                [
                    "        p1 = visual_box.get_center() + LEFT * 1.05 + DOWN * 0.75",
                    "        p2 = visual_box.get_center() + RIGHT * 1.0 + DOWN * 0.75",
                    "        p3 = visual_box.get_center() + LEFT * 1.05 + UP * 0.75",
                    "        visual_shape = Polygon(p1, p2, p3, color=BLACK, stroke_width=2)",
                    "        right_mark = Square(side_length=0.18, color=GREY_C, stroke_width=1.5).move_to(p1 + RIGHT * 0.09 + UP * 0.09)",
                    "        base_label = teacher_text('base', size=18, color=VAR_COLOR, max_width=1.2).next_to(Line(p1, p2), DOWN, buff=0.08)",
                    "        height_label = teacher_text('height', size=18, color=VAR_COLOR, max_width=1.4).next_to(Line(p1, p3), LEFT, buff=0.08)",
                    "        visual = VGroup(visual_box, visual_shape, right_mark, base_label, height_label)",
                ]
            )
        elif visual_focus == "circle":
            visual_lines.extend(
                [
                    "        center = visual_box.get_center()",
                    "        visual_shape = Circle(radius=0.82, color=BLACK, stroke_width=2).move_to(center)",
                    "        radius_line = Line(center, center + RIGHT * 0.82, color=VAR_COLOR, stroke_width=2.5)",
                    "        radius_label = teacher_text('r', size=22, color=VAR_COLOR, max_width=0.5).next_to(radius_line, UP, buff=0.06)",
                    "        visual = VGroup(visual_box, visual_shape, radius_line, radius_label)",
                ]
            )
        elif visual_focus == "numberline":
            visual_lines.extend(
                [
                    "        number_line = NumberLine(x_range=[-3, 3, 1], length=3.0, include_tip=True, color=BLACK)",
                    "        number_line.move_to(visual_box.get_center())",
                    "        dot_left = Dot(number_line.n2p(-2), color=CONST_COLOR, radius=0.07)",
                    "        dot_mid = Dot(number_line.n2p(0), color=VAR_COLOR, radius=0.07)",
                    "        dot_right = Dot(number_line.n2p(2), color=ANSWER_COLOR, radius=0.07)",
                    "        visual = VGroup(visual_box, number_line, dot_left, dot_mid, dot_right)",
                ]
            )
        else:
            visual_lines.extend(
                [
                    "        center = visual_box.get_center()",
                    "        core_dot = Dot(center, color=BLACK, radius=0.08)",
                    "        branch_a = Dot(center + LEFT * 1.0 + UP * 0.55, color=VAR_COLOR, radius=0.06)",
                    "        branch_b = Dot(center + RIGHT * 0.95 + UP * 0.55, color=CONST_COLOR, radius=0.06)",
                    "        branch_c = Dot(center + DOWN * 0.72, color=ANSWER_COLOR, radius=0.06)",
                    "        link_a = Line(center, branch_a.get_center(), color=GREY_C, stroke_width=1.8)",
                    "        link_b = Line(center, branch_b.get_center(), color=GREY_C, stroke_width=1.8)",
                    "        link_c = Line(center, branch_c.get_center(), color=GREY_C, stroke_width=1.8)",
                    "        visual = VGroup(visual_box, link_a, link_b, link_c, core_dot, branch_a, branch_b, branch_c)",
                ]
            )

        lines = [
            "from manim import *",
            "",
            f"class {self._scene_name}(MovingCameraScene):",
            "    def construct(self):",
            "        self.camera.background_color = WHITE",
            "        self.camera.frame.set(width=13.5)",
            "",
            "        TEXT_COLOR = BLACK",
            "        VAR_COLOR = '#2563EB'",
            "        CONST_COLOR = '#16A34A'",
            "        ANSWER_COLOR = '#EAB308'",
            "        MUTED = GREY_D",
            "",
            f"        variable_tokens = {repr(variable_tokens)}",
            f"        constant_tokens = {repr(constant_tokens)}",
            f"        key_ideas = {repr(key_ideas)}",
            f"        steps = {repr(steps)}",
            f"        example1_eqs = {repr(example1_eqs)}",
            f"        example2_eqs = {repr(example2_eqs)}",
            f"        component_rows = {repr(component_rows)}",
            "",
            "        def teacher_text(content, size=28, color=TEXT_COLOR, max_width=11.0, weight=NORMAL):",
            "            text = Text(content, font_size=size, color=color, weight=weight)",
            "            if text.width > max_width:",
            "                text.scale_to_fit_width(max_width)",
            "            return text",
            "",
            "        def fit_math(eq, max_width=5.8, max_height=1.35):",
            "            if eq.width > max_width:",
            "                eq.scale_to_fit_width(max_width)",
            "            if eq.height > max_height:",
            "                eq.scale_to_fit_height(max_height)",
            "            return eq",
            "",
            "        def build_math(expr, size=54):",
            "            eq = MathTex(expr, font_size=size, color=TEXT_COLOR)",
            "            for token in variable_tokens:",
            "                eq.set_color_by_tex(token, VAR_COLOR)",
            "            for token in constant_tokens:",
            "                eq.set_color_by_tex(token, CONST_COLOR)",
            "            return fit_math(eq)",
            "",
            "        brand = teacher_text('GyanDeep', size=22, color=MUTED, max_width=2.4, weight=BOLD).to_corner(UR, buff=0.25)",
            f"        title = teacher_text({repr(title)}, size=40, color=TEXT_COLOR, max_width=11.2, weight=BOLD).to_edge(UP, buff=0.3)",
            f"        learning_goal = teacher_text({repr(learning_goal)}, size=24, color=MUTED, max_width=10.8).next_to(title, DOWN, buff=0.2)",
            f"        hook = teacher_text({repr(hook)}, size=28, color=TEXT_COLOR, max_width=10.6).next_to(learning_goal, DOWN, buff=0.55)",
            "",
            "        self.add(brand)",
            "        self.play(Write(title), FadeIn(learning_goal, shift=UP * 0.1), run_time=1.0)",
            "        self.play(Write(hook), run_time=1.0)",
            "        self.wait(1.0)",
            "",
            f"        example_header = teacher_text({repr(example1_caption)}, size=24, color=MUTED, max_width=7.0, weight=BOLD)",
            "        current_eq = build_math(example1_eqs[0], size=50)",
            "        problem_group = VGroup(example_header, current_eq).arrange(DOWN, buff=0.25).move_to(DOWN * 0.1)",
            "        self.play(ReplacementTransform(hook, example_header), Write(current_eq), run_time=1.2)",
            "        self.wait(1.0)",
            "",
            f"        concept_note = teacher_text({repr(concept_intro)}, size=24, color=MUTED, max_width=10.8).next_to(problem_group, DOWN, buff=0.4)",
            "        self.play(FadeIn(concept_note, shift=UP * 0.08), run_time=0.8)",
            "        self.wait(0.8)",
            "",
            "        formula_header = teacher_text('Core Formula', size=24, color=MUTED, max_width=4.6, weight=BOLD).move_to(example_header)",
            f"        formula_eq = build_math({repr(formula_latex)}, size=48).move_to(current_eq)",
            "        self.play(",
            "            Transform(example_header, formula_header),",
            "            FadeOut(concept_note, shift=DOWN * 0.08),",
            "            TransformMatchingTex(current_eq, formula_eq),",
            "            self.camera.frame.animate.move_to(formula_eq).set(width=13.2),",
            "            run_time=1.2,",
            "        )",
            "        current_eq = formula_eq",
            "        self.wait(0.9)",
            "",
            "        formula_anchor = VGroup(example_header, current_eq).arrange(DOWN, buff=0.22)",
            "        formula_anchor.move_to(UP * 0.9)",
            "        example_header.move_to(formula_anchor[0])",
            "        current_eq.move_to(formula_anchor[1])",
            "        key_title = teacher_text('Key Ideas', size=24, color=MUTED, max_width=3.6, weight=BOLD)",
            "        idea_rows = []",
            "        for idx, text in enumerate(key_ideas):",
            "            bullet = Dot(radius=0.055, color=[VAR_COLOR, CONST_COLOR, ANSWER_COLOR][idx % 3])",
            "            note = teacher_text(text, size=18, color=TEXT_COLOR, max_width=4.2)",
            "            row = VGroup(bullet, note).arrange(RIGHT, buff=0.16)",
            "            idea_rows.append(row)",
            "        key_group = VGroup(key_title, *idea_rows).arrange(DOWN, aligned_edge=LEFT, buff=0.16)",
            "        key_group.to_edge(LEFT, buff=0.85).shift(DOWN * 1.35)",
            "",
            "        part_title = teacher_text('Key Parts', size=22, color=MUTED, max_width=4.0, weight=BOLD)",
            "        part_rows = []",
            "        for label, description, kind in component_rows:",
            "            swatch = {'variable': VAR_COLOR, 'constant': CONST_COLOR, 'answer': ANSWER_COLOR}.get(kind, GREY_C)",
            "            symbol = teacher_text(label, size=18, color=swatch, max_width=1.6, weight=BOLD)",
            "            note = teacher_text(description, size=16, color=TEXT_COLOR, max_width=3.8)",
            "            row = VGroup(symbol, note).arrange(RIGHT, buff=0.2, aligned_edge=UP)",
            "            part_rows.append(row)",
            "        part_group = VGroup(part_title, *part_rows).arrange(DOWN, aligned_edge=LEFT, buff=0.14)",
            "        part_group.to_edge(LEFT, buff=0.85).shift(UP * 0.05)",
            "",
        ]
        lines.extend(visual_lines)
        lines.extend(
            [
                "        visual.scale_to_fit_width(min(visual.width, 3.2))",
                "        if visual.height > 2.2:",
                "            visual.scale_to_fit_height(2.2)",
                "        visual.to_edge(RIGHT, buff=0.75).shift(DOWN * 1.4)",
                "        self.play(FadeIn(visual, shift=LEFT * 0.08), run_time=0.7)",
                "        self.play(FadeIn(key_title, shift=UP * 0.08), run_time=0.35)",
                "        for row in idea_rows:",
                "            self.play(FadeIn(row, shift=RIGHT * 0.08), run_time=0.35)",
                "        self.play(FadeIn(part_group, shift=RIGHT * 0.08), run_time=0.45)",
                "        self.play(Indicate(current_eq, color=VAR_COLOR, scale_factor=1.03), run_time=0.7)",
                "        self.wait(1.0)",
                "",
                "        solution_header = teacher_text('Solution Steps', size=24, color=MUTED, max_width=5.6, weight=BOLD).move_to(example_header)",
                "        step_note = teacher_text(f'Step 1: {steps[0]}', size=20, color=TEXT_COLOR, max_width=4.8).to_edge(LEFT, buff=0.85).shift(DOWN * 0.1)",
                "        example_eq = build_math(example1_eqs[0], size=50).move_to(current_eq)",
                "        self.play(",
                "            FadeOut(key_group, shift=DOWN * 0.08),",
                "            FadeOut(part_group, shift=LEFT * 0.08),",
                "            FadeOut(visual, shift=RIGHT * 0.08),",
                "            Transform(example_header, solution_header),",
                "            TransformMatchingTex(current_eq, example_eq),",
                "            FadeIn(step_note, shift=RIGHT * 0.08),",
                "            self.camera.frame.animate.move_to(example_eq).set(width=12.4),",
                "            run_time=1.0,",
                "        )",
                "        current_eq = example_eq",
                "        current_eq.shift(DOWN * 0.45)",
                "        self.play(current_eq.animate.move_to(ORIGIN + DOWN * 0.15), run_time=0.3)",
                "        self.wait(0.9)",
                "",
                "        for idx, expr in enumerate(example1_eqs[1:], start=2):",
                "            next_eq = build_math(expr, size=50).move_to(current_eq)",
                "            note_index = min(idx - 1, len(steps) - 1)",
                "            next_note = teacher_text(f'Step {min(idx, 3)}: {steps[note_index]}', size=20, color=TEXT_COLOR, max_width=4.8).move_to(step_note)",
                "            self.play(",
                "                Transform(step_note, next_note),",
                "                TransformMatchingTex(current_eq, next_eq),",
                "                run_time=1.15,",
                "            )",
                "            current_eq = next_eq",
                "            self.wait(0.8)",
                "",
                "        answer_header = teacher_text('Final answer', size=22, color=ANSWER_COLOR, max_width=3.2, weight=BOLD)",
                "        answer_header.move_to(ORIGIN + UP * 1.55)",
                "        answer_eq = current_eq.copy()",
                "        answer_eq.move_to(ORIGIN + DOWN * 0.15)",
                "        final_box = SurroundingRectangle(current_eq, color=ANSWER_COLOR, buff=0.22, corner_radius=0.14)",
                "        final_box = SurroundingRectangle(answer_eq, color=ANSWER_COLOR, buff=0.28, corner_radius=0.14)",
                f"        insight_note = teacher_text({repr(insight)}, size=18, color=MUTED, max_width=8.8).next_to(final_box, DOWN, buff=0.55)",
                f"        interpretation = teacher_text({repr(interpretation)}, size=19, color=MUTED, max_width=8.8).next_to(final_box, DOWN, buff=0.58)",
                "        self.play(",
                "            FadeOut(solution_header, shift=UP * 0.08),",
                "            TransformMatchingTex(current_eq, answer_eq),",
                "            FadeOut(step_note, shift=LEFT * 0.08),",
                "            FadeIn(answer_header, shift=UP * 0.08),",
                "            Create(final_box),",
                "            run_time=0.8,",
                "        )",
                "        current_eq = answer_eq",
                "        self.play(FadeIn(insight_note, shift=UP * 0.08), Indicate(current_eq, color=ANSWER_COLOR, scale_factor=1.04), run_time=0.8)",
                "        self.wait(1.0)",
                "        self.play(FadeOut(insight_note, shift=DOWN * 0.08), FadeIn(interpretation, shift=UP * 0.08), run_time=0.7)",
                "        self.wait(1.0)",
                "",
                f"        example2_header = teacher_text({repr(example2_caption)}, size=24, color=MUTED, max_width=7.0, weight=BOLD).move_to(answer_header)",
                "        faster_note = teacher_text('Same method, a little faster.', size=22, color=MUTED, max_width=5.0).move_to(step_note)",
                "        next_eq = build_math(example2_eqs[0], size=48).move_to(current_eq)",
                "        self.play(",
                "            FadeOut(final_box),",
                "            FadeOut(interpretation),",
                "            Transform(answer_header, example2_header),",
                "            FadeIn(faster_note, shift=RIGHT * 0.08),",
                "            TransformMatchingTex(current_eq, next_eq),",
                "            run_time=1.0,",
                "        )",
                "        answer_header = example2_header",
                "        step_note = faster_note",
                "        current_eq = next_eq",
                "        self.wait(0.7)",
                "",
                "        for expr in example2_eqs[1:]:",
                "            quick_eq = build_math(expr, size=48).move_to(current_eq)",
                "            self.play(TransformMatchingTex(current_eq, quick_eq), run_time=0.85)",
                "            current_eq = quick_eq",
                "            self.wait(0.55)",
                "",
                "        second_box = SurroundingRectangle(current_eq, color=ANSWER_COLOR, buff=0.2, corner_radius=0.12)",
                "        self.play(Create(second_box), Indicate(current_eq, color=ANSWER_COLOR, scale_factor=1.03), run_time=0.65)",
                "        self.wait(0.8)",
                "",
                "        mistake_title = teacher_text('Common mistake', size=22, color='#B91C1C', max_width=4.6, weight=BOLD)",
                f"        mistake_eq = build_math({repr(mistake)}, size=42)",
                "        mistake_eq.set_color('#B91C1C')",
                "        mistake_group = VGroup(mistake_title, mistake_eq).arrange(DOWN, buff=0.16)",
                "        mistake_group.move_to(ORIGIN + DOWN * 1.55)",
                "        mistake_mark = Cross(mistake_eq, stroke_color='#B91C1C', stroke_width=4)",
                "        self.play(FadeIn(mistake_group, shift=UP * 0.08), Create(mistake_mark), run_time=0.85)",
                "        self.wait(0.8)",
                "",
                "        recap_title = teacher_text('Recap', size=24, color=MUTED, max_width=3.0, weight=BOLD)",
                "        recap_rows = [teacher_text(f'{idx + 1}. {step}', size=18, color=TEXT_COLOR, max_width=10.0) for idx, step in enumerate(steps)]",
                "        recap_group = VGroup(recap_title, *recap_rows).arrange(DOWN, aligned_edge=LEFT, buff=0.26)",
                "        recap_group.to_edge(LEFT, buff=0.85).shift(UP * 0.25)",
                "        self.play(",
                "            FadeOut(title),",
                "            FadeOut(learning_goal),",
                "            FadeOut(answer_header),",
                "            FadeOut(step_note),",
                "            FadeOut(second_box),",
                "            FadeOut(current_eq),",
                "            FadeOut(mistake_group),",
                "            FadeOut(mistake_mark),",
                "            self.camera.frame.animate.move_to(ORIGIN).set(width=13.5),",
                "            run_time=0.9,",
                "        )",
                "        self.play(FadeIn(recap_title, shift=UP * 0.08), run_time=0.4)",
                "        for row in recap_rows:",
                "            self.play(FadeIn(row, shift=RIGHT * 0.08), run_time=0.35)",
                "        self.wait(1.6)",
            ]
        )

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
