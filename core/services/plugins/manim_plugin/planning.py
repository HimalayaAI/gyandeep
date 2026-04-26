from __future__ import annotations

import re
from textwrap import dedent

from ..runtime import PluginJobRequest
from .text_utils import clip_text, latex_to_text


def build_plan_prompt(query: str, context_text: str, style_context: str) -> str:
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
        - "visual_focus": one of "triangle", "circle", "algebra", "numberline", "generic"
        - "answer_line": one sentence with the direct answer idea
        """
    ).strip()


def fallback_formula(query: str, context_text: str) -> str:
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


def fallback_plan(query: str, context_text: str, scene_name: str = "LessonScene") -> dict[str, object]:
    scope = f"{query} {context_text}".lower()
    if "real number" in scope or "real numbers" in scope:
        return {
            "title": "Understanding real numbers",
            "learning_goal": "See how rational and irrational numbers together make the real number system.",
            "formula_latex": "Real numbers = rational numbers + irrational numbers",
            "steps": [
                "Start with the number line because every real number can be placed on it.",
                "Separate rational numbers such as integers and fractions from irrational numbers.",
                "Show that both groups still live on the same continuous number line.",
                "Conclude that real numbers include every rational and irrational value.",
            ],
            "worked_example": [
                "-2 and 5 are rational because they can be written as fractions.",
                "1/2 is rational, while sqrt(2) and pi are irrational.",
                "All of them are real because each one has a point on the number line.",
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

    formula_latex = fallback_formula(query, context_text)
    query_line = clip_text(query, 78)
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
            f"Use formula: {latex_to_text(formula_latex)}",
            "Compute each step and present the final result clearly.",
        ],
        "visual_focus": visual_focus,
        "answer_line": "The answer follows from applying the formula step by step.",
    }


def normalize_plan(
    raw_plan: dict | None,
    query: str,
    context_text: str,
    scene_name: str = "LessonScene",
) -> dict[str, object]:
    base = fallback_plan(query, context_text, scene_name=scene_name)
    if not raw_plan:
        return base

    plan = dict(base)
    for field in ("title", "learning_goal", "formula_latex", "visual_focus", "answer_line"):
        value = raw_plan.get(field)
        if isinstance(value, str) and value.strip():
            plan[field] = clip_text(value, 220)

    for field in ("steps", "worked_example"):
        value = raw_plan.get(field)
        cleaned: list[str] = []
        if isinstance(value, list):
            cleaned = [clip_text(str(item), 180) for item in value if str(item).strip()]
        elif isinstance(value, str) and value.strip():
            pieces = re.split(r"(?:\n+|•|- )", value.strip())
            cleaned = [clip_text(piece, 180) for piece in pieces if piece.strip()]
        if cleaned:
            plan[field] = cleaned[:6] if field == "steps" else cleaned[:5]

    visual = str(plan.get("visual_focus", "generic")).strip().lower()
    if visual not in {"triangle", "circle", "algebra", "numberline", "generic"}:
        visual = "generic"
    plan["visual_focus"] = visual

    formula = str(plan.get("formula_latex", "")).strip()
    if not formula:
        plan["formula_latex"] = fallback_formula(query, context_text)

    if not plan["steps"]:
        plan["steps"] = base["steps"]
    if not plan["worked_example"]:
        plan["worked_example"] = base["worked_example"]
    return plan


def plan_to_markdown(request: PluginJobRequest, plan: dict[str, object], plan_mode: str) -> str:
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
