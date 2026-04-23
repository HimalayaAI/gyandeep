from __future__ import annotations

import json
import re
import textwrap


def clip_text(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(text or "").strip())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def extract_python_block(text: str) -> str:
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


def extract_json_object(text: str) -> dict | None:
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


def wrap_text(text: str, width: int = 34, max_lines: int = 3) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    lines = textwrap.wrap(text, width=width)
    if not lines:
        return ""
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if not lines[-1].endswith("..."):
            lines[-1] = lines[-1].rstrip(".") + "..."
    return "\n".join(lines)


def latex_to_text(expr: str) -> str:
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
