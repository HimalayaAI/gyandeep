from __future__ import annotations

import importlib.util
import shutil
import sys

from .manim_plugin import (
    ManimVideoService,
    clip_text as _clip_text,
    extract_json_object as _extract_json_object,
    extract_python_block as _extract_python_block,
    latex_to_text as _latex_to_text,
    wrap_text as _wrap_text,
)
from .manim_plugin.planning import fallback_formula as _fallback_formula, fallback_plan as _fallback_plan
from .manim_plugin.rendering import resolve_manim_cli as _resolve_manim_cli_impl
from .manim_plugin.rendering import resolve_manim_command as _resolve_manim_command_impl
from .manim_plugin.scripting import script_looks_valid as _script_looks_valid_impl


class ManimVideoPlugin(ManimVideoService):
    @staticmethod
    def _clip(text: str, limit: int = 220) -> str:
        return _clip_text(text, limit)

    @staticmethod
    def _extract_python_block(text: str) -> str:
        return _extract_python_block(text)

    @staticmethod
    def _extract_json_object(text: str) -> dict | None:
        return _extract_json_object(text)

    @staticmethod
    def _wrap_text(text: str, width: int = 34, max_lines: int = 3) -> str:
        return _wrap_text(text, width=width, max_lines=max_lines)

    @staticmethod
    def _latex_to_text(expr: str) -> str:
        return _latex_to_text(expr)

    def _fallback_formula(self, query: str, context_text: str) -> str:
        return _fallback_formula(query, context_text)

    def _fallback_plan(self, query: str, context_text: str) -> dict[str, object]:
        return _fallback_plan(query, context_text, scene_name=self._scene_name)

    @staticmethod
    def _script_looks_valid(script: str, scene_name: str) -> bool:
        return _script_looks_valid_impl(script, scene_name)

    @staticmethod
    def _resolve_manim_command() -> list[str]:
        return _resolve_manim_command_impl(
            which_fn=shutil.which,
            find_spec_fn=importlib.util.find_spec,
            executable_path=sys.executable,
        )

    @staticmethod
    def _resolve_manim_cli() -> str:
        return _resolve_manim_cli_impl(
            which_fn=shutil.which,
            find_spec_fn=importlib.util.find_spec,
            executable_path=sys.executable,
        )

