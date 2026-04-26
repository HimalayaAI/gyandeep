from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_manim_command(
    which_fn=shutil.which,
    find_spec_fn=importlib.util.find_spec,
    executable_path: str | None = None,
) -> list[str]:
    executable_path = executable_path or sys.executable
    direct = which_fn("manim") or which_fn("manim.exe")
    if direct:
        return [direct]

    scripts_dir = Path(executable_path).resolve().parent
    for candidate_name in ("manim", "manim.exe"):
        sibling = scripts_dir / candidate_name
        if sibling.exists():
            return [str(sibling)]

    if find_spec_fn("manim") is not None:
        return [executable_path, "-m", "manim"]

    raise RuntimeError(
        f"Manim CLI not found for Python at {executable_path}. Install with `python -m pip install manim` in this environment."
    )


def resolve_manim_cli(
    which_fn=shutil.which,
    find_spec_fn=importlib.util.find_spec,
    executable_path: str | None = None,
) -> str:
    return " ".join(
        resolve_manim_command(which_fn=which_fn, find_spec_fn=find_spec_fn, executable_path=executable_path)
    )


def render_manim(
    script_path: Path,
    media_dir: Path,
    scene_name: str,
    quality: str,
    timeout_seconds: int,
) -> Path:
    media_dir.mkdir(parents=True, exist_ok=True)
    command = [
        *resolve_manim_command(),
        f"-q{quality}",
        str(script_path),
        scene_name,
        "-o",
        "lesson.mp4",
        "--media_dir",
        str(media_dir),
    ]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
        details = stderr or stdout or "No additional output captured before timeout."
        raise RuntimeError(
            f"Manim render timed out after {timeout_seconds} seconds. "
            f"Try increasing ANIMATION_RENDER_TIMEOUT_SECONDS or simplifying the scene. "
            f"Output: {details[:1200]}"
        ) from exc
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
