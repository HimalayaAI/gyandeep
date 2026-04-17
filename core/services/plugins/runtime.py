from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable, Protocol

from core.models.plugins import PluginJobRequest, PluginJobResult

EmitFn = Callable[[str, str], Awaitable[None]]


class PluginHandler(Protocol):
    plugin_id: str

    async def run(self, request: PluginJobRequest, emit: EmitFn) -> PluginJobResult:
        ...


class PluginRuntime:
    """Generic plugin host for background job execution."""

    def __init__(self, artifact_root: str | Path):
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._handlers: dict[str, PluginHandler] = {}

    def register(self, handler: PluginHandler) -> None:
        self._handlers[handler.plugin_id] = handler

    def has_handler(self, plugin_id: str) -> bool:
        return plugin_id in self._handlers

    def create_job_dir(self, job_id: str) -> Path:
        job_dir = self.artifact_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    async def run_job(self, request: PluginJobRequest, emit: EmitFn) -> PluginJobResult:
        if request.plugin_id not in self._handlers:
            raise ValueError(f"Unknown plugin: {request.plugin_id}")
        return await self._handlers[request.plugin_id].run(request, emit)
