"""Construct the configured query-wiki explorer backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skillfabric.wiki.explorer.backends.base import (
    ExplorerBackendName,
    WikiExplorerBackend,
    normalize_explorer_backend,
)
from skillfabric.wiki.explorer.backends.claude_code import ClaudeCodeWikiExplorerBackend
from skillfabric.wiki.explorer.backends.codex import CodexWikiExplorerBackend


def create_explorer_backend(
    name: str,
    *,
    env_file: str | Path,
    max_selected_skills: int,
    required_selected_skills: int | None,
    model: str | None,
    reasoning_effort: str | None,
    max_turns: int,
    load_timeout_ms: int,
    execution_timeout_seconds: float,
    sdk_runtime: Any = None,
    tool_budget: dict[str, int] | None = None,
) -> WikiExplorerBackend:
    """Create one of the supported SDK-backed explorer implementations."""

    backend_name: ExplorerBackendName = normalize_explorer_backend(name)
    common = {
        "env_file": env_file,
        "max_selected_skills": max_selected_skills,
        "required_selected_skills": required_selected_skills,
        "model": model,
        "execution_timeout_seconds": execution_timeout_seconds,
        "sdk_runtime": sdk_runtime,
        "tool_budget": tool_budget,
    }
    if backend_name == "claude":
        return ClaudeCodeWikiExplorerBackend(
            **common,
            max_turns=max_turns,
            load_timeout_ms=load_timeout_ms,
            reasoning_effort=reasoning_effort,
        )
    return CodexWikiExplorerBackend(
        **common,
        reasoning_effort=reasoning_effort or "medium",
    )


__all__ = ["create_explorer_backend"]
