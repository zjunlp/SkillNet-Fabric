"""Configuration for final router orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

RouterSdkRuntime = Any


@dataclass(slots=True)
class RouterConfig:
    """Configuration for final skill routing."""

    workspace: str | Path = ".skillfabric"
    query: str = ""
    env_file: str | Path = ".env"
    use_llm_router: bool = True
    max_selected_skills: int = 8
    seed_limit: int = 8
    expanded_limit: int = 32
    workflow_confidence_threshold: float = 0.95
    max_workflow_hints: int = 12
    trace_id: str | None = None
    explorer_backend: str = "claude-code"
    explorer_model: str | None = None
    strict_explorer: bool = False
    explorer_max_turns: int = 24
    explorer_load_timeout_ms: int = 30_000
    explorer_timeout_seconds: float = 300.0
