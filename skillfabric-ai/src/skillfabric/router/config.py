"""Configuration for query-wiki routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillfabric.router.traces import validate_trace_id

RouterSdkRuntime = Any


@dataclass(frozen=True, slots=True)
class RouterConfig:
    workspace: str | Path = ".skillfabric"
    query: str = ""
    env_file: str | Path = ".env"
    max_selected_skills: int = 8
    seed_limit: int = 24
    expanded_limit: int = 100
    max_depth: int = 2
    trace_id: str | None = None
    explorer_model: str | None = None
    explorer_max_turns: int = 24
    explorer_load_timeout_ms: int = 30_000
    explorer_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.trace_id is not None:
            validate_trace_id(self.trace_id)

        for name in ("max_selected_skills", "seed_limit", "max_depth"):
            _require_int_at_least(getattr(self, name), name=name, minimum=0)
        _require_int_at_least(
            self.expanded_limit,
            name="expanded_limit",
            minimum=self.seed_limit,
        )
        _require_int_at_least(
            self.explorer_max_turns,
            name="explorer_max_turns",
            minimum=1,
        )
        _require_int_at_least(
            self.explorer_load_timeout_ms,
            name="explorer_load_timeout_ms",
            minimum=1_000,
        )
        timeout = self.explorer_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 1.0
        ):
            raise ValueError("explorer_timeout_seconds must be finite and at least 1")


def _require_int_at_least(value: Any, *, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
