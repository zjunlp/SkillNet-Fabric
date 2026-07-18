"""Single public runtime defaults for build and routing."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RouterOptions:
    max_selected_skills: int = 8
    seed_limit: int = 24
    expanded_limit: int = 100
    max_depth: int = 2
    explorer_max_turns: int = 24
    explorer_load_timeout_ms: int = 30_000
    explorer_timeout_seconds: float = 300.0
    explorer_max_attempts: int = 2
    explorer_retry_delay_seconds: float = 1.0

    def __post_init__(self) -> None:
        for name in ("max_selected_skills", "seed_limit", "max_depth"):
            _require_nonnegative_int(getattr(self, name), name=name)
        if (
            isinstance(self.expanded_limit, bool)
            or not isinstance(self.expanded_limit, int)
            or self.expanded_limit < self.seed_limit
        ):
            raise ValueError("expanded_limit must be an integer at least seed_limit")
        _require_positive_int(self.explorer_max_turns, name="explorer_max_turns")
        _require_positive_int(
            self.explorer_load_timeout_ms,
            name="explorer_load_timeout_ms",
        )
        _require_positive_int(self.explorer_max_attempts, name="explorer_max_attempts")
        if (
            isinstance(self.explorer_timeout_seconds, bool)
            or not isinstance(self.explorer_timeout_seconds, (int, float))
            or not math.isfinite(self.explorer_timeout_seconds)
            or self.explorer_timeout_seconds <= 0
        ):
            raise ValueError("explorer_timeout_seconds must be finite and positive")
        if (
            isinstance(self.explorer_retry_delay_seconds, bool)
            or not isinstance(self.explorer_retry_delay_seconds, (int, float))
            or not math.isfinite(self.explorer_retry_delay_seconds)
            or self.explorer_retry_delay_seconds < 0
        ):
            raise ValueError("explorer_retry_delay_seconds must be finite and non-negative")


def default_router_options() -> RouterOptions:
    return RouterOptions(
        max_selected_skills=_nonnegative_int("SKILLFABRIC_MAX_SELECTED_SKILLS", 8),
        seed_limit=_nonnegative_int("SKILLFABRIC_SEED_LIMIT", 24),
        expanded_limit=_nonnegative_int("SKILLFABRIC_EXPANDED_LIMIT", 100),
        max_depth=_nonnegative_int("SKILLFABRIC_MAX_GRAPH_DEPTH", 2),
        explorer_max_turns=_positive_int("SKILLFABRIC_EXPLORER_MAX_TURNS", 24),
        explorer_load_timeout_ms=_positive_int(
            "SKILLFABRIC_EXPLORER_LOAD_TIMEOUT_MS",
            30_000,
        ),
        explorer_timeout_seconds=_positive_float(
            "SKILLFABRIC_EXPLORER_TIMEOUT_SECONDS",
            300.0,
        ),
        explorer_max_attempts=_positive_int("SKILLFABRIC_EXPLORER_MAX_ATTEMPTS", 2),
        explorer_retry_delay_seconds=_nonnegative_float(
            "SKILLFABRIC_EXPLORER_RETRY_DELAY_SECONDS",
            1.0,
        ),
    )


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _require_nonnegative_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "RouterOptions",
    "default_router_options",
]
