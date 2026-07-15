"""Orchestrate one strict query-wiki exploration run."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from skillfabric.storage import atomic_write_text
from skillfabric.wiki.explorer.backends.base import WikiExplorerBackend
from skillfabric.wiki.explorer.backends.claude_code import (
    ClaudeCodeSdkRuntime,
    ClaudeCodeWikiExplorerBackend,
)
from skillfabric.wiki.explorer.skill_package import SkillPackage
from skillfabric.wiki.explorer.validation import (
    SkillPackageValidationResult,
    validate_skill_package,
)


@dataclass(frozen=True, slots=True)
class WikiExplorerConfig:
    env_file: str | Path = ".env"
    max_selected_skills: int = 8
    model: str | None = None
    max_turns: int = 24
    load_timeout_ms: int = 30_000
    execution_timeout_seconds: float = 300.0
    max_attempts: int = 2
    retry_delay_seconds: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        if (
            isinstance(self.retry_delay_seconds, bool)
            or not isinstance(self.retry_delay_seconds, (int, float))
            or not math.isfinite(self.retry_delay_seconds)
            or self.retry_delay_seconds < 0
        ):
            raise ValueError("retry_delay_seconds must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class WikiExplorerRun:
    package: SkillPackage
    validation: SkillPackageValidationResult


def explore_query_wiki(
    config: WikiExplorerConfig,
    *,
    query: str,
    query_wiki_root: Path,
    trace_dir: Path,
    sdk_runtime: ClaudeCodeSdkRuntime | None = None,
    backend: WikiExplorerBackend | None = None,
) -> WikiExplorerRun:
    """Retry exploration until the selected package passes validation."""

    resolved_backend = backend or ClaudeCodeWikiExplorerBackend(
        env_file=config.env_file,
        max_selected_skills=config.max_selected_skills,
        model=config.model,
        sdk_runtime=sdk_runtime,
        max_turns=config.max_turns,
        load_timeout_ms=config.load_timeout_ms,
        execution_timeout_seconds=config.execution_timeout_seconds,
    )
    if not query_wiki_root.is_dir():
        raise FileNotFoundError(f"query_wiki root does not exist: {query_wiki_root}")
    cc_dir = trace_dir / "cc_explorer"
    for attempt in range(1, config.max_attempts + 1):
        if attempt > 1:
            _clear_attempt_artifacts(cc_dir)
        try:
            package = resolved_backend.explore(
                query=query,
                query_wiki_root=query_wiki_root,
                trace_dir=trace_dir,
            )
            validation = validate_skill_package(
                package,
                query_wiki_root,
                max_selected_skills=config.max_selected_skills,
            )
            cc_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                cc_dir / "validation.json",
                json.dumps(validation.to_dict(), ensure_ascii=False, indent=2) + "\n",
            )
            if validation.valid:
                return WikiExplorerRun(package=package, validation=validation)
            error = ValueError("; ".join(validation.errors) or "invalid SkillPackage")
        except Exception as exc:  # noqa: BLE001 - one bounded route retry policy.
            error = exc
        if attempt == config.max_attempts:
            raise error
        if config.retry_delay_seconds:
            time.sleep(config.retry_delay_seconds)
    raise AssertionError("explorer retry loop ended unexpectedly")


def _clear_attempt_artifacts(cc_dir: Path) -> None:
    for name in ("skill_package.json", "usage.json", "validation.json", "error.json"):
        path = cc_dir / name
        if path.exists():
            path.unlink()


__all__ = ["WikiExplorerConfig", "WikiExplorerRun", "explore_query_wiki"]
