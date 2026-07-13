"""Orchestrate one strict query-wiki exploration run."""

from __future__ import annotations

import json
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
    """Run the explorer once and validate its exact structured output."""

    resolved_backend = backend or ClaudeCodeWikiExplorerBackend(
        env_file=config.env_file,
        max_selected_skills=config.max_selected_skills,
        model=config.model,
        sdk_runtime=sdk_runtime,
        max_turns=config.max_turns,
        load_timeout_ms=config.load_timeout_ms,
        execution_timeout_seconds=config.execution_timeout_seconds,
    )
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
    cc_dir = trace_dir / "cc_explorer"
    cc_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        cc_dir / "validation.json",
        json.dumps(validation.to_dict(), ensure_ascii=False, indent=2) + "\n",
    )
    return WikiExplorerRun(package=package, validation=validation)


__all__ = ["WikiExplorerConfig", "WikiExplorerRun", "explore_query_wiki"]
