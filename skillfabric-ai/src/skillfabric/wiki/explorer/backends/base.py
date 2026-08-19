"""Protocol for Task Wiki Explorer backends."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from skillfabric.wiki.explorer.skill_package import SkillPackage

ExplorerBackendName = Literal["claude", "codex"]
EXPLORER_BACKENDS: tuple[ExplorerBackendName, ...] = ("claude", "codex")


def normalize_explorer_backend(value: str) -> ExplorerBackendName:
    """Validate the short name used to select an explorer SDK."""

    if value not in EXPLORER_BACKENDS:
        choices = ", ".join(EXPLORER_BACKENDS)
        raise ValueError(f"unsupported explorer backend: {value!r}; choose one of {choices}")
    return value


class WikiExplorerBackend(Protocol):
    """Task time Explorer backend over a materialized task_wiki directory."""

    def explore(
        self,
        *,
        query: str,
        task_wiki_root: Path,
        trace_dir: Path,
    ) -> SkillPackage:
        """Read task_wiki and return a proposed SkillPackage."""


__all__ = [
    "EXPLORER_BACKENDS",
    "ExplorerBackendName",
    "WikiExplorerBackend",
    "normalize_explorer_backend",
]
