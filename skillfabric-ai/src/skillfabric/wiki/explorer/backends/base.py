"""Protocol for query-wiki explorer backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from skillfabric.wiki.explorer.skill_package import SkillPackage


class WikiExplorerBackend(Protocol):
    """Route-time explorer backend over a materialized query_wiki directory."""

    def explore(
        self,
        *,
        query: str,
        query_wiki_root: Path,
        trace_dir: Path,
    ) -> SkillPackage:
        """Read query_wiki and return a proposed SkillPackage."""
