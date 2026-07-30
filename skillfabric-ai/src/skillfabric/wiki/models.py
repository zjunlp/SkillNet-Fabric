"""Models for Wiki materialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

WikiPageType = Literal["skill", "workflow", "index"]
NO_WORKFLOW_GUIDANCE = "No strong workflow guidance."


@dataclass(slots=True)
class WikiBuildConfig:
    """Configuration for building the graph-backed wiki."""

    workspace: str | Path = ".skillfabric"
    max_neighbors_per_section: int = 10

    def __post_init__(self) -> None:
        _required_path(self.workspace, label="workspace")
        if (
            isinstance(self.max_neighbors_per_section, bool)
            or not isinstance(self.max_neighbors_per_section, int)
            or self.max_neighbors_per_section <= 0
        ):
            raise ValueError("max_neighbors_per_section must be a positive integer")


@dataclass(slots=True)
class WikiPage:
    """Markdown page emitted by the wiki materializer."""

    path: Path
    page_type: WikiPageType
    entity_id: str
    text: str


@dataclass(slots=True)
class WikiSummaryRecord:
    """Cached summary for one wiki entity."""

    page_type: str
    entity_id: str
    content_hash: str
    routing_summary: str
    workflow_summary: str
    summary: str

    def __post_init__(self) -> None:
        for field_name in (
            "page_type",
            "entity_id",
            "content_hash",
            "routing_summary",
            "workflow_summary",
            "summary",
        ):
            _required_string(getattr(self, field_name), label=field_name)


@dataclass(slots=True)
class WikiHealthReport:
    """Health report for a generated wiki."""

    missing_skill_pages: list[str] = field(default_factory=list)
    broken_links: list[str] = field(default_factory=list)
    orphan_skill_pages: list[str] = field(default_factory=list)
    skills_without_contract_sections: list[str] = field(default_factory=list)
    skills_without_graph_links: list[str] = field(default_factory=list)
    raw_llm_output_leaks: list[str] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, int]:
        return {
            "missing_skill_page_count": len(self.missing_skill_pages),
            "broken_link_count": len(self.broken_links),
            "orphan_skill_page_count": len(self.orphan_skill_pages),
            "skill_without_contract_sections_count": len(self.skills_without_contract_sections),
            "skill_without_graph_links_count": len(self.skills_without_graph_links),
            "raw_llm_output_leak_count": len(self.raw_llm_output_leaks),
        }


@dataclass(slots=True)
class WikiBuildResult:
    """Result returned by wiki materialization."""

    pages_written: int
    health: WikiHealthReport
    workspace: Path

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "pages_written": self.pages_written,
            **self.health.summary,
            "workspace": str(self.workspace),
        }


def _required_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _required_path(value: object, *, label: str) -> None:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{label} must be a path")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{label} must not be empty")
