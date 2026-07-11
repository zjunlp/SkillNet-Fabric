"""Models for Wiki materialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

WikiPageType = Literal["skill", "workflow", "index"]


@dataclass(slots=True)
class WikiBuildConfig:
    """Configuration for building the graph-backed wiki."""

    workspace: str | Path = ".skillfabric"
    env_file: str | Path = ".env"
    use_llm_summaries: bool = True
    max_neighbors_per_section: int = 10
    llm_concurrency: int | None = None
    llm_rate_limit_per_minute: float | None = None
    llm_max_retries: int | None = None
    llm_retry_backoff_seconds: float | None = None
    llm_progress_every: int | None = None
    llm_batch_size: int | None = None


@dataclass(slots=True)
class WikiPage:
    """Markdown page emitted by the wiki materializer."""

    path: Path
    page_type: WikiPageType
    entity_id: str
    title: str
    text: str


@dataclass(slots=True)
class WikiSummaryRecord:
    """Cached summary for one wiki entity."""

    page_type: str
    entity_id: str
    content_hash: str
    model_id: str
    routing_summary: str = ""
    workflow_summary: str = ""
    summary: str = ""
    provenance: str = "deterministic_fallback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_type": self.page_type,
            "entity_id": self.entity_id,
            "content_hash": self.content_hash,
            "model_id": self.model_id,
            "routing_summary": self.routing_summary,
            "workflow_summary": self.workflow_summary,
            "summary": self.summary,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WikiSummaryRecord:
        return cls(
            page_type=str(payload.get("page_type", "")),
            entity_id=str(payload.get("entity_id", "")),
            content_hash=str(payload.get("content_hash", "")),
            model_id=str(payload.get("model_id", "")),
            routing_summary=str(payload.get("routing_summary", "")),
            workflow_summary=str(payload.get("workflow_summary", "")),
            summary=str(payload.get("summary", "")),
            provenance=str(payload.get("provenance", "deterministic_fallback")),
        )


@dataclass(slots=True)
class WikiHealthReport:
    """Health report for a generated wiki."""

    missing_skill_pages: list[str] = field(default_factory=list)
    broken_links: list[str] = field(default_factory=list)
    orphan_skill_pages: list[str] = field(default_factory=list)
    skills_without_interface: list[str] = field(default_factory=list)
    skills_without_graph_links: list[str] = field(default_factory=list)
    raw_llm_output_leaks: list[str] = field(default_factory=list)
    fallback_count: int = 0

    @property
    def summary(self) -> dict[str, int]:
        return {
            "missing_skill_page_count": len(self.missing_skill_pages),
            "broken_link_count": len(self.broken_links),
            "orphan_skill_page_count": len(self.orphan_skill_pages),
            "skill_without_interface_count": len(self.skills_without_interface),
            "skill_without_graph_links_count": len(self.skills_without_graph_links),
            "raw_llm_output_leak_count": len(self.raw_llm_output_leaks),
            "summary_fallback_count": self.fallback_count,
        }


@dataclass(slots=True)
class WikiBuildResult:
    """Result returned by wiki materialization."""

    pages_written: int
    cache_hits: int
    llm_calls: int
    fallback_count: int
    health: WikiHealthReport
    workspace: Path

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "pages_written": self.pages_written,
            "cache_hits": self.cache_hits,
            "llm_calls": self.llm_calls,
            "fallback_count": self.fallback_count,
            **self.health.summary,
            "workspace": str(self.workspace),
        }
