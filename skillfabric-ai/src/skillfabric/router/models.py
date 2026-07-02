"""Models for final skill routing results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillfabric.router.task_atoms import TaskDecomposition


@dataclass(slots=True)
class RouterQuery:
    """User task passed into the Router."""

    task: str
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RouterQuery:
        return cls(
            task=str(payload.get("task", "")),
            metadata={str(key): str(value) for key, value in dict(payload.get("metadata", {})).items()},
        )


@dataclass(slots=True)
class RouterBundleConfig:
    """Configuration for query-local routing context assembly."""

    workspace: str | Path = ".skillfabric"
    query: str = ""
    task_atoms: TaskDecomposition | None = None
    env_file: str | Path | None = ".env"
    seed_limit: int = 8
    expanded_limit: int = 50
    candidate_pool_limit: int = 250
    workflow_confidence_threshold: float = 0.95
    max_workflow_hints: int = 12
    graph_expansion_mode: str = "ppr"
    ppr_alpha: float = 0.85
    ppr_max_iter: int = 50
    ppr_tol: float = 1e-8


@dataclass(slots=True)
class RouterSkillCandidate:
    """One selected skill with retrieval and graph-expansion evidence."""

    skill_id: str
    name: str
    score: float
    sources: list[str] = field(default_factory=list)
    graph_depth: int = 0
    reason: str = ""
    seed_score: float = 0.0
    ppr_score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    atom_coverage: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "score": round(self.score, 6),
            "seed_score": round(self.seed_score, 6),
            "ppr_score": round(self.ppr_score, 6),
            "score_breakdown": {
                str(key): round(float(value), 6)
                for key, value in sorted(self.score_breakdown.items())
            },
            "sources": sorted(set(self.sources)),
            "graph_depth": self.graph_depth,
            "reason": self.reason,
            "atom_coverage": {
                str(key): sorted({str(item) for item in value})
                for key, value in sorted(self.atom_coverage.items())
                if value
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RouterSkillCandidate:
        return cls(
            skill_id=str(payload.get("skill_id", "")),
            name=str(payload.get("name", "")),
            score=_safe_float(payload.get("score"), 0.0),
            sources=_string_list(payload.get("sources", [])),
            graph_depth=_safe_int(payload.get("graph_depth"), 0),
            reason=str(payload.get("reason", "")),
            seed_score=_safe_float(payload.get("seed_score"), 0.0),
            ppr_score=_safe_float(payload.get("ppr_score"), 0.0),
            score_breakdown={
                str(key): _safe_float(value, 0.0)
                for key, value in dict(payload.get("score_breakdown", {}) or {}).items()
            },
            atom_coverage={
                str(key): _string_list(value)
                for key, value in dict(payload.get("atom_coverage", {}) or {}).items()
            },
        )


@dataclass(slots=True)
class RouterCommunityContext:
    """Community context attached to selected skills."""

    community_id: str
    name: str
    summary: str
    selected_member_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "community_id": self.community_id,
            "name": self.name,
            "summary": self.summary,
            "selected_member_ids": sorted(set(self.selected_member_ids)),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RouterCommunityContext:
        return cls(
            community_id=str(payload.get("community_id", "")),
            name=str(payload.get("name", "")),
            summary=str(payload.get("summary", "")),
            selected_member_ids=_string_list(payload.get("selected_member_ids", [])),
        )


@dataclass(slots=True)
class RouterWorkflowHint:
    """Workflow hint scoped to selected skills for route ordering."""

    source_skill: str
    target_skill: str
    relation_type: str
    canonical_object: str
    confidence: float
    projected_edge_type: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_skill": self.source_skill,
            "target_skill": self.target_skill,
            "relation_type": self.relation_type,
            "canonical_object": self.canonical_object,
            "confidence": round(self.confidence, 6),
            "projected_edge_type": self.projected_edge_type,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RouterWorkflowHint:
        return cls(
            source_skill=str(payload.get("source_skill", "")),
            target_skill=str(payload.get("target_skill", "")),
            relation_type=str(payload.get("relation_type", "")),
            canonical_object=str(payload.get("canonical_object", "")),
            confidence=_safe_float(payload.get("confidence"), 0.0),
            projected_edge_type=str(payload.get("projected_edge_type", "")),
            reason=str(payload.get("reason", "")),
        )


@dataclass(slots=True)
class RouterBundle:
    """Query-local bundle consumed by routing and query-wiki exploration."""

    query: str
    selected_skills: list[RouterSkillCandidate]
    communities: list[RouterCommunityContext]
    workflow_hints: list[RouterWorkflowHint]
    wiki_pages: list[str]
    task_atoms: TaskDecomposition = field(default_factory=TaskDecomposition.empty)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "task_atoms": self.task_atoms.to_dict(),
            "selected_skills": [item.to_dict() for item in self.selected_skills],
            "communities": [item.to_dict() for item in self.communities],
            "workflow_hints": [item.to_dict() for item in self.workflow_hints],
            "wiki_pages": list(self.wiki_pages),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RouterBundle:
        return cls(
            query=str(payload.get("query", "")),
            task_atoms=TaskDecomposition.from_dict(
                dict(payload.get("task_atoms", {}) or {})
            ) if isinstance(payload.get("task_atoms", {}), dict) else TaskDecomposition.empty(),
            selected_skills=[
                RouterSkillCandidate.from_dict(item)
                for item in payload.get("selected_skills", [])
                if isinstance(item, dict)
            ],
            communities=[
                RouterCommunityContext.from_dict(item)
                for item in payload.get("communities", [])
                if isinstance(item, dict)
            ],
            workflow_hints=[
                RouterWorkflowHint.from_dict(item)
                for item in payload.get("workflow_hints", [])
                if isinstance(item, dict)
            ],
            wiki_pages=_string_list(payload.get("wiki_pages", [])),
            warnings=_string_list(payload.get("warnings", [])),
        )


@dataclass(slots=True)
class RouteSelectedSkill:
    """One final skill selected for the task."""

    skill_id: str
    name: str
    rank: int
    score: float = 0.0
    reason: str = ""
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "rank": self.rank,
            "score": round(float(self.score), 6),
            "reason": self.reason,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RouteSelectedSkill:
        return cls(
            skill_id=str(payload.get("skill_id", "")),
            name=str(payload.get("name", "")),
            rank=_safe_int(payload.get("rank"), 0),
            score=_safe_float(payload.get("score"), 0.0),
            reason=str(payload.get("reason", "")),
            evidence=_string_list(payload.get("evidence", [])),
        )


@dataclass(slots=True)
class RouteEdge:
    """Ordering edge for route and execution package generation."""

    before_skill: str
    after_skill: str
    edge_type: str
    confidence: float = 0.0
    reason: str = ""
    source: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.before_skill, self.after_skill, self.edge_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_skill": self.before_skill,
            "after_skill": self.after_skill,
            "edge_type": self.edge_type,
            "confidence": round(float(self.confidence), 6),
            "reason": self.reason,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RouteEdge:
        return cls(
            before_skill=str(payload.get("before_skill", "")),
            after_skill=str(payload.get("after_skill", "")),
            edge_type=str(payload.get("edge_type", "depend_on")),
            confidence=_safe_float(payload.get("confidence"), 0.0),
            reason=str(payload.get("reason", "")),
            source=str(payload.get("source", "")),
        )


@dataclass(slots=True)
class RouteResult:
    """Final router output consumed by execution package generation."""

    query: str
    trace_id: str
    trace_dir: Path
    selected_skills: list[RouteSelectedSkill]
    required_edges: list[RouteEdge] = field(default_factory=list)
    ordered_hints: list[RouteEdge] = field(default_factory=list)
    near_misses: list[dict[str, str]] = field(default_factory=list)
    wiki_pages_read: list[str] = field(default_factory=list)
    rationale: str = ""
    provenance: str = "deterministic_fallback"
    task_atoms: TaskDecomposition = field(default_factory=TaskDecomposition.empty)
    warnings: list[str] = field(default_factory=list)

    @property
    def selected_skill_ids(self) -> list[str]:
        return [item.skill_id for item in self.selected_skills]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "trace_id": self.trace_id,
            "trace_dir": str(self.trace_dir),
            "selected_skills": [item.to_dict() for item in self.selected_skills],
            "required_edges": [item.to_dict() for item in self.required_edges],
            "ordered_hints": [item.to_dict() for item in self.ordered_hints],
            "near_misses": [dict(item) for item in self.near_misses],
            "wiki_pages_read": list(self.wiki_pages_read),
            "rationale": self.rationale,
            "provenance": self.provenance,
            "task_atoms": self.task_atoms.to_dict(),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RouteResult:
        return cls(
            query=str(payload.get("query", "")),
            trace_id=str(payload.get("trace_id", "")),
            trace_dir=Path(str(payload.get("trace_dir", ""))),
            selected_skills=[
                RouteSelectedSkill.from_dict(item)
                for item in payload.get("selected_skills", [])
                if isinstance(item, dict)
            ],
            required_edges=[
                RouteEdge.from_dict(item)
                for item in payload.get("required_edges", [])
                if isinstance(item, dict)
            ],
            ordered_hints=[
                RouteEdge.from_dict(item)
                for item in payload.get("ordered_hints", [])
                if isinstance(item, dict)
            ],
            near_misses=[
                {str(key): str(value) for key, value in item.items()}
                for item in payload.get("near_misses", [])
                if isinstance(item, dict)
            ],
            wiki_pages_read=[str(item) for item in payload.get("wiki_pages_read", [])],
            rationale=str(payload.get("rationale", "")),
            provenance=str(payload.get("provenance", "deterministic_fallback")),
            task_atoms=TaskDecomposition.from_dict(
                dict(payload.get("task_atoms", {}) or {})
            ) if isinstance(payload.get("task_atoms", {}), dict) else TaskDecomposition.empty(),
            warnings=[str(item) for item in payload.get("warnings", [])],
        )


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []
