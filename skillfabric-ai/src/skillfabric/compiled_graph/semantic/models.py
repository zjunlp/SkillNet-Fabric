"""Canonical records for semantic candidate retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from skillfabric.compiled_graph.models import Edge, EvidenceRef

CandidateChannel = Literal["handoff", "explicit_reference", "similarity", "lexical"]
EmbeddingKind = Literal["skill", "requires", "produces"]
RelationType = Literal["depend_on", "compose_with", "similar_to", "none"]
_CHANNEL_ORDER = {"handoff": 0, "explicit_reference": 1, "similarity": 2, "lexical": 3}


@dataclass(frozen=True, slots=True)
class CandidateHit:
    """One retrieval observation that justifies reviewing a skill pair."""

    channel: CandidateChannel
    query_skill: str
    matched_skill: str
    rank: int
    query_field: str = ""
    matched_field: str = ""
    evidence: tuple[EvidenceRef, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "query_skill": self.query_skill,
            "matched_skill": self.matched_skill,
            "rank": self.rank,
            "query_field": self.query_field,
            "matched_field": self.matched_field,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class CandidatePair:
    """One unordered skill pair awaiting exactly one semantic decision."""

    skill_a: str
    skill_b: str
    hits: tuple[CandidateHit, ...]

    def __post_init__(self) -> None:
        if not self.skill_a or not self.skill_b or self.skill_a >= self.skill_b:
            raise ValueError("candidate endpoints must be distinct and in canonical order")
        if not self.hits:
            raise ValueError("candidate pair must contain retrieval evidence")

    @property
    def key(self) -> tuple[str, str]:
        return (self.skill_a, self.skill_b)

    @property
    def channels(self) -> tuple[CandidateChannel, ...]:
        return tuple(
            sorted(
                {hit.channel for hit in self.hits},
                key=lambda channel: _CHANNEL_ORDER[channel],
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_a": self.skill_a,
            "skill_b": self.skill_b,
            "hits": [hit.to_dict() for hit in self.hits],
        }


@dataclass(frozen=True, slots=True)
class EmbeddingRecord:
    """One cached vector for a contract document or contract field."""

    key: str
    skill_id: str
    kind: EmbeddingKind
    field_name: str
    text_hash: str
    vector: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "skill_id": self.skill_id,
            "kind": self.kind,
            "field_name": self.field_name,
            "text_hash": self.text_hash,
            "vector": list(self.vector),
        }


@dataclass(frozen=True, slots=True)
class CandidateRetrievalResult:
    """Bounded candidate pairs plus reusable embedding records."""

    pairs: tuple[CandidatePair, ...]
    metrics: dict[str, int | float | str]


@dataclass(frozen=True, slots=True)
class RelationDecision:
    """One validated final semantic decision for a candidate pair."""

    candidate: CandidatePair
    relation: RelationType
    source_skill: str
    target_skill: str
    confidence: float
    reason: str
    evidence: tuple[EvidenceRef, ...]
    cache_hit: bool = False

    def judge_dict(self) -> dict[str, Any]:
        """Return the compact validated payload persisted in the decision cache."""

        return {
            "relation": self.relation,
            "source_skill": self.source_skill,
            "target_skill": self.target_skill,
            "confidence": round(float(self.confidence), 6),
            "reason": self.reason,
            "evidence": [{"skill": item.skill, "line": item.line} for item in self.evidence],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "relation": self.relation,
            "source_skill": self.source_skill,
            "target_skill": self.target_skill,
            "confidence": round(float(self.confidence), 6),
            "reason": self.reason,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class GraphProjectionResult:
    """Projected semantic graph edges and any cycle-reviewed decisions."""

    edges: tuple[Edge, ...]
    decisions: tuple[RelationDecision, ...]
    cycle_review_count: int = 0
