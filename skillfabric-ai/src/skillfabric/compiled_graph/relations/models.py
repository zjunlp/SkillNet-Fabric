"""Build-time models for relation construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from skillfabric.compiled_graph.models import Edge

DirectionHint = str


@dataclass(slots=True)
class RelationEvidence:
    """Evidence explaining why a pair became a relation candidate."""

    source: str
    skill_id: str
    line: int
    text: str
    kind: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, int, str, str]:
        return (self.source, self.skill_id, self.line, self.text, self.kind)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RelationEvidence:
        return cls(
            source=str(payload.get("source", "")),
            skill_id=str(payload.get("skill_id", "")),
            line=int(payload.get("line", 0)),
            text=str(payload.get("text", "")),
            kind=str(payload.get("kind", "")),
            metadata={str(k): str(v) for k, v in dict(payload.get("metadata", {})).items()},
        )


@dataclass(slots=True)
class SkillMention:
    """Explicit mention of one skill inside another skill document."""

    from_skill: str
    to_skill: str
    line: int
    text: str
    mention_type: str
    direction_hint: DirectionHint = "none"

    def to_evidence(self) -> RelationEvidence:
        return RelationEvidence(
            source="explicit_mention",
            skill_id=self.from_skill,
            line=self.line,
            text=self.text,
            kind="mention",
            metadata={
                "to_skill": self.to_skill,
                "mention_type": self.mention_type,
                "direction_hint": self.direction_hint,
            },
        )


@dataclass(slots=True)
class CandidatePair:
    """Potential compose_with or depend_on relation."""

    skill_a: str
    skill_b: str
    prior: float
    sources: list[str] = field(default_factory=list)
    evidence: list[RelationEvidence] = field(default_factory=list)
    direction_hint: DirectionHint = "none"

    def __post_init__(self) -> None:
        if self.skill_b < self.skill_a:
            self.skill_a, self.skill_b = self.skill_b, self.skill_a
            self.direction_hint = _flip_direction(self.direction_hint)

    @property
    def key(self) -> tuple[str, str]:
        return (self.skill_a, self.skill_b)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "skill_a": self.skill_a,
            "skill_b": self.skill_b,
            "prior": self.prior,
            "sources": list(self.sources),
            "direction_hint": self.direction_hint,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(slots=True)
class ValidationRecord:
    """Auditable pairwise validation result."""

    pair: CandidatePair
    raw_output: dict[str, Any]
    normalized: dict[str, Any]
    accepted: bool
    rejection_reason: str
    edge: Edge | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "skill_a": self.pair.skill_a,
            "skill_b": self.pair.skill_b,
            "candidate_sources": list(self.pair.sources),
            "candidate_prior": self.pair.prior,
            "direction_hint": self.pair.direction_hint,
            "candidate_evidence": [item.to_dict() for item in self.pair.evidence],
            "raw_output": self.raw_output,
            "normalized": self.normalized,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "edge": self.edge.to_dict() if self.edge else None,
        }


def _flip_direction(direction: str) -> str:
    if direction == "A->B":
        return "B->A"
    if direction == "B->A":
        return "A->B"
    return direction
