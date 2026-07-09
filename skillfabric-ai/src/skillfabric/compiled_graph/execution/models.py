"""Build-time models for the Execution Flow Layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ExecutionNodeType = Literal["artifact", "scenario"]
ExecutionEdgeType = Literal[
    "produces_artifact",
    "consumes_artifact",
    "enables_scenario",
    "requires_scenario",
    "artifact_handoff",
    "state_handoff",
]
ExecutionFlowType = Literal["artifact_handoff", "state_handoff"]
ExecutionRelationType = Literal["artifact_compatibility", "state_compatibility", "tool_handoff"]


@dataclass(slots=True)
class ExecutionEvidence:
    """Line-level evidence for execution graph construction."""

    skill: str
    line: int
    text: str

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.skill, self.line, self.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExecutionEvidence:
        return cls(
            skill=str(payload.get("skill", "")),
            line=int(payload.get("line", 0)),
            text=str(payload.get("text", "")),
        )


@dataclass(slots=True)
class ArtifactNode:
    """Artifact node in the execution layer."""

    id: str
    type: Literal["artifact"]
    name: str
    normalized_name: str
    kind: str = "artifact"
    evidence: list[ExecutionEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "normalized_name": self.normalized_name,
            "kind": self.kind,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactNode:
        return cls(
            id=str(payload.get("id", "")),
            type="artifact",
            name=str(payload.get("name", "")),
            normalized_name=str(payload.get("normalized_name", "")),
            kind=str(payload.get("kind", "artifact")),
            evidence=_evidence_from_payload(payload.get("evidence", [])),
        )


@dataclass(slots=True)
class ScenarioNode:
    """Scenario node in the execution layer."""

    id: str
    type: Literal["scenario"]
    name: str
    normalized_name: str
    kind: str = "condition"
    evidence: list[ExecutionEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "normalized_name": self.normalized_name,
            "kind": self.kind,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScenarioNode:
        return cls(
            id=str(payload.get("id", "")),
            type="scenario",
            name=str(payload.get("name", "")),
            normalized_name=str(payload.get("normalized_name", "")),
            kind=str(payload.get("kind", "condition")),
            evidence=_evidence_from_payload(payload.get("evidence", [])),
        )


@dataclass(slots=True)
class ExecutionEdge:
    """Execution-layer edge."""

    source: str
    target: str
    type: ExecutionEdgeType
    confidence: float = 1.0
    weight: float = 0.0
    evidence: list[ExecutionEvidence] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(float(self.confidence), 1.0))
        self.weight = max(0.0, float(self.weight))

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source, self.target, self.type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "confidence": self.confidence,
            "weight": self.weight,
            "evidence": [item.to_dict() for item in self.evidence],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExecutionEdge:
        return cls(
            source=str(payload.get("source", "")),
            target=str(payload.get("target", "")),
            type=payload.get("type", "artifact_handoff"),
            confidence=float(payload.get("confidence", 1.0) or 1.0),
            weight=float(payload.get("weight", 0.0) or 0.0),
            evidence=_evidence_from_payload(payload.get("evidence", [])),
            metadata={str(key): str(value) for key, value in dict(payload.get("metadata", {})).items()},
        )


@dataclass(slots=True)
class ExecutionFlowCandidate:
    """Potential artifact flow or scenario transition before validation."""

    source_skill: str
    target_skill: str
    flow_type: ExecutionFlowType
    matched_node_id: str
    matched_name: str
    evidence: list[ExecutionEvidence] = field(default_factory=list)
    prior: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.source_skill, self.target_skill, self.flow_type, self.matched_node_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_skill": self.source_skill,
            "target_skill": self.target_skill,
            "flow_type": self.flow_type,
            "matched_node_id": self.matched_node_id,
            "matched_name": self.matched_name,
            "evidence": [item.to_dict() for item in self.evidence],
            "prior": self.prior,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExecutionFlowCandidate:
        return cls(
            source_skill=str(payload.get("source_skill", "")),
            target_skill=str(payload.get("target_skill", "")),
            flow_type=payload.get("flow_type", "artifact_handoff"),
            matched_node_id=str(payload.get("matched_node_id", "")),
            matched_name=str(payload.get("matched_name", "")),
            evidence=_evidence_from_payload(payload.get("evidence", [])),
            prior=float(payload.get("prior", 1.0) or 1.0),
            metadata={str(key): str(value) for key, value in dict(payload.get("metadata", {})).items()},
        )


@dataclass(slots=True)
class ExecutionIndexRecord:
    """Canonical skill-to-skill compatibility record used as orchestration sidecar."""

    source_skill: str
    target_skill: str
    relation_type: ExecutionRelationType
    canonical_object: str
    direction: str
    confidence: float = 1.0
    evidence: list[ExecutionEvidence] = field(default_factory=list)
    projected_edge_type: str = "depend_on"
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_skill": self.source_skill,
            "target_skill": self.target_skill,
            "relation_type": self.relation_type,
            "canonical_object": self.canonical_object,
            "direction": self.direction,
            "confidence": self.confidence,
            "evidence": [item.to_dict() for item in self.evidence],
            "projected_edge_type": self.projected_edge_type,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExecutionIndexRecord:
        return cls(
            source_skill=str(payload.get("source_skill", "")),
            target_skill=str(payload.get("target_skill", "")),
            relation_type=payload.get("relation_type", "artifact_compatibility"),
            canonical_object=str(payload.get("canonical_object", "")),
            direction=str(payload.get("direction", "source_to_target")),
            confidence=float(payload.get("confidence", 1.0) or 1.0),
            evidence=_evidence_from_payload(payload.get("evidence", [])),
            projected_edge_type=str(payload.get("projected_edge_type", "depend_on")),
            metadata={str(key): str(value) for key, value in dict(payload.get("metadata", {})).items()},
        )


@dataclass(slots=True)
class ExecutionValidationRecord:
    """Auditable execution flow validation result."""

    candidate: ExecutionFlowCandidate
    raw_output: dict[str, Any]
    normalized: dict[str, Any]
    accepted: bool
    rejection_reason: str
    flow_edge: ExecutionEdge | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "raw_output": self.raw_output,
            "normalized": self.normalized,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "flow_edge": self.flow_edge.to_dict() if self.flow_edge else None,
        }


def _evidence_from_payload(payload: Any) -> list[ExecutionEvidence]:
    if not isinstance(payload, list):
        return []
    return [ExecutionEvidence.from_dict(item) for item in payload if isinstance(item, dict)]
