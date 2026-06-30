"""Canonical KG schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from skillfabric.registry.models import SkillNode

NodeType = Literal["skill", "community"]
EdgeType = Literal["similar_to", "member_of", "compose_with", "depend_on"]


@dataclass(slots=True)
class CommunityNode:
    """Community node in the canonical KG."""

    id: str
    type: Literal["community"]
    name: str
    summary: str
    member_count: int
    representative_skill_ids: list[str] = field(default_factory=list)
    cohesion_score: float = 0.0
    task_patterns: list[str] = field(default_factory=list)
    summary_provenance: str = "deterministic_fallback"
    model_id: str = "deterministic-community"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CommunityNode:
        return cls(
            id=str(payload["id"]),
            type="community",
            name=str(payload.get("name", "")),
            summary=str(payload.get("summary", "")),
            member_count=int(payload.get("member_count", 0)),
            representative_skill_ids=[
                str(item) for item in payload.get("representative_skill_ids", [])
            ],
            cohesion_score=float(payload.get("cohesion_score", 0.0)),
            task_patterns=[str(item) for item in payload.get("task_patterns", [])],
            summary_provenance=str(payload.get("summary_provenance", "deterministic_fallback")),
            model_id=str(payload.get("model_id", "deterministic-community")),
        )


@dataclass(slots=True)
class EvidenceRef:
    """Line-level evidence reference for a canonical edge."""

    skill: str
    line: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvidenceRef:
        return cls(
            skill=str(payload.get("skill", "")),
            line=int(payload.get("line", 0)),
            text=str(payload.get("text", "")),
        )


@dataclass(slots=True)
class Edge:
    """Canonical KG edge."""

    source: str
    target: str
    type: EdgeType
    confidence: float
    weight: float = 0.0
    provenance: str = "computed"
    evidence: list[EvidenceRef] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "confidence": self.confidence,
            "weight": self.weight,
            "provenance": self.provenance,
            "evidence": [item.to_dict() for item in self.evidence],
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Edge:
        return cls(
            source=str(payload.get("source", "")),
            target=str(payload.get("target", "")),
            type=payload.get("type", "similar_to"),
            confidence=float(payload.get("confidence", 0.0)),
            weight=float(payload.get("weight", 0.0)),
            provenance=str(payload.get("provenance", "computed")),
            evidence=[
                EvidenceRef.from_dict(item)
                for item in payload.get("evidence", [])
                if isinstance(item, dict)
            ],
            reason=str(payload.get("reason", "")),
        )


@dataclass(slots=True)
class GraphDocument:
    """Root object serialized to graph.json."""

    schema_version: str
    build_id: str
    nodes: list[SkillNode | CommunityNode]
    edges: list[Edge]
    stats: dict[str, Any]
    config_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "build_id": self.build_id,
            "nodes": [
                node.to_dict(include_raw_text=False)
                if isinstance(node, SkillNode)
                else node.to_dict()
                for node in self.nodes
            ],
            "edges": [edge.to_dict() for edge in self.edges],
            "stats": self.stats,
            "config_digest": self.config_digest,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GraphDocument:
        nodes: list[SkillNode | CommunityNode] = []
        for item in payload.get("nodes", []):
            if item.get("type") == "skill":
                nodes.append(SkillNode.from_dict(item))
            elif item.get("type") == "community":
                nodes.append(CommunityNode.from_dict(item))
        return cls(
            schema_version=str(payload.get("schema_version", "1.0")),
            build_id=str(payload.get("build_id", "")),
            nodes=nodes,
            edges=[
                Edge.from_dict(item) for item in payload.get("edges", []) if isinstance(item, dict)
            ],
            stats=dict(payload.get("stats", {})),
            config_digest=str(payload.get("config_digest", "")),
        )
