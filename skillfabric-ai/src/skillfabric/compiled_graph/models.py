"""Canonical KG schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from skillfabric.registry.models import SkillNode

NodeType = Literal["skill"]
EdgeType = Literal["similar_to", "compose_with", "depend_on"]
_EDGE_TYPES = {"similar_to", "compose_with", "depend_on"}
_NODE_TYPES = {"skill"}


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
        if "type" not in payload:
            raise ValueError("missing edge type")
        edge_type = str(payload["type"])
        if edge_type not in _EDGE_TYPES:
            raise ValueError(f"unsupported edge type: {edge_type}")
        return cls(
            source=str(payload.get("source", "")),
            target=str(payload.get("target", "")),
            type=edge_type,
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
    nodes: list[SkillNode]
    edges: list[Edge]
    stats: dict[str, Any]
    config_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "build_id": self.build_id,
            "nodes": [node.to_dict(include_raw_text=False) for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "stats": self.stats,
            "config_digest": self.config_digest,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GraphDocument:
        nodes: list[SkillNode] = []
        for item in payload.get("nodes", []):
            if not isinstance(item, dict):
                raise ValueError("graph nodes must be JSON objects")
            node_type = str(item.get("type", ""))
            if node_type not in _NODE_TYPES:
                raise ValueError(f"unsupported node type: {node_type}")
            nodes.append(SkillNode.from_dict(item))
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
