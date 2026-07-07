"""Models for pool-level contract object canonicalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from skillfabric.compiled_graph.interface.models import InterfaceEvidence


@dataclass(slots=True)
class RawContractObject:
    """One raw requires/produces term extracted from a skill contract."""

    skill_id: str
    role: str
    name: str
    kind: str
    description: str = ""
    confidence: float = 0.0
    evidence: list[InterfaceEvidence] = field(default_factory=list)

    @property
    def key(self) -> str:
        return "|".join([self.skill_id, self.role, self.name.lower(), self.kind.lower()])

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "role": self.role,
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "confidence": self.confidence,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(slots=True)
class CanonicalizationCluster:
    """Small pre-cluster sent to a canonicalization provider."""

    cluster_id: str
    object_type: str
    terms: list[RawContractObject]
    candidate_edges: list[dict[str, Any]] = field(default_factory=list)
    ambiguous: bool = False
    methods_present: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "object_type": self.object_type,
            "terms": [item.to_dict() for item in self.terms],
            "candidate_edges": list(self.candidate_edges),
            "ambiguous": self.ambiguous,
            "methods_present": list(self.methods_present),
        }


@dataclass(slots=True)
class CanonicalObject:
    """Canonical object used by execution compatibility."""

    canonical_id: str
    name: str
    type: str
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    required_by: list[str] = field(default_factory=list)
    produced_by: list[str] = field(default_factory=list)
    reuse_count: int = 0
    promoted: bool = False
    confidence: float = 0.0
    provenance: str = "deterministic_fallback"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "aliases": sorted(set(self.aliases)),
            "required_by": sorted(set(self.required_by)),
            "produced_by": sorted(set(self.produced_by)),
            "reuse_count": self.reuse_count,
            "promoted": self.promoted,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CanonicalObject:
        return cls(
            canonical_id=str(payload.get("canonical_id", "")),
            name=str(payload.get("name", "")),
            type=str(payload.get("type", "artifact")),
            description=str(payload.get("description", "")),
            aliases=[str(item) for item in payload.get("aliases", [])],
            required_by=[str(item) for item in payload.get("required_by", [])],
            produced_by=[str(item) for item in payload.get("produced_by", [])],
            reuse_count=int(payload.get("reuse_count", 0) or 0),
            promoted=bool(payload.get("promoted", False)),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            provenance=str(payload.get("provenance", "deterministic_fallback")),
            reason=str(payload.get("reason", "")),
        )


@dataclass(slots=True)
class CanonicalAssignment:
    """Assignment from a raw contract object to a canonical object."""

    raw_key: str
    skill_id: str
    role: str
    raw_name: str
    raw_kind: str
    canonical_id: str
    confidence: float = 0.0
    reason: str = ""
    provenance: str = "deterministic_fallback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_key": self.raw_key,
            "skill_id": self.skill_id,
            "role": self.role,
            "raw_name": self.raw_name,
            "raw_kind": self.raw_kind,
            "canonical_id": self.canonical_id,
            "confidence": self.confidence,
            "reason": self.reason,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CanonicalAssignment:
        return cls(
            raw_key=str(payload.get("raw_key", "")),
            skill_id=str(payload.get("skill_id", "")),
            role=str(payload.get("role", "")),
            raw_name=str(payload.get("raw_name", "")),
            raw_kind=str(payload.get("raw_kind", "")),
            canonical_id=str(payload.get("canonical_id", "")),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            reason=str(payload.get("reason", "")),
            provenance=str(payload.get("provenance", "deterministic_fallback")),
        )


@dataclass(slots=True)
class CanonicalizationBuild:
    """Pool-level canonicalization result."""

    objects: list[CanonicalObject] = field(default_factory=list)
    assignments: list[CanonicalAssignment] = field(default_factory=list)
    raw_terms: list[RawContractObject] = field(default_factory=list)
    candidate_edges: list[Any] = field(default_factory=list)
    candidate_components: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_id: str = "deterministic-canonicalization"

    def lookup(self, skill_id: str, role: str, raw_name: str, raw_kind: str) -> str:
        key = "|".join([skill_id, role, raw_name.lower(), raw_kind.lower()])
        assignment = {item.raw_key: item for item in self.assignments}.get(key)
        if assignment is None:
            return ""
        objects = {item.canonical_id: item for item in self.objects}
        canonical = objects.get(assignment.canonical_id)
        if canonical is None or not canonical.promoted:
            return ""
        return assignment.canonical_id


class CanonicalizationProvider(Protocol):
    """Provider protocol for canonicalizing pre-clustered raw terms."""

    model_id: str

    def canonicalize(self, cluster: CanonicalizationCluster) -> dict[str, Any]:
        """Return canonicalization JSON for one cluster."""
