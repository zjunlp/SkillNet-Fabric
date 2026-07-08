"""Models for interface term canonicalization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol

from skillfabric.compiled_graph.interface.models import InterfaceEvidence


@dataclass(slots=True)
class RawContractObject:
    """One raw requires/produces term extracted from a skill interface."""

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

    @property
    def term_id(self) -> str:
        digest = hashlib.sha256(self.key.encode("utf-8")).hexdigest()[:16]
        return f"term:{digest}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id,
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
    """A small set of raw terms to resolve into canonical interface objects."""

    cluster_id: str
    terms: list[RawContractObject]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "terms": [item.to_dict() for item in self.terms],
        }


@dataclass(slots=True)
class CanonicalObject:
    """Accepted canonical interface object used by downstream candidate generation."""

    canonical_id: str
    name: str
    type: str
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    required_by: list[str] = field(default_factory=list)
    produced_by: list[str] = field(default_factory=list)
    confidence: float = 0.0
    provenance: str = "deterministic_exact"

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "aliases": sorted(set(self.aliases)),
            "required_by": sorted(set(self.required_by)),
            "produced_by": sorted(set(self.produced_by)),
            "confidence": self.confidence,
            "provenance": self.provenance,
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
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            provenance=str(payload.get("provenance", "deterministic_exact")),
        )


@dataclass(slots=True)
class CanonicalAssignment:
    """Assignment from a raw interface term to an accepted canonical object."""

    raw_key: str
    skill_id: str
    role: str
    raw_name: str
    raw_kind: str
    canonical_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_key": self.raw_key,
            "skill_id": self.skill_id,
            "role": self.role,
            "raw_name": self.raw_name,
            "raw_kind": self.raw_kind,
            "canonical_id": self.canonical_id,
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
        )


@dataclass(slots=True)
class CanonicalizationBuild:
    """Pool-level canonicalization result."""

    objects: list[CanonicalObject] = field(default_factory=list)
    assignments: list[CanonicalAssignment] = field(default_factory=list)
    raw_terms: list[RawContractObject] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_id: str = "deterministic-canonicalization"

    def lookup(self, skill_id: str, role: str, raw_name: str, raw_kind: str) -> str:
        key = "|".join([skill_id, role, raw_name.lower(), raw_kind.lower()])
        for assignment in self.assignments:
            if assignment.raw_key == key:
                return assignment.canonical_id
        return ""


class CanonicalizationProvider(Protocol):
    """Provider protocol for resolving unresolved candidate groups."""

    model_id: str

    def canonicalize(self, cluster: CanonicalizationCluster) -> dict[str, Any]:
        """Return canonicalization JSON for one candidate group."""
