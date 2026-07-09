"""Models for the Interface Semantics Layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

INTERFACE_FIELD_KINDS = frozenset(
    {
        "artifact",
        "data",
        "text",
        "world_state",
        "belief_state",
        "planning_state",
        "credential",
        "environment",
        "report",
        "tool",
    }
)


@dataclass(slots=True)
class InterfaceEvidence:
    """Line-level evidence for an interface field."""

    skill: str
    line: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InterfaceEvidence:
        return cls(
            skill=str(payload.get("skill", "")),
            line=int(payload.get("line", 0)),
            text=str(payload.get("text", "")),
        )


@dataclass(slots=True)
class InterfaceField:
    """Structured field in a skill interface."""

    name: str
    description: str = ""
    kind: str = "text"
    confidence: float = 0.0
    evidence: list[InterfaceEvidence] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.kind = normalize_interface_kind(self.kind)
        if self.kind not in INTERFACE_FIELD_KINDS:
            raise ValueError(f"unsupported interface field kind: {self.kind}")
        self.confidence = max(0.0, min(float(self.confidence), 1.0))

    @property
    def key(self) -> tuple[str, str]:
        return (_normalize_token(self.name), self.kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "confidence": self.confidence,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InterfaceField:
        return cls(
            name=str(payload.get("name", "")),
            description=str(payload.get("description", "")),
            kind=str(payload.get("kind", "text")),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            evidence=[
                InterfaceEvidence.from_dict(item)
                for item in payload.get("evidence", [])
                if isinstance(item, dict)
            ],
        )


@dataclass(slots=True)
class SkillInterface:
    """Compiled interface for one skill."""

    skill_id: str
    content_hash: str
    capability_summary: str
    when_to_use: str = ""
    requires: list[InterfaceField] = field(default_factory=list)
    produces: list[InterfaceField] = field(default_factory=list)
    uses_tools: list[InterfaceField] = field(default_factory=list)
    evidence: list[InterfaceEvidence] = field(default_factory=list)
    model_id: str = "deterministic-interface"

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "content_hash": self.content_hash,
            "capability_summary": self.capability_summary,
            "when_to_use": self.when_to_use,
            "requires": _fields_to_dict(self.requires),
            "produces": _fields_to_dict(self.produces),
            "uses_tools": _fields_to_dict(self.uses_tools),
            "evidence": [item.to_dict() for item in self.evidence],
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillInterface:
        return cls(
            skill_id=str(payload.get("skill_id", "")),
            content_hash=str(payload.get("content_hash", "")),
            capability_summary=str(payload.get("capability_summary", "")),
            when_to_use=str(payload.get("when_to_use", "")),
            requires=_fields_from_payload(payload.get("requires", [])),
            produces=_fields_from_payload(payload.get("produces", [])),
            uses_tools=_fields_from_payload(payload.get("uses_tools", [])),
            evidence=[
                InterfaceEvidence.from_dict(item)
                for item in payload.get("evidence", [])
                if isinstance(item, dict)
            ],
            model_id=str(payload.get("model_id", "deterministic-interface")),
        )


@dataclass(slots=True)
class InterfaceExtractionRecord:
    """Auditable interface extraction result."""

    skill_id: str
    raw_output: dict[str, Any]
    interface: SkillInterface
    accepted: bool
    rejection_reason: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "raw_output": self.raw_output,
            "interface": self.interface.to_dict(),
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


def _fields_to_dict(fields: list[InterfaceField]) -> list[dict[str, Any]]:
    return [item.to_dict() for item in fields]


def _fields_from_payload(payload: Any) -> list[InterfaceField]:
    if not isinstance(payload, list):
        return []
    return [InterfaceField.from_dict(item) for item in payload if isinstance(item, dict)]


def _normalize_token(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def normalize_interface_kind(kind: str) -> str:
    normalized_kind = kind.lower().strip().replace("-", "_").replace(" ", "_")
    return normalized_kind
