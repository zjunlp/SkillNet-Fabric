"""Models for the Interface Semantics Layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
    inferred: bool = False
    evidence: list[InterfaceEvidence] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.kind = normalize_interface_kind(self.kind, name=self.name)
        self.confidence = max(0.0, min(float(self.confidence), 1.0))
        if not self.evidence:
            self.inferred = True

    @property
    def key(self) -> tuple[str, str]:
        return (_normalize_token(self.name), self.kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "confidence": self.confidence,
            "inferred": self.inferred,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InterfaceField:
        return cls(
            name=str(payload.get("name", "")),
            description=str(payload.get("description", "")),
            kind=str(payload.get("kind", "text")),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            inferred=bool(payload.get("inferred", False)),
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
    granularity: str = "utility"
    execution_role: str = "helper"
    requires: list[InterfaceField] = field(default_factory=list)
    produces: list[InterfaceField] = field(default_factory=list)
    uses_tools: list[InterfaceField] = field(default_factory=list)
    failure_modes: list[InterfaceField] = field(default_factory=list)
    evidence: list[InterfaceEvidence] = field(default_factory=list)
    provenance: str = "deterministic_fallback"
    model_id: str = "deterministic-interface"

    def __post_init__(self) -> None:
        self.granularity = normalize_granularity(self.granularity)
        self.execution_role = normalize_execution_role(self.execution_role, granularity=self.granularity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "content_hash": self.content_hash,
            "capability_summary": self.capability_summary,
            "when_to_use": self.when_to_use,
            "granularity": self.granularity,
            "execution_role": self.execution_role,
            "requires": _fields_to_dict(self.requires),
            "produces": _fields_to_dict(self.produces),
            "uses_tools": _fields_to_dict(self.uses_tools),
            "failure_modes": _fields_to_dict(self.failure_modes),
            "evidence": [item.to_dict() for item in self.evidence],
            "provenance": self.provenance,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillInterface:
        return cls(
            skill_id=str(payload.get("skill_id", "")),
            content_hash=str(payload.get("content_hash", "")),
            capability_summary=str(payload.get("capability_summary", "")),
            when_to_use=str(payload.get("when_to_use", "")),
            granularity=str(payload.get("granularity", "utility")),
            execution_role=str(payload.get("execution_role", "helper")),
            requires=_fields_from_payload(payload.get("requires", [])),
            produces=_fields_from_payload(payload.get("produces", [])),
            uses_tools=_fields_from_payload(payload.get("uses_tools", [])),
            failure_modes=_fields_from_payload(payload.get("failure_modes", [])),
            evidence=[
                InterfaceEvidence.from_dict(item)
                for item in payload.get("evidence", [])
                if isinstance(item, dict)
            ],
            provenance=str(payload.get("provenance", "deterministic_fallback")),
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


def normalize_granularity(value: str) -> str:
    normalized = value.lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "macro": "macro",
        "macro_skill": "macro",
        "workflow": "macro",
        "multi_step": "macro",
        "end_to_end": "macro",
        "primitive": "primitive",
        "primitive_skill": "primitive",
        "action": "primitive",
        "single_action": "primitive",
        "atomic": "primitive",
        "utility": "utility",
        "helper": "utility",
        "support": "utility",
        "inspection": "utility",
        "planning": "planning",
        "planner": "planning",
        "plan": "planning",
        "routing": "planning",
        "goal_parser": "planning",
    }
    return aliases.get(normalized, "utility")


def normalize_execution_role(value: str, *, granularity: str = "utility") -> str:
    normalized = value.lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "actor": "actor",
        "act": "actor",
        "action": "actor",
        "take": "actor",
        "pickup": "actor",
        "picker": "actor",
        "navigator": "navigator",
        "navigation": "navigator",
        "move": "navigator",
        "inspector": "inspector",
        "inspect": "inspector",
        "scanner": "inspector",
        "searcher": "inspector",
        "locator": "inspector",
        "transformer": "transformer",
        "transform": "transformer",
        "modifier": "transformer",
        "cleaner": "transformer",
        "heater": "transformer",
        "cooler": "transformer",
        "verifier": "verifier",
        "verify": "verifier",
        "validator": "verifier",
        "planner": "planner",
        "planning": "planner",
        "parser": "planner",
        "helper": "helper",
        "utility": "helper",
    }
    resolved = aliases.get(normalized)
    if resolved:
        return resolved
    defaults = {
        "planning": "planner",
        "primitive": "actor",
        "macro": "actor",
        "utility": "helper",
    }
    return defaults.get(granularity, "helper")


def normalize_interface_kind(kind: str, *, name: str = "") -> str:
    normalized_kind = kind.lower().strip().replace("-", "_").replace(" ", "_")
    normalized_name = _normalize_token(name)
    if _looks_like_belief_state_name(normalized_name):
        return "belief_state"
    if _looks_like_planning_state_name(normalized_name):
        return "planning_state"
    if _looks_like_world_state_name(normalized_name):
        return "world_state"
    aliases = {
        "state": "world_state",
        "condition": "world_state",
        "physical_state": "world_state",
        "environment_state": "world_state",
        "worldstate": "world_state",
        "belief": "belief_state",
        "memory_state": "belief_state",
        "knowledge_state": "belief_state",
        "observation_state": "belief_state",
        "planning": "planning_state",
        "plan_state": "planning_state",
        "routing_state": "planning_state",
    }
    return aliases.get(normalized_kind, normalized_kind or "data")


def _looks_like_belief_state_name(normalized_name: str) -> bool:
    if not normalized_name:
        return False
    tokens = set(normalized_name.split())
    return bool(
        {
            "permanence",
            "remember",
            "remembered",
            "observed",
            "observation",
            "belief",
            "memory",
            "knowledge",
            "inferred",
            "inference",
        }
        & tokens
    ) or "object permanence" in normalized_name


def _looks_like_planning_state_name(normalized_name: str) -> bool:
    if not normalized_name:
        return False
    tokens = set(normalized_name.split())
    return bool(
        {
            "parsed",
            "plan",
            "planning",
            "routing",
            "objective",
            "subobjective",
            "goal",
        }
        & tokens
    ) or any(
        phrase in normalized_name
        for phrase in (
            "sequential sub objective",
            "structured task parse",
            "goal parse",
            "task parse",
            "sub objective",
        )
    )


def _looks_like_world_state_name(normalized_name: str) -> bool:
    if not normalized_name:
        return False
    tokens = set(normalized_name.split())
    if {"inventory", "held", "holding", "open", "closed", "clean", "cleaned", "heated", "cooled"} & tokens:
        return True
    if "located" in tokens or "accessible" in tokens or "authenticated" in tokens:
        return True
    return any(
        phrase in normalized_name
        for phrase in (
            "in hand",
            "in inventory",
            "object inventory",
            "object hand",
            "receptacle open",
            "receptacle closed",
        )
    )
