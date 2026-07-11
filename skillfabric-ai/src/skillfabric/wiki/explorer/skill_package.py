"""Structured route-time explorer output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SkillPackageEvidence:
    """Evidence path for one selected skill."""

    path: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "reason": self.reason}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillPackageEvidence:
        return cls(path=str(payload.get("path", "")), reason=str(payload.get("reason", "")))


@dataclass(slots=True)
class SkillPackageSelectedSkill:
    """One selected skill from the query wiki."""

    skill_id: str
    role: str
    evidence: list[SkillPackageEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "role": self.role,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillPackageSelectedSkill:
        return cls(
            skill_id=str(payload.get("skill_id", "")),
            role=str(payload.get("role", payload.get("reason", ""))),
            evidence=[
                SkillPackageEvidence.from_dict(item)
                for item in payload.get("evidence", [])
                if isinstance(item, dict)
            ],
        )


@dataclass(slots=True)
class SkillPackageRequiredEdge:
    """Ordering edge selected by query-wiki exploration."""

    before: str
    after: str
    relation_type: str = "depend_on"
    evidence_path: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": self.before,
            "after": self.after,
            "relation_type": self.relation_type,
            "evidence_path": self.evidence_path,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillPackageRequiredEdge:
        return cls(
            before=str(payload.get("before", payload.get("before_skill", ""))),
            after=str(payload.get("after", payload.get("after_skill", ""))),
            relation_type=str(payload.get("relation_type", payload.get("edge_type", "depend_on"))),
            evidence_path=str(payload.get("evidence_path", "")),
            reason=str(payload.get("reason", "")),
        )


@dataclass(slots=True)
class SkillPackageOrderedHint:
    """Coarse ordering hint."""

    skill_id: str
    hint: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"skill_id": self.skill_id, "hint": self.hint}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillPackageOrderedHint:
        return cls(skill_id=str(payload.get("skill_id", "")), hint=str(payload.get("hint", "")))


@dataclass(slots=True)
class SkillPackageNearMiss:
    """Rejected but plausible skill."""

    skill_id: str
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"skill_id": self.skill_id, "reason": self.reason}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillPackageNearMiss:
        return cls(skill_id=str(payload.get("skill_id", "")), reason=str(payload.get("reason", "")))


@dataclass(slots=True)
class SkillPackageCoverageNote:
    """Coverage note returned by the explorer."""

    requirement_id: str
    status: str
    skill_ids: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "status": self.status,
            "skill_ids": list(self.skill_ids),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillPackageCoverageNote:
        raw_skill_ids = payload.get("skill_ids", [])
        if isinstance(raw_skill_ids, str):
            skill_ids = [raw_skill_ids]
        elif isinstance(raw_skill_ids, list):
            skill_ids = [str(item) for item in raw_skill_ids if str(item)]
        else:
            skill_ids = []
        return cls(
            requirement_id=str(payload.get("requirement_id", payload.get("id", ""))),
            status=str(payload.get("status", "")),
            skill_ids=skill_ids,
            reason=str(payload.get("reason", "")),
        )


@dataclass(slots=True)
class SkillPackage:
    """Explorer-selected skill package."""

    selected_skills: list[SkillPackageSelectedSkill] = field(default_factory=list)
    required_edges: list[SkillPackageRequiredEdge] = field(default_factory=list)
    ordered_hints: list[SkillPackageOrderedHint] = field(default_factory=list)
    near_misses: list[SkillPackageNearMiss] = field(default_factory=list)
    coverage_notes: list[SkillPackageCoverageNote] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_skills": [item.to_dict() for item in self.selected_skills],
            "required_edges": [item.to_dict() for item in self.required_edges],
            "ordered_hints": [item.to_dict() for item in self.ordered_hints],
            "near_misses": [item.to_dict() for item in self.near_misses],
            "coverage_notes": [item.to_dict() for item in self.coverage_notes],
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillPackage:
        return cls(
            selected_skills=[
                SkillPackageSelectedSkill.from_dict(item)
                for item in payload.get("selected_skills", [])
                if isinstance(item, dict)
            ],
            required_edges=[
                SkillPackageRequiredEdge.from_dict(item)
                for item in payload.get("required_edges", [])
                if isinstance(item, dict)
            ],
            ordered_hints=[
                SkillPackageOrderedHint.from_dict(item)
                for item in payload.get("ordered_hints", [])
                if isinstance(item, dict)
            ],
            near_misses=[
                SkillPackageNearMiss.from_dict(item)
                for item in payload.get("near_misses", [])
                if isinstance(item, dict)
            ],
            coverage_notes=[
                SkillPackageCoverageNote.from_dict(item)
                for item in payload.get("coverage_notes", [])
                if isinstance(item, dict)
            ],
            rationale=str(payload.get("rationale", "")),
        )


def skill_package_json_schema() -> dict[str, Any]:
    """Return the structured output schema for query-wiki explorers."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_skills": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "skill_id": {
                            "type": "string",
                            "description": "Manifest skill_id selected from query_wiki.",
                        },
                        "role": {
                            "type": "string",
                            "description": "Short evidence-grounded reason for selecting this skill.",
                        },
                        "evidence": {
                            "type": "array",
                            "description": "Files under query_wiki that justify selecting this skill.",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "path": {
                                        "type": "string",
                                        "description": "Relative query_wiki evidence path.",
                                    },
                                    "reason": {
                                        "type": "string",
                                        "description": "Why this file supports the selection.",
                                    },
                                },
                                "required": ["path", "reason"],
                            },
                        },
                    },
                    "required": ["skill_id", "role", "evidence"],
                },
            },
            "required_edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "before": {
                            "type": "string",
                            "description": "Skill id that must run before the after skill.",
                        },
                        "after": {
                            "type": "string",
                            "description": "Skill id that consumes context, artifacts, or state.",
                        },
                        "relation_type": {
                            "type": "string",
                            "enum": ["depend_on", "compose_with", "artifact_compatibility", "state_compatibility"],
                            "description": "Dependency type for the before -> after edge.",
                        },
                        "evidence_path": {
                            "type": "string",
                            "description": "Relative query_wiki evidence path for the edge.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why the edge direction is before -> after.",
                        },
                    },
                    "required": ["before", "after", "relation_type", "evidence_path", "reason"],
                },
            },
            "ordered_hints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"skill_id": {"type": "string"}, "hint": {"type": "string"}},
                    "required": ["skill_id", "hint"],
                },
            },
            "near_misses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"skill_id": {"type": "string"}, "reason": {"type": "string"}},
                    "required": ["skill_id", "reason"],
                },
            },
            "coverage_notes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "requirement_id": {"type": "string"},
                        "status": {"type": "string"},
                        "reason": {"type": "string"},
                        "skill_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["requirement_id", "status", "reason", "skill_ids"],
                },
            },
            "rationale": {"type": "string"},
        },
        "required": ["selected_skills", "required_edges", "ordered_hints", "near_misses", "coverage_notes", "rationale"],
    }


def validate_skill_package_payload(payload: Any) -> list[str]:
    """Validate raw explorer JSON before tolerant model conversion."""

    errors: list[str] = []
    _validate_schema_value(payload, skill_package_json_schema(), path="$", errors=errors)
    return errors


def _validate_schema_value(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str,
    errors: list[str],
) -> None:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            errors.append(f"{path}: expected object")
            return
        properties = dict(schema.get("properties", {}))
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required property is missing")
        if schema.get("additionalProperties") is False:
            for key in sorted(set(value) - set(properties)):
                errors.append(f"{path}.{key}: additional property is not allowed")
        for key, child_schema in properties.items():
            if key in value:
                _validate_schema_value(
                    value[key],
                    child_schema,
                    path=f"{path}.{key}",
                    errors=errors,
                )
        return
    if expected_type == "array":
        if not isinstance(value, list):
            errors.append(f"{path}: expected array")
            return
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_schema_value(
                    item,
                    item_schema,
                    path=f"{path}[{index}]",
                    errors=errors,
                )
        return
    if expected_type == "string" and not isinstance(value, str):
        errors.append(f"{path}: expected string")
        return
    allowed = schema.get("enum")
    if isinstance(allowed, list) and value not in allowed:
        errors.append(f"{path}: unsupported value {value!r}; expected one of {allowed}")
