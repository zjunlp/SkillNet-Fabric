"""Structured explorer output aligned with the canonical route contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SkillPackageEvidence:
    path: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillPackageEvidence:
        _require_keys(payload, {"path"}, "selected skill evidence")
        return cls(path=_task_wiki_path(payload["path"], "evidence path"))


@dataclass(frozen=True, slots=True)
class SkillPackageSelectedSkill:
    skill_id: str
    role: str
    evidence: tuple[SkillPackageEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "role": self.role,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillPackageSelectedSkill:
        _require_keys(payload, {"skill_id", "role", "evidence"}, "selected skill")
        evidence = payload["evidence"]
        evidence_items = _object_list(
            evidence,
            label="selected skill evidence",
            require_item=True,
        )
        return cls(
            skill_id=_string(payload["skill_id"], "selected skill id"),
            role=_string(payload["role"], "selected skill role"),
            evidence=tuple(SkillPackageEvidence.from_dict(item) for item in evidence_items),
        )


@dataclass(frozen=True, slots=True)
class SkillPackageNearMiss:
    skill_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"skill_id": self.skill_id, "reason": self.reason}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillPackageNearMiss:
        _require_keys(payload, {"skill_id", "reason"}, "near miss")
        return cls(
            skill_id=_string(payload["skill_id"], "near miss skill_id"),
            reason=_string(payload["reason"], "near miss reason"),
        )


@dataclass(frozen=True, slots=True)
class SkillPackage:
    selected_skills: tuple[SkillPackageSelectedSkill, ...]
    near_misses: tuple[SkillPackageNearMiss, ...]
    coverage_gaps: tuple[str, ...]
    wiki_pages_read: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_skills": [item.to_dict() for item in self.selected_skills],
            "near_misses": [item.to_dict() for item in self.near_misses],
            "coverage_gaps": list(self.coverage_gaps),
            "wiki_pages_read": list(self.wiki_pages_read),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillPackage:
        expected = {
            "selected_skills",
            "near_misses",
            "coverage_gaps",
            "wiki_pages_read",
            "rationale",
        }
        _require_keys(payload, expected, "skill package")
        for key in expected - {"rationale"}:
            if not isinstance(payload[key], list):
                raise ValueError(f"skill package {key} must be a list")
        return cls(
            selected_skills=tuple(
                SkillPackageSelectedSkill.from_dict(item)
                for item in _object_list(payload["selected_skills"], label="selected_skills")
            ),
            near_misses=tuple(
                SkillPackageNearMiss.from_dict(item)
                for item in _object_list(payload["near_misses"], label="near_misses")
            ),
            coverage_gaps=tuple(_string(item, "coverage gap") for item in payload["coverage_gaps"]),
            wiki_pages_read=tuple(
                _task_wiki_path(item, "wiki page path") for item in payload["wiki_pages_read"]
            ),
            rationale=_string(payload["rationale"], "rationale"),
        )


def skill_package_json_schema() -> dict[str, Any]:
    evidence_path = {"type": "string", "minLength": 1}
    selected_skill = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "skill_id": {"type": "string", "minLength": 1},
            "role": {"type": "string", "minLength": 1},
            "evidence": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "path": evidence_path,
                    },
                    "required": ["path"],
                },
            },
        },
        "required": ["skill_id", "role", "evidence"],
    }
    near_miss = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "skill_id": {"type": "string", "minLength": 1},
            "reason": {"type": "string", "minLength": 1},
        },
        "required": ["skill_id", "reason"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_skills": {"type": "array", "items": selected_skill},
            "near_misses": {"type": "array", "items": near_miss},
            "coverage_gaps": {"type": "array", "items": {"type": "string"}},
            "wiki_pages_read": {"type": "array", "items": evidence_path},
            "rationale": {"type": "string", "minLength": 1},
        },
        "required": [
            "selected_skills",
            "near_misses",
            "coverage_gaps",
            "wiki_pages_read",
            "rationale",
        ],
    }


def _require_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} must contain exactly: {', '.join(sorted(expected))}")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _task_wiki_path(value: Any, label: str) -> str:
    path = _string(value, label).removeprefix("./").removeprefix("task_wiki/")
    if not path:
        raise ValueError(f"{label} must identify a file inside task_wiki")
    return path


def _object_list(
    value: Any,
    *,
    label: str,
    require_item: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    if require_item and not value:
        raise ValueError(f"{label} must contain at least one item")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{label}[{index}] must be an object")
    return value
