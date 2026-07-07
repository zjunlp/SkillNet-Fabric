"""Skill registry data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class SkillNode:
    """Skill node in the canonical KG."""

    id: str
    type: Literal["skill"]
    name: str
    description: str
    content_hash: str
    raw_text: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self, *, include_raw_text: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "content_hash": self.content_hash,
        }
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        if include_raw_text:
            payload["raw_text"] = self.raw_text
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillNode:
        return cls(
            id=str(payload["id"]),
            type="skill",
            name=str(payload.get("name", "")),
            description=str(payload.get("description", "")),
            content_hash=str(payload.get("content_hash", "")),
            raw_text=str(payload.get("raw_text", "")),
            warnings=[str(v) for v in payload.get("warnings", [])],
        )
