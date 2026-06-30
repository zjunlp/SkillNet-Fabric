"""Data models for page-level wiki exploration traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class WikiPageEntry:
    """One searchable wiki page."""

    page_id: str
    path: str
    page_type: str
    entity_id: str
    title: str
    summary: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "path": self.path,
            "page_type": self.page_type,
            "entity_id": self.entity_id,
            "title": self.title,
            "summary": self.summary,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WikiPageEntry:
        return cls(
            page_id=str(payload.get("page_id", "")),
            path=str(payload.get("path", "")),
            page_type=str(payload.get("page_type", "")),
            entity_id=str(payload.get("entity_id", "")),
            title=str(payload.get("title", "")),
            summary=str(payload.get("summary", "")),
            text=str(payload.get("text", "")),
        )
