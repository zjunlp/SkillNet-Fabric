"""Shared wiki page formatting helpers."""

from __future__ import annotations

import re


def slug(value: str) -> str:
    """Return a stable markdown file slug."""

    cleaned = value
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[1]
    return re.sub(r"[^a-zA-Z0-9]+", "-", cleaned.lower()).strip("-") or "unnamed"


def wiki_link(category: str, entity_id: str, label: str | None = None) -> str:
    """Return an Obsidian-style wikilink."""

    target = f"{category}/{slug(entity_id)}"
    if label and label != slug(entity_id):
        return f"[[{target}|{label}]]"
    return f"[[{target}]]"


def bullet_list(values: list[str], *, empty: str = "- None") -> str:
    """Render markdown bullets."""

    if not values:
        return empty
    return "\n".join(f"- {value}" for value in values)


def frontmatter(fields: dict[str, object]) -> str:
    """Render simple YAML frontmatter."""

    lines = ["---"]
    for key, value in fields.items():
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value)
            lines.append(f"{key}: [{rendered}]")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)
