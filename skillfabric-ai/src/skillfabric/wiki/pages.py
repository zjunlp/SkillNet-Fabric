"""Shared wiki page formatting helpers."""

from __future__ import annotations

import re

import yaml


def slug(value: str) -> str:
    """Return a stable markdown file slug."""

    cleaned = value
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[1]
    return re.sub(r"[^a-zA-Z0-9]+", "-", cleaned.lower()).strip("-") or "unnamed"


def wiki_link(category: str, entity_id: str, label: str | None = None) -> str:
    """Return an Obsidian-style wikilink."""

    target_category = "skills/cards" if category == "skills" else category
    target = f"{target_category}/{slug(entity_id)}"
    if label and label != slug(entity_id):
        return f"[[{target}|{label}]]"
    return f"[[{target}]]"


def bullet_list(values: list[str], *, empty: str = "- None") -> str:
    """Render markdown bullets."""

    if not values:
        return empty
    return "\n".join(f"- {value}" for value in values)


def frontmatter(fields: dict[str, object]) -> str:
    """Render YAML frontmatter without allowing content to alter its structure."""

    document = yaml.safe_dump(
        fields,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    ).rstrip()
    return f"---\n{document}\n---"
