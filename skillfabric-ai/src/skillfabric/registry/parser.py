"""SKILL.md parser."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from skillfabric.registry.models import SkillNode

_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_NAME_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")
_MAX_NAME_LENGTH = 64


def parse_skill_file(path: str | Path) -> SkillNode:
    """Parse a single SKILL.md file."""

    skill_path = Path(path)
    raw_text = skill_path.read_text(encoding="utf-8")
    frontmatter = _parse_frontmatter(raw_text)
    name = frontmatter.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{skill_path} frontmatter name must be a non-empty string")
    name = _normalize_name(name)
    if len(name) > _MAX_NAME_LENGTH or _NAME_RE.fullmatch(name) is None:
        raise ValueError(
            f"{skill_path} frontmatter name must normalize to at most {_MAX_NAME_LENGTH} "
            "lowercase letters, numbers, and single hyphens"
        )
    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{skill_path} frontmatter description must be a non-empty string")

    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    return SkillNode(
        id=f"skill:{name}",
        type="skill",
        name=name,
        description=description.strip(),
        content_hash=content_hash,
        raw_text=raw_text,
    )


def _normalize_name(value: str) -> str:
    return _NAME_SEPARATOR_RE.sub("-", value.strip().lower()).strip("-")


def _parse_frontmatter(raw_text: str) -> dict[str, Any]:
    if not raw_text.startswith("---\n"):
        raise ValueError("SKILL.md requires YAML frontmatter")
    end = raw_text.find("\n---", 3)
    if end == -1:
        raise ValueError("frontmatter start found but closing delimiter is missing")
    block = raw_text[3:end].strip()
    try:
        parsed: Any = yaml.safe_load(block) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML frontmatter: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("YAML frontmatter must be a mapping")
    return parsed
