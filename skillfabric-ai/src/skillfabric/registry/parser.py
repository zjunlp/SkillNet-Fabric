"""SKILL.md parser."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from skillfabric.registry.models import SkillNode

try:  # pragma: no cover - fall back when PyYAML is unavailable
    import yaml
except Exception:  # pragma: no cover
    yaml = None


_NAME_RE = re.compile(r"[^a-z0-9]+")


def parse_skill_file(path: str | Path) -> SkillNode:
    """Parse a single SKILL.md file."""

    skill_path = Path(path)
    raw_text = skill_path.read_text(encoding="utf-8")
    frontmatter, body, warnings = _split_frontmatter(raw_text)
    fallback_name = _slugify(skill_path.parent.name)
    name = _slugify(str(frontmatter.get("name") or fallback_name))
    description = str(frontmatter.get("description") or _first_useful_paragraph(body))
    if not frontmatter.get("name"):
        warnings.append("missing frontmatter name; used directory name")
    if not frontmatter.get("description"):
        warnings.append("missing frontmatter description; used first paragraph")

    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    return SkillNode(
        id=f"skill:{name}",
        type="skill",
        name=name,
        description=description.strip(),
        content_hash=content_hash,
        raw_text=raw_text,
        warnings=warnings,
    )


def _split_frontmatter(raw_text: str) -> tuple[dict[str, Any], str, list[str]]:
    warnings: list[str] = []
    if not raw_text.startswith("---"):
        return {}, raw_text, warnings
    end = raw_text.find("\n---", 3)
    if end == -1:
        warnings.append("frontmatter start found but closing delimiter missing")
        return {}, raw_text, warnings
    block = raw_text[3:end].strip()
    body = raw_text[raw_text.find("\n", end + 1) + 1 :]
    parsed: Any = None
    if yaml is not None:
        try:
            parsed = yaml.safe_load(block) or {}
        except Exception as exc:
            warnings.append(f"failed to parse yaml frontmatter: {exc}")
    if not isinstance(parsed, dict):
        parsed = _parse_simple_frontmatter(block)
    return parsed, body, warnings


def _parse_simple_frontmatter(block: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_key: str | None = None
    for line in block.splitlines():
        if ":" in line and not line.startswith((" ", "\t", "-")):
            key, _, value = line.partition(":")
            current_key = key.strip()
            data[current_key] = value.strip().strip('"').strip("'")
        elif current_key and line.strip().startswith("-"):
            existing = data.get(current_key)
            if not isinstance(existing, list):
                existing = []
                data[current_key] = existing
            existing.append(line.strip()[1:].strip())
    return data


def _first_useful_paragraph(body: str) -> str:
    for paragraph in re.split(r"\n\s*\n", body):
        text = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if not text or text.startswith("#"):
            continue
        text = re.sub(r"^#+\s*", "", text).strip()
        if text:
            return text
    return "No description provided."


def _slugify(value: str) -> str:
    slug = _NAME_RE.sub("-", value.strip().lower()).strip("-")
    return slug or "unnamed-skill"
