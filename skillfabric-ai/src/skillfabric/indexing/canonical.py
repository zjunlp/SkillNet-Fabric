"""Canonical skill text construction."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skillfabric.registry.models import SkillNode


def canonical_skill_text(skill: SkillNode) -> str:
    """Build retrieval text from name, description, and full SKILL.md."""

    return "\n".join(
        [
            skill.name.strip(),
            skill.description.strip(),
            skill.raw_text.strip(),
        ]
    ).strip()


def hash_text(text: str) -> str:
    """Compute a stable text hash."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
