from __future__ import annotations

from skillfabric.registry.models import SkillNode


def make_skill(
    skill_id: str,
    name: str,
    raw_text: str,
    *,
    tools: list[str] | None = None,
    artifacts: list[str] | None = None,
    actions: list[str] | None = None,
) -> SkillNode:
    del tools, artifacts, actions
    return SkillNode(
        id=skill_id,
        type="skill",
        name=name,
        description=f"{name} description",
        content_hash=f"hash-{name}",
        raw_text=raw_text,
    )
