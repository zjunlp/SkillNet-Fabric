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
        source_path=f"/tmp/{name}/SKILL.md",
        wiki_path=f"/tmp/wiki/{name}.md",
        content_hash=f"hash-{name}",
        token_count=len(raw_text.split()),
        canonical_skill_text_hash=f"canonical-{name}",
        raw_text=raw_text,
    )
