"""Generate wiki navigation pages."""

from __future__ import annotations

from pathlib import Path

from skillfabric.wiki.loader import WikiSource
from skillfabric.wiki.pages import slug


def render_index(
    source: WikiSource,
    page_summaries: dict[str, str],
) -> str:
    """Render the root LLM-readable wiki catalog."""

    return _root_index(source, page_summaries)


def _root_index(source: WikiSource, page_summaries: dict[str, str]) -> str:
    lines = [
        "# SkillFabric Wiki",
        "",
        "> Agent-readable skill knowledge bundle. Use skill cards for routing. "
        "Open full sources when exact boundaries, prerequisites, or execution details matter.",
        "",
        "## Corpus",
        f"- skills: {len(source.skills)}",
        "",
        "## Skill Cards",
    ]
    for skill_id, skill in sorted(source.skills.items(), key=lambda item: item[1].name):
        description = _clean_summary(page_summaries.get(skill_id) or skill.description)
        card_path = f"skills/cards/{slug(skill_id)}.md"
        source_path = f"skills/sources/{slug(skill_id)}.md"
        lines.append(f"- [{skill.name}]({card_path})")
        if description:
            lines.append(f"  summary: {description}")
        lines.append(f"  source: [full SKILL.md]({source_path})")
    lines.extend(["", "## Full Skill Sources"])
    for skill_id, skill in sorted(source.skills.items(), key=lambda item: item[1].name):
        lines.append(f"- [{skill.name}](skills/sources/{slug(skill_id)}.md)")
    return "\n".join(lines).rstrip() + "\n"


def _clean_summary(value: str) -> str:
    text = " ".join(str(value).split())
    if len(text) <= 240:
        return text
    clipped = text[:237].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{clipped}..."


def page_path(root: Path, category: str, entity_id: str) -> Path:
    """Return page path for one entity."""

    return root / category / f"{slug(entity_id)}.md"
