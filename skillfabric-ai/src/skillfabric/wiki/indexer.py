"""Generate wiki navigation pages."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from skillfabric.storage import atomic_write_text
from skillfabric.wiki.loader import WikiSource
from skillfabric.wiki.models import WikiBuildResult
from skillfabric.wiki.pages import slug


def render_index(
    source: WikiSource,
    page_summaries: dict[str, str],
    *,
    section: str = "root",
) -> str:
    """Render the root LLM-readable wiki catalog."""

    del section
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
        f"- workflows: {len(source.execution_index)}",
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
    if source.execution_index:
        lines.extend(["", "## Workflows"])
        for record in sorted(source.execution_index, key=lambda item: (item.source_skill, item.target_skill, item.relation_type)):
            entity_id = f"{record.source_skill}__{record.target_skill}__{record.relation_type}"
            title = f"{record.source_skill} -> {record.target_skill}"
            lines.append(f"- [{title}](workflows/{slug(entity_id)}.md): {record.relation_type} via {record.canonical_object}.")
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


def append_log(path: Path, *, result: WikiBuildResult, build_id: str) -> None:
    """Append one wiki-build log entry."""

    existing = path.read_text(encoding="utf-8") if path.exists() else "# Wiki Log\n\n"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"## [{now}] wiki-build | build_id={build_id}\n\n"
        f"- pages_written: {result.pages_written}\n"
        f"- cache_hits: {result.cache_hits}\n"
        f"- llm_calls: {result.llm_calls}\n"
        f"- fallback_count: {result.fallback_count}\n"
        f"- health_warnings: {sum(value for key, value in result.health.summary.items() if key != 'summary_fallback_count')}\n\n"
    )
    atomic_write_text(path, existing.rstrip() + "\n\n" + entry)


def page_path(root: Path, category: str, entity_id: str) -> Path:
    """Return page path for one entity."""

    return root / category / f"{slug(entity_id)}.md"
