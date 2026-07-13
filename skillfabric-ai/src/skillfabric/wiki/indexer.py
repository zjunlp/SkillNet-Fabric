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
        f"- workflows: {len(source.operational_edges)}",
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
    if source.operational_edges:
        lines.extend(["", "## Workflows"])
        for edge in sorted(
            source.operational_edges, key=lambda item: (item.type, item.source, item.target)
        ):
            entity_id = f"{edge.source}__{edge.target}__{edge.type}"
            title = f"{edge.source} -> {edge.target}"
            lines.append(f"- [{title}](workflows/{slug(entity_id)}.md): {edge.type}. {edge.reason}")
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
        f"- health_warnings: {sum(result.health.summary.values())}\n\n"
    )
    atomic_write_text(path, existing.rstrip() + "\n\n" + entry)


def page_path(root: Path, category: str, entity_id: str) -> Path:
    """Return page path for one entity."""

    return root / category / f"{slug(entity_id)}.md"
