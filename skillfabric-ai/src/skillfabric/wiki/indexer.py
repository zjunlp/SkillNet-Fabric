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
    """Render lightweight directory indexes for progressive reading."""

    if section == "skills":
        return _skills_index(source, page_summaries)
    if section == "communities":
        return _communities_index(source, page_summaries)
    if section == "workflows":
        return _workflows_index(source)
    if section == "references":
        return _references_index()
    if section == "skill-sources":
        return _skill_sources_index(source)
    return _root_index(source)


def _root_index(source: WikiSource) -> str:
    return "\n".join(
        [
            "# SkillFabric Wiki",
            "",
            "Start here. Use directory indexes for navigation, then open individual skill cards to evaluate candidates.",
            "",
            "## Directories",
            f"* [Skills](skills/) - routing cards for {len(source.skills)} skills.",
            f"* [Communities](communities/) - capability clusters for {len(source.communities)} groups.",
            "* [Workflows](workflows/) - validated ordering and handoff hints.",
            "* [Skill Sources](skills/source/) - authoritative SKILL.md files for second-stage reading.",
            "",
            "## Reading Order",
            "* Read [Skills](skills/) to find skill pages.",
            "* Open skill cards to evaluate routing fit.",
            "* Read full source under skills/source/ when the card is insufficient for final routing or execution.",
        ]
    ) + "\n"


def _skills_index(source: WikiSource, page_summaries: dict[str, str]) -> str:
    del page_summaries
    lines = ["# Skills", "", "Skill directory. Open a card to evaluate routing fit.", ""]
    for skill_id, skill in sorted(source.skills.items(), key=lambda item: item[1].name):
        lines.append(f"* [{skill.name}]({slug(skill_id)}.md)")
    return "\n".join(lines).rstrip() + "\n"


def _communities_index(source: WikiSource, page_summaries: dict[str, str]) -> str:
    lines = ["# Communities", "", "Capability clusters that group related skills.", ""]
    for community_id, community in sorted(source.communities.items(), key=lambda item: item[1].name):
        summary = page_summaries.get(community_id, community.summary)
        lines.append(f"* [{community.name}]({slug(community_id)}.md) - {_clean_summary(summary)}")
    return "\n".join(lines).rstrip() + "\n"


def _workflows_index(source: WikiSource) -> str:
    lines = ["# Workflows", "", "Validated handoff and ordering hints between skills.", ""]
    for record in sorted(source.execution_index, key=lambda item: (item.source_skill, item.target_skill, item.relation_type)):
        entity_id = f"{record.source_skill}__{record.target_skill}__{record.relation_type}"
        title = f"{record.source_skill} -> {record.target_skill}"
        lines.append(f"* [{title}]({slug(entity_id)}.md) - {record.relation_type} via {record.canonical_object}.")
    if len(lines) == 4:
        lines.append("* None")
    return "\n".join(lines).rstrip() + "\n"


def _references_index() -> str:
    return "\n".join(
        [
            "# References",
            "",
            "Reference pages are supplementary material. Full skill source files live under skills/source/.",
            "",
            "* [Skill Sources](../skills/source/) - authoritative original SKILL.md files.",
        ]
    ) + "\n"


def _skill_sources_index(source: WikiSource) -> str:
    lines = ["# Skill Sources", "", "Full original SKILL.md files. Prefer skill cards before opening these.", ""]
    for skill_id, skill in sorted(source.skills.items(), key=lambda item: item[1].name):
        lines.append(f"* [{skill.name}]({slug(skill_id)}.md) - full source for {skill_id}.")
    return "\n".join(lines).rstrip() + "\n"


def _clean_summary(value: str) -> str:
    text = " ".join(str(value).split())
    return text[:240].rstrip() if len(text) > 240 else text


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
