"""Generate wiki navigation pages."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from skillfabric.storage import atomic_write_text
from skillfabric.wiki.loader import WikiSource
from skillfabric.wiki.models import WikiBuildResult
from skillfabric.wiki.pages import bullet_list, slug, wiki_link


def render_index(source: WikiSource, page_summaries: dict[str, str]) -> str:
    """Render the content-oriented wiki index."""

    community_sections = []
    for community_id, members in sorted(source.community_members.items()):
        community = source.communities.get(community_id)
        label = community.name if community else community_id
        links = [
            wiki_link("skills", skill_id, source.skills[skill_id].name)
            for skill_id in members
            if skill_id in source.skills
        ]
        community_sections.append(f"### {wiki_link('communities', community_id, label)}\n\n{bullet_list(links)}")

    pages = [
        f"{wiki_link('skills', skill_id, skill.name)}: {page_summaries.get(skill_id, skill.description)}"
        for skill_id, skill in sorted(source.skills.items(), key=lambda item: item[1].name)
    ]
    workflow_links = [
        f"{wiki_link('workflows', f'{record.source_skill}__{record.target_skill}__{record.relation_type}', record.canonical_object)}: "
        f"{record.source_skill} -> {record.target_skill}"
        for record in source.execution_index[:20]
    ]
    return "\n\n".join(
        [
            "# SkillFabric Wiki",
            "Clean skill-level wiki materialized from the compiled skill knowledge bundle.",
            "## Counts",
            bullet_list(
                [
                    f"Skills: {len(source.skills)}",
                    f"Communities: {len(source.communities)}",
                    f"Core links: {len(source.core_edges)}",
                    f"Execution compatibility records: {len(source.execution_index)}",
                    f"Raw artifacts: {source.stats.get('raw_artifact_count', len(source.raw_artifacts))}",
                    f"Raw scenarios: {source.stats.get('raw_scenario_count', len(source.raw_scenarios))}",
                ]
            ),
            "## Skills by Community",
            "\n\n".join(community_sections) or "- None",
            "## Workflow Recipes",
            bullet_list(workflow_links),
            "## Pages",
            bullet_list(pages),
        ]
    ) + "\n"


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
