"""Materialize the Compiled Skill Graph into markdown wiki pages."""

from __future__ import annotations

import shutil

from skillfabric.storage import Workspace, atomic_write_text
from skillfabric.wiki.explorer.search_index import build_wiki_page_index
from skillfabric.wiki.health import analyze_wiki_health, write_wiki_health_report
from skillfabric.wiki.indexer import append_log, render_index
from skillfabric.wiki.loader import WikiSource, load_wiki_source
from skillfabric.wiki.models import WikiBuildConfig, WikiBuildResult, WikiPage, WikiSummaryRecord
from skillfabric.wiki.renderers import (
    _common_interface_terms,
    _community_page,
    _content_hash,
    _debug_pages,
    _deliverables_page,
    _first_paragraph,
    _overview_page,
    _resolver_page,
    _skill_page,
    _skill_summary_payload,
    _workflow_page,
)
from skillfabric.wiki.summarizer import WikiSummarizer


def build_wiki(config: WikiBuildConfig) -> WikiBuildResult:
    """Build wiki markdown pages from an existing compiled graph."""

    workspace = Workspace(config.workspace)
    workspace.ensure()
    _prepare_wiki_dirs(workspace, include_debug_pages=config.include_debug_pages)
    source = load_wiki_source(workspace)
    summarizer = WikiSummarizer(config)
    pages = _entity_pages(source, config, summarizer, workspace)
    page_summaries = {
        page.entity_id: _first_paragraph(page.text)
        for page in pages
    }
    pages.extend(
        [
            _overview_page(source, workspace),
            _resolver_page(source, workspace),
            _deliverables_page(source, workspace),
            WikiPage(
                path=workspace.wiki_dir / "index.md",
                page_type="index",
                entity_id="index",
                title="SkillFabric Wiki",
                text=render_index(source, page_summaries),
            ),
        ]
    )
    for page in pages:
        atomic_write_text(page.path, page.text)
    summarizer.save()
    health = analyze_wiki_health(workspace, fallback_count=summarizer.fallback_count)
    write_wiki_health_report(workspace, health)
    result = WikiBuildResult(
        pages_written=len(pages) + 1,
        cache_hits=summarizer.cache_hits,
        llm_calls=summarizer.llm_calls,
        fallback_count=summarizer.fallback_count,
        health=health,
        workspace=workspace.root,
    )
    build_wiki_page_index(workspace)
    append_log(workspace.wiki_dir / "log.md", result=result, build_id=source.build_id)
    return result


def _entity_pages(
    source: WikiSource,
    config: WikiBuildConfig,
    summarizer: WikiSummarizer,
    workspace: Workspace,
) -> list[WikiPage]:
    pages: list[WikiPage] = []
    summaries = _summary_records(source, summarizer)
    for _skill_id, skill in sorted(source.skills.items(), key=lambda item: item[1].name):
        pages.append(_skill_page(source, skill, config, summaries, workspace))
    for community_id, _community in sorted(source.communities.items(), key=lambda item: item[1].name):
        pages.append(_community_page(source, community_id, config, summaries, workspace))
    for record in sorted(source.execution_index, key=lambda item: (item.source_skill, item.target_skill, item.relation_type)):
        pages.append(_workflow_page(source, record, workspace))
    if config.include_debug_pages:
        pages.extend(_debug_pages(source, workspace))
    return pages


def _prepare_wiki_dirs(workspace: Workspace, *, include_debug_pages: bool) -> None:
    """Remove stale generated pages that no longer belong to the main wiki view."""

    for path in _stale_main_wiki_dirs(workspace):
        if path.exists():
            shutil.rmtree(path)
    stale_hot = workspace.wiki_dir / "hot.md"
    if stale_hot.exists():
        stale_hot.unlink()
    if not include_debug_pages and workspace.wiki_debug_dir.exists():
        shutil.rmtree(workspace.wiki_debug_dir)
    workspace.wiki_workflows_dir.mkdir(parents=True, exist_ok=True)
    if include_debug_pages:
        workspace.wiki_debug_raw_artifacts_dir.mkdir(parents=True, exist_ok=True)
        workspace.wiki_debug_raw_scenarios_dir.mkdir(parents=True, exist_ok=True)


def _stale_main_wiki_dirs(workspace: Workspace) -> tuple:
    return (
        workspace.wiki_dir / "artifacts",
        workspace.wiki_dir / "scenarios",
    )


def _summary_records(
    source: WikiSource,
    summarizer: WikiSummarizer,
) -> dict[tuple[str, str], WikiSummaryRecord]:
    requests: list[dict[str, object]] = []
    for skill_id, skill in sorted(source.skills.items(), key=lambda item: item[1].name):
        requests.append(
            {
                "page_type": "skill",
                "entity_id": skill_id,
                "content_hash": skill.content_hash,
                "payload": _skill_summary_payload(skill, source.interfaces.get(skill_id)),
            }
        )
    for community_id, community in sorted(source.communities.items(), key=lambda item: item[1].name):
        members = source.community_members.get(community_id, [])
        requests.append(
            {
                "page_type": "community",
                "entity_id": community_id,
                "content_hash": _content_hash([community_id, *members]),
                "payload": {
                    "name": community.name,
                    "summary": community.summary,
                    "member_count": len(members),
                    **_common_interface_terms(source, members),
                },
            }
        )
    return summarizer.summarize_many(requests)
