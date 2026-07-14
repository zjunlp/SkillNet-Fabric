"""Materialize the Compiled Skill Graph into markdown wiki pages."""

from __future__ import annotations

import shutil

from skillfabric.storage import Workspace, atomic_write_text
from skillfabric.wiki.health import analyze_wiki_health, write_wiki_health_report
from skillfabric.wiki.indexer import append_log, render_index
from skillfabric.wiki.loader import WikiSource, load_wiki_source
from skillfabric.wiki.models import WikiBuildConfig, WikiBuildResult, WikiPage, WikiSummaryRecord
from skillfabric.wiki.renderers import (
    _first_paragraph,
    _skill_page,
    _skill_source_page,
    _skill_summary_payload,
    _workflow_page,
)
from skillfabric.wiki.summarizer import WikiSummarizer


def build_wiki(config: WikiBuildConfig) -> WikiBuildResult:
    """Build wiki markdown pages from an existing compiled graph."""

    workspace = Workspace(config.workspace)
    workspace.ensure()
    source = load_wiki_source(workspace)
    summarizer = WikiSummarizer(config)
    pages = _entity_pages(source, config, summarizer, workspace)
    page_summaries = _directory_page_summaries(pages)
    pages.extend(_directory_pages(source, page_summaries, workspace))
    _reset_wiki_output(workspace)
    for page in pages:
        atomic_write_text(page.path, page.text)
    health = analyze_wiki_health(workspace)
    write_wiki_health_report(workspace, health)
    result = WikiBuildResult(
        pages_written=len(pages),
        cache_hits=summarizer.cache_hits,
        llm_calls=summarizer.llm_calls,
        health=health,
        workspace=workspace.root,
    )
    append_log(workspace.reports_dir / "wiki_log.md", result=result, build_id=source.build_id)
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
        pages.append(_skill_source_page(skill, workspace))
    pages.extend(
        _workflow_page(source, edge, workspace)
        for edge in sorted(
            source.operational_edges,
            key=lambda item: (item.type, item.source, item.target),
        )
    )
    return pages


def _directory_page_summaries(pages: list[WikiPage]) -> dict[str, str]:
    """Return summaries from routing pages, excluding second-stage source references."""

    summaries: dict[str, str] = {}
    for page in pages:
        if _is_skill_source_page(page):
            continue
        if page.page_type in {"skill", "workflow"}:
            summaries[page.entity_id] = _first_paragraph(page.text)
    return summaries


def _is_skill_source_page(page: WikiPage) -> bool:
    return (
        page.path.parent.name in {"source", "sources"} and page.path.parent.parent.name == "skills"
    )


def _directory_pages(
    source: WikiSource,
    page_summaries: dict[str, str],
    workspace: Workspace,
) -> list[WikiPage]:
    """Render the root wiki catalog."""

    return [
        WikiPage(
            path=workspace.wiki_dir / "index.md",
            page_type="index",
            entity_id="index",
            text=render_index(source, page_summaries),
        ),
    ]


def _reset_wiki_output(workspace: Workspace) -> None:
    """Replace the generated wiki only after source loading and summaries succeed."""

    if workspace.wiki_dir.exists():
        shutil.rmtree(workspace.wiki_dir)
    workspace.wiki_dir.mkdir(parents=True)


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
                "payload": _skill_summary_payload(skill, source.contracts.get(skill_id)),
            }
        )
    return summarizer.summarize_many(requests)
