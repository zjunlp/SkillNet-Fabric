"""Materialize the stable Full Wiki from canonical graph artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil

from skillfabric.storage import Workspace, atomic_write_text
from skillfabric.wiki.health import analyze_wiki_health, write_wiki_health_report
from skillfabric.wiki.indexer import render_index
from skillfabric.wiki.loader import WikiSource, load_wiki_source
from skillfabric.wiki.models import WikiBuildConfig, WikiBuildResult, WikiPage
from skillfabric.wiki.pages import slug
from skillfabric.wiki.renderers import (
    _first_paragraph,
    _skill_page,
    _skill_source_page,
)


def build_wiki(config: WikiBuildConfig) -> WikiBuildResult:
    """Build wiki markdown pages from an existing compiled graph."""

    workspace = Workspace(config.workspace)
    workspace.ensure()
    source = load_wiki_source(workspace)
    pages = _entity_pages(source, config, workspace)
    page_summaries = _directory_page_summaries(pages)
    pages.extend(_directory_pages(source, page_summaries, workspace))
    _reset_wiki_output(workspace)
    for page in pages:
        atomic_write_text(page.path, page.text)
    _write_manifest(workspace, source, pages)
    health = analyze_wiki_health(workspace)
    write_wiki_health_report(workspace, health)
    return WikiBuildResult(
        pages_written=len(pages),
        health=health,
        workspace=workspace.root,
    )


def _write_manifest(workspace: Workspace, source: WikiSource, pages: list[WikiPage]) -> None:
    """Publish stable page identities for task-wiki projections."""

    page_hashes = {
        page.path.relative_to(workspace.wiki_dir).as_posix(): _sha256(page.text) for page in pages
    }
    skills = []
    for skill_id, skill in sorted(source.skills.items(), key=lambda item: item[1].name):
        card_path = f"skills/cards/{slug(skill_id)}.md"
        source_path = f"skills/sources/{slug(skill_id)}.md"
        skills.append(
            {
                "skill_id": skill_id,
                "name": skill.name,
                "card_path": card_path,
                "source_path": source_path,
                "content_hash": skill.content_hash,
                "card_hash": page_hashes[card_path],
                "source_hash": page_hashes[source_path],
            }
        )
    manifest = {
        "build_id": source.build_id,
        "skills": skills,
        "page_hashes": dict(sorted(page_hashes.items())),
    }
    atomic_write_text(
        workspace.wiki_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _entity_pages(
    source: WikiSource,
    config: WikiBuildConfig,
    workspace: Workspace,
) -> list[WikiPage]:
    pages: list[WikiPage] = []
    for _skill_id, skill in sorted(source.skills.items(), key=lambda item: item[1].name):
        pages.append(_skill_page(source, skill, config, workspace))
        pages.append(_skill_source_page(skill, workspace))
    return pages


def _directory_page_summaries(pages: list[WikiPage]) -> dict[str, str]:
    """Return summaries from routing pages, excluding second-stage source references."""

    summaries: dict[str, str] = {}
    for page in pages:
        if _is_skill_source_page(page):
            continue
        if page.page_type == "skill":
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
