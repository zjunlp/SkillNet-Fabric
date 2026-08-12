"""Health checks for materialized wiki pages."""

from __future__ import annotations

import re
from pathlib import Path

from skillfabric.storage import Workspace, atomic_write_text
from skillfabric.wiki.loader import load_wiki_source
from skillfabric.wiki.models import WikiHealthReport
from skillfabric.wiki.pages import slug

WIKILINK_RE = re.compile(r"\[\[([^]|]+)(?:\|[^]]+)?]]")


def analyze_wiki_health(workspace: Workspace) -> WikiHealthReport:
    """Analyze wiki page consistency."""

    source = load_wiki_source(workspace)
    report = WikiHealthReport()
    wiki_dir = workspace.wiki_dir
    expected_skill_pages = {
        str(workspace.wiki_skill_cards_dir / f"{slug(skill_id)}.md") for skill_id in source.skills
    }
    for path in expected_skill_pages:
        if not Path(path).exists():
            report.missing_skill_pages.append(path)

    inbound: dict[str, int] = dict.fromkeys(source.skills, 0)
    for page in wiki_dir.rglob("*.md"):
        text = page.read_text(encoding="utf-8")
        rel = page.relative_to(wiki_dir).as_posix()
        if rel.startswith(("skills/source/", "skills/sources/", "references/skill-sources/")):
            continue
        generated_text = _generated_wiki_text(text)
        if "raw_output" in generated_text:
            report.raw_llm_output_leaks.append(str(page))
        if page.parent == workspace.wiki_skill_cards_dir and "## Inputs" not in text:
            report.skills_without_contract_sections.append(page.stem)
        if page.parent == workspace.wiki_skill_cards_dir and "## Related Skills" not in text:
            report.skills_without_related_links.append(page.stem)
        for target in WIKILINK_RE.findall(_strip_fenced_code_blocks(generated_text)):
            target_path = _wikilink_target_path(wiki_dir, target)
            if not target_path.exists():
                report.broken_links.append(f"{page}: {target}")
            skill_slug = _wikilink_skill_slug(target)
            if skill_slug:
                for skill_id in source.skills:
                    if slug(skill_id) == skill_slug:
                        inbound[skill_id] += 1
                        break
    for skill_id, count in inbound.items():
        if count == 0:
            report.orphan_skill_pages.append(skill_id)
    return report


def _generated_wiki_text(text: str) -> str:
    """Return generated routing/wiki content, excluding raw source excerpts."""

    if "\n## Source" in text:
        return text.split("\n## Source", 1)[0]
    if text.startswith("## Source"):
        return ""
    return text


def _strip_fenced_code_blocks(text: str) -> str:
    lines: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        stripped = line.lstrip()
        marker = stripped[:3]
        if fence is None and marker in {"```", "~~~"}:
            fence = marker
            continue
        if fence is not None:
            if stripped.startswith(fence):
                fence = None
            continue
        lines.append(line)
    return "\n".join(lines)


def _wikilink_target_path(wiki_dir: Path, target: str) -> Path:
    if target.startswith("skills/cards/"):
        return wiki_dir / f"{target}.md"
    if target.startswith("skills/"):
        return wiki_dir / "skills" / "cards" / f"{target.removeprefix('skills/')}.md"
    return wiki_dir / f"{target}.md"


def _wikilink_skill_slug(target: str) -> str:
    if target.startswith("skills/cards/"):
        return target.removeprefix("skills/cards/")
    if target.startswith("skills/"):
        return target.removeprefix("skills/")
    return ""


def render_wiki_health_report(report: WikiHealthReport) -> str:
    """Render wiki health report markdown."""

    sections = ["# Wiki Health Report", "## Summary"]
    sections.append("\n".join(f"- {key}: {value}" for key, value in report.summary.items()))
    details = {
        "Missing Skill Pages": report.missing_skill_pages,
        "Broken Links": report.broken_links,
        "Orphan Skill Pages": report.orphan_skill_pages,
        "Skills Without Contract Sections": report.skills_without_contract_sections,
        "Skills Without Related Links": report.skills_without_related_links,
        "Raw LLM Output Leaks": report.raw_llm_output_leaks,
    }
    for title, values in details.items():
        sections.append(f"## {title}")
        sections.append("\n".join(f"- {value}" for value in values) if values else "- None")
    return "\n\n".join(sections) + "\n"


def write_wiki_health_report(workspace: Workspace, report: WikiHealthReport) -> None:
    """Write wiki health markdown report."""

    atomic_write_text(workspace.wiki_dir / "health.md", render_wiki_health_report(report))
