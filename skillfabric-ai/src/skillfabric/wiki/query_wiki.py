"""Materialize query-local wiki directories for route-time exploration."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.models import Edge
from skillfabric.router.assembly import _load_graph, _load_registry_skills
from skillfabric.router.models import RouterBundle
from skillfabric.router.sidecars import load_execution_index
from skillfabric.storage import Workspace, atomic_write_text
from skillfabric.wiki.explorer.prompting import render_query_wiki_explorer_md
from skillfabric.wiki.explorer.search_index import _split_frontmatter, _stable_id
from skillfabric.wiki.pages import slug

ORIGIN_ORDER = {
    "router_bundle": 0,
    "workflow_bridge": 1,
    "graph_frontier": 2,
}
SKILL_WIKI_LINK_PATTERN = re.compile(r"\[\[skills/(?:cards/)?([^\]|#]+)")
SKILL_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_:-])skill:[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?(?=->|$|[^A-Za-z0-9_.-])"
)


@dataclass(slots=True)
class QueryWikiBuildResult:
    """Query-local wiki build result."""

    root: Path
    manifest: dict[str, Any]


def render_query_wiki_skill_card(query_wiki_root: str | Path, skill_id: str) -> str:
    """Render the bounded card/header view for one query-wiki skill."""

    root = Path(query_wiki_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing query_wiki manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    skills = manifest.get("skills", [])
    row = next(
        (
            item
            for item in skills
            if isinstance(item, dict) and str(item.get("skill_id", "")) == skill_id
        ),
        None,
    )
    if row is None:
        raise KeyError(f"skill not found in query_wiki manifest: {skill_id}")
    if not row.get("selectable", True):
        raise ValueError(f"skill is not selectable in query_wiki: {skill_id}")
    card_path = str(row.get("card_path", ""))
    source_path = str(row.get("source_path", ""))
    lines = [
        f"# {skill_id}",
        "",
        "## Skill Card",
        f"- origin: {row.get('origin', '')}",
        f"- route_score: {float(row.get('score', 0.0) or 0.0):.6f}",
        f"- sources: {_format_card_value(row.get('sources')) or 'none'}",
        f"- card: {card_path}",
        f"- source: {source_path}",
    ]
    introduced_by = _format_introduced_by(row.get("introduced_by", []))
    if introduced_by:
        lines.append(f"- introduced_by: {introduced_by}")
    header = _skill_page_header(root, card_path)
    if header:
        lines.extend(["", header.rstrip()])
    return "\n".join(lines).rstrip() + "\n"


def materialize_query_wiki(
    workspace: Workspace,
    bundle: RouterBundle,
    *,
    trace_dir: Path,
    bridge_min_confidence: float = 0.95,
    frontier_min_confidence: float = 0.85,
    max_bridge_skills: int = 8,
    max_bridge_workflows: int = 8,
    max_frontier_skills: int = 8,
    frontier_top_k_per_source: int = 2,
) -> QueryWikiBuildResult:
    """Create runs/{trace_id}/query_wiki as the route explorer read root."""

    query_root = trace_dir / "query_wiki"
    if query_root.exists():
        shutil.rmtree(query_root)
    for path in (
        query_root / "skills" / "cards",
        query_root / "skills" / "sources",
        query_root / "communities",
        query_root / "workflows",
        query_root / "edges",
    ):
        path.mkdir(parents=True, exist_ok=True)

    graph = _load_graph(workspace)
    registry_skills = _load_registry_skills(workspace)
    execution_index = load_execution_index(workspace)
    core_ids = [item.skill_id for item in bundle.selected_skills]
    core_set = set(core_ids)
    score_lookup = {item.skill_id: item.score for item in bundle.selected_skills}
    source_lookup = {item.skill_id: sorted(set(item.sources)) for item in bundle.selected_skills}

    bridge_records = [
        record
        for record in sorted(
            execution_index,
            key=lambda item: (-item.confidence, item.source_skill, item.target_skill, item.canonical_object),
        )
        if record.confidence >= bridge_min_confidence
        and (record.source_skill in core_set or record.target_skill in core_set)
    ][:max_bridge_workflows]
    bridge_ids: list[str] = []
    for record in bridge_records:
        for skill_id in (record.source_skill, record.target_skill):
            if skill_id not in core_set and skill_id not in bridge_ids:
                bridge_ids.append(skill_id)
            if len(bridge_ids) >= max_bridge_skills:
                break
        if len(bridge_ids) >= max_bridge_skills:
            break
    bridge_set = set(bridge_ids)

    frontier_edges = _frontier_edges(
        graph.edges,
        source_ids=core_set | bridge_set,
        excluded_ids=core_set | bridge_set,
        min_confidence=frontier_min_confidence,
        top_k_per_source=frontier_top_k_per_source,
        max_frontier_skills=max_frontier_skills,
    )
    frontier_set = {
        edge.target if edge.source in core_set | bridge_set else edge.source
        for edge in frontier_edges
    } - core_set - bridge_set

    origins: dict[str, str] = {}
    for skill_id in core_ids:
        origins[skill_id] = "router_bundle"
    for skill_id in sorted(bridge_set):
        origins.setdefault(skill_id, "workflow_bridge")
    for skill_id in sorted(frontier_set):
        origins.setdefault(skill_id, "graph_frontier")
    included_skill_ids = set(origins)
    all_skill_ids = _workspace_skill_ids(workspace) | included_skill_ids
    external_slug_pattern = _external_slug_pattern(all_skill_ids - included_skill_ids)

    copied_pages: list[str] = []
    missing_pages: list[dict[str, str]] = []
    skills_manifest = []
    for skill_id, origin in sorted(origins.items(), key=_origin_sort_key):
        source = workspace.wiki_skill_cards_dir / f"{slug(skill_id)}.md"
        target = query_root / "skills" / "cards" / f"{slug(skill_id)}.md"
        description = ""
        if _copy_sanitized_page(source, target, included_skill_ids, external_slug_pattern):
            copied_pages.append(_rel(query_root, target))
            card_path = _rel(query_root, target)
            description = (
                registry_skills.get(skill_id).description
                if skill_id in registry_skills
                else _page_description(target)
            )
            source_target = query_root / "skills" / "sources" / target.name
            source_path = (
                _rel(query_root, source_target)
                if _copy_page(
                    workspace.wiki_skill_sources_dir / target.name,
                    source_target,
                )
                else ""
            )
            selectable = True
        else:
            missing_pages.append({"skill_id": skill_id, "source_path": str(source)})
            card_path = ""
            source_path = ""
            selectable = False
        skills_manifest.append(
            {
                "skill_id": skill_id,
                "origin": origin,
                "selectable": selectable,
                "score": score_lookup.get(skill_id, 0.0),
                "description": description,
                "sources": _manifest_sources(
                    skill_id,
                    origin=origin,
                    source_lookup=source_lookup,
                    bridge_records=bridge_records,
                    frontier_edges=frontier_edges,
                ),
                "card_path": card_path,
                "source_path": source_path,
                "introduced_by": _introduced_by(skill_id, core_set, included_skill_ids, bridge_records, frontier_edges),
            }
        )

    community_rows = _copy_communities(
        workspace,
        query_root,
        graph.edges,
        included_skill_ids,
        external_slug_pattern,
        copied_pages,
        missing_pages,
    )
    included_workflows, excluded_workflows = _copy_closed_workflows(
        workspace,
        query_root,
        execution_index,
        included_skill_ids,
        external_slug_pattern,
        copied_pages,
        missing_pages,
    )
    bridge_edge_rows = [
        _sanitize_edge_row(record.to_dict(), included_skill_ids, external_slug_pattern)
        for record in bridge_records
        if {record.source_skill, record.target_skill}.issubset(included_skill_ids)
    ]
    frontier_edge_rows = [
        _sanitize_edge_row(edge.to_dict(), included_skill_ids, external_slug_pattern)
        for edge in frontier_edges
        if {edge.source, edge.target}.issubset(included_skill_ids)
    ]
    _write_jsonl(query_root / "edges" / "bridge_edges.jsonl", bridge_edge_rows)
    _write_jsonl(query_root / "edges" / "frontier_edges.jsonl", frontier_edge_rows)

    debug_manifest = {
        "query": bundle.query,
        "source_wiki": str(workspace.wiki_dir),
        "query_wiki": str(query_root),
        "selection_policy": {
            "skill_expansion_depth": 1,
            "recursive_workflow_expansion": False,
            "workflow_closure": "included_skills_only",
            "include_similar_to": False,
            "bridge_min_confidence": bridge_min_confidence,
            "frontier_edge_types": ["depend_on", "compose_with"],
            "frontier_min_confidence": frontier_min_confidence,
            "max_bridge_skills": max_bridge_skills,
            "max_frontier_skills": max_frontier_skills,
        },
        "skills": skills_manifest,
        "communities": community_rows,
        "included_workflows": included_workflows,
        "excluded_workflows": excluded_workflows,
        "copied_pages": copied_pages,
        "missing_pages": missing_pages,
        "bridge_edges": bridge_edge_rows,
        "frontier_edges": frontier_edge_rows,
    }
    manifest = _explorer_manifest(debug_manifest)
    atomic_write_text(
        trace_dir / "query_wiki_debug_manifest.json",
        json.dumps(debug_manifest, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_text(query_root / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(query_root / "index.md", _render_index(manifest))
    atomic_write_text(query_root / "EXPLORER.md", _render_explorer_instructions())
    _write_page_index(query_root)
    return QueryWikiBuildResult(root=query_root, manifest=manifest)


def _frontier_edges(
    edges: list[Edge],
    *,
    source_ids: set[str],
    excluded_ids: set[str],
    min_confidence: float,
    top_k_per_source: int,
    max_frontier_skills: int,
) -> list[Edge]:
    by_source: dict[str, list[Edge]] = {}
    for edge in edges:
        if edge.type not in {"depend_on", "compose_with"} or edge.confidence < min_confidence:
            continue
        if edge.source in source_ids and edge.target not in excluded_ids:
            by_source.setdefault(edge.source, []).append(edge)
        elif edge.target in source_ids and edge.source not in excluded_ids:
            by_source.setdefault(edge.target, []).append(edge)
    selected: list[Edge] = []
    frontier_ids: set[str] = set()
    for source in sorted(by_source):
        candidates = sorted(by_source[source], key=lambda item: (-item.confidence, item.source, item.target))
        for edge in candidates[:top_k_per_source]:
            other = edge.target if edge.source == source else edge.source
            if other in frontier_ids and len(frontier_ids) >= max_frontier_skills:
                continue
            selected.append(edge)
            frontier_ids.add(other)
            if len(frontier_ids) >= max_frontier_skills:
                return selected
    return selected


def _copy_communities(
    workspace: Workspace,
    query_root: Path,
    edges: list[Edge],
    included_skill_ids: set[str],
    external_slug_pattern: re.Pattern[str] | None,
    copied_pages: list[str],
    missing_pages: list[dict[str, str]],
) -> list[dict[str, Any]]:
    members: dict[str, list[str]] = {}
    for edge in edges:
        if edge.type == "member_of" and edge.source in included_skill_ids:
            members.setdefault(edge.target, []).append(edge.source)
    rows: list[dict[str, Any]] = []
    for community_id, skill_ids in sorted(members.items()):
        source = workspace.wiki_communities_dir / f"{slug(community_id)}.md"
        target = query_root / "communities" / f"{slug(community_id)}.md"
        page_path = ""
        if _copy_sanitized_page(source, target, included_skill_ids, external_slug_pattern):
            copied_pages.append(_rel(query_root, target))
            page_path = _rel(query_root, target)
        else:
            missing_pages.append({"community_id": community_id, "source_path": str(source)})
        rows.append({"community_id": community_id, "skill_ids": sorted(skill_ids), "page_path": page_path})
    return rows


def _copy_closed_workflows(
    workspace: Workspace,
    query_root: Path,
    execution_index: list[Any],
    included_skill_ids: set[str],
    external_slug_pattern: re.Pattern[str] | None,
    copied_pages: list[str],
    missing_pages: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for record in sorted(execution_index, key=lambda item: (item.source_skill, item.target_skill, item.relation_type)):
        skill_ids = [record.source_skill, record.target_skill]
        entity_id = f"{record.source_skill}__{record.target_skill}__{record.relation_type}"
        if not set(skill_ids).issubset(included_skill_ids):
            excluded.append(
                {
                    "workflow_id": entity_id,
                    "skill_ids": skill_ids,
                    "reason": "referenced skill outside query_wiki",
                }
            )
            continue
        source = workspace.wiki_workflows_dir / f"{slug(entity_id)}.md"
        target = query_root / "workflows" / f"{slug(entity_id)}.md"
        page_path = ""
        if _copy_sanitized_page(source, target, included_skill_ids, external_slug_pattern):
            copied_pages.append(_rel(query_root, target))
            page_path = _rel(query_root, target)
        else:
            missing_pages.append({"workflow_id": entity_id, "source_path": str(source)})
        included.append({"workflow_id": entity_id, "skill_ids": skill_ids, "page_path": page_path})
    return included, excluded


def _introduced_by(
    skill_id: str,
    core_ids: set[str],
    included_skill_ids: set[str],
    bridge_records: list[Any],
    frontier_edges: list[Edge],
) -> list[dict[str, Any]]:
    if skill_id in core_ids:
        return [{"source": "router_bundle"}]
    rows: list[dict[str, Any]] = []
    for record in bridge_records:
        if skill_id in {record.source_skill, record.target_skill} and {
            record.source_skill,
            record.target_skill,
        }.issubset(included_skill_ids):
            rows.append(
                {
                    "source": "workflow_bridge",
                    "source_skill": record.source_skill,
                    "target_skill": record.target_skill,
                    "confidence": record.confidence,
                    "canonical_object": record.canonical_object,
                }
            )
    for edge in frontier_edges:
        if skill_id in {edge.source, edge.target} and {edge.source, edge.target}.issubset(included_skill_ids):
            rows.append(
                {
                    "source": "graph_frontier",
                    "edge_type": edge.type,
                    "source_skill": edge.source,
                    "target_skill": edge.target,
                    "confidence": edge.confidence,
                }
            )
    return rows


def _manifest_sources(
    skill_id: str,
    *,
    origin: str,
    source_lookup: dict[str, list[str]],
    bridge_records: list[Any],
    frontier_edges: list[Edge],
) -> list[str]:
    sources = list(source_lookup.get(skill_id, []))
    if origin == "workflow_bridge":
        for record in bridge_records:
            if skill_id in {record.source_skill, record.target_skill}:
                sources.append(f"query_wiki:workflow_bridge:{record.relation_type}")
    if origin == "graph_frontier":
        for edge in frontier_edges:
            if skill_id in {edge.source, edge.target}:
                sources.append(f"query_wiki:graph_frontier:{edge.type}")
    return sorted(set(sources))


def _origin_sort_key(item: tuple[str, str]) -> tuple[int, str, str]:
    skill_id, origin = item
    return (ORIGIN_ORDER.get(origin, len(ORIGIN_ORDER)), origin, skill_id)


def _copy_sanitized_page(
    source: Path,
    target: Path,
    included_skill_ids: set[str],
    external_slug_pattern: re.Pattern[str] | None,
) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8")
    atomic_write_text(target, _sanitize_markdown_skill_refs(text, included_skill_ids, external_slug_pattern))
    return True


def _copy_page(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, source.read_text(encoding="utf-8"))
    return True


def _workspace_skill_ids(workspace: Workspace) -> set[str]:
    if not workspace.wiki_skill_cards_dir.exists():
        return set()
    return {
        f"skill:{path.stem}"
        for path in workspace.wiki_skill_cards_dir.glob("*.md")
        if path.is_file()
    }


def _external_slug_pattern(external_skill_ids: set[str]) -> re.Pattern[str] | None:
    slugs = [
        re.escape(skill_id.removeprefix("skill:"))
        for skill_id in sorted(external_skill_ids, key=len, reverse=True)
        if "-" in skill_id.removeprefix("skill:") or "_" in skill_id.removeprefix("skill:")
    ]
    if not slugs:
        return None
    return re.compile(r"(?<![A-Za-z0-9_-])(?:" + "|".join(slugs) + r")(?![A-Za-z0-9_-])")


def _sanitize_markdown_skill_refs(
    text: str,
    included_skill_ids: set[str],
    external_slug_pattern: re.Pattern[str] | None,
) -> str:
    lines = [
        line
        for line in text.splitlines()
        if not _has_external_skill_ref(line, included_skill_ids, external_slug_pattern)
    ]
    return _fill_empty_sections("\n".join(lines)).rstrip() + "\n"


def _has_external_skill_ref(
    text: str,
    included_skill_ids: set[str],
    external_slug_pattern: re.Pattern[str] | None,
) -> bool:
    return any(skill_id not in included_skill_ids for skill_id in _explicit_skill_refs(text)) or (
        external_slug_pattern is not None and external_slug_pattern.search(text) is not None
    )


def _explicit_skill_refs(text: str) -> set[str]:
    refs = {f"skill:{match.group(1).strip()}" for match in SKILL_WIKI_LINK_PATTERN.finditer(text)}
    refs.update(match.group(0) for match in SKILL_ID_PATTERN.finditer(text))
    return refs


def _fill_empty_sections(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    section_lines: list[str] | None = None

    def flush_section() -> None:
        nonlocal section_lines
        if section_lines is None:
            return
        if not any(line.strip() and line.strip().lower() != "- none" for line in section_lines):
            output.append("- None")
        else:
            output.extend(section_lines)
        section_lines = None

    for line in lines:
        if line.startswith("## "):
            flush_section()
            output.append(line)
            section_lines = []
            continue
        if section_lines is None:
            output.append(line)
        else:
            section_lines.append(line)
    flush_section()
    return "\n".join(output)


def _sanitize_edge_row(
    row: dict[str, Any],
    included_skill_ids: set[str],
    external_slug_pattern: re.Pattern[str] | None,
) -> dict[str, Any]:
    sanitized = dict(row)
    evidence = []
    for item in row.get("evidence", []):
        if not isinstance(item, dict):
            continue
        skill_id = str(item.get("skill", ""))
        text = str(item.get("text", ""))
        if skill_id and skill_id not in included_skill_ids:
            continue
        if _has_external_skill_ref(text, included_skill_ids, external_slug_pattern):
            continue
        evidence.append(dict(item))
    sanitized["evidence"] = evidence
    if _has_external_skill_ref(str(sanitized.get("reason", "")), included_skill_ids, external_slug_pattern):
        sanitized["reason"] = ""
    return sanitized


def _explorer_manifest(debug_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in debug_manifest.items()
        if key not in {"source_wiki", "query_wiki", "excluded_workflows"}
    }


def _write_page_index(query_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(query_root.rglob("*.md")):
        rel = path.relative_to(query_root).as_posix()
        if rel.startswith(("skills/source/", "skills/sources/")):
            continue
        if rel == "EXPLORER.md":
            page_type = "instructions"
            entity_id = "explorer"
        else:
            text = path.read_text(encoding="utf-8")
            metadata, body = _split_frontmatter(text)
            page_type, entity_id = _page_identity(rel, metadata)
            title = _title(body) or path.stem
            rows.append(
                {
                    "page_id": _stable_id(rel),
                    "path": rel,
                    "page_type": page_type,
                    "entity_id": entity_id,
                    "title": title,
                    "summary": _summary(body),
                }
            )
            continue
        rows.append(
            {
                "page_id": _stable_id(rel),
                "path": rel,
                "page_type": page_type,
                "entity_id": entity_id,
                "title": path.stem,
                "summary": "",
            }
        )
    _write_jsonl(query_root / "page_index.jsonl", rows)


def _page_identity(rel: str, metadata: dict[str, Any]) -> tuple[str, str]:
    if rel == "index.md":
        return "index", "index"
    if rel.endswith("/index.md"):
        return "index", rel.removesuffix("/index.md").replace("/", "-") + "-index"
    if rel.startswith("skills/cards/"):
        return "skill", str(metadata.get("skill_id", f"skill:{Path(rel).stem}"))
    if rel.startswith("communities/"):
        return "community", str(metadata.get("community_id", Path(rel).stem))
    if rel.startswith("workflows/"):
        return "workflow", str(metadata.get("workflow_id", Path(rel).stem))
    return str(metadata.get("type", "page")).lower(), Path(rel).stem


def _render_index(manifest: dict[str, Any]) -> str:
    lines = [
        "# Query Wiki",
        "",
        "Read this file first. It is the compact routing map for the current task. "
        "Use it to locate candidate skill cards, then open cards for routing decisions, "
        "then open full source only when the card is insufficient.",
        "",
        f"Query: {manifest['query']}",
        "",
        "## Skill Cards",
    ]
    for skill in manifest["skills"]:
        if not skill.get("selectable", True):
            continue
        lines.extend(
            [
                f"- {skill['skill_id']} | score={float(skill.get('score', 0.0)):.6f} "
                f"| card: {skill.get('card_path', '')} | source: {skill.get('source_path', '')}",
            ]
        )
    lines.append("")
    lines.append("## Skills")
    for skill in manifest["skills"]:
        status = "selectable" if skill.get("selectable", True) else "missing"
        lines.append(
            f"- {skill['skill_id']} | score={float(skill.get('score', 0.0)):.6f} "
            f"| sources={_format_card_value(skill.get('sources')) or 'none'} | {status} "
            f"| card={skill.get('card_path', '')} | source={skill.get('source_path', '')}"
        )
    lines.append("")
    lines.append("## Communities")
    for community in manifest["communities"]:
        lines.append(f"- {community['community_id']} | {', '.join(community['skill_ids'])}")
    lines.append("")
    lines.append("## Workflows")
    for workflow in manifest["included_workflows"]:
        lines.append(f"- {workflow['workflow_id']} | {workflow['page_path']}")
    lines.append("")
    lines.append("## Edge Evidence")
    lines.append("- bridge_edges: edges/bridge_edges.jsonl")
    lines.append("- frontier_edges: edges/frontier_edges.jsonl")
    return "\n".join(lines) + "\n"


def _render_explorer_instructions() -> str:
    return render_query_wiki_explorer_md()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _title(body: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _summary(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:300]
    return ""


def _page_description(page_path: Path) -> str:
    text = page_path.read_text(encoding="utf-8")
    metadata, _body = _split_frontmatter(text)
    return str(metadata.get("description", "")).strip()


def _skill_page_header(query_root: Path, page_path: str) -> str:
    if not page_path:
        return ""
    candidate = (query_root / page_path).resolve()
    try:
        candidate.relative_to(query_root.resolve())
    except ValueError:
        return ""
    if not candidate.exists() or not candidate.is_file():
        return ""
    text = candidate.read_text(encoding="utf-8")
    _metadata, body = _split_frontmatter(text)
    return body.strip()


def _section(body: str, heading: str) -> str:
    marker = f"## {heading}"
    lines = body.splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == marker:
            start = index + 1
            break
    if start is None:
        return ""
    collected: list[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        collected.append(line)
    return "\n".join(collected).strip()


def _field_value(section: str, label: str) -> str:
    prefix = f"{label}:".lower()
    for line in section.splitlines():
        stripped = line.strip().lstrip("- ").strip()
        if stripped.lower().startswith(prefix):
            return stripped.split(":", 1)[1].strip()
    return ""


def _first_content_line(section: str) -> str:
    for line in section.splitlines():
        stripped = line.strip().lstrip("- ").strip()
        if stripped and stripped.lower() != "none":
            return stripped
    return ""


def _split_field(value: str) -> list[str]:
    if not value:
        return []
    return [_truncate(item.strip().strip("`"), 80) for item in value.split(",") if item.strip()]


def _section_items(section: str, *, limit: int) -> list[str]:
    items: list[str] = []
    for line in section.splitlines():
        stripped = line.strip().lstrip("- ").strip()
        if not stripped or stripped.lower() == "none":
            continue
        items.append(_truncate(stripped, 140))
        if len(items) >= limit:
            break
    return items


def _format_card_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item))
    if value is None:
        return ""
    return str(value)


def _format_introduced_by(rows: Any) -> str:
    if not isinstance(rows, list):
        return ""
    values: list[str] = []
    for row in rows[:4]:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source", ""))
        if source == "router_bundle":
            values.append(source)
            continue
        source_skill = str(row.get("source_skill", ""))
        target_skill = str(row.get("target_skill", ""))
        confidence = row.get("confidence")
        suffix = f" confidence={float(confidence):.2f}" if isinstance(confidence, int | float) else ""
        values.append(f"{source} {source_skill}->{target_skill}{suffix}".strip())
    return "; ".join(values)


def _truncate(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()
