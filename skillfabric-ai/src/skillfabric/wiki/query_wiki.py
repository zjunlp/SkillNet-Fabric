"""Materialize a bounded query wiki for route-time exploration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.contracts.models import SkillContract
from skillfabric.compiled_graph.models import Edge
from skillfabric.registry.models import SkillNode
from skillfabric.router.models import RouterAlternative, RouterBundle, RouterSkillCandidate
from skillfabric.storage import Workspace, atomic_write_text
from skillfabric.wiki.contract_pages import render_contract_card, render_untrusted_skill_source
from skillfabric.wiki.explorer.prompting import render_query_wiki_explorer_md
from skillfabric.wiki.loader import load_wiki_source
from skillfabric.wiki.pages import slug


@dataclass(frozen=True, slots=True)
class QueryWikiBuildResult:
    root: Path


def materialize_query_wiki(
    workspace: Workspace | str | Path,
    bundle: RouterBundle,
    *,
    trace_dir: Path,
) -> QueryWikiBuildResult:
    """Create one self-contained query wiki from canonical semantic artifacts."""

    workspace = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
    source = load_wiki_source(workspace)
    query_root = trace_dir / "query_wiki"
    if query_root.exists():
        raise FileExistsError(f"query_wiki already exists: {query_root}")
    cards_dir = query_root / "skills" / "cards"
    sources_dir = query_root / "skills" / "sources"
    edges_dir = query_root / "edges"
    for directory in (cards_dir, sources_dir, edges_dir):
        directory.mkdir(parents=True, exist_ok=True)

    candidates = {candidate.skill_id: candidate for candidate in bundle.selected_skills}
    alternatives = {alternative.skill_id: alternative for alternative in bundle.alternatives}
    included_ids = list(candidates)
    included_ids.extend(skill_id for skill_id in alternatives if skill_id not in candidates)
    missing = sorted(set(included_ids) - set(source.skills))
    if missing:
        raise ValueError(f"router bundle references unknown skills: {', '.join(missing)}")

    skills_manifest: list[dict[str, Any]] = []
    for skill_id in included_ids:
        skill = source.skills[skill_id]
        contract = source.contracts[skill_id]
        candidate = candidates.get(skill_id)
        alternative = alternatives.get(skill_id)
        card_path = f"skills/cards/{slug(skill_id)}.md"
        source_path = f"skills/sources/{slug(skill_id)}.md"
        card = _render_skill_card(skill, contract, candidate=candidate, alternative=alternative)
        source_page = render_untrusted_skill_source(skill)
        atomic_write_text(query_root / card_path, card)
        atomic_write_text(query_root / source_path, source_page)
        skills_manifest.append(
            _manifest_skill(
                skill,
                candidate=candidate,
                alternative=alternative,
                card_path=card_path,
                source_path=source_path,
            )
        )

    semantic_edges_path = "edges/semantic_edges.jsonl"
    edge_rows = [_query_edge(edge) for edge in bundle.graph_edges]
    _write_jsonl(query_root / semantic_edges_path, edge_rows)
    manifest = {
        "query": bundle.query,
        "skills": skills_manifest,
        "semantic_edges_path": semantic_edges_path,
        "alternatives": [item.to_dict() for item in bundle.alternatives],
    }
    atomic_write_text(
        query_root / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    index = _render_index(manifest)
    atomic_write_text(query_root / "index.md", index)
    atomic_write_text(query_root / "EXPLORER.md", render_query_wiki_explorer_md())
    return QueryWikiBuildResult(root=query_root)


def render_query_wiki_skill_card(query_wiki_root: str | Path, skill_id: str) -> str:
    """Return one manifest-listed card without exposing files outside query_wiki."""

    root = Path(query_wiki_root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next(
        (
            item
            for item in manifest.get("skills", [])
            if isinstance(item, dict) and item.get("skill_id") == skill_id
        ),
        None,
    )
    if row is None:
        raise KeyError(f"skill not found in query_wiki manifest: {skill_id}")
    if not row.get("selectable"):
        raise ValueError(f"skill is not selectable in query_wiki: {skill_id}")
    card_path = _resolve_inside(root, str(row.get("card_path", "")))
    if not card_path.is_file():
        raise FileNotFoundError(f"query_wiki skill card is missing: {card_path}")
    return card_path.read_text(encoding="utf-8")


def _manifest_skill(
    skill: SkillNode,
    *,
    candidate: RouterSkillCandidate | None,
    alternative: RouterAlternative | None,
    card_path: str,
    source_path: str,
) -> dict[str, Any]:
    if candidate is not None:
        origin = "seed" if candidate.is_seed else "semantic_expansion"
        route = candidate.to_dict()
    else:
        origin = "similar_alternative"
        route = None
    return {
        "skill_id": skill.id,
        "name": skill.name,
        "description": skill.description,
        "selectable": True,
        "origin": origin,
        "card_path": card_path,
        "source_path": source_path,
        "route": route,
        "alternative": alternative.to_dict() if alternative is not None else None,
    }


def _render_skill_card(
    skill: SkillNode,
    contract: SkillContract,
    *,
    candidate: RouterSkillCandidate | None,
    alternative: RouterAlternative | None,
) -> str:
    route_lines: list[str] = []
    if candidate is not None:
        route_lines.extend(
            [
                f"- Origin: {'seed' if candidate.is_seed else 'semantic expansion'}",
                f"- RRF score: {candidate.score:.8f}",
                f"- Graph depth: {candidate.graph_depth}",
                f"- Retrieval ranks: {_mapping(candidate.retrieval_ranks)}",
            ]
        )
        if candidate.introduced_by:
            route_lines.append("- Expansion paths:")
            route_lines.extend(
                f"  - {_render_path(path.to_dict())}" for path in candidate.introduced_by
            )
    if alternative is not None:
        route_lines.append(
            f"- Similar alternative to {alternative.alternative_to} "
            f"(confidence {alternative.confidence:.3f}): {alternative.reason}"
        )
    return render_contract_card(
        skill,
        contract,
        context_lines=route_lines or ("- Similarity alternative for comparison.",),
    )


def _query_edge(edge: Edge) -> dict[str, Any]:
    if edge.type not in {"depend_on", "compose_with"}:
        raise ValueError(f"query_wiki received non-operational edge: {edge.type}")
    return edge.to_dict()


def _render_index(manifest: dict[str, Any]) -> str:
    lines = [
        "# Query Wiki",
        "",
        "Read this file first. Compare contract cards, then inspect source or semantic edges only "
        "when a routing decision needs more evidence.",
        "",
        "## Skill Cards",
    ]
    for row in manifest["skills"]:
        route = row.get("route") or {}
        score = float(route.get("score", 0.0))
        lines.extend(
            [
                f"- `{row['skill_id']}` ({row['origin']}, score={score:.8f})",
                f"  - card: {row['card_path']}",
                f"  - source: {row['source_path']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Edge Evidence",
            "",
            f"- canonical operational edges: {manifest['semantic_edges_path']}",
        ]
    )
    if manifest["alternatives"]:
        lines.extend(["", "## Similar Alternatives", ""])
        lines.extend(
            f"- `{item['skill_id']}` alternative to `{item['alternative_to']}`: {item['reason']}"
            for item in manifest["alternatives"]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_path(payload: dict[str, Any]) -> str:
    steps = payload.get("steps", [])
    if not steps:
        return str(payload.get("seed_skill", ""))
    parts = [str(payload.get("seed_skill", ""))]
    parts.extend(
        f"--{step.get('edge_type', '')}--> {step.get('target', '')}"
        for step in steps
        if isinstance(step, dict)
    )
    return " ".join(parts)


def _mapping(values: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(values.items())) or "none"


def _resolve_inside(root: Path, relative_path: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError(f"query_wiki path must be relative: {relative_path}")
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"query_wiki path escapes read root: {relative_path}") from exc
    return candidate


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


__all__ = [
    "QueryWikiBuildResult",
    "materialize_query_wiki",
    "render_query_wiki_skill_card",
]
