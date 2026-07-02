"""Deterministic and explorer-backed route selection."""

from __future__ import annotations

from pathlib import Path

from skillfabric.router.models import (
    RouterBundle,
    RouteResult,
    RouterSkillCandidate,
    RouteSelectedSkill,
)
from skillfabric.router.route_edges import _edges_from_workflow_hints


def _fallback_route(
    bundle: RouterBundle,
    *,
    query: str,
    trace_id: str,
    trace_dir: Path,
    max_selected_skills: int,
    warnings: list[str],
) -> RouteResult:
    selected_candidates = _select_fallback_candidates(bundle.selected_skills, max_selected_skills)
    selected = [
        RouteSelectedSkill(
            skill_id=item.skill_id,
            name=item.name,
            rank=index + 1,
            score=item.score,
            reason=item.reason or "selected by deterministic bundle score",
            evidence=item.sources,
        )
        for index, item in enumerate(selected_candidates)
    ]
    selected_ids = {item.skill_id for item in selected}
    required_edges = _edges_from_workflow_hints(bundle, selected_ids)
    return RouteResult(
        query=query,
        trace_id=trace_id,
        trace_dir=trace_dir,
        selected_skills=selected,
        required_edges=required_edges,
        ordered_hints=list(required_edges),
        near_misses=[],
        wiki_pages_read=list(bundle.wiki_pages),
        rationale="Deterministic fallback selected the highest-scoring query-local skills from the router bundle.",
        provenance="deterministic_fallback",
        task_atoms=bundle.task_atoms,
        warnings=warnings,
    )


def _select_fallback_candidates(
    candidates: list[RouterSkillCandidate],
    max_selected_skills: int,
) -> list[RouterSkillCandidate]:
    limit = max(max_selected_skills, 0)
    if limit <= 0:
        return []
    return list(candidates[:limit])
