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
from skillfabric.task_understanding import coverage_diagnostics


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
        task_understanding=bundle.task_understanding,
        coverage_diagnostics=coverage_diagnostics(bundle.task_understanding, selected_ids),
        rationale="Deterministic fallback selected high-scoring query-local skills while preserving task-facet specialists.",
        provenance="deterministic_fallback",
        warnings=warnings,
    )


def _select_fallback_candidates(
    candidates: list[RouterSkillCandidate],
    max_selected_skills: int,
) -> list[RouterSkillCandidate]:
    limit = max(max_selected_skills, 0)
    if limit <= 0:
        return []
    candidates = _quality_filtered_candidates(candidates, limit)
    selected = list(candidates[:limit])
    if len(selected) < limit:
        return selected

    reserve = _facet_reserve(limit)
    if reserve <= 0:
        return selected

    selected_ids = {item.skill_id for item in selected}
    selected_facet_count = sum(1 for item in selected if _is_facet_candidate(item))
    open_slots = max(0, reserve - selected_facet_count)
    if open_slots == 0:
        return selected

    drop_ids: set[str] = set()
    additions: list[RouterSkillCandidate] = []
    for candidate in candidates[limit:]:
        if len(additions) >= open_slots:
            break
        if candidate.skill_id in selected_ids or not _is_facet_candidate(candidate):
            continue
        drop = _lowest_replaceable_candidate(selected, drop_ids)
        if drop is None:
            break
        drop_ids.add(drop.skill_id)
        additions.append(candidate)

    if not additions:
        return selected
    return [item for item in selected if item.skill_id not in drop_ids] + additions


def _facet_reserve(max_selected_skills: int) -> int:
    if max_selected_skills < 10:
        return 0
    return min(2, max_selected_skills // 3)


def _quality_filtered_candidates(
    candidates: list[RouterSkillCandidate],
    limit: int,
) -> list[RouterSkillCandidate]:
    """Keep fallback routes compact while preserving strong and protected candidates."""

    if not candidates:
        return []
    minimum = min(3, limit)
    top_score = max(max(float(candidate.score), 0.0) for candidate in candidates)
    score_floor = max(0.35, top_score * 0.25) if top_score > 0 else 0.0
    output: list[RouterSkillCandidate] = []
    for index, candidate in enumerate(candidates):
        if index < minimum or _is_protected_candidate(candidate) or _is_facet_candidate(candidate):
            output.append(candidate)
            continue
        if candidate.score >= score_floor and _has_strong_evidence(candidate):
            output.append(candidate)
    return output


def _lowest_replaceable_candidate(
    selected: list[RouterSkillCandidate],
    drop_ids: set[str],
) -> RouterSkillCandidate | None:
    for candidate in reversed(selected):
        if candidate.skill_id in drop_ids:
            continue
        if _is_protected_candidate(candidate) or _is_facet_candidate(candidate):
            continue
        return candidate
    return None


def _is_facet_candidate(candidate: RouterSkillCandidate) -> bool:
    return any(source.startswith("facet:") for source in candidate.sources)


def _is_protected_candidate(candidate: RouterSkillCandidate) -> bool:
    return any(source.startswith("coverage:") for source in candidate.sources)


def _has_strong_evidence(candidate: RouterSkillCandidate) -> bool:
    sources = set(candidate.sources)
    if sources & {"bm25", "embedding"}:
        return True
    if any(source.startswith("ppr:") for source in sources) and candidate.ppr_score >= 0.05:
        return True
    lexical = float(candidate.score_breakdown.get("lexical", 0.0) or 0.0)
    return lexical >= 0.25
