"""Coverage-aware RouterBundle candidate selection."""

from __future__ import annotations

from skillfabric.registry.models import SkillNode
from skillfabric.router.models import RouterSkillCandidate
from skillfabric.router.task_atoms import TaskDecomposition

_ATOM_SOURCE_WEIGHTS = {
    "object:produces": 1.4,
    "object:requires": 1.1,
    "interface": 1.2,
    "execution": 1.2,
    "bm25": 0.7,
    "embedding": 0.7,
    "lexical": 0.5,
}


def select_coverage_candidates(
    expanded: list[RouterSkillCandidate],
    *,
    seed_scores: dict[str, RouterSkillCandidate],
    skills: dict[str, SkillNode],
    task_atoms: TaskDecomposition,
    expanded_limit: int,
) -> list[RouterSkillCandidate]:
    """Select a compact candidate bundle while preserving atom coverage."""

    if expanded_limit <= 0:
        return []
    if not task_atoms.atoms:
        return _rank_candidates(expanded)[:expanded_limit]

    candidate_pool = _candidate_pool(expanded, seed_scores, skills)
    required_atom_ids = {atom.id for atom in task_atoms.atoms if atom.required}
    selected: list[RouterSkillCandidate] = []
    selected_ids: set[str] = set()
    covered: set[str] = set()
    while len(selected) < expanded_limit and len(selected_ids) < len(candidate_pool):
        best = _best_candidate(candidate_pool, selected_ids, covered, required_atom_ids)
        if best is None:
            break
        selected.append(best)
        selected_ids.add(best.skill_id)
        covered.update(set(best.atom_coverage) & required_atom_ids)
    return sorted(selected, key=lambda item: (-item.score, item.graph_depth, item.skill_id))


def _candidate_pool(
    expanded: list[RouterSkillCandidate],
    seed_scores: dict[str, RouterSkillCandidate],
    skills: dict[str, SkillNode],
) -> dict[str, RouterSkillCandidate]:
    pool: dict[str, RouterSkillCandidate] = {}
    for item in expanded:
        if item.skill_id in skills:
            pool[item.skill_id] = _copy_candidate(item)
    for skill_id, item in seed_scores.items():
        if skill_id not in skills:
            continue
        existing = pool.get(skill_id)
        if existing is None or item.score > existing.seed_score:
            candidate = _copy_candidate(item)
            if existing is not None:
                candidate.ppr_score = existing.ppr_score
                candidate.graph_depth = min(existing.graph_depth, candidate.graph_depth)
                for source in existing.sources:
                    if source not in candidate.sources:
                        candidate.sources.append(source)
                for key, value in existing.score_breakdown.items():
                    candidate.score_breakdown[key] = max(candidate.score_breakdown.get(key, 0.0), value)
                for atom_id, sources in existing.atom_coverage.items():
                    candidate.atom_coverage.setdefault(atom_id, [])
                    candidate.atom_coverage[atom_id].extend(sources)
                candidate.score = max(candidate.score, existing.score)
            pool[skill_id] = candidate
    return pool


def _best_candidate(
    candidates: dict[str, RouterSkillCandidate],
    selected_ids: set[str],
    covered: set[str],
    required_atom_ids: set[str],
) -> RouterSkillCandidate | None:
    best: RouterSkillCandidate | None = None
    best_key: tuple[float, float, int, str] | None = None
    for candidate in candidates.values():
        if candidate.skill_id in selected_ids:
            continue
        utility = candidate.score + _coverage_bonus(candidate, covered, required_atom_ids)
        key = (utility, candidate.score, -candidate.graph_depth, candidate.skill_id)
        if best_key is None or key > best_key:
            best_key = key
            best = candidate
    return best


def _coverage_bonus(
    candidate: RouterSkillCandidate,
    covered: set[str],
    required_atom_ids: set[str],
) -> float:
    bonus = 0.0
    for atom_id, sources in candidate.atom_coverage.items():
        if atom_id not in required_atom_ids:
            continue
        source_weight = max((_source_weight(source) for source in sources), default=0.0)
        if atom_id not in covered:
            bonus += 1.5 + source_weight
        else:
            bonus += 0.15 * source_weight
    return bonus


def _source_weight(source: str) -> float:
    suffix = source.split(":", 2)[-1]
    for key, weight in _ATOM_SOURCE_WEIGHTS.items():
        if key in suffix:
            return weight
    return 0.25


def _rank_candidates(candidates) -> list[RouterSkillCandidate]:
    return sorted(candidates, key=lambda item: (-item.score, item.graph_depth, item.skill_id))


def _copy_candidate(item: RouterSkillCandidate) -> RouterSkillCandidate:
    return RouterSkillCandidate(
        skill_id=item.skill_id,
        name=item.name,
        score=item.score,
        sources=list(item.sources),
        graph_depth=item.graph_depth,
        reason=item.reason,
        seed_score=item.seed_score,
        ppr_score=item.ppr_score,
        score_breakdown=dict(item.score_breakdown),
        atom_coverage={key: list(value) for key, value in item.atom_coverage.items()},
    )
