"""Graph expansion for router bundle candidates."""

from __future__ import annotations

from skillfabric.compiled_graph.models import Edge
from skillfabric.registry.models import SkillNode
from skillfabric.router.models import RouterSkillCandidate


def _expand_seed_skills(
    edges: list[Edge],
    skills: dict[str, SkillNode],
    seeds: dict[str, RouterSkillCandidate],
    *,
    seed_limit: int,
    expanded_limit: int,
) -> list[RouterSkillCandidate]:
    ranked_seeds = _ranked_seeds(seeds, seed_limit)
    selected = {item.skill_id: item for item in ranked_seeds}
    seed_ids = set(selected)
    for edge in edges:
        if edge.type == "member_of":
            continue
        if edge.source in seed_ids and edge.target in skills:
            _add_graph_expansion(selected, seeds, skills[edge.target], selected[edge.source], edge)
        if edge.target in seed_ids and edge.source in skills:
            _add_graph_expansion(selected, seeds, skills[edge.source], selected[edge.target], edge)
    return _limit_candidates(selected, expanded_limit)


def _add_graph_expansion(
    selected: dict[str, RouterSkillCandidate],
    seeds: dict[str, RouterSkillCandidate],
    skill: SkillNode,
    seed: RouterSkillCandidate,
    edge: Edge,
) -> None:
    priority = _edge_priority(edge.type)
    score = seed.score * priority * max(edge.confidence, 0.0)
    candidate = selected.get(skill.id)
    if candidate is None:
        seed_candidate = seeds.get(skill.id)
        if seed_candidate is not None:
            candidate = RouterSkillCandidate(
                skill_id=seed_candidate.skill_id,
                name=seed_candidate.name,
                score=seed_candidate.score,
                sources=list(seed_candidate.sources),
                graph_depth=1,
                reason=f"expanded from {seed.skill_id}",
                seed_score=seed_candidate.seed_score,
                ppr_score=0.0,
                score_breakdown=dict(seed_candidate.score_breakdown),
                atom_coverage={key: list(value) for key, value in seed_candidate.atom_coverage.items()},
            )
        else:
            candidate = RouterSkillCandidate(
                skill_id=skill.id,
                name=skill.name,
                score=0.0,
                sources=[],
                graph_depth=1,
                reason=f"expanded from {seed.skill_id}",
                seed_score=0.0,
                ppr_score=0.0,
                score_breakdown={},
                atom_coverage={},
            )
        selected[skill.id] = candidate
    if candidate.graph_depth > 0:
        candidate.reason = f"expanded from {seed.skill_id}"
    candidate.score = max(candidate.score, score)
    candidate.score_breakdown[f"graph:{edge.type}"] = max(
        candidate.score_breakdown.get(f"graph:{edge.type}", 0.0),
        score,
    )
    candidate.graph_depth = min(candidate.graph_depth, 1)
    candidate.sources.append(f"graph:{edge.type}")


def _expand_seed_skills_ppr(
    edges: list[Edge],
    skills: dict[str, SkillNode],
    seeds: dict[str, RouterSkillCandidate],
    *,
    seed_limit: int,
    expanded_limit: int,
    alpha: float,
    max_iter: int,
    tol: float,
) -> list[RouterSkillCandidate]:
    if expanded_limit <= 0:
        return []
    ranked_seeds = _ranked_seeds(seeds, seed_limit)
    if not ranked_seeds:
        return []
    seed_ids = {item.skill_id for item in ranked_seeds}
    adjacency = _ppr_adjacency(edges, skills)
    personalization = _personalization(ranked_seeds)
    ppr_scores = _personalized_pagerank(adjacency, personalization, alpha=alpha, max_iter=max_iter, tol=tol)
    depths = _ppr_depths(adjacency, seed_ids)
    support_sources = _ppr_support_sources(edges, skills, seed_ids)
    candidates: dict[str, RouterSkillCandidate] = {}
    for seed in ranked_seeds:
        candidates[seed.skill_id] = RouterSkillCandidate(
            skill_id=seed.skill_id,
            name=seed.name,
            score=seed.score,
            sources=list(seed.sources),
            graph_depth=0,
            reason="query seed",
            seed_score=seed.seed_score,
            ppr_score=ppr_scores.get(seed.skill_id, 0.0),
            score_breakdown=dict(seed.score_breakdown),
            atom_coverage={key: list(value) for key, value in seed.atom_coverage.items()},
        )
    for skill_id, ppr_score in ppr_scores.items():
        if skill_id not in skills or ppr_score <= 0:
            continue
        candidate = candidates.get(skill_id)
        if candidate is None:
            skill = skills[skill_id]
            seed_candidate = seeds.get(skill_id)
            candidate = RouterSkillCandidate(
                skill_id=skill_id,
                name=skill.name,
                score=0.0,
                sources=[],
                graph_depth=depths.get(skill_id, 99),
                reason="ppr propagated from query seeds",
                seed_score=seed_candidate.seed_score if seed_candidate is not None else 0.0,
                ppr_score=ppr_score,
                score_breakdown=dict(seed_candidate.score_breakdown) if seed_candidate is not None else {},
                atom_coverage={
                    key: list(value)
                    for key, value in (seed_candidate.atom_coverage.items() if seed_candidate is not None else [])
                },
            )
            candidates[skill_id] = candidate
        else:
            candidate.reason = "seed + ppr support"
            candidate.ppr_score = ppr_score
        candidate.score = max(candidate.score, candidate.seed_score + (0.75 * ppr_score))
        candidate.score_breakdown["ppr"] = max(candidate.score_breakdown.get("ppr", 0.0), 0.75 * ppr_score)
        candidate.graph_depth = min(candidate.graph_depth, depths.get(skill_id, candidate.graph_depth))
        for source in support_sources.get(skill_id, []):
            candidate.sources.append(source)
    return _limit_candidates(candidates, expanded_limit)


def _ranked_seeds(
    seeds: dict[str, RouterSkillCandidate],
    seed_limit: int,
) -> list[RouterSkillCandidate]:
    ranked = sorted(seeds.values(), key=lambda item: (-item.score, item.skill_id))
    return ranked[: max(seed_limit, 0)]


def _limit_candidates(
    candidates: dict[str, RouterSkillCandidate],
    expanded_limit: int,
) -> list[RouterSkillCandidate]:
    ranked = sorted(candidates.values(), key=lambda item: (-item.score, item.graph_depth, item.skill_id))
    return ranked[: max(expanded_limit, 0)]


def _ppr_adjacency(edges: list[Edge], skills: dict[str, SkillNode]) -> dict[str, dict[str, float]]:
    adjacency: dict[str, dict[str, float]] = {skill_id: {} for skill_id in skills}
    for edge in edges:
        if edge.type == "member_of":
            continue
        if edge.source not in skills or edge.target not in skills:
            continue
        confidence = max(edge.confidence, 0.0)
        if edge.type == "similar_to":
            _add_transition(adjacency, edge.source, edge.target, 0.55 * confidence)
            _add_transition(adjacency, edge.target, edge.source, 0.55 * confidence)
        elif edge.type == "compose_with":
            _add_transition(adjacency, edge.source, edge.target, 0.85 * confidence)
            _add_transition(adjacency, edge.target, edge.source, 0.85 * confidence)
        elif edge.type == "depend_on":
            _add_transition(adjacency, edge.source, edge.target, 1.00 * confidence)
            _add_transition(adjacency, edge.target, edge.source, 0.35 * confidence)
    return adjacency


def _add_transition(adjacency: dict[str, dict[str, float]], source: str, target: str, weight: float) -> None:
    if weight <= 0:
        return
    adjacency.setdefault(source, {})
    adjacency[source][target] = adjacency[source].get(target, 0.0) + weight


def _personalization(seeds: list[RouterSkillCandidate]) -> dict[str, float]:
    total = sum(max(seed.score, 0.0) for seed in seeds)
    if total <= 0:
        return {seed.skill_id: 1.0 / len(seeds) for seed in seeds}
    return {seed.skill_id: max(seed.score, 0.0) / total for seed in seeds}


def _personalized_pagerank(
    adjacency: dict[str, dict[str, float]],
    personalization: dict[str, float],
    *,
    alpha: float,
    max_iter: int,
    tol: float,
) -> dict[str, float]:
    nodes = sorted(adjacency)
    if not nodes:
        return {}
    alpha = min(max(alpha, 0.0), 1.0)
    restart = {node: personalization.get(node, 0.0) for node in nodes}
    rank = dict(restart)
    for _ in range(max_iter):
        next_rank = {node: (1.0 - alpha) * restart.get(node, 0.0) for node in nodes}
        leaked = 0.0
        for source, neighbors in adjacency.items():
            source_rank = rank.get(source, 0.0)
            total_weight = sum(neighbors.values())
            if total_weight <= 0:
                leaked += source_rank
                continue
            for target, weight in neighbors.items():
                next_rank[target] = next_rank.get(target, 0.0) + alpha * source_rank * (weight / total_weight)
        if leaked:
            for node, value in restart.items():
                next_rank[node] = next_rank.get(node, 0.0) + alpha * leaked * value
        delta = sum(abs(next_rank.get(node, 0.0) - rank.get(node, 0.0)) for node in nodes)
        rank = next_rank
        if delta <= tol:
            break
    return _normalize_scores(rank)


def _ppr_depths(adjacency: dict[str, dict[str, float]], seed_ids: set[str]) -> dict[str, int]:
    depths = {seed_id: 0 for seed_id in seed_ids}
    queue = list(seed_ids)
    while queue:
        source = queue.pop(0)
        for target in adjacency.get(source, {}):
            if target in depths:
                continue
            depths[target] = depths[source] + 1
            queue.append(target)
    return depths


def _ppr_support_sources(edges: list[Edge], skills: dict[str, SkillNode], seed_ids: set[str]) -> dict[str, list[str]]:
    adjacency = _ppr_adjacency(edges, skills)
    depths = _ppr_depths(adjacency, seed_ids)
    output: dict[str, list[str]] = {}
    for edge in edges:
        if edge.type == "member_of":
            continue
        if edge.source not in skills or edge.target not in skills:
            continue
        source_depth = depths.get(edge.source)
        target_depth = depths.get(edge.target)
        if source_depth is not None and target_depth is not None and target_depth == source_depth + 1:
            output.setdefault(edge.target, []).append(f"ppr:{edge.type}")
        if edge.type in {"similar_to", "compose_with"} and source_depth is not None and target_depth is not None and source_depth == target_depth + 1:
            output.setdefault(edge.source, []).append(f"ppr:{edge.type}")
        if edge.type == "depend_on" and source_depth is not None and target_depth is not None and source_depth == target_depth + 1:
            output.setdefault(edge.source, []).append("ppr:depend_on")
    return output


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values()) or 1.0
    return {key: value / max_score for key, value in scores.items()}


def _edge_priority(edge_type: str) -> float:
    return {
        "depend_on": 0.95,
        "compose_with": 0.75,
        "similar_to": 0.45,
    }.get(edge_type, 0.0)
