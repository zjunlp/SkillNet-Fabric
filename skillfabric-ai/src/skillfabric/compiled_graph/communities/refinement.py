"""Community metadata refinement orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.communities.cache import _cache_key, _load_cache, _write_cache
from skillfabric.compiled_graph.communities.common import _members_by_community, _string_list
from skillfabric.compiled_graph.communities.providers import (
    CommunityRefinementProvider,
    DeterministicCommunityRefinementProvider,
)
from skillfabric.compiled_graph.interface.models import SkillInterface
from skillfabric.compiled_graph.models import CommunityNode, Edge
from skillfabric.registry.models import SkillNode
from skillfabric.runtime.jobs import LLMJobOptions, run_llm_jobs


def refine_communities(
    communities: list[CommunityNode],
    skills: list[SkillNode],
    internal_edges: list[Edge],
    membership: dict[str, str],
    *,
    provider: CommunityRefinementProvider | None = None,
    cache_path: str | Path | None = None,
    interfaces: dict[str, SkillInterface] | None = None,
    job_options: LLMJobOptions | None = None,
) -> list[CommunityNode]:
    """Refine community names and summaries without changing membership."""

    provider = provider or DeterministicCommunityRefinementProvider()
    cache = _load_cache(cache_path)
    by_skill = {skill.id: skill for skill in skills}
    by_community = _members_by_community(membership)
    results: list[CommunityNode | None] = [None] * len(communities)
    pending: list[tuple[int, CommunityNode, dict[str, Any], str]] = []
    for index, community in enumerate(communities):
        payload = _payload_for_community(community, by_community.get(community.id, []), by_skill, internal_edges, interfaces or {})
        key = _cache_key(community, payload, provider.model_id)
        cached = cache.get(key)
        if isinstance(cached, dict):
            results[index] = _community_from_refinement(
                community,
                cached,
                model_id=provider.model_id,
                member_ids=by_community.get(community.id, []),
            )
            continue
        pending.append((index, community, payload, key))

    def refine_one(item: tuple[int, CommunityNode, dict[str, Any], str]) -> dict[str, Any]:
        _index, _community, payload, _key = item
        raw = provider.refine(payload)
        if not isinstance(raw, dict):
            raise ValueError("community refinement output must be a JSON object")
        return _normalize_refinement(raw)

    def on_success(outcome) -> None:
        index, community, _payload, key = outcome.item
        raw = outcome.value
        cache[key] = raw
        results[index] = _community_from_refinement(
            community,
            raw,
            model_id=provider.model_id,
            member_ids=by_community.get(community.id, []),
        )
        _write_cache(cache_path, cache)

    outcomes = run_llm_jobs(
        pending,
        refine_one,
        options=job_options,
        label="community-refinement",
        on_success=on_success,
    )
    for outcome in outcomes:
        if outcome.ok:
            continue
        index, community, _payload, _key = outcome.item
        results[index] = community
    _write_cache(cache_path, cache)
    return [item if item is not None else communities[index] for index, item in enumerate(results)]


def _payload_for_community(
    community: CommunityNode,
    member_ids: list[str],
    by_skill: dict[str, SkillNode],
    edges: list[Edge],
    interfaces: dict[str, SkillInterface],
) -> dict[str, Any]:
    member_set = set(member_ids)
    internal_edges = [
        edge
        for edge in sorted(edges, key=_edge_sort_key)
        if edge.source in member_set and edge.target in member_set
    ][:12]
    boundary_edges = [
        edge
        for edge in sorted(edges, key=_edge_sort_key)
        if (edge.source in member_set) != (edge.target in member_set)
    ][:12]
    return {
        "community": community.to_dict(),
        "members": [
            {
                "id": skill_id,
                "name": by_skill[skill_id].name,
                "description": by_skill[skill_id].description,
                "capability_summary": interfaces[skill_id].capability_summary if skill_id in interfaces else "",
                "when_to_use": interfaces[skill_id].when_to_use if skill_id in interfaces else "",
                "requires": _interface_field_records(interfaces[skill_id].requires) if skill_id in interfaces else [],
                "produces": _interface_field_records(interfaces[skill_id].produces) if skill_id in interfaces else [],
                "uses_tools": _interface_field_records(interfaces[skill_id].uses_tools[:8]) if skill_id in interfaces else [],
            }
            for skill_id in sorted(member_ids)
            if skill_id in by_skill
        ],
        "top_internal_edges": [edge.to_dict() for edge in internal_edges],
        "selected_boundary_edges": [edge.to_dict() for edge in boundary_edges],
        "projection_edge_evidence": [
            _projection_edge_record(edge)
            for edge in internal_edges
            if edge.type in {"similar_to", "compose_with"}
        ],
        "cohesion_score": community.cohesion_score,
    }


def _edge_sort_key(edge: Edge) -> tuple[float, float, str, str, str]:
    usable_weight = max(float(edge.weight or 0.0), float(edge.confidence or 0.0))
    return (-usable_weight, -float(edge.confidence or 0.0), edge.type, edge.source, edge.target)


def _projection_edge_record(edge: Edge) -> dict[str, Any]:
    projection_weight = max(float(edge.weight or 0.0), float(edge.confidence or 0.0))
    if edge.type == "compose_with":
        projection_weight *= 0.35
    return {
        "source": edge.source,
        "target": edge.target,
        "edge_type": edge.type,
        "projection_weight": round(projection_weight, 4),
        "membership_role": "weak" if edge.type == "compose_with" else "strong",
        "reason": edge.reason,
    }


def _interface_field_records(fields: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(field.name),
            "kind": str(field.kind),
            "confidence": round(float(field.confidence), 4),
        }
        for field in fields[:10]
    ]


def _community_from_refinement(
    community: CommunityNode,
    raw: dict[str, Any],
    *,
    model_id: str,
    member_ids: list[str],
) -> CommunityNode:
    member_set = set(member_ids)
    representatives = [
        item
        for item in _string_list(raw.get("representative_skill_ids", community.representative_skill_ids))
        if not member_set or item in member_set
    ] or list(community.representative_skill_ids)
    return replace(
        community,
        name=str(raw.get("name") or community.name),
        summary=str(raw.get("summary") or community.summary),
        representative_skill_ids=representatives[:5],
        task_patterns=_string_list(raw.get("task_patterns", []))[:8],
        summary_provenance="llm_refined" if model_id != "deterministic-community" else "deterministic_fallback",
        model_id=model_id,
    )


def _normalize_refinement(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(raw.get("name", "")).strip(),
        "summary": str(raw.get("summary", "")).strip(),
        "task_patterns": _string_list(raw.get("task_patterns", [])),
        "representative_skill_ids": _string_list(raw.get("representative_skill_ids", [])),
    }
