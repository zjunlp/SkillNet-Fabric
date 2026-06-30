"""Final community orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.communities.clustering import cluster_communities
from skillfabric.compiled_graph.communities.providers import (
    CommunityRefinementProvider,
    DeterministicCommunityRefinementProvider,
)
from skillfabric.compiled_graph.communities.refinement import refine_communities
from skillfabric.compiled_graph.interface.models import SkillInterface
from skillfabric.compiled_graph.models import CommunityNode, Edge
from skillfabric.llm_jobs import LLMJobOptions
from skillfabric.registry.models import SkillNode


def assign_final_communities(
    skills: list[SkillNode],
    *,
    provider: CommunityRefinementProvider | None = None,
    refinement_cache_path: str | Path | None = None,
    similar_edges: list[Edge] | None = None,
    relation_edges: list[Edge] | None = None,
    interfaces: dict[str, SkillInterface] | None = None,
    job_options: LLMJobOptions | None = None,
) -> tuple[list[CommunityNode], list[Edge], dict[str, str], dict[str, Any]]:
    """Return graph-clustered final communities and member_of edges."""

    provider = provider or DeterministicCommunityRefinementProvider()
    communities, member_edges, membership, stats = cluster_communities(
        skills,
        similar_edges or [],
        relation_edges or [],
    )
    context_edges = [*(similar_edges or []), *(relation_edges or [])]
    refined = refine_communities(
        communities,
        skills,
        context_edges,
        membership,
        provider=provider,
        cache_path=refinement_cache_path,
        interfaces=interfaces,
        job_options=job_options,
    )
    stats = {
        **stats,
        "community_refinement_model_id": provider.model_id,
    }
    return refined, member_edges, membership, stats
