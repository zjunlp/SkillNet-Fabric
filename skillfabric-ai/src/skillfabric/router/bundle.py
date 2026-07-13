"""Build bounded query-local evidence bundles for the explorer."""

from __future__ import annotations

from skillfabric.indexing.embeddings import EmbeddingProvider
from skillfabric.router.expansion import expand_semantic_candidates
from skillfabric.router.models import RouterBundle, RouterBundleConfig, RouterSkillCandidate
from skillfabric.router.retrieval import retrieve_seed_candidates
from skillfabric.storage import Workspace
from skillfabric.wiki.loader import load_wiki_source


def build_router_bundle(
    config: RouterBundleConfig,
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> RouterBundle:
    """Retrieve seeds and add bounded operational graph context."""

    workspace = Workspace(config.workspace)
    source = load_wiki_source(workspace)
    seeds = retrieve_seed_candidates(
        workspace,
        config.query,
        source.skills,
        limit=config.seed_limit,
        env_file=config.env_file,
        embedding_provider=embedding_provider,
    )
    expanded = expand_semantic_candidates(
        seeds,
        source.core_edges,
        source.skills,
        max_depth=config.max_depth,
        limit=config.expanded_limit,
    )
    context_ids = {candidate.skill_id for candidate in expanded.candidates}
    context_ids.update(alternative.skill_id for alternative in expanded.alternatives)
    graph_edges = tuple(
        sorted(
            (
                edge
                for edge in source.operational_edges
                if edge.source in context_ids and edge.target in context_ids
            ),
            key=lambda edge: (edge.type, edge.source, edge.target),
        )
    )
    return RouterBundle(
        query=config.query,
        selected_skills=expanded.candidates,
        graph_edges=graph_edges,
        alternatives=expanded.alternatives,
    )


__all__ = [
    "RouterBundle",
    "RouterBundleConfig",
    "RouterSkillCandidate",
    "build_router_bundle",
]
