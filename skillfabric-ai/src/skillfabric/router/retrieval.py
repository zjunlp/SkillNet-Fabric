"""BM25 and contract-embedding reciprocal-rank retrieval."""

from __future__ import annotations

import math
from pathlib import Path

from skillfabric.indexing.bm25 import search_bm25
from skillfabric.indexing.embeddings import (
    EmbeddingProvider,
    cosine_similarity,
    embed_query,
    embedding_provider_for_model,
    load_skill_embedding_store,
)
from skillfabric.indexing.ranking import reciprocal_rank_fusion
from skillfabric.registry.models import SkillNode
from skillfabric.router.models import RouterSkillCandidate
from skillfabric.storage import Workspace


class RouterRetrievalError(RuntimeError):
    """Raised when a configured retrieval channel cannot run correctly."""


def retrieve_seed_candidates(
    workspace: Workspace,
    query: str,
    skills: dict[str, SkillNode],
    *,
    limit: int,
    env_file: str | Path | None,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[RouterSkillCandidate]:
    """Retrieve the configured seed budget from two independent channels."""

    if not query.strip():
        raise ValueError("route query must not be empty")
    if limit <= 0:
        return []
    bm25_hits = search_bm25(
        workspace.graph_dir / "bm25.sqlite",
        query,
        limit=len(skills),
    )
    store = load_skill_embedding_store(workspace.graph_dir / "embeddings.json")
    if set(store.vectors) != set(skills):
        raise RouterRetrievalError(
            "skill embedding ids differ from the registry; rebuild the workspace"
        )
    provider = embedding_provider
    if provider is None:
        provider = embedding_provider_for_model(
            store.model_id,
            dimension=store.dimension,
            env_path=env_file,
        )
    query_vector = embed_query(provider, query)
    if len(query_vector) != store.dimension:
        raise RouterRetrievalError(
            f"query embedding dimension {len(query_vector)} does not match store dimension {store.dimension}"
        )
    if not query_vector or any(not math.isfinite(value) for value in query_vector):
        raise RouterRetrievalError("query embedding must be a non-empty finite vector")
    if math.sqrt(sum(value * value for value in query_vector)) <= 0:
        raise RouterRetrievalError("query embedding must have non-zero norm")
    embedding_order = [
        skill_id
        for skill_id, _score in sorted(
            (
                (skill_id, cosine_similarity(query_vector, vector))
                for skill_id, vector in store.vectors.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    fused = reciprocal_rank_fusion(
        {
            "bm25": [hit.skill_id for hit in bm25_hits],
            "embedding": embedding_order,
        }
    )
    seeds: list[RouterSkillCandidate] = []
    for row in fused[:limit]:
        skill = skills.get(row.skill_id)
        if skill is None:
            raise RouterRetrievalError(f"retrieval returned unknown skill id: {row.skill_id}")
        seeds.append(
            RouterSkillCandidate(
                skill_id=skill.id,
                name=skill.name,
                score=row.score,
                is_seed=True,
                retrieval_ranks=row.ranks,
            )
        )
    return seeds
