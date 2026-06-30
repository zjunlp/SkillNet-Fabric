"""Unified neighbor scoring for retrieval."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from skillfabric.indexing.canonical import canonical_skill_text
from skillfabric.indexing.embeddings import cosine_similarity
from skillfabric.registry.models import SkillNode


@dataclass(slots=True)
class NeighborScore:
    """Hybrid similarity details for a pair of skills."""

    source: str
    target: str
    score: float
    embedding_score: float
    bm25_overlap_score: float
    lexical_score: float
    sources: list[str] = field(default_factory=list)


def build_neighbor_scores(
    skills: list[SkillNode],
    embeddings: dict[str, list[float]],
    *,
    top_k: int = 5,
    bm25_neighbors: dict[str, dict[str, float]] | None = None,
) -> list[NeighborScore]:
    """Compute top-k similar neighbors for each skill."""

    bm25_neighbors = bm25_neighbors or {}
    token_sets = {skill.id: set(_tokens(canonical_skill_text(skill))) for skill in skills}
    title_sets = {
        skill.id: set(_tokens(f"{skill.name} {skill.description}"))
        for skill in skills
    }
    by_id = {skill.id: skill for skill in skills}
    results: list[NeighborScore] = []
    for source in skills:
        scored: list[NeighborScore] = []
        for target in skills:
            if source.id == target.id:
                continue
            embedding_score = max(
                cosine_similarity(embeddings.get(source.id, []), embeddings.get(target.id, [])),
                0.0,
            )
            bm25_overlap = bm25_neighbors.get(source.id, {}).get(target.id)
            if bm25_overlap is None:
                bm25_overlap = _jaccard(token_sets[source.id], token_sets[target.id])
            lexical = _jaccard(title_sets[source.id], title_sets[target.id])
            score = 0.55 * embedding_score + 0.25 * bm25_overlap + 0.20 * lexical
            sources = []
            if embedding_score > 0:
                sources.append("embedding_knn")
            if bm25_overlap > 0:
                sources.append("bm25_neighbor_overlap")
            if lexical > 0:
                sources.append("title_description_lexical_overlap")
            if score > 0 and target.id in by_id:
                scored.append(
                    NeighborScore(
                        source=source.id,
                        target=target.id,
                        score=round(score, 6),
                        embedding_score=round(embedding_score, 6),
                        bm25_overlap_score=round(bm25_overlap, 6),
                        lexical_score=round(lexical, 6),
                        sources=sources,
                    )
                )
        scored.sort(key=lambda item: (-item.score, item.target))
        results.extend(scored[: max(top_k, 0)])
    return results


def lexical_overlap(left: SkillNode, right: SkillNode) -> float:
    """Compute title and description lexical overlap."""

    return _jaccard(
        set(_tokens(f"{left.name} {left.description}")),
        set(_tokens(f"{right.name} {right.description}")),
    )


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_.+-]*", text.lower())


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if intersection == 0:
        return 0.0
    return intersection / math.sqrt(len(left) * len(right))
