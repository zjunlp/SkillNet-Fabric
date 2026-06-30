"""Build similar_to edges."""

from __future__ import annotations

from skillfabric.compiled_graph.models import Edge
from skillfabric.indexing.neighbors import NeighborScore


def build_similar_edges(neighbor_scores: list[NeighborScore]) -> list[Edge]:
    """Build similar_to edges from hybrid neighbor scores."""

    edges: list[Edge] = []
    for score in neighbor_scores:
        edges.append(
            Edge(
                source=score.source,
                target=score.target,
                type="similar_to",
                confidence=score.score,
                weight=score.score,
                provenance="computed",
                reason=(
                    "hybrid similarity "
                    f"embedding={score.embedding_score}, "
                    f"bm25_overlap={score.bm25_overlap_score}, "
                    f"lexical={score.lexical_score}"
                ),
            )
        )
    return edges
