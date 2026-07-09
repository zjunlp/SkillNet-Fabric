"""Relation utilities retained by the compiled graph build."""

from skillfabric.compiled_graph.relations.edge_safety import (
    EdgeSafetyResult,
    enforce_depend_on_acyclicity,
)
from skillfabric.compiled_graph.relations.similarity import build_similar_edges

__all__ = [
    "EdgeSafetyResult",
    "build_similar_edges",
    "enforce_depend_on_acyclicity",
]
