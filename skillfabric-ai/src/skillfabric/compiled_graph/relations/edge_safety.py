"""Canonical edge safety checks applied before graph serialization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from skillfabric.compiled_graph.models import Edge


@dataclass(slots=True)
class EdgeSafetyResult:
    """Result of applying deterministic edge safety gates."""

    edges: list[Edge]
    removed_depend_on_cycle_edges: list[Edge] = field(default_factory=list)
    depend_on_cycles: list[list[str]] = field(default_factory=list)


def enforce_depend_on_acyclicity(edges: Iterable[Edge]) -> EdgeSafetyResult:
    """Remove the weakest cycle-closing ``depend_on`` edges.

    ``depend_on`` edges define handoff ordering constraints. A directed cycle is
    therefore unusable for execution-package construction, even when each pairwise edge
    has local evidence. This gate keeps the strongest evidence in each cycle and
    removes the weakest edge deterministically.
    """

    remaining = list(edges)
    removed: list[Edge] = []
    cycles: list[list[str]] = []
    while True:
        cycle = _find_first_depend_on_cycle(remaining)
        if cycle is None:
            break
        edge = _weakest_cycle_edge(cycle, remaining)
        if edge is None:
            break
        cycles.append(cycle)
        remaining = [item for item in remaining if item is not edge]
        removed.append(edge)
    return EdgeSafetyResult(
        edges=sorted(remaining, key=lambda item: (item.type, item.source, item.target)),
        removed_depend_on_cycle_edges=removed,
        depend_on_cycles=cycles,
    )


def _find_first_depend_on_cycle(edges: list[Edge]) -> list[str] | None:
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        if edge.type != "depend_on":
            continue
        adjacency.setdefault(edge.source, []).append(edge.target)
    for neighbors in adjacency.values():
        neighbors.sort()

    visiting: list[str] = []
    visited: set[str] = set()

    def visit(node: str) -> list[str] | None:
        if node in visiting:
            return visiting[visiting.index(node) :] + [node]
        if node in visited:
            return None
        visiting.append(node)
        for neighbor in adjacency.get(node, []):
            cycle = visit(neighbor)
            if cycle is not None:
                return cycle
        visiting.pop()
        visited.add(node)
        return None

    for node in sorted(adjacency):
        cycle = visit(node)
        if cycle is not None:
            return cycle
    return None


def _weakest_cycle_edge(cycle: list[str], edges: list[Edge]) -> Edge | None:
    edge_by_pair = {
        (edge.source, edge.target): edge
        for edge in edges
        if edge.type == "depend_on"
    }
    cycle_edges = [
        edge_by_pair[(source, target)]
        for source, target in zip(cycle, cycle[1:], strict=False)
        if (source, target) in edge_by_pair
    ]
    if not cycle_edges:
        return None
    return min(cycle_edges, key=_cycle_prune_rank)


def _cycle_prune_rank(edge: Edge) -> tuple[float, float, int, int, str, str]:
    return (
        edge.confidence,
        edge.weight,
        _provenance_strength(edge.provenance),
        len(edge.evidence),
        edge.source,
        edge.target,
    )


def _provenance_strength(provenance: str) -> int:
    strengths = {
        "computed": 1,
        "execution_projected": 2,
        "llm_validated": 3,
        "explicit_mention": 4,
    }
    return strengths.get(provenance, 2)
