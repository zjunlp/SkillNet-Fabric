from __future__ import annotations

import unittest

from skillfabric.compiled_graph.health import analyze_health
from skillfabric.compiled_graph.models import Edge, EvidenceRef, GraphDocument
from skillfabric.compiled_graph.relations.edge_safety import enforce_depend_on_acyclicity


def _edge(
    source: str,
    target: str,
    confidence: float,
    *,
    weight: float | None = None,
    provenance: str = "llm_validated",
) -> Edge:
    return Edge(
        source=source,
        target=target,
        type="depend_on",
        confidence=confidence,
        weight=confidence if weight is None else weight,
        provenance=provenance,
        evidence=[EvidenceRef(skill=source, line=1, text=f"{source} depends on {target}.")],
    )


class EdgeSafetyTests(unittest.TestCase):
    def test_prunes_weaker_reciprocal_dependency(self) -> None:
        strong = _edge(
            "skill:presentation",
            "skill:converter",
            0.96,
            provenance="explicit_mention",
        )
        weak = _edge(
            "skill:converter",
            "skill:presentation",
            0.93,
            weight=0.837,
            provenance="execution_projected",
        )

        result = enforce_depend_on_acyclicity([weak, strong])

        self.assertEqual(result.removed_depend_on_cycle_edges, [weak])
        self.assertIn(strong, result.edges)
        self.assertNotIn(weak, result.edges)
        self.assertEqual(result.depend_on_cycles, [["skill:converter", "skill:presentation", "skill:converter"]])

    def test_prunes_weakest_edge_from_longer_dependency_cycle(self) -> None:
        a_to_b = _edge("skill:a", "skill:b", 0.97)
        b_to_c = _edge("skill:b", "skill:c", 0.92)
        c_to_a = _edge("skill:c", "skill:a", 0.88)
        similar = Edge("skill:a", "skill:c", "similar_to", confidence=0.7, weight=0.7)

        result = enforce_depend_on_acyclicity([a_to_b, b_to_c, c_to_a, similar])
        graph = GraphDocument(
            schema_version="1.0",
            build_id="edge-safety-test",
            nodes=[],
            edges=result.edges,
            stats={},
            config_digest="x",
        )

        self.assertEqual(result.removed_depend_on_cycle_edges, [c_to_a])
        self.assertIn(similar, result.edges)
        self.assertFalse(analyze_health(graph, communities=[]).depend_on_cycles)

    def test_deterministic_accept_has_strong_cycle_rank(self) -> None:
        deterministic = _edge(
            "skill:a",
            "skill:b",
            0.92,
            provenance="deterministic_accept",
        )
        projected = _edge(
            "skill:b",
            "skill:a",
            0.92,
            provenance="execution_projected",
        )

        result = enforce_depend_on_acyclicity([deterministic, projected])

        self.assertIn(deterministic, result.edges)
        self.assertNotIn(projected, result.edges)


if __name__ == "__main__":
    unittest.main()
