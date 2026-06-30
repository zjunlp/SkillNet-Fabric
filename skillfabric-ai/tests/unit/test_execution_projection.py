from __future__ import annotations

import unittest

from skillfabric.compiled_graph.execution.models import (
    ExecutionEdge,
    ExecutionEvidence,
    ExecutionFlowCandidate,
    ExecutionValidationRecord,
)
from skillfabric.compiled_graph.execution.projection import project_execution_records
from skillfabric.compiled_graph.models import Edge, EvidenceRef


def _record(projected_edge_type: str = "depend_on") -> ExecutionValidationRecord:
    evidence = [ExecutionEvidence(skill="skill:producer", line=2, text="Produce csv.")]
    candidate = ExecutionFlowCandidate(
        source_skill="skill:producer",
        target_skill="skill:consumer",
        flow_type="artifact_flow",
        matched_node_id="artifact:csv",
        matched_name="csv",
        evidence=evidence,
    )
    return ExecutionValidationRecord(
        candidate=candidate,
        raw_output={},
        normalized={
            "accepted": True,
            "flow_type": "artifact_flow",
            "projected_edge_type": projected_edge_type,
            "confidence": 0.91,
            "evidence": [item.to_dict() for item in evidence],
            "reason": "Consumer needs producer output.",
        },
        accepted=True,
        rejection_reason="",
        flow_edge=ExecutionEdge(
            source="skill:producer",
            target="skill:consumer",
            type="artifact_flow",
            confidence=0.91,
            evidence=evidence,
            reason="Consumer needs producer output.",
            metadata={"artifact_id": "artifact:csv"},
        ),
    )


class ExecutionProjectionTests(unittest.TestCase):
    def test_accepted_artifact_flow_projects_to_depend_on(self) -> None:
        edges = project_execution_records([_record()], [])

        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].source, "skill:consumer")
        self.assertEqual(edges[0].target, "skill:producer")
        self.assertEqual(edges[0].type, "depend_on")
        self.assertEqual(edges[0].provenance, "execution_projected")

    def test_compose_with_projection_uses_stable_order(self) -> None:
        edges = project_execution_records([_record("compose_with")], [])

        self.assertEqual(edges[0].type, "compose_with")
        self.assertEqual((edges[0].source, edges[0].target), ("skill:consumer", "skill:producer"))

    def test_projection_merges_existing_edge_without_duplicates(self) -> None:
        existing = [
            Edge(
                source="skill:consumer",
                target="skill:producer",
                type="depend_on",
                confidence=0.7,
                weight=0.7,
                provenance="llm_validated",
                evidence=[EvidenceRef(skill="skill:consumer", line=4, text="Needs csv.")],
                reason="Existing reason.",
            )
        ]

        edges = project_execution_records([_record()], existing)

        self.assertEqual(len(edges), 1)
        self.assertGreater(edges[0].confidence, 0.7)
        self.assertEqual(len(edges[0].evidence), 2)
        self.assertIn("Consumer needs producer output.", edges[0].reason)


if __name__ == "__main__":
    unittest.main()
