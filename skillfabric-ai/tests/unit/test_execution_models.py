from __future__ import annotations

import unittest

from skillfabric.compiled_graph.execution.models import (
    ArtifactNode,
    ExecutionEdge,
    ExecutionEvidence,
    ExecutionFlowCandidate,
    ExecutionValidationRecord,
    ScenarioNode,
)


class ExecutionModelTests(unittest.TestCase):
    def test_execution_models_round_trip_stably(self) -> None:
        evidence = [ExecutionEvidence(skill="skill:producer", line=3, text="Write CSV output.")]
        artifact = ArtifactNode(
            id="artifact:csv",
            type="artifact",
            name="CSV",
            normalized_name="csv",
            kind="artifact",
            evidence=evidence,
        )
        scenario = ScenarioNode(
            id="scenario:authenticated-session",
            type="scenario",
            name="Authenticated session",
            normalized_name="authenticated session",
            kind="condition",
            evidence=evidence,
        )
        edge = ExecutionEdge(
            source="skill:producer",
            target="artifact:csv",
            type="produces_artifact",
            confidence=0.9,
            evidence=evidence,
            metadata={"artifact_id": "artifact:csv"},
        )
        candidate = ExecutionFlowCandidate(
            source_skill="skill:producer",
            target_skill="skill:consumer",
            flow_type="artifact_handoff",
            matched_node_id="artifact:csv",
            matched_name="csv",
            evidence=evidence,
        )
        record = ExecutionValidationRecord(
            candidate=candidate,
            raw_output={"accepted": True},
            normalized={"accepted": True, "confidence": 0.9},
            accepted=True,
            rejection_reason="",
            flow_edge=edge,
        )

        self.assertEqual(ArtifactNode.from_dict(artifact.to_dict()).to_dict(), artifact.to_dict())
        self.assertEqual(ScenarioNode.from_dict(scenario.to_dict()).to_dict(), scenario.to_dict())
        self.assertEqual(ExecutionEdge.from_dict(edge.to_dict()).to_dict(), edge.to_dict())
        self.assertEqual(ExecutionFlowCandidate.from_dict(candidate.to_dict()).to_dict(), candidate.to_dict())
        self.assertEqual(record.to_record()["flow_edge"], edge.to_dict())


if __name__ == "__main__":
    unittest.main()
