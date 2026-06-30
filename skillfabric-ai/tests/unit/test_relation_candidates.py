from __future__ import annotations

import unittest

from skillfabric.compiled_graph.execution.models import (
    ExecutionEdge,
    ExecutionEvidence,
    ExecutionFlowCandidate,
    ExecutionValidationRecord,
)
from skillfabric.compiled_graph.models import Edge
from skillfabric.compiled_graph.relations.candidates import generate_relation_candidates
from tests.unit.relation_helpers import make_skill


class RelationCandidateTests(unittest.TestCase):
    def test_merges_sources_evidence_and_direction_hints(self) -> None:
        parser = make_skill("skill:pdf-table-parser", "pdf-table-parser", "Parse PDF tables.")
        kpi = make_skill(
            "skill:financial-kpi-extractor",
            "financial-kpi-extractor",
            "Use after pdf-table-parser has produced CSV tables.",
            artifacts=["csv"],
        )
        similar = [
            Edge(
                source="skill:financial-kpi-extractor",
                target="skill:pdf-table-parser",
                type="similar_to",
                confidence=0.7,
                weight=0.7,
            )
        ]

        pairs = generate_relation_candidates(
            [parser, kpi],
            similar,
            per_skill_limit=10,
        )

        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        self.assertEqual(pair.key, ("skill:financial-kpi-extractor", "skill:pdf-table-parser"))
        self.assertIn("explicit_mention", pair.sources)
        self.assertIn("similar_neighbor", pair.sources)
        self.assertNotIn("same_community", pair.sources)
        self.assertEqual(pair.direction_hint, "A->B")
        self.assertTrue(pair.evidence)

    def test_does_not_generate_candidates_from_community_membership(self) -> None:
        left = make_skill("skill:left", "left", "Left skill.")
        right = make_skill("skill:right", "right", "Right skill.")

        pairs = generate_relation_candidates([left, right], [], per_skill_limit=10)

        self.assertEqual(pairs, [])

    def test_applies_per_skill_limit_stably(self) -> None:
        hub = make_skill("skill:hub", "hub", "Hub skill.")
        skills = [hub] + [
            make_skill(f"skill:s{i}", f"s{i}", "Use after hub.")
            for i in range(4)
        ]
        similar = [
            Edge(source="skill:hub", target=f"skill:s{i}", type="similar_to", confidence=0.9 - i * 0.1, weight=0.9 - i * 0.1)
            for i in range(4)
        ]

        pairs = generate_relation_candidates(skills, similar, per_skill_limit=2)

        self.assertLessEqual(sum(1 for pair in pairs if "skill:hub" in pair.key), 2)

    def test_direction_hint_is_flipped_when_pair_ids_are_sorted(self) -> None:
        alpha = make_skill("skill:alpha", "alpha", "Alpha prepares data.")
        zeta = make_skill("skill:zeta", "zeta", "Use after alpha has prepared data.")

        pairs = generate_relation_candidates([alpha, zeta], [], per_skill_limit=10)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].key, ("skill:alpha", "skill:zeta"))
        self.assertEqual(pairs[0].direction_hint, "B->A")

    def test_accepted_execution_flow_generates_relation_candidate(self) -> None:
        producer = make_skill("skill:producer", "producer", "Produce CSV.")
        consumer = make_skill("skill:consumer", "consumer", "Consume CSV.")
        evidence = [ExecutionEvidence(skill="skill:producer", line=2, text="Produce CSV.")]
        candidate = ExecutionFlowCandidate(
            source_skill="skill:producer",
            target_skill="skill:consumer",
            flow_type="artifact_flow",
            matched_node_id="artifact:csv",
            matched_name="csv",
            evidence=evidence,
        )
        records = [
            ExecutionValidationRecord(
                candidate=candidate,
                raw_output={},
                normalized={
                    "accepted": True,
                    "flow_type": "artifact_flow",
                    "projected_edge_type": "depend_on",
                    "confidence": 0.9,
                    "evidence": [item.to_dict() for item in evidence],
                    "reason": "Consumer needs producer output.",
                },
                accepted=True,
                rejection_reason="",
                flow_edge=ExecutionEdge(
                    source="skill:producer",
                    target="skill:consumer",
                    type="artifact_flow",
                    confidence=0.9,
                    evidence=evidence,
                    metadata={"artifact_id": "artifact:csv"},
                ),
            )
        ]

        pairs = generate_relation_candidates(
            [producer, consumer],
            [],
            per_skill_limit=10,
            execution_records=records,
        )

        self.assertEqual(len(pairs), 1)
        self.assertIn("execution_flow", pairs[0].sources)
        self.assertEqual(pairs[0].key, ("skill:consumer", "skill:producer"))
        self.assertEqual(pairs[0].direction_hint, "A->B")


if __name__ == "__main__":
    unittest.main()
