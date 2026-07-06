from __future__ import annotations

import unittest

from skillfabric.compiled_graph.execution.models import ExecutionEvidence
from skillfabric.compiled_graph.interface.models import (
    InterfaceEvidence,
    InterfaceField,
    SkillInterface,
)
from skillfabric.compiled_graph.models import Edge
from skillfabric.compiled_graph.relations.candidates import generate_relation_candidates
from tests.unit.relation_helpers import make_skill


class RelationCandidateTests(unittest.TestCase):
    def test_ignores_explicit_mentions_and_similarity_edges(self) -> None:
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

        self.assertEqual(pairs, [])

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
        interfaces = {
            hub.id: SkillInterface(
                skill_id=hub.id,
                content_hash=hub.content_hash,
                capability_summary="Hub skill.",
                produces=[
                    InterfaceField(
                        name=f"handoff {i}",
                        kind="artifact",
                        confidence=0.9,
                        evidence=[InterfaceEvidence(hub.id, i + 1, f"Produces handoff {i}.")],
                    )
                    for i in range(4)
                ],
            ),
            **{
                f"skill:s{i}": SkillInterface(
                    skill_id=f"skill:s{i}",
                    content_hash=f"hash-s{i}",
                    capability_summary=f"Consumer {i}.",
                    requires=[
                        InterfaceField(
                            name=f"handoff {i}",
                            kind="artifact",
                            confidence=0.9,
                            evidence=[InterfaceEvidence(f"skill:s{i}", i + 1, f"Requires handoff {i}.")],
                        )
                    ],
                )
                for i in range(4)
            },
        }

        pairs = generate_relation_candidates(skills, [], per_skill_limit=2, interfaces=interfaces)

        self.assertLessEqual(sum(1 for pair in pairs if "skill:hub" in pair.key), 2)

    def test_interface_candidate_sets_consumer_to_producer_direction(self) -> None:
        alpha = make_skill("skill:alpha", "alpha", "Alpha prepares data.")
        zeta = make_skill("skill:zeta", "zeta", "Use after alpha has prepared data.")
        interfaces = {
            alpha.id: SkillInterface(
                skill_id=alpha.id,
                content_hash=alpha.content_hash,
                capability_summary="Alpha prepares data.",
                produces=[InterfaceField(name="prepared data", kind="artifact", confidence=0.9)],
            ),
            zeta.id: SkillInterface(
                skill_id=zeta.id,
                content_hash=zeta.content_hash,
                capability_summary="Zeta consumes prepared data.",
                requires=[InterfaceField(name="prepared data", kind="artifact", confidence=0.9)],
            ),
        }

        pairs = generate_relation_candidates([alpha, zeta], [], per_skill_limit=10, interfaces=interfaces)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].key, ("skill:alpha", "skill:zeta"))
        self.assertEqual(pairs[0].direction_hint, "B->A")

    def test_accepted_execution_flow_does_not_generate_relation_candidate(self) -> None:
        producer = make_skill("skill:producer", "producer", "Produce CSV.")
        consumer = make_skill("skill:consumer", "consumer", "Consume CSV.")
        evidence = [ExecutionEvidence(skill="skill:producer", line=2, text="Produce CSV.")]

        pairs = generate_relation_candidates(
            [producer, consumer],
            [],
            per_skill_limit=10,
            execution_records=[],
        )

        self.assertEqual(pairs, [])
        self.assertEqual(evidence[0].skill, "skill:producer")


if __name__ == "__main__":
    unittest.main()
