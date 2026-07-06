from __future__ import annotations

import json
import unittest

from skillfabric.compiled_graph.execution.models import (
    ExecutionEdge,
    ExecutionEvidence,
    ExecutionFlowCandidate,
    ExecutionValidationRecord,
)
from skillfabric.compiled_graph.interface.models import (
    InterfaceEvidence,
    InterfaceField,
    SkillInterface,
)
from skillfabric.compiled_graph.relations.candidates import generate_relation_candidates
from skillfabric.compiled_graph.relations.prompts import build_pair_validation_messages
from tests.unit.relation_helpers import make_skill


class RelationInterfaceCandidateTests(unittest.TestCase):
    def test_output_input_overlap_generates_directed_interface_candidate(self) -> None:
        producer = make_skill("skill:producer", "producer", "Produce CSV.")
        consumer = make_skill("skill:consumer", "consumer", "Consume CSV.")
        interfaces = {
            producer.id: SkillInterface(
                skill_id=producer.id,
                content_hash=producer.content_hash,
                capability_summary="Produce CSV.",
                produces=[
                    InterfaceField(
                        name="csv",
                        kind="artifact",
                        confidence=0.9,
                        evidence=[InterfaceEvidence(producer.id, 1, "Produce CSV.")],
                    )
                ],
            ),
            consumer.id: SkillInterface(
                skill_id=consumer.id,
                content_hash=consumer.content_hash,
                capability_summary="Consume CSV.",
                requires=[
                    InterfaceField(
                        name="csv",
                        kind="artifact",
                        confidence=0.9,
                        evidence=[InterfaceEvidence(consumer.id, 1, "Consume CSV.")],
                    )
                ],
            ),
        }

        pairs = generate_relation_candidates([producer, consumer], [], per_skill_limit=10, interfaces=interfaces)

        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        self.assertIn("interface_compatibility", pair.sources)
        self.assertEqual(pair.key, ("skill:consumer", "skill:producer"))
        self.assertEqual(pair.direction_hint, "A->B")
        self.assertTrue(any(item.source == "interface_compatibility" for item in pair.evidence))

    def test_postcondition_precondition_overlap_generates_candidate(self) -> None:
        setup = make_skill("skill:setup", "setup", "Authenticate.")
        action = make_skill("skill:action", "action", "Requires authentication.")
        interfaces = {
            setup.id: SkillInterface(
                skill_id=setup.id,
                content_hash=setup.content_hash,
                capability_summary="Authenticate.",
                produces=[InterfaceField(name="authenticated session", kind="state", confidence=0.8)],
            ),
            action.id: SkillInterface(
                skill_id=action.id,
                content_hash=action.content_hash,
                capability_summary="Use authenticated session.",
                requires=[InterfaceField(name="authenticated session", kind="state", confidence=0.8)],
            ),
        }

        pairs = generate_relation_candidates([setup, action], [], per_skill_limit=10, interfaces=interfaces)

        self.assertEqual(len(pairs), 1)
        self.assertIn("interface_compatibility", pairs[0].sources)

    def test_shared_tool_does_not_generate_interface_candidate(self) -> None:
        left = make_skill("skill:left", "left", "Use Python.")
        right = make_skill("skill:right", "right", "Use Python.")
        interfaces = {
            left.id: SkillInterface(
                skill_id=left.id,
                content_hash=left.content_hash,
                capability_summary="Use Python.",
                uses_tools=[InterfaceField(name="python", kind="tool", confidence=0.8)],
            ),
            right.id: SkillInterface(
                skill_id=right.id,
                content_hash=right.content_hash,
                capability_summary="Use Python.",
                uses_tools=[InterfaceField(name="python", kind="tool", confidence=0.8)],
            ),
        }

        pairs = generate_relation_candidates([left, right], [], per_skill_limit=10, interfaces=interfaces)

        self.assertEqual(pairs, [])

    def test_validation_prompt_includes_interface_summary_and_evidence(self) -> None:
        producer = make_skill("skill:producer", "producer", "Produce CSV.")
        consumer = make_skill("skill:consumer", "consumer", "Consume CSV.")
        interfaces = {
            producer.id: SkillInterface(
                skill_id=producer.id,
                content_hash=producer.content_hash,
                capability_summary="Produces CSV artifacts.",
                produces=[InterfaceField(name="csv", kind="artifact", confidence=0.9)],
            ),
            consumer.id: SkillInterface(
                skill_id=consumer.id,
                content_hash=consumer.content_hash,
                capability_summary="Consumes CSV artifacts.",
                requires=[InterfaceField(name="csv", kind="artifact", confidence=0.9)],
            ),
        }
        pair = generate_relation_candidates([producer, consumer], [], per_skill_limit=10, interfaces=interfaces)[0]

        prompt = json.dumps(
            build_pair_validation_messages(producer, consumer, pair, interfaces=interfaces),
            ensure_ascii=False,
        )

        self.assertIn("Produces CSV artifacts.", prompt)
        self.assertIn("Consumes CSV artifacts.", prompt)
        self.assertIn("interface_compatibility", prompt)

    def test_execution_covered_pair_skips_relation_candidate(self) -> None:
        producer = make_skill("skill:producer", "producer", "Produce CSV.")
        consumer = make_skill("skill:consumer", "consumer", "Consume CSV.")
        evidence = [ExecutionEvidence(skill="skill:producer", line=2, text="Produce CSV.")]
        interfaces = {
            producer.id: SkillInterface(
                skill_id=producer.id,
                content_hash=producer.content_hash,
                capability_summary="Produces CSV.",
                produces=[InterfaceField(name="csv", kind="artifact", confidence=0.9)],
            ),
            consumer.id: SkillInterface(
                skill_id=consumer.id,
                content_hash=consumer.content_hash,
                capability_summary="Consumes CSV.",
                requires=[InterfaceField(name="csv", kind="artifact", confidence=0.9)],
            ),
        }
        execution_records = [
            ExecutionValidationRecord(
                candidate=ExecutionFlowCandidate(
                    source_skill="skill:producer",
                    target_skill="skill:consumer",
                    flow_type="artifact_flow",
                    matched_node_id="artifact:csv",
                    matched_name="csv",
                    evidence=evidence,
                ),
                raw_output={},
                normalized={
                    "accepted": True,
                    "flow_type": "artifact_flow",
                    "projected_edge_type": "depend_on",
                    "confidence": 0.9,
                    "evidence": [item.to_dict() for item in evidence],
                    "reason": "Consumer needs CSV.",
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
            interfaces=interfaces,
            execution_records=execution_records,
        )

        self.assertEqual(pairs, [])


if __name__ == "__main__":
    unittest.main()
