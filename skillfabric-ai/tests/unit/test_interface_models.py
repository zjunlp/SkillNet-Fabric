from __future__ import annotations

import unittest

from skillfabric.compiled_graph.interface.models import (
    InterfaceEvidence,
    InterfaceField,
    SkillInterface,
)


class InterfaceModelTests(unittest.TestCase):
    def test_interface_field_normalizes_state_semantics_by_name(self) -> None:
        belief = InterfaceField(
            name="object_permanence_state",
            kind="state",
            confidence=0.7,
        )
        planning = InterfaceField(
            name="sequential_sub_objective_plan",
            kind="state",
            confidence=0.7,
        )
        inventory = InterfaceField(
            name="object_in_inventory",
            kind="state",
            confidence=0.7,
        )
        verified = InterfaceField(
            name="task_verified",
            kind="state",
            confidence=0.7,
        )

        self.assertEqual(belief.kind, "belief_state")
        self.assertEqual(planning.kind, "planning_state")
        self.assertEqual(inventory.kind, "world_state")
        self.assertEqual(verified.kind, "world_state")

    def test_skill_interface_round_trips_stably(self) -> None:
        interface = SkillInterface(
            skill_id="skill:pdf-table-parser",
            content_hash="hash-pdf",
            capability_summary="Extract PDF tables.",
            when_to_use="Use when tabular data must be extracted from PDFs.",
            granularity="utility",
            execution_role="inspector",
            produces=[
                InterfaceField(
                    name="csv tables",
                    description="Structured CSV tables.",
                    kind="artifact",
                    confidence=0.91,
                    inferred=False,
                    evidence=[InterfaceEvidence("skill:pdf-table-parser", 8, "Write CSV tables.")],
                )
            ],
            provenance="llm_extracted",
            model_id="openai/test-model",
        )

        payload = interface.to_dict()
        restored = SkillInterface.from_dict(payload)

        self.assertEqual(restored.to_dict(), payload)
        self.assertEqual(restored.granularity, "utility")
        self.assertEqual(restored.execution_role, "inspector")
        self.assertEqual(restored.produces[0].evidence[0].line, 8)
        self.assertNotIn("inputs", payload)
        self.assertNotIn("outputs", payload)
        self.assertNotIn("preconditions", payload)
        self.assertNotIn("postconditions", payload)
        self.assertNotIn("artifacts", payload)
        self.assertNotIn("tools", payload)
        self.assertNotIn("raw_output", payload)

    def test_skill_interface_normalizes_granularity_and_execution_role(self) -> None:
        planning = SkillInterface(
            skill_id="skill:goal-interpreter",
            content_hash="hash-goal",
            capability_summary="Parse goals into plans.",
            granularity="plan",
            execution_role="planning",
        )
        primitive = SkillInterface(
            skill_id="skill:object-picker",
            content_hash="hash-picker",
            capability_summary="Take an object.",
            granularity="action",
            execution_role="take",
        )

        self.assertEqual(planning.granularity, "planning")
        self.assertEqual(planning.execution_role, "planner")
        self.assertEqual(primitive.granularity, "primitive")
        self.assertEqual(primitive.execution_role, "actor")


if __name__ == "__main__":
    unittest.main()
