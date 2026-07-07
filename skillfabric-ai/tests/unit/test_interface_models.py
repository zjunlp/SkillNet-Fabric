from __future__ import annotations

import unittest

from skillfabric.compiled_graph.interface.models import (
    InterfaceEvidence,
    InterfaceField,
    SkillInterface,
)


class InterfaceModelTests(unittest.TestCase):
    def test_interface_field_normalizes_kind_format_only(self) -> None:
        field = InterfaceField(
            name="object_in_inventory",
            kind="World State",
            confidence=0.7,
        )

        self.assertEqual(field.kind, "world_state")

    def test_interface_field_kind_is_closed_set(self) -> None:
        credential = InterfaceField(name="authenticated oauth session", kind="credential", confidence=0.7)

        self.assertEqual(credential.kind, "credential")
        for alias in ("state", "condition", "dependency", "observation_state"):
            with self.subTest(alias=alias):
                with self.assertRaises(ValueError):
                    InterfaceField(name=f"{alias} field", kind=alias, confidence=0.7)

    def test_common_kind_aliases_are_not_semantically_normalized(self) -> None:
        for alias in ("dataset", "table", "library", "api", "command"):
            with self.subTest(alias=alias):
                with self.assertRaises(ValueError):
                    InterfaceField(name=f"{alias} field", kind=alias, confidence=0.7)

    def test_skill_interface_round_trips_stably(self) -> None:
        interface = SkillInterface(
            skill_id="skill:pdf-table-parser",
            content_hash="hash-pdf",
            capability_summary="Extract PDF tables.",
            when_to_use="Use when tabular data must be extracted from PDFs.",
            produces=[
                InterfaceField(
                    name="csv tables",
                    description="Structured CSV tables.",
                    kind="artifact",
                    confidence=0.91,
                    evidence=[InterfaceEvidence("skill:pdf-table-parser", 8, "Write CSV tables.")],
                )
            ],
            provenance="llm_extracted",
            model_id="openai/test-model",
        )

        payload = interface.to_dict()
        restored = SkillInterface.from_dict(payload)

        self.assertEqual(restored.to_dict(), payload)
        self.assertEqual(restored.produces[0].evidence[0].line, 8)
        self.assertNotIn("inferred", payload["produces"][0])
        self.assertNotIn("granularity", payload)
        self.assertNotIn("execution_role", payload)
        self.assertNotIn("failure_modes", payload)
        self.assertNotIn("inputs", payload)
        self.assertNotIn("outputs", payload)
        self.assertNotIn("preconditions", payload)
        self.assertNotIn("postconditions", payload)
        self.assertNotIn("artifacts", payload)
        self.assertNotIn("tools", payload)
        self.assertNotIn("raw_output", payload)


if __name__ == "__main__":
    unittest.main()
