from __future__ import annotations

import unittest

from skillfabric.compiled_graph.interface.health import (
    analyze_interface_health,
    render_interface_health_report,
)
from skillfabric.compiled_graph.interface.models import InterfaceField, SkillInterface


class InterfaceHealthTests(unittest.TestCase):
    def test_detects_missing_summary_empty_contract_and_missing_evidence(self) -> None:
        interface = SkillInterface(
            skill_id="skill:a",
            content_hash="hash-a",
            capability_summary="",
            produces=[InterfaceField(name="csv", kind="artifact", confidence=0.2)],
            model_id="model",
        )

        report = analyze_interface_health([interface])
        rendered = render_interface_health_report(report)

        self.assertIn("skill:a", report.missing_summary)
        self.assertIn("skill:a", report.empty_requires)
        self.assertTrue(report.fields_missing_evidence)
        self.assertTrue(report.low_confidence_fields)
        self.assertIn("Interface Health Report", rendered)
        self.assertIn("empty requires", rendered)


if __name__ == "__main__":
    unittest.main()
