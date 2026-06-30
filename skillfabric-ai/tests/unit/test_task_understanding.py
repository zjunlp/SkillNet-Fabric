from __future__ import annotations

import unittest

from skillfabric.task_understanding import (
    analyze_task,
    coverage_diagnostics,
    filter_task_understanding_skills,
)

PENGUIN_QUERY = """I have collected observational data at artifacts/penguins.csv.
Please perform statistical analysis, generate at least 4 publication-quality PNG figures,
write report.docx, and create academic slides as presentation.pptx."""


class TaskUnderstandingTests(unittest.TestCase):
    def test_extracts_inputs_deliverables_and_analysis_intents(self) -> None:
        understanding = analyze_task(PENGUIN_QUERY)
        payload = understanding.to_dict()

        self.assertEqual(payload["input_artifacts"][0]["path"], "artifacts/penguins.csv")
        formats = {item["format"] for item in payload["required_deliverables"]}
        self.assertIn("docx", formats)
        self.assertIn("pptx", formats)
        self.assertIn("png", formats)
        png = next(item for item in payload["required_deliverables"] if item["format"] == "png")
        self.assertEqual(png["minimum_count"], 4)
        self.assertIn("statistical_analysis", payload["analysis_intents"])

        requirement_ids = {item["id"] for item in payload["coverage_requirements"]}
        self.assertIn("deliverable:docx", requirement_ids)
        self.assertIn("deliverable:pptx", requirement_ids)
        self.assertIn("deliverable:png", requirement_ids)
        self.assertIn("intent:tabular_or_statistical_analysis", requirement_ids)
        tabular = next(
            item
            for item in payload["coverage_requirements"]
            if item["id"] == "intent:tabular_or_statistical_analysis"
        )
        self.assertTrue(tabular["requires_preferred"])

    def test_extracts_direct_object_input_artifact(self) -> None:
        understanding = analyze_task(
            "Analyze artifacts/penguins.csv with statistical tests, generate at least 4 PNG figures, "
            "write report.docx, and create presentation.pptx."
        )
        payload = understanding.to_dict()

        self.assertEqual(payload["input_artifacts"][0]["path"], "artifacts/penguins.csv")
        self.assertEqual(payload["input_artifacts"][0]["format"], "csv")

    def test_separates_input_markdown_from_output_document_paths(self) -> None:
        understanding = analyze_task("Read notes.md, create report.docx, and produce presentation.pptx.")
        payload = understanding.to_dict()

        self.assertEqual(
            payload["input_artifacts"],
            [{"path": "notes.md", "format": "md", "source_text": "notes.md"}],
        )
        deliverables_by_format = {
            item["format"]: item
            for item in payload["required_deliverables"]
        }
        self.assertEqual(deliverables_by_format["docx"]["path"], "report.docx")
        self.assertEqual(deliverables_by_format["pptx"]["path"], "presentation.pptx")

    def test_extracts_financial_statement_analysis_intent(self) -> None:
        understanding = analyze_task(
            "Analyze a company's financial statements, extract key financial KPIs, "
            "compare year-over-year trends, generate charts, and produce an executive summary report."
        )
        payload = understanding.to_dict()

        self.assertIn("financial_statement_analysis", payload["analysis_intents"])
        requirement = next(
            item
            for item in payload["coverage_requirements"]
            if item["id"] == "intent:financial_statement_analysis"
        )
        self.assertEqual(requirement["preferred_skill_ids"], [])
        self.assertEqual(requirement["acceptable_skill_ids"], [])

    def test_filter_keeps_resolved_coverage_skills_to_available_registry(self) -> None:
        understanding = analyze_task(
            "Analyze financial statements, extract KPIs, compare trends, and generate charts."
        )
        financial = next(
            item
            for item in understanding.coverage_requirements
            if item.id == "intent:financial_statement_analysis"
        )
        financial.preferred_skill_ids = ["skill:analyzing-financial-statements", "skill:not-installed"]
        financial.acceptable_skill_ids = [
            "skill:analyzing-financial-statements",
            "skill:creating-financial-models",
        ]

        filtered = filter_task_understanding_skills(
            understanding,
            {"skill:analyzing-financial-statements"},
        )
        requirement = next(
            item
            for item in filtered.to_dict()["coverage_requirements"]
            if item["id"] == "intent:financial_statement_analysis"
        )

        self.assertEqual(requirement["preferred_skill_ids"], ["skill:analyzing-financial-statements"])
        self.assertEqual(requirement["acceptable_skill_ids"], ["skill:analyzing-financial-statements"])

    def test_analyze_task_does_not_emit_concrete_skill_ids(self) -> None:
        understanding = analyze_task(
            "Analyze artifacts/penguins.csv with statistical tests and generate PNG figures."
        )

        for requirement in understanding.coverage_requirements:
            self.assertEqual(requirement.preferred_skill_ids, [])
            self.assertEqual(requirement.acceptable_skill_ids, [])

    def test_tabular_statistical_intent_requires_preferred_analysis_skill_after_resolution(self) -> None:
        understanding = analyze_task(
            "Analyze artifacts/penguins.csv with statistical tests and generate PNG figures."
        )
        tabular = next(
            item
            for item in understanding.coverage_requirements
            if item.id == "intent:tabular_or_statistical_analysis"
        )
        tabular.preferred_skill_ids = ["skill:xlsx"]
        tabular.acceptable_skill_ids = ["skill:xlsx", "skill:data-visualization"]

        visualization_only = coverage_diagnostics(
            understanding,
            {"skill:data-visualization"},
        )
        with_xlsx = coverage_diagnostics(
            understanding,
            {"skill:data-visualization", "skill:xlsx"},
        )

        missing_ids = {item["id"] for item in visualization_only["missing"]}
        self.assertIn("intent:tabular_or_statistical_analysis", missing_ids)
        tabular_covered = next(
            item
            for item in with_xlsx["covered"]
            if item["id"] == "intent:tabular_or_statistical_analysis"
        )
        self.assertEqual(tabular_covered["covered_by"], ["skill:xlsx"])


if __name__ == "__main__":
    unittest.main()
