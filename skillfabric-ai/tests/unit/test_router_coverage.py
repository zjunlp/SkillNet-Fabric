from __future__ import annotations

import unittest

from skillfabric.compiled_graph.execution.models import ExecutionIndexRecord
from skillfabric.compiled_graph.interface.models import InterfaceField, SkillInterface
from skillfabric.registry.models import SkillNode
from skillfabric.router.coverage import resolve_coverage_requirements
from skillfabric.task_understanding import analyze_task


def _skill(
    skill_id: str,
    name: str,
    description: str,
) -> SkillNode:
    return SkillNode(
        id=skill_id,
        type="skill",
        name=name,
        description=description,
        source_path=f"/skills/{name}/SKILL.md",
        wiki_path="",
        content_hash=f"hash-{skill_id}",
        token_count=10,
        canonical_skill_text_hash=f"text-hash-{skill_id}",
        raw_text=description,
    )


def _interface(
    skill_id: str,
    *,
    produces: list[str] | None = None,
    requires: list[str] | None = None,
    summary: str = "",
    when_to_use: str = "",
    execution_role: str = "helper",
) -> SkillInterface:
    return SkillInterface(
        skill_id=skill_id,
        content_hash=f"hash-{skill_id}",
        capability_summary=summary,
        when_to_use=when_to_use,
        execution_role=execution_role,
        produces=[InterfaceField(name=item, kind="artifact", confidence=0.9) for item in produces or []],
        requires=[InterfaceField(name=item, kind="artifact", confidence=0.9) for item in requires or []],
    )


class RouterCoverageTests(unittest.TestCase):
    def test_deliverable_requirement_resolves_from_interface_produces_not_skill_id(self) -> None:
        understanding = analyze_task("Create an academic slide deck as presentation.pptx.")
        skills = {
            "skill:renamed-deck-maker": _skill(
                "skill:renamed-deck-maker",
                "renamed-deck-maker",
                "Create presentation deliverables.",
            )
        }
        interfaces = {
            "skill:renamed-deck-maker": _interface(
                "skill:renamed-deck-maker",
                produces=["presentation_document"],
                summary="Creates PowerPoint slide decks.",
            )
        }

        resolved = resolve_coverage_requirements(
            understanding,
            skills=skills,
            interfaces=interfaces,
            execution_index=[],
        )

        requirement = next(item for item in resolved.coverage_requirements if item.id == "deliverable:pptx")
        self.assertEqual(requirement.preferred_skill_ids, ["skill:renamed-deck-maker"])
        self.assertEqual(requirement.acceptable_skill_ids, ["skill:renamed-deck-maker"])
        self.assertEqual(resolved.coverage_diagnostics[0]["status"], "resolved")

    def test_missing_requirement_is_diagnostic_not_filled_with_nonexistent_skill(self) -> None:
        understanding = analyze_task("Write report.docx.")
        resolved = resolve_coverage_requirements(
            understanding,
            skills={},
            interfaces={},
            execution_index=[],
        )

        requirement = next(item for item in resolved.coverage_requirements if item.id == "deliverable:docx")
        self.assertEqual(requirement.preferred_skill_ids, [])
        self.assertEqual(requirement.acceptable_skill_ids, [])
        self.assertEqual(resolved.coverage_diagnostics[0]["status"], "missing")

    def test_intent_requirement_resolves_from_capability_text_and_execution_role(self) -> None:
        understanding = analyze_task("Analyze a CSV dataset with statistical tests.")
        skill_id = "skill:renamed-table-analyst"
        skills = {
            skill_id: _skill(
                skill_id,
                "renamed-table-analyst",
                "Analyze tabular datasets and statistical summaries.",
            )
        }
        interfaces = {
            skill_id: _interface(
                skill_id,
                requires=["csv_table"],
                produces=["statistical_summary"],
                summary="Performs tabular statistical analysis over CSV data.",
                when_to_use="Use for CSV, spreadsheet, or dataframe analysis.",
                execution_role="transformer",
            )
        }
        execution_index = [
            ExecutionIndexRecord(
                source_skill=skill_id,
                target_skill="skill:report-writer",
                relation_type="artifact_compatibility",
                canonical_object="statistical_summary",
                direction="source_to_target",
                confidence=0.91,
            )
        ]

        resolved = resolve_coverage_requirements(
            understanding,
            skills=skills,
            interfaces=interfaces,
            execution_index=execution_index,
        )

        requirement = next(
            item
            for item in resolved.coverage_requirements
            if item.id == "intent:tabular_or_statistical_analysis"
        )
        self.assertEqual(requirement.preferred_skill_ids, [skill_id])
        self.assertEqual(requirement.acceptable_skill_ids, [skill_id])


if __name__ == "__main__":
    unittest.main()
