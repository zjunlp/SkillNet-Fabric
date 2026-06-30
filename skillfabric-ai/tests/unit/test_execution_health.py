from __future__ import annotations

import unittest

from skillfabric.compiled_graph.execution.compiler import ExecutionGraphBuild
from skillfabric.compiled_graph.execution.health import (
    analyze_execution_health,
    render_execution_health_report,
)
from skillfabric.compiled_graph.execution.models import (
    ExecutionEdge,
    ExecutionFlowCandidate,
    ExecutionIndexRecord,
    ExecutionValidationRecord,
)


class ExecutionHealthTests(unittest.TestCase):
    def test_health_report_counts_missing_nodes_rejections_and_orphans(self) -> None:
        candidate = ExecutionFlowCandidate(
            source_skill="skill:a",
            target_skill="skill:b",
            flow_type="artifact_flow",
            matched_node_id="artifact:csv",
            matched_name="csv",
            evidence=[],
        )
        build = ExecutionGraphBuild(
            candidates=[candidate],
        )
        self.assertFalse(hasattr(build, "artifact_nodes"))
        self.assertFalse(hasattr(build, "scenario_nodes"))
        records = [
            ExecutionValidationRecord(
                candidate=candidate,
                raw_output={"accepted": False},
                normalized={"accepted": False},
                accepted=False,
                rejection_reason="missing evidence",
            )
        ]

        report = analyze_execution_health(build, records)
        rendered = render_execution_health_report(report)

        self.assertEqual(report.raw_artifact_count, 0)
        self.assertEqual(report.execution_compatibility_count, 0)
        self.assertEqual(report.rejected_flow_count, 1)
        self.assertIn("missing evidence", rendered)

    def test_health_report_counts_duplicate_accepted_execution_records(self) -> None:
        candidate = ExecutionFlowCandidate(
            source_skill="skill:a",
            target_skill="skill:b",
            flow_type="artifact_flow",
            matched_node_id="canonical:artifact_compatibility:csv_table",
            matched_name="csv_table",
            metadata={
                "relation_type": "artifact_compatibility",
                "canonical_object": "csv_table",
                "direction": "source_to_target",
            },
        )
        edge = ExecutionEdge(
            source="skill:a",
            target="skill:b",
            type="artifact_flow",
            confidence=0.93,
            metadata=candidate.metadata,
        )
        build = ExecutionGraphBuild(
            candidates=[candidate],
            execution_index=[
                ExecutionIndexRecord(
                    source_skill="skill:a",
                    target_skill="skill:b",
                    relation_type="artifact_compatibility",
                    canonical_object="csv_table",
                    direction="source_to_target",
                    confidence=0.93,
                    projected_edge_type="depend_on",
                )
            ],
        )
        records = [
            ExecutionValidationRecord(
                candidate=candidate,
                raw_output={"accepted": True},
                normalized={"accepted": True, "projected_edge_type": "depend_on"},
                accepted=True,
                rejection_reason="",
                flow_edge=edge,
            ),
            ExecutionValidationRecord(
                candidate=candidate,
                raw_output={"accepted": True},
                normalized={"accepted": True, "projected_edge_type": "depend_on"},
                accepted=True,
                rejection_reason="",
                flow_edge=edge,
            ),
        ]

        report = analyze_execution_health(build, records)
        rendered = render_execution_health_report(report)

        self.assertEqual(report.duplicate_execution_record_count, 1)
        self.assertIn("duplicate accepted execution records: 1", rendered)

    def test_health_report_warns_when_candidates_have_no_compatibility_records(self) -> None:
        candidate = ExecutionFlowCandidate(
            source_skill="skill:a",
            target_skill="skill:b",
            flow_type="artifact_flow",
            matched_node_id="canonical:artifact_compatibility:csv_table",
            matched_name="csv_table",
            evidence=[],
        )
        build = ExecutionGraphBuild(candidates=[candidate], execution_index=[])

        report = analyze_execution_health(build, records=[])
        rendered = render_execution_health_report(report)

        self.assertTrue(report.empty_execution_index_warning)
        self.assertIn("empty execution index warning: yes", rendered)

    def test_health_report_includes_execution_compiler_warnings(self) -> None:
        build = ExecutionGraphBuild(
            warnings=["execution compatibility requires accepted canonicalization assignments"]
        )

        report = analyze_execution_health(build, records=[])
        rendered = render_execution_health_report(report)

        self.assertEqual(report.compiler_warnings, build.warnings)
        self.assertIn("canonicalization assignments", rendered)


if __name__ == "__main__":
    unittest.main()
