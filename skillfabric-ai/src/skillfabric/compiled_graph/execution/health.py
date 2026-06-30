"""Health checks for the Execution Flow Layer."""

from __future__ import annotations

from dataclasses import dataclass, field

from skillfabric.compiled_graph.execution.compiler import ExecutionGraphBuild
from skillfabric.compiled_graph.execution.models import ExecutionValidationRecord


@dataclass(slots=True)
class ExecutionHealthReport:
    """Execution health check result."""

    raw_artifact_count: int
    raw_scenario_count: int
    canonical_artifact_count: int
    reusable_state_count: int
    execution_compatibility_count: int
    candidate_count: int
    accepted_flow_count: int
    rejected_flow_count: int
    duplicate_execution_record_count: int = 0
    empty_execution_index_warning: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    candidates_missing_evidence: list[str] = field(default_factory=list)
    compiler_warnings: list[str] = field(default_factory=list)


def analyze_execution_health(
    build: ExecutionGraphBuild,
    records: list[ExecutionValidationRecord],
) -> ExecutionHealthReport:
    """Analyze execution graph quality."""

    accepted_count = sum(1 for record in records if record.accepted)
    accepted_keys = [_accepted_record_key(record) for record in records if record.accepted and record.flow_edge is not None]
    duplicate_count = len(accepted_keys) - len(set(accepted_keys))
    return ExecutionHealthReport(
        raw_artifact_count=len(build.raw_artifact_nodes),
        raw_scenario_count=len(build.raw_scenario_nodes),
        canonical_artifact_count=len(
            {record.canonical_object for record in build.execution_index if record.relation_type == "artifact_compatibility"}
        ),
        reusable_state_count=len(
            {record.canonical_object for record in build.execution_index if record.relation_type == "state_compatibility"}
        ),
        execution_compatibility_count=len(build.execution_index),
        candidate_count=len(build.candidates),
        accepted_flow_count=accepted_count,
        rejected_flow_count=sum(1 for record in records if not record.accepted),
        duplicate_execution_record_count=duplicate_count,
        empty_execution_index_warning=bool(build.candidates and not build.execution_index),
        rejection_reasons=[record.rejection_reason for record in records if record.rejection_reason],
        candidates_missing_evidence=[
            f"{candidate.source_skill}->{candidate.target_skill}:{candidate.flow_type}"
            for candidate in build.candidates
            if not candidate.evidence
        ],
        compiler_warnings=list(build.warnings),
    )


def _accepted_record_key(record: ExecutionValidationRecord) -> tuple[str, str, str, str, str, str]:
    candidate = record.candidate
    return (
        candidate.source_skill,
        candidate.target_skill,
        candidate.metadata.get("relation_type", "artifact_compatibility"),
        candidate.metadata.get("canonical_object", candidate.matched_name),
        candidate.metadata.get("direction", "source_to_target"),
        str(record.normalized.get("projected_edge_type", "depend_on")),
    )


def render_execution_health_report(report: ExecutionHealthReport) -> str:
    """Render execution_health_report.md."""

    lines = [
        "# Execution Health Report",
        "",
        f"- raw artifacts: {report.raw_artifact_count}",
        f"- raw scenarios: {report.raw_scenario_count}",
        f"- canonical artifacts: {report.canonical_artifact_count}",
        f"- reusable states: {report.reusable_state_count}",
        f"- execution compatibility records: {report.execution_compatibility_count}",
        f"- candidates: {report.candidate_count}",
        f"- accepted flows: {report.accepted_flow_count}",
        f"- rejected flows: {report.rejected_flow_count}",
        f"- duplicate accepted execution records: {report.duplicate_execution_record_count}",
        f"- empty execution index warning: {'yes' if report.empty_execution_index_warning else 'no'}",
        f"- candidates missing evidence: {len(report.candidates_missing_evidence)}",
        f"- compiler warnings: {len(report.compiler_warnings)}",
        "",
        "## Compiler Warnings",
        "",
    ]
    if report.compiler_warnings:
        lines.extend(f"- {warning}" for warning in report.compiler_warnings)
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Rejection Reasons",
            "",
        ]
    )
    if report.rejection_reasons:
        lines.extend(f"- {reason}" for reason in report.rejection_reasons)
    else:
        lines.append("None.")
    lines.extend(["", "## Candidates Missing Evidence", ""])
    if report.candidates_missing_evidence:
        lines.extend(f"- {item}" for item in report.candidates_missing_evidence)
    else:
        lines.append("None.")
    return "\n".join(lines).rstrip() + "\n"
