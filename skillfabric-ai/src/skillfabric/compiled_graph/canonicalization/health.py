"""Health reporting for interface term canonicalization."""

from __future__ import annotations

from dataclasses import dataclass

from skillfabric.compiled_graph.canonicalization.models import CanonicalizationBuild


@dataclass(slots=True)
class CanonicalizationHealthReport:
    raw_contract_object_count: int
    canonical_object_count: int
    assignment_count: int
    alias_merge_ratio: float
    producer_consumer_match_count: int
    warning_count: int = 0


def analyze_canonicalization_health(build: CanonicalizationBuild) -> CanonicalizationHealthReport:
    """Analyze canonicalization coverage and downstream usefulness."""

    raw_count = len(build.raw_terms)
    canonical_count = len(build.objects)
    merge_ratio = 0.0 if not raw_count else round(1.0 - (canonical_count / raw_count), 6)
    return CanonicalizationHealthReport(
        raw_contract_object_count=raw_count,
        canonical_object_count=canonical_count,
        assignment_count=len(build.assignments),
        alias_merge_ratio=merge_ratio,
        producer_consumer_match_count=sum(1 for item in build.objects if item.required_by and item.produced_by),
        warning_count=len(build.warnings),
    )


def render_canonicalization_health_report(report: CanonicalizationHealthReport) -> str:
    """Render canonicalization_health_report.md."""

    return "\n".join(
        [
            "# Canonicalization Health Report",
            "",
            f"- raw contract objects: {report.raw_contract_object_count}",
            f"- canonical objects: {report.canonical_object_count}",
            f"- assignments: {report.assignment_count}",
            f"- alias merge ratio: {report.alias_merge_ratio}",
            f"- producer-consumer matches: {report.producer_consumer_match_count}",
            f"- warnings: {report.warning_count}",
            "",
        ]
    )
