"""Health reporting for pool-level canonicalization."""

from __future__ import annotations

from dataclasses import dataclass

from skillfabric.compiled_graph.canonicalization.models import CanonicalizationBuild


@dataclass(slots=True)
class CanonicalizationHealthReport:
    raw_contract_object_count: int
    canonical_object_count: int
    promoted_object_count: int
    alias_merge_ratio: float
    producer_consumer_match_count: int
    candidate_edge_count: int = 0
    candidate_component_count: int = 0
    ambiguous_component_count: int = 0
    lexical_candidate_count: int = 0
    semantic_candidate_count: int = 0
    warning_count: int = 0


def analyze_canonicalization_health(build: CanonicalizationBuild) -> CanonicalizationHealthReport:
    """Analyze canonicalization quality and usefulness."""

    canonical_count = len(build.objects)
    raw_count = len(build.raw_terms)
    producer_consumer = sum(1 for item in build.objects if item.required_by and item.produced_by)
    merge_ratio = 0.0
    if raw_count:
        merge_ratio = round(1.0 - (canonical_count / raw_count), 6)
    return CanonicalizationHealthReport(
        raw_contract_object_count=raw_count,
        canonical_object_count=canonical_count,
        promoted_object_count=sum(1 for item in build.objects if item.promoted),
        alias_merge_ratio=merge_ratio,
        producer_consumer_match_count=producer_consumer,
        candidate_edge_count=len(build.candidate_edges),
        candidate_component_count=len(build.candidate_components),
        ambiguous_component_count=sum(
            1 for item in build.candidate_components if bool(getattr(item, "ambiguous", False))
        ),
        lexical_candidate_count=sum(
            1 for item in build.candidate_edges if getattr(item, "method", "") == "lexical"
        ),
        semantic_candidate_count=sum(
            1 for item in build.candidate_edges if getattr(item, "method", "") == "semantic"
        ),
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
            f"- promoted objects: {report.promoted_object_count}",
            f"- alias merge ratio: {report.alias_merge_ratio}",
            f"- producer-consumer matches: {report.producer_consumer_match_count}",
            f"- candidate edges: {report.candidate_edge_count}",
            f"- candidate components: {report.candidate_component_count}",
            f"- ambiguous components: {report.ambiguous_component_count}",
            f"- lexical candidates: {report.lexical_candidate_count}",
            f"- semantic candidates: {report.semantic_candidate_count}",
            f"- warnings: {report.warning_count}",
            "",
        ]
    )
