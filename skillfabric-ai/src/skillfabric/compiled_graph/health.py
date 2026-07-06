"""Graph health checks."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from skillfabric.compiled_graph.models import Edge, GraphDocument


@dataclass(slots=True)
class HealthReport:
    """Graph health check result."""

    isolated_skills: list[str] = field(default_factory=list)
    high_degree_hubs: list[tuple[str, int]] = field(default_factory=list)
    low_confidence_edges: list[Edge] = field(default_factory=list)
    depend_on_cycles: list[list[str]] = field(default_factory=list)
    edges_missing_evidence: int = 0


def analyze_health(graph: GraphDocument) -> HealthReport:
    """Run graph health checks."""

    skill_ids = [node.id for node in graph.nodes if getattr(node, "type", "") == "skill"]
    degree = Counter()
    for edge in graph.edges:
        degree[edge.source] += 1
        degree[edge.target] += 1
    isolated = sorted(skill_id for skill_id in skill_ids if degree[skill_id] == 0)
    hub_threshold = max(6, int(len(graph.edges) ** 0.5) if graph.edges else 6)
    hubs = sorted(
        [(skill_id, count) for skill_id, count in degree.items() if skill_id.startswith("skill:") and count >= hub_threshold],
        key=lambda item: (-item[1], item[0]),
    )[:20]
    low_confidence = [
        edge
        for edge in graph.edges
        if edge.type in {"compose_with", "depend_on"} and edge.confidence < 0.9
    ]
    missing_evidence = sum(
        1 for edge in graph.edges if edge.type in {"compose_with", "depend_on"} and not edge.evidence
    )
    return HealthReport(
        isolated_skills=isolated,
        high_degree_hubs=hubs,
        low_confidence_edges=low_confidence,
        depend_on_cycles=_find_depend_cycles(graph.edges),
        edges_missing_evidence=missing_evidence,
    )


def render_health_report(
    graph: GraphDocument,
    *,
    cache_hits: int = 0,
    llm_validations: int = 0,
    skipped_unchanged: int = 0,
) -> str:
    """Render graph_health_report.md."""

    report = analyze_health(graph)
    edge_counts = Counter(edge.type for edge in graph.edges)
    lines = [
        "# Graph Health Report",
        "",
        "## Summary",
        "",
        f"- skill_count: {sum(1 for node in graph.nodes if getattr(node, 'type', '') == 'skill')}",
        f"- edge_count: {len(graph.edges)}",
        f"- similar_to_edges: {edge_counts.get('similar_to', 0)}",
        f"- compose_with_edges: {edge_counts.get('compose_with', 0)}",
        f"- depend_on_edges: {edge_counts.get('depend_on', 0)}",
        f"- cache_hits: {cache_hits}",
        f"- llm_validations: {llm_validations}",
        f"- skipped_unchanged: {skipped_unchanged}",
        "",
        "## Isolated Skills",
        "",
        *_list(report.isolated_skills),
        "",
        "## High-Degree Hub Skills",
        "",
        *_list(f"{skill_id}: {count}" for skill_id, count in report.high_degree_hubs),
        "",
        "## Low-Confidence Edges",
        "",
        *_list(
            f"{edge.type} {edge.source} -> {edge.target} confidence={edge.confidence}"
            for edge in report.low_confidence_edges
        ),
        "",
        "## depend_on Cycles",
        "",
        *_list(" -> ".join(cycle) for cycle in report.depend_on_cycles),
        "",
        "## Edges Missing Evidence",
        "",
        f"- count: {report.edges_missing_evidence}",
        "",
    ]
    return "\n".join(lines)


def _find_depend_cycles(edges: list[Edge]) -> list[list[str]]:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.type == "depend_on":
            adjacency[edge.source].append(edge.target)
    cycles: list[list[str]] = []
    visiting: list[str] = []
    seen: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            cycle = visiting[visiting.index(node) :] + [node]
            if cycle not in cycles:
                cycles.append(cycle)
            return
        if node in seen:
            return
        visiting.append(node)
        for neighbor in adjacency.get(node, []):
            visit(neighbor)
        visiting.pop()
        seen.add(node)

    for node in sorted(adjacency):
        visit(node)
    return cycles[:20]


def _list(values) -> list[str]:
    output = [f"- {value}" for value in values]
    return output or ["- None"]
