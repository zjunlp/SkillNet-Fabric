"""Graph health checks。"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean

from skillfabric.compiled_graph.models import CommunityNode, Edge, GraphDocument
from skillfabric.registry.models import SkillNode


@dataclass(slots=True)
class CommunityTextOutlier:
    """Skill assignment whose text fit is much stronger in another community."""

    skill_id: str
    assigned_community_id: str
    assigned_community_name: str
    suggested_community_id: str
    suggested_community_name: str
    current_score: float
    suggested_score: float
    shared_terms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WeakCrossCommunityComposeEdge:
    """Low-confidence compose edge that spans two different communities."""

    source: str
    target: str
    confidence: float
    source_community_name: str
    target_community_name: str
    provenance: str = ""


@dataclass(slots=True)
class LowCohesionLargeCommunity:
    """Large community whose internal relation density suggests over-merge."""

    community_id: str
    community_name: str
    member_count: int
    average_member_count: float
    cohesion_score: float


@dataclass(slots=True)
class HealthReport:
    """Graph health check result."""

    isolated_skills: list[str] = field(default_factory=list)
    high_degree_hubs: list[tuple[str, int]] = field(default_factory=list)
    low_confidence_edges: list[Edge] = field(default_factory=list)
    depend_on_cycles: list[list[str]] = field(default_factory=list)
    community_size_outliers: list[tuple[str, int]] = field(default_factory=list)
    community_text_outliers: list[CommunityTextOutlier] = field(default_factory=list)
    weak_cross_community_compose_edges: list[WeakCrossCommunityComposeEdge] = field(default_factory=list)
    low_cohesion_large_communities: list[LowCohesionLargeCommunity] = field(default_factory=list)
    edges_missing_evidence: int = 0


def analyze_health(graph: GraphDocument, communities: list[CommunityNode]) -> HealthReport:
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
    cycles = _find_depend_cycles(graph.edges)
    sizes = [community.member_count for community in communities]
    outliers: list[tuple[str, int]] = []
    if sizes:
        avg = mean(sizes)
        for community in communities:
            if community.member_count == 1 or community.member_count > max(10, avg * 3):
                outliers.append((community.id, community.member_count))
    text_outliers = _find_community_text_outliers(graph, communities)
    weak_cross_community = _find_weak_cross_community_compose_edges(graph, communities)
    low_cohesion_large = _find_low_cohesion_large_communities(communities)
    return HealthReport(
        isolated_skills=isolated,
        high_degree_hubs=hubs,
        low_confidence_edges=low_confidence,
        depend_on_cycles=cycles,
        community_size_outliers=outliers,
        community_text_outliers=text_outliers,
        weak_cross_community_compose_edges=weak_cross_community,
        low_cohesion_large_communities=low_cohesion_large,
        edges_missing_evidence=missing_evidence,
    )


def render_health_report(
    graph: GraphDocument,
    communities: list[CommunityNode],
    *,
    cache_hits: int = 0,
    llm_validations: int = 0,
    skipped_unchanged: int = 0,
) -> str:
    """Render graph_health_report.md."""

    report = analyze_health(graph, communities)
    edge_counts = Counter(edge.type for edge in graph.edges)
    lines = [
        "# Graph Health Report",
        "",
        "## Summary",
        "",
        f"- skill_count: {sum(1 for node in graph.nodes if getattr(node, 'type', '') == 'skill')}",
        f"- community_count: {len(communities)}",
        f"- edge_count: {len(graph.edges)}",
        f"- similar_to_edges: {edge_counts.get('similar_to', 0)}",
        f"- member_of_edges: {edge_counts.get('member_of', 0)}",
        f"- compose_with_edges: {edge_counts.get('compose_with', 0)}",
        f"- depend_on_edges: {edge_counts.get('depend_on', 0)}",
        f"- community_text_outliers: {len(report.community_text_outliers)}",
        f"- weak_cross_community_compose_edges: {len(report.weak_cross_community_compose_edges)}",
        f"- low_cohesion_large_communities: {len(report.low_cohesion_large_communities)}",
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
        "## Community Size Outliers",
        "",
        *_list(f"{community_id}: {size}" for community_id, size in report.community_size_outliers),
        "",
        "## Low-Cohesion Large Communities",
        "",
        *_list(
            (
                f"{item.community_id}: {item.community_name} "
                f"members={item.member_count} avg_members={item.average_member_count:.2f} "
                f"cohesion={item.cohesion_score:.4f}"
            )
            for item in report.low_cohesion_large_communities
        ),
        "",
        "## Community Text Outliers",
        "",
        *_list(
            (
                f"{item.skill_id}: {item.assigned_community_name} -> {item.suggested_community_name} "
                f"(current={item.current_score:.2f}, suggested={item.suggested_score:.2f}, "
                f"shared_terms={', '.join(item.shared_terms) or 'none'})"
            )
            for item in report.community_text_outliers
        ),
        "",
        "## Weak Cross-Community Compose Edges",
        "",
        *_list(
            (
                f"{item.source} -> {item.target} confidence={item.confidence:.2f} "
                f"communities={item.source_community_name} / {item.target_community_name} "
                f"provenance={item.provenance or 'unknown'}"
            )
            for item in report.weak_cross_community_compose_edges
        ),
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


def _find_weak_cross_community_compose_edges(
    graph: GraphDocument,
    communities: list[CommunityNode],
) -> list[WeakCrossCommunityComposeEdge]:
    if not communities:
        return []
    membership = _skill_membership(graph.edges)
    community_names = {community.id: community.name or community.id for community in communities}
    flagged: list[WeakCrossCommunityComposeEdge] = []
    for edge in graph.edges:
        if edge.type != "compose_with" or edge.confidence >= 0.9:
            continue
        source_community = membership.get(edge.source)
        target_community = membership.get(edge.target)
        if not source_community or not target_community or source_community == target_community:
            continue
        flagged.append(
            WeakCrossCommunityComposeEdge(
                source=edge.source,
                target=edge.target,
                confidence=edge.confidence,
                source_community_name=community_names.get(source_community, source_community),
                target_community_name=community_names.get(target_community, target_community),
                provenance=edge.provenance,
            )
        )
    flagged.sort(key=lambda item: (item.confidence, item.source, item.target))
    return flagged[:30]


def _find_low_cohesion_large_communities(
    communities: list[CommunityNode],
) -> list[LowCohesionLargeCommunity]:
    if len(communities) < 2:
        return []
    average_member_count = mean(community.member_count for community in communities)
    member_threshold = max(12.0, average_member_count * 1.5)
    flagged: list[LowCohesionLargeCommunity] = []
    for community in communities:
        if community.member_count < member_threshold:
            continue
        if community.cohesion_score > 0.2:
            continue
        flagged.append(
            LowCohesionLargeCommunity(
                community_id=community.id,
                community_name=community.name or community.id,
                member_count=community.member_count,
                average_member_count=average_member_count,
                cohesion_score=community.cohesion_score,
            )
        )
    flagged.sort(key=lambda item: (item.cohesion_score, -item.member_count, item.community_id))
    return flagged[:20]


def _find_community_text_outliers(
    graph: GraphDocument,
    communities: list[CommunityNode],
) -> list[CommunityTextOutlier]:
    if len(communities) < 2:
        return []
    skills = {
        node.id: node
        for node in graph.nodes
        if isinstance(node, SkillNode)
    }
    community_by_id = {community.id: community for community in communities}
    membership = _skill_membership(graph.edges)
    flagged: list[CommunityTextOutlier] = []
    for skill_id, assigned_community_id in sorted(membership.items()):
        skill = skills.get(skill_id)
        assigned = community_by_id.get(assigned_community_id)
        if skill is None or assigned is None:
            continue
        current_score, _current_terms = _skill_community_text_score(skill, assigned)
        alternatives: list[tuple[float, list[str], CommunityNode]] = []
        for community in communities:
            if community.id == assigned.id:
                continue
            score, terms = _skill_community_text_score(skill, community)
            alternatives.append((score, terms, community))
        if not alternatives:
            continue
        suggested_score, shared_terms, suggested = max(
            alternatives,
            key=lambda item: (item[0], item[2].id),
        )
        if not _is_strong_community_text_outlier(
            current_score=current_score,
            suggested_score=suggested_score,
        ):
            continue
        flagged.append(
            CommunityTextOutlier(
                skill_id=skill_id,
                assigned_community_id=assigned.id,
                assigned_community_name=assigned.name or assigned.id,
                suggested_community_id=suggested.id,
                suggested_community_name=suggested.name or suggested.id,
                current_score=current_score,
                suggested_score=suggested_score,
                shared_terms=shared_terms[:8],
            )
        )
    flagged.sort(
        key=lambda item: (
            -(item.suggested_score - item.current_score),
            item.skill_id,
            item.suggested_community_id,
        )
    )
    return flagged[:20]


def _skill_membership(edges: list[Edge]) -> dict[str, str]:
    return {
        edge.source: edge.target
        for edge in edges
        if edge.type == "member_of" and edge.source.startswith("skill:")
    }


def _is_strong_community_text_outlier(*, current_score: float, suggested_score: float) -> bool:
    return (
        current_score <= 0.4
        and suggested_score >= 0.8
        and suggested_score - current_score >= 0.8
    )


def _skill_community_text_score(
    skill: SkillNode,
    community: CommunityNode,
) -> tuple[float, list[str]]:
    skill_tokens = _semantic_tokens(f"{skill.name} {skill.description}")
    if not skill_tokens:
        return 0.0, []
    community_text = " ".join(
        [
            community.name,
            community.summary,
            " ".join(community.task_patterns),
        ]
    )
    community_tokens = _semantic_tokens(community_text)
    if not community_tokens:
        return 0.0, []
    overlap = sorted(skill_tokens & community_tokens)
    if not overlap:
        return 0.0, []
    name_bonus = 0.8 if _semantic_tokens(skill.name) & community_tokens else 0.0
    overlap_score = min(0.8, 0.12 * len(overlap))
    return name_bonus + overlap_score, overlap


_SEMANTIC_STOPWORDS = {
    "and",
    "any",
    "artifact",
    "artifacts",
    "are",
    "as",
    "build",
    "builder",
    "can",
    "content",
    "create",
    "creating",
    "creator",
    "design",
    "engineer",
    "engineering",
    "for",
    "from",
    "generate",
    "generating",
    "generator",
    "guide",
    "high",
    "in",
    "into",
    "maker",
    "media",
    "on",
    "optimization",
    "optimized",
    "optimizing",
    "or",
    "other",
    "output",
    "pattern",
    "patterns",
    "processing",
    "quality",
    "review",
    "sharing",
    "share",
    "skill",
    "skills",
    "state",
    "system",
    "systems",
    "table",
    "tables",
    "that",
    "the",
    "this",
    "to",
    "tool",
    "tools",
    "use",
    "using",
    "web",
    "when",
    "with",
    "work",
    "workflow",
    "workflows",
    "you",
}


def _semantic_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", value.lower().replace("_", " ").replace("-", " ")):
        token = raw[:-1] if len(raw) > 3 and raw.endswith("s") else raw
        if len(token) < 2 or token in _SEMANTIC_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _list(items) -> list[str]:
    values = list(items)
    if not values:
        return ["- none"]
    return [f"- {item}" for item in values]
