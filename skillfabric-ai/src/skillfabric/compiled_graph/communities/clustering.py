"""Graph-first community detection for routing communities."""

from __future__ import annotations

import contextlib
import inspect
import io
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.communities.requirements import _assignment_health_requirements
from skillfabric.compiled_graph.models import CommunityNode, Edge
from skillfabric.registry.models import SkillNode


@dataclass(slots=True)
class _PartitionResult:
    components: list[list[str]]
    algorithm: str


@dataclass(slots=True)
class _ProjectionResult:
    graph: Any
    similar_edge_count: int
    compose_edge_count: int
    depend_on_ignored_count: int


def cluster_communities(
    skills: list[SkillNode],
    similar_edges: list[Edge],
    relation_edges: list[Edge],
) -> tuple[list[CommunityNode], list[Edge], dict[str, str], dict[str, Any]]:
    """Build final routing communities from a typed weighted graph projection."""

    projection = _build_projection_graph(skills, similar_edges, relation_edges)
    partition = _partition_graph(projection.graph)
    requirements = _assignment_health_requirements(len(skills))
    preferred_range = list(requirements.get("preferred_member_count_range", []))
    max_member_count = (
        int(preferred_range[1])
        if len(preferred_range) >= 2
        else max(1, math.floor(len(skills) * float(requirements["max_largest_fraction"])))
    )
    components, split_count = _rebalance_components(
        partition.components,
        projection.graph,
        minimum_communities=int(requirements["minimum_communities"]),
        max_member_count=max_member_count,
    )
    communities, member_edges, membership = _communities_from_components(
        components,
        skills,
        [*similar_edges, *relation_edges],
    )
    stats = {
        "community_clustering_algorithm": partition.algorithm,
        "community_projection_similar_to_count": projection.similar_edge_count,
        "community_projection_compose_with_count": projection.compose_edge_count,
        "community_projection_depend_on_ignored_count": projection.depend_on_ignored_count,
        "community_oversize_split_count": split_count,
    }
    return communities, member_edges, membership, stats


def _build_projection_graph(
    skills: list[SkillNode],
    similar_edges: list[Edge],
    relation_edges: list[Edge],
) -> _ProjectionResult:
    import networkx as nx

    graph = nx.Graph()
    for skill in skills:
        graph.add_node(skill.id)
    similar_count = 0
    compose_count = 0
    ignored_depend_on_count = 0
    for edge in similar_edges:
        if edge.type != "similar_to":
            continue
        similar_count += 1
        _add_weighted_edge(graph, edge.source, edge.target, _edge_weight(edge), "similar_to")
    for edge in relation_edges:
        if edge.type == "compose_with":
            compose_count += 1
            _add_weighted_edge(graph, edge.source, edge.target, _edge_weight(edge) * 0.35, "compose_with")
        elif edge.type == "depend_on":
            ignored_depend_on_count += 1
    return _ProjectionResult(graph, similar_count, compose_count, ignored_depend_on_count)


def _add_weighted_edge(graph: Any, source: str, target: str, weight: float, edge_type: str) -> None:
    if source == target or weight <= 0:
        return
    existing = graph.get_edge_data(source, target, default={})
    current_weight = float(existing.get("weight", 0.0) or 0.0)
    if weight <= current_weight:
        return
    graph.add_edge(source, target, weight=weight, projection_type=edge_type)


def _edge_weight(edge: Edge) -> float:
    return max(float(edge.weight or 0.0), float(edge.confidence or 0.0))


def _partition_graph(graph: Any) -> _PartitionResult:
    if graph.number_of_nodes() == 0:
        return _PartitionResult([], "empty")
    if graph.number_of_edges() == 0:
        return _PartitionResult([[str(node)] for node in sorted(graph.nodes)], "singletons")
    isolates = [str(node) for node in graph.nodes if graph.degree(node) == 0]
    connected_nodes = [node for node in graph.nodes if graph.degree(node) > 0]
    connected = graph.subgraph(connected_nodes).copy()
    partition = _partition_connected_graph(connected)
    components_by_id: dict[int, list[str]] = {}
    for node, community_id in partition.components:
        components_by_id.setdefault(community_id, []).append(str(node))
    components = [sorted(nodes) for nodes in components_by_id.values()]
    components.extend([node] for node in sorted(isolates))
    components.sort(key=lambda item: (-len(item), item[0] if item else ""))
    return _PartitionResult(components, partition.algorithm)


@dataclass(slots=True)
class _RawPartition:
    components: list[tuple[str, int]]
    algorithm: str


def _partition_connected_graph(graph: Any) -> _RawPartition:
    _prepare_native_library_cache_dirs()
    try:
        from graspologic.partition import leiden
    except ImportError as exc:
        raise RuntimeError("Leiden community detection requires graspologic.partition.leiden") from exc

    kwargs: dict[str, Any] = {}
    parameters = inspect.signature(leiden).parameters
    if "weight_attribute" in parameters:
        kwargs["weight_attribute"] = "weight"
    if "random_seed" in parameters:
        kwargs["random_seed"] = 42
    try:
        with _suppress_partition_output():
            raw = leiden(graph, **kwargs)
    except Exception as exc:  # noqa: BLE001 - surface partition failures without fallback.
        raise RuntimeError("Leiden community detection failed") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("Leiden community detection returned an unsupported partition payload")
    return _RawPartition([(str(node), int(cid)) for node, cid in raw.items()], "leiden")


def _prepare_native_library_cache_dirs() -> None:
    cache_root = Path(os.environ.get("SKILLFABRIC_NATIVE_CACHE_DIR", _default_native_cache_root()))
    defaults = {
        "NUMBA_CACHE_DIR": cache_root / "numba",
        "MPLCONFIGDIR": cache_root / "matplotlib",
    }
    for key, path in defaults.items():
        if os.environ.get(key):
            continue
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)


def _default_native_cache_root() -> Path:
    return Path(__file__).resolve().parents[6] / "tmp" / "skillfabric-native-cache"


@contextlib.contextmanager
def _suppress_partition_output():
    old_stderr = sys.stderr
    try:
        sys.stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        sys.stderr = old_stderr


def _rebalance_components(
    components: list[list[str]],
    graph: Any,
    *,
    minimum_communities: int,
    max_member_count: int,
) -> tuple[list[list[str]], int]:
    if not components:
        return [], 0
    max_size = max(1, max_member_count)
    balanced = [sorted(component) for component in components]
    split_count = 0
    changed = True
    while changed:
        changed = False
        next_components: list[list[str]] = []
        for component in balanced:
            if len(component) <= max_size:
                next_components.append(component)
                continue
            split = _split_component(component, graph, max_size=max_size)
            split_count += max(0, len(split) - 1)
            next_components.extend(split)
            changed = True
        balanced = _sort_components(next_components)
    while len(balanced) < minimum_communities:
        largest_index = max(range(len(balanced)), key=lambda index: len(balanced[index]))
        largest = balanced[largest_index]
        if len(largest) <= 1:
            break
        split = _split_component(largest, graph, max_size=max(1, len(largest) // 2))
        if len(split) <= 1:
            split = _deterministic_chunks(largest, max(1, math.ceil(len(largest) / 2)))
        balanced = [
            component
            for index, component in enumerate(balanced)
            if index != largest_index
        ]
        balanced.extend(split)
        balanced = _sort_components(balanced)
        split_count += max(0, len(split) - 1)
    return balanced, split_count


def _split_component(component: list[str], graph: Any, *, max_size: int) -> list[list[str]]:
    if len(component) <= max_size:
        return [sorted(component)]
    subgraph = graph.subgraph(component).copy()
    if subgraph.number_of_edges() > 0:
        partition = _partition_graph(subgraph)
        split = [nodes for nodes in partition.components if nodes]
        if len(split) > 1 and max(len(nodes) for nodes in split) < len(component):
            output: list[list[str]] = []
            for nodes in split:
                if len(nodes) > max_size:
                    output.extend(_deterministic_chunks(nodes, max_size))
                else:
                    output.append(sorted(nodes))
            return _sort_components(output)
    return _deterministic_chunks(component, max_size)


def _deterministic_chunks(component: list[str], max_size: int) -> list[list[str]]:
    members = sorted(component)
    return [members[index : index + max_size] for index in range(0, len(members), max_size)]


def _sort_components(components: list[list[str]]) -> list[list[str]]:
    return sorted((sorted(component) for component in components), key=lambda item: (-len(item), item[0] if item else ""))


def _communities_from_components(
    components: list[list[str]],
    skills: list[SkillNode],
    edges: list[Edge],
) -> tuple[list[CommunityNode], list[Edge], dict[str, str]]:
    by_id = {skill.id: skill for skill in skills}
    communities: list[CommunityNode] = []
    member_edges: list[Edge] = []
    membership: dict[str, str] = {}
    for index, members in enumerate(_sort_components(components)):
        community_id = f"community:{index:04d}-{_community_slug(members, by_id)}"
        community = CommunityNode(
            id=community_id,
            type="community",
            name=_community_name(members, by_id),
            summary=_community_summary(members, by_id),
            member_count=len(members),
            representative_skill_ids=_representatives(members, edges, limit=5),
            cohesion_score=_cohesion(members, edges),
        )
        communities.append(community)
        for skill_id in sorted(members):
            membership[skill_id] = community_id
            member_edges.append(
                Edge(
                    source=skill_id,
                    target=community_id,
                    type="member_of",
                    confidence=1.0,
                    weight=1.0,
                    provenance="graph_clustered",
                    reason=f"{skill_id} assigned to {community_id} by graph-first clustering.",
                )
            )
    return communities, member_edges, membership


def _representatives(members: list[str], edges: list[Edge], *, limit: int) -> list[str]:
    member_set = set(members)
    degree = Counter()
    for edge in edges:
        if edge.type == "depend_on":
            continue
        if not (edge.source in member_set and edge.target in member_set):
            continue
        weight = _edge_weight(edge)
        degree[edge.source] += weight
        degree[edge.target] += weight
    return [
        skill_id
        for skill_id, _score in sorted(degree.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ] or sorted(members)[:limit]


def _community_name(members: list[str], by_id: dict[str, SkillNode]) -> str:
    tokens = Counter()
    for skill_id in members:
        skill = by_id[skill_id]
        for part in skill.name.replace("-", " ").replace("_", " ").split():
            normalized = part.strip().title()
            if len(normalized) > 2 and normalized.lower() not in {"skill", "with"}:
                tokens[normalized] += 1
    if not tokens:
        return "General Skills"
    return " ".join(token for token, _count in tokens.most_common(3))


def _community_slug(members: list[str], by_id: dict[str, SkillNode]) -> str:
    return _community_name(members, by_id).lower().replace(" ", "-") or "general"


def _community_summary(members: list[str], by_id: dict[str, SkillNode]) -> str:
    name = _community_name(members, by_id)
    examples = ", ".join(by_id[skill_id].name for skill_id in sorted(members)[:5])
    return f"{name} routing community covering {len(members)} skills: {examples}."


def _cohesion(members: list[str], edges: list[Edge]) -> float:
    if len(members) <= 1:
        return 1.0
    member_set = set(members)
    actual = 0
    for edge in edges:
        if edge.type == "depend_on":
            continue
        if edge.source in member_set and edge.target in member_set:
            actual += 1
    possible = len(members) * (len(members) - 1) / 2
    return round(actual / possible, 4) if possible else 0.0
