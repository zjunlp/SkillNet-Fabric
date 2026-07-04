"""Route edge parsing, merging, and reconciliation."""

from __future__ import annotations

from typing import Any

from skillfabric.router.models import RouteEdge, RouterBundle


def _edges_from_payload(
    payload: Any,
    selected_ids: set[str],
    *,
    source: str,
    warnings: list[str],
    allow_sequence: bool = False,
    default_edge_type: str = "depend_on",
) -> list[RouteEdge]:
    if not isinstance(payload, list):
        return []
    edges: list[RouteEdge] = []
    if allow_sequence:
        edges.extend(_edges_from_ordered_sequence(payload, selected_ids, source=source, warnings=warnings))
    for item in payload:
        if not isinstance(item, dict):
            continue
        before = str(item.get("before_skill", ""))
        after = str(item.get("after_skill", ""))
        if not before and not after:
            continue
        if before not in selected_ids or after not in selected_ids:
            warnings.append(f"dropped invalid route edge: {before} -> {after}")
            continue
        edges.append(
            RouteEdge(
                before_skill=before,
                after_skill=after,
                edge_type=str(item.get("edge_type", default_edge_type) or default_edge_type),
                confidence=_safe_float(item.get("confidence"), 0.0),
                reason=str(item.get("reason", "")),
                source=str(item.get("source", source) or source),
            )
        )
    return edges


def _edges_from_ordered_sequence(
    payload: list[Any],
    selected_ids: set[str],
    *,
    source: str,
    warnings: list[str],
) -> list[RouteEdge]:
    sequence: list[str] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, str):
            continue
        skill_id = str(item)
        if skill_id not in selected_ids:
            warnings.append(f"dropped invalid ordered hint skill: {skill_id}")
            continue
        if skill_id in seen:
            warnings.append(f"dropped duplicate ordered hint skill: {skill_id}")
            continue
        seen.add(skill_id)
        sequence.append(skill_id)
    if len(sequence) < 2:
        return []
    return [
        RouteEdge(
            before_skill=before,
            after_skill=after,
            edge_type="hint",
            confidence=0.0,
            reason="ordered_hints sequence",
            source=source,
        )
        for before, after in zip(sequence, sequence[1:], strict=False)
    ]


def _reconcile_route_edges(
    required_edges: list[RouteEdge],
    ordered_hints: list[RouteEdge],
    *,
    warnings: list[str],
) -> list[RouteEdge]:
    """Drop conflicting same-LLM hard edges while keeping ordered hints soft."""

    required_edges = _merge_edges(required_edges)
    ordered_hints = _merge_edges(ordered_hints)
    hint_graph = _edge_graph(ordered_hints)
    filtered_required: list[RouteEdge] = []
    for edge in required_edges:
        if _has_path(hint_graph, edge.after_skill, edge.before_skill) and _is_llm_edge(edge):
            warnings.append(
                "dropped conflicting LLM required edge: "
                f"{edge.before_skill} -> {edge.after_skill}; ordered hint requires "
                f"{edge.after_skill} before {edge.before_skill}"
            )
            continue
        filtered_required.append(edge)

    return _merge_edges(filtered_required)


def _reconcile_ordered_hints(
    ordered_hints: list[RouteEdge],
    required_edges: list[RouteEdge],
    *,
    warnings: list[str],
) -> list[RouteEdge]:
    """Return soft hints that do not contradict hard required edges."""

    ordered_hints = _merge_edges(ordered_hints)
    required_graph = _edge_graph(_merge_edges(required_edges))
    filtered: list[RouteEdge] = []
    for hint in ordered_hints:
        if _has_path(required_graph, hint.after_skill, hint.before_skill):
            warnings.append(
                "dropped conflicting ordered hint: "
                f"{hint.before_skill} -> {hint.after_skill}; required edge already forces reverse order"
            )
            continue
        filtered.append(hint)
    return _merge_edges(filtered)


def _edges_from_ordered_skill_ids(
    skill_ids: list[str],
    selected_ids: set[str],
    *,
    source: str,
    warnings: list[str],
) -> list[RouteEdge]:
    """Build hint edges from an explicit selected skill sequence."""

    return _edges_from_ordered_sequence(skill_ids, selected_ids, source=source, warnings=warnings)


def _is_llm_edge(edge: RouteEdge) -> bool:
    return edge.source in {"wiki_agent", "llm_router", ""}


def _edge_graph(edges: list[RouteEdge]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for edge in edges:
        graph.setdefault(edge.before_skill, set()).add(edge.after_skill)
        graph.setdefault(edge.after_skill, set())
    return graph


def _has_path(graph: dict[str, set[str]], start: str, target: str) -> bool:
    if not start or not target or start == target:
        return bool(start and start == target)
    seen: set[str] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(sorted(graph.get(node, set()) - seen))
    return False


def _edges_from_workflow_hints(bundle: RouterBundle, selected_ids: set[str]) -> list[RouteEdge]:
    edges: list[RouteEdge] = []
    for hint in bundle.workflow_hints:
        if hint.source_skill not in selected_ids or hint.target_skill not in selected_ids:
            continue
        edges.append(
            RouteEdge(
                before_skill=hint.source_skill,
                after_skill=hint.target_skill,
                edge_type=hint.projected_edge_type or "depend_on",
                confidence=hint.confidence,
                reason=hint.reason,
                source="execution_index",
            )
        )
    return _merge_edges(edges)


def _merge_edges(edges: list[RouteEdge]) -> list[RouteEdge]:
    by_key: dict[tuple[str, str, str], RouteEdge] = {}
    for edge in edges:
        existing = by_key.get(edge.key)
        if existing is None or edge.confidence > existing.confidence:
            by_key[edge.key] = edge
    return sorted(by_key.values(), key=lambda item: (-item.confidence, item.before_skill, item.after_skill))


def _near_misses_from_payload(payload: Any, candidates: dict[str, Any], selected_ids: set[str]) -> list[dict[str, str]]:
    if not isinstance(payload, list):
        return []
    output: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        skill_id = str(item.get("skill_id", ""))
        if skill_id not in candidates or skill_id in selected_ids:
            continue
        output.append({"skill_id": skill_id, "reason": str(item.get("reason", ""))})
    return output


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []
