"""Project accepted execution flows into canonical skill-skill edges."""

from __future__ import annotations

from skillfabric.compiled_graph.execution.models import ExecutionEvidence, ExecutionValidationRecord
from skillfabric.compiled_graph.models import Edge, EvidenceRef


def project_execution_records(records: list[ExecutionValidationRecord], existing_edges: list[Edge]) -> list[Edge]:
    """Return canonical edges with accepted execution projections merged in."""

    by_key = {(edge.source, edge.target, edge.type): edge for edge in existing_edges}
    for record in records:
        edge = _edge_from_record(record)
        if edge is None:
            continue
        key = (edge.source, edge.target, edge.type)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = edge
            continue
        _merge_edge(existing, edge)
    return sorted(by_key.values(), key=lambda item: (item.type, item.source, item.target))


def _edge_from_record(record: ExecutionValidationRecord) -> Edge | None:
    if not record.accepted or record.flow_edge is None:
        return None
    projected_edge_type = str(record.normalized.get("projected_edge_type", "none"))
    confidence = float(record.normalized.get("confidence", 0.0) or 0.0)
    if projected_edge_type == "depend_on":
        source = record.candidate.target_skill
        target = record.candidate.source_skill
    elif projected_edge_type == "compose_with":
        source, target = sorted([record.candidate.source_skill, record.candidate.target_skill])
    else:
        return None
    weight = _projected_weight(projected_edge_type, confidence)
    return Edge(
        source=source,
        target=target,
        type=projected_edge_type,
        confidence=round(confidence, 6),
        weight=round(weight, 6),
        provenance="execution_projected",
        evidence=[_evidence_ref(item) for item in record.flow_edge.evidence],
        reason=str(record.normalized.get("reason", "")),
    )


def _merge_edge(existing: Edge, incoming: Edge) -> None:
    if incoming.confidence > existing.confidence:
        existing.confidence = incoming.confidence
    if incoming.weight > existing.weight:
        existing.weight = incoming.weight
        existing.provenance = incoming.provenance
    seen = {(item.skill, item.line, item.text) for item in existing.evidence}
    for item in incoming.evidence:
        key = (item.skill, item.line, item.text)
        if key not in seen:
            existing.evidence.append(item)
            seen.add(key)
    if incoming.reason and incoming.reason not in existing.reason:
        existing.reason = f"{existing.reason} {incoming.reason}".strip()


def _projected_weight(edge_type: str, confidence: float) -> float:
    if edge_type == "depend_on":
        return 1.0 * 0.9 * confidence
    return 0.8 * 0.85 * confidence


def _evidence_ref(evidence: ExecutionEvidence) -> EvidenceRef:
    return EvidenceRef(skill=evidence.skill, line=evidence.line, text=evidence.text)
