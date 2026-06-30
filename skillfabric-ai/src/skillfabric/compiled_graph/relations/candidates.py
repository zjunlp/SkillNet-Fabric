"""Generate relation candidates before pairwise validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from skillfabric.compiled_graph.execution.models import ExecutionValidationRecord
from skillfabric.compiled_graph.interface.models import InterfaceField, SkillInterface
from skillfabric.compiled_graph.models import Edge
from skillfabric.compiled_graph.relations.mentions import extract_skill_mentions
from skillfabric.compiled_graph.relations.models import CandidatePair, RelationEvidence
from skillfabric.registry.models import SkillNode


def generate_relation_candidates(
    skills: list[SkillNode],
    similar_edges: list[Edge],
    *,
    per_skill_limit: int,
    interfaces: dict[str, SkillInterface] | None = None,
    execution_records: list[ExecutionValidationRecord] | None = None,
) -> list[CandidatePair]:
    """Generate bounded candidates for compose_with and depend_on validation."""

    by_id = {skill.id: skill for skill in skills}
    pairs: dict[tuple[str, str], CandidatePair] = {}
    for mention in extract_skill_mentions(skills):
        if mention.from_skill not in by_id or mention.to_skill not in by_id:
            continue
        _merge_pair(
            pairs,
            CandidatePair(
                mention.from_skill,
                mention.to_skill,
                1.0,
                sources=["explicit_mention"],
                evidence=[mention.to_evidence()],
                direction_hint=mention.direction_hint,
            ),
        )
    for edge in similar_edges:
        _merge_pair(
            pairs,
            CandidatePair(
                edge.source,
                edge.target,
                0.55 + min(edge.weight, 0.4),
                sources=["similar_neighbor"],
            ),
        )
    for pair in _interface_candidate_pairs(interfaces or {}):
        _merge_pair(pairs, pair)
    for pair in _execution_candidate_pairs(execution_records or []):
        _merge_pair(pairs, pair)
    ordered = sorted(pairs.values(), key=lambda item: (-item.prior, -len(item.evidence), item.key))
    return _limit_per_skill(ordered, per_skill_limit)


def _merge_pair(pairs: dict[tuple[str, str], CandidatePair], incoming: CandidatePair) -> None:
    existing = pairs.get(incoming.key)
    if existing is None:
        pairs[incoming.key] = incoming
        return
    existing.prior = max(existing.prior, incoming.prior)
    for source in incoming.sources:
        if source not in existing.sources:
            existing.sources.append(source)
    seen = {item.key for item in existing.evidence}
    for item in incoming.evidence:
        if item.key not in seen:
            existing.evidence.append(item)
            seen.add(item.key)
    existing.direction_hint = _merge_direction(existing.direction_hint, incoming.direction_hint)


def _merge_direction(left: str, right: str) -> str:
    if left == "none":
        return right
    if right == "none" or left == right:
        return left
    return "none"


def _limit_per_skill(pairs: list[CandidatePair], limit: int) -> list[CandidatePair]:
    counts: dict[str, int] = {}
    kept: list[CandidatePair] = []
    for pair in pairs:
        if counts.get(pair.skill_a, 0) >= limit or counts.get(pair.skill_b, 0) >= limit:
            continue
        kept.append(pair)
        counts[pair.skill_a] = counts.get(pair.skill_a, 0) + 1
        counts[pair.skill_b] = counts.get(pair.skill_b, 0) + 1
    return kept


def _interface_candidate_pairs(interfaces: dict[str, SkillInterface]) -> list[CandidatePair]:
    pairs: dict[tuple[str, str], CandidatePair] = {}
    produce_index = _field_index(interfaces.values(), "produces")
    _add_interface_matches(pairs, produce_index, interfaces.values(), "requires")
    return list(pairs.values())


def _field_index(
    interfaces: Iterable[SkillInterface],
    field_group: str,
) -> dict[str, list[tuple[SkillInterface, InterfaceField]]]:
    index: dict[str, list[tuple[SkillInterface, InterfaceField]]] = defaultdict(list)
    for interface in interfaces:
        for field in _interface_fields(interface, field_group):
            normalized = _normalize_field_name(field.name)
            if normalized:
                index[normalized].append((interface, field))
    return index


def _add_interface_matches(
    pairs: dict[tuple[str, str], CandidatePair],
    producer_index: dict[str, list[tuple[SkillInterface, InterfaceField]]],
    consumers: Iterable[SkillInterface],
    consumer_field_group: str,
) -> None:
    for consumer in consumers:
        for consumed in _interface_fields(consumer, consumer_field_group):
            for producer, produced in producer_index.get(_normalize_field_name(consumed.name), []):
                if producer.skill_id == consumer.skill_id:
                    continue
                pair = CandidatePair(
                    consumer.skill_id,
                    producer.skill_id,
                    0.85,
                    sources=["interface_compatibility"],
                    evidence=_interface_evidence(producer, consumer, [(produced, consumed)]),
                    direction_hint="A->B",
                )
                _merge_pair(pairs, pair)


def _interface_fields(interface: SkillInterface, field_group: str) -> list[InterfaceField]:
    return list(getattr(interface, field_group))


def _interface_evidence(
    producer: SkillInterface,
    consumer: SkillInterface,
    matches: list[tuple[InterfaceField, InterfaceField]],
) -> list[RelationEvidence]:
    evidence: list[RelationEvidence] = []
    for produced, consumed in matches[:3]:
        evidence.extend(_field_evidence(producer.skill_id, produced, "producer_field"))
        evidence.extend(_field_evidence(consumer.skill_id, consumed, "consumer_field"))
    return evidence


def _field_evidence(skill_id: str, field: InterfaceField, role: str) -> list[RelationEvidence]:
    if not field.evidence:
        return [
            RelationEvidence(
                source="interface_compatibility",
                skill_id=skill_id,
                line=0,
                text=field.name,
                kind="interface",
                metadata={"field_role": role, "field_kind": field.kind, "field_name": field.name},
            )
        ]
    return [
        RelationEvidence(
            source="interface_compatibility",
            skill_id=skill_id,
            line=item.line,
            text=item.text,
            kind="interface",
            metadata={"field_role": role, "field_kind": field.kind, "field_name": field.name},
        )
        for item in field.evidence
    ]


def _normalize_field_name(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def _execution_candidate_pairs(records: list[ExecutionValidationRecord]) -> list[CandidatePair]:
    pairs: list[CandidatePair] = []
    for record in records:
        if not record.accepted or record.flow_edge is None:
            continue
        projected_edge_type = str(record.normalized.get("projected_edge_type", "none"))
        if projected_edge_type == "depend_on":
            pairs.append(
                CandidatePair(
                    record.candidate.target_skill,
                    record.candidate.source_skill,
                    0.95,
                    sources=["execution_flow"],
                    evidence=_execution_evidence(record),
                    direction_hint="A->B",
                )
            )
        elif projected_edge_type == "compose_with":
            pairs.append(
                CandidatePair(
                    record.candidate.source_skill,
                    record.candidate.target_skill,
                    0.9,
                    sources=["execution_flow"],
                    evidence=_execution_evidence(record),
                    direction_hint="undirected",
                )
            )
    return pairs


def _execution_evidence(record: ExecutionValidationRecord) -> list[RelationEvidence]:
    if record.flow_edge is None:
        return []
    return [
        RelationEvidence(
            source="execution_flow",
            skill_id=item.skill,
            line=item.line,
            text=item.text,
            kind=record.candidate.flow_type,
            metadata={
                "matched_node_id": record.candidate.matched_node_id,
                "matched_name": record.candidate.matched_name,
                "projected_edge_type": str(record.normalized.get("projected_edge_type", "none")),
            },
        )
        for item in record.flow_edge.evidence
    ]
