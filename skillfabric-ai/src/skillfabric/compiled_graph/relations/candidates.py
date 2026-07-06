"""Generate relation candidates before pairwise validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from skillfabric.compiled_graph.execution.models import ExecutionValidationRecord
from skillfabric.compiled_graph.interface.models import InterfaceField, SkillInterface
from skillfabric.compiled_graph.models import Edge
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
    """Generate bounded workflow-relation candidates from interface contracts.

    ``similar_to`` edges are retrieval/community signals, not workflow evidence.
    Explicit textual mentions are intentionally ignored here because broad tool
    names and examples create noisy graph frontier edges. Accepted execution
    flows are projected directly into graph edges by the execution layer, so
    re-validating them as relation candidates would duplicate cost.
    """

    pairs: dict[tuple[str, str], CandidatePair] = {}
    execution_covered_keys = _execution_covered_pair_keys(execution_records or [])
    for pair in _interface_candidate_pairs(interfaces or {}):
        if pair.key in execution_covered_keys:
            continue
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


def _execution_covered_pair_keys(records: Iterable[ExecutionValidationRecord]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for record in records:
        candidate = getattr(record, "candidate", None)
        if candidate is None:
            continue
        keys.add(tuple(sorted([candidate.source_skill, candidate.target_skill])))
    return keys
