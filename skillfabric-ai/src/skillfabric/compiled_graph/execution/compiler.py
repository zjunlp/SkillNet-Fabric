"""Compile skill interfaces into execution graph candidates."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from skillfabric.compiled_graph.canonicalization.models import CanonicalizationBuild
from skillfabric.compiled_graph.execution.models import (
    ArtifactNode,
    ExecutionEdge,
    ExecutionEvidence,
    ExecutionFlowCandidate,
    ExecutionIndexRecord,
    ExecutionValidationRecord,
    ScenarioNode,
)
from skillfabric.compiled_graph.interface.models import InterfaceField, SkillInterface


@dataclass(slots=True)
class ExecutionGraphBuild:
    """Execution graph compilation output before validation."""

    candidates: list[ExecutionFlowCandidate] = field(default_factory=list)
    execution_index: list[ExecutionIndexRecord] = field(default_factory=list)
    raw_artifact_nodes: list[ArtifactNode] = field(default_factory=list)
    raw_scenario_nodes: list[ScenarioNode] = field(default_factory=list)
    raw_skill_artifact_edges: list[ExecutionEdge] = field(default_factory=list)
    raw_skill_scenario_edges: list[ExecutionEdge] = field(default_factory=list)
    canonical_aliases: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def compile_execution_graph(
    interfaces: dict[str, SkillInterface],
    *,
    bucket_limit: int = 100,
    canonicalization: CanonicalizationBuild | None = None,
) -> ExecutionGraphBuild:
    """Compile interfaces into execution nodes, skill-node edges, and flow candidates."""

    raw_artifact_nodes: dict[str, ArtifactNode] = {}
    raw_scenario_nodes: dict[str, ScenarioNode] = {}
    raw_skill_artifact_edges: list[ExecutionEdge] = []
    raw_skill_scenario_edges: list[ExecutionEdge] = []

    for interface in interfaces.values():
        _add_contract_fields(
            interface,
            "produces",
            "produces_artifact",
            "enables_scenario",
            raw_artifact_nodes,
            raw_scenario_nodes,
            raw_skill_artifact_edges,
            raw_skill_scenario_edges,
        )
        _add_contract_fields(
            interface,
            "requires",
            "consumes_artifact",
            "requires_scenario",
            raw_artifact_nodes,
            raw_scenario_nodes,
            raw_skill_artifact_edges,
            raw_skill_scenario_edges,
        )

    warnings: list[str] = []
    if canonicalization is None:
        warnings.append("execution compatibility requires accepted canonicalization assignments")
    candidates = [
        *_compatibility_candidates(
            interfaces.values(),
            producer_group="produces",
            consumer_group="requires",
            flow_type="artifact_flow",
            relation_type="artifact_compatibility",
            bucket_limit=bucket_limit,
            kind_filter=_is_artifact_like,
            canonicalization=canonicalization,
        ),
        *_compatibility_candidates(
            interfaces.values(),
            producer_group="produces",
            consumer_group="requires",
            flow_type="scenario_transition",
            relation_type="state_compatibility",
            bucket_limit=bucket_limit,
            kind_filter=_is_execution_state_like,
            canonicalization=canonicalization,
        ),
    ]
    candidates.sort(key=lambda item: item.key)
    raw_artifacts = sorted(raw_artifact_nodes.values(), key=lambda item: item.id)
    raw_scenarios = sorted(raw_scenario_nodes.values(), key=lambda item: item.id)
    return ExecutionGraphBuild(
        raw_artifact_nodes=raw_artifacts,
        raw_scenario_nodes=raw_scenarios,
        raw_skill_artifact_edges=sorted(raw_skill_artifact_edges, key=lambda item: item.key),
        raw_skill_scenario_edges=sorted(raw_skill_scenario_edges, key=lambda item: item.key),
        canonical_aliases=_canonical_aliases(raw_artifacts, raw_scenarios),
        candidates=candidates,
        warnings=warnings,
    )


def _add_contract_fields(
    interface: SkillInterface,
    field_group: str,
    artifact_edge_type: str,
    scenario_edge_type: str,
    artifact_nodes: dict[str, ArtifactNode],
    scenario_nodes: dict[str, ScenarioNode],
    artifact_edges: list[ExecutionEdge],
    scenario_edges: list[ExecutionEdge],
) -> None:
    for interface_field in _interface_fields(interface, field_group):
        if _is_state_like(interface_field):
            _add_scenario_field(interface, interface_field, scenario_edge_type, scenario_nodes, scenario_edges)
            continue
        _add_artifact_field(interface, interface_field, artifact_edge_type, artifact_nodes, artifact_edges)


def _add_artifact_field(
    interface: SkillInterface,
    field: InterfaceField,
    edge_type: str,
    nodes: dict[str, ArtifactNode],
    edges: list[ExecutionEdge],
) -> None:
    normalized = _normalize_name(field.name)
    if not normalized:
        return
    node_id = f"artifact:{_slug(normalized)}"
    evidence = _execution_evidence(field)
    node = nodes.setdefault(
        node_id,
        ArtifactNode(
            id=node_id,
            type="artifact",
            name=field.name,
            normalized_name=normalized,
            kind=field.kind,
            evidence=[],
        )
    )
    _merge_evidence(node.evidence, evidence)
    edges.append(
        ExecutionEdge(
            source=interface.skill_id,
            target=node_id,
            type=edge_type,
            confidence=field.confidence or 0.65,
            evidence=evidence,
            metadata={"artifact_id": node_id},
        )
    )


def _add_scenario_field(
    interface: SkillInterface,
    field: InterfaceField,
    edge_type: str,
    nodes: dict[str, ScenarioNode],
    edges: list[ExecutionEdge],
) -> None:
    normalized = _normalize_name(field.name)
    if not normalized:
        return
    node_id = f"scenario:{_slug(normalized)}"
    evidence = _execution_evidence(field)
    node = nodes.setdefault(
        node_id,
        ScenarioNode(
            id=node_id,
            type="scenario",
            name=field.name,
            normalized_name=normalized,
            kind=field.kind,
            evidence=[],
        )
    )
    _merge_evidence(node.evidence, evidence)
    edges.append(
        ExecutionEdge(
            source=interface.skill_id,
            target=node_id,
            type=edge_type,
            confidence=field.confidence or 0.65,
            evidence=evidence,
            metadata={"scenario_id": node_id},
        )
    )


def _compatibility_candidates(
    interfaces: Iterable[SkillInterface],
    *,
    producer_group: str,
    consumer_group: str,
    flow_type: str,
    relation_type: str,
    bucket_limit: int,
    kind_filter,
    canonicalization: CanonicalizationBuild | None,
) -> list[ExecutionFlowCandidate]:
    interface_list = list(interfaces)
    producer_index = _field_index(
        interface_list,
        producer_group,
        kind_filter=kind_filter,
        canonicalization=canonicalization,
    )
    consumer_index = _field_index(
        interface_list,
        consumer_group,
        kind_filter=kind_filter,
        canonicalization=canonicalization,
    )
    candidates: dict[tuple[str, str, str, str], ExecutionFlowCandidate] = {}
    for canonical_id, producers in producer_index.items():
        if _is_broad_handoff_object(canonical_id):
            continue
        consumers = consumer_index.get(canonical_id, [])
        if not consumers:
            continue
        if _unique_candidate_pair_count(producers, consumers) > bucket_limit:
            continue
        matched_name = canonical_id.split(":", 1)[-1]
        node_id = f"canonical:{relation_type}:{_slug(matched_name)}"
        for producer, produced in producers:
            for consumer, consumed in consumers:
                if producer.skill_id == consumer.skill_id:
                    continue
                evidence = [*_execution_evidence(produced), *_execution_evidence(consumed)]
                candidate = ExecutionFlowCandidate(
                    source_skill=producer.skill_id,
                    target_skill=consumer.skill_id,
                    flow_type=flow_type,
                    matched_node_id=node_id,
                    matched_name=matched_name,
                    evidence=_unique_evidence(evidence),
                    metadata={
                        "matched_name": matched_name,
                        "relation_type": relation_type,
                        "canonical_object": matched_name,
                        "canonical_object_id": canonical_id,
                        "direction": "source_to_target",
                    },
                )
                existing = candidates.get(candidate.key)
                if existing is None:
                    candidates[candidate.key] = candidate
                else:
                    existing.evidence = _unique_evidence([*existing.evidence, *candidate.evidence])
                    existing.prior = max(existing.prior, candidate.prior)
                    existing.metadata = {**existing.metadata, **candidate.metadata}
    return list(candidates.values())


def _unique_candidate_pair_count(
    producers: list[tuple[SkillInterface, InterfaceField]],
    consumers: list[tuple[SkillInterface, InterfaceField]],
) -> int:
    pairs = {
        (producer.skill_id, consumer.skill_id)
        for producer, _produced in producers
        for consumer, _consumed in consumers
        if producer.skill_id != consumer.skill_id
    }
    return len(pairs)


def _field_index(
    interfaces: Iterable[SkillInterface],
    field_group: str,
    *,
    kind_filter,
    canonicalization: CanonicalizationBuild | None,
) -> dict[str, list[tuple[SkillInterface, InterfaceField]]]:
    index: dict[str, list[tuple[SkillInterface, InterfaceField]]] = defaultdict(list)
    for interface in interfaces:
        for interface_field in _interface_fields(interface, field_group):
            if not kind_filter(interface_field):
                continue
            if field_group == "requires" and _is_non_consumed_artifact_requirement(interface_field):
                continue
            canonical_id = _canonical_id(interface, field_group, interface_field, canonicalization)
            if canonical_id:
                index[canonical_id].append((interface, interface_field))
    return index


def _canonical_id(
    interface: SkillInterface,
    field_group: str,
    field: InterfaceField,
    canonicalization: CanonicalizationBuild | None,
) -> str:
    if canonicalization is not None:
        canonical_id = canonicalization.lookup(interface.skill_id, field_group, field.name, field.kind)
        if canonical_id and not _canonical_id_compatible_with_field(canonical_id, field):
            return ""
        if canonical_id:
            return canonical_id
    return ""


def _canonical_id_compatible_with_field(canonical_id: str, field: InterfaceField) -> bool:
    canonical_type = canonical_id.split(":", 1)[0]
    if canonical_type in {"belief_state", "planning_state"}:
        return False
    if canonical_type == "state":
        return _is_execution_state_like(field)
    return _is_artifact_like(field)


def _is_broad_handoff_object(canonical_id: str) -> bool:
    _object_type, _, raw_name = canonical_id.partition(":")
    tokens = set(_normalize_name(raw_name).split())
    if not tokens:
        return True
    if _is_exact_underspecified_name(tokens):
        return True
    if _looks_like_context_or_input_placeholder(tokens):
        return True
    if _looks_like_path_placeholder(tokens):
        return True
    return False


def _is_exact_underspecified_name(tokens: set[str]) -> bool:
    return len(tokens) == 1 and next(iter(tokens)) in {
        "artifact",
        "content",
        "data",
        "file",
        "input",
        "object",
        "output",
        "result",
        "state",
        "text",
    }


def _looks_like_context_or_input_placeholder(tokens: set[str]) -> bool:
    broad_context_tokens = {
        "context",
        "input",
        "material",
        "materials",
        "reference",
        "references",
        "snippet",
        "snippets",
        "source",
    }
    if not (tokens & broad_context_tokens):
        return False
    informative_tokens = tokens - broad_context_tokens - {
        "artifact",
        "artifacts",
        "data",
        "document",
        "documents",
        "file",
        "files",
        "project",
        "task",
        "text",
    }
    return not informative_tokens


def _looks_like_path_placeholder(tokens: set[str]) -> bool:
    if not ({"path", "directory"} & tokens):
        return False
    return tokens <= {"directory", "file", "input", "output", "path", "project", "target"}


def execution_index_from_validation_records(records: list[ExecutionValidationRecord]) -> list[ExecutionIndexRecord]:
    """Build handoff-facing compatibility records from accepted validation results."""

    output: dict[tuple[str, str, str, str, str, str], ExecutionIndexRecord] = {}
    for record in records:
        if not record.accepted or record.flow_edge is None:
            continue
        candidate = record.candidate
        relation_type = candidate.metadata.get("relation_type", "artifact_compatibility")
        item = ExecutionIndexRecord(
            source_skill=candidate.source_skill,
            target_skill=candidate.target_skill,
            relation_type=relation_type,
            canonical_object=candidate.metadata.get("canonical_object", candidate.matched_name),
            direction=candidate.metadata.get("direction", "source_to_target"),
            confidence=float(record.normalized.get("confidence", record.flow_edge.confidence) or 0.0),
            evidence=record.flow_edge.evidence,
            projected_edge_type=str(record.normalized.get("projected_edge_type", "depend_on")),
            reason=str(record.normalized.get("reason", record.flow_edge.reason)),
            metadata={**candidate.metadata, "flow_type": candidate.flow_type},
        )
        key = _execution_index_key(item)
        if key in output:
            _merge_execution_index_record(output[key], item)
        else:
            output[key] = item
    return sorted(
        output.values(),
        key=lambda item: (item.source_skill, item.target_skill, item.relation_type, item.canonical_object),
    )


def _execution_index_key(item: ExecutionIndexRecord) -> tuple[str, str, str, str, str, str]:
    return (
        item.source_skill,
        item.target_skill,
        item.relation_type,
        item.canonical_object,
        item.direction,
        item.projected_edge_type,
    )


def _merge_execution_index_record(existing: ExecutionIndexRecord, incoming: ExecutionIndexRecord) -> None:
    existing.confidence = max(existing.confidence, incoming.confidence)
    existing.evidence = _unique_evidence([*existing.evidence, *incoming.evidence])
    existing.reason = _merge_reasons(existing.reason, incoming.reason)
    existing.metadata = {**existing.metadata, **incoming.metadata}


def _merge_reasons(first: str, second: str) -> str:
    reasons: list[str] = []
    for reason in (first, second):
        cleaned = reason.strip()
        if cleaned and cleaned not in reasons:
            reasons.append(cleaned)
    return " | ".join(reasons)


def _canonical_aliases(
    raw_artifacts: list[ArtifactNode],
    raw_scenarios: list[ScenarioNode],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in raw_artifacts:
        canonical = "_".join(_normalize_name(node.name).split())
        if canonical:
            aliases[node.id] = f"artifact:{canonical}"
    for node in raw_scenarios:
        canonical = _canonical_state_name(node.name)
        if canonical:
            aliases[node.id] = f"state:{canonical}"
    return dict(sorted(aliases.items()))


def _interface_fields(interface: SkillInterface, field_group: str) -> list[InterfaceField]:
    return list(getattr(interface, field_group))


def _is_state_like(field: InterfaceField) -> bool:
    return _field_kind(field) in {
        "state",
        "condition",
        "world_state",
        "physical_state",
        "environment_state",
        "belief_state",
        "memory_state",
        "knowledge_state",
        "observation_state",
        "planning_state",
        "plan_state",
        "routing_state",
        "credential",
        "environment",
    }


def _is_artifact_like(field: InterfaceField) -> bool:
    return not _is_state_like(field) and not _looks_like_world_state(field.name)


def _is_non_consumed_artifact_requirement(field: InterfaceField) -> bool:
    normalized = _normalize_name(field.name)
    tokens = set(normalized.split())
    if not tokens:
        return False
    if {"destination", "sink", "target"} & tokens and {"output", "markdown", "file", "path", "directory"} & tokens:
        return True
    if "output" in tokens and {"path", "directory"} & tokens:
        return True
    evidence_text = " ".join(item.text.lower() for item in field.evidence)
    if _looks_like_optional_user_supplied_artifact(tokens, evidence_text):
        return True
    if _looks_like_view_only_reference_artifact(tokens, evidence_text):
        return True
    if _looks_like_local_template_reference_artifact(tokens, evidence_text):
        return True
    if {"md", "markdown"} & tokens and _looks_like_local_reference_evidence(evidence_text):
        return True
    return False


def _looks_like_optional_user_supplied_artifact(tokens: set[str], evidence_text: str) -> bool:
    if not evidence_text:
        return False
    if "optional" in tokens or "uploaded" in tokens:
        return True
    markers = (
        "if a user uploads",
        "if user uploads",
        "if the user uploads",
        "if a user provides",
        "if user provides",
        "if the user provides",
        "input image path for editing",
        "enables edit mode",
        "optional input",
        "user uploads",
        "user uploaded",
    )
    return any(marker in evidence_text for marker in markers)


def _looks_like_view_only_reference_artifact(tokens: set[str], evidence_text: str) -> bool:
    if not evidence_text:
        return False
    if not ({"showcase", "sample", "example", "demo", "reference"} & tokens):
        return False
    view_markers = (
        "display",
        "show the",
        "showcase",
        "viewing",
        "for viewing",
        "see all available",
        "do not make any modifications",
        "simply show",
    )
    return any(marker in evidence_text for marker in view_markers)


def _looks_like_local_template_reference_artifact(tokens: set[str], evidence_text: str) -> bool:
    if not evidence_text:
        return False
    if "template" not in tokens and "templates" not in tokens:
        return False
    template_markers = (
        "required starting point",
        "literal starting point",
        "read tool",
        "read the template",
        "read `templates/",
        "templates/",
        "bundled template",
        "template file",
    )
    return any(marker in evidence_text for marker in template_markers)


def _looks_like_local_reference_evidence(value: str) -> bool:
    if not value:
        return False
    reference_markers = (
        "read entire file",
        "read the full file",
        "syntax",
        "best practices",
        "guidance",
        "reference",
        "instructions",
    )
    return any(marker in value for marker in reference_markers)


def _is_execution_state_like(field: InterfaceField) -> bool:
    kind = _field_kind(field)
    if kind in {
        "belief_state",
        "memory_state",
        "knowledge_state",
        "observation_state",
        "planning_state",
        "plan_state",
        "routing_state",
    }:
        return False
    if kind in {"state", "condition", "world_state", "physical_state", "environment_state", "credential", "environment"}:
        return True
    return _looks_like_world_state(field.name)


def _field_kind(field: InterfaceField) -> str:
    return field.kind.lower().strip().replace("-", "_").replace(" ", "_")


def _execution_evidence(field: InterfaceField) -> list[ExecutionEvidence]:
    return [
        ExecutionEvidence(skill=item.skill, line=item.line, text=item.text)
        for item in field.evidence
    ]


def _merge_evidence(target: list[ExecutionEvidence], incoming: list[ExecutionEvidence]) -> None:
    seen = {item.key for item in target}
    for item in incoming:
        if item.key not in seen:
            target.append(item)
            seen.add(item.key)


def _unique_evidence(evidence: list[ExecutionEvidence]) -> list[ExecutionEvidence]:
    output: list[ExecutionEvidence] = []
    _merge_evidence(output, evidence)
    return output


def _normalize_name(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def _canonical_state_name(value: str) -> str:
    normalized = _normalize_name(value)
    if not normalized:
        return ""
    tokens = normalized.split()
    token_set = set(tokens)
    if "authenticated" in token_set or "authorization" in token_set or "oauth" in token_set:
        return "authenticated_session"
    if "validated" in token_set or "verified" in token_set:
        return "validated_result_available"
    if "created" in token_set or "generated" in token_set or "available" in token_set:
        domain = [token for token in tokens if token not in {"created", "generated", "available", "ready", "exists"}]
        if not domain:
            return ""
        return "_".join(domain[:4] + ["available"])
    return "_".join(tokens)


def _looks_like_world_state(value: str) -> bool:
    tokens = set(_normalize_name(value).split())
    phrase = " ".join(tokens)
    if not tokens:
        return False
    if {"inventory", "held", "holding", "open", "closed", "clean", "cleaned", "heated", "cooled"} & tokens:
        return True
    if "located" in tokens or "accessible" in tokens or "authenticated" in tokens:
        return True
    raw = value.lower()
    if "in_hand" in raw or "in_inventory" in raw:
        return True
    return any(item in phrase for item in ("object inventory", "object hand", "receptacle open", "receptacle closed"))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
