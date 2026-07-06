"""Prompt construction for relation validation."""

from __future__ import annotations

import json
from typing import Any

from skillfabric.compiled_graph.execution.models import ExecutionValidationRecord
from skillfabric.compiled_graph.interface.models import SkillInterface
from skillfabric.compiled_graph.relations.models import CandidatePair
from skillfabric.registry.models import SkillNode

RELATION_PROMPT_ID = "relation_validation_low_redundancy"
COMPACT_RELATION_PROMPT_ID = "relation_validation_compact_interface_first"


def build_compact_pair_validation_messages(
    skill_a: SkillNode,
    skill_b: SkillNode,
    pair: CandidatePair,
    *,
    interfaces: dict[str, SkillInterface] | None = None,
    execution_records: list[ExecutionValidationRecord] | None = None,
) -> list[dict[str, str]]:
    """Build a compact interface-first relation validation prompt."""

    payload = {
        "todo": (
            "Validate one SkillFabric relation candidate from compressed interface and evidence signals. "
            "Use the available contract fields and candidate evidence to decide whether the edge is useful for routing."
        ),
        "task": (
            "Return compose_with, depend_on, or none for this exact pair. Prefer a precise edge when interface fields, "
            "execution summaries, or explicit evidence support a reusable collaboration or ordering signal."
        ),
        "prompt_id": COMPACT_RELATION_PROMPT_ID,
        "input": {
            "skill_a": "First candidate skill represented by metadata and extracted skill_interface.",
            "skill_b": "Second candidate skill represented by metadata and extracted skill_interface.",
            "candidate_evidence": "Heuristic signals and line-level snippets that proposed this pair.",
            "execution_summary": "Accepted execution-level flows for this pair when available.",
            "direction_hint": "Optional direction hint from deterministic candidate generation.",
        },
        "output": {
            "format": "Return one strict JSON object, with no markdown, comments, or extra keys.",
            "required_top_level_keys": ["edge_type", "direction", "confidence", "evidence", "reason"],
            "edge_type_values": ["compose_with", "depend_on", "none"],
            "direction_values": ["A->B", "B->A", "undirected", "none"],
            "purpose": "The output becomes a graph edge only when the compressed evidence is precise enough.",
        },
        "workflow": [
            "Step 1: Read the extracted interfaces and candidate evidence as the source of truth.",
            "Step 2: Check for producer-consumer evidence across requires, produces, accepted execution summaries, or explicit mentions.",
            "Step 3: Use depend_on only when the direction follows a concrete prerequisite, artifact, state, credential, or validation handoff.",
            "Step 4: Use compose_with when the skills cover distinct complementary stages without a strict prerequisite.",
            "Step 5: Return none when the evidence only shows topical similarity, shared tools, overlap, or a redundant alternative.",
        ],
        "rules": [
            "Return JSON only.",
            "Use the candidate evidence and skill_interface fields before broader inference.",
            "Accepted edges must cite evidence that connects both candidate skills.",
            "For depend_on, point from the dependent skill to the prerequisite skill.",
            "Keep confidence calibrated to the compressed evidence. Use confidence below 0.85 for uncertain depend_on candidates.",
            "Prefer none when the compressed evidence does not support a reusable relation.",
        ],
        "constraints": [
            "Validate only skill_a and skill_b.",
            "Do not invent capabilities outside skill_interface, candidate_evidence, or execution_summary.",
            "Do not make workflow claims from shared domain or similarity alone.",
            "Do not add schema fields beyond output_schema.",
        ],
        "output_schema": {
            "edge_type": "compose_with|depend_on|none",
            "direction": "A->B|B->A|undirected|none",
            "confidence": "float between 0 and 1",
            "evidence": [{"skill": "skill id", "line": 1, "text": "verbatim evidence"}],
            "reason": "short reason",
        },
        "candidate": {
            "sources": pair.sources,
            "prior": pair.prior,
            "direction_hint": pair.direction_hint,
            "candidate_evidence": [item.to_dict() for item in pair.evidence],
            "execution_summary": _execution_summary(pair, execution_records or []),
            "skill_a": _compact_skill_payload(skill_a, interfaces or {}),
            "skill_b": _compact_skill_payload(skill_b, interfaces or {}),
        },
    }
    return [
        {"role": "system", "content": "You validate SkillFabric KG relations from compact interface evidence."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def build_pair_validation_messages(
    skill_a: SkillNode,
    skill_b: SkillNode,
    pair: CandidatePair,
    *,
    interfaces: dict[str, SkillInterface] | None = None,
    execution_records: list[ExecutionValidationRecord] | None = None,
) -> list[dict[str, str]]:
    """Build the LiteLLM messages for pairwise relation validation."""

    payload = {
        "todo": (
            "Decide whether this exact skill pair deserves one canonical KG relation for future routing and orchestration. "
            "Return no edge unless the evidence supports a reusable collaboration or ordering signal."
        ),
        "task": (
            "Validate whether the two skills have compose_with, depend_on, or no canonical KG edge. "
            "Use the full SKILL.md content, extracted interfaces, execution summaries, and candidate evidence as the source of truth. "
            "The graph should expose only reusable, evidence-backed relations that help later skill recommendation and orchestration."
        ),
        "prompt_id": RELATION_PROMPT_ID,
        "input": {
            "skill_a": "First candidate skill with metadata, full_skill_md, and optional extracted skill_interface.",
            "skill_b": "Second candidate skill with metadata, full_skill_md, and optional extracted skill_interface.",
            "candidate_evidence": "Heuristic signals that suggested this pair. These are hints, not proof.",
            "execution_summary": "Optional accepted execution-level flows for this pair. Use them as supporting evidence only.",
            "direction_hint": "Optional heuristic direction hint. Validate it against producer-consumer or prerequisite evidence before using it.",
        },
        "output": {
            "format": "Return one strict JSON object, with no markdown, comments, or extra keys.",
            "required_top_level_keys": ["edge_type", "direction", "confidence", "evidence", "reason"],
            "edge_type_values": ["compose_with", "depend_on", "none"],
            "direction_values": ["A->B", "B->A", "undirected", "none"],
            "purpose": "The output becomes a graph edge only if it is high precision enough for downstream routing.",
        },
        "workflow": [
            "Step 1: Identify each skill's primary capability, execution role, prerequisites, products, tools, and explicit boundaries.",
            "Step 2: Check for concrete producer-consumer evidence: artifact, data, state, credential, environment, or validation result.",
            "Step 3: If producer-consumer evidence exists, verify the dependency direction from consumer/dependent to producer/prerequisite.",
            "Step 4: If no strict dependency exists, check whether the skills cover distinct complementary stages of a reusable task.",
            "Step 5: Reject relations based only on shared tools, shared domain, similar names, community membership, or one-off co-occurrence.",
            "Step 6: Check whether one skill is merely a broader, narrower, or redundant alternative to the other. If so, output none.",
            "Step 7: Calibrate confidence from the strength, specificity, and directionality of evidence.",
            "Step 8: Return the strict schema with short evidence and a reason that explains the edge decision.",
        ],
        "decision_workflow": [
            "Identify each skill's primary capability, input requirements, output artifacts/states, tools, and scope boundaries.",
            "Check whether one skill's concrete output, environment state, or validated intermediate artifact is required by the other skill.",
            "Check whether the skills cover distinct complementary stages of a larger reusable task.",
            "Reject relations based only on topical similarity, shared tools, broad community membership, or the fact that both could appear in one task.",
            "Prefer no edge unless the relation would help a downstream recommender expose a smaller, more useful skill set or order selected skills safely.",
        ],
        "edge_semantics": {
            "depend_on": (
                "A depend_on B means skill A requires skill B to run first, or A consumes a state/data/artifact "
                "that B produces. The edge points from the dependent skill to its prerequisite skill."
            ),
            "compose_with": (
                "A compose_with B means the skills are complementary parts of a larger task, but neither direction "
                "is a strict prerequisite. Use compose_with for reusable collaboration, not for plain similarity."
            ),
            "producer_consumer_rule": (
                "producer -> consumer compatibility implies consumer depend_on producer. "
                "If skill A produces object X and skill B requires object X, output edge_type=depend_on "
                "and direction=B->A."
            ),
            "state_taxonomy": (
                "A world_state is a physical/environment state actually established by execution. "
                "A belief_state or planning_state is not a world-state producer and must not justify depend_on."
            ),
            "low_redundancy_goal": (
                "A graph edge is useful only when it preserves a reusable collaboration or ordering signal. "
                "Do not use edges to cluster redundant alternatives or loosely related skills."
            ),
        },
        "output_schema": {
            "edge_type": "compose_with|depend_on|none",
            "direction": "A->B|B->A|undirected|none",
            "confidence": "float between 0 and 1",
            "evidence": [{"skill": "skill id", "line": 1, "text": "verbatim evidence"}],
            "reason": "short reason",
        },
        "confidence_calibration": {
            "0.95-1.0": "Use only for explicit execution prerequisites with concrete producer-consumer or state transition evidence.",
            "0.90-0.94": "Use for clear but non-unique prerequisites, where several skills could satisfy the same requirement.",
            "0.85-0.89": "Use for conditional or context-dependent dependencies, including evidence containing may, if needed, optional, in some cases, or similar wording.",
            "<0.85": "Do not emit depend_on; use none or compose_with if appropriate.",
            "planning_context": "Dependencies based on parsed goals, task plans, routing context, or other planning_state evidence must not receive high confidence.",
        },
        "rules": [
            "Return JSON only.",
            "Interpret direction as edge direction, not execution order. A->B means A depend_on B when edge_type is depend_on.",
            "For depend_on, never point from producer to consumer. Point from consumer/dependent to producer/prerequisite.",
            "When execution_summary states source_skill produces/enables something required by target_skill, the canonical depend_on direction is target_skill -> source_skill.",
            "Do not create depend_on from belief_state or planning_state to a world_state requirement.",
            "object_permanence_state, parsed goals, plans, observations, and remembered facts are not physical object_in_inventory producers.",
            "A skill produces object_in_inventory only if it actually performs or confirms take/pickup/acquire.",
            "Use compose_with only when the skills are complementary, not merely similar.",
            "For compose_with, require that the skills cover distinct roles or stages such as extract -> transform, analyze -> report, implement -> verify, generate -> package, or configure -> validate.",
            "Do not emit compose_with when one skill is merely a broader or narrower alternative to the other; use none for redundant alternatives.",
            "Use depend_on only when there is explicit directional evidence.",
            "For depend_on, require a concrete producer-consumer, prerequisite-dependent, or state-transition relationship. A useful order hint without a required handoff should be compose_with or none.",
            "Only validate an edge between skill_a and skill_b. If the text says skill_a or skill_b depends on a third skill that is not this pair, output edge_type=none.",
            "Do not substitute skill_a or skill_b for a missing third prerequisite skill. A third skill dependency is not evidence for this candidate pair.",
            "Accepted edges must cite evidence that connects both candidate skills, either by citing lines from both skills or by citing a line in one skill that explicitly names the other skill.",
            "Shared tools, shared communities, or textual similarity are not sufficient evidence by themselves.",
            "Use none when evidence is weak or absent.",
            "Cite evidence lines from full_skill_md or candidate_evidence.",
            "Do not invent evidence not present in the provided skill content.",
            "Prefer none over a weak edge. depend_on affects workflow order and must be high precision.",
            "For large skill pools, false positive edges are worse than missing weak edges. Keep the graph small and evidence-backed.",
            "Do not create both directions of depend_on for the same operational object. Choose the direction that follows the producer/consumer or prerequisite/dependent evidence.",
            "Use confidence as the only edge usability signal; do not output planner_usable, support_level, dependency_scope, or any extra schema fields.",
            "If a dependency is conditional, optional, or context-dependent, keep confidence between 0.85 and 0.89 and say so in reason.",
            "If a dependency is based on planning context rather than execution state, keep confidence below 0.90.",
            "If a dependency is an explicit hard execution prerequisite, use confidence >= 0.95.",
        ],
        "constraints": [
            "Validate only skill_a and skill_b. Do not substitute a missing third skill.",
            "Do not create both directions for the same operational object.",
            "Do not output planner_usable, support_level, dependency_scope, or any schema field not listed in output_schema.",
            "Do not invent evidence or infer hidden requirements from general world knowledge.",
            "Do not accept weak edges to make the graph denser. A small high-precision graph is preferred.",
            "If the pair could be useful together only for one specific task but lacks reusable handoff evidence, output none.",
        ],
        "candidate": {
            "sources": pair.sources,
            "prior": pair.prior,
            "direction_hint": pair.direction_hint,
            "candidate_evidence": [item.to_dict() for item in pair.evidence],
            "execution_summary": _execution_summary(pair, execution_records or []),
            "skill_a": _skill_payload(skill_a, interfaces or {}),
            "skill_b": _skill_payload(skill_b, interfaces or {}),
        },
    }
    return [
        {"role": "system", "content": "You validate SkillFabric KG relations using strict textual evidence."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def _skill_payload(skill: SkillNode, interfaces: dict[str, SkillInterface]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": skill.id,
        "name": skill.name,
        "description": skill.description,
        "full_skill_md": [
            {"line": line_number, "text": line}
            for line_number, line in enumerate(skill.raw_text.splitlines(), start=1)
        ],
    }
    interface = interfaces.get(skill.id)
    if interface is not None:
        payload["skill_interface"] = {
            "capability_summary": interface.capability_summary,
            "when_to_use": interface.when_to_use,
            "requires": [item.to_dict() for item in interface.requires],
            "produces": [item.to_dict() for item in interface.produces],
            "uses_tools": [item.to_dict() for item in interface.uses_tools],
        }
    return payload


def _compact_skill_payload(skill: SkillNode, interfaces: dict[str, SkillInterface]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": skill.id,
        "name": skill.name,
        "description": skill.description,
    }
    interface = interfaces.get(skill.id)
    if interface is not None:
        payload["skill_interface"] = {
            "capability_summary": interface.capability_summary,
            "when_to_use": interface.when_to_use,
            "requires": [item.to_dict() for item in interface.requires],
            "produces": [item.to_dict() for item in interface.produces],
            "uses_tools": [item.to_dict() for item in interface.uses_tools],
        }
    return payload


def _execution_summary(
    pair: CandidatePair,
    records: list[ExecutionValidationRecord],
) -> list[dict[str, Any]]:
    pair_skills = {pair.skill_a, pair.skill_b}
    summary: list[dict[str, Any]] = []
    for record in records:
        if not record.accepted or record.flow_edge is None:
            continue
        if {record.candidate.source_skill, record.candidate.target_skill} != pair_skills:
            continue
        summary.append(
            {
                "flow_type": record.candidate.flow_type,
                "source_skill": record.candidate.source_skill,
                "target_skill": record.candidate.target_skill,
                "matched_node_id": record.candidate.matched_node_id,
                "matched_name": record.candidate.matched_name,
                "projected_edge_type": record.normalized.get("projected_edge_type", "none"),
                "confidence": record.normalized.get("confidence", 0.0),
                "evidence": [item.to_dict() for item in record.flow_edge.evidence],
                "reason": record.normalized.get("reason", ""),
            }
        )
    return summary
