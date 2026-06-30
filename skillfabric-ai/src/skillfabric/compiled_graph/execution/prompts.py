"""Prompt construction for execution flow validation."""

from __future__ import annotations

import json
from typing import Any

from skillfabric.compiled_graph.execution.models import ExecutionFlowCandidate
from skillfabric.compiled_graph.interface.models import SkillInterface
from skillfabric.registry.models import SkillNode

EXECUTION_PROMPT_ID = "execution_validation_handoff_precision"
COMPACT_EXECUTION_PROMPT_ID = "execution_validation_compact_interface_first"


def build_compact_execution_validation_messages(
    candidate: ExecutionFlowCandidate,
    source_skill: SkillNode,
    target_skill: SkillNode,
    *,
    interfaces: dict[str, SkillInterface],
) -> list[dict[str, str]]:
    """Build a compact interface-first execution validation prompt."""

    payload = {
        "todo": (
            "Validate one execution handoff candidate from compressed interface and candidate evidence. "
            "Use the available contract fields to decide whether the handoff should become orchestration evidence."
        ),
        "task": (
            "Validate whether source_skill makes a concrete artifact, data object, state, credential, environment condition, "
            "or validation result available for target_skill. Prefer accepted=false when the compressed evidence is not precise."
        ),
        "prompt_id": COMPACT_EXECUTION_PROMPT_ID,
        "input": {
            "candidate": "A directioned source_skill -> target_skill flow from canonical requires/produces matching.",
            "source_skill": "Possible producer or enabler represented by metadata and skill_interface.",
            "target_skill": "Possible consumer or dependent represented by metadata and skill_interface.",
            "matched_name": "The proposed reusable handoff object.",
        },
        "output": {
            "format": "Return one strict JSON object, with no markdown, comments, or extra keys.",
            "required_top_level_keys": [
                "accepted",
                "flow_type",
                "projected_edge_type",
                "confidence",
                "evidence",
                "reason",
            ],
            "flow_type_values": ["artifact_flow", "scenario_transition", "none"],
            "projected_edge_type_values": ["depend_on", "compose_with", "none"],
            "purpose": "Accepted flows become evidence for KG relations and execution ordering hints.",
        },
        "workflow": [
            "Step 1: Inspect candidate.matched_name, source_skill.skill_interface, and target_skill.skill_interface.",
            "Step 2: Verify source_skill produces, establishes, validates, or makes available the handoff.",
            "Step 3: Verify target_skill requires, consumes, or materially benefits from the same handoff.",
            "Step 4: Project strict prerequisites to depend_on and weaker collaboration handoffs to compose_with.",
            "Step 5: Return accepted=false when the evidence is generic, local-only, or not useful for routing order.",
        ],
        "rules": [
            "Return JSON only.",
            "Use artifact_flow for concrete artifact, data, text, or report handoffs.",
            "Use scenario_transition for workflow state, credential, environment, or condition handoffs.",
            "For projected_edge_type=depend_on, target_skill depends on source_skill.",
            "Reject broad matches based only on generic words such as data, output, result, text, or observation.",
            "Prefer accepted=false when the compressed evidence does not support reusable ordering or handoff value.",
        ],
        "constraints": [
            "Do not reverse source_skill and target_skill.",
            "Do not invent capabilities outside skill_interface or candidate evidence.",
            "Do not treat planning_state or belief_state as physical world_state.",
            "Do not add schema fields beyond output_schema.",
        ],
        "output_schema": {
            "accepted": True,
            "flow_type": "artifact_flow|scenario_transition|none",
            "projected_edge_type": "depend_on|compose_with|none",
            "confidence": "float between 0 and 1",
            "evidence": [{"skill": "skill id", "line": 1, "text": "verbatim evidence"}],
            "reason": "short reason",
        },
        "candidate": candidate.to_dict(),
        "source_skill": _compact_skill_payload(source_skill, interfaces),
        "target_skill": _compact_skill_payload(target_skill, interfaces),
    }
    return [
        {"role": "system", "content": "You validate SkillFabric execution flows from compact interface evidence."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def build_execution_validation_messages(
    candidate: ExecutionFlowCandidate,
    source_skill: SkillNode,
    target_skill: SkillNode,
    *,
    interfaces: dict[str, SkillInterface],
) -> list[dict[str, str]]:
    """Build LiteLLM messages for validating an execution flow candidate."""

    payload = {
        "todo": (
            "Validate one proposed execution handoff from source_skill to target_skill. Accept it only when the handoff is "
            "specific, reusable, and useful for routing, ordering, or downstream execution quality."
        ),
        "task": (
            "Validate whether an execution-level flow exists between two skills. "
            "The candidate is already directioned: source_skill is the possible producer/enabler, "
            "and target_skill is the possible consumer/dependent. Accept only flows that provide reusable "
            "ordering or handoff value for later orchestration."
        ),
        "prompt_id": EXECUTION_PROMPT_ID,
        "input": {
            "candidate": "A directioned source_skill -> target_skill flow proposed from canonical requires/produces matching.",
            "source_skill": "Possible producer or enabler, including full_skill_md and optional skill_interface.",
            "target_skill": "Possible consumer or dependent, including full_skill_md and optional skill_interface.",
            "matched_name": "The proposed artifact, data object, state, environment condition, or validation result.",
        },
        "output": {
            "format": "Return one strict JSON object, with no markdown, comments, or extra keys.",
            "required_top_level_keys": [
                "accepted",
                "flow_type",
                "projected_edge_type",
                "confidence",
                "evidence",
                "reason",
            ],
            "flow_type_values": ["artifact_flow", "scenario_transition", "none"],
            "projected_edge_type_values": ["depend_on", "compose_with", "none"],
            "purpose": "Accepted flows become evidence for KG relation validation and execution ordering hints.",
        },
        "workflow": [
            "Step 1: Name the exact handoff claimed by candidate.matched_name and candidate.flow_type.",
            "Step 2: Verify source_skill actually produces, establishes, validates, or makes available that handoff.",
            "Step 3: Verify target_skill actually consumes, requires, or is materially improved by that handoff.",
            "Step 4: Distinguish artifacts/data/text/reports from world_state, belief_state, planning_state, credentials, and environment conditions.",
            "Step 5: Reject generic matches where the handoff is only object, data, text, output, result, observation, or command.",
            "Step 6: Reject local-only outputs, examples, logs, or intermediate text that do not transfer across tasks.",
            "Step 7: Project accepted strict prerequisites to depend_on and weaker collaboration hints to compose_with.",
            "Step 8: Return accepted=false when the flow would not change routing, ordering, or downstream execution quality.",
        ],
        "decision_workflow": [
            "Identify the exact artifact, data object, world state, environment condition, or validation result claimed as the handoff.",
            "Verify that source_skill actually produces or establishes that handoff, not merely describes, plans, observes, or recommends it.",
            "Verify that target_skill actually consumes, requires, or is materially improved by that handoff.",
            "Reject broad topical matches, shared tools, generic output words, and local-only artifacts that do not transfer across tasks.",
            "Prefer accepted=false when the flow would not change routing, ordering, or downstream execution quality.",
        ],
        "direction_semantics": {
            "candidate_direction": "source_skill produces or enables; target_skill consumes or requires",
            "artifact_flow": "source_skill produces a concrete artifact/data/text/report that target_skill consumes.",
            "scenario_transition": "source_skill establishes a workflow state that target_skill requires.",
            "state_taxonomy": (
                "A world_state is a physical/environment state actually established by execution. "
                "A belief_state or planning_state is not a world-state producer and must not create depend_on workflow order."
            ),
            "projected_depend_on": (
                "For projected_edge_type=depend_on, the target skill depends on the source skill. "
                "The canonical KG edge will point target_skill -> source_skill."
            ),
            "projected_compose_with": (
                "Use projected_edge_type=compose_with only when the flow is useful as a collaboration hint "
                "but does not impose a strict prerequisite."
            ),
            "precision_goal": (
                "Execution flows should be small and high precision. A missing weak hint is safer than a false dependency."
            ),
        },
        "output_schema": {
            "accepted": True,
            "flow_type": "artifact_flow|scenario_transition|none",
            "projected_edge_type": "depend_on|compose_with|none",
            "confidence": "float between 0 and 1",
            "evidence": [{"skill": "skill id", "line": 1, "text": "verbatim evidence"}],
            "reason": "short reason",
        },
        "rules": [
            "Return JSON only.",
            "Use artifact_flow only when the target skill consumes an artifact produced by the source skill.",
            "Use scenario_transition only when the source skill enables a scenario required by the target skill.",
            "The artifact or scenario must be named specifically enough to support reuse; generic data, text, output, result, observation, or command is not enough.",
            "For projected_edge_type=depend_on, the target skill depends on the source skill; do not describe the reverse direction.",
            "Reject flows where source_skill only produces belief_state or planning_state while target_skill requires a world_state.",
            "Do not treat object_permanence_state, parsed goals, plans, remembered facts, or observations as physical object_in_inventory state.",
            "Accept object_in_inventory only when source_skill actually performs or confirms take/pickup/acquire, not when it merely reasons about object permanence.",
            "Do not accept local-only outputs that are not consumed or required by the target skill.",
            "Do not accept generic matches such as object, data, text, output, result, observation, or command unless the evidence names a specific reusable workflow object.",
            "Do not accept a flow from shared tools or broad topical similarity.",
            "Do not accept a flow merely because the two skills could be useful in the same overall task.",
            "Cite evidence lines from full_skill_md, skill_interface, or candidate_evidence.",
            "Do not invent evidence not present in the input.",
            "Prefer accepted=false when the flow would not help route ordering or state/data handoff.",
        ],
        "constraints": [
            "Do not reverse source_skill and target_skill. The candidate is already directioned.",
            "Do not treat planning_state or belief_state as a physical world_state producer.",
            "Do not accept object_in_inventory unless source_skill actually performs or confirms take, pickup, acquire, or equivalent physical possession.",
            "Do not accept flows solely from shared tools, broad topic similarity, or both skills being useful in the same overall task.",
            "Do not add schema fields beyond output_schema.",
            "If evidence is missing or ambiguous, return accepted=false with flow_type=none and projected_edge_type=none.",
        ],
        "candidate": candidate.to_dict(),
        "source_skill": _skill_payload(source_skill, interfaces),
        "target_skill": _skill_payload(target_skill, interfaces),
    }
    return [
        {"role": "system", "content": "You validate SkillFabric execution flows using strict textual evidence."},
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
        payload["skill_interface"] = interface.to_dict()
    return payload


def _compact_skill_payload(skill: SkillNode, interfaces: dict[str, SkillInterface]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": skill.id,
        "name": skill.name,
        "description": skill.description,
    }
    interface = interfaces.get(skill.id)
    if interface is not None:
        payload["skill_interface"] = interface.to_dict()
    return payload
