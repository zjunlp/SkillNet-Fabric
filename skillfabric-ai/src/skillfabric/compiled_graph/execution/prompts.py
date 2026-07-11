"""Prompt construction for execution flow validation."""

from __future__ import annotations

import json
from typing import Any

from skillfabric.compiled_graph.execution.models import ExecutionFlowCandidate
from skillfabric.compiled_graph.interface.models import SkillInterface
from skillfabric.registry.models import SkillNode

EXECUTION_PROMPT_ID = "execution_validation_handoff_precision_v2"
COMPACT_EXECUTION_PROMPT_ID = "execution_validation_compact_handoff_v2"

_OUTPUT_SCHEMA: dict[str, Any] = {
    "accepted": True,
    "flow_type": "artifact_handoff|state_handoff|none",
    "projected_edge_type": "depend_on|compose_with|none",
    "confidence": "float between 0 and 1",
    "evidence": [{"skill": "skill id", "line": 1, "text": "verbatim evidence"}],
    "needs_full_context": False,
}

_OUTPUT_CONTRACT: dict[str, Any] = {
    "format": "Return exactly one strict JSON object, with no markdown, comments, or extra keys.",
    "required_top_level_keys": list(_OUTPUT_SCHEMA),
    "flow_type_values": ["artifact_handoff", "state_handoff", "none"],
    "projected_edge_type_values": ["depend_on", "compose_with", "none"],
    "purpose": "Accepted flows become SkillFabric graph evidence and execution ordering hints.",
}

_EDGE_SEMANTICS: dict[str, str] = {
    "candidate_direction": "source_skill produces or enables a post-state; target_skill consumes or requires a pre-state",
    "depend_on": (
        "Use projected_edge_type=depend_on for a strict consumable handoff: "
        "target_skill cannot run correctly, or loses its core purpose, without the concrete artifact/state from source_skill. "
        "For projected_edge_type=depend_on, the target skill depends on the source skill."
    ),
    "compose_with": (
        "Use projected_edge_type=compose_with for a non-strict workflow progression: "
        "source_skill creates context or partial material that makes target_skill materially better, "
        "but target_skill is not a hard consumer of that exact handoff."
    ),
    "none": (
        "Use projected_edge_type=none when the pair is only topically related, interchangeable, "
        "directionally wrong, generic, local-only, or not a reusable handoff."
    ),
    "artifact_handoff": "source_skill produces a concrete artifact/data/text/report and target_skill consumes that artifact.",
    "state_handoff": "source_skill establishes a reusable workflow state, credential, or environment condition required by target_skill.",
    "state_taxonomy": (
        "A world_state is a physical/environment state actually established by execution. "
        "belief_state and planning_state are non-execution cognitive context and should not appear as execution handoff candidates."
    ),
    "precision_goal": "Execution flows should be small and high precision. A missing weak hint is safer than a false dependency.",
}

_REJECTION_TAXONOMY: dict[str, str] = {
    "topical_only": "Reject shared topic, shared tool, broad domain overlap, or both skills being useful in the same overall task.",
    "duplicate_or_alternative": "Reject skills that solve the same step, are substitutes, or restate the same state without a consumer handoff.",
    "wrong_direction": "Reject when target_skill produces the state or source_skill consumes it; do not reverse the candidate.",
    "generic_or_underspecified": "Reject generic data, text, output, result, object, observation, file, or command matches.",
    "local_only": "Reject examples, logs, prompts, temporary paths, setup notes, and intermediate text that do not transfer across skills.",
    "state_mismatch": "Reject cognitive context or planning evidence as proof of required world_state, inventory, or physical environment state.",
    "unsupported_by_evidence": "Reject when candidate.evidence, skill_interface, or full_skill_md does not prove both sides of the handoff.",
}

_COMPACT_REJECTION_TAXONOMY = {
    key: value.replace(", or full_skill_md", "")
    for key, value in _REJECTION_TAXONOMY.items()
}

_DECISION_PROCEDURE: list[str] = [
    "Identify the exact post-state claimed by source_skill and the exact pre-state claimed by target_skill.",
    "Verify source_skill actually produces, establishes, validates, or makes available that post-state.",
    "Verify target_skill actually requires, consumes, or materially benefits from that same pre-state.",
    "Classify as depend_on only for a strict consumable handoff.",
    "Classify as compose_with only for a non-strict workflow progression with material handoff value.",
    "Return accepted=false for topical_only, duplicate_or_alternative, wrong_direction, generic_or_underspecified, local_only, state_mismatch, or unsupported_by_evidence.",
]


def build_compact_execution_validation_messages(
    candidate: ExecutionFlowCandidate,
    source_skill: SkillNode,
    target_skill: SkillNode,
    *,
    interfaces: dict[str, SkillInterface],
) -> list[dict[str, str]]:
    """Build a compact interface-first execution validation prompt."""

    payload: dict[str, Any] = {
        "role": "You are a precise SkillFabric execution handoff judge.",
        "goal": (
            "Decide whether source_skill's produced post-state can satisfy target_skill's required pre-state. "
            "Use compact interface evidence first; reject or report uncertainty instead of guessing."
        ),
        "prompt_id": COMPACT_EXECUTION_PROMPT_ID,
        "input": {
            "candidate": "A directed source_skill -> target_skill candidate from canonical produces/requires matching.",
            "source_skill": "Possible post-state producer/enabler represented by metadata and skill_interface.",
            "target_skill": "Possible pre-state consumer/dependent represented by metadata and skill_interface.",
            "matched_name": "The proposed reusable artifact/state handoff.",
        },
        "output": _OUTPUT_CONTRACT,
        "edge_semantics": _EDGE_SEMANTICS,
        "decision_procedure": _DECISION_PROCEDURE,
        "rejection_taxonomy": _COMPACT_REJECTION_TAXONOMY,
        "rules": [
            "Return JSON only.",
            "Use artifact_handoff when the post-state/pre-state is a concrete artifact, data object, text, or report.",
            "Use state_handoff when the post-state/pre-state is a reusable workflow state, credential, environment, or condition.",
            "For projected_edge_type=depend_on, target_skill depends on source_skill.",
            "Reject broad matches based only on generic data, text, output, result, object, observation, file, or command.",
            "Prefer accepted=false when compact evidence does not prove a reusable state or artifact handoff.",
        ],
        "constraints": [
            "Do not reverse source_skill and target_skill.",
            "Do not invent capabilities outside skill_interface or candidate evidence.",
            "Do not treat cognitive context or planning evidence as physical world_state.",
            "Do not add schema fields beyond output_schema.",
        ],
        "output_schema": _OUTPUT_SCHEMA,
        "candidate": candidate.to_dict(),
        "source_skill": _compact_skill_payload(source_skill, interfaces),
        "target_skill": _compact_skill_payload(target_skill, interfaces),
    }
    return [
        {"role": "system", "content": "You validate SkillFabric post-state to pre-state handoffs from compact evidence."},
        {"role": "user", "content": _prompt_json(payload)},
    ]


def build_execution_validation_messages(
    candidate: ExecutionFlowCandidate,
    source_skill: SkillNode,
    target_skill: SkillNode,
    *,
    interfaces: dict[str, SkillInterface],
) -> list[dict[str, str]]:
    """Build LiteLLM messages for validating an execution flow candidate."""

    payload: dict[str, Any] = {
        "role": "You are a precise SkillFabric execution handoff judge.",
        "goal": (
            "Decide whether source_skill's produced post-state can satisfy target_skill's required pre-state. "
            "Accept only evidence-backed handoffs that improve graph routing, wiki exploration, or final execution prompts."
        ),
        "prompt_id": EXECUTION_PROMPT_ID,
        "input": {
            "candidate": "A directed source_skill -> target_skill candidate from canonical produces/requires matching.",
            "source_skill": "Possible post-state producer/enabler, including full_skill_md and optional skill_interface.",
            "target_skill": "Possible pre-state consumer/dependent, including full_skill_md and optional skill_interface.",
            "matched_name": "The proposed artifact, data object, state, environment condition, credential, or validation result.",
        },
        "output": _OUTPUT_CONTRACT,
        "workflow": [
            "Step 1: Name the exact post-state claimed by source_skill and pre-state claimed by target_skill.",
            "Step 2: Verify source_skill actually produces, establishes, validates, or makes available that post-state.",
            "Step 3: Verify target_skill actually consumes, requires, or is materially improved by the same pre-state.",
            "Step 4: Distinguish artifacts/data/text/reports from world_state, credentials, and environment conditions.",
            "Step 5: Project strict consumable handoff cases to depend_on.",
            "Step 6: Project non-strict workflow progression cases to compose_with.",
            "Step 7: Return accepted=false when the pair matches a rejection taxonomy item or would not help routing, wiki search, or prompt assembly.",
        ],
        "decision_workflow": _DECISION_PROCEDURE,
        "direction_semantics": _EDGE_SEMANTICS,
        "rejection_taxonomy": _REJECTION_TAXONOMY,
        "output_schema": _OUTPUT_SCHEMA,
        "rules": [
            "Return JSON only.",
            "Use artifact_handoff only when target_skill consumes an artifact produced by source_skill.",
            "Use state_handoff only when source_skill establishes a state required by target_skill.",
            "The artifact or scenario must be named specifically enough to support reuse; generic data, text, output, result, observation, or command is not enough.",
            "For projected_edge_type=depend_on, the target skill depends on the source skill; do not describe the reverse direction.",
            "Reject flows where source_skill only provides cognitive context or planning evidence while target_skill requires a world_state.",
            "Do not treat object_permanence_state, parsed goals, plans, remembered facts, or observations as physical object_in_inventory state.",
            "Accept object_in_inventory only when source_skill actually performs or confirms take/pickup/acquire, not when it merely reasons about object permanence.",
            "Do not accept local-only outputs that are not consumed or required by the target skill.",
            "Do not accept generic matches such as object, data, text, output, result, observation, or command unless the evidence names a specific reusable workflow object.",
            "Do not accept a flow from shared tools or broad topical similarity.",
            "Do not accept a flow merely because the two skills could be useful in the same overall task.",
            "Cite evidence lines from full_skill_md, skill_interface, or candidate.evidence.",
            "Do not invent evidence not present in the input.",
            "Prefer accepted=false when the flow would not help graph routing, wiki search, or prompt assembly.",
        ],
        "constraints": [
            "Do not reverse source_skill and target_skill. The candidate is already directioned.",
            "Do not treat cognitive context or planning evidence as a physical world_state producer.",
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
        {"role": "system", "content": "You validate SkillFabric post-state to pre-state handoffs using strict textual evidence."},
        {"role": "user", "content": _prompt_json(payload)},
    ]


def _prompt_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


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
