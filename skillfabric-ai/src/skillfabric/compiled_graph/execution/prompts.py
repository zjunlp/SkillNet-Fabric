"""Prompt construction for execution flow validation."""

from __future__ import annotations

import json
from typing import Any

from skillfabric.compiled_graph.execution.models import ExecutionFlowCandidate
from skillfabric.compiled_graph.interface.models import SkillInterface
from skillfabric.registry.models import SkillNode

EXECUTION_PROMPT_ID = "execution_validation_handoff_precision_v5"
COMPACT_EXECUTION_PROMPT_ID = "execution_validation_compact_handoff_v4"

_OUTPUT_SCHEMA: dict[str, Any] = {
    "accepted": True,
    "flow_type": "artifact_handoff|state_handoff|none",
    "projected_edge_type": "depend_on|compose_with|none",
    "confidence": "float between 0 and 1",
    "evidence": [{"skill": "skill id", "line": 1, "text": "verbatim evidence"}],
    "needs_full_context": False,
}

_DIRECTION = "source_skill produces or enables a post-state; target_skill consumes or requires a pre-state"

_DECISION_RULES = [
    (
        "Verify both sides of the same named handoff from supplied evidence. Do not reverse the candidate or invent "
        "capabilities. Missing or ambiguous evidence means accepted=false."
    ),
    (
        "Use artifact_handoff for concrete artifact/data/text/report transfer and state_handoff for a reusable "
        "world state, credential, environment, or condition established by execution."
    ),
    (
        "Use projected_edge_type=depend_on only for a strict consumable handoff. For "
        "projected_edge_type=depend_on, the target skill depends on the source skill."
    ),
    (
        "Use projected_edge_type=compose_with only for a non-strict workflow progression that materially improves "
        "the target without being a hard prerequisite."
    ),
    (
        "belief_state and planning_state are non-execution cognitive context and should not appear as execution "
        "handoff candidates or prove physical world_state or object_in_inventory."
    ),
    (
        "Reject topical_only, duplicate_or_alternative, wrong_direction, generic_or_underspecified, local_only, "
        "state_mismatch, and unsupported_by_evidence. Shared tools or usefulness in the same overall task are not "
        "handoffs; generic data, text, output, result, object, observation, file, or command matches are insufficient."
    ),
]


def build_compact_execution_validation_messages(
    candidate: ExecutionFlowCandidate,
    source_skill: SkillNode,
    target_skill: SkillNode,
    *,
    interfaces: dict[str, SkillInterface],
) -> list[dict[str, str]]:
    """Build a compact interface-first execution validation prompt."""

    payload = _validation_payload(
        prompt_id=COMPACT_EXECUTION_PROMPT_ID,
        prompt_tier="compact",
        task=(
            "Decide whether the directed post-state can satisfy the pre-state using interface evidence. Return "
            "needs_full_context=true only when full SKILL.md could resolve a specific uncertainty."
        ),
        candidate=candidate,
        source_skill=_compact_skill_payload(source_skill, interfaces),
        target_skill=_compact_skill_payload(target_skill, interfaces),
    )
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

    payload = _validation_payload(
        prompt_id=EXECUTION_PROMPT_ID,
        prompt_tier="full",
        task=(
            "Decide whether the directed post-state can satisfy the pre-state using candidate, interface, and full "
            "SKILL.md evidence. Keep only high-precision reusable handoffs."
        ),
        candidate=candidate,
        source_skill=_skill_payload(source_skill, interfaces),
        target_skill=_skill_payload(target_skill, interfaces),
    )
    return [
        {"role": "system", "content": "You validate SkillFabric post-state to pre-state handoffs using strict textual evidence."},
        {"role": "user", "content": _prompt_json(payload)},
    ]


def _validation_payload(
    *,
    prompt_id: str,
    prompt_tier: str,
    task: str,
    candidate: ExecutionFlowCandidate,
    source_skill: dict[str, Any],
    target_skill: dict[str, Any],
) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "prompt_tier": prompt_tier,
        "task": task,
        "direction": _DIRECTION,
        "decision_rules": _DECISION_RULES,
        "output_contract": (
            "Return exactly one JSON object matching output_schema, with no prose or Markdown and no comments or extra keys."
        ),
        "output_schema": _OUTPUT_SCHEMA,
        "candidate": candidate.to_dict(),
        "source_skill": source_skill,
        "target_skill": target_skill,
    }


def _prompt_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _skill_payload(skill: SkillNode, interfaces: dict[str, SkillInterface]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": skill.id,
        "name": skill.name,
        "description": skill.description,
        "full_skill_md": {
            "line_numbered_text": "\n".join(
                f"{line_number}: {line}"
                for line_number, line in enumerate(skill.raw_text.splitlines(), start=1)
            )
        },
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
