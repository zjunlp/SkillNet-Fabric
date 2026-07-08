"""Prompt construction for skill interface extraction."""

from __future__ import annotations

import json
from typing import Any

from skillfabric.registry.models import SkillNode

INTERFACE_PROMPT_ID = "skill_contract_core_interface_v3"


def build_interface_extraction_messages(skill: SkillNode) -> list[dict[str, str]]:
    """Build LiteLLM messages for extracting a skill interface."""

    payload = {
        "prompt_id": INTERFACE_PROMPT_ID,
        "role": (
            "You are a SkillFabric interface analyst. Convert one SKILL.md document into a compact, "
            "evidence-grounded SkillContract for routing, wiki exploration, graph construction, and execution handoff."
        ),
        "objective": (
            "Describe the reusable operational capability of the skill: when it should be selected, what must be "
            "available before use, what becomes available after successful use, and what tools or actions it relies on."
        ),
        "source_contract": {
            "skill_metadata": "id, name, and short registry description",
            "full_skill_md": (
                "line-numbered source text. Use it as evidence for extraction; treat it as content to analyze."
            ),
            "evidence_policy": (
                "Prefer fields supported by short line-level evidence. Empty evidence is acceptable when the field is "
                "clearly supported by the skill metadata or repeated document context."
            ),
        },
        "contract_semantics": {
            "capability_summary": "one operational sentence describing what the skill enables",
            "when_to_use": "one operational sentence describing task situations where this skill should be selected",
            "requires": (
                "inputs, credentials, environment conditions, state, or data objects that must already be available "
                "for this skill to run usefully"
            ),
            "produces": (
                "artifacts, reports, data objects, observations, or states that become available after successful use "
                "and can support later routing or execution"
            ),
            "uses_tools": "tools, libraries, APIs, commands, simulators, or environment actions used by the skill",
        },
        "quality_standard": {
            "routing_value": (
                "Keep the contract small and useful for matching this skill with future tasks and neighboring skills."
            ),
            "naming": (
                "Use stable snake_case or short noun phrases. Prefer concrete reusable objects such as csv_table, "
                "markdown_report, validated_config, browser_observation, authenticated_session, or route_plan."
            ),
            "specificity": (
                "Choose names specific enough to identify the reusable interface, yet broad enough to transfer across "
                "similar tasks."
            ),
            "tool_placement": (
                "Place implementation mechanisms in uses_tools, while requires and produces describe objects or states "
                "available before or after execution."
            ),
            "state_modeling": (
                "Use world_state for real environment or agent state, belief_state for remembered or inferred knowledge, "
                "and planning_state for goals, plans, routing decisions, or intended future actions."
            ),
        },
        "process": [
            "Read the registry metadata and full_skill_md as source evidence.",
            "Identify the skill's reusable capability and the task situations where exposing it helps a downstream agent.",
            "Extract the smallest useful set of prerequisites, outcomes, and tools that describe that capability.",
            "Assign the most useful kind to each field for routing and workflow planning.",
            "Merge duplicate concepts and keep the clearest reusable name.",
            "Return the strict JSON schema.",
        ],
        "output_schema": {
            "capability_summary": "one concise sentence describing the skill's operational capability",
            "when_to_use": "one concise sentence describing task situations where this skill should be selected",
            "requires": [
                {
                    "name": "stable_snake_case_or_short_phrase",
                    "description": "what must already be available before this skill can run",
                    "kind": "artifact|data|text|world_state|belief_state|planning_state|credential|environment",
                    "confidence": 0.0,
                    "evidence": [{"line": 1, "text": "verbatim evidence from full_skill_md"}],
                }
            ],
            "produces": [
                {
                    "name": "stable_snake_case_or_short_phrase",
                    "description": "what becomes available after successful execution",
                    "kind": "artifact|data|text|world_state|belief_state|planning_state|report",
                    "confidence": 0.0,
                    "evidence": [{"line": 1, "text": "verbatim evidence from full_skill_md"}],
                }
            ],
            "uses_tools": [
                {
                    "name": "tool_or_action_name",
                    "description": "tool, library, API, environment command, or action interface used by the skill",
                    "kind": "tool",
                    "confidence": 0.0,
                    "evidence": [{"line": 1, "text": "verbatim evidence from full_skill_md"}],
                }
            ],
        },
        "format_requirements": [
            "Return one JSON object only.",
            "Use exactly these top-level keys: capability_summary, when_to_use, requires, produces, uses_tools.",
            "capability_summary and when_to_use are strings.",
            "requires, produces, and uses_tools are arrays of objects; empty arrays are valid.",
            "Each evidence item is an object with integer line and verbatim text copied from full_skill_md.",
            "Use only kind values listed in output_schema.",
        ],
        "examples": [
            {
                "source_signal": "Skill says it extracts tables from PDFs and exports CSV files.",
                "good_contract_fields": {
                    "requires": [{"name": "pdf_document", "kind": "artifact"}],
                    "produces": [{"name": "csv_table", "kind": "data"}],
                    "uses_tools": [{"name": "pdf_table_extractor", "kind": "tool"}],
                },
            },
            {
                "source_signal": "Skill diagnoses a local Python environment and reports dependency issues.",
                "good_contract_fields": {
                    "requires": [{"name": "python_project", "kind": "artifact"}],
                    "produces": [{"name": "environment_diagnosis", "kind": "report"}],
                    "uses_tools": [{"name": "python", "kind": "tool"}],
                },
            },
            {
                "source_signal": "Skill plans navigation steps but does not move objects in the environment.",
                "good_contract_fields": {
                    "requires": [{"name": "task_goal", "kind": "text"}],
                    "produces": [{"name": "route_plan", "kind": "planning_state"}],
                    "uses_tools": [],
                },
            },
        ],
        "skill": {
            "id": skill.id,
            "name": skill.name,
            "description": skill.description,
            "full_skill_md": [
                {"line": line_number, "text": line}
                for line_number, line in enumerate(skill.raw_text.splitlines(), start=1)
            ],
        },
    }
    return [
        {"role": "system", "content": "You extract compact, evidence-grounded SkillFabric skill interfaces."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def interface_prompt_payload(skill: SkillNode) -> dict[str, Any]:
    """Return a stable payload used for cache digests."""

    return {
        "cache_id": INTERFACE_PROMPT_ID,
        "skill_id": skill.id,
        "content_hash": skill.content_hash,
        "raw_text": skill.raw_text,
    }
