"""LLM task atomization for route-time SkillFabric retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skillfabric.llm import LLMConfig, litellm_completion, response_to_jsonable
from skillfabric.router.task_atoms import (
    TASK_ATOMS_SCHEMA_VERSION,
    TaskDecomposition,
    task_atoms_json_schema,
    validate_task_decomposition,
)

TASK_ATOMIZER_PROMPT_ID = "skillfabric_task_atomizer_v1"


def atomize_task(query: str, *, env_file: str | Path | None = ".env") -> TaskDecomposition:
    """Call the configured LLM once to extract route-time task atoms."""

    config = LLMConfig.from_env(env_path=env_file)
    response = litellm_completion(
        messages=build_task_atomizer_messages(query),
        config=config,
        usage_operation="route.task_atomizer",
        usage_metadata={"prompt_id": TASK_ATOMIZER_PROMPT_ID},
    )
    text = _extract_response_text(response)
    payload = json.loads(_strip_fence(text))
    if not isinstance(payload, dict):
        raise ValueError("task atomizer response must be a JSON object")
    return validate_task_decomposition(payload, query=query)


def build_task_atomizer_messages(query: str) -> list[dict[str, str]]:
    """Build the strict JSON task atomizer prompt."""

    payload = {
        "prompt_id": TASK_ATOMIZER_PROMPT_ID,
        "todo": "Decompose one user task into a compact set of route-time requirement atoms.",
        "task": (
            "Extract atomic requirements that help a graph retriever cover distinct user-requested stages, artifacts, "
            "and constraints. Do not choose skills or map the task to fixed ontology labels."
        ),
        "output_schema": task_atoms_json_schema(),
        "allowed_atom_kinds": {
            "action": "An operation or subtask the downstream agent must perform.",
            "artifact": "A requested input, output, file, data object, screenshot, report, image, page, or other concrete deliverable.",
            "constraint": "A user-specified quality, format, ordering, verification, or scope constraint.",
        },
        "rules": [
            "Return one strict JSON object only. Do not include markdown, comments, or explanation.",
            f"Use schema_version {TASK_ATOMS_SCHEMA_VERSION}.",
            "Create at most 12 atoms and merge duplicates.",
            "Every atom.evidence must be a short exact substring copied from the user query.",
            "Do not output skill_id, skill_ids, intent, domain_hints, deliverable labels, or any graph vocabulary.",
            "Do not recommend, name, or rank skills.",
            "Do not map vague words to file formats. For example, do not turn 'slides' into '.pptx' unless the query explicitly contains .pptx or PowerPoint.",
            "If the query explicitly names files, URLs, extensions, JSON keys, or screenshots, keep those details in artifact atoms.",
            "Use depends_on only when the query clearly states an order or one atom must precede another.",
            "Use required=false only for optional or nice-to-have wording from the user.",
        ],
        "examples": [
            {
                "query": "Research Trello and Asana, capture homepage screenshots, and write competitor_report.docx.",
                "output": {
                    "schema_version": TASK_ATOMS_SCHEMA_VERSION,
                    "atoms": [
                        {
                            "id": "a1",
                            "kind": "action",
                            "text": "research Trello and Asana",
                            "evidence": "Research Trello and Asana",
                            "required": True,
                            "depends_on": [],
                        },
                        {
                            "id": "a2",
                            "kind": "artifact",
                            "text": "capture homepage screenshots",
                            "evidence": "capture homepage screenshots",
                            "required": True,
                            "depends_on": ["a1"],
                        },
                        {
                            "id": "a3",
                            "kind": "artifact",
                            "text": "write competitor_report.docx",
                            "evidence": "write competitor_report.docx",
                            "required": True,
                            "depends_on": ["a1", "a2"],
                        },
                    ],
                },
            }
        ],
        "user_query": query,
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a precise task decomposition component for SkillFabric routing. "
                "You extract user-stated requirements as JSON atoms. You never select skills."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def parse_task_atomizer_response(response: Any, *, query: str = "") -> TaskDecomposition:
    """Parse and validate a raw LiteLLM-style task atomizer response."""

    text = _extract_response_text(response)
    payload = json.loads(_strip_fence(text))
    if not isinstance(payload, dict):
        raise ValueError("task atomizer response must be a JSON object")
    return validate_task_decomposition(payload, query=query)


def _extract_response_text(response: Any) -> str:
    payload = response_to_jsonable(response)
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and message.get("content") is not None:
                    return str(message.get("content", ""))
                if first.get("text") is not None:
                    return str(first.get("text", ""))
        output = payload.get("output")
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict):
                    for content in item.get("content", []) or []:
                        if isinstance(content, dict) and content.get("text") is not None:
                            return str(content["text"])
        if payload.get("content") is not None:
            return str(payload["content"])
    return str(payload)


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped

