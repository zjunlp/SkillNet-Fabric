"""Community metadata refinement providers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from skillfabric.runtime.llm import LLMConfig, litellm_completion, response_to_jsonable

COMMUNITY_REFINEMENT_PROMPT_ID = "community_refinement_graph_routing_boundaries"


class CommunityRefinementProvider(Protocol):
    """Provider protocol for community metadata refinement."""

    model_id: str

    def refine(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return refined community metadata."""


class DeterministicCommunityRefinementProvider:
    """No-op provider used for offline builds."""

    model_id = "deterministic-community"

    def refine(self, payload: dict[str, Any]) -> dict[str, Any]:
        community = dict(payload.get("community", {}))
        return {
            "name": str(community.get("name", "")),
            "summary": str(community.get("summary", "")),
            "task_patterns": [],
            "representative_skill_ids": list(community.get("representative_skill_ids", [])),
        }


class LiteLLMCommunityRefinementProvider:
    """LiteLLM-backed provider for community metadata refinement."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @property
    def model_id(self) -> str:
        return self.config.model

    @classmethod
    def from_env(cls, *, env_path: str | Path | None = None) -> LiteLLMCommunityRefinementProvider:
        return cls(LLMConfig.from_env(env_path=env_path))

    def refine(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Refine SkillFabric community metadata. Return strict JSON with keys "
                    "name, summary, task_patterns, representative_skill_ids. Do not change community membership. "
                    "Write metadata for route-time skill recommendation over graph-derived routing communities."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "prompt_id": COMMUNITY_REFINEMENT_PROMPT_ID,
                        "todo": (
                            "Refine one graph-derived skill community into concise route-time metadata without changing membership."
                        ),
                        "task": "Rename and summarize this routing community from graph-topology evidence.",
                        "input": {
                            "payload": (
                                "A fixed community with member skills, interface summaries, graph-topology cohesion, "
                                "projection_edge_evidence, top internal relation edges, and selected boundary edges. "
                                "Membership is fixed."
                            ),
                            "goal": (
                                "Produce metadata that helps a router understand when to inspect this community, "
                                "what routing boundary it covers, and where adjacent workflow stages remain separate."
                            ),
                        },
                        "output": {
                            "format": "Return one strict JSON object, with no markdown, comments, or extra keys.",
                            "required_top_level_keys": [
                                "name",
                                "summary",
                                "task_patterns",
                                "representative_skill_ids",
                            ],
                            "purpose": "Community metadata becomes query_wiki context for route-time skill recommendation.",
                        },
                        "workflow": [
                            "Step 1: Inspect member capabilities, interface fields, projection_edge_evidence, top internal edges, selected boundary edges, and cohesion_score.",
                            "Step 2: Identify the shared routing boundary: domain, artifact family, operation, constraint, validation role, or support role.",
                            "Step 3: Use graph-topology evidence to distinguish overlapping alternatives from adjacent workflow stages.",
                            "Step 4: Name the community by capability boundary, not by frequent words or taxonomy labels.",
                            "Step 5: Write task_patterns as short trigger phrases that help a router decide when to inspect this community.",
                            "Step 6: Choose representatives from member ids that best cover the community's central capability.",
                            "Step 7: Keep adjacent workflow stages described as boundary context unless validated relation edges show they belong together.",
                        ],
                        "rules": [
                            "Return JSON only.",
                            "Do not change community membership.",
                            "Do not split or merge communities.",
                            "Use a short capability-cluster name that is useful in router and wiki context.",
                            "Summarize the common task pattern shared by the member skills in downstream-agent terms.",
                            "Mention the strongest shared capability facet: domain, input artifact, output artifact, operation, constraint, validation role, or support role.",
                            "representative_skill_ids must be selected from members only.",
                            "task_patterns must be short strings useful for Router context and should mention trigger conditions, deliverable families, or support roles.",
                            "If the community contains overlapping alternatives, describe the shared boundary without implying all members should be selected together.",
                            "If selected_boundary_edges connect adjacent workflow stages, describe the boundary without collapsing those stages into this community.",
                            "Use validated relation edges as evidence only; do not invent an unvalidated dependency or compose relationship.",
                        ],
                        "constraints": [
                            "Do not change community membership.",
                            "Do not invent task patterns absent from member capability evidence.",
                            "Do not imply all member skills should be selected together when they are alternatives.",
                            "Do not turn boundary edges into membership changes.",
                            "Do not choose representative ids outside members.",
                            "Do not output schema keys beyond the required top-level keys.",
                        ],
                        "payload": payload,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        response = litellm_completion(
            messages=messages,
            config=self.config,
            usage_operation="kg_build.community_refinement",
            usage_metadata={"community_id": payload.get("community", {}).get("id")},
        )
        text = _extract_response_text(response)
        parsed = json.loads(_strip_fence(text))
        if not isinstance(parsed, dict):
            raise ValueError("community refinement response must be a JSON object")
        return parsed


def _extract_response_text(response: Any) -> str:
    payload = response_to_jsonable(response)
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and message.get("content") is not None:
                    return str(message["content"])
                if first.get("text") is not None:
                    return str(first["text"])
        if payload.get("output_text") is not None:
            return str(payload["output_text"])
    return str(payload)


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()
