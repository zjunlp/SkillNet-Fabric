"""Providers for skill interface extraction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from skillfabric.compiled_graph.interface.fallback import _fallback_interface
from skillfabric.compiled_graph.interface.prompts import build_interface_extraction_messages
from skillfabric.registry.models import SkillNode
from skillfabric.runtime.llm import LLMConfig, litellm_completion, response_to_jsonable


class InterfaceSchemaError(ValueError):
    """Raised when an interface extraction payload is not a SkillInterface payload."""


class SkillInterfaceExtractor(Protocol):
    """Protocol for skill interface extractors."""

    model_id: str

    def extract(self, skill: SkillNode) -> dict[str, Any]:
        """Return raw interface extraction output."""


@dataclass(slots=True)
class DeterministicInterfaceExtractor:
    """Deterministic fallback extractor for offline builds."""

    model_id: str = "deterministic-interface"

    def extract(self, skill: SkillNode) -> dict[str, Any]:
        return _fallback_interface(skill, model_id=self.model_id).to_dict()


@dataclass(slots=True)
class LiteLLMInterfaceExtractor:
    """LiteLLM-backed skill interface extractor."""

    config: LLMConfig

    @property
    def model_id(self) -> str:
        return self.config.model

    @classmethod
    def from_env(cls, *, env_path: str | Path | None = None) -> LiteLLMInterfaceExtractor:
        return cls(config=LLMConfig.from_env(env_path=env_path))

    def extract(self, skill: SkillNode) -> dict[str, Any]:
        messages = build_interface_extraction_messages(skill)
        response = litellm_completion(
            messages=messages,
            config=self.config,
            usage_operation="kg_build.interface_extraction",
            usage_metadata={"skill_id": skill.id},
        )
        response_text = _extract_response_text(response)
        return _parse_json_response(response_text)


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
        output = payload.get("output")
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("text") is not None:
                            parts.append(str(part["text"]))
            if parts:
                return "\n".join(parts)
    return str(payload)


def _parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise InterfaceSchemaError("interface JSON root must be an object")
    return payload


def _provenance(extractor: SkillInterfaceExtractor) -> str:
    if isinstance(extractor, DeterministicInterfaceExtractor):
        return "deterministic_fallback"
    return "llm_extracted"
