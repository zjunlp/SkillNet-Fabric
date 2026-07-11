"""Providers for skill interface extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from skillfabric.compiled_graph.interface.fallback import _fallback_interface
from skillfabric.compiled_graph.interface.prompts import build_interface_extraction_messages
from skillfabric.registry.models import SkillNode
from skillfabric.runtime.json_utils import parse_json_response
from skillfabric.runtime.llm import LLMConfig, litellm_completion

INTERFACE_MAX_TOKENS = 6144


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
            max_tokens=INTERFACE_MAX_TOKENS,
            reasoning_effort="low",
            usage_operation="kg_build.interface_extraction",
            usage_metadata={"skill_id": skill.id},
        )
        try:
            return parse_json_response(response)
        except json.JSONDecodeError:
            raise
        except ValueError as exc:
            raise InterfaceSchemaError(str(exc)) from exc
