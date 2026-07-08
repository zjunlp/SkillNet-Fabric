"""Minimal interface fallback for offline or failed LLM extraction paths."""

from __future__ import annotations

from skillfabric.compiled_graph.interface.models import SkillInterface
from skillfabric.registry.models import SkillNode


def _fallback_interface(skill: SkillNode, *, model_id: str = "deterministic-interface") -> SkillInterface:
    """Return a non-inferential interface placeholder for a skill."""

    return SkillInterface(
        skill_id=skill.id,
        content_hash=skill.content_hash,
        capability_summary=skill.description,
        when_to_use=skill.description,
        requires=[],
        produces=[],
        uses_tools=[],
        evidence=[],
        provenance="deterministic_fallback",
        model_id=model_id,
    )
