"""Cache helpers for interface extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.interface.models import SkillInterface
from skillfabric.compiled_graph.interface.prompts import interface_prompt_payload
from skillfabric.registry.models import SkillNode


def interface_cache_key(skill: SkillNode, model_id: str) -> str:
    """Build the stable cache key for one skill interface extraction."""

    raw = json.dumps(
        {
            "skill_id": skill.id,
            "content_hash": skill.content_hash,
            "model_id": model_id,
            "input_digest": _input_digest(skill),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_interface_cache(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load cached interface payloads."""

    if path is None:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def write_interface_cache(path: str | Path | None, payload: dict[str, dict[str, Any]]) -> None:
    """Write cached interface payloads."""

    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cached_interface_from_payload(payload: dict[str, Any]) -> SkillInterface:
    """Load a cached interface payload."""

    return SkillInterface.from_dict(payload)


def _input_digest(skill: SkillNode) -> str:
    raw = json.dumps(interface_prompt_payload(skill), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
