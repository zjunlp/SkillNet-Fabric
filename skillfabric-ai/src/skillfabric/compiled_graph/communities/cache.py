"""Cache keys and persistence for community LLM calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.communities.providers import COMMUNITY_REFINEMENT_PROMPT_ID
from skillfabric.compiled_graph.models import CommunityNode


def _cache_key(community: CommunityNode, payload: dict[str, Any], model_id: str) -> str:
    raw = json.dumps(
        {
            "prompt_id": COMMUNITY_REFINEMENT_PROMPT_ID,
            "community_id": community.id,
            "payload": payload,
            "model_id": model_id,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def _write_cache(path: str | Path | None, cache: dict[str, dict[str, Any]]) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
