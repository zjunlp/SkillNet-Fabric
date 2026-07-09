"""Skill interface extraction orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.interface.cache import (
    cached_interface_from_payload,
    interface_cache_key,
    load_interface_cache,
    write_interface_cache,
)
from skillfabric.compiled_graph.interface.fallback import _fallback_interface
from skillfabric.compiled_graph.interface.models import InterfaceExtractionRecord
from skillfabric.compiled_graph.interface.normalization import (
    _error_payload,
    _interface_from_raw,
    _normalize_recoverable_interface_payload,
    _rejection_reason,
    _validate_interface_payload,
)
from skillfabric.compiled_graph.interface.providers import (
    DeterministicInterfaceExtractor,
    InterfaceSchemaError,
    LiteLLMInterfaceExtractor,
    SkillInterfaceExtractor,
)
from skillfabric.registry.models import SkillNode
from skillfabric.runtime.jobs import LLMJobOptions, run_llm_jobs


def extract_skill_interfaces(
    skills: list[SkillNode],
    *,
    extractor: SkillInterfaceExtractor | None = None,
    cache_path: str | Path | None = None,
    job_options: LLMJobOptions | None = None,
) -> list[InterfaceExtractionRecord]:
    """Extract interfaces for skills with cache and fallback behavior."""

    extractor = extractor or DeterministicInterfaceExtractor()
    cache = load_interface_cache(cache_path)
    records: list[InterfaceExtractionRecord | None] = [None] * len(skills)
    pending: list[tuple[int, SkillNode]] = []
    for index, skill in enumerate(skills):
        key = interface_cache_key(skill, extractor.model_id)
        cached = cache.get(key)
        if isinstance(cached, dict):
            interface = cached_interface_from_payload(cached)
            records[index] = (
                InterfaceExtractionRecord(
                    skill_id=skill.id,
                    raw_output={"cache_hit": True},
                    interface=interface,
                    accepted=True,
                )
            )
            continue
        pending.append((index, skill))

    def extract_one(item: tuple[int, SkillNode]) -> dict[str, Any]:
        _, skill = item
        raw = extractor.extract(skill)
        if not isinstance(raw, dict):
            raise InterfaceSchemaError("extractor output must be a JSON object")
        raw = _normalize_recoverable_interface_payload(raw)
        _validate_interface_payload(raw)
        return raw

    def on_success(outcome) -> None:
        index, skill = outcome.item
        raw = outcome.value
        if not isinstance(raw, dict):
            return
        interface = _interface_from_raw(skill, raw, model_id=extractor.model_id)
        records[index] = InterfaceExtractionRecord(
            skill_id=skill.id,
            raw_output=raw,
            interface=interface,
            accepted=True,
        )
        cache[interface_cache_key(skill, extractor.model_id)] = interface.to_dict()
        write_interface_cache(cache_path, cache)

    outcomes = run_llm_jobs(
        pending,
        extract_one,
        options=job_options,
        label="interface",
        on_success=on_success,
    )
    for outcome in outcomes:
        if outcome.ok:
            continue
        index, skill = outcome.item
        exc = outcome.error
        if isinstance(exc, json.JSONDecodeError):
            raw = _error_payload("json_parse_error", f"failed to parse interface JSON: {exc}")
        elif isinstance(exc, InterfaceSchemaError):
            raw = _error_payload("schema_error", str(exc))
        else:
            raw = _error_payload("api_error", f"{type(exc).__name__}: {exc}")
        interface = _fallback_interface(skill)
        records[index] = InterfaceExtractionRecord(skill.id, raw, interface, False, _rejection_reason(raw))

    write_interface_cache(cache_path, cache)
    return [record for record in records if record is not None]


__all__ = [
    "DeterministicInterfaceExtractor",
    "InterfaceSchemaError",
    "LiteLLMInterfaceExtractor",
    "SkillInterfaceExtractor",
    "extract_skill_interfaces",
]
