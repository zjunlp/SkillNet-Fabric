"""Fail-closed SkillContract extraction with validated result caching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from skillfabric.compiled_graph.contracts.models import ContractSchemaError, SkillContract
from skillfabric.compiled_graph.contracts.prompts import (
    CONTRACT_PROMPT_ID,
    build_contract_extraction_messages,
)
from skillfabric.registry.models import SkillNode
from skillfabric.runtime.jobs import LLMJobOptions, run_llm_jobs
from skillfabric.runtime.json_utils import parse_json_response
from skillfabric.runtime.llm import LLMConfig, litellm_completion
from skillfabric.storage import atomic_write_text


class ContractExtractionError(RuntimeError):
    """Raised when any enabled contract extraction cannot produce a valid result."""


class ContractExtractor(Protocol):
    """Provider protocol for raw contract extraction payloads."""

    model_id: str

    def extract(self, skill: SkillNode) -> dict[str, Any]:
        """Extract one raw contract payload."""


@dataclass(slots=True)
class ContractExtractionRecord:
    """One validated contract and its cache provenance."""

    contract: SkillContract
    cache_hit: bool = False


@dataclass(slots=True)
class LiteLLMContractExtractor:
    """Production contract extractor backed by the configured LiteLLM endpoint."""

    config: LLMConfig

    @property
    def model_id(self) -> str:
        return self.config.model

    @classmethod
    def from_env(cls, *, env_path: str | Path | None = None) -> LiteLLMContractExtractor:
        return cls(LLMConfig.from_env(env_path=env_path))

    def extract(self, skill: SkillNode) -> dict[str, Any]:
        response = litellm_completion(
            messages=build_contract_extraction_messages(skill),
            config=self.config,
            usage_operation="graph.contract_extraction",
            usage_metadata={"skill_id": skill.id},
        )
        return parse_json_response(response)


def extract_skill_contracts(
    skills: list[SkillNode],
    *,
    extractor: ContractExtractor,
    cache_path: str | Path | None = None,
    job_options: LLMJobOptions | None = None,
) -> list[ContractExtractionRecord]:
    """Extract all contracts, failing the batch if any result is invalid."""

    cache = _load_cache(cache_path)
    records: list[ContractExtractionRecord | None] = [None] * len(skills)
    pending: list[tuple[int, SkillNode, str]] = []
    for index, skill in enumerate(skills):
        key = _cache_key(skill, extractor.model_id)
        cached = cache.get(key)
        if cached is None:
            pending.append((index, skill, key))
            continue
        try:
            contract = SkillContract.from_dict(cached)
            _validate_cached_identity(contract, skill)
        except (ContractSchemaError, TypeError, ValueError) as exc:
            raise ContractExtractionError(f"invalid contract cache for {skill.id}: {exc}") from exc
        records[index] = ContractExtractionRecord(
            contract=contract,
            cache_hit=True,
        )

    def extract_one(item: tuple[int, SkillNode, str]) -> SkillContract:
        _, skill, _ = item
        raw = extractor.extract(skill)
        if not isinstance(raw, dict):
            raise ContractSchemaError("contract extractor output must be an object")
        return SkillContract.from_extraction(skill, raw)

    outcomes = run_llm_jobs(
        pending,
        extract_one,
        options=job_options,
        label="contract",
    )
    additions: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        if not outcome.ok:
            continue
        index, _skill, key = outcome.item
        contract = outcome.value
        if not isinstance(contract, SkillContract):
            raise ContractExtractionError("contract extraction returned an invalid internal result")
        records[index] = ContractExtractionRecord(contract=contract)
        additions[key] = contract.to_dict()

    if additions:
        cache.update(additions)
        _write_cache(cache_path, cache)
    failures = [outcome for outcome in outcomes if not outcome.ok]
    if failures:
        first = failures[0]
        _, skill, _ = first.item
        error = first.error or RuntimeError("unknown contract extraction failure")
        raise ContractExtractionError(
            f"contract extraction failed for {skill.id}: {error}"
        ) from error
    return [record for record in records if record is not None]


def _cache_key(skill: SkillNode, model_id: str) -> str:
    payload = {
        "prompt_id": CONTRACT_PROMPT_ID,
        "skill_id": skill.id,
        "content_hash": skill.content_hash,
        "model_id": model_id,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_cached_identity(
    contract: SkillContract,
    skill: SkillNode,
) -> None:
    if contract.skill_id != skill.id:
        raise ContractSchemaError("cached skill_id does not match the source skill")
    if contract.content_hash != skill.content_hash:
        raise ContractSchemaError("cached content_hash does not match the source skill")


def _load_cache(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractExtractionError(f"failed to read contract cache: {exc}") from exc
    if not isinstance(payload, dict) or any(
        not isinstance(value, dict) for value in payload.values()
    ):
        raise ContractExtractionError("contract cache must map keys to contract objects")
    return {str(key): value for key, value in payload.items()}


def _write_cache(path: str | Path | None, cache: dict[str, dict[str, Any]]) -> None:
    if path is None:
        return
    atomic_write_text(
        Path(path),
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
