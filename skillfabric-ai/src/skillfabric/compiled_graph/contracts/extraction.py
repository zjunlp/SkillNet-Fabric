"""Fail-closed SkillContract extraction with validated result caching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from skillfabric.compiled_graph.contracts.models import ContractSchemaError, SkillContract
from skillfabric.compiled_graph.contracts.prompts import (
    CONTRACT_PROMPT_FINGERPRINT,
    CONTRACT_PROMPT_ID,
    build_contract_extraction_messages,
)
from skillfabric.registry.models import SkillNode
from skillfabric.runtime.jobs import LLMJobOptions, LLMJobOutcome, run_llm_jobs
from skillfabric.runtime.json_utils import parse_json_response
from skillfabric.runtime.llm import LLMConfig, litellm_completion
from skillfabric.storage.checkpoint_cache import (
    CheckpointCacheError,
    JsonObjectCheckpointCache,
)


class ContractExtractionError(RuntimeError):
    """Raised when any enabled contract extraction cannot produce a valid result."""


class ContractExtractor(Protocol):
    """Provider protocol for raw contract extraction payloads."""

    model_id: str

    def extract(self, skill: SkillNode) -> dict[str, Any]:
        """Extract one raw contract payload."""


@dataclass(slots=True)
class ContractExtractionRecord:
    """One validated contract and its cache status."""

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

    options = (job_options or LLMJobOptions()).normalized()
    checkpoint_cache = JsonObjectCheckpointCache(
        cache_path,
        interval=options.checkpoint_interval,
    )
    try:
        cache = checkpoint_cache.load()
    except CheckpointCacheError as exc:
        raise ContractExtractionError(f"invalid contract cache: {exc}") from exc
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

    def accept(outcome: LLMJobOutcome[SkillContract]) -> None:
        index, _skill, key = outcome.item
        contract = outcome.value
        if not isinstance(contract, SkillContract):
            raise ContractExtractionError("contract extraction returned an invalid internal result")
        records[index] = ContractExtractionRecord(contract=contract)
        checkpoint_cache.record(key, contract.to_dict())

    try:
        outcomes = run_llm_jobs(
            pending,
            extract_one,
            options=options,
            label="contract",
            on_success=accept,
        )
    except Exception as exc:
        _flush_checkpoint(checkpoint_cache)
        if isinstance(exc, ContractExtractionError):
            raise
        raise ContractExtractionError(f"contract extraction aborted: {exc}") from exc
    except BaseException:
        _flush_checkpoint(checkpoint_cache)
        raise
    _flush_checkpoint(checkpoint_cache)
    failures = [outcome for outcome in outcomes if not outcome.ok]
    if failures:
        first = failures[0]
        _, skill, _ = first.item
        error = first.error or RuntimeError("unknown contract extraction failure")
        raise ContractExtractionError(
            f"contract extraction failed for {skill.id}: {error}"
        ) from error
    try:
        checkpoint_cache.compact()
    except CheckpointCacheError as exc:
        raise ContractExtractionError(f"failed to compact contract cache: {exc}") from exc
    return [record for record in records if record is not None]


def _cache_key(skill: SkillNode, model_id: str) -> str:
    payload = {
        "prompt_name": CONTRACT_PROMPT_ID,
        "prompt_fingerprint": CONTRACT_PROMPT_FINGERPRINT,
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


def _flush_checkpoint(cache: JsonObjectCheckpointCache) -> None:
    try:
        cache.flush()
    except CheckpointCacheError as exc:
        raise ContractExtractionError(f"failed to checkpoint contract cache: {exc}") from exc
