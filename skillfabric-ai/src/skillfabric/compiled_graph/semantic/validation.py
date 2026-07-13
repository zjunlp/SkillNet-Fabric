"""Strict semantic relation requests with pair-level validated caching."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from skillfabric.compiled_graph.contracts.models import SkillContract
from skillfabric.compiled_graph.models import EvidenceRef
from skillfabric.compiled_graph.semantic.models import (
    CandidatePair,
    RelationDecision,
)
from skillfabric.compiled_graph.semantic.prompts import (
    RELATION_PROMPT_FINGERPRINT,
    RELATION_PROMPT_ID,
    build_relation_judge_messages,
)
from skillfabric.registry.models import SkillNode
from skillfabric.runtime.jobs import LLMJobOptions, run_llm_jobs
from skillfabric.runtime.json_utils import parse_json_response
from skillfabric.runtime.llm import LLMConfig, litellm_completion
from skillfabric.storage import atomic_write_text

_DECISION_KEYS = frozenset(
    {"relation", "source_skill", "target_skill", "confidence", "reason", "evidence"}
)
_RELATIONS = frozenset({"depend_on", "compose_with", "similar_to", "none"})
RELATION_PAIRS_PER_REQUEST = 4


class RelationValidationError(RuntimeError):
    """Raised when any candidate cannot receive a valid semantic decision."""


class RelationJudge(Protocol):
    """Provider protocol for one exact semantic relation request."""

    model_id: str

    def judge(
        self,
        pairs: tuple[CandidatePair, ...],
        skills: dict[str, SkillNode],
        contracts: dict[str, SkillContract],
    ) -> dict[str, Any]:
        """Return one raw decision for every requested pair."""


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    index: int
    pair: CandidatePair
    cache_key: str


@dataclass(slots=True)
class LiteLLMRelationJudge:
    """Production Skill Profile semantic judge."""

    config: LLMConfig

    @property
    def model_id(self) -> str:
        return self.config.model

    @classmethod
    def from_env(cls, *, env_path: str | Path | None = None) -> LiteLLMRelationJudge:
        return cls(LLMConfig.from_env(env_path=env_path))

    def judge(
        self,
        pairs: tuple[CandidatePair, ...],
        skills: dict[str, SkillNode],
        contracts: dict[str, SkillContract],
    ) -> dict[str, Any]:
        response = litellm_completion(
            messages=build_relation_judge_messages(pairs, skills, contracts),
            config=self.config,
            usage_operation="graph.semantic_relation",
            usage_metadata={"candidate_pairs": [list(pair.key) for pair in pairs]},
        )
        return parse_json_response(response)


def validate_candidate_pairs(
    pairs: list[CandidatePair] | tuple[CandidatePair, ...],
    skills: list[SkillNode],
    contracts: dict[str, SkillContract],
    *,
    judge: RelationJudge,
    cache_path: str | Path | None = None,
    job_options: LLMJobOptions | None = None,
) -> list[RelationDecision]:
    """Judge every candidate exactly once and fail the batch on invalid output."""

    skills_by_id = {skill.id: skill for skill in skills}
    if set(skills_by_id) != set(contracts):
        raise RelationValidationError(
            "skill and contract ids must match before relation validation"
        )
    ordered_pairs = list(pairs)
    if len({pair.key for pair in ordered_pairs}) != len(ordered_pairs):
        raise RelationValidationError("each unordered candidate pair may be judged only once")
    cache = _load_cache(cache_path)
    records: list[RelationDecision | None] = [None] * len(ordered_pairs)
    pending: list[_PendingDecision] = []
    for index, pair in enumerate(ordered_pairs):
        key = _cache_key(pair, skills_by_id, contracts, judge.model_id)
        cached = cache.get(key)
        if cached is None:
            pending.append(_PendingDecision(index=index, pair=pair, cache_key=key))
            continue
        try:
            records[index] = decision_from_payload(
                pair,
                cached,
                skills_by_id,
                cache_hit=True,
            )
        except (TypeError, ValueError) as exc:
            raise RelationValidationError(f"invalid relation cache for {pair.key}: {exc}") from exc

    requests = _pack_requests(pending, limit=RELATION_PAIRS_PER_REQUEST)

    def judge_one(request: tuple[_PendingDecision, ...]) -> list[tuple[_PendingDecision, RelationDecision]]:
        request_pairs = tuple(item.pair for item in request)
        raw = judge.judge(request_pairs, skills_by_id, contracts)
        if not isinstance(raw, dict):
            raise ValueError("relation judge output must be an object")
        decisions = _decisions_from_response(request_pairs, raw, skills_by_id)
        return list(zip(request, decisions, strict=True))

    outcomes = run_llm_jobs(requests, judge_one, options=job_options, label="relation")
    additions: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        if not outcome.ok:
            continue
        request_decisions = outcome.value
        if not isinstance(request_decisions, list):
            raise RelationValidationError("relation validation returned an invalid internal result")
        for pending_item, decision in request_decisions:
            if not isinstance(decision, RelationDecision):
                raise RelationValidationError(
                    "relation validation returned an invalid internal result"
                )
            records[pending_item.index] = decision
            additions[pending_item.cache_key] = decision.judge_dict()
    if additions:
        cache.update(additions)
        _write_cache(cache_path, cache)
    failures = [outcome for outcome in outcomes if not outcome.ok]
    if failures:
        first = failures[0]
        error = first.error or RuntimeError("unknown relation validation failure")
        pair_keys = [item.pair.key for item in first.item]
        raise RelationValidationError(
            f"relation validation failed for {pair_keys}: {error}"
        ) from error
    return [record for record in records if record is not None]


def _pack_requests(
    pending: list[_PendingDecision],
    *,
    limit: int,
) -> list[tuple[_PendingDecision, ...]]:
    if limit <= 0:
        raise ValueError("relation request limit must be positive")
    remaining = {item.pair.key: item for item in pending}
    requests: list[tuple[_PendingDecision, ...]] = []
    while remaining:
        endpoint_counts = Counter(
            skill_id
            for item in remaining.values()
            for skill_id in item.pair.key
        )
        highest_count = max(endpoint_counts.values())
        anchor = min(
            skill_id for skill_id, count in endpoint_counts.items() if count == highest_count
        )
        request = tuple(
            item
            for item in sorted(remaining.values(), key=lambda row: row.pair.key)
            if anchor in item.pair.key
        )[:limit]
        requests.append(request)
        for item in request:
            remaining.pop(item.pair.key)
    return requests


def _decisions_from_response(
    pairs: tuple[CandidatePair, ...],
    payload: dict[str, Any],
    skills: dict[str, SkillNode],
) -> list[RelationDecision]:
    if set(payload) != {"decisions"}:
        raise ValueError("relation judge response must contain exactly decisions")
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError("relation judge decisions must be a list")
    pairs_by_key = {pair.key: pair for pair in pairs}
    decisions_by_key: dict[tuple[str, str], RelationDecision] = {}
    for raw in raw_decisions:
        if not isinstance(raw, dict):
            raise ValueError("relation judge decisions must be objects")
        source = _required_string(raw.get("source_skill"), "source_skill")
        target = _required_string(raw.get("target_skill"), "target_skill")
        key = tuple(sorted((source, target)))
        if key in decisions_by_key:
            raise ValueError(f"relation judge returned duplicate candidate pair: {key}")
        pair = pairs_by_key.get(key)
        if pair is None:
            raise ValueError(f"relation judge returned unexpected candidate pair: {key}")
        decisions_by_key[key] = decision_from_payload(pair, raw, skills)
    missing = sorted(set(pairs_by_key) - set(decisions_by_key))
    if missing:
        raise ValueError(f"relation judge response is missing candidate pairs: {missing}")
    return [decisions_by_key[pair.key] for pair in pairs]


def decision_from_payload(
    pair: CandidatePair,
    payload: dict[str, Any],
    skills: dict[str, SkillNode],
    *,
    cache_hit: bool = False,
) -> RelationDecision:
    """Validate one raw decision against pair identity and verbatim source lines."""

    actual_keys = set(payload)
    if actual_keys != _DECISION_KEYS:
        missing = _DECISION_KEYS - actual_keys
        unexpected = actual_keys - _DECISION_KEYS
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected keys: {', '.join(sorted(unexpected))}")
        raise ValueError("relation decision " + "; ".join(details))
    relation = payload.get("relation")
    if not isinstance(relation, str) or relation not in _RELATIONS:
        raise ValueError("relation must be depend_on, compose_with, similar_to, or none")
    source = _required_string(payload.get("source_skill"), "source_skill")
    target = _required_string(payload.get("target_skill"), "target_skill")
    if source == target or {source, target} != set(pair.key):
        raise ValueError("source_skill and target_skill must be the candidate pair")
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    reason = _required_string(payload.get("reason"), "reason")
    evidence = _validated_evidence(payload.get("evidence"), pair=pair, skills=skills)
    if relation != "none" and {item.skill for item in evidence} != set(pair.key):
        raise ValueError("non-none relations require evidence from both candidate skills")
    if relation in {"compose_with", "similar_to", "none"}:
        source, target = pair.key
    return RelationDecision(
        candidate=pair,
        relation=relation,
        source_skill=source,
        target_skill=target,
        confidence=confidence,
        reason=reason,
        evidence=tuple(evidence),
        cache_hit=cache_hit,
    )


def _validated_evidence(
    value: Any,
    *,
    pair: CandidatePair,
    skills: dict[str, SkillNode],
) -> list[EvidenceRef]:
    if not isinstance(value, list):
        raise ValueError("evidence must be a list")
    evidence: list[EvidenceRef] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"skill", "line"}:
            raise ValueError(f"evidence[{index}] must contain exactly skill and line")
        skill_id = _required_string(item.get("skill"), f"evidence[{index}].skill")
        if skill_id not in pair.key:
            raise ValueError("evidence skill must belong to the candidate pair")
        line = item.get("line")
        if isinstance(line, bool) or not isinstance(line, int):
            raise ValueError(f"evidence[{index}].line must be an integer")
        source_lines = skills[skill_id].raw_text.splitlines()
        if line < 1 or line > len(source_lines):
            raise ValueError(f"evidence[{index}].line is outside the skill source")
        source_text = source_lines[line - 1]
        if not source_text.strip():
            raise ValueError(f"evidence[{index}].line must reference a non-empty source line")
        evidence.append(
            EvidenceRef(
                skill=skill_id,
                line=line,
                text=source_text,
            )
        )
    return evidence


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _cache_key(
    pair: CandidatePair,
    skills: dict[str, SkillNode],
    contracts: dict[str, SkillContract],
    model_id: str,
) -> str:
    payload = {
        "prompt_name": RELATION_PROMPT_ID,
        "prompt_fingerprint": RELATION_PROMPT_FINGERPRINT,
        "model_id": model_id,
        "pair": list(pair.key),
        "skills": {skill_id: skills[skill_id].content_hash for skill_id in pair.key},
        "contracts": {skill_id: contracts[skill_id].to_dict() for skill_id in pair.key},
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cache(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not Path(path).exists():
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RelationValidationError(f"failed to read relation cache: {exc}") from exc
    if not isinstance(payload, dict) or any(
        not isinstance(value, dict) for value in payload.values()
    ):
        raise RelationValidationError("relation cache must map keys to decision objects")
    return {str(key): value for key, value in payload.items()}


def _write_cache(path: str | Path | None, cache: dict[str, dict[str, Any]]) -> None:
    if path is None:
        return
    atomic_write_text(
        Path(path),
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
