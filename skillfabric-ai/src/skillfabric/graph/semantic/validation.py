"""Strict semantic relation requests with pair-level validated caching."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from skillfabric.graph.contracts.models import SkillContract
from skillfabric.graph.models import EvidenceRef
from skillfabric.graph.semantic.models import (
    CandidatePair,
    RelationDecision,
)
from skillfabric.graph.semantic.prompts import (
    RELATION_POLICY_FINGERPRINT,
    RELATION_PROMPT_ID,
    build_relation_judge_messages,
)
from skillfabric.registry.models import SkillNode
from skillfabric.runtime.jobs import LLMJobOptions, LLMJobOutcome, run_llm_jobs
from skillfabric.runtime.json_utils import parse_json_response
from skillfabric.runtime.llm import LLMConfig, litellm_completion
from skillfabric.storage.checkpoint_cache import (
    CheckpointCacheError,
    JsonObjectCheckpointCache,
)

_DECISION_KEYS = frozenset(
    {"relation", "source_skill", "target_skill", "confidence", "reason", "evidence"}
)
_JUDGE_DECISION_KEYS = frozenset(
    {"pair_index", "relation", "direction", "confidence", "reason", "evidence"}
)
_JUDGE_EVIDENCE_KEYS = frozenset({"skill_a_lines", "skill_b_lines"})
_RELATIONS = frozenset({"depend_on", "compose_with", "similar_to", "none"})
_DIRECTIONS = frozenset({"skill_a_to_skill_b", "skill_b_to_skill_a", "symmetric"})


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
    options = (job_options or LLMJobOptions()).normalized()
    checkpoint_cache = JsonObjectCheckpointCache(
        cache_path,
        interval=options.checkpoint_interval,
    )
    try:
        cache = checkpoint_cache.load()
    except CheckpointCacheError as exc:
        raise RelationValidationError(f"invalid relation cache: {exc}") from exc
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

    requests = [(item,) for item in pending]

    def judge_one(
        request: tuple[_PendingDecision, ...],
    ) -> list[tuple[_PendingDecision, RelationDecision]]:
        request_pairs = tuple(item.pair for item in request)
        raw = judge.judge(request_pairs, skills_by_id, contracts)
        if not isinstance(raw, dict):
            raise ValueError("relation judge output must be an object")
        decisions = _decisions_from_response(request_pairs, raw, skills_by_id)
        return list(zip(request, decisions, strict=True))

    def accept(
        outcome: LLMJobOutcome[list[tuple[_PendingDecision, RelationDecision]]],
    ) -> None:
        request_decisions = outcome.value
        if not isinstance(request_decisions, list):
            raise RelationValidationError("relation validation returned an invalid internal result")
        for pending_item, decision in request_decisions:
            if not isinstance(decision, RelationDecision):
                raise RelationValidationError(
                    "relation validation returned an invalid internal result"
                )
            records[pending_item.index] = decision
            checkpoint_cache.record(pending_item.cache_key, decision.judge_dict())

    try:
        outcomes = run_llm_jobs(
            requests,
            judge_one,
            options=options,
            label="relation",
            on_success=accept,
        )
    except Exception as exc:
        _flush_checkpoint(checkpoint_cache)
        if isinstance(exc, RelationValidationError):
            raise
        raise RelationValidationError(f"relation validation aborted: {exc}") from exc
    except BaseException:
        _flush_checkpoint(checkpoint_cache)
        raise
    _flush_checkpoint(checkpoint_cache)
    failures = [outcome for outcome in outcomes if not outcome.ok]
    if failures:
        first = failures[0]
        error = first.error or RuntimeError("unknown relation validation failure")
        pair_keys = [item.pair.key for item in first.item]
        raise RelationValidationError(
            f"relation validation failed for {pair_keys}: {error}"
        ) from error
    try:
        checkpoint_cache.retain(
            {_cache_key(pair, skills_by_id, contracts, judge.model_id) for pair in ordered_pairs}
        )
        checkpoint_cache.compact()
    except CheckpointCacheError as exc:
        raise RelationValidationError(f"failed to compact relation cache: {exc}") from exc
    return [record for record in records if record is not None]


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
    decisions_by_index: dict[int, RelationDecision] = {}
    for raw in raw_decisions:
        if not isinstance(raw, dict):
            raise ValueError("relation judge decisions must be objects")
        pair_index = _required_pair_index(raw.get("pair_index"), len(pairs))
        if pair_index in decisions_by_index:
            raise ValueError(f"relation judge returned duplicate pair_index: {pair_index}")
        decisions_by_index[pair_index] = decision_from_judge_payload(
            pairs[pair_index],
            raw,
            skills,
            pair_index=pair_index,
        )
    missing = sorted(set(range(len(pairs))) - set(decisions_by_index))
    if missing:
        raise ValueError(f"relation judge response is missing pair_index values: {missing}")
    return [decisions_by_index[index] for index in range(len(pairs))]


def decision_from_judge_payload(
    pair: CandidatePair,
    payload: dict[str, Any],
    skills: dict[str, SkillNode],
    *,
    pair_index: int,
) -> RelationDecision:
    """Map pair-local judge output to one canonical relation decision."""

    actual_keys = set(payload)
    if actual_keys != _JUDGE_DECISION_KEYS:
        _raise_key_mismatch("relation judge decision", actual_keys, _JUDGE_DECISION_KEYS)
    actual_pair_index = _required_pair_index(payload.get("pair_index"), pair_index + 1)
    if actual_pair_index != pair_index:
        raise ValueError(f"relation judge decision must use pair_index {pair_index}")
    relation = payload.get("relation")
    if not isinstance(relation, str) or relation not in _RELATIONS:
        raise ValueError("relation must be depend_on, compose_with, similar_to, or none")
    direction = payload.get("direction")
    if not isinstance(direction, str) or direction not in _DIRECTIONS:
        raise ValueError("direction must be skill_a_to_skill_b, skill_b_to_skill_a, or symmetric")
    if relation in {"depend_on", "compose_with"} and direction == "symmetric":
        raise ValueError(f"{relation} requires a directed pair-local direction")
    if relation in {"similar_to", "none"} and direction != "symmetric":
        raise ValueError(f"{relation} requires symmetric direction")
    confidence = _validated_confidence(payload.get("confidence"))
    reason = _required_string(payload.get("reason"), "reason")
    evidence = _validated_pair_local_evidence(payload.get("evidence"), pair=pair, skills=skills)
    if relation == "none" and evidence:
        raise ValueError("none relations require empty evidence line lists")
    if relation != "none" and {item.skill for item in evidence} != set(pair.key):
        raise ValueError("non-none relations require evidence from both candidate skills")
    if direction == "skill_a_to_skill_b":
        source, target = pair.skill_a, pair.skill_b
    elif direction == "skill_b_to_skill_a":
        source, target = pair.skill_b, pair.skill_a
    else:
        source, target = pair.key
    return RelationDecision(
        candidate=pair,
        relation=relation,
        source_skill=source,
        target_skill=target,
        confidence=confidence,
        reason=reason,
        evidence=tuple(evidence),
    )


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
        _raise_key_mismatch("relation decision", actual_keys, _DECISION_KEYS)
    relation = payload.get("relation")
    if not isinstance(relation, str) or relation not in _RELATIONS:
        raise ValueError("relation must be depend_on, compose_with, similar_to, or none")
    source = _required_string(payload.get("source_skill"), "source_skill")
    target = _required_string(payload.get("target_skill"), "target_skill")
    if source == target or {source, target} != set(pair.key):
        raise ValueError("source_skill and target_skill must be the candidate pair")
    confidence = _validated_confidence(payload.get("confidence"))
    reason = _required_string(payload.get("reason"), "reason")
    evidence = _validated_evidence(payload.get("evidence"), pair=pair, skills=skills)
    if relation != "none" and {item.skill for item in evidence} != set(pair.key):
        raise ValueError("non-none relations require evidence from both candidate skills")
    if relation in {"similar_to", "none"}:
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


def _validated_pair_local_evidence(
    value: Any,
    *,
    pair: CandidatePair,
    skills: dict[str, SkillNode],
) -> list[EvidenceRef]:
    if not isinstance(value, dict) or set(value) != _JUDGE_EVIDENCE_KEYS:
        raise ValueError("evidence must contain exactly skill_a_lines and skill_b_lines")
    return [
        *_validated_line_evidence(
            value["skill_a_lines"],
            label="evidence.skill_a_lines",
            skill_id=pair.skill_a,
            skills=skills,
        ),
        *_validated_line_evidence(
            value["skill_b_lines"],
            label="evidence.skill_b_lines",
            skill_id=pair.skill_b,
            skills=skills,
        ),
    ]


def _validated_line_evidence(
    value: Any,
    *,
    label: str,
    skill_id: str,
    skills: dict[str, SkillNode],
) -> list[EvidenceRef]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    source_lines = skills[skill_id].raw_text.splitlines()
    evidence: list[EvidenceRef] = []
    for index, line in enumerate(value):
        if isinstance(line, bool) or not isinstance(line, int):
            raise ValueError(f"{label}[{index}] must be an integer")
        if line < 1 or line > len(source_lines):
            raise ValueError(f"{label}[{index}] is outside the skill source")
        source_text = source_lines[line - 1]
        if not source_text.strip():
            raise ValueError(f"{label}[{index}] must reference a non-empty source line")
        evidence.append(EvidenceRef(skill=skill_id, line=line, text=source_text))
    return evidence


def _validated_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be a number")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return confidence


def _required_pair_index(value: Any, pair_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("pair_index must be an integer")
    if value < 0 or value >= pair_count:
        raise ValueError(f"relation judge returned unexpected pair_index: {value}")
    return value


def _raise_key_mismatch(label: str, actual: set[str], expected: frozenset[str]) -> None:
    missing = expected - actual
    unexpected = actual - expected
    details = []
    if missing:
        details.append(f"missing keys: {', '.join(sorted(missing))}")
    if unexpected:
        details.append(f"unexpected keys: {', '.join(sorted(unexpected))}")
    raise ValueError(f"{label} " + "; ".join(details))


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
        "prompt_fingerprint": RELATION_POLICY_FINGERPRINT,
        "model_id": model_id,
        "pair": list(pair.key),
        "skills": {skill_id: skills[skill_id].content_hash for skill_id in pair.key},
        "contracts": {skill_id: contracts[skill_id].to_dict() for skill_id in pair.key},
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _flush_checkpoint(cache: JsonObjectCheckpointCache) -> None:
    try:
        cache.flush()
    except CheckpointCacheError as exc:
        raise RelationValidationError(f"failed to checkpoint relation cache: {exc}") from exc
