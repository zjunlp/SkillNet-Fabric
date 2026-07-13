"""Strict pair-level semantic validation with validated result caching."""

from __future__ import annotations

import hashlib
import json
import math
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


class RelationValidationError(RuntimeError):
    """Raised when any candidate cannot receive a valid semantic decision."""


class RelationJudge(Protocol):
    """Provider protocol for one pair-level semantic judgment."""

    model_id: str

    def judge(
        self,
        pair: CandidatePair,
        skills: dict[str, SkillNode],
        contracts: dict[str, SkillContract],
    ) -> dict[str, Any]:
        """Return one raw semantic decision."""


@dataclass(slots=True)
class LiteLLMRelationJudge:
    """Production full-source semantic judge."""

    config: LLMConfig

    @property
    def model_id(self) -> str:
        return self.config.model

    @classmethod
    def from_env(cls, *, env_path: str | Path | None = None) -> LiteLLMRelationJudge:
        return cls(LLMConfig.from_env(env_path=env_path))

    def judge(
        self,
        pair: CandidatePair,
        skills: dict[str, SkillNode],
        contracts: dict[str, SkillContract],
    ) -> dict[str, Any]:
        response = litellm_completion(
            messages=build_relation_judge_messages(pair, skills, contracts),
            config=self.config,
            usage_operation="graph.semantic_relation",
            usage_metadata={"skill_a": pair.skill_a, "skill_b": pair.skill_b},
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
    pending: list[tuple[int, CandidatePair, str]] = []
    for index, pair in enumerate(ordered_pairs):
        key = _cache_key(pair, skills_by_id, contracts, judge.model_id)
        cached = cache.get(key)
        if cached is None:
            pending.append((index, pair, key))
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

    def judge_one(item: tuple[int, CandidatePair, str]) -> RelationDecision:
        _, pair, _ = item
        raw = judge.judge(pair, skills_by_id, contracts)
        if not isinstance(raw, dict):
            raise ValueError("relation judge output must be an object")
        return decision_from_payload(
            pair,
            raw,
            skills_by_id,
        )

    outcomes = run_llm_jobs(pending, judge_one, options=job_options, label="relation")
    additions: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        if not outcome.ok:
            continue
        index, _pair, key = outcome.item
        decision = outcome.value
        if not isinstance(decision, RelationDecision):
            raise RelationValidationError("relation validation returned an invalid internal result")
        records[index] = decision
        additions[key] = decision.judge_dict()
    if additions:
        cache.update(additions)
        _write_cache(cache_path, cache)
    failures = [outcome for outcome in outcomes if not outcome.ok]
    if failures:
        first = failures[0]
        _, pair, _ = first.item
        error = first.error or RuntimeError("unknown relation validation failure")
        raise RelationValidationError(
            f"relation validation failed for {pair.key}: {error}"
        ) from error
    return [record for record in records if record is not None]


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
