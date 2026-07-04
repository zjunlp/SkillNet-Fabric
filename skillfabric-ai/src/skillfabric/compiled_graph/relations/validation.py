"""Pairwise relation validation and canonical edge conversion."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from skillfabric.compiled_graph.execution.models import ExecutionValidationRecord
from skillfabric.compiled_graph.interface.models import SkillInterface
from skillfabric.compiled_graph.models import Edge, EvidenceRef
from skillfabric.compiled_graph.relations.models import CandidatePair, ValidationRecord
from skillfabric.compiled_graph.relations.policy import (
    RELATION_POLICY_DIGEST,
    RELATION_POLICY_VERSION,
    classify_relation_candidate,
)
from skillfabric.compiled_graph.relations.prompts import (
    COMPACT_RELATION_PROMPT_ID,
    RELATION_PROMPT_ID,
    build_compact_pair_validation_messages,
    build_pair_validation_messages,
)
from skillfabric.registry.models import SkillNode
from skillfabric.runtime.jobs import LLMJobOptions, run_llm_jobs
from skillfabric.runtime.llm import LLMConfig, litellm_completion, response_to_jsonable


class PairValidator(Protocol):
    """Protocol for pairwise relation validators."""

    model_id: str

    def validate(
        self,
        skill_a: SkillNode,
        skill_b: SkillNode,
        pair: CandidatePair,
        *,
        interfaces: dict[str, SkillInterface] | None = None,
        execution_records: list[ExecutionValidationRecord] | None = None,
    ) -> dict[str, Any]:
        """Return a relation validation payload."""


@dataclass(slots=True)
class NoopPairValidator:
    """Validator used when LLM relation validation is disabled."""

    model_id: str = "noop"

    def validate(
        self,
        skill_a: SkillNode,
        skill_b: SkillNode,
        pair: CandidatePair,
        *,
        interfaces: dict[str, SkillInterface] | None = None,
        execution_records: list[ExecutionValidationRecord] | None = None,
    ) -> dict[str, Any]:
        return {
            "edge_type": "none",
            "direction": "none",
            "confidence": 0.0,
            "evidence": [],
            "reason": "LLM validation disabled.",
        }


@dataclass(slots=True)
class StaticPairValidator:
    """Deterministic validator for tests."""

    responses: dict[tuple[str, str], dict[str, Any]]
    model_id: str = "static"

    def validate(
        self,
        skill_a: SkillNode,
        skill_b: SkillNode,
        pair: CandidatePair,
        *,
        interfaces: dict[str, SkillInterface] | None = None,
        execution_records: list[ExecutionValidationRecord] | None = None,
    ) -> dict[str, Any]:
        direct = self.responses.get((skill_a.id, skill_b.id))
        if direct is not None:
            return direct
        reverse = self.responses.get((skill_b.id, skill_a.id))
        if reverse is not None:
            payload = dict(reverse)
            payload["direction"] = _flip_direction(str(payload.get("direction", "none")))
            return payload
        return {
            "edge_type": "none",
            "direction": "none",
            "confidence": 0.0,
            "evidence": [],
            "reason": "No static validation response.",
        }


@dataclass(slots=True)
class LiteLLMPairValidator:
    """LiteLLM-backed validator for pairwise relation checks."""

    config: LLMConfig

    @property
    def model_id(self) -> str:
        return self.config.model

    @classmethod
    def from_env(cls, *, env_path: str | Path | None = None) -> LiteLLMPairValidator:
        return cls(config=LLMConfig.from_env(env_path=env_path))

    def validate(
        self,
        skill_a: SkillNode,
        skill_b: SkillNode,
        pair: CandidatePair,
        *,
        interfaces: dict[str, SkillInterface] | None = None,
        execution_records: list[ExecutionValidationRecord] | None = None,
    ) -> dict[str, Any]:
        messages = build_compact_pair_validation_messages(
            skill_a,
            skill_b,
            pair,
            interfaces=interfaces,
            execution_records=execution_records,
        )
        raw = self._validate_messages(
            messages,
            usage_metadata={"skill_a": skill_a.id, "skill_b": skill_b.id, "prompt_tier": "compact"},
        )
        if not _should_escalate_to_full_prompt(raw):
            return raw
        full_messages = build_pair_validation_messages(
            skill_a,
            skill_b,
            pair,
            interfaces=interfaces,
            execution_records=execution_records,
        )
        full_raw = self._validate_messages(
            full_messages,
            usage_metadata={"skill_a": skill_a.id, "skill_b": skill_b.id, "prompt_tier": "full"},
        )
        return _with_compact_escalation_meta(full_raw)

    def _validate_messages(
        self,
        messages: list[dict[str, str]],
        *,
        usage_metadata: dict[str, str],
    ) -> dict[str, Any]:
        try:
            response = litellm_completion(
                messages=messages,
                config=self.config,
                usage_operation="kg_build.relation_validation",
                usage_metadata=usage_metadata,
            )
            response_text = _extract_response_text(response)
            return _with_validation_meta(
                _parse_validation_json(response_text),
                source="llm",
                prompt_tier=str(usage_metadata.get("prompt_tier", "")),
                model_id=self.model_id,
                cache_hit=False,
            )
        except json.JSONDecodeError as exc:
            return _with_validation_meta(
                _error_payload(
                    "json_parse_error",
                    f"failed to parse validator JSON: {exc}",
                    raw_response=locals().get("response_text", ""),
                ),
                source="llm",
                prompt_tier=str(usage_metadata.get("prompt_tier", "")),
                model_id=self.model_id,
                cache_hit=False,
            )
        except Exception as exc:
            return _with_validation_meta(
                _error_payload("api_error", f"{type(exc).__name__}: {exc}"),
                source="llm",
                prompt_tier=str(usage_metadata.get("prompt_tier", "")),
                model_id=self.model_id,
                cache_hit=False,
            )


def validate_relation_candidates(
    candidates: list[CandidatePair],
    skills: list[SkillNode],
    *,
    validator: PairValidator | None = None,
    cache_path: str | Path | None = None,
    interfaces: dict[str, SkillInterface] | None = None,
    execution_records: list[ExecutionValidationRecord] | None = None,
    job_options: LLMJobOptions | None = None,
) -> list[ValidationRecord]:
    """Validate relation candidates and return auditable records."""

    validator = validator or NoopPairValidator()
    by_id = {skill.id: skill for skill in skills}
    cache = _load_cache(cache_path)
    records: list[ValidationRecord | None] = [None] * len(candidates)
    pending: list[tuple[int, CandidatePair]] = []
    for index, pair in enumerate(candidates):
        skill_a = by_id[pair.skill_a]
        skill_b = by_id[pair.skill_b]
        decision = classify_relation_candidate(pair)
        if decision.action != "llm":
            records[index] = _normalize_record(
                pair,
                _with_validation_meta(
                    _raw_from_decision(pair, decision),
                    source=f"deterministic_{decision.action}",
                    prompt_tier="none",
                    model_id=validator.model_id,
                    cache_hit=False,
                ),
            )
            continue
        key = _cache_key(pair, skill_a, skill_b, validator.model_id, execution_records=execution_records)
        raw = cache.get(key)
        if raw is None:
            pending.append((index, pair))
            continue
        if not isinstance(raw, dict):
            raw = _error_payload("schema_error", "validator output must be a JSON object")
        if _is_retryable_error_payload(raw):
            cache.pop(key, None)
            pending.append((index, pair))
            continue
        records[index] = _normalize_record(
            pair,
            _with_validation_meta(
                raw,
                source=_meta_value(raw, "source", default="cache"),
                prompt_tier=_meta_value(raw, "prompt_tier", default=""),
                model_id=validator.model_id,
                cache_hit=True,
            ),
        )

    def validate_one(item: tuple[int, CandidatePair]) -> dict[str, Any]:
        _, pair = item
        skill_a = by_id[pair.skill_a]
        skill_b = by_id[pair.skill_b]
        try:
            return validator.validate(
                skill_a,
                skill_b,
                pair,
                interfaces=interfaces,
                execution_records=execution_records,
            )
        except Exception as exc:
            return _error_payload("validator_error", f"{type(exc).__name__}: {exc}")

    def on_success(outcome) -> None:
        index, pair = outcome.item
        skill_a = by_id[pair.skill_a]
        skill_b = by_id[pair.skill_b]
        raw = outcome.value
        if not isinstance(raw, dict):
            raw = _error_payload("schema_error", "validator output must be a JSON object")
        raw = _with_validation_meta(
            raw,
            source=_meta_value(raw, "source", default="validator"),
            prompt_tier=_meta_value(raw, "prompt_tier", default="none"),
            model_id=validator.model_id,
            cache_hit=False,
        )
        if not _is_retryable_error_payload(raw):
            cache[_cache_key(pair, skill_a, skill_b, validator.model_id, execution_records=execution_records)] = raw
            _write_cache(cache_path, cache)
        records[index] = _normalize_record(pair, raw)

    outcomes = run_llm_jobs(
        pending,
        validate_one,
        options=job_options,
        label="relations",
        retry_on_result=_is_retryable_error_payload,
        on_success=on_success,
    )
    for outcome in outcomes:
        if outcome.ok:
            continue
        index, pair = outcome.item
        raw = _with_validation_meta(
            _error_payload("validator_error", f"{type(outcome.error).__name__}: {outcome.error}"),
            source="validator",
            prompt_tier="none",
            model_id=validator.model_id,
            cache_hit=False,
        )
        skill_a = by_id[pair.skill_a]
        skill_b = by_id[pair.skill_b]
        records[index] = _normalize_record(pair, raw)
    _write_cache(cache_path, cache)
    return [record for record in records if record is not None]


def _raw_from_decision(pair: CandidatePair, decision) -> dict[str, Any]:
    if decision.action == "accept":
        return {
            "edge_type": decision.edge_type,
            "direction": decision.direction,
            "confidence": decision.confidence,
            "evidence": [_relation_evidence_to_edge_evidence(item) for item in pair.evidence[:4]],
            "reason": decision.reason,
        }
    return {
        "edge_type": "none",
        "direction": "none",
        "confidence": 0.0,
        "evidence": [],
        "reason": decision.reason,
        "error_type": "deterministic_reject",
    }


def summarize_relation_validation_records(records: list[ValidationRecord]) -> dict[str, Any]:
    """Summarize relation validation policy decisions for build metrics."""

    summary: dict[str, Any] = {
        "policy_version": RELATION_POLICY_VERSION,
        "policy_digest": RELATION_POLICY_DIGEST,
        "prompt_id": RELATION_PROMPT_ID,
        "compact_prompt_id": COMPACT_RELATION_PROMPT_ID,
        "total": len(records),
        "accepted": sum(1 for record in records if record.accepted),
        "rejected": sum(1 for record in records if not record.accepted),
        "deterministic_accept": 0,
        "deterministic_reject": 0,
        "llm_compact": 0,
        "llm_full": 0,
        "cache_hits": 0,
        "validator_calls": 0,
    }
    for record in records:
        meta = _validation_meta(record.raw_output)
        source = str(meta.get("source", ""))
        prompt_tier = str(meta.get("prompt_tier", ""))
        compact_attempted = bool(meta.get("compact_attempted", prompt_tier == "compact"))
        full_attempted = bool(meta.get("full_attempted", prompt_tier == "full"))
        cache_hit = bool(meta.get("cache_hit", False))
        if bool(meta.get("cache_hit", False)):
            summary["cache_hits"] += 1
        if source == "deterministic_accept":
            summary["deterministic_accept"] += 1
        elif source == "deterministic_reject":
            summary["deterministic_reject"] += 1
        elif compact_attempted or prompt_tier == "compact":
            summary["llm_compact"] += 1
        if full_attempted or prompt_tier == "full":
            summary["llm_full"] += 1
        if source not in {"deterministic_accept", "deterministic_reject"} and not cache_hit:
            summary["validator_calls"] += int(compact_attempted) + int(full_attempted)
    return summary


def relation_validation_audit_rows(records: list[ValidationRecord]) -> list[dict[str, Any]]:
    """Return compact relation validation audit rows without skill content or secrets."""

    rows: list[dict[str, Any]] = []
    for record in records:
        meta = _validation_meta(record.raw_output)
        rows.append(
            {
                "stage": "relation_validation",
                "policy_version": RELATION_POLICY_VERSION,
                "policy_digest": RELATION_POLICY_DIGEST,
                "prompt_id": RELATION_PROMPT_ID,
                "compact_prompt_id": COMPACT_RELATION_PROMPT_ID,
                "action": _validation_action(meta),
                "skill_a": record.pair.skill_a,
                "skill_b": record.pair.skill_b,
                "candidate_sources": list(record.pair.sources),
                "candidate_prior": round(float(record.pair.prior), 6),
                "direction_hint": record.pair.direction_hint,
                "source": str(meta.get("source", "")),
                "prompt_tier": str(meta.get("prompt_tier", "")),
                "compact_attempted": bool(meta.get("compact_attempted", False)),
                "full_attempted": bool(meta.get("full_attempted", False)),
                "escalated_from_compact": bool(meta.get("escalated_from_compact", False)),
                "cache_hit": bool(meta.get("cache_hit", False)),
                "model_id": str(meta.get("model_id", "")),
                "accepted": record.accepted,
                "rejection_reason": record.rejection_reason,
                "reason": str(record.normalized.get("reason", record.raw_output.get("reason", ""))),
                "edge_type": str(record.normalized.get("edge_type", "none")),
                "direction": str(record.normalized.get("direction", "none")),
                "confidence": round(float(record.normalized.get("confidence", 0.0) or 0.0), 6),
            }
        )
    return rows


def _relation_evidence_to_edge_evidence(evidence: Any) -> dict[str, Any]:
    return {
        "skill": str(getattr(evidence, "skill_id", "")),
        "line": int(getattr(evidence, "line", 0) or 0),
        "text": str(getattr(evidence, "text", "")),
    }


def _normalize_record(pair: CandidatePair, raw: dict[str, Any]) -> ValidationRecord:
    if raw.get("error_type"):
        normalized = {
            "edge_type": "none",
            "direction": "none",
            "confidence": 0.0,
            "evidence": [],
            "reason": str(raw.get("reason", "")),
        }
        return ValidationRecord(pair, raw, normalized, False, f"{raw.get('error_type')}: {raw.get('reason', '')}")
    schema_error = _schema_error(raw)
    if schema_error:
        normalized = {
            "edge_type": "none",
            "direction": "none",
            "confidence": 0.0,
            "evidence": [],
            "reason": schema_error,
        }
        return ValidationRecord(pair, raw, normalized, False, f"schema_error: {schema_error}")
    edge_type = str(raw.get("edge_type", "none"))
    direction = str(raw.get("direction", "none"))
    confidence = float(raw.get("confidence", 0.0) or 0.0)
    evidence = _evidence_from_raw(raw.get("evidence", []))
    normalized = {
        "edge_type": edge_type,
        "direction": direction,
        "confidence": round(confidence, 6),
        "evidence": [item.to_dict() for item in evidence],
        "reason": str(raw.get("reason", "")),
    }
    rejection = _rejection_reason(pair, edge_type, direction, confidence, evidence, reason=str(raw.get("reason", "")))
    if rejection:
        return ValidationRecord(pair, raw, normalized, False, rejection)
    source, target = _resolve_direction(pair, edge_type, direction)
    provenance = _provenance(pair, raw)
    edge = Edge(
        source=source,
        target=target,
        type=edge_type,
        confidence=round(confidence, 6),
        weight=round(_edge_weight(edge_type, provenance, confidence), 6),
        provenance=provenance,
        evidence=evidence,
        reason=str(raw.get("reason", "")),
    )
    return ValidationRecord(pair, raw, normalized, True, "", edge)


def _schema_error(raw: dict[str, Any]) -> str:
    if str(raw.get("edge_type", "none")) not in {"compose_with", "depend_on", "none"}:
        return "unsupported edge_type"
    if str(raw.get("direction", "none")) not in {"A->B", "B->A", "undirected", "none"}:
        return "unsupported direction"
    if not _is_float(raw.get("confidence", 0.0)):
        return "confidence must be numeric"
    if not isinstance(raw.get("evidence", []), list):
        return "evidence must be a list"
    if not _valid_evidence_payload(raw.get("evidence", [])):
        return "evidence items must include numeric line values"
    return ""


def _rejection_reason(
    pair: CandidatePair,
    edge_type: str,
    direction: str,
    confidence: float,
    evidence: list[EvidenceRef],
    *,
    reason: str = "",
) -> str:
    if edge_type not in {"compose_with", "depend_on"}:
        return "edge_type is none or unsupported"
    if not evidence:
        return "missing evidence"
    if _reason_uses_missing_third_skill(reason):
        return "reason relies on a third skill outside the candidate pair"
    if not _evidence_connects_pair(pair, evidence):
        return "evidence does not connect both candidate skills"
    if edge_type == "depend_on":
        if confidence < 0.85:
            return "depend_on confidence below threshold"
        if direction not in {"A->B", "B->A"}:
            return "depend_on requires directed evidence"
        if _conflicts_with_execution_direction_hint(pair, direction):
            return f"depend_on direction conflicts with execution_flow direction_hint {pair.direction_hint}"
    if edge_type == "compose_with" and confidence < 0.75:
        return "compose_with confidence below threshold"
    return ""


def _evidence_connects_pair(pair: CandidatePair, evidence: list[EvidenceRef]) -> bool:
    pair_ids = {pair.skill_a, pair.skill_b}
    cited_skills = {item.skill for item in evidence}
    if pair_ids.issubset(cited_skills):
        return True
    for item in evidence:
        text = item.text.lower()
        if item.skill == pair.skill_a and _mentions_skill(text, pair.skill_b):
            return True
        if item.skill == pair.skill_b and _mentions_skill(text, pair.skill_a):
            return True
    return False


def _mentions_skill(text: str, skill_id: str) -> bool:
    suffix = skill_id.split(":", 1)[-1].lower()
    aliases = {
        skill_id.lower(),
        suffix,
        suffix.replace("-", " "),
        suffix.replace("-", "_"),
    }
    return any(alias and alias in text for alias in aliases)


def _reason_uses_missing_third_skill(reason: str) -> bool:
    normalized = reason.lower()
    third_skill_phrases = (
        "other skill id was not provided",
        "not provided in the candidate",
        "outside the candidate",
        "third skill",
    )
    return any(phrase in normalized for phrase in third_skill_phrases)


def _conflicts_with_execution_direction_hint(pair: CandidatePair, direction: str) -> bool:
    if "execution_flow" not in pair.sources:
        return False
    if pair.direction_hint not in {"A->B", "B->A"}:
        return False
    return direction != pair.direction_hint


def _evidence_from_raw(payload: Any) -> list[EvidenceRef]:
    if not isinstance(payload, list):
        return []
    return [EvidenceRef.from_dict(item) for item in payload if isinstance(item, dict)]


def _is_float(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _valid_evidence_payload(payload: Any) -> bool:
    if not isinstance(payload, list):
        return False
    for item in payload:
        if not isinstance(item, dict):
            return False
        try:
            int(item.get("line", 0))
        except (TypeError, ValueError):
            return False
    return True


def _extract_response_text(response: Any) -> str:
    payload = response_to_jsonable(response)
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and message.get("content") is not None:
                    return str(message["content"])
                if first.get("text") is not None:
                    return str(first["text"])
        if payload.get("output_text") is not None:
            return str(payload["output_text"])
        output = payload.get("output")
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("text") is not None:
                            parts.append(str(part["text"]))
            if parts:
                return "\n".join(parts)
    return str(payload)


def _parse_validation_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        return _error_payload("schema_error", "validator JSON root must be an object", raw_response=text)
    return payload


def _error_payload(error_type: str, reason: str, *, raw_response: str = "") -> dict[str, Any]:
    return {
        "edge_type": "none",
        "direction": "none",
        "confidence": 0.0,
        "evidence": [],
        "reason": reason,
        "error_type": error_type,
        "raw_response": raw_response,
    }


def _is_retryable_error_payload(raw: dict[str, Any]) -> bool:
    return str(raw.get("error_type", "")) in {"api_error", "json_parse_error", "validator_error"}


def _should_escalate_to_full_prompt(raw: dict[str, Any]) -> bool:
    reason = str(raw.get("reason", "")).lower()
    if str(raw.get("needs_full_context", "")).lower() == "true":
        return True
    return any(marker in reason for marker in ("need full", "insufficient context", "uncertain"))


def _with_validation_meta(
    raw: dict[str, Any],
    *,
    source: str,
    prompt_tier: str,
    model_id: str,
    cache_hit: bool,
) -> dict[str, Any]:
    payload = dict(raw)
    existing = _validation_meta(payload)
    resolved_source = source or str(existing.get("source", ""))
    resolved_prompt_tier = prompt_tier or str(existing.get("prompt_tier", ""))
    meta = dict(existing)
    meta.update(
        {
            "source": resolved_source,
            "prompt_tier": resolved_prompt_tier,
            "model_id": model_id or str(existing.get("model_id", "")),
            "cache_hit": cache_hit,
            "policy_version": RELATION_POLICY_VERSION,
            "policy_digest": RELATION_POLICY_DIGEST,
            "prompt_id": RELATION_PROMPT_ID,
            "compact_prompt_id": COMPACT_RELATION_PROMPT_ID,
            "reason": str(payload.get("reason", existing.get("reason", ""))),
        }
    )
    if resolved_prompt_tier == "compact":
        meta.setdefault("compact_attempted", True)
        meta.setdefault("full_attempted", False)
    elif resolved_prompt_tier == "full":
        meta.setdefault("compact_attempted", False)
        meta.setdefault("full_attempted", True)
    else:
        meta.setdefault("compact_attempted", False)
        meta.setdefault("full_attempted", False)
    meta["action"] = str(meta.get("action") or _validation_action(meta))
    payload["_validation_meta"] = meta
    return payload


def _with_compact_escalation_meta(raw: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw)
    meta = dict(_validation_meta(payload))
    meta["compact_attempted"] = True
    meta["full_attempted"] = True
    meta["escalated_from_compact"] = True
    meta["action"] = "llm_full"
    payload["_validation_meta"] = meta
    return payload


def _validation_meta(raw: dict[str, Any]) -> dict[str, Any]:
    meta = raw.get("_validation_meta", {})
    return meta if isinstance(meta, dict) else {}


def _meta_value(raw: dict[str, Any], key: str, *, default: str) -> str:
    value = _validation_meta(raw).get(key, default)
    return str(value or default)


def _validation_action(meta: dict[str, Any]) -> str:
    source = str(meta.get("source", ""))
    if source in {"deterministic_accept", "deterministic_reject"}:
        return source
    prompt_tier = str(meta.get("prompt_tier", ""))
    if prompt_tier == "full" or bool(meta.get("full_attempted", False)):
        return "llm_full"
    if prompt_tier == "compact" or bool(meta.get("compact_attempted", False)):
        return "llm_compact"
    if bool(meta.get("cache_hit", False)):
        return "cache"
    return source or "validator"


def _resolve_direction(pair: CandidatePair, edge_type: str, direction: str) -> tuple[str, str]:
    if edge_type == "compose_with":
        return pair.skill_a, pair.skill_b
    if direction == "A->B":
        return pair.skill_a, pair.skill_b
    if direction == "B->A":
        return pair.skill_b, pair.skill_a
    return pair.skill_a, pair.skill_b


def _flip_direction(direction: str) -> str:
    if direction == "A->B":
        return "B->A"
    if direction == "B->A":
        return "A->B"
    return direction


def _provenance(pair: CandidatePair, raw: dict[str, Any]) -> str:
    meta = _validation_meta(raw)
    if str(meta.get("source", "")) == "deterministic_accept":
        return "deterministic_accept"
    if "explicit_mention" in pair.sources:
        return "explicit_mention"
    return "llm_validated"


def _edge_weight(edge_type: str, provenance: str, confidence: float) -> float:
    base = 1.0 if edge_type == "depend_on" else 0.8
    factor = 1.0 if provenance in {"explicit_mention", "deterministic_accept"} else 0.8
    return base * factor * confidence


def _cache_key(
    pair: CandidatePair,
    skill_a: SkillNode,
    skill_b: SkillNode,
    model_id: str,
    *,
    execution_records: list[ExecutionValidationRecord] | None = None,
) -> str:
    raw = json.dumps(
        {
            "skill_a": skill_a.id,
            "hash_a": skill_a.content_hash,
            "skill_b": skill_b.id,
            "hash_b": skill_b.content_hash,
            "model_id": model_id,
            "prompt_id": RELATION_PROMPT_ID,
            "compact_prompt_id": COMPACT_RELATION_PROMPT_ID,
            "policy_version": RELATION_POLICY_VERSION,
            "policy_digest": RELATION_POLICY_DIGEST,
            "candidate": pair.to_prompt_dict(),
            "execution_summary": _execution_cache_payload(pair, execution_records or []),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _execution_cache_payload(
    pair: CandidatePair,
    records: list[ExecutionValidationRecord],
) -> list[dict[str, Any]]:
    pair_skills = {pair.skill_a, pair.skill_b}
    return [
        {
            "candidate": record.candidate.to_dict(),
            "normalized": record.normalized,
            "accepted": record.accepted,
        }
        for record in records
        if {record.candidate.source_skill, record.candidate.target_skill} == pair_skills
    ]


def _load_cache(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def _write_cache(path: str | Path | None, payload: dict[str, dict[str, Any]]) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
