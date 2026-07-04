"""Execution flow validation and normalization."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from skillfabric.compiled_graph.execution.models import (
    ExecutionEdge,
    ExecutionEvidence,
    ExecutionFlowCandidate,
    ExecutionValidationRecord,
)
from skillfabric.compiled_graph.execution.policy import (
    EXECUTION_POLICY_DIGEST,
    EXECUTION_POLICY_VERSION,
    classify_execution_candidate,
)
from skillfabric.compiled_graph.execution.prompts import (
    COMPACT_EXECUTION_PROMPT_ID,
    EXECUTION_PROMPT_ID,
    build_compact_execution_validation_messages,
    build_execution_validation_messages,
)
from skillfabric.compiled_graph.interface.models import SkillInterface
from skillfabric.registry.models import SkillNode
from skillfabric.runtime.jobs import LLMJobOptions, run_llm_jobs
from skillfabric.runtime.llm import LLMConfig, litellm_completion, response_to_jsonable


class ExecutionFlowValidator(Protocol):
    """Protocol for execution flow validators."""

    model_id: str

    def validate(
        self,
        candidate: ExecutionFlowCandidate,
        source_skill: SkillNode,
        target_skill: SkillNode,
        *,
        interfaces: dict[str, SkillInterface],
    ) -> dict[str, Any]:
        """Return an execution validation payload."""


@dataclass(slots=True)
class DeterministicExecutionFlowValidator:
    """Deterministic validator used when LLM validation is disabled."""

    model_id: str = "deterministic-execution"

    def validate(
        self,
        candidate: ExecutionFlowCandidate,
        source_skill: SkillNode,
        target_skill: SkillNode,
        *,
        interfaces: dict[str, SkillInterface],
    ) -> dict[str, Any]:
        accepted = bool(candidate.evidence)
        return {
            "accepted": accepted,
            "flow_type": candidate.flow_type if accepted else "none",
            "projected_edge_type": "depend_on" if accepted else "none",
            "confidence": 0.9 if accepted else 0.0,
            "evidence": [item.to_dict() for item in candidate.evidence],
            "reason": "Deterministic exact interface match." if accepted else "Missing candidate evidence.",
        }


@dataclass(slots=True)
class LiteLLMExecutionFlowValidator:
    """LiteLLM-backed validator for execution flow checks."""

    config: LLMConfig

    @property
    def model_id(self) -> str:
        return self.config.model

    @classmethod
    def from_env(cls, *, env_path: str | Path | None = None) -> LiteLLMExecutionFlowValidator:
        return cls(config=LLMConfig.from_env(env_path=env_path))

    def validate(
        self,
        candidate: ExecutionFlowCandidate,
        source_skill: SkillNode,
        target_skill: SkillNode,
        *,
        interfaces: dict[str, SkillInterface],
    ) -> dict[str, Any]:
        messages = build_compact_execution_validation_messages(
            candidate,
            source_skill,
            target_skill,
            interfaces=interfaces,
        )
        raw = self._validate_messages(
            messages,
            usage_metadata={
                "source_skill": source_skill.id,
                "target_skill": target_skill.id,
                "flow_type": candidate.flow_type,
                "prompt_tier": "compact",
            },
        )
        if not _should_escalate_to_full_prompt(raw):
            return raw
        full_messages = build_execution_validation_messages(
            candidate,
            source_skill,
            target_skill,
            interfaces=interfaces,
        )
        full_raw = self._validate_messages(
            full_messages,
            usage_metadata={
                "source_skill": source_skill.id,
                "target_skill": target_skill.id,
                "flow_type": candidate.flow_type,
                "prompt_tier": "full",
            },
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
                usage_operation="kg_build.execution_validation",
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
                    f"failed to parse execution JSON: {exc}",
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


def validate_execution_flow_candidates(
    candidates: list[ExecutionFlowCandidate],
    skills: list[SkillNode],
    *,
    interfaces: dict[str, SkillInterface],
    validator: ExecutionFlowValidator | None = None,
    cache_path: str | Path | None = None,
    job_options: LLMJobOptions | None = None,
) -> list[ExecutionValidationRecord]:
    """Validate execution flow candidates with cache and audit records."""

    validator = validator or DeterministicExecutionFlowValidator()
    by_id = {skill.id: skill for skill in skills}
    cache = _load_cache(cache_path)
    records: list[ExecutionValidationRecord | None] = [None] * len(candidates)
    pending: list[tuple[int, ExecutionFlowCandidate]] = []
    for index, candidate in enumerate(candidates):
        source_skill = by_id[candidate.source_skill]
        target_skill = by_id[candidate.target_skill]
        decision = classify_execution_candidate(candidate)
        if decision.action != "llm":
            records[index] = _normalize_record(
                candidate,
                _with_validation_meta(
                    _raw_from_decision(candidate, decision),
                    source=f"deterministic_{decision.action}",
                    prompt_tier="none",
                    model_id=validator.model_id,
                    cache_hit=False,
                ),
            )
            continue
        key = _cache_key(candidate, source_skill, target_skill, validator.model_id)
        raw = cache.get(key)
        if raw is None:
            pending.append((index, candidate))
            continue
        if not isinstance(raw, dict):
            raw = _error_payload("schema_error", "validator output must be a JSON object")
        if _is_retryable_error_payload(raw):
            cache.pop(key, None)
            pending.append((index, candidate))
            continue
        records[index] = _normalize_record(
            candidate,
            _with_validation_meta(
                raw,
                source=_meta_value(raw, "source", default="cache"),
                prompt_tier=_meta_value(raw, "prompt_tier", default=""),
                model_id=validator.model_id,
                cache_hit=True,
            ),
        )

    def validate_one(item: tuple[int, ExecutionFlowCandidate]) -> dict[str, Any]:
        _, candidate = item
        source_skill = by_id[candidate.source_skill]
        target_skill = by_id[candidate.target_skill]
        try:
            return validator.validate(candidate, source_skill, target_skill, interfaces=interfaces)
        except Exception as exc:
            return _error_payload("validator_error", f"{type(exc).__name__}: {exc}")

    def on_success(outcome) -> None:
        index, candidate = outcome.item
        source_skill = by_id[candidate.source_skill]
        target_skill = by_id[candidate.target_skill]
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
            cache[_cache_key(candidate, source_skill, target_skill, validator.model_id)] = raw
            _write_cache(cache_path, cache)
        records[index] = _normalize_record(candidate, raw)

    outcomes = run_llm_jobs(
        pending,
        validate_one,
        options=job_options,
        label="execution",
        retry_on_result=_is_retryable_error_payload,
        on_success=on_success,
    )
    for outcome in outcomes:
        if outcome.ok:
            continue
        index, candidate = outcome.item
        raw = _with_validation_meta(
            _error_payload("validator_error", f"{type(outcome.error).__name__}: {outcome.error}"),
            source="validator",
            prompt_tier="none",
            model_id=validator.model_id,
            cache_hit=False,
        )
        source_skill = by_id[candidate.source_skill]
        target_skill = by_id[candidate.target_skill]
        records[index] = _normalize_record(candidate, raw)
    _write_cache(cache_path, cache)
    return [record for record in records if record is not None]


def _raw_from_decision(candidate: ExecutionFlowCandidate, decision) -> dict[str, Any]:
    if decision.action == "accept":
        return {
            "accepted": decision.accepted,
            "flow_type": decision.flow_type,
            "projected_edge_type": decision.projected_edge_type,
            "confidence": decision.confidence,
            "evidence": [item.to_dict() for item in candidate.evidence[:4]],
            "reason": decision.reason,
        }
    return {
        "accepted": False,
        "flow_type": "none",
        "projected_edge_type": "none",
        "confidence": 0.0,
        "evidence": [],
        "reason": decision.reason,
    }


def summarize_execution_validation_records(records: list[ExecutionValidationRecord]) -> dict[str, Any]:
    """Summarize execution validation policy decisions for build metrics."""

    summary: dict[str, Any] = {
        "policy_version": EXECUTION_POLICY_VERSION,
        "policy_digest": EXECUTION_POLICY_DIGEST,
        "prompt_id": EXECUTION_PROMPT_ID,
        "compact_prompt_id": COMPACT_EXECUTION_PROMPT_ID,
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


def execution_validation_audit_rows(records: list[ExecutionValidationRecord]) -> list[dict[str, Any]]:
    """Return compact execution validation audit rows without skill content or secrets."""

    rows: list[dict[str, Any]] = []
    for record in records:
        meta = _validation_meta(record.raw_output)
        rows.append(
            {
                "stage": "execution_validation",
                "policy_version": EXECUTION_POLICY_VERSION,
                "policy_digest": EXECUTION_POLICY_DIGEST,
                "prompt_id": EXECUTION_PROMPT_ID,
                "compact_prompt_id": COMPACT_EXECUTION_PROMPT_ID,
                "action": _validation_action(meta),
                "source_skill": record.candidate.source_skill,
                "target_skill": record.candidate.target_skill,
                "flow_type": record.candidate.flow_type,
                "matched_node_id": record.candidate.matched_node_id,
                "matched_name": record.candidate.matched_name,
                "candidate_prior": round(float(record.candidate.prior), 6),
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
                "projected_edge_type": str(record.normalized.get("projected_edge_type", "none")),
                "confidence": round(float(record.normalized.get("confidence", 0.0) or 0.0), 6),
            }
        )
    return rows


def _normalize_record(candidate: ExecutionFlowCandidate, raw: dict[str, Any]) -> ExecutionValidationRecord:
    if raw.get("error_type"):
        normalized = _none_normalized(str(raw.get("reason", "")))
        return ExecutionValidationRecord(candidate, raw, normalized, False, f"{raw.get('error_type')}: {raw.get('reason', '')}")
    schema_error = _schema_error(raw)
    if schema_error:
        normalized = _none_normalized(schema_error)
        return ExecutionValidationRecord(candidate, raw, normalized, False, f"schema_error: {schema_error}")

    accepted = bool(raw.get("accepted", False))
    flow_type = str(raw.get("flow_type", "none"))
    projected_edge_type = str(raw.get("projected_edge_type", "none"))
    confidence = float(raw.get("confidence", 0.0) or 0.0)
    evidence = _evidence_from_raw(raw.get("evidence", []))
    normalized = {
        "accepted": accepted,
        "flow_type": flow_type,
        "projected_edge_type": projected_edge_type,
        "confidence": round(confidence, 6),
        "evidence": [item.to_dict() for item in evidence],
        "reason": str(raw.get("reason", "")),
    }
    rejection = _rejection_reason(accepted, flow_type, projected_edge_type, confidence, evidence)
    if rejection:
        return ExecutionValidationRecord(candidate, raw, normalized, False, rejection)
    flow_edge = ExecutionEdge(
        source=candidate.source_skill,
        target=candidate.target_skill,
        type=flow_type,
        confidence=round(confidence, 6),
        weight=round(confidence, 6),
        evidence=evidence,
        reason=str(raw.get("reason", "")),
        metadata=_flow_metadata(candidate),
    )
    return ExecutionValidationRecord(candidate, raw, normalized, True, "", flow_edge)


def _schema_error(raw: dict[str, Any]) -> str:
    if not isinstance(raw.get("accepted", False), bool):
        return "accepted must be a boolean"
    if str(raw.get("flow_type", "none")) not in {"artifact_flow", "scenario_transition", "none"}:
        return "unsupported flow_type"
    if str(raw.get("projected_edge_type", "none")) not in {"depend_on", "compose_with", "none"}:
        return "unsupported projected_edge_type"
    if not _is_float(raw.get("confidence", 0.0)):
        return "confidence must be numeric"
    if not isinstance(raw.get("evidence", []), list):
        return "evidence must be a list"
    if not _valid_evidence_payload(raw.get("evidence", [])):
        return "evidence items must include numeric line values"
    return ""


def _rejection_reason(
    accepted: bool,
    flow_type: str,
    projected_edge_type: str,
    confidence: float,
    evidence: list[ExecutionEvidence],
) -> str:
    if not accepted:
        return "accepted is false"
    if flow_type not in {"artifact_flow", "scenario_transition"}:
        return "flow_type is none or unsupported"
    if projected_edge_type not in {"depend_on", "compose_with"}:
        return "projected_edge_type is none or unsupported"
    if confidence < 0.85:
        return "confidence below threshold"
    if not evidence:
        return "missing evidence"
    return ""


def _flow_metadata(candidate: ExecutionFlowCandidate) -> dict[str, str]:
    key = "artifact_id" if candidate.flow_type == "artifact_flow" else "scenario_id"
    return {
        key: candidate.matched_node_id,
        "matched_name": candidate.matched_name,
        **candidate.metadata,
    }


def _evidence_from_raw(payload: Any) -> list[ExecutionEvidence]:
    if not isinstance(payload, list):
        return []
    return [ExecutionEvidence.from_dict(item) for item in payload if isinstance(item, dict)]


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


def _none_normalized(reason: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "flow_type": "none",
        "projected_edge_type": "none",
        "confidence": 0.0,
        "evidence": [],
        "reason": reason,
    }


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
        return _error_payload("schema_error", "execution JSON root must be an object", raw_response=text)
    return payload


def _error_payload(error_type: str, reason: str, *, raw_response: str = "") -> dict[str, Any]:
    return {
        "accepted": False,
        "flow_type": "none",
        "projected_edge_type": "none",
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
            "policy_version": EXECUTION_POLICY_VERSION,
            "policy_digest": EXECUTION_POLICY_DIGEST,
            "prompt_id": EXECUTION_PROMPT_ID,
            "compact_prompt_id": COMPACT_EXECUTION_PROMPT_ID,
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


def _cache_key(
    candidate: ExecutionFlowCandidate,
    source_skill: SkillNode,
    target_skill: SkillNode,
    model_id: str,
) -> str:
    raw = json.dumps(
        {
            "source_skill": source_skill.id,
            "source_hash": source_skill.content_hash,
            "target_skill": target_skill.id,
            "target_hash": target_skill.content_hash,
            "model_id": model_id,
            "prompt_id": EXECUTION_PROMPT_ID,
            "compact_prompt_id": COMPACT_EXECUTION_PROMPT_ID,
            "policy_version": EXECUTION_POLICY_VERSION,
            "policy_digest": EXECUTION_POLICY_DIGEST,
            "candidate": candidate.to_dict(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
