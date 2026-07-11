"""Local LLM token and cost usage accounting."""

from __future__ import annotations

import json
import math
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skillfabric.runtime.json_utils import extract_response_text

_PATH_LOCKS: dict[Path, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-token model pricing used when LiteLLM does not know a model alias."""

    input_per_token: float
    output_per_token: float
    cached_input_per_token: float | None = None
    source: str = "skillfabric"


_TOKENS_PER_MILLION = 1_000_000

# Official OpenAI API pricing snapshot used for SkillFabric's custom model aliases.
# Prices are USD per token. Update this table when upstream provider prices change.
_SKILLFABRIC_PRICING_OVERRIDES: dict[str, ModelPricing] = {
    "gpt-5.4-mini": ModelPricing(
        input_per_token=0.75 / _TOKENS_PER_MILLION,
        cached_input_per_token=0.075 / _TOKENS_PER_MILLION,
        output_per_token=4.50 / _TOKENS_PER_MILLION,
        source="skillfabric_openai_official",
    ),
}


@dataclass(slots=True)
class LLMUsageRecord:
    """Usage data for one LLM completion call."""

    timestamp: str
    call_id: str
    operation: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None
    estimated: bool
    pricing_known: bool
    duration_ms: int
    status: str
    cached_prompt_tokens: int = 0
    billable_prompt_tokens: int = 0
    pricing_source: str = "unknown"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "call_id": self.call_id,
            "operation": self.operation,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "billable_prompt_tokens": self.billable_prompt_tokens,
            "cost_usd": self.cost_usd,
            "estimated": self.estimated,
            "pricing_known": self.pricing_known,
            "pricing_source": self.pricing_source,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LLMUsageRecord:
        return cls(
            timestamp=str(payload.get("timestamp", "")),
            call_id=str(payload.get("call_id", "")),
            operation=str(payload.get("operation", "")),
            model=str(payload.get("model", "")),
            prompt_tokens=int(payload.get("prompt_tokens", 0) or 0),
            completion_tokens=int(payload.get("completion_tokens", 0) or 0),
            total_tokens=int(payload.get("total_tokens", 0) or 0),
            cached_prompt_tokens=max(0, int(payload.get("cached_prompt_tokens", 0) or 0)),
            billable_prompt_tokens=max(
                0,
                int(
                    payload.get(
                        "billable_prompt_tokens",
                        int(payload.get("prompt_tokens", 0) or 0)
                        - int(payload.get("cached_prompt_tokens", 0) or 0),
                    )
                    or 0
                ),
            ),
            cost_usd=_optional_float(payload.get("cost_usd")),
            estimated=bool(payload.get("estimated", False)),
            pricing_known=bool(payload.get("pricing_known", False)),
            pricing_source=str(payload.get("pricing_source", "unknown") or "unknown"),
            duration_ms=int(payload.get("duration_ms", 0) or 0),
            status=str(payload.get("status", "")),
            error=str(payload["error"]) if payload.get("error") is not None else None,
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(slots=True)
class LLMUsageTotals:
    """Aggregate usage data across many LLM calls."""

    total_calls: int = 0
    completed_calls: int = 0
    failed_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    billable_prompt_tokens: int = 0
    cost_usd: float | None = None
    estimated_calls: int = 0
    pricing_unknown_calls: int = 0
    duration_ms: int = 0
    by_operation: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "completed_calls": self.completed_calls,
            "failed_calls": self.failed_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "billable_prompt_tokens": self.billable_prompt_tokens,
            "cost_usd": self.cost_usd,
            "estimated_calls": self.estimated_calls,
            "pricing_unknown_calls": self.pricing_unknown_calls,
            "duration_ms": self.duration_ms,
            "by_operation": {key: dict(value) for key, value in self.by_operation.items()},
        }


class LLMUsageTracker:
    """Append-only usage tracker for local LiteLLM calls."""

    def __init__(self, log_path: str | Path | None = None) -> None:
        self.log_path = Path(log_path).resolve() if log_path else None
        self.records: list[LLMUsageRecord] = []
        self._lock = threading.Lock()

    def record_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response: Any,
        operation: str,
        duration_ms: int,
        status: str,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMUsageRecord:
        """Record one completion call using API usage when available and local estimates otherwise."""

        payload = _to_jsonable(response)
        usage = _extract_usage(payload)
        if usage is None:
            prompt_tokens = _count_messages(model, messages)
            completion_tokens = 0 if status == "failed" else _count_text(model, extract_response_text(payload))
            total_tokens = prompt_tokens + completion_tokens
            cached_prompt_tokens = 0
            estimated = True
        else:
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = 0 if status == "failed" else usage["completion_tokens"]
            total_tokens = usage.get("total_tokens") or prompt_tokens + completion_tokens
            cached_prompt_tokens = min(prompt_tokens, max(0, usage.get("cached_prompt_tokens", 0)))
            estimated = False
        billable_prompt_tokens = max(0, prompt_tokens - cached_prompt_tokens)
        estimated_cost_usd, pricing_known, pricing_source = _estimate_cost(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
        )
        cost_usd = estimated_cost_usd if status != "failed" else None
        record = LLMUsageRecord(
            timestamp=_now_timestamp(),
            call_id=f"call_{uuid.uuid4().hex}",
            operation=operation or "llm",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            billable_prompt_tokens=billable_prompt_tokens,
            cost_usd=cost_usd,
            estimated=estimated,
            pricing_known=pricing_known,
            pricing_source=pricing_source,
            duration_ms=max(0, int(duration_ms)),
            status=status,
            error=error,
            metadata=_jsonable_metadata(metadata or {}),
        )
        self._append(record)
        return record

    def _append(self, record: LLMUsageRecord) -> None:
        with self._lock:
            self.records.append(record)
        if self.log_path is None:
            return
        lock = _path_lock(self.log_path)
        with lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def load_usage_records(path: str | Path) -> list[LLMUsageRecord]:
    """Load usage records from an append-only JSONL file."""

    usage_path = Path(path)
    if not usage_path.exists():
        return []
    records: list[LLMUsageRecord] = []
    for line in usage_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(LLMUsageRecord.from_dict(payload))
    return records


def summarize_usage(records: list[LLMUsageRecord]) -> LLMUsageTotals:
    """Summarize usage records into run-level totals."""

    totals = LLMUsageTotals()
    known_cost = 0.0
    has_known_cost = False
    by_operation_records: dict[str, list[LLMUsageRecord]] = {}
    for record in records:
        totals.total_calls += 1
        if record.status == "failed":
            totals.failed_calls += 1
        else:
            totals.completed_calls += 1
        totals.prompt_tokens += record.prompt_tokens
        totals.completion_tokens += record.completion_tokens
        totals.total_tokens += record.total_tokens
        totals.cached_prompt_tokens += record.cached_prompt_tokens
        totals.billable_prompt_tokens += record.billable_prompt_tokens
        totals.duration_ms += record.duration_ms
        if record.estimated:
            totals.estimated_calls += 1
        if not record.pricing_known:
            totals.pricing_unknown_calls += 1
        if record.cost_usd is not None:
            known_cost += record.cost_usd
            has_known_cost = True
        by_operation_records.setdefault(record.operation, []).append(record)
    totals.cost_usd = round(known_cost, 10) if has_known_cost else None
    totals.by_operation = {
        operation: _summarize_operation(operation_records)
        for operation, operation_records in sorted(by_operation_records.items())
    }
    return totals


def _summarize_operation(records: list[LLMUsageRecord]) -> dict[str, Any]:
    known_cost = 0.0
    has_known_cost = False
    for record in records:
        if record.cost_usd is not None:
            known_cost += record.cost_usd
            has_known_cost = True
    return {
        "total_calls": len(records),
        "completed_calls": sum(1 for item in records if item.status != "failed"),
        "failed_calls": sum(1 for item in records if item.status == "failed"),
        "prompt_tokens": sum(item.prompt_tokens for item in records),
        "completion_tokens": sum(item.completion_tokens for item in records),
        "total_tokens": sum(item.total_tokens for item in records),
        "cached_prompt_tokens": sum(item.cached_prompt_tokens for item in records),
        "billable_prompt_tokens": sum(item.billable_prompt_tokens for item in records),
        "cost_usd": round(known_cost, 10) if has_known_cost else None,
        "estimated_calls": sum(1 for item in records if item.estimated),
        "pricing_unknown_calls": sum(1 for item in records if not item.pricing_known),
        "duration_ms": sum(item.duration_ms for item in records),
    }


def _extract_usage(payload: Any) -> dict[str, int] | None:
    if not isinstance(payload, dict):
        return None
    raw_usage = payload.get("usage")
    if raw_usage is None and isinstance(payload.get("response"), dict):
        raw_usage = payload["response"].get("usage")
    if hasattr(raw_usage, "model_dump"):
        raw_usage = raw_usage.model_dump()
    elif hasattr(raw_usage, "dict"):
        raw_usage = raw_usage.dict()
    if not isinstance(raw_usage, dict):
        return None
    prompt_tokens = _first_int(raw_usage, "prompt_tokens", "input_tokens", "inputTokens")
    completion_tokens = _first_int(raw_usage, "completion_tokens", "output_tokens", "outputTokens")
    if prompt_tokens is None or completion_tokens is None:
        return None
    total_tokens = _first_int(raw_usage, "total_tokens", "totalTokens")
    cached_prompt_tokens = _extract_cached_prompt_tokens(raw_usage)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens or prompt_tokens + completion_tokens,
        "cached_prompt_tokens": min(prompt_tokens, cached_prompt_tokens),
    }


def _count_messages(model: str, messages: list[dict[str, Any]]) -> int:
    try:
        import litellm

        return max(0, int(litellm.token_counter(model=model, messages=messages)))
    except Exception:  # noqa: BLE001 - usage accounting must not break LLM calls.
        return _fallback_count(_messages_to_text(messages))


def _count_text(model: str, text: str) -> int:
    if not text:
        return 0
    try:
        import litellm

        return max(0, int(litellm.token_counter(model=model, text=text)))
    except Exception:  # noqa: BLE001 - usage accounting must not break LLM calls.
        return _fallback_count(text)


def _extract_cached_prompt_tokens(raw_usage: dict[str, Any]) -> int:
    for key in (
        "cached_prompt_tokens",
        "cached_input_tokens",
        "cached_tokens",
        "cache_read_input_tokens",
    ):
        value = _first_int(raw_usage, key)
        if value is not None:
            return max(0, value)
    for detail_key in (
        "prompt_tokens_details",
        "prompt_token_details",
        "input_tokens_details",
        "input_token_details",
    ):
        details = raw_usage.get(detail_key)
        if hasattr(details, "model_dump"):
            details = details.model_dump()
        elif hasattr(details, "dict"):
            details = details.dict()
        if not isinstance(details, dict):
            continue
        value = _first_int(
            details,
            "cached_tokens",
            "cached_prompt_tokens",
            "cached_input_tokens",
            "cache_read_input_tokens",
        )
        if value is not None:
            return max(0, value)
    return 0


def _estimate_cost(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
) -> tuple[float | None, bool, str]:
    override = _pricing_override_for_model(model)
    if override is not None:
        cached_tokens = min(prompt_tokens, max(0, cached_prompt_tokens))
        billable_prompt_tokens = max(0, prompt_tokens - cached_tokens)
        cached_input_per_token = (
            override.cached_input_per_token
            if override.cached_input_per_token is not None
            else override.input_per_token
        )
        cost = (
            billable_prompt_tokens * override.input_per_token
            + cached_tokens * cached_input_per_token
            + completion_tokens * override.output_per_token
        )
        return round(cost, 10), True, override.source
    try:
        import litellm

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception:  # noqa: BLE001 - unknown pricing is expected for local/custom models.
        return None, False, "unknown"
    return round(float(prompt_cost or 0.0) + float(completion_cost or 0.0), 10), True, "litellm"


def _pricing_override_for_model(model: str) -> ModelPricing | None:
    normalized = _normalize_model_name(model)
    return _SKILLFABRIC_PRICING_OVERRIDES.get(normalized)


def _normalize_model_name(model: str) -> str:
    normalized = str(model).strip().lower()
    for prefix in ("openai/responses/", "openai/chat/", "openai/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized


def _fallback_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    word_estimate = len(stripped.split())
    char_estimate = math.ceil(len(stripped) / 4)
    return max(1, word_estimate, char_estimate)


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role is not None:
            parts.append(str(role))
        parts.append(_content_to_text(content))
    return "\n".join(part for part in parts if part)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("text") is not None:
                    parts.append(str(item["text"]))
                elif item.get("content") is not None:
                    parts.append(_content_to_text(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        if content.get("text") is not None:
            return str(content["text"])
        return json.dumps(_to_jsonable(content), ensure_ascii=False, sort_keys=True)
    return str(content)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _jsonable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    payload = _to_jsonable(metadata)
    return dict(payload) if isinstance(payload, dict) else {"value": payload}


def _first_int(payload: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _path_lock(path: Path) -> threading.Lock:
    resolved = path.resolve()
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(resolved)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[resolved] = lock
        return lock


def _now_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
