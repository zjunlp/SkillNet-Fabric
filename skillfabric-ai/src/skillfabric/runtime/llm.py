"""LiteLLM configuration and call wrappers."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from skillfabric.runtime.usage import LLMUsageTracker

# Keep default offline test/build paths from making LiteLLM fetch pricing metadata.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

DEFAULT_MODEL = "openai/responses/gpt-5.4-mini"
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MAX_TOKENS = 32768
DEFAULT_TIMEOUT = 120


@dataclass(frozen=True, slots=True)
class LLMUsageContext:
    """Thread-local usage overrides for nested SkillFabric runs."""

    enabled: bool | None = None
    log_path: Path | None = None
    operation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_USAGE_CONTEXT: ContextVar[LLMUsageContext | None] = ContextVar(
    "skillfabric_llm_usage_context",
    default=None,
)


def _current_usage_context() -> LLMUsageContext:
    return _USAGE_CONTEXT.get() or LLMUsageContext()


@dataclass(slots=True)
class LLMConfig:
    """LLM configuration for SkillFabric workflows."""

    api_base: str
    api_key: str
    model: str = DEFAULT_MODEL
    credential_source: str = "api_key"
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: float = DEFAULT_TIMEOUT
    usage_enabled: bool = True
    usage_log_path: Path | None = None
    usage_operation: str = "llm"
    usage_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, *, env_path: str | Path | None = None) -> LLMConfig:
        """Read LiteLLM configuration from an env file, with shell fallback."""

        values = read_env_file(env_path)
        api_base, api_base_key = _first_config_entry(
            values,
            ("SKILLFABRIC_LLM_API_BASE", "BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"),
            ("ANTHROPIC_BASE_URL",),
        )
        api_key, credential_key = _first_config_entry(
            values,
            ("SKILLFABRIC_LLM_API_KEY", "API_KEY", "OPENAI_API_KEY"),
            ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        )
        model = _normalize_litellm_model(
            _first_config_value(values, ("SKILLFABRIC_LLM_MODEL", "MODEL"), ("ANTHROPIC_MODEL",), default=DEFAULT_MODEL),
            values=values,
            credential_key=credential_key,
            api_base_key=api_base_key,
        )
        api_base = _normalize_api_base(api_base or DEFAULT_API_BASE, model=model)
        reasoning_effort = _first_value(
            values,
            "SKILLFABRIC_LLM_REASONING_EFFORT",
            default=DEFAULT_REASONING_EFFORT,
        )
        max_tokens = int(_first_value(values, "SKILLFABRIC_LLM_MAX_TOKENS", "MAX_TOKENS", default=str(DEFAULT_MAX_TOKENS)))
        timeout = float(_first_value(values, "SKILLFABRIC_LLM_TIMEOUT", "TIMEOUT", default=str(DEFAULT_TIMEOUT)))
        usage_enabled = _parse_bool(_first_value(values, "SKILLFABRIC_USAGE_ENABLED", "USAGE_ENABLED", default="1"), default=True)
        usage_log_value = _first_value(values, "SKILLFABRIC_USAGE_LOG_PATH", "USAGE_LOG_PATH")
        usage_log_path = _resolve_optional_path(usage_log_value, env_path=env_path)
        usage_operation = _first_value(values, "SKILLFABRIC_USAGE_OPERATION", "USAGE_OPERATION", default="llm")
        usage_metadata = _parse_json_object(_first_value(values, "SKILLFABRIC_USAGE_METADATA", "USAGE_METADATA", default=""))
        usage_context = _current_usage_context()
        usage_enabled = usage_enabled if usage_context.enabled is None else usage_context.enabled
        usage_log_path = usage_context.log_path or usage_log_path
        usage_operation = usage_context.operation or usage_operation
        usage_metadata = {**usage_metadata, **usage_context.metadata}
        if not api_key:
            raise ValueError(
                "missing API key. Set API_KEY, OPENAI_API_KEY, or ANTHROPIC_AUTH_TOKEN. "
                "Run `skillfabric help config` for details."
            )
        return cls(
            api_base=api_base,
            api_key=api_key,
            model=model,
            credential_source=_credential_source(credential_key),
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            timeout=timeout,
            usage_enabled=usage_enabled,
            usage_log_path=usage_log_path,
            usage_operation=usage_operation,
            usage_metadata=usage_metadata,
        )


@contextmanager
def llm_usage_context(
    *,
    enabled: bool | None = None,
    log_path: str | Path | None = None,
    operation: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Temporarily override usage tracking for the current thread/context."""

    parent = _current_usage_context()
    child = LLMUsageContext(
        enabled=parent.enabled if enabled is None else enabled,
        log_path=parent.log_path if log_path is None else Path(log_path).expanduser().resolve(),
        operation=parent.operation if operation is None else operation,
        metadata={**parent.metadata, **(metadata or {})},
    )
    token = _USAGE_CONTEXT.set(child)
    try:
        yield
    finally:
        _USAGE_CONTEXT.reset(token)


def load_llm_env(*, env_path: str | Path = ".env", override: bool = False) -> dict[str, str]:
    """Load LLM env file values into the current process environment."""

    values = read_env_file(env_path)
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values


def read_env_file(env_path: str | Path | None = ".env") -> dict[str, str]:
    """Read simple KEY=VALUE env files without mutating the process environment."""

    return _read_env_file(Path(env_path) if env_path is not None else Path(".env"))


def litellm_completion(
    *,
    messages: list[dict[str, Any]],
    config: LLMConfig | None = None,
    env_path: str | Path | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    usage_operation: str | None = None,
    usage_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Call litellm.completion with project env configuration."""

    resolved = config or LLMConfig.from_env(env_path=env_path)
    resolved_model = model or resolved.model
    provider_messages = _provider_messages(resolved_model, messages)
    _ensure_provider_env(resolved, resolved_model)
    call_kwargs = {
        **kwargs,
        "model": resolved_model,
        "messages": provider_messages,
        "max_tokens": resolved.max_tokens if max_tokens is None else max_tokens,
        "api_base": resolved.api_base,
        "api_key": _provider_api_key(resolved, resolved_model),
        "timeout": resolved.timeout,
        "request_timeout": resolved.timeout,
        "force_timeout": resolved.timeout,
    }
    resolved_reasoning_effort = resolved.reasoning_effort if reasoning_effort is None else reasoning_effort
    if resolved_reasoning_effort:
        call_kwargs["reasoning_effort"] = resolved_reasoning_effort
    usage_context = _current_usage_context()
    operation = usage_operation or usage_context.operation or resolved.usage_operation
    metadata = {**resolved.usage_metadata, **usage_context.metadata, **(usage_metadata or {})}
    usage_enabled = resolved.usage_enabled if usage_context.enabled is None else usage_context.enabled
    usage_log_path = usage_context.log_path or resolved.usage_log_path
    started = time.monotonic()
    try:
        if _should_use_process_timeout(resolved.timeout):
            response = _completion_with_process_timeout(call_kwargs, resolved.timeout)
        else:
            response = _direct_litellm_completion(call_kwargs, resolved.timeout)
    except BaseException as exc:
        _record_usage(
            resolved,
            model=resolved_model,
            messages=provider_messages,
            response=None,
            operation=operation,
            started=started,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            metadata=metadata,
            usage_enabled=usage_enabled,
            usage_log_path=usage_log_path,
        )
        raise
    _record_usage(
        resolved,
        model=resolved_model,
        messages=provider_messages,
        response=response,
        operation=operation,
        started=started,
        status="completed",
        error=None,
        metadata=metadata,
        usage_enabled=usage_enabled,
        usage_log_path=usage_log_path,
    )
    return response


def _direct_litellm_completion(call_kwargs: dict[str, Any], timeout: float) -> Any:
    import litellm

    litellm.suppress_debug_info = True
    litellm.request_timeout = timeout
    return litellm.completion(**call_kwargs)


def _should_use_process_timeout(timeout: float) -> bool:
    if timeout <= 0:
        return False
    if os.environ.get("SKILLFABRIC_LLM_PROCESS_TIMEOUT", "1").lower() in {"0", "false", "no"}:
        return False
    if threading.current_thread() is threading.main_thread():
        return False
    return not _using_injected_litellm_module()


def _using_injected_litellm_module() -> bool:
    module = sys.modules.get("litellm")
    return module is not None and not hasattr(module, "__file__")


def _completion_with_process_timeout(call_kwargs: dict[str, Any], timeout: float) -> Any:
    ctx = mp.get_context("spawn")
    output: mp.Queue[dict[str, Any]] = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_completion_process_worker, args=(call_kwargs, timeout, output))
    process.start()
    grace_seconds = min(5.0, max(0.5, timeout * 0.1))
    try:
        result = output.get(timeout=timeout + grace_seconds)
    except queue.Empty as exc:
        process.terminate()
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)
        raise TimeoutError(f"LLM call exceeded process timeout of {timeout} seconds") from exc
    finally:
        if process.is_alive():
            process.join(timeout=1.0)
    if not result.get("ok"):
        raise RuntimeError(str(result.get("error") or "LLM subprocess failed"))
    return result.get("response")


def _completion_process_worker(
    call_kwargs: dict[str, Any],
    timeout: float,
    output: mp.Queue[dict[str, Any]],
) -> None:
    try:
        sleep_seconds = float(call_kwargs.pop("_skillfabric_test_sleep_seconds", 0.0) or 0.0)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        response = _direct_litellm_completion(call_kwargs, timeout)
        output.put({"ok": True, "response": response_to_jsonable(response)})
    except BaseException as exc:  # noqa: BLE001 - subprocess boundary normalizes failures.
        output.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def response_to_jsonable(response: Any) -> Any:
    """Convert a LiteLLM response object into a JSON-serializable payload."""

    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    try:
        json.dumps(response)
    except TypeError:
        return str(response)
    return response


def _record_usage(
    config: LLMConfig,
    *,
    model: str,
    messages: list[dict[str, Any]],
    response: Any,
    operation: str,
    started: float,
    status: str,
    error: str | None,
    metadata: dict[str, Any] | None,
    usage_enabled: bool | None = None,
    usage_log_path: Path | None = None,
) -> None:
    enabled = config.usage_enabled if usage_enabled is None else usage_enabled
    log_path = usage_log_path or config.usage_log_path
    if not enabled or log_path is None:
        return
    try:
        tracker = LLMUsageTracker(log_path=log_path)
        tracker.record_completion(
            model=model,
            messages=messages,
            response=response,
            operation=operation,
            duration_ms=int((time.monotonic() - started) * 1000),
            status=status,
            error=error,
            metadata=metadata,
        )
    except Exception:
        return


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def _first_value(values: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        if values.get(key):
            return values[key]
    for key in keys:
        if key in os.environ and os.environ[key]:
            return os.environ[key]
    return default


def _first_config_value(
    values: dict[str, str],
    primary_keys: tuple[str, ...],
    fallback_keys: tuple[str, ...],
    *,
    default: str = "",
) -> str:
    for key in primary_keys:
        if values.get(key):
            return values[key]
    for key in primary_keys:
        if key in os.environ and os.environ[key]:
            return os.environ[key]
    for key in fallback_keys:
        if values.get(key):
            return values[key]
    for key in fallback_keys:
        if key in os.environ and os.environ[key]:
            return os.environ[key]
    return default


def _first_config_entry(
    values: dict[str, str],
    primary_keys: tuple[str, ...],
    fallback_keys: tuple[str, ...],
) -> tuple[str, str]:
    for key in primary_keys:
        if values.get(key):
            return values[key], key
    for key in primary_keys:
        if key in os.environ and os.environ[key]:
            return os.environ[key], key
    for key in fallback_keys:
        if values.get(key):
            return values[key], key
    for key in fallback_keys:
        if key in os.environ and os.environ[key]:
            return os.environ[key], key
    return "", ""


def _credential_source(key: str) -> str:
    if key == "ANTHROPIC_AUTH_TOKEN":
        return "anthropic_auth_token"
    if key == "ANTHROPIC_API_KEY":
        return "anthropic_api_key"
    return "api_key"


def _normalize_litellm_model(
    model: str,
    *,
    values: dict[str, str],
    credential_key: str,
    api_base_key: str,
) -> str:
    """Normalize env model names for LiteLLM without changing explicit provider ids."""

    normalized = model.strip()
    if "/" in normalized:
        return normalized
    anthropic_model = _first_value(values, "ANTHROPIC_MODEL")
    anthropic_base = _first_value(values, "ANTHROPIC_BASE_URL")
    primary_config = credential_key in {"SKILLFABRIC_LLM_API_KEY", "API_KEY", "OPENAI_API_KEY"} or api_base_key in {
        "SKILLFABRIC_LLM_API_BASE",
        "BASE_URL",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    }
    if primary_config:
        return f"openai/{normalized}"
    if anthropic_base and credential_key.startswith("ANTHROPIC_") and _looks_like_anthropic_model(normalized):
        return f"anthropic/{normalized}"
    openai_model = _first_value(values, "MODEL")
    if anthropic_model and not openai_model and anthropic_base:
        if _looks_like_anthropic_model(normalized):
            return f"anthropic/{normalized}"
        return f"openai/{normalized}"
    if not _looks_like_anthropic_model(normalized):
        return f"openai/{normalized}"
    return f"anthropic/{normalized}"


def _looks_like_anthropic_model(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized.startswith(("claude-", "anthropic."))


def _normalize_api_base(api_base: str, *, model: str) -> str:
    """Normalize provider base URLs without exposing provider-specific details upstream."""

    normalized = str(api_base or DEFAULT_API_BASE).strip().rstrip("/")
    if model.startswith("openai/"):
        return _ensure_openai_v1_base(normalized)
    if model.startswith("anthropic/") and normalized.endswith("/v1"):
        return normalized[:-3]
    return normalized


def _ensure_openai_v1_base(api_base: str) -> str:
    parsed = urlsplit(api_base)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        return urlunsplit(parsed._replace(path=path))
    for endpoint in ("/v1/chat/completions", "/v1/responses", "/v1/embeddings"):
        if path.endswith(endpoint):
            return urlunsplit(parsed._replace(path=path[: -len(endpoint)] + "/v1"))
    if "/v1/" in f"{path}/":
        return urlunsplit(parsed._replace(path=path))
    if not path or path == "/":
        return urlunsplit(parsed._replace(path="/v1"))
    return urlunsplit(parsed._replace(path=path))


def _provider_api_key(config: LLMConfig, model: str) -> str | None:
    """Return the api_key argument LiteLLM should receive for this provider."""

    if model.startswith("anthropic/") and config.credential_source == "anthropic_auth_token":
        return None
    return config.api_key


def _ensure_provider_env(config: LLMConfig, model: str) -> None:
    """Expose env-only credentials required by provider adapters."""

    if model.startswith("anthropic/") and config.credential_source == "anthropic_auth_token":
        if not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            os.environ["ANTHROPIC_AUTH_TOKEN"] = config.api_key


def _provider_messages(model: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize messages for providers that do not accept explicit system messages."""

    if not model.startswith("anthropic/"):
        return messages
    system_parts: list[str] = []
    non_system_messages: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "system":
            content = str(message.get("content", "")).strip()
            if content:
                system_parts.append(content)
            continue
        non_system_messages.append(dict(message))
    if not system_parts:
        return messages
    system_text = "\n\n".join(system_parts)
    if non_system_messages and non_system_messages[0].get("role") == "user":
        first = dict(non_system_messages[0])
        first_content = str(first.get("content", ""))
        first["content"] = f"{system_text}\n\n{first_content}" if first_content else system_text
        return [first, *non_system_messages[1:]]
    return [{"role": "user", "content": system_text}, *non_system_messages]


def _parse_bool(value: str, *, default: bool) -> bool:
    if not value:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_json_object(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _resolve_optional_path(value: str, *, env_path: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if env_path is not None:
        env_file = Path(env_path)
        base = env_file.parent if env_file.name else env_file
        return (base / path).resolve()
    return path.resolve()
