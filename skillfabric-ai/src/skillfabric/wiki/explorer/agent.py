"""Orchestrate one strict query-wiki exploration run."""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillfabric.storage import atomic_write_text
from skillfabric.wiki.explorer.backends.base import WikiExplorerBackend
from skillfabric.wiki.explorer.backends.claude_code import (
    ClaudeCodeSdkRuntime,
    ClaudeCodeWikiExplorerBackend,
)
from skillfabric.wiki.explorer.prompting import validate_required_selected_skills
from skillfabric.wiki.explorer.redaction import sanitize_error_text
from skillfabric.wiki.explorer.skill_package import SkillPackage
from skillfabric.wiki.explorer.validation import (
    SkillPackageValidationResult,
    validate_skill_package,
)

LOGGER = logging.getLogger(__name__)


class _RetryableExplorerValidationError(ValueError):
    __skillfabric_recoverable_route_failure__ = True


@dataclass(frozen=True, slots=True)
class WikiExplorerConfig:
    env_file: str | Path = ".env"
    max_selected_skills: int = 8
    model: str | None = None
    max_turns: int = 24
    load_timeout_ms: int = 30_000
    execution_timeout_seconds: float = 300.0
    max_attempts: int = 2
    retry_delay_seconds: float = 1.0
    reasoning_effort: str | None = None
    required_selected_skills: int | None = None

    def __post_init__(self) -> None:
        validate_required_selected_skills(
            self.required_selected_skills,
            max_selected_skills=self.max_selected_skills,
        )
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        if (
            isinstance(self.retry_delay_seconds, bool)
            or not isinstance(self.retry_delay_seconds, (int, float))
            or not math.isfinite(self.retry_delay_seconds)
            or self.retry_delay_seconds < 0
        ):
            raise ValueError("retry_delay_seconds must be finite and non-negative")
        if self.reasoning_effort is not None and (
            not isinstance(self.reasoning_effort, str) or not self.reasoning_effort.strip()
        ):
            raise ValueError("reasoning_effort must be a non-empty string when provided")


@dataclass(frozen=True, slots=True)
class WikiExplorerRun:
    package: SkillPackage
    validation: SkillPackageValidationResult


def explore_query_wiki(
    config: WikiExplorerConfig,
    *,
    query: str,
    query_wiki_root: Path,
    trace_dir: Path,
    sdk_runtime: ClaudeCodeSdkRuntime | None = None,
    backend: WikiExplorerBackend | None = None,
) -> WikiExplorerRun:
    """Retry exploration until the selected package passes validation."""

    if sdk_runtime is not None and backend is not None:
        raise TypeError("sdk_runtime and backend cannot be used together")
    resolved_backend = (
        backend
        if backend is not None
        else ClaudeCodeWikiExplorerBackend(
            env_file=config.env_file,
            max_selected_skills=config.max_selected_skills,
            required_selected_skills=config.required_selected_skills,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            sdk_runtime=sdk_runtime,
            max_turns=config.max_turns,
            load_timeout_ms=config.load_timeout_ms,
            execution_timeout_seconds=config.execution_timeout_seconds,
        )
    )
    if not query_wiki_root.is_dir():
        raise FileNotFoundError(f"query_wiki root does not exist: {query_wiki_root}")
    cc_dir = trace_dir / "cc_explorer"
    attempts_root = trace_dir / ".cc_explorer_attempts"
    if cc_dir.exists() or attempts_root.exists():
        raise FileExistsError("explorer trace already contains Codex attempt artifacts")
    attempts_root.mkdir(parents=True, exist_ok=True)
    attempt_records: list[dict[str, Any]] = []
    winner: int | None = None
    terminal_error: Exception | None = None
    for attempt in range(1, config.max_attempts + 1):
        attempt_dir = attempts_root / f"attempt-{attempt:02d}"
        attempt_trace = trace_dir / f".cc_explorer_run-{attempt:02d}"
        attempt_trace.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        package: SkillPackage | None = None
        validation: SkillPackageValidationResult | None = None
        try:
            package = resolved_backend.explore(
                query=query,
                query_wiki_root=query_wiki_root,
                trace_dir=attempt_trace,
            )
            validation = validate_skill_package(
                package,
                query_wiki_root,
                max_selected_skills=config.max_selected_skills,
                required_selected_skills=config.required_selected_skills,
            )
            if validation.valid:
                winner = attempt
                terminal_error = None
            else:
                terminal_error = _RetryableExplorerValidationError(
                    "; ".join(validation.errors) or "invalid SkillPackage"
                )
        except Exception as exc:  # noqa: BLE001 - one bounded route retry policy.
            terminal_error = exc
        finally:
            _archive_attempt(
                attempt_trace,
                attempt_dir,
                package=package,
                validation=validation,
                error=terminal_error,
                started=started,
                trace_root=trace_dir,
                redaction_roots=(
                    query_wiki_root.resolve(),
                    trace_dir.resolve(),
                    Path(config.env_file).expanduser().resolve(),
                ),
            )
        attempt_record = _load_attempt_record(attempt_dir, trace_root=trace_dir)
        attempt_records.append(attempt_record)
        if winner is not None:
            break
        if terminal_error is None:
            terminal_error = RuntimeError("explorer attempt ended without a result")
        if attempt == config.max_attempts or not _retryable_explorer_error(terminal_error):
            break
        LOGGER.warning(
            "explorer_retry attempt=%d/%d delay_seconds=%.3f error_type=%s",
            attempt,
            config.max_attempts,
            config.retry_delay_seconds,
            type(terminal_error).__name__,
        )
        if config.retry_delay_seconds:
            time.sleep(config.retry_delay_seconds)
    if terminal_error is None:
        terminal_error = RuntimeError("explorer retry loop ended without a terminal result")
    _publish_explorer_closure(
        trace_dir,
        attempt_records=attempt_records,
        winner=winner,
    )
    if winner is None:
        raise terminal_error
    winner_package = SkillPackage.from_dict(
        _read_json(trace_dir / "cc_explorer" / "skill_package.json", label="skill package")
    )
    winner_validation = _read_validation(trace_dir / "cc_explorer" / "validation.json")
    return WikiExplorerRun(package=winner_package, validation=winner_validation)


def _archive_attempt(
    attempt_trace: Path,
    attempt_dir: Path,
    *,
    package: SkillPackage | None,
    validation: SkillPackageValidationResult | None,
    error: Exception | None,
    started: float,
    trace_root: Path,
    redaction_roots: tuple[Path, ...],
) -> None:
    source = attempt_trace / "cc_explorer"
    attempt_dir.parent.mkdir(parents=True, exist_ok=True)
    if attempt_dir.exists():
        raise FileExistsError(f"explorer attempt already exists: {attempt_dir.name}")
    if source.exists():
        shutil.move(str(source), str(attempt_dir))
    else:
        attempt_dir.mkdir(parents=True)
    if package is not None:
        atomic_write_text(
            attempt_dir / "skill_package.json",
            json.dumps(package.to_dict(), ensure_ascii=False, indent=2) + "\n",
        )
    if validation is not None:
        atomic_write_text(
            attempt_dir / "validation.json",
            json.dumps(validation.to_dict(), ensure_ascii=False, indent=2) + "\n",
        )
    safe_error = None if error is None else _sanitized_exception(error, paths=redaction_roots)
    if error is not None and not (attempt_dir / "error.json").exists():
        atomic_write_text(
            attempt_dir / "error.json",
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "error": safe_error,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
    duration = time.monotonic() - started
    if not math.isfinite(duration) or duration < 0:
        duration = 0.0
    payload = {
        "schema_version": 1,
        "attempt": int(attempt_dir.name.removeprefix("attempt-")),
        "status": "completed" if error is None else "failed",
        "failure_kind": None if error is None else _failure_kind(error),
        "retryable": False if error is None else _retryable_explorer_error(error),
        "duration_seconds": duration,
        "operational_access": _optional_json(attempt_dir / "operational_access.json"),
        "validation": validation.to_dict() if validation is not None else None,
        "error_class": None if error is None else type(error).__name__,
        "sanitized_error": safe_error,
        "artifact_paths": sorted(
            [
                *_artifact_paths(attempt_dir, trace_root),
                str(Path("cc_explorer") / "attempts" / attempt_dir.name / "attempt.json"),
            ]
        ),
    }
    atomic_write_text(
        attempt_dir / "attempt.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    shutil.rmtree(attempt_trace, ignore_errors=True)


def _publish_explorer_closure(
    trace_dir: Path,
    *,
    attempt_records: list[dict[str, Any]],
    winner: int | None,
) -> None:
    cc_dir = trace_dir / "cc_explorer"
    attempts_root = trace_dir / ".cc_explorer_attempts"
    if cc_dir.exists():
        shutil.rmtree(cc_dir)
    cc_dir.mkdir(parents=True, exist_ok=True)
    published_attempts = cc_dir / "attempts"
    if attempts_root.exists():
        shutil.move(str(attempts_root), str(published_attempts))
    winner_dir = published_attempts / f"attempt-{winner:02d}" if winner else None
    if winner_dir is None and attempt_records:
        last_attempt = attempt_records[-1].get("attempt")
        if isinstance(last_attempt, int):
            failure_dir = published_attempts / f"attempt-{last_attempt:02d}"
            error = failure_dir / "error.json"
            if error.exists():
                shutil.copy2(error, cc_dir / "error.json")
            validation = failure_dir / "validation.json"
            if validation.exists():
                shutil.copy2(validation, cc_dir / "validation.json")
    if winner_dir is not None and winner_dir.is_dir():
        for child in winner_dir.iterdir():
            if child.name in {"attempt.json", "validation.json", "error.json"}:
                continue
            destination = cc_dir / child.name
            if child.is_file():
                shutil.copy2(child, destination)
            elif child.is_dir():
                shutil.copytree(child, destination)
        validation = winner_dir / "validation.json"
        if validation.exists():
            shutil.copy2(validation, cc_dir / "validation.json")
    closure = {
        "schema_version": 1,
        "status": "completed" if winner is not None else "route_failed",
        "outcome": _closure_outcome(winner_dir) if winner_dir is not None else "route_failed",
        "winning_attempt": winner,
        "attempts": attempt_records,
    }
    atomic_write_text(
        cc_dir / "closure.json",
        json.dumps(closure, ensure_ascii=False, indent=2) + "\n",
    )


def _closure_outcome(winner_dir: Path) -> str:
    package = _read_json(winner_dir / "skill_package.json", label="winning skill package")
    selected = package.get("selected_skills")
    if not isinstance(selected, list):
        raise ValueError("winning skill package selected_skills must be a list")
    if selected:
        return "completed_nonempty"
    backend = _optional_json(winner_dir / "backend.json")
    if backend is not None and backend.get("backend") == "codex":
        access = _optional_json(winner_dir / "operational_access.json")
        if access is None or access.get("semantic_empty_valid") is not True:
            raise ValueError("Codex empty closure requires validated semantic-empty evidence")
    return "completed_empty"


def _load_attempt_record(attempt_dir: Path, *, trace_root: Path) -> dict[str, Any]:
    payload = _read_json(attempt_dir / "attempt.json", label="explorer attempt")
    payload["artifact_paths"] = _artifact_paths(attempt_dir, trace_root)
    return payload


def _artifact_paths(root: Path, trace_root: Path) -> list[str]:
    del trace_root
    published_root = Path("cc_explorer") / "attempts" / root.name
    return sorted(
        str(published_root / path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    )


def _optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = _read_json(path, label=path.name)
    return value if isinstance(value, dict) else None


def _read_validation(path: Path) -> SkillPackageValidationResult:
    payload = _read_json(path, label="explorer validation")
    if set(payload) != {"valid", "errors"}:
        raise ValueError("explorer validation must use canonical fields")
    errors = payload["errors"]
    if not isinstance(payload["valid"], bool) or not isinstance(errors, list):
        raise ValueError("explorer validation has invalid fields")
    return SkillPackageValidationResult(
        valid=payload["valid"], errors=tuple(str(item) for item in errors)
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _retryable_explorer_error(error: Exception) -> bool:
    if getattr(error, "__skillfabric_non_retryable__", False):
        return False
    if getattr(error, "__skillfabric_recoverable_route_failure__", False):
        return True
    for attribute in ("retryable", "is_retryable"):
        marker = getattr(error, attribute, None)
        if isinstance(marker, bool):
            return marker
    try:
        from openai_codex import is_retryable_error

        if is_retryable_error(error):
            return True
    except (ImportError, TypeError):
        pass
    if isinstance(error, TimeoutError):
        return True
    message = str(error).casefold()
    if any(
        marker in message
        for marker in (
            "authentication",
            "unauthorized",
            "invalid api key",
            "login failed",
            "permission denied",
            "policy violation",
            "outside_query_wiki",
            "write_attempt",
            "runtime mismatch",
            "config mismatch",
        )
    ):
        return False
    retry_markers = (
        "timeout",
        "transient",
        "temporarily",
        "service unavailable",
        "server busy",
        "transport",
        "connection",
        "503",
        "429",
        "valid skillpackage json",
        "structured skillpackage",
    )
    return any(marker in message for marker in retry_markers)


def _failure_kind(error: Exception) -> str:
    if isinstance(error, _RetryableExplorerValidationError):
        return "validation"
    if type(error).__name__ == "CodexOperationalAccessError":
        return "operational_access"
    if isinstance(error, TimeoutError):
        return "timeout"
    if getattr(error, "__skillfabric_non_retryable__", False):
        return "policy_violation"
    return "retryable_runtime" if _retryable_explorer_error(error) else "non_retryable_runtime"


def _safe_error(value: str, *, paths: tuple[Path, ...] = ()) -> str:
    return (
        sanitize_error_text(
            value,
            paths=paths,
            path_replacement="[redacted-path]",
            collapse_whitespace=True,
        )
        or "explorer failed"
    )


def _sanitized_exception(error: Exception, *, paths: tuple[Path, ...]) -> str:
    sanitized = getattr(error, "__skillfabric_sanitized_error__", None)
    value = sanitized if isinstance(sanitized, str) and sanitized.strip() else str(error)
    return _safe_error(value, paths=paths)


__all__ = ["WikiExplorerConfig", "WikiExplorerRun", "explore_query_wiki"]
