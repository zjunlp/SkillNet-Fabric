"""Concurrent LLM job execution helpers."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from skillfabric.runtime.llm import LLMRequestError

T = TypeVar("T")
R = TypeVar("R")
_PERMANENT_HTTP_STATUS_CODES = frozenset({401, 403, 404})


@dataclass(slots=True)
class LLMJobOptions:
    """Runtime controls for batches of LLM-backed jobs."""

    concurrency: int = 4
    rate_limit_per_minute: float = 0.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    batch_size: int | None = None
    checkpoint_interval: int = 100
    circuit_breaker_threshold: int = 10

    def normalized(self) -> LLMJobOptions:
        concurrency = _require_int(self.concurrency, name="concurrency", minimum=1)
        rate_limit = _require_float(
            self.rate_limit_per_minute,
            name="rate_limit_per_minute",
            minimum=0.0,
        )
        max_retries = _require_int(self.max_retries, name="max_retries", minimum=0)
        retry_backoff = _require_float(
            self.retry_backoff_seconds,
            name="retry_backoff_seconds",
            minimum=0.0,
        )
        batch_size = (
            concurrency * 4
            if self.batch_size is None
            else _require_int(self.batch_size, name="batch_size", minimum=1)
        )
        if batch_size < concurrency:
            raise ValueError("batch_size must be at least concurrency")
        checkpoint_interval = _require_int(
            self.checkpoint_interval,
            name="checkpoint_interval",
            minimum=1,
        )
        circuit_breaker_threshold = _require_int(
            self.circuit_breaker_threshold,
            name="circuit_breaker_threshold",
            minimum=1,
        )
        return LLMJobOptions(
            concurrency=concurrency,
            rate_limit_per_minute=rate_limit,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff,
            batch_size=batch_size,
            checkpoint_interval=checkpoint_interval,
            circuit_breaker_threshold=circuit_breaker_threshold,
        )


@dataclass(slots=True)
class LLMJobOutcome(Generic[R]):
    """Result for one LLM job after retries."""

    index: int
    item: Any
    ok: bool
    value: R | None = None
    error: BaseException | None = None
    attempts: int = 0


class LLMJobBatchAbortedError(RuntimeError):
    """Raised after a provider-wide failure stops an LLM job batch."""

    def __init__(
        self,
        message: str,
        *,
        label: str,
        reason: str,
        error: BaseException,
        consecutive_failures: int,
    ) -> None:
        super().__init__(message)
        self.label = label
        self.reason = reason
        self.error = error
        self.consecutive_failures = consecutive_failures


class _RateLimiter:
    def __init__(self, per_minute: float) -> None:
        self.interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            sleep_for = self.next_allowed - now
            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.monotonic()
            self.next_allowed = now + self.interval


def run_llm_jobs(
    items: Iterable[T],
    worker: Callable[[T], R],
    *,
    options: LLMJobOptions | None = None,
    label: str = "llm",
    on_success: Callable[[LLMJobOutcome[R]], None] | None = None,
) -> list[LLMJobOutcome[R]]:
    """Run LLM jobs with concurrency, rate limiting, retries, and progress logging."""

    job_options = (options or LLMJobOptions()).normalized()
    job_items = list(items)
    total = len(job_items)
    if not job_items:
        return []
    limiter = _RateLimiter(job_options.rate_limit_per_minute)
    outcomes: list[LLMJobOutcome[R] | None] = [None] * total
    terminal = 0
    succeeded = 0
    failed = 0

    def run_one(index: int, item: T) -> LLMJobOutcome[R]:
        attempts = 0
        while True:
            attempts += 1
            try:
                limiter.wait()
                value = worker(item)
                return LLMJobOutcome(
                    index=index, item=item, ok=True, value=value, attempts=attempts
                )
            except Exception as exc:  # noqa: BLE001 - caller normalizes final failures.
                should_retry = attempts <= job_options.max_retries and (
                    not isinstance(exc, LLMRequestError) or exc.retryable
                )
                if not should_retry:
                    return LLMJobOutcome(
                        index=index, item=item, ok=False, error=exc, attempts=attempts
                    )
                _sleep_before_retry(job_options.retry_backoff_seconds, attempts)

    aborted: LLMJobBatchAbortedError | None = None
    consecutive_error_key: tuple[str, int | str] | None = None
    consecutive_failures = 0
    with ThreadPoolExecutor(max_workers=job_options.concurrency) as executor:
        next_index = 0
        futures: dict[Future[LLMJobOutcome[R]], int] = {}
        completed_futures: queue.SimpleQueue[Future[LLMJobOutcome[R]]] = queue.SimpleQueue()
        max_queued = min(job_options.batch_size or total, total)

        def submit_next() -> None:
            nonlocal next_index
            if next_index >= total:
                return
            index = next_index
            next_index += 1
            future = executor.submit(run_one, index, job_items[index])
            futures[future] = index
            future.add_done_callback(completed_futures.put)

        for _ in range(max_queued):
            submit_next()

        while futures:
            future = completed_futures.get()
            futures.pop(future, None)
            if future.cancelled():
                continue
            outcome = future.result()
            outcomes[outcome.index] = outcome
            if outcome.ok and on_success is not None:
                on_success(outcome)
            if outcome.ok:
                consecutive_error_key = None
                consecutive_failures = 0
            elif aborted is None:
                error = outcome.error or RuntimeError("unknown LLM job failure")
                status_code = getattr(error, "status_code", None)
                if status_code in _PERMANENT_HTTP_STATUS_CODES:
                    aborted = LLMJobBatchAbortedError(
                        f"{label} LLM job batch aborted after permanent provider "
                        f"error {status_code} ({_error_type(error)})",
                        label=label,
                        reason="permanent_provider_error",
                        error=error,
                        consecutive_failures=1,
                    )
                else:
                    error_key = _transient_error_key(error)
                    if error_key is None:
                        consecutive_error_key = None
                        consecutive_failures = 0
                    elif error_key == consecutive_error_key:
                        consecutive_failures += 1
                    else:
                        consecutive_error_key = error_key
                        consecutive_failures = 1
                    if consecutive_failures >= job_options.circuit_breaker_threshold:
                        aborted = LLMJobBatchAbortedError(
                            f"{label} LLM job circuit breaker opened after "
                            f"{consecutive_failures} consecutive {_error_type(error)} "
                            f"errors{_status_suffix(status_code)}",
                            label=label,
                            reason="circuit_breaker_open",
                            error=error,
                            consecutive_failures=consecutive_failures,
                        )
            terminal += 1
            if outcome.ok:
                succeeded += 1
            else:
                failed += 1
            if aborted is not None:
                for pending in futures:
                    pending.cancel()
            else:
                submit_next()

    if aborted is not None:
        raise aborted from aborted.error
    return [outcome for outcome in outcomes if outcome is not None]


def _transient_error_key(error: BaseException) -> tuple[str, int | str] | None:
    if isinstance(error, LLMRequestError) and error.retryable:
        if error.status_code is not None:
            return ("status_code", error.status_code)
        return ("error_type", error.error_type)
    if isinstance(error, TimeoutError):
        return ("error_type", type(error).__name__)
    return None


def _error_type(error: BaseException) -> str:
    if isinstance(error, LLMRequestError):
        return error.error_type
    return type(error).__name__


def _status_suffix(status_code: object) -> str:
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        return f" (HTTP {status_code})"
    return ""


def _sleep_before_retry(base_delay: float, attempts: int) -> None:
    if base_delay <= 0:
        return
    time.sleep(base_delay * max(1, attempts))


def _require_int(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _require_float(value: object, *, name: str, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number greater than or equal to {minimum}")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < minimum:
        raise ValueError(f"{name} must be a finite number greater than or equal to {minimum}")
    return resolved
