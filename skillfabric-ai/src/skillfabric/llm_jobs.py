"""Concurrent LLM job execution helpers."""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass(slots=True)
class LLMJobOptions:
    """Runtime controls for batches of LLM-backed jobs."""

    concurrency: int = 4
    rate_limit_per_minute: float = 0.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    progress_every: int = 10
    batch_size: int | None = None

    @classmethod
    def from_env(
        cls,
        *,
        env_path: str | Path | None = None,
        concurrency: int | None = None,
        rate_limit_per_minute: float | None = None,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
        progress_every: int | None = None,
        batch_size: int | None = None,
    ) -> LLMJobOptions:
        values = _read_env_file(Path(env_path) if env_path is not None else None)
        return cls(
            concurrency=_int_value(values, "SKILLFABRIC_LLM_CONCURRENCY", concurrency, 4),
            rate_limit_per_minute=_float_value(
                values,
                "SKILLFABRIC_LLM_RATE_LIMIT_PER_MINUTE",
                rate_limit_per_minute,
                0.0,
            ),
            max_retries=_int_value(values, "SKILLFABRIC_LLM_MAX_RETRIES", max_retries, 2),
            retry_backoff_seconds=_float_value(
                values,
                "SKILLFABRIC_LLM_RETRY_BACKOFF_SECONDS",
                retry_backoff_seconds,
                1.0,
            ),
            progress_every=_int_value(values, "SKILLFABRIC_LLM_PROGRESS_EVERY", progress_every, 10),
            batch_size=_int_value(values, "SKILLFABRIC_LLM_BATCH_SIZE", batch_size, 0),
        )

    def normalized(self) -> LLMJobOptions:
        concurrency = max(1, int(self.concurrency))
        batch_size = int(self.batch_size or 0)
        return LLMJobOptions(
            concurrency=concurrency,
            rate_limit_per_minute=max(0.0, float(self.rate_limit_per_minute)),
            max_retries=max(0, int(self.max_retries)),
            retry_backoff_seconds=max(0.0, float(self.retry_backoff_seconds)),
            progress_every=max(0, int(self.progress_every)),
            batch_size=max(concurrency, batch_size) if batch_size > 0 else max(concurrency * 4, concurrency),
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
    retry_on_result: Callable[[R], bool] | None = None,
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
    completed = 0

    def run_one(index: int, item: T) -> LLMJobOutcome[R]:
        attempts = 0
        while True:
            attempts += 1
            try:
                limiter.wait()
                value = worker(item)
                if retry_on_result is not None and retry_on_result(value):
                    raise _RetryableResult(value)
                return LLMJobOutcome(index=index, item=item, ok=True, value=value, attempts=attempts)
            except _RetryableResult as exc:
                if attempts > job_options.max_retries:
                    return LLMJobOutcome(
                        index=index,
                        item=item,
                        ok=True,
                        value=exc.value,
                        attempts=attempts,
                    )
                _sleep_before_retry(job_options.retry_backoff_seconds, attempts)
            except TimeoutError as exc:
                return LLMJobOutcome(index=index, item=item, ok=False, error=exc, attempts=attempts)
            except Exception as exc:  # noqa: BLE001 - caller normalizes final failures.
                if attempts > job_options.max_retries:
                    return LLMJobOutcome(index=index, item=item, ok=False, error=exc, attempts=attempts)
                _sleep_before_retry(job_options.retry_backoff_seconds, attempts)

    with ThreadPoolExecutor(max_workers=job_options.concurrency) as executor:
        next_index = 0
        futures: dict[Future[LLMJobOutcome[R]], int] = {}
        max_queued = min(job_options.batch_size or total, total)

        def submit_next() -> None:
            nonlocal next_index
            if next_index >= total:
                return
            index = next_index
            next_index += 1
            context = copy_context()
            futures[executor.submit(context.run, run_one, index, job_items[index])] = index

        for _ in range(max_queued):
            submit_next()

        while futures:
            done, _pending = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future, None)
                outcome = future.result()
                outcomes[outcome.index] = outcome
                if outcome.ok and on_success is not None:
                    on_success(outcome)
                completed += 1
                _log_progress(label, completed, total, job_options.progress_every)
                submit_next()

    return [outcome for outcome in outcomes if outcome is not None]


class _RetryableResult(Exception):
    def __init__(self, value: Any) -> None:
        super().__init__("retryable LLM result")
        self.value = value


def _sleep_before_retry(base_delay: float, attempts: int) -> None:
    if base_delay <= 0:
        return
    time.sleep(base_delay * max(1, attempts))


def _log_progress(label: str, completed: int, total: int, progress_every: int) -> None:
    if progress_every <= 0:
        return
    if completed != total and completed % progress_every != 0:
        return
    sys.stderr.write(f"[{label}] completed {completed}/{total}\n")
    sys.stderr.flush()


def _read_env_file(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _int_value(values: dict[str, str], key: str, override: int | None, default: int) -> int:
    if override is not None:
        return int(override)
    if os.environ.get(key):
        return int(os.environ[key])
    if values.get(key):
        return int(values[key])
    return default


def _float_value(values: dict[str, str], key: str, override: float | None, default: float) -> float:
    if override is not None:
        return float(override)
    if os.environ.get(key):
        return float(os.environ[key])
    if values.get(key):
        return float(values[key])
    return default
