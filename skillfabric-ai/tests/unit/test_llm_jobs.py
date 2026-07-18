from __future__ import annotations

import json
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from skillfabric.runtime.jobs import LLMJobOptions, run_llm_jobs
from skillfabric.runtime.llm import (
    LLMConfig,
    LLMRequestError,
    litellm_completion,
    llm_usage_context,
)
from skillfabric.runtime.usage import load_usage_records


class LLMJobRunnerTests(unittest.TestCase):
    @staticmethod
    def _request_error(
        *,
        status_code: int | None = None,
        error_type: str = "ProviderError",
        retryable: bool = False,
    ) -> LLMRequestError:
        return LLMRequestError(
            f"{error_type}: provider request failed",
            status_code=status_code,
            error_type=error_type,
            retryable=retryable,
        )

    def test_job_options_reject_invalid_runtime_limits(self) -> None:
        invalid_options = [
            LLMJobOptions(concurrency=0),
            LLMJobOptions(concurrency=True),
            LLMJobOptions(rate_limit_per_minute=-1),
            LLMJobOptions(rate_limit_per_minute=float("nan")),
            LLMJobOptions(max_retries=-1),
            LLMJobOptions(retry_backoff_seconds=-1),
            LLMJobOptions(progress_every=-1),
            LLMJobOptions(batch_size=0),
            LLMJobOptions(checkpoint_interval=0),
            LLMJobOptions(circuit_breaker_threshold=0),
        ]

        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                options.normalized()

    def test_job_options_reject_batch_smaller_than_concurrency(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            LLMJobOptions(concurrency=3, batch_size=2).normalized()

    def test_job_options_default_to_formal_checkpoint_and_breaker_intervals(self) -> None:
        options = LLMJobOptions().normalized()

        self.assertEqual(options.checkpoint_interval, 100)
        self.assertEqual(options.circuit_breaker_threshold, 10)

    def test_job_options_load_checkpoint_and_breaker_intervals_from_env_file(self) -> None:
        with TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "SKILLFABRIC_LLM_CHECKPOINT_INTERVAL=25\n"
                "SKILLFABRIC_LLM_CIRCUIT_BREAKER_THRESHOLD=7\n",
                encoding="utf-8",
            )

            options = LLMJobOptions.from_env(env_path=env_path)

        self.assertEqual(options.checkpoint_interval, 25)
        self.assertEqual(options.circuit_breaker_threshold, 7)

    def test_retries_failed_jobs_and_preserves_input_order(self) -> None:
        attempts: dict[str, int] = {}

        def worker(item: str) -> str:
            attempts[item] = attempts.get(item, 0) + 1
            if item == "alpha" and attempts[item] == 1:
                raise RuntimeError("transient")
            return item.upper()

        outcomes = run_llm_jobs(
            ["alpha", "beta"],
            worker,
            options=LLMJobOptions(concurrency=2, max_retries=1, progress_every=0),
            label="test",
        )

        self.assertEqual([outcome.value for outcome in outcomes], ["ALPHA", "BETA"])
        self.assertTrue(all(outcome.ok for outcome in outcomes))
        self.assertEqual(attempts["alpha"], 2)
        self.assertEqual(attempts["beta"], 1)

    def test_timeout_errors_retry_within_configured_limit(self) -> None:
        attempts = 0

        def worker(_item: str) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TimeoutError("stuck LLM request")
            return "recovered"

        outcomes = run_llm_jobs(
            ["alpha"],
            worker,
            options=LLMJobOptions(
                concurrency=1,
                max_retries=1,
                retry_backoff_seconds=0,
                progress_every=0,
            ),
            label="test",
        )

        self.assertEqual(attempts, 2)
        self.assertTrue(outcomes[0].ok)
        self.assertEqual(outcomes[0].value, "recovered")

    def test_timeout_errors_stop_after_configured_retries(self) -> None:
        attempts = 0

        def worker(_item: str) -> str:
            nonlocal attempts
            attempts += 1
            raise TimeoutError("stuck LLM request")

        outcomes = run_llm_jobs(
            ["alpha"],
            worker,
            options=LLMJobOptions(
                concurrency=1,
                max_retries=2,
                retry_backoff_seconds=0,
                progress_every=0,
            ),
            label="test",
        )

        self.assertEqual(attempts, 3)
        self.assertFalse(outcomes[0].ok)
        self.assertIsInstance(outcomes[0].error, TimeoutError)

    def test_retryable_provider_request_failure_uses_job_retry_limit(self) -> None:
        provider_calls = 0

        def completion(**kwargs):
            nonlocal provider_calls
            provider_calls += 1
            self.assertEqual(kwargs["max_retries"], 2)
            raise TimeoutError("provider retries exhausted")

        fake_litellm = types.SimpleNamespace(
            completion=completion,
            suppress_debug_info=False,
            request_timeout=None,
        )
        config = LLMConfig(
            api_base="https://example.test/v1",
            api_key="test-key",
            model="openai/responses/gpt-5.4-mini",
        )

        def worker(_item: str) -> str:
            litellm_completion(
                messages=[{"role": "user", "content": "Hello"}],
                config=config,
            )
            return "unreachable"

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            outcomes = run_llm_jobs(
                ["item"],
                worker,
                options=LLMJobOptions(
                    concurrency=1,
                    max_retries=2,
                    retry_backoff_seconds=0,
                    progress_every=0,
                ),
            )

        self.assertEqual(provider_calls, 3)
        self.assertFalse(outcomes[0].ok)
        self.assertIsInstance(outcomes[0].error, LLMRequestError)

    def test_ten_consecutive_matching_transient_errors_open_circuit(self) -> None:
        calls: list[int] = []

        def worker(item: int) -> None:
            calls.append(item)
            raise self._request_error(status_code=503, retryable=True)

        with self.assertRaisesRegex(RuntimeError, "circuit breaker.*10 consecutive"):
            run_llm_jobs(
                range(30),
                worker,
                options=LLMJobOptions(
                    concurrency=1,
                    max_retries=0,
                    retry_backoff_seconds=0,
                    progress_every=0,
                    batch_size=20,
                    circuit_breaker_threshold=10,
                ),
                label="relation",
            )

        self.assertGreaterEqual(len(calls), 10)
        self.assertLess(len(calls), 30)

    def test_success_resets_consecutive_transient_error_count(self) -> None:
        calls: list[int] = []

        def worker(item: int) -> int:
            calls.append(item)
            if item == 9:
                return item
            raise self._request_error(status_code=503, retryable=True)

        outcomes = run_llm_jobs(
            range(19),
            worker,
            options=LLMJobOptions(
                concurrency=1,
                max_retries=0,
                retry_backoff_seconds=0,
                progress_every=0,
                batch_size=19,
                circuit_breaker_threshold=10,
            ),
            label="relation",
        )

        self.assertEqual(calls, list(range(19)))
        self.assertEqual(len(outcomes), 19)
        self.assertTrue(outcomes[9].ok)

    def test_permanent_provider_error_stops_batch_immediately(self) -> None:
        calls: list[int] = []

        def worker(item: int) -> None:
            calls.append(item)
            raise self._request_error(status_code=401, retryable=False)

        with self.assertRaisesRegex(RuntimeError, "permanent.*401"):
            run_llm_jobs(
                range(10),
                worker,
                options=LLMJobOptions(
                    concurrency=1,
                    max_retries=8,
                    retry_backoff_seconds=0,
                    progress_every=0,
                    batch_size=1,
                ),
                label="contract",
            )

        self.assertEqual(calls, [0])

    def test_uses_concurrent_workers(self) -> None:
        active = 0
        max_active = 0
        lock = threading.Lock()

        def worker(item: int) -> int:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return item

        outcomes = run_llm_jobs(
            list(range(6)),
            worker,
            options=LLMJobOptions(concurrency=3, max_retries=0, progress_every=0, batch_size=3),
            label="test",
        )

        self.assertEqual([outcome.value for outcome in outcomes], list(range(6)))
        self.assertGreaterEqual(max_active, 2)

    def test_slow_early_job_does_not_block_later_batch_submission(self) -> None:
        item_two_started = threading.Event()
        slow_observed_later_start: list[bool] = []

        def worker(item: int) -> int:
            if item == 0:
                slow_observed_later_start.append(item_two_started.wait(timeout=0.2))
                return item
            if item == 2:
                item_two_started.set()
            time.sleep(0.01)
            return item

        outcomes = run_llm_jobs(
            [0, 1, 2, 3],
            worker,
            options=LLMJobOptions(concurrency=2, max_retries=0, progress_every=0, batch_size=2),
            label="test",
        )

        self.assertEqual([outcome.value for outcome in outcomes], [0, 1, 2, 3])
        self.assertEqual(slow_observed_later_start, [True])

    def test_usage_contains_only_the_accepted_attempt(self) -> None:
        responses = [
            {
                "choices": [{"message": {"content": "not json"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
            {
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 3},
            },
        ]
        fake_litellm = types.SimpleNamespace(
            completion=lambda **_kwargs: responses.pop(0),
            suppress_debug_info=False,
            request_timeout=None,
        )
        config = LLMConfig(
            api_base="https://example.test/v1",
            api_key="test-key",
            model="openai/responses/gpt-5.4-mini",
        )

        with TemporaryDirectory() as tmp, patch.dict(sys.modules, {"litellm": fake_litellm}):
            usage_path = Path(tmp) / "usage.jsonl"

            def worker(_item: str) -> dict[str, bool]:
                response = litellm_completion(
                    messages=[{"role": "user", "content": "Return JSON."}],
                    config=config,
                )
                return json.loads(response["choices"][0]["message"]["content"])

            with llm_usage_context(log_path=usage_path):
                outcomes = run_llm_jobs(
                    ["item"],
                    worker,
                    options=LLMJobOptions(
                        concurrency=1,
                        max_retries=1,
                        retry_backoff_seconds=0,
                        progress_every=0,
                    ),
                )

            records = load_usage_records(usage_path)

        self.assertTrue(outcomes[0].ok)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].prompt_tokens, 20)
        self.assertEqual(records[0].completion_tokens, 3)

    def test_usage_write_failure_does_not_retry_an_accepted_job(self) -> None:
        fake_litellm = types.SimpleNamespace(
            completion=lambda **_kwargs: {
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 3},
            },
            suppress_debug_info=False,
            request_timeout=None,
        )
        config = LLMConfig(
            api_base="https://example.test/v1",
            api_key="test-key",
            model="openai/responses/gpt-5.4-mini",
        )
        worker_calls = 0

        def worker(_item: str) -> dict[str, bool]:
            nonlocal worker_calls
            worker_calls += 1
            response = litellm_completion(
                messages=[{"role": "user", "content": "Return JSON."}],
                config=config,
            )
            return json.loads(response["choices"][0]["message"]["content"])

        with (
            TemporaryDirectory() as tmp,
            patch.dict(sys.modules, {"litellm": fake_litellm}),
            patch("skillfabric.runtime.llm.LLMUsageTracker.append", side_effect=OSError("disk")),
            patch("skillfabric.runtime.llm.LOGGER.warning") as warning,
            llm_usage_context(log_path=Path(tmp) / "usage.jsonl"),
        ):
            outcomes = run_llm_jobs(
                ["item"],
                worker,
                options=LLMJobOptions(
                    concurrency=1,
                    max_retries=2,
                    retry_backoff_seconds=0,
                    progress_every=0,
                ),
            )

        self.assertEqual(worker_calls, 1)
        self.assertTrue(outcomes[0].ok)
        warning.assert_called_once_with("usage_write_failed error_type=%s", "OSError")


if __name__ == "__main__":
    unittest.main()
