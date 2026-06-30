from __future__ import annotations

import threading
import time
import unittest

from skillfabric.llm_jobs import LLMJobOptions, run_llm_jobs


class LLMJobRunnerTests(unittest.TestCase):
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

    def test_timeout_errors_do_not_retry(self) -> None:
        attempts = 0

        def worker(_item: str) -> str:
            nonlocal attempts
            attempts += 1
            raise TimeoutError("stuck LLM request")

        outcomes = run_llm_jobs(
            ["alpha"],
            worker,
            options=LLMJobOptions(concurrency=1, max_retries=3, progress_every=0),
            label="test",
        )

        self.assertEqual(attempts, 1)
        self.assertFalse(outcomes[0].ok)
        self.assertIsInstance(outcomes[0].error, TimeoutError)

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

    def test_retries_error_payloads_when_requested(self) -> None:
        attempts = 0

        def worker(item: str) -> dict[str, str]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return {"error_type": "api_error", "reason": "timeout"}
            return {"ok": item}

        outcomes = run_llm_jobs(
            ["alpha"],
            worker,
            options=LLMJobOptions(concurrency=1, max_retries=1, progress_every=0),
            label="test",
            retry_on_result=lambda payload: bool(payload.get("error_type")),
        )

        self.assertTrue(outcomes[0].ok)
        self.assertEqual(outcomes[0].value, {"ok": "alpha"})
        self.assertEqual(attempts, 2)


if __name__ == "__main__":
    unittest.main()
