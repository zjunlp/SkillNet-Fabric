from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.runtime.usage import LLMUsageTracker, load_usage_records, summarize_usage


class LLMUsageTests(unittest.TestCase):
    def test_record_completion_estimates_tokens_when_response_has_no_usage(self) -> None:
        with TemporaryDirectory() as tmp:
            tracker = LLMUsageTracker(log_path=Path(tmp) / "usage.jsonl")

            record = tracker.record_completion(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "Summarize this short text."}],
                response={"choices": [{"message": {"content": "Short summary."}}]},
                operation="wiki_build",
                duration_ms=12,
                status="completed",
            )

            self.assertEqual(record.operation, "wiki_build")
            self.assertGreater(record.prompt_tokens, 0)
            self.assertGreater(record.completion_tokens, 0)
            self.assertEqual(record.total_tokens, record.prompt_tokens + record.completion_tokens)
            self.assertTrue(record.estimated)
            self.assertTrue(record.pricing_known)
            self.assertIsNotNone(record.cost_usd)

    def test_response_usage_is_preferred_over_local_estimate(self) -> None:
        record = LLMUsageTracker().record_completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Count exactly."}],
            response={
                "choices": [{"message": {"content": "Done"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            },
            operation="route",
            duration_ms=3,
            status="completed",
        )

        self.assertEqual(record.prompt_tokens, 11)
        self.assertEqual(record.completion_tokens, 7)
        self.assertEqual(record.total_tokens, 18)
        self.assertFalse(record.estimated)

    def test_skillfabric_pricing_override_prices_gpt_5_4_mini_with_cached_input(self) -> None:
        record = LLMUsageTracker().record_completion(
            model="openai/responses/gpt-5.4-mini",
            messages=[{"role": "user", "content": "Reply with OK only."}],
            response={
                "choices": [{"message": {"content": "OK"}}],
                "usage": {
                    "prompt_tokens": 5091,
                    "completion_tokens": 5,
                    "total_tokens": 5096,
                    "prompt_tokens_details": {"cached_tokens": 4864},
                },
            },
            operation="llm_smoke",
            duration_ms=3583,
            status="completed",
        )

        expected_cost = ((227 * 0.75) + (4864 * 0.075) + (5 * 4.50)) / 1_000_000
        payload = record.to_dict()
        self.assertEqual(record.prompt_tokens, 5091)
        self.assertEqual(record.completion_tokens, 5)
        self.assertEqual(record.cached_prompt_tokens, 4864)
        self.assertEqual(record.billable_prompt_tokens, 227)
        self.assertAlmostEqual(record.cost_usd or 0.0, expected_cost, places=10)
        self.assertTrue(record.pricing_known)
        self.assertEqual(record.pricing_source, "skillfabric_openai_official")
        self.assertEqual(payload["cached_prompt_tokens"], 4864)
        self.assertEqual(payload["billable_prompt_tokens"], 227)

    def test_skillfabric_pricing_override_normalizes_gpt_5_4_mini_alias(self) -> None:
        record = LLMUsageTracker().record_completion(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": "Reply with OK only."}],
            response={
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            },
            operation="llm_smoke",
            duration_ms=10,
            status="completed",
        )

        expected_cost = ((100 * 0.75) + (10 * 4.50)) / 1_000_000
        self.assertAlmostEqual(record.cost_usd or 0.0, expected_cost, places=10)
        self.assertEqual(record.pricing_source, "skillfabric_openai_official")

    def test_cached_tokens_are_clamped_to_prompt_tokens(self) -> None:
        record = LLMUsageTracker().record_completion(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": "Reply with OK only."}],
            response={
                "choices": [{"message": {"content": "OK"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                    "prompt_tokens_details": {"cached_tokens": 99},
                },
            },
            operation="llm_smoke",
            duration_ms=10,
            status="completed",
        )

        self.assertEqual(record.cached_prompt_tokens, 10)
        self.assertEqual(record.billable_prompt_tokens, 0)

    def test_unknown_model_pricing_does_not_fail(self) -> None:
        record = LLMUsageTracker().record_completion(
            model="openai/unknown-local-model",
            messages=[{"role": "user", "content": "Hello"}],
            response={"choices": [{"message": {"content": "Hi"}}]},
            operation="validation",
            duration_ms=1,
            status="completed",
        )

        self.assertGreater(record.total_tokens, 0)
        self.assertFalse(record.pricing_known)
        self.assertIsNone(record.cost_usd)

    def test_jsonl_records_load_and_aggregate(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.jsonl"
            tracker = LLMUsageTracker(log_path=path)
            tracker.record_completion(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "First"}],
                response={
                    "choices": [{"message": {"content": "One"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                },
                operation="kg_build",
                duration_ms=10,
                status="completed",
            )
            tracker.record_completion(
                model="openai/unknown-local-model",
                messages=[{"role": "user", "content": "Second"}],
                response={"choices": [{"message": {"content": "Two"}}]},
                operation="route",
                duration_ms=20,
                status="completed",
            )

            records = load_usage_records(path)
            totals = summarize_usage(records)

            self.assertEqual(len(records), 2)
            self.assertEqual(totals.total_calls, 2)
            self.assertEqual(totals.prompt_tokens, sum(item.prompt_tokens for item in records))
            self.assertEqual(totals.by_operation["kg_build"]["total_calls"], 1)
            self.assertEqual(totals.by_operation["route"]["total_calls"], 1)
            self.assertGreater(totals.cost_usd or 0.0, 0.0)
            self.assertEqual(totals.pricing_unknown_calls, 1)

    def test_usage_summary_can_be_scoped_to_one_build(self) -> None:
        tracker = LLMUsageTracker()
        for build_id in ("build-old", "build-current"):
            tracker.record_completion(
                model="gpt-5.4-mini",
                messages=[{"role": "user", "content": build_id}],
                response={
                    "choices": [{"message": {"content": "done"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                },
                operation="graph.contract_extraction",
                duration_ms=1,
                status="completed",
                metadata={"build_id": build_id},
            )

        totals = summarize_usage(tracker.records, metadata={"build_id": "build-current"})

        self.assertEqual(totals.total_calls, 1)
        self.assertEqual(totals.total_tokens, 7)

    def test_jsonl_rejects_records_missing_current_pricing_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "timestamp": "2026-06-14T00:00:00Z",
                        "call_id": "call_legacy",
                        "operation": "route",
                        "model": "openai/gpt-4o-mini",
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 7,
                        "cost_usd": 0.001,
                        "estimated": False,
                        "pricing_known": True,
                        "duration_ms": 10,
                        "status": "completed",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "exact schema"):
                load_usage_records(path)

    def test_failed_call_records_prompt_tokens_and_error(self) -> None:
        record = LLMUsageTracker().record_completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "This request failed."}],
            response=None,
            operation="llm_smoke",
            duration_ms=4,
            status="failed",
            error="TimeoutError: exceeded",
        )

        self.assertEqual(record.status, "failed")
        self.assertGreater(record.prompt_tokens, 0)
        self.assertEqual(record.completion_tokens, 0)
        self.assertIn("TimeoutError", record.error or "")
        self.assertTrue(record.pricing_known)
        self.assertIsNone(record.cost_usd)

    def test_litellm_token_counter_failure_uses_local_estimate(self) -> None:
        fake_litellm = types.SimpleNamespace()

        def token_counter(**_kwargs):
            raise RuntimeError("counter unavailable")

        fake_litellm.token_counter = token_counter
        fake_litellm.cost_per_token = lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("unknown")
        )
        original = sys.modules.get("litellm")
        sys.modules["litellm"] = fake_litellm
        try:
            record = LLMUsageTracker().record_completion(
                model="openai/unknown-local-model",
                messages=[{"role": "user", "content": "fallback words"}],
                response={"choices": [{"message": {"content": "fallback output"}}]},
                operation="route",
                duration_ms=5,
                status="completed",
            )
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

        self.assertGreater(record.prompt_tokens, 0)
        self.assertGreater(record.completion_tokens, 0)
        self.assertTrue(record.estimated)


if __name__ == "__main__":
    unittest.main()
