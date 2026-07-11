from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.wiki.models import WikiBuildConfig
from skillfabric.wiki.summarizer import LiteLLMSummaryProvider, WikiSummarizer


class CountingSummaryProvider:
    model_id = "counting-model"

    def __init__(self) -> None:
        self.calls = 0

    def summarize(self, *, page_type: str, entity_id: str, payload: dict[str, object]) -> dict[str, str]:
        self.calls += 1
        return {
            "routing_summary": f"{entity_id} routing",
            "workflow_summary": f"{entity_id} workflow",
            "summary": f"{entity_id} summary",
        }


class FlakySummaryProvider:
    model_id = "flaky-model"

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def summarize(self, *, page_type: str, entity_id: str, payload: dict[str, object]) -> dict[str, str]:
        self.calls[entity_id] = self.calls.get(entity_id, 0) + 1
        if self.calls[entity_id] == 1:
            raise RuntimeError("transient")
        return {
            "routing_summary": f"{entity_id} routing",
            "workflow_summary": f"{entity_id} workflow",
            "summary": f"{entity_id} summary",
        }


class WikiSummaryCacheTests(unittest.TestCase):
    def test_deterministic_summary_reuses_interface_capability_without_duplication(self) -> None:
        with TemporaryDirectory() as tmp:
            config = WikiBuildConfig(workspace=Path(tmp) / ".skillfabric", use_llm_summaries=False)
            summarizer = WikiSummarizer(config)

            record = summarizer.summarize_entity(
                page_type="skill",
                entity_id="skill:parser",
                content_hash="hash-parser",
                payload={
                    "name": "parser",
                    "description": "Registry description.",
                    "capability_summary": "Extract evidence-grounded tables from PDF files.",
                    "when_to_use": "Use for PDF table extraction.",
                    "requires": ["pdf_document"],
                    "produces": ["csv_table"],
                    "uses_tools": ["pdfplumber"],
                },
            )

        self.assertEqual(record.summary, "Extract evidence-grounded tables from PDF files.")
        self.assertEqual(record.routing_summary, record.summary)
        self.assertEqual(record.workflow_summary, "")
        self.assertNotIn("Requires:", record.summary)

    def test_litellm_summary_provider_uses_bounded_low_effort_call(self) -> None:
        calls: list[dict[str, object]] = []
        fake_litellm = types.SimpleNamespace()

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "summary",
                                    "routing_summary": "routing",
                                    "workflow_summary": "workflow",
                                }
                            )
                        }
                    }
                ]
            }

        fake_litellm.completion = fake_completion
        original = sys.modules.get("litellm")
        sys.modules["litellm"] = fake_litellm
        try:
            with TemporaryDirectory() as tmp:
                env_file = Path(tmp) / ".env"
                env_file.write_text(
                    "\n".join(
                        [
                            "BASE_URL=https://example.test/api",
                            "API_KEY=sk-test",
                            "MODEL=openai/test-model",
                            "MAX_TOKENS=32768",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                provider = LiteLLMSummaryProvider(env_file=env_file)

                provider.summarize(page_type="skill", entity_id="skill:a", payload={"name": "a"})
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

        self.assertEqual(calls[0]["max_tokens"], 2048)
        self.assertEqual(calls[0]["reasoning_effort"], "low")
        prompt_payload = json.loads(calls[0]["messages"][1]["content"])
        prompt_text = json.dumps(calls[0]["messages"], ensure_ascii=False)
        self.assertEqual(prompt_payload["prompt_id"], "skillcontract_summary_routing_guidance_v3")
        for field in ("task", "rules", "output_schema", "entity"):
            self.assertIn(field, prompt_payload)
        self.assertIn("routing_summary", prompt_text)
        self.assertIn("workflow_summary", prompt_text)
        self.assertIn("coverage-gap", prompt_text)
        self.assertIn("Do not solve", prompt_text)
        self.assertLess(len(prompt_text), 2000)

    def test_summary_cache_reuses_content_hash_and_model_id(self) -> None:
        with TemporaryDirectory() as tmp:
            provider = CountingSummaryProvider()
            config = WikiBuildConfig(workspace=Path(tmp) / ".skillfabric", use_llm_summaries=True)
            summarizer = WikiSummarizer(config, provider=provider)

            first = summarizer.summarize_skill(
                entity_id="skill:test",
                content_hash="hash-1",
                payload={"name": "test"},
            )
            summarizer.save()
            second = WikiSummarizer(config, provider=provider).summarize_skill(
                entity_id="skill:test",
                content_hash="hash-1",
                payload={"name": "test"},
            )

            self.assertEqual(first.routing_summary, second.routing_summary)
            self.assertEqual(provider.calls, 1)

    def test_summary_batch_retries_and_preserves_records(self) -> None:
        with TemporaryDirectory() as tmp:
            provider = FlakySummaryProvider()
            config = WikiBuildConfig(
                workspace=Path(tmp) / ".skillfabric",
                use_llm_summaries=True,
                llm_concurrency=2,
                llm_max_retries=1,
                llm_progress_every=0,
            )
            summarizer = WikiSummarizer(config, provider=provider)

            records = summarizer.summarize_many(
                [
                    {
                        "page_type": "skill",
                        "entity_id": "skill:a",
                        "content_hash": "hash-a",
                        "payload": {"name": "a"},
                    },
                    {
                        "page_type": "skill",
                        "entity_id": "skill:b",
                        "content_hash": "hash-b",
                        "payload": {"name": "b"},
                    },
                ]
            )

            self.assertEqual(records[("skill", "skill:a")].summary, "skill:a summary")
            self.assertEqual(records[("skill", "skill:b")].summary, "skill:b summary")
            self.assertEqual(provider.calls["skill:a"], 2)
            self.assertEqual(provider.calls["skill:b"], 2)

    def test_cached_fallback_records_are_counted_as_fallbacks(self) -> None:
        with TemporaryDirectory() as tmp:
            config = WikiBuildConfig(workspace=Path(tmp) / ".skillfabric", use_llm_summaries=False)
            first = WikiSummarizer(config)
            first.summarize_many(
                [
                    {
                        "page_type": "skill",
                        "entity_id": "skill:a",
                        "content_hash": "hash-a",
                        "payload": {"name": "a"},
                    }
                ]
            )

            second = WikiSummarizer(config)
            second.summarize_many(
                [
                    {
                        "page_type": "skill",
                        "entity_id": "skill:a",
                        "content_hash": "hash-a",
                        "payload": {"name": "a"},
                    }
                ]
            )

            self.assertEqual(second.cache_hits, 1)
            self.assertEqual(second.fallback_count, 1)


if __name__ == "__main__":
    unittest.main()
