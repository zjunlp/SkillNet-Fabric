from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from skillfabric.runtime.jobs import LLMJobOptions
from skillfabric.wiki.models import WikiBuildConfig
from skillfabric.wiki.summarizer import (
    CONTRACT_SUMMARY_MODEL_ID,
    WIKI_SUMMARY_PROMPT_ID,
    LiteLLMSummaryProvider,
    WikiSummarizer,
    WikiSummaryError,
)


class CountingSummaryProvider:
    model_id = "counting-model"

    def __init__(self) -> None:
        self.calls = 0

    def summarize(
        self, *, page_type: str, entity_id: str, payload: dict[str, object]
    ) -> dict[str, str]:
        del page_type, payload
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

    def summarize(
        self, *, page_type: str, entity_id: str, payload: dict[str, object]
    ) -> dict[str, str]:
        del page_type, payload
        self.calls[entity_id] = self.calls.get(entity_id, 0) + 1
        if self.calls[entity_id] == 1:
            raise RuntimeError("transient")
        return {
            "routing_summary": f"{entity_id} routing",
            "workflow_summary": f"{entity_id} workflow",
            "summary": f"{entity_id} summary",
        }


class StaticSummaryProvider:
    model_id = "static-model"

    def __init__(self, response: dict[str, object] | BaseException) -> None:
        self.response = response
        self.calls = 0

    def summarize(
        self, *, page_type: str, entity_id: str, payload: dict[str, object]
    ) -> dict[str, str]:
        del page_type, entity_id, payload
        self.calls += 1
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response  # type: ignore[return-value]


class WikiSummaryCacheTests(unittest.TestCase):
    def test_summary_cache_record_contains_only_rendered_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            record = WikiSummarizer(
                WikiBuildConfig(
                    workspace=Path(tmp) / ".skillfabric",
                    use_llm_summaries=False,
                )
            ).summarize_skill(
                entity_id="skill:test",
                content_hash="hash-test",
                payload={"capability": "Test tasks."},
            )

            self.assertEqual(
                set(record.to_dict()),
                {
                    "page_type",
                    "entity_id",
                    "content_hash",
                    "routing_summary",
                    "workflow_summary",
                    "summary",
                },
            )

    def test_wiki_config_rejects_invalid_runtime_values(self) -> None:
        invalid = (
            ({"workspace": ""}, "workspace"),
            ({"env_file": ""}, "env_file"),
            ({"use_llm_summaries": 1}, "use_llm_summaries"),
            ({"max_neighbors_per_section": True}, "max_neighbors_per_section"),
            ({"max_neighbors_per_section": 0}, "max_neighbors_per_section"),
            ({"llm_options": object()}, "llm_options"),
        )

        for kwargs, message in invalid:
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(
                    (TypeError, ValueError),
                    message,
                ),
            ):
                WikiBuildConfig(**kwargs)

    def test_wiki_config_normalizes_shared_llm_job_options(self) -> None:
        config = WikiBuildConfig(
            llm_options=LLMJobOptions(concurrency=2, batch_size=None),
        )

        self.assertEqual(config.llm_options, LLMJobOptions(concurrency=2, batch_size=8))

    def test_disabled_summary_mode_does_not_parse_llm_job_configuration(self) -> None:
        with TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "SKILLFABRIC_LLM_CONCURRENCY=not-an-integer\n",
                encoding="utf-8",
            )
            summarizer = WikiSummarizer(
                WikiBuildConfig(
                    workspace=Path(tmp) / ".skillfabric",
                    env_file=env_file,
                    use_llm_summaries=False,
                )
            )

            record = summarizer.summarize_skill(
                entity_id="skill:test",
                content_hash="hash-test",
                payload={"capability": "Test tasks."},
            )

            self.assertEqual(record.summary, "Test tasks.")

    def test_litellm_summary_provider_uses_env_max_tokens(self) -> None:
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

        self.assertEqual(calls[0]["max_tokens"], 32768)
        prompt_text = json.dumps(calls[0]["messages"], ensure_ascii=False)
        self.assertEqual(WIKI_SUMMARY_PROMPT_ID, "wiki_summary")
        self.assertEqual(CONTRACT_SUMMARY_MODEL_ID, "contract-derived")
        self.assertIn(WIKI_SUMMARY_PROMPT_ID, prompt_text)
        self.assertIn("<output_schema>", prompt_text)
        self.assertIn("<source_data>", prompt_text)
        self.assertIn("untrusted data", prompt_text)
        self.assertIn("routing_summary", prompt_text)
        self.assertIn("workflow_summary", prompt_text)
        self.assertNotIn('"todo"', prompt_text)
        self.assertNotIn('"constraints"', prompt_text)
        user_prompt = calls[0]["messages"][1]["content"]
        self.assertLess(user_prompt.index("<source_data>"), user_prompt.index("<task>"))
        self.assertLess(user_prompt.index("<task>"), user_prompt.index("<output_schema>"))

    def test_llm_summary_requires_exact_nonempty_string_fields(self) -> None:
        invalid_responses = [
            {"summary": "summary", "routing_summary": "routing"},
            {
                "summary": "summary",
                "routing_summary": "routing",
                "workflow_summary": "workflow",
                "extra": "unexpected",
            },
            {
                "summary": " ",
                "routing_summary": "routing",
                "workflow_summary": "workflow",
            },
            {
                "summary": "summary",
                "routing_summary": ["routing"],
                "workflow_summary": "workflow",
            },
        ]
        for response in invalid_responses:
            with self.subTest(response=response), TemporaryDirectory() as tmp:
                provider = StaticSummaryProvider(response)
                config = WikiBuildConfig(
                    workspace=Path(tmp) / ".skillfabric",
                    use_llm_summaries=True,
                    llm_options=LLMJobOptions(max_retries=0, progress_every=0),
                )

                with self.assertRaisesRegex(WikiSummaryError, "summary generation failed"):
                    WikiSummarizer(config, provider=provider).summarize_skill(
                        entity_id="skill:test",
                        content_hash="hash-1",
                        payload={"name": "test"},
                    )

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

    def test_summary_cache_key_includes_prompt_fingerprint(self) -> None:
        with TemporaryDirectory() as tmp:
            config = WikiBuildConfig(
                workspace=Path(tmp) / ".skillfabric",
                use_llm_summaries=False,
            )
            summarizer = WikiSummarizer(config)

            summarizer.summarize_skill(
                entity_id="skill:test",
                content_hash="hash-1",
                payload={"capability": "Test tasks."},
            )

            self.assertEqual(len(summarizer.cache), 1)
            cache_key = next(iter(summarizer.cache))
            prompt_id, fingerprint, *_parts = cache_key.split("|")
            self.assertEqual(prompt_id, WIKI_SUMMARY_PROMPT_ID)
            self.assertEqual(len(fingerprint), 64)
            self.assertTrue(all(character in "0123456789abcdef" for character in fingerprint))

    def test_summary_batch_retries_and_preserves_records(self) -> None:
        with TemporaryDirectory() as tmp:
            provider = FlakySummaryProvider()
            config = WikiBuildConfig(
                workspace=Path(tmp) / ".skillfabric",
                use_llm_summaries=True,
                llm_options=LLMJobOptions(
                    concurrency=2,
                    max_retries=1,
                    progress_every=0,
                ),
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

    def test_llm_batch_failure_does_not_generate_contract_summary(self) -> None:
        with TemporaryDirectory() as tmp:
            provider = StaticSummaryProvider(RuntimeError("provider unavailable"))
            config = WikiBuildConfig(
                workspace=Path(tmp) / ".skillfabric",
                use_llm_summaries=True,
                llm_options=LLMJobOptions(max_retries=0, progress_every=0),
            )
            summarizer = WikiSummarizer(config, provider=provider)

            with self.assertRaisesRegex(WikiSummaryError, "skill:a"):
                summarizer.summarize_many(
                    [
                        {
                            "page_type": "skill",
                            "entity_id": "skill:a",
                            "content_hash": "hash-a",
                            "payload": {"name": "a"},
                        }
                    ]
                )

            self.assertFalse(summarizer.cache_path.exists())

    def test_llm_provider_initialization_failure_is_not_silenced(self) -> None:
        with TemporaryDirectory() as tmp:
            config = WikiBuildConfig(
                workspace=Path(tmp) / ".skillfabric",
                use_llm_summaries=True,
            )
            with (
                patch(
                    "skillfabric.wiki.summarizer.LiteLLMSummaryProvider",
                    side_effect=ValueError("missing model"),
                ),
                self.assertRaisesRegex(WikiSummaryError, "provider initialization failed"),
            ):
                WikiSummarizer(config).summarize_many([])

    def test_disabled_llm_mode_caches_contract_derived_records(self) -> None:
        with TemporaryDirectory() as tmp:
            config = WikiBuildConfig(workspace=Path(tmp) / ".skillfabric", use_llm_summaries=False)
            first = WikiSummarizer(config)
            first_records = first.summarize_many(
                [
                    {
                        "page_type": "skill",
                        "entity_id": "skill:a",
                        "content_hash": "hash-a",
                        "payload": {
                            "name": "a",
                            "capability": "Parse normalized tables.",
                            "when_to_use": "Use for tabular documents.",
                            "requires": ["document"],
                            "produces": ["normalized table"],
                        },
                    }
                ]
            )

            second = WikiSummarizer(config)
            second_records = second.summarize_many(
                [
                    {
                        "page_type": "skill",
                        "entity_id": "skill:a",
                        "content_hash": "hash-a",
                        "payload": {
                            "name": "a",
                            "capability": "Parse normalized tables.",
                            "when_to_use": "Use for tabular documents.",
                            "requires": ["document"],
                            "produces": ["normalized table"],
                        },
                    }
                ]
            )

            record = first_records[("skill", "skill:a")]
            self.assertEqual(record.summary, "Parse normalized tables.")
            self.assertEqual(record.routing_summary, "Use for tabular documents.")
            self.assertEqual(second_records[("skill", "skill:a")].to_dict(), record.to_dict())
            self.assertEqual(second.cache_hits, 1)


if __name__ == "__main__":
    unittest.main()
