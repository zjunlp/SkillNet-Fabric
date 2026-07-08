from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.compiled_graph.canonicalization.candidates import (
    CanonicalSemanticEmbedder,
    EmbeddingProviderCanonicalEmbedder,
    candidate_groups_from_terms,
    generate_semantic_candidate_pairs,
    normalized_candidate_text,
)
from skillfabric.compiled_graph.canonicalization.compiler import (
    DeterministicCanonicalizationProvider,
    LiteLLMCanonicalizationProvider,
    canonicalize_contract_objects,
)
from skillfabric.compiled_graph.canonicalization.models import (
    CanonicalizationCluster,
    RawContractObject,
)
from skillfabric.compiled_graph.canonicalization.prompts import build_canonicalization_messages
from skillfabric.compiled_graph.execution.compiler import compile_execution_graph
from skillfabric.compiled_graph.interface.models import (
    InterfaceEvidence,
    InterfaceField,
    SkillInterface,
)
from skillfabric.runtime.jobs import LLMJobOptions
from skillfabric.runtime.llm import LLMConfig


def _interface(
    skill_id: str,
    *,
    requires: list[InterfaceField] | None = None,
    produces: list[InterfaceField] | None = None,
) -> SkillInterface:
    return SkillInterface(
        skill_id=skill_id,
        content_hash=f"hash-{skill_id}",
        capability_summary=f"{skill_id} summary",
        requires=requires or [],
        produces=produces or [],
    )


def _field(skill_id: str, name: str, kind: str = "artifact", description: str = "") -> InterfaceField:
    return InterfaceField(
        name=name,
        kind=kind,
        description=description,
        confidence=0.9,
        evidence=[InterfaceEvidence(skill=skill_id, line=1, text=f"{skill_id} mentions {name}.")],
    )


class StaticCanonicalizationProvider:
    model_id = "static-canonicalizer"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[CanonicalizationCluster] = []

    def canonicalize(self, cluster: CanonicalizationCluster):
        self.calls.append(cluster)
        if self.fail:
            raise RuntimeError("canonicalizer unavailable")
        return {
            "canonical_objects": [
                {
                    "name": "spreadsheet_table",
                    "type": "data",
                    "term_ids": [term.term_id for term in cluster.terms],
                    "confidence": 0.9,
                }
            ],
            "omitted_term_ids": [],
        }


class OmitCanonicalizationProvider:
    model_id = "omit-canonicalizer"

    def __init__(self) -> None:
        self.calls: list[CanonicalizationCluster] = []

    def canonicalize(self, cluster: CanonicalizationCluster):
        self.calls.append(cluster)
        return {
            "canonical_objects": [],
            "omitted_term_ids": [term.term_id for term in cluster.terms],
        }


class SplitSingletonCanonicalizationProvider:
    model_id = "split-singleton-canonicalizer"

    def canonicalize(self, cluster: CanonicalizationCluster):
        return {
            "canonical_objects": [
                {
                    "name": term.name,
                    "type": "environment",
                    "term_ids": [term.term_id],
                    "confidence": 0.98,
                }
                for term in cluster.terms
            ],
            "omitted_term_ids": [],
        }


class WrongTypeCanonicalizationProvider:
    model_id = "wrong-type-canonicalizer"

    def canonicalize(self, cluster: CanonicalizationCluster):
        return {
            "canonical_objects": [
                {
                    "name": "markdown_report",
                    "type": "environment",
                    "term_ids": [term.term_id for term in cluster.terms],
                    "confidence": 0.98,
                }
            ],
            "omitted_term_ids": [],
        }


class FixedSemanticEmbedder(CanonicalSemanticEmbedder):
    model_id = "fixed-semantic-test"

    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "spreadsheet export" in lowered or "worksheet rows" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "md file" in lowered or "markdown file" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class NoSimilarityEmbedder(CanonicalSemanticEmbedder):
    model_id = "no-similarity-test"

    def embed_texts(self, texts):
        vectors = []
        for index, _text in enumerate(texts):
            vector = [0.0] * max(len(texts), 1)
            vector[index] = 1.0
            vectors.append(vector)
        return vectors


class CountingSemanticEmbedder(CanonicalSemanticEmbedder):
    model_id = "counting-semantic-test"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            vector = [0.0, 0.0, 0.0]
            vector[len(text) % 3] = 1.0
            vectors.append(vector)
        return vectors


class EmbedManyOnlyProvider:
    model_id = "embed-many-only-test"
    dimension = 2

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, text: str) -> list[float]:
        raise AssertionError(f"embed should not be called for batched provider: {text}")

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1, 0] for _text in texts]


class CanonicalizationTests(unittest.TestCase):
    def test_normalized_text_is_mechanical_only(self) -> None:
        self.assertEqual(normalized_candidate_text("CSV-table"), "csv table")
        self.assertEqual(normalized_candidate_text("object_in_inventory"), "object in inventory")
        self.assertEqual(normalized_candidate_text("markdown file"), "markdown file")
        self.assertNotEqual(normalized_candidate_text("md file"), normalized_candidate_text("markdown file"))

    def test_prompt_is_short_positive_task_with_schema_only(self) -> None:
        cluster = CanonicalizationCluster(
            cluster_id="canonical-group:test",
            terms=[
                RawContractObject("skill:a", "produces", "spreadsheet export", "artifact"),
                RawContractObject("skill:b", "requires", "worksheet rows", "data"),
            ],
        )

        messages = build_canonicalization_messages(cluster)
        payload = json.loads(messages[1]["content"])
        prompt_text = json.dumps(messages, ensure_ascii=False)

        self.assertEqual(payload["prompt_id"], "interface_term_canonicalization_v2")
        self.assertEqual(set(payload), {"prompt_id", "task", "context", "input", "output_schema", "terms"})
        self.assertEqual(
            set(payload["output_schema"]),
            {"canonical_objects", "omitted_term_ids"},
        )
        self.assertNotIn("depend_on", prompt_text)
        self.assertNotIn("compose_with", prompt_text)
        self.assertNotIn("Do not", prompt_text)
        self.assertNotIn("candidate_edges", prompt_text)
        self.assertLess(len(prompt_text), 3500)

    def test_exact_normalized_terms_are_accepted_without_provider(self) -> None:
        provider = StaticCanonicalizationProvider()
        producer = _interface("skill:producer", produces=[_field("skill:producer", "CSV-table")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "csv_table")])

        build = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=provider,
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=NoSimilarityEmbedder(),
        )

        self.assertEqual(provider.calls, [])
        self.assertEqual(build.lookup("skill:producer", "produces", "CSV-table", "artifact"), "artifact:csv_table")
        self.assertEqual(build.lookup("skill:consumer", "requires", "csv_table", "artifact"), "artifact:csv_table")
        self.assertEqual(len(build.objects), 1)
        self.assertFalse(hasattr(build.objects[0], "promoted"))
        self.assertFalse(hasattr(build.objects[0], "reuse_count"))
        self.assertEqual(build.objects[0].produced_by, ["skill:producer"])
        self.assertEqual(build.objects[0].required_by, ["skill:consumer"])

    def test_same_role_exact_terms_are_not_canonicalized_without_consumer(self) -> None:
        build = canonicalize_contract_objects(
            {
                "skill:first": _interface("skill:first", produces=[_field("skill:first", "verification_report", "report")]),
                "skill:second": _interface("skill:second", produces=[_field("skill:second", "verification report", "report")]),
            },
            provider=DeterministicCanonicalizationProvider(),
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=NoSimilarityEmbedder(),
        )

        self.assertEqual(build.objects, [])
        self.assertEqual(build.assignments, [])

    def test_format_aliases_are_not_heuristically_merged(self) -> None:
        provider = OmitCanonicalizationProvider()
        producer = _interface("skill:producer", produces=[_field("skill:producer", "md file")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "markdown file")])

        build = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=provider,
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=FixedSemanticEmbedder(),
        )

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(build.objects, [])
        self.assertEqual(build.assignments, [])
        self.assertEqual(build.lookup("skill:producer", "produces", "md file", "artifact"), "")
        self.assertEqual(build.lookup("skill:consumer", "requires", "markdown file", "artifact"), "")

    def test_isolated_terms_are_not_canonicalized(self) -> None:
        build = canonicalize_contract_objects(
            {
                "skill:producer": _interface(
                    "skill:producer",
                    produces=[_field("skill:producer", "single_use_artifact")],
                )
            },
            provider=DeterministicCanonicalizationProvider(),
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=NoSimilarityEmbedder(),
        )

        self.assertEqual(build.objects, [])
        self.assertEqual(build.assignments, [])
        self.assertEqual(build.raw_terms[0].name, "single_use_artifact")

    def test_pure_generic_interface_names_are_dropped_but_specific_phrases_remain(self) -> None:
        producer = _interface(
            "skill:producer",
            produces=[
                _field("skill:producer", "output"),
                _field("skill:producer", "command_result"),
                _field("skill:producer", "project_directory_path"),
            ],
        )
        consumer = _interface(
            "skill:consumer",
            requires=[
                _field("skill:consumer", "files"),
                _field("skill:consumer", "command result"),
                _field("skill:consumer", "project directory path"),
            ],
        )

        build = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=DeterministicCanonicalizationProvider(),
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=NoSimilarityEmbedder(),
        )

        self.assertEqual(
            {term.name for term in build.raw_terms},
            {"command_result", "project_directory_path", "command result", "project directory path"},
        )
        self.assertEqual(build.lookup("skill:producer", "produces", "output", "artifact"), "")
        self.assertEqual(build.lookup("skill:consumer", "requires", "files", "artifact"), "")
        self.assertEqual(
            build.lookup("skill:producer", "produces", "command_result", "artifact"),
            "artifact:command_result",
        )
        self.assertEqual(
            build.lookup("skill:consumer", "requires", "command result", "artifact"),
            "artifact:command_result",
        )

    def test_deterministic_provider_skips_semantic_embedding(self) -> None:
        embedder = CountingSemanticEmbedder()
        producer = _interface("skill:producer", produces=[_field("skill:producer", "spreadsheet export")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "worksheet rows", "data")])

        build = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=DeterministicCanonicalizationProvider(),
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=embedder,
        )

        self.assertEqual(embedder.calls, [])
        self.assertEqual(build.objects, [])
        self.assertEqual(build.assignments, [])

    def test_semantic_candidates_require_provider_acceptance(self) -> None:
        provider = StaticCanonicalizationProvider()
        producer = _interface("skill:producer", produces=[_field("skill:producer", "spreadsheet export")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "worksheet rows", "data")])

        build = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=provider,
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=FixedSemanticEmbedder(),
        )

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(build.objects[0].canonical_id, "data:spreadsheet_table")
        self.assertEqual(build.lookup("skill:producer", "produces", "spreadsheet export", "artifact"), "data:spreadsheet_table")
        self.assertEqual(build.lookup("skill:consumer", "requires", "worksheet rows", "data"), "data:spreadsheet_table")

    def test_provider_singleton_splits_are_ignored(self) -> None:
        producer = _interface("skill:producer", produces=[_field("skill:producer", "target object", "data")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "target receptacle object", "data")])

        build = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=SplitSingletonCanonicalizationProvider(),
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=FixedSemanticEmbedder(),
        )

        self.assertEqual(build.objects, [])
        self.assertEqual(build.assignments, [])

    def test_provider_type_mismatch_is_ignored(self) -> None:
        producer = _interface("skill:producer", produces=[_field("skill:producer", "markdown report", "report")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "md report", "artifact")])

        build = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=WrongTypeCanonicalizationProvider(),
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=FixedSemanticEmbedder(),
        )

        self.assertEqual(build.objects, [])
        self.assertEqual(build.assignments, [])

    def test_kind_mismatch_is_sent_to_provider_when_text_matches(self) -> None:
        provider = StaticCanonicalizationProvider()
        producer = _interface("skill:producer", produces=[_field("skill:producer", "csv table", "artifact")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "csv table", "data")])

        build = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=provider,
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=NoSimilarityEmbedder(),
        )

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(build.objects[0].canonical_id, "data:spreadsheet_table")

    def test_omitted_terms_do_not_generate_execution_candidates(self) -> None:
        provider = OmitCanonicalizationProvider()
        producer = _interface("skill:producer", produces=[_field("skill:producer", "spreadsheet export")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "worksheet rows")])
        canonicalization = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=provider,
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=FixedSemanticEmbedder(),
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            canonicalization=canonicalization,
            bucket_limit=100,
        )

        self.assertEqual(canonicalization.objects, [])
        self.assertEqual(canonicalization.assignments, [])
        self.assertEqual(compiled.candidates, [])

    def test_accepted_canonicalization_prepares_downstream_execution_candidates(self) -> None:
        producer = _interface("skill:producer", produces=[_field("skill:producer", "spreadsheet export")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "worksheet rows", "data")])
        canonicalization = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=StaticCanonicalizationProvider(),
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=FixedSemanticEmbedder(),
        )

        compiled = compile_execution_graph(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            canonicalization=canonicalization,
            bucket_limit=100,
        )

        self.assertEqual(len(compiled.candidates), 1)
        self.assertEqual(compiled.candidates[0].source_skill, "skill:producer")
        self.assertEqual(compiled.candidates[0].target_skill, "skill:consumer")
        self.assertEqual(compiled.candidates[0].metadata["canonical_object_id"], "data:spreadsheet_table")

    def test_semantic_embedding_cache_is_keyed_per_text(self) -> None:
        first_terms = [
            RawContractObject("skill:a", "produces", "alpha artifact", "artifact"),
            RawContractObject("skill:b", "requires", "beta artifact", "artifact"),
        ]
        second_terms = [
            *first_terms,
            RawContractObject("skill:c", "requires", "gamma artifact", "artifact"),
        ]

        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "canonical_embeddings.json"
            embedder = CountingSemanticEmbedder()
            generate_semantic_candidate_pairs(
                first_terms,
                embedder=embedder,
                threshold=0.0,
                top_k=2,
                cache_path=cache_path,
            )
            self.assertEqual([len(call) for call in embedder.calls], [2])

            generate_semantic_candidate_pairs(
                second_terms,
                embedder=embedder,
                threshold=0.0,
                top_k=2,
                cache_path=cache_path,
            )
            self.assertEqual([len(call) for call in embedder.calls], [2, 1])

    def test_embedding_provider_adapter_uses_shared_embed_many_interface(self) -> None:
        provider = EmbedManyOnlyProvider()
        embedder = EmbeddingProviderCanonicalEmbedder(provider)

        vectors = embedder.embed_texts(["alpha", "beta"])

        self.assertEqual(provider.calls, [["alpha", "beta"]])
        self.assertEqual(vectors, [[1.0, 0.0], [1.0, 0.0]])

    def test_candidate_groups_do_not_create_full_pair_audit_structures(self) -> None:
        terms = [
            RawContractObject("skill:a", "produces", "spreadsheet export", "artifact"),
            RawContractObject("skill:b", "requires", "worksheet rows", "data"),
            RawContractObject("skill:c", "requires", "clinical protocol", "artifact"),
        ]

        groups = candidate_groups_from_terms(
            terms,
            semantic_embedder=FixedSemanticEmbedder(),
            semantic_threshold=0.9,
            semantic_top_k=2,
        )

        self.assertEqual(len(groups), 1)
        self.assertTrue(all(not hasattr(group, "candidate_edges") for group in groups))
        semantic_group = next(group for group in groups if len(group.terms) == 2)
        self.assertEqual({term.name for term in semantic_group.terms}, {"spreadsheet export", "worksheet rows"})

    def test_litellm_canonicalization_provider_uses_config_max_tokens(self) -> None:
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
                                    "canonical_objects": [],
                                    "omitted_term_ids": [],
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
            provider = LiteLLMCanonicalizationProvider(
                LLMConfig(
                    api_base="https://example.test/api",
                    api_key="sk-test",
                    model="openai/test-model",
                    max_tokens=32768,
                )
            )
            provider.canonicalize(
                CanonicalizationCluster(
                    cluster_id="canonical-group:csv",
                    terms=[RawContractObject("skill:producer", "produces", "csv", "artifact")],
                )
            )
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

        self.assertEqual(calls[0]["max_tokens"], 32768)

    def test_llm_canonicalization_uses_cache_and_falls_back_to_omit_on_failure(self) -> None:
        producer = _interface("skill:producer", produces=[_field("skill:producer", "spreadsheet export")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "worksheet rows", "data")])
        provider = StaticCanonicalizationProvider()

        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "canonicalization_cache.json"
            first = canonicalize_contract_objects(
                {producer.skill_id: producer, consumer.skill_id: consumer},
                provider=provider,
                cache_path=cache_path,
                job_options=LLMJobOptions(progress_every=0),
                semantic_embedder=FixedSemanticEmbedder(),
            )
            second = canonicalize_contract_objects(
                {producer.skill_id: producer, consumer.skill_id: consumer},
                provider=provider,
                cache_path=cache_path,
                job_options=LLMJobOptions(progress_every=0),
                semantic_embedder=FixedSemanticEmbedder(),
            )

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(first.objects[0].canonical_id, second.objects[0].canonical_id)

        fallback = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=StaticCanonicalizationProvider(fail=True),
            job_options=LLMJobOptions(progress_every=0, max_retries=0),
            semantic_embedder=FixedSemanticEmbedder(),
        )
        self.assertEqual(fallback.objects, [])
        self.assertEqual(fallback.assignments, [])


if __name__ == "__main__":
    unittest.main()
