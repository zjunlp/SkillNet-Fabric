from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.compiled_graph.canonicalization.candidates import (
    CanonicalCandidateEdge,
    generate_lexical_candidates,
    generate_semantic_candidates,
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


def _field(skill_id: str, name: str, kind: str = "artifact") -> InterfaceField:
    return InterfaceField(
        name=name,
        kind=kind,
        confidence=0.9,
        evidence=[InterfaceEvidence(skill=skill_id, line=1, text=f"{skill_id} mentions {name}.")],
    )


class StaticCanonicalizationProvider:
    model_id = "static-canonicalizer"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def canonicalize(self, cluster):
        self.calls += 1
        if self.fail:
            raise RuntimeError("canonicalizer unavailable")
        return {
            "canonical_objects": [
                {
                    "canonical_name": "spreadsheet_table",
                    "type": "artifact",
                    "description": "Tabular spreadsheet data.",
                    "aliases": [term.name for term in cluster.terms],
                    "promoted": True,
                    "confidence": 0.88,
                    "reason": "The cluster describes reusable tabular spreadsheet data.",
                }
            ],
            "assignments": [
                {
                    "raw_name": term.name,
                    "canonical_name": "spreadsheet_table",
                    "confidence": 0.86,
                    "reason": "Same reusable spreadsheet object.",
                }
                for term in cluster.terms
            ],
        }


class StaticInventoryStateCanonicalizationProvider:
    model_id = "static-inventory-state-canonicalizer"

    def canonicalize(self, cluster):
        return {
            "canonical_objects": [
                {
                    "canonical_name": "object_in_inventory",
                    "type": "world_state",
                    "description": "Physical object is held in the agent inventory.",
                    "aliases": [term.name for term in cluster.terms],
                    "promoted": True,
                    "confidence": 0.99,
                    "reason": "Provider attempts to merge all inventory-like terms.",
                }
            ],
            "assignments": [
                {
                    "raw_name": term.name,
                    "canonical_name": "object_in_inventory",
                    "confidence": 0.99,
                    "reason": "Provider attempts to merge all inventory-like terms.",
                }
                for term in cluster.terms
            ],
        }


class FixedSemanticEmbedder:
    model_id = "fixed-semantic-test"

    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if (
                "spreadsheet export" in lowered
                or "worksheet rows" in lowered
                or "formatted_excel_report" in lowered
                or "workbook_or_tabular_data" in lowered
            ):
                vectors.append([1.0, 0.0, 0.0])
            elif "csv table" in lowered or "csv tables" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class NoSimilarityEmbedder:
    model_id = "no-similarity-test"

    def embed_texts(self, texts):
        vectors = []
        for index, _text in enumerate(texts):
            vector = [0.0] * max(len(texts), 1)
            vector[index] = 1.0
            vectors.append(vector)
        return vectors


class CanonicalizationTests(unittest.TestCase):
    def test_canonicalization_prompt_defines_operational_decision_workflow(self) -> None:
        cluster = CanonicalizationCluster(
            cluster_id="artifact:reports",
            object_type="artifact",
            terms=[
                RawContractObject("skill:report", "produces", "report", "artifact"),
                RawContractObject("skill:chart", "produces", "chart", "artifact"),
            ],
            ambiguous=True,
        )

        messages = build_canonicalization_messages(cluster)
        payload = json.loads(messages[1]["content"])
        prompt_text = json.dumps(messages, ensure_ascii=False)

        self.assertEqual(payload["prompt_id"], "canonicalization_operational_objects")
        for field in ("todo", "input", "output", "workflow", "rules", "constraints"):
            self.assertIn(field, payload)
        self.assertIn("decision_workflow", payload)
        self.assertEqual(
            set(payload["output_schema"]),
            {"canonical_objects", "assignments"},
        )
        self.assertIn("Do not merge terms that merely belong to the same broad domain", prompt_text)
        self.assertIn("example, placeholder, section heading", prompt_text)

    def test_candidate_edge_model_serializes_evidence_fields(self) -> None:
        edge = CanonicalCandidateEdge(
            left_object_id="left",
            right_object_id="right",
            left_text="CSV table",
            right_text="csv tables",
            object_type="artifact",
            method="lexical",
            score=0.91,
            features={"token_set_ratio": 0.91},
        )

        payload = edge.to_dict()

        self.assertEqual(payload["left_object_id"], "left")
        self.assertEqual(payload["method"], "lexical")
        self.assertEqual(payload["score"], 0.91)
        self.assertEqual(payload["features"]["token_set_ratio"], 0.91)

    def test_lexical_candidates_use_rapidfuzz_without_fixed_artifact_ontology(self) -> None:
        terms = [
            RawContractObject("skill:a", "produces", "CSV table", "artifact"),
            RawContractObject("skill:b", "requires", "csv tables", "artifact"),
            RawContractObject("skill:c", "requires", "PowerPoint slides", "artifact"),
        ]

        edges = generate_lexical_candidates(terms, threshold=0.8)

        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].method, "lexical")
        self.assertEqual({edges[0].left_text, edges[0].right_text}, {"CSV table", "csv tables"})
        self.assertNotIn("presentation_document", json.dumps([edge.to_dict() for edge in edges]))

    def test_semantic_candidates_use_exact_cosine_nearest_neighbors(self) -> None:
        terms = [
            RawContractObject("skill:a", "produces", "spreadsheet export", "artifact"),
            RawContractObject("skill:b", "requires", "worksheet rows", "artifact"),
            RawContractObject("skill:c", "requires", "clinical protocol", "artifact"),
        ]

        edges = generate_semantic_candidates(
            terms,
            embedder=FixedSemanticEmbedder(),
            threshold=0.9,
            top_k=2,
        )

        pairs = {frozenset([edge.left_text, edge.right_text]) for edge in edges}
        self.assertIn(frozenset(["spreadsheet export", "worksheet rows"]), pairs)
        self.assertTrue(all(edge.method == "semantic" for edge in edges))

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
                                    "assignments": [],
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
                    cluster_id="artifact:csv",
                    object_type="artifact",
                    terms=[
                        RawContractObject(
                            skill_id="skill:producer",
                            role="produces",
                            name="csv",
                            kind="artifact",
                        )
                    ],
                )
            )
        finally:
            if original is None:
                sys.modules.pop("litellm", None)
            else:
                sys.modules["litellm"] = original

        self.assertEqual(calls[0]["max_tokens"], 32768)

    def test_deterministic_canonicalizer_uses_candidate_graph_and_drops_exact_generic_terms(self) -> None:
        producer = _interface(
            "skill:producer",
            produces=[_field("skill:producer", "csv table"), _field("skill:producer", "output")],
        )
        consumer = _interface(
            "skill:consumer",
            requires=[_field("skill:consumer", "csv tables")],
        )

        build = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=DeterministicCanonicalizationProvider(),
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=NoSimilarityEmbedder(),
        )

        promoted = {item.canonical_id for item in build.objects if item.promoted}
        self.assertIn("artifact:csv_table", promoted)
        self.assertNotIn("artifact:output", promoted)
        self.assertTrue(build.candidate_edges)
        self.assertTrue(build.candidate_components)
        self.assertEqual(build.lookup("skill:producer", "produces", "csv table", "artifact"), "artifact:csv_table")
        self.assertEqual(build.lookup("skill:consumer", "requires", "csv tables", "artifact"), "artifact:csv_table")
        self.assertFalse(any(item.name == "output" for item in build.raw_terms))
        self.assertFalse(any(item.raw_name == "output" for item in build.assignments))

    def test_singleton_component_is_canonicalized_without_provider_call(self) -> None:
        provider = StaticCanonicalizationProvider()
        producer = _interface(
            "skill:producer",
            produces=[_field("skill:producer", "isolated artifact")],
        )

        build = canonicalize_contract_objects(
            {producer.skill_id: producer},
            provider=provider,
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=NoSimilarityEmbedder(),
        )

        self.assertEqual(provider.calls, 0)
        self.assertEqual(build.assignments[0].canonical_id, "artifact:isolated_artifact")
        self.assertEqual(build.assignments[0].provenance, "deterministic_exact")
        self.assertFalse(build.objects[0].promoted)

    def test_broad_handoff_object_is_not_promoted_by_canonicalization(self) -> None:
        producer = _interface("skill:producer", produces=[_field("skill:producer", "source_data", "data")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "source_data", "data")])

        build = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=DeterministicCanonicalizationProvider(),
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=NoSimilarityEmbedder(),
        )

        source_data = {item.canonical_id: item for item in build.objects}["data:source_data"]
        self.assertFalse(source_data.promoted)
        self.assertEqual(
            build.lookup("skill:producer", "produces", "source_data", "data"),
            "",
        )
        self.assertEqual(
            build.lookup("skill:consumer", "requires", "source_data", "data"),
            "",
        )

    def test_normalized_exact_duplicates_are_canonicalized_without_provider_call(self) -> None:
        provider = StaticCanonicalizationProvider()
        producer = _interface("skill:producer", produces=[_field("skill:producer", "CSV-table")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "csv table")])

        build = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=provider,
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=NoSimilarityEmbedder(),
        )

        self.assertEqual(provider.calls, 0)
        self.assertEqual(build.lookup("skill:producer", "produces", "CSV-table", "artifact"), "artifact:csv_table")
        self.assertEqual(build.lookup("skill:consumer", "requires", "csv table", "artifact"), "artifact:csv_table")
        self.assertEqual(build.objects[0].provenance, "deterministic_exact")

    def test_non_exact_candidate_component_still_uses_provider(self) -> None:
        provider = StaticCanonicalizationProvider()
        producer = _interface("skill:producer", produces=[_field("skill:producer", "spreadsheet export")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "worksheet rows")])

        build = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=provider,
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=FixedSemanticEmbedder(),
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(build.objects[0].canonical_id, "artifact:spreadsheet_table")

    def test_duplicate_raw_names_in_one_cluster_keep_all_skill_roles(self) -> None:
        producer = _interface("skill:producer", produces=[_field("skill:producer", "csv")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "csv")])

        build = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=DeterministicCanonicalizationProvider(),
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=NoSimilarityEmbedder(),
        )

        csv = {item.canonical_id: item for item in build.objects}["artifact:csv"]
        self.assertTrue(csv.promoted)
        self.assertEqual(csv.produced_by, ["skill:producer"])
        self.assertEqual(csv.required_by, ["skill:consumer"])
        self.assertEqual(build.lookup("skill:producer", "produces", "csv", "artifact"), "artifact:csv")
        self.assertEqual(build.lookup("skill:consumer", "requires", "csv", "artifact"), "artifact:csv")

    def test_deterministic_canonicalizer_does_not_use_old_spreadsheet_hint_bucket(self) -> None:
        producer = _interface(
            "skill:financial-analysis",
            produces=[_field("skill:financial-analysis", "formatted_excel_report")],
        )
        consumer = _interface(
            "skill:xlsx",
            requires=[_field("skill:xlsx", "workbook_or_tabular_data")],
        )

        build = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=DeterministicCanonicalizationProvider(),
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=NoSimilarityEmbedder(),
        )

        promoted = {item.canonical_id for item in build.objects if item.promoted}
        self.assertNotIn("artifact:spreadsheet_table", promoted)
        self.assertEqual(
            build.lookup("skill:financial-analysis", "produces", "formatted_excel_report", "artifact"),
            "",
        )

    def test_execution_compiler_uses_pool_level_canonicalization(self) -> None:
        producer = _interface("skill:producer", produces=[_field("skill:producer", "spreadsheet export")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "worksheet rows")])
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
        self.assertEqual(compiled.candidates[0].matched_name, "spreadsheet_table")
        self.assertEqual(compiled.candidates[0].metadata["canonical_object_id"], "artifact:spreadsheet_table")

    def test_candidate_components_split_by_object_type(self) -> None:
        producer = _interface("skill:picker", produces=[_field("skill:picker", "object_in_inventory", "state")])
        consumer = _interface("skill:storer", requires=[_field("skill:storer", "object description note", "text")])
        provider = StaticCanonicalizationProvider()
        canonicalization = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=provider,
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=FixedSemanticEmbedder(),
        )

        self.assertEqual(provider.calls, 0)
        component_types = {item.object_type for item in canonicalization.candidate_components}
        self.assertEqual(component_types, {"state", "text"})

    def test_belief_state_does_not_merge_into_world_state_execution_object(self) -> None:
        planner = _interface(
            "skill:goal-interpreter",
            produces=[_field("skill:goal-interpreter", "object_permanence_state", "belief_state")],
        )
        cleaner = _interface(
            "skill:clean-object",
            requires=[_field("skill:clean-object", "object_in_inventory", "world_state")],
        )

        canonicalization = canonicalize_contract_objects(
            {planner.skill_id: planner, cleaner.skill_id: cleaner},
            provider=StaticInventoryStateCanonicalizationProvider(),
            job_options=LLMJobOptions(progress_every=0),
            semantic_embedder=FixedSemanticEmbedder(),
        )

        inventory = {item.canonical_id: item for item in canonicalization.objects}.get("state:object_in_inventory")
        self.assertIsNotNone(inventory)
        self.assertEqual(inventory.produced_by, [])
        self.assertEqual(inventory.required_by, ["skill:clean-object"])
        self.assertEqual(
            canonicalization.lookup("skill:goal-interpreter", "produces", "object_permanence_state", "belief_state"),
            "",
        )
        self.assertTrue(
            any(
                item.skill_id == "skill:clean-object"
                and item.raw_name == "object_in_inventory"
                and item.canonical_id == "state:object_in_inventory"
                for item in canonicalization.assignments
            )
        )

    def test_llm_canonicalization_uses_cache_and_falls_back_on_failure(self) -> None:
        producer = _interface("skill:producer", produces=[_field("skill:producer", "spreadsheet export")])
        consumer = _interface("skill:consumer", requires=[_field("skill:consumer", "worksheet rows")])
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

        self.assertEqual(provider.calls, 1)
        self.assertEqual(first.objects[0].canonical_id, second.objects[0].canonical_id)

        fallback = canonicalize_contract_objects(
            {producer.skill_id: producer, consumer.skill_id: consumer},
            provider=StaticCanonicalizationProvider(fail=True),
            job_options=LLMJobOptions(progress_every=0, max_retries=0),
            semantic_embedder=FixedSemanticEmbedder(),
        )
        self.assertTrue(fallback.objects)
        self.assertEqual(fallback.objects[0].provenance, "deterministic_fallback")


if __name__ == "__main__":
    unittest.main()
