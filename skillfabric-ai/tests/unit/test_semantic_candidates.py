from __future__ import annotations

import json
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from skillfabric.compiled_graph.contracts.models import SkillContract
from skillfabric.compiled_graph.semantic.candidates import (
    CandidateRetrievalError,
    retrieve_candidate_pairs,
)
from tests.unit.relation_helpers import make_skill


@dataclass
class SemanticEmbeddingProvider:
    model_id: str = "semantic-test-model"
    dimension: int = 3
    calls: list[list[str]] = field(default_factory=list)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        if "normalized_table" in lowered:
            return [1.0, 0.0, 0.0]
        if "pdf" in lowered:
            return [0.95, 0.05, 0.0]
        if "testing" in lowered or "ci" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def _contract(
    skill,
    *,
    capability: str,
    requires: list[str] | None = None,
    produces: list[str] | None = None,
) -> SkillContract:
    def fields(names: list[str]) -> list[dict[str, object]]:
        return [
            {
                "name": name,
                "description": f"Contract field {name}.",
                "evidence": [{"line": 1}],
            }
            for name in names
        ]

    return SkillContract.from_extraction(
        skill,
        {
            "capability": capability,
            "when_to_use": f"Use {skill.name} for its documented task.",
            "requires": fields(requires or []),
            "produces": fields(produces or []),
            "tools": [],
            "evidence": [{"line": 1}],
        },
    )


def test_handoff_ann_queries_produces_against_requires() -> None:
    producer = make_skill(
        "skill:pdf-parser", "pdf-parser", "Produces normalized_table from PDF files."
    )
    consumer = make_skill("skill:report", "report", "Requires normalized_table to write a report.")
    unrelated = make_skill("skill:testing", "testing", "Runs testing for Python projects.")
    contracts = {
        producer.id: _contract(
            producer,
            capability="Parse PDF tables.",
            produces=["normalized_table"],
        ),
        consumer.id: _contract(
            consumer,
            capability="Write a report.",
            requires=["normalized_table"],
        ),
        unrelated.id: _contract(unrelated, capability="Run testing."),
    }

    result = retrieve_candidate_pairs(
        contracts,
        [producer, consumer, unrelated],
        provider=SemanticEmbeddingProvider(),
        handoff_top_k=1,
        similarity_top_k=0,
    )

    pair = next(item for item in result.pairs if item.key == (producer.id, consumer.id))
    assert pair.channels == ("handoff",)
    assert pair.hits[0].query_skill == producer.id
    assert pair.hits[0].matched_skill == consumer.id
    assert pair.hits[0].query_field == "produces:normalized_table"
    assert pair.hits[0].matched_field == "requires:normalized_table"


def test_handoff_top_k_counts_distinct_candidate_skills() -> None:
    producer = make_skill("skill:producer", "producer", "Produces shared_output.")
    repeated = make_skill("skill:repeated", "repeated", "Requires two forms of shared_input.")
    distinct = make_skill("skill:distinct", "distinct", "Requires another shared_input.")
    contracts = {
        producer.id: _contract(
            producer,
            capability="Produce shared data.",
            produces=["shared_output"],
        ),
        repeated.id: _contract(
            repeated,
            capability="Consume shared data in two forms.",
            requires=["shared_input_primary", "shared_input_secondary"],
        ),
        distinct.id: _contract(
            distinct,
            capability="Consume shared data independently.",
            requires=["shared_input_distinct"],
        ),
    }

    class DuplicateFieldProvider(SemanticEmbeddingProvider):
        @staticmethod
        def _vector(text: str) -> list[float]:
            lowered = text.lower()
            if "shared_output" in lowered or "shared_input_primary" in lowered:
                return [1.0, 0.0, 0.0]
            if "shared_input_secondary" in lowered:
                return [0.99, 0.01, 0.0]
            if "shared_input_distinct" in lowered:
                return [0.98, 0.02, 0.0]
            return [0.0, 0.0, 1.0]

    result = retrieve_candidate_pairs(
        contracts,
        [producer, repeated, distinct],
        provider=DuplicateFieldProvider(),
        handoff_top_k=2,
        similarity_top_k=0,
    )

    handoff_pairs = {pair.key for pair in result.pairs if "handoff" in pair.channels}
    assert handoff_pairs == {
        tuple(sorted((producer.id, repeated.id))),
        tuple(sorted((producer.id, distinct.id))),
    }


def test_exact_handoff_fields_are_not_dropped_by_ann_top_k() -> None:
    producer = make_skill("skill:producer", "producer", "Produces normalized_table.")
    consumers = [
        make_skill(
            f"skill:consumer-{index}",
            f"consumer-{index}",
            "Requires normalized_table.",
        )
        for index in range(3)
    ]
    contracts = {
        producer.id: _contract(
            producer,
            capability="Produce normalized tables.",
            produces=["normalized_table"],
        ),
        **{
            consumer.id: _contract(
                consumer,
                capability="Consume normalized tables.",
                requires=["normalized_table"],
            )
            for consumer in consumers
        },
    }

    result = retrieve_candidate_pairs(
        contracts,
        [producer, *consumers],
        provider=SemanticEmbeddingProvider(),
        handoff_top_k=1,
        similarity_top_k=0,
    )

    assert {pair.key for pair in result.pairs if "handoff" in pair.channels} == {
        tuple(sorted((producer.id, consumer.id))) for consumer in consumers
    }


def test_exact_handoff_ignores_outer_markdown_code_quotes() -> None:
    producer = make_skill(
        "skill:quoted-producer",
        "quoted-producer",
        "Produces `normalized_table`.",
    )
    consumers = [
        make_skill(
            f"skill:plain-consumer-{index}",
            f"plain-consumer-{index}",
            "Requires normalized_table.",
        )
        for index in range(3)
    ]
    contracts = {
        producer.id: _contract(
            producer,
            capability="Produce normalized tables.",
            produces=["`normalized_table`"],
        ),
        **{
            consumer.id: _contract(
                consumer,
                capability="Consume normalized tables.",
                requires=["normalized_table"],
            )
            for consumer in consumers
        },
    }

    result = retrieve_candidate_pairs(
        contracts,
        [producer, *consumers],
        provider=SemanticEmbeddingProvider(),
        handoff_top_k=1,
        similarity_top_k=0,
    )

    assert {pair.key for pair in result.pairs if "handoff" in pair.channels} == {
        tuple(sorted((producer.id, consumer.id))) for consumer in consumers
    }


@pytest.mark.parametrize("field", ["handoff_top_k", "similarity_top_k"])
def test_candidate_retrieval_rejects_invalid_top_k(field) -> None:
    skill = make_skill("skill:only", "only", "Only skill.")
    contracts = {skill.id: _contract(skill, capability="Only capability.")}
    options = {"handoff_top_k": 0, "similarity_top_k": 0, field: -1}

    with pytest.raises(CandidateRetrievalError, match=field):
        retrieve_candidate_pairs(
            contracts,
            [skill],
            provider=SemanticEmbeddingProvider(),
            **options,
        )


def test_similarity_ann_is_only_candidate_evidence() -> None:
    parser = make_skill("skill:pdf-parser", "pdf-parser", "Extract PDF tables.")
    extractor = make_skill("skill:pdf-extractor", "pdf-extractor", "Read PDF table data.")
    contracts = {
        parser.id: _contract(parser, capability="Parse PDF tables."),
        extractor.id: _contract(extractor, capability="Extract PDF tables."),
    }

    result = retrieve_candidate_pairs(
        contracts,
        [parser, extractor],
        provider=SemanticEmbeddingProvider(),
        handoff_top_k=0,
        similarity_top_k=1,
    )

    assert len(result.pairs) == 1
    assert result.pairs[0].channels == ("similarity",)
    assert not hasattr(result.pairs[0], "relation")
    assert not hasattr(result.pairs[0], "edge")


def test_explicit_reference_is_retrieved_without_ann_neighbors() -> None:
    analyze = make_skill("skill:analyze-ci", "analyze-ci", "Inspect failed CI logs.")
    testing = make_skill(
        "skill:testing-python",
        "testing-python",
        "Use `analyze-ci` before rerunning the Python test suite.",
    )
    contracts = {
        analyze.id: _contract(analyze, capability="Analyze CI failures."),
        testing.id: _contract(testing, capability="Run Python testing."),
    }

    result = retrieve_candidate_pairs(
        contracts,
        [analyze, testing],
        provider=SemanticEmbeddingProvider(),
        handoff_top_k=0,
        similarity_top_k=0,
    )

    assert len(result.pairs) == 1
    assert result.pairs[0].channels == ("explicit_reference",)
    assert result.pairs[0].hits[0].evidence[0].text == testing.raw_text


def test_plain_prose_skill_name_is_not_an_explicit_reference() -> None:
    architecture = make_skill(
        "skill:architecture",
        "architecture",
        "Design software architecture.",
    )
    designer = make_skill(
        "skill:architecture-designer",
        "architecture-designer",
        "Review the existing system architecture before proposing changes.",
    )
    contracts = {
        architecture.id: _contract(architecture, capability="Design software systems."),
        designer.id: _contract(designer, capability="Review software designs."),
    }

    result = retrieve_candidate_pairs(
        contracts,
        [architecture, designer],
        provider=SemanticEmbeddingProvider(),
        handoff_top_k=0,
        similarity_top_k=0,
    )

    assert result.pairs == ()


def test_candidate_artifact_omits_derived_channels_and_similarity_scores() -> None:
    parser = make_skill("skill:pdf-parser", "pdf-parser", "Extract PDF tables.")
    extractor = make_skill("skill:pdf-extractor", "pdf-extractor", "Read PDF table data.")
    contracts = {
        parser.id: _contract(parser, capability="Parse PDF tables."),
        extractor.id: _contract(extractor, capability="Extract PDF tables."),
    }

    pair = retrieve_candidate_pairs(
        contracts,
        [parser, extractor],
        provider=SemanticEmbeddingProvider(),
        handoff_top_k=0,
        similarity_top_k=1,
    ).pairs[0]
    payload = pair.to_dict()

    assert set(payload) == {"skill_a", "skill_b", "hits"}
    assert "score" not in payload["hits"][0]


def test_channels_merge_into_one_unordered_pair() -> None:
    producer = make_skill(
        "skill:pdf-parser",
        "pdf-parser",
        "Produces normalized_table and references `report`.",
    )
    consumer = make_skill("skill:report", "report", "Requires normalized_table for a PDF report.")
    contracts = {
        producer.id: _contract(
            producer,
            capability="Parse PDF tables.",
            produces=["normalized_table"],
        ),
        consumer.id: _contract(
            consumer,
            capability="Write PDF reports.",
            requires=["normalized_table"],
        ),
    }

    result = retrieve_candidate_pairs(
        contracts,
        [producer, consumer],
        provider=SemanticEmbeddingProvider(),
        handoff_top_k=1,
        similarity_top_k=1,
    )

    assert len(result.pairs) == 1
    assert result.pairs[0].channels == ("handoff", "explicit_reference", "similarity")
    assert {hit.channel for hit in result.pairs[0].hits} == {
        "handoff",
        "explicit_reference",
        "similarity",
    }


def test_embedding_store_is_reused_by_text_hash(tmp_path) -> None:
    skill = make_skill("skill:pdf-parser", "pdf-parser", "Extract PDF tables.")
    contracts = {skill.id: _contract(skill, capability="Parse PDF tables.")}
    provider = SemanticEmbeddingProvider()
    store = tmp_path / "embeddings.json"

    first = retrieve_candidate_pairs(
        contracts,
        [skill],
        provider=provider,
        store_path=store,
        handoff_top_k=0,
        similarity_top_k=0,
    )
    calls_after_first = len(provider.calls)
    second = retrieve_candidate_pairs(
        contracts,
        [skill],
        provider=provider,
        store_path=store,
        handoff_top_k=0,
        similarity_top_k=0,
    )

    assert calls_after_first == 1
    assert len(provider.calls) == calls_after_first
    assert first.metrics["embedded_record_count"] == 1
    assert second.metrics["cache_hit_count"] == 1
    assert not hasattr(second, "embeddings")
    payload = json.loads(store.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2.0"
    assert payload["records"][0]["kind"] == "skill"
    assert "content_hash" not in payload["records"][0]


def test_all_embedding_cache_hits_do_not_rewrite_the_store(tmp_path) -> None:
    skill = make_skill("skill:pdf-parser", "pdf-parser", "Extract PDF tables.")
    contracts = {skill.id: _contract(skill, capability="Parse PDF tables.")}
    provider = SemanticEmbeddingProvider()
    store = tmp_path / "embeddings.json"
    retrieve_candidate_pairs(
        contracts,
        [skill],
        provider=provider,
        store_path=store,
        handoff_top_k=0,
        similarity_top_k=0,
    )

    with patch("skillfabric.compiled_graph.semantic.candidates.atomic_write_text") as write:
        retrieve_candidate_pairs(
            contracts,
            [skill],
            provider=provider,
            store_path=store,
            handoff_top_k=0,
            similarity_top_k=0,
        )

    write.assert_not_called()


def test_embedding_cache_is_invalidated_when_provider_dimension_changes(tmp_path) -> None:
    skill = make_skill("skill:pdf-parser", "pdf-parser", "Extract PDF tables.")
    contracts = {skill.id: _contract(skill, capability="Parse PDF tables.")}
    store = tmp_path / "embeddings.json"
    first_provider = SemanticEmbeddingProvider(model_id="stable-model", dimension=3)
    retrieve_candidate_pairs(
        contracts,
        [skill],
        provider=first_provider,
        store_path=store,
        handoff_top_k=0,
        similarity_top_k=0,
    )

    @dataclass
    class TwoDimensionalProvider(SemanticEmbeddingProvider):
        model_id: str = "stable-model"
        dimension: int = 2

        @staticmethod
        def _vector(_text: str) -> list[float]:
            return [1.0, 0.0]

    second_provider = TwoDimensionalProvider()
    result = retrieve_candidate_pairs(
        contracts,
        [skill],
        provider=second_provider,
        store_path=store,
        handoff_top_k=0,
        similarity_top_k=0,
    )

    assert second_provider.calls
    assert result.metrics["new_embedding_count"] == 1
    assert json.loads(store.read_text(encoding="utf-8"))["dimension"] == 2


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda payload: payload["records"].append(dict(payload["records"][0])), "duplicate key"),
        (lambda payload: payload.update({"dimension": 999}), "dimension"),
        (lambda payload: payload["records"][0].update({"vector": ["1", 0, 0]}), "vector"),
    ],
)
def test_embedding_cache_rejects_inconsistent_schema(tmp_path, mutate, match) -> None:
    skill = make_skill("skill:pdf-parser", "pdf-parser", "Extract PDF tables.")
    contracts = {skill.id: _contract(skill, capability="Parse PDF tables.")}
    provider = SemanticEmbeddingProvider()
    store = tmp_path / "embeddings.json"
    retrieve_candidate_pairs(
        contracts,
        [skill],
        provider=provider,
        store_path=store,
        handoff_top_k=0,
        similarity_top_k=0,
    )
    payload = json.loads(store.read_text(encoding="utf-8"))
    mutate(payload)
    store.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CandidateRetrievalError, match=match):
        retrieve_candidate_pairs(
            contracts,
            [skill],
            provider=provider,
            store_path=store,
            handoff_top_k=0,
            similarity_top_k=0,
        )


def test_empty_vectors_fail_instead_of_using_brute_force() -> None:
    skill = make_skill("skill:empty", "empty", "Empty vector source.")
    contracts = {skill.id: _contract(skill, capability="Empty vector capability.")}

    class EmptyProvider(SemanticEmbeddingProvider):
        def embed_many(self, texts: list[str]) -> list[list[float]]:
            return [[] for _ in texts]

    with pytest.raises(CandidateRetrievalError, match="non-empty"):
        retrieve_candidate_pairs(
            contracts,
            [skill],
            provider=EmptyProvider(),
            handoff_top_k=0,
            similarity_top_k=0,
        )


def test_zero_norm_vectors_fail_instead_of_creating_arbitrary_candidates() -> None:
    skill = make_skill("skill:zero", "zero", "Zero vector source.")
    contracts = {skill.id: _contract(skill, capability="Zero vector capability.")}

    class ZeroProvider(SemanticEmbeddingProvider):
        def embed_many(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 0.0, 0.0] for _ in texts]

    with pytest.raises(CandidateRetrievalError, match="non-zero norm"):
        retrieve_candidate_pairs(
            contracts,
            [skill],
            provider=ZeroProvider(),
            handoff_top_k=0,
            similarity_top_k=0,
        )


def test_missing_contract_is_an_explicit_error() -> None:
    skill = make_skill("skill:missing", "missing", "No contract.")

    with pytest.raises(CandidateRetrievalError, match="contract ids"):
        retrieve_candidate_pairs(
            {},
            [skill],
            provider=SemanticEmbeddingProvider(),
        )
