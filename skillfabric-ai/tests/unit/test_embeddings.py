from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import skillfabric.indexing.embeddings as embedding_module
from skillfabric.indexing.embeddings import (
    DEFAULT_EMBEDDING_MODEL_ID,
    ApiEmbeddingProvider,
    default_embedding_provider,
    embedding_provider_for_model,
    load_skill_embedding_store,
)
from skillfabric.registry.models import SkillNode
from skillfabric.router.retrieval import retrieve_seed_candidates
from skillfabric.storage import Workspace


class QueryProvider:
    model_id = DEFAULT_EMBEDDING_MODEL_ID
    dimension = 2

    def __init__(self) -> None:
        self.embed_calls: list[str] = []
        self.query_calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        return [0.0, 1.0]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [1.0, 0.0]


def test_api_provider_uses_litellm_embedding() -> None:
    provider = ApiEmbeddingProvider(
        dimension=2,
        api_key="test-key",
        api_base="https://example.test/v1",
    )
    calls: list[dict[str, object]] = []

    def fake_embedding(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"data": [{"embedding": [0.25, 0.75]}]}

    fake_litellm = type("FakeLiteLLM", (), {"embedding": staticmethod(fake_embedding)})
    with patch.dict("sys.modules", {"litellm": fake_litellm}):
        vector = provider.embed("find pdf table parser")

    assert vector == [0.25, 0.75]
    assert calls[0]["model"] == DEFAULT_EMBEDDING_MODEL_ID
    assert calls[0]["input"] == ["find pdf table parser"]
    assert calls[0]["api_key"] == "test-key"
    assert calls[0]["api_base"] == "https://example.test/v1"


def test_api_provider_rejects_a_response_with_the_wrong_dimension() -> None:
    provider = ApiEmbeddingProvider(dimension=3)
    fake_litellm = type(
        "FakeLiteLLM",
        (),
        {"embedding": staticmethod(lambda **_kwargs: {"data": [{"embedding": [0.2, 0.8]}]})},
    )

    with (
        patch.dict("sys.modules", {"litellm": fake_litellm}),
        pytest.raises(
            RuntimeError,
            match="dimension",
        ),
    ):
        provider.embed("dimension mismatch")


@pytest.mark.parametrize("vector", [[0.0, 0.0], [float("nan"), 1.0]])
def test_api_provider_rejects_invalid_vector_values(vector) -> None:
    provider = ApiEmbeddingProvider(dimension=2)
    fake_litellm = type(
        "FakeLiteLLM",
        (),
        {"embedding": staticmethod(lambda **_kwargs: {"data": [{"embedding": vector}]})},
    )

    with (
        patch.dict("sys.modules", {"litellm": fake_litellm}),
        pytest.raises(
            RuntimeError,
            match="finite and non-zero",
        ),
    ):
        provider.embed("invalid vector")


def test_api_provider_rejects_duplicate_response_indexes() -> None:
    provider = ApiEmbeddingProvider(dimension=2)
    fake_litellm = type(
        "FakeLiteLLM",
        (),
        {
            "embedding": staticmethod(
                lambda **_kwargs: {
                    "data": [
                        {"index": 0, "embedding": [1.0, 0.0]},
                        {"index": 0, "embedding": [0.0, 1.0]},
                    ]
                }
            )
        },
    )

    with (
        patch.dict("sys.modules", {"litellm": fake_litellm}),
        pytest.raises(
            RuntimeError,
            match="indexes",
        ),
    ):
        provider.embed_many(["first", "second"])


def test_embedding_specific_env_overrides_shared_api_env(tmp_path) -> None:
    env_path = tmp_path / ".env.test"
    env_path.write_text(
        "API_KEY=shared-test-key\n"
        "BASE_URL=https://shared.example/v1\n"
        "EMBEDDING_API_KEY=embedding-test-key\n"
        "EMBEDDING_BASE_URL=https://embedding.example/v1\n"
        "EMBEDDING_MODEL=openai/custom-embedding\n",
        encoding="utf-8",
    )

    provider = ApiEmbeddingProvider.from_env(env_path=env_path)

    assert provider.api_key == "embedding-test-key"
    assert provider.api_base == "https://embedding.example/v1"
    assert provider.model_id == "openai/custom-embedding"


def test_embedding_runtime_limits_are_loaded_from_the_selected_env_file(tmp_path) -> None:
    env_path = tmp_path / ".env.test"
    env_path.write_text(
        "EMBEDDING_BATCH_SIZE=7\nEMBEDDING_TEXT_CHARS=2500\nEMBEDDING_TIMEOUT=45\n",
        encoding="utf-8",
    )

    provider = ApiEmbeddingProvider.from_env(env_path=env_path)

    assert provider.batch_size == 7
    assert provider.max_text_chars == 2500
    assert provider.timeout == 45


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_id": ""},
        {"dimension": 0},
        {"dimension": True},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"batch_size": 0},
        {"max_text_chars": -1},
    ],
)
def test_api_provider_rejects_invalid_runtime_configuration(overrides) -> None:
    with pytest.raises(ValueError):
        ApiEmbeddingProvider(**overrides)


def test_api_provider_rejects_invalid_numeric_env_instead_of_using_defaults(tmp_path) -> None:
    env_path = tmp_path / ".env.test"
    env_path.write_text("EMBEDDING_BATCH_SIZE=not-an-integer\n", encoding="utf-8")

    with pytest.raises(ValueError, match="EMBEDDING_BATCH_SIZE"):
        ApiEmbeddingProvider.from_env(env_path=env_path)


def test_default_provider_rejects_unknown_provider(tmp_path) -> None:
    env_path = tmp_path / ".env.test"
    env_path.write_text("EMBEDDING_PROVIDER=custom-provider\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported embedding provider"):
        default_embedding_provider(env_path=env_path)


def test_provider_for_store_model_preserves_dimension() -> None:
    provider = embedding_provider_for_model("custom/openai-compatible-embedding", dimension=32)

    assert isinstance(provider, ApiEmbeddingProvider)
    assert provider.model_id == "custom/openai-compatible-embedding"
    assert provider.dimension == 32


def test_schema_v2_loader_returns_only_skill_document_vectors(tmp_path) -> None:
    path = tmp_path / "embeddings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "model_id": "embedding-test-model",
                "dimension": 2,
                "records": [
                    _record("skill:skill:alpha", "skill:alpha", "skill", [1.0, 0.0]),
                    _record("requires:skill:alpha:0", "skill:alpha", "requires", [0.0, 1.0]),
                ],
            }
        ),
        encoding="utf-8",
    )

    store = load_skill_embedding_store(path)

    assert store.model_id == "embedding-test-model"
    assert store.dimension == 2
    assert store.vectors == {"skill:alpha": [1.0, 0.0]}


def test_schema_v2_loader_rejects_duplicate_skill_records(tmp_path) -> None:
    path = tmp_path / "embeddings.json"
    row = _record("skill:skill:alpha", "skill:alpha", "skill", [1.0, 0.0])
    path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "model_id": "embedding-test-model",
                "dimension": 2,
                "records": [row, {**row, "key": "skill:duplicate"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate skill embedding"):
        load_skill_embedding_store(path)


def test_schema_v2_loader_rejects_zero_norm_skill_vectors(tmp_path) -> None:
    path = tmp_path / "embeddings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "model_id": "embedding-test-model",
                "dimension": 2,
                "records": [
                    _record("skill:skill:alpha", "skill:alpha", "skill", [0.0, 0.0]),
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-zero norm"):
        load_skill_embedding_store(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"model_id": 123}),
        lambda payload: payload.update({"dimension": "2"}),
        lambda payload: payload["records"][0].update({"skill_id": 123}),
        lambda payload: payload["records"][0].update({"vector": ["1.0", 0.0]}),
        lambda payload: payload["records"][0].update({"vector": [True, 0.0]}),
    ],
)
def test_schema_v2_loader_rejects_coerced_metadata_and_vectors(tmp_path, mutate) -> None:
    path = tmp_path / "embeddings.json"
    payload = {
        "schema_version": "2.0",
        "model_id": "embedding-test-model",
        "dimension": 2,
        "records": [
            _record("skill:skill:alpha", "skill:alpha", "skill", [1.0, 0.0]),
        ],
    }
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_skill_embedding_store(path)


def test_router_query_uses_store_model_and_query_embedding(tmp_path) -> None:
    workspace = Workspace(tmp_path / ".skillfabric")
    workspace.ensure()
    workspace.write_json(
        workspace.graph_dir / "embeddings.json",
        {
            "schema_version": "2.0",
            "model_id": DEFAULT_EMBEDDING_MODEL_ID,
            "dimension": 2,
            "records": [
                _record("skill:skill:alpha", "skill:alpha", "skill", [1.0, 0.0]),
            ],
        },
    )
    skills = {"skill:alpha": _skill("skill:alpha", "alpha")}
    query_provider = QueryProvider()

    with (
        patch("skillfabric.router.retrieval.search_bm25", return_value=[]),
        patch(
            "skillfabric.router.retrieval.embedding_provider_for_model",
            return_value=query_provider,
        ) as provider_factory,
    ):
        seeds = retrieve_seed_candidates(
            workspace,
            "find alpha",
            skills,
            limit=1,
            env_file=None,
        )

    provider_factory.assert_called_once_with(
        DEFAULT_EMBEDDING_MODEL_ID,
        dimension=2,
        env_path=None,
    )
    assert query_provider.query_calls == ["find alpha"]
    assert query_provider.embed_calls == []
    assert seeds[0].skill_id == "skill:alpha"
    assert seeds[0].retrieval_ranks == {"embedding": 1}


def test_legacy_embedding_store_api_is_removed() -> None:
    assert not hasattr(embedding_module, "build_embedding_store")
    assert not hasattr(embedding_module, "load_embedding_store")
    assert not hasattr(embedding_module, "load_embedding_store_payload")
    assert "DISABLE_DENSE_EMBEDDINGS" not in Path(embedding_module.__file__).read_text(
        encoding="utf-8"
    )


def _record(key: str, skill_id: str, kind: str, vector: list[float]) -> dict[str, object]:
    return {
        "key": key,
        "skill_id": skill_id,
        "kind": kind,
        "field_name": "",
        "text_hash": "text-hash",
        "vector": vector,
    }


def _skill(skill_id: str, name: str) -> SkillNode:
    return SkillNode(
        id=skill_id,
        type="skill",
        name=name,
        description=f"{name} description",
        content_hash="content-hash",
        raw_text=f"# {name}\n\n{name} raw text.",
    )
