from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from skillfabric.indexing.embeddings import (
    DEFAULT_EMBEDDING_MODEL_ID,
    LOCAL_SENTENCE_TRANSFORMER_MODEL_ID,
    ApiEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    build_embedding_store,
    embedding_provider_for_model,
)
from skillfabric.registry.models import SkillNode
from skillfabric.router.retrieval import _seed_scores
from skillfabric.storage import Workspace


class _BatchProvider:
    model_id = "test-batch-model"
    dimension = 2

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, text: str) -> list[float]:
        raise AssertionError("build_embedding_store should use embed_many when available")

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]


class _QueryProvider:
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


class _RecordingSentenceTransformer:
    def __init__(self) -> None:
        self.text_batches: list[list[str]] = []
        self.kwargs: list[dict[str, object]] = []

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        self.text_batches.append(list(texts))
        self.kwargs.append(dict(kwargs))
        return [[1.0, 0.0] for _text in texts]


class EmbeddingTests(unittest.TestCase):
    def test_api_embedding_provider_is_default_and_uses_litellm_embedding(self) -> None:
        provider = ApiEmbeddingProvider(api_key="test-key", api_base="https://example.test/v1")
        calls: list[dict[str, object]] = []

        def fake_embedding(**kwargs: object) -> dict[str, object]:
            calls.append(dict(kwargs))
            return {"data": [{"embedding": [0.25, 0.75]}]}

        fake_litellm = type("FakeLiteLLM", (), {"embedding": staticmethod(fake_embedding)})
        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            vector = provider.embed("find pdf table parser")

        self.assertEqual(vector, [0.25, 0.75])
        self.assertEqual(calls[0]["model"], DEFAULT_EMBEDDING_MODEL_ID)
        self.assertEqual(calls[0]["input"], ["find pdf table parser"])
        self.assertEqual(calls[0]["api_key"], "test-key")
        self.assertEqual(calls[0]["api_base"], "https://example.test/v1")

    def test_api_embedding_provider_uses_skillnet_style_api_env_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "API_KEY=sk-shared",
                        "BASE_URL=https://shared.example/v1",
                        "EMBEDDING_MODEL=openai/custom-embedding",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "API_KEY": "",
                    "BASE_URL": "",
                    "EMBEDDING_API_KEY": "",
                    "EMBEDDING_BASE_URL": "",
                    "OPENAI_API_KEY": "",
                    "OPENAI_BASE_URL": "",
                    "OPENAI_API_BASE": "",
                },
                clear=False,
            ):
                provider = ApiEmbeddingProvider.from_env(env_path=env_path)

        self.assertEqual(provider.api_key, "sk-shared")
        self.assertEqual(provider.api_base, "https://shared.example/v1")
        self.assertEqual(provider.model_id, "openai/custom-embedding")

    def test_api_embedding_provider_specific_env_overrides_shared_api_env(self) -> None:
        with TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "API_KEY=sk-shared",
                        "BASE_URL=https://shared.example/v1",
                        "EMBEDDING_API_KEY=sk-embedding",
                        "EMBEDDING_BASE_URL=https://embedding.example/v1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            provider = ApiEmbeddingProvider.from_env(env_path=env_path)

        self.assertEqual(provider.api_key, "sk-embedding")
        self.assertEqual(provider.api_base, "https://embedding.example/v1")

    def test_default_provider_for_model_resolves_api_provider(self) -> None:
        provider = embedding_provider_for_model(DEFAULT_EMBEDDING_MODEL_ID, dimension=1536)

        self.assertIsInstance(provider, ApiEmbeddingProvider)
        self.assertEqual(provider.dimension, 1536)

    def test_local_sentence_transformer_provider_uses_configured_model_path(self) -> None:
        provider = SentenceTransformerEmbeddingProvider()

        self.assertEqual(provider.model_id, LOCAL_SENTENCE_TRANSFORMER_MODEL_ID)
        self.assertIn("bge-large-en-v1.5", str(provider.model_path))
        self.assertEqual(provider.dimension, 1024)

    def test_embedding_provider_for_model_accepts_api_model_ids(self) -> None:
        provider = embedding_provider_for_model("custom/openai-compatible-embedding", dimension=32)

        self.assertIsInstance(provider, ApiEmbeddingProvider)
        self.assertEqual(provider.model_id, "custom/openai-compatible-embedding")
        self.assertEqual(provider.dimension, 32)

    def test_embedding_provider_for_model_uses_local_model_path_from_env_file(self) -> None:
        with TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "models" / "BAAI" / "bge-large-en-v1.5"
            env_path = Path(tmp) / ".env"
            env_path.write_text(f"EMBEDDING_MODEL_PATH={model_path}\n", encoding="utf-8")

            provider = embedding_provider_for_model(
                LOCAL_SENTENCE_TRANSFORMER_MODEL_ID,
                dimension=1024,
                env_path=env_path,
            )

        self.assertIsInstance(provider, SentenceTransformerEmbeddingProvider)
        self.assertEqual(Path(provider.model_path), model_path)
        self.assertEqual(provider.dimension, 1024)

    def test_bge_provider_uses_query_instruction_for_retrieval_queries(self) -> None:
        provider = SentenceTransformerEmbeddingProvider()
        recorder = _RecordingSentenceTransformer()

        with patch.object(SentenceTransformerEmbeddingProvider, "_model", return_value=recorder):
            provider.embed_query("find pdf table parser")
            provider.embed("pdf table parser skill")

        self.assertEqual(
            recorder.text_batches[0],
            ["Represent this sentence for searching relevant passages: find pdf table parser"],
        )
        self.assertEqual(recorder.text_batches[1], ["pdf table parser skill"])

    def test_bge_provider_batches_and_truncates_long_embedding_texts(self) -> None:
        provider = SentenceTransformerEmbeddingProvider(batch_size=2, max_text_chars=12)
        recorder = _RecordingSentenceTransformer()

        with patch.object(SentenceTransformerEmbeddingProvider, "_model", return_value=recorder):
            vectors = provider.embed_many(["alpha" * 10, "beta", "gamma"])

        self.assertEqual(len(vectors), 3)
        self.assertEqual([len(batch) for batch in recorder.text_batches], [2, 1])
        self.assertEqual(recorder.kwargs[0]["batch_size"], 2)
        self.assertIn("...", recorder.text_batches[0][0])
        self.assertLessEqual(len(recorder.text_batches[0][0]), 20)

    def test_build_embedding_store_batches_sentence_vectors_and_records_model(self) -> None:
        with TemporaryDirectory() as tmp:
            provider = _BatchProvider()
            target = Path(tmp) / "embeddings.json"

            build_embedding_store([_skill("skill:alpha", "alpha"), _skill("skill:beta", "beta")], target, provider=provider)

            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["model_id"], "test-batch-model")
            self.assertEqual(payload["dimension"], 2)
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(len(payload["embeddings"]), 2)

    def test_build_embedding_store_can_disable_dense_embeddings(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"DISABLE_DENSE_EMBEDDINGS": "1"}):
            provider = _BatchProvider()
            target = Path(tmp) / "embeddings.json"

            vectors = build_embedding_store([_skill("skill:alpha", "alpha")], target, provider=provider)

            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(vectors, {"skill:alpha": []})
            self.assertTrue(payload["disabled"])
            self.assertEqual(payload["dimension"], 0)
            self.assertEqual(payload["embeddings"][0]["vector"], [])
            self.assertEqual(provider.calls, [])

    def test_router_query_embedding_uses_store_model_id(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp) / ".skillfabric")
            workspace.ensure()
            workspace.write_json(
                workspace.index_dir / "embeddings.json",
                {
                    "schema_version": "1.0",
                    "model_id": DEFAULT_EMBEDDING_MODEL_ID,
                    "dimension": 2,
                    "embeddings": [
                        {
                            "skill_id": "skill:alpha",
                            "content_hash": "h",
                            "canonical_skill_text_hash": "ct",
                            "vector": [1.0, 0.0],
                        }
                    ],
                },
            )
            skills = {"skill:alpha": _skill("skill:alpha", "alpha")}

            query_provider = _QueryProvider()
            with patch("skillfabric.router.retrieval.search_bm25", return_value=[]), patch(
                "skillfabric.router.retrieval.embedding_provider_for_model",
                return_value=query_provider,
            ) as provider_factory:
                seeds = _seed_scores(workspace, "find alpha", skills, warnings=[])

            provider_factory.assert_called_once_with(DEFAULT_EMBEDDING_MODEL_ID, dimension=2)
            self.assertEqual(query_provider.query_calls, ["find alpha"])
            self.assertEqual(query_provider.embed_calls, [])
            self.assertIn("skill:alpha", seeds)
            self.assertIn("embedding", seeds["skill:alpha"].sources)


def _skill(skill_id: str, name: str) -> SkillNode:
    return SkillNode(
        id=skill_id,
        type="skill",
        name=name,
        description=f"{name} description",
        source_path=f"{name}/SKILL.md",
        wiki_path=f"skills/{name}.md",
        content_hash="h",
        token_count=10,
        canonical_skill_text_hash="ct",
        raw_text=f"# {name}\n\n{name} raw text.",
    )


if __name__ == "__main__":
    unittest.main()
