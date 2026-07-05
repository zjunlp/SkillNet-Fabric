"""Embedding provider interfaces and embedding stores."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from skillfabric.indexing.canonical import canonical_skill_text
from skillfabric.registry.models import SkillNode
from skillfabric.runtime.llm import DEFAULT_API_BASE, read_env_file

DEFAULT_EMBEDDING_MODEL_ID = "openai/text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSION = 1536
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_EMBEDDING_TEXT_CHARS = 4_000


class EmbeddingProvider(Protocol):
    """Embedding provider protocol."""

    model_id: str
    dimension: int

    def embed(self, text: str) -> list[float]:
        """Embed a single text."""


@dataclass(slots=True)
class LoadedEmbeddingStore:
    """Loaded embedding store with vectors and model metadata."""

    vectors: dict[str, list[float]]
    model_id: str
    dimension: int
    provider: str = ""


@dataclass(slots=True)
class ApiEmbeddingProvider:
    """LiteLLM-backed embedding provider for public SkillFabric builds."""

    model_id: str = DEFAULT_EMBEDDING_MODEL_ID
    dimension: int = DEFAULT_EMBEDDING_DIMENSION
    api_key: str = ""
    api_base: str = ""
    timeout: float = 120.0
    batch_size: int = field(default_factory=lambda: _int_env("EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE))
    max_text_chars: int = field(
        default_factory=lambda: _int_env("EMBEDDING_TEXT_CHARS", DEFAULT_EMBEDDING_TEXT_CHARS)
    )
    provider_name: str = "api"

    @classmethod
    def from_env(cls, *, env_path: str | Path | None = ".env", model_id: str | None = None, dimension: int = 0) -> ApiEmbeddingProvider:
        """Create an API embedding provider from environment variables and an optional env file."""

        values = read_env_file(env_path)
        resolved_model = model_id or _first_value(values, "EMBEDDING_MODEL", default=DEFAULT_EMBEDDING_MODEL_ID)
        return cls(
            model_id=resolved_model,
            dimension=dimension or _int_value(
                _first_value(values, "EMBEDDING_DIMENSION", default=str(DEFAULT_EMBEDDING_DIMENSION)),
                DEFAULT_EMBEDDING_DIMENSION,
            ),
            api_key=_first_value(
                values,
                "EMBEDDING_API_KEY",
            )
            or _first_value(
                values,
                "API_KEY",
                "OPENAI_API_KEY",
            ),
            api_base=_first_value(
                values,
                "EMBEDDING_BASE_URL",
            )
            or _first_value(
                values,
                "BASE_URL",
                "OPENAI_BASE_URL",
                "OPENAI_API_BASE",
                default=DEFAULT_API_BASE,
            ),
            timeout=float(_first_value(values, "EMBEDDING_TIMEOUT", default="120")),
        )

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed texts through LiteLLM's OpenAI-compatible embedding API."""

        if not texts:
            return []
        try:
            import litellm
        except Exception as exc:  # pragma: no cover - dependency smoke covers this path.
            raise RuntimeError("litellm is required for API embeddings") from exc
        rows: list[list[float]] = []
        batch_size = max(1, int(self.batch_size))
        prepared = [_truncate_embedding_text(text, self.max_text_chars) for text in texts]
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            kwargs: dict[str, Any] = {
                "model": self.model_id,
                "input": batch,
                "timeout": self.timeout,
                "request_timeout": self.timeout,
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["api_base"] = self.api_base
            response = litellm.embedding(**kwargs)
            rows.extend(_vectors_from_embedding_response(response))
        if rows:
            self.dimension = len(rows[0])
        return rows


def default_embedding_provider(*, env_path: str | Path | None = ".env") -> EmbeddingProvider:
    """Return the default embedding provider for public SkillFabric builds."""

    values = read_env_file(env_path)
    provider = _first_value(values, "EMBEDDING_PROVIDER", default="api").strip().lower()
    if provider in {"", "api"}:
        return ApiEmbeddingProvider.from_env(env_path=env_path)
    if provider == "disabled":
        return DisabledEmbeddingProvider()
    raise ValueError(f"unsupported embedding provider: {provider}. Use 'api' or 'disabled'.")


@dataclass(slots=True)
class DisabledEmbeddingProvider:
    """Embedding provider that writes empty vectors while preserving build flow."""

    model_id: str = "disabled"
    dimension: int = 0
    provider_name: str = "disabled"

    def embed(self, text: str) -> list[float]:
        del text
        return []

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[] for _text in texts]


def embedding_provider_for_model(
    model_id: str,
    *,
    dimension: int = 0,
    env_path: str | Path | None = ".env",
) -> EmbeddingProvider:
    """Resolve the query embedding provider matching an embedding store model id."""

    if model_id in {"", "disabled"}:
        return DisabledEmbeddingProvider()
    return ApiEmbeddingProvider.from_env(
        env_path=env_path,
        model_id=model_id,
        dimension=dimension or DEFAULT_EMBEDDING_DIMENSION,
    )


def build_embedding_store(
    skills: list[SkillNode],
    path: str | Path,
    *,
    provider: EmbeddingProvider | None = None,
) -> dict[str, list[float]]:
    """Build and write an embedding store."""

    provider = provider or default_embedding_provider()
    target = Path(path)
    if _disable_dense_embeddings() or getattr(provider, "provider_name", "") == "disabled":
        vectors = {skill.id: [] for skill in skills}
        payload = {
            "schema_version": "1.0",
            "provider": getattr(provider, "provider_name", provider.__class__.__name__),
            "model_id": provider.model_id,
            "dimension": 0,
            "disabled": True,
            "embeddings": [
                {
                    "skill_id": skill.id,
                    "content_hash": skill.content_hash,
                    "canonical_skill_text_hash": skill.canonical_skill_text_hash,
                    "vector": [],
                }
                for skill in skills
            ],
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return vectors
    existing = _load_existing_payload(target, provider.model_id)
    reusable = {
        str(item.get("skill_id")): item
        for item in existing.get("embeddings", [])
        if isinstance(item, dict)
    }
    vectors: dict[str, list[float]] = {}
    rows: list[dict[str, object]] = []
    for skill in skills:
        old = reusable.get(skill.id)
        if old and old.get("content_hash") == skill.content_hash:
            vector = [float(value) for value in old.get("vector", [])]
        else:
            vector = []
        vectors[skill.id] = vector

    pending = [skill for skill in skills if not vectors[skill.id]]
    pending_vectors = _embed_pending(provider, [canonical_skill_text(skill) for skill in pending])
    for skill, vector in zip(pending, pending_vectors, strict=False):
        vectors[skill.id] = vector

    for skill in skills:
        vector = vectors[skill.id]
        rows.append(
            {
                "skill_id": skill.id,
                "content_hash": skill.content_hash,
                "canonical_skill_text_hash": skill.canonical_skill_text_hash,
                "vector": vector,
            }
        )
    payload = {
        "schema_version": "1.0",
        "provider": getattr(provider, "provider_name", provider.__class__.__name__),
        "model_id": provider.model_id,
        "dimension": getattr(provider, "dimension", len(next(iter(vectors.values()), [])) or 0),
        "embeddings": rows,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return vectors


def load_embedding_store(path: str | Path) -> dict[str, list[float]]:
    """Load an embedding store."""

    return load_embedding_store_payload(path).vectors


def load_embedding_store_payload(path: str | Path) -> LoadedEmbeddingStore:
    """Load embedding vectors and model metadata."""

    payload = _load_embedding_store_payload_json(*_file_cache_key(Path(path)))
    vectors = {
        str(item["skill_id"]): [float(value) for value in item["vector"]]
        for item in payload.get("embeddings", [])
    }
    dimension = int(payload.get("dimension") or len(next(iter(vectors.values()), [])) or 0)
    return LoadedEmbeddingStore(
        vectors=vectors,
        model_id=str(payload.get("model_id", "")),
        dimension=dimension,
        provider=str(payload.get("provider", "")),
    )


@lru_cache(maxsize=8)
def _load_embedding_store_payload_json(path: str, _mtime_ns: int, _size: int) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_cache_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _embed_pending(provider: EmbeddingProvider, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    embed_many = getattr(provider, "embed_many", None)
    if callable(embed_many):
        return [list(map(float, vector)) for vector in embed_many(texts)]
    return [provider.embed(text) for text in texts]


def _truncate_embedding_text(text: str, max_chars: int) -> str:
    limit = max(0, int(max_chars))
    if limit <= 0 or len(text) <= limit:
        return text
    head = max(1, limit * 2 // 3)
    tail = max(1, limit - head)
    return text[:head].rstrip() + "\n\n...\n\n" + text[-tail:].lstrip()


def _vectors_from_embedding_response(response: Any) -> list[list[float]]:
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    elif hasattr(response, "dict"):
        response = response.dict()
    data = response.get("data", []) if isinstance(response, dict) else getattr(response, "data", [])
    rows: list[tuple[int, list[float]]] = []
    for index, item in enumerate(data):
        if isinstance(item, dict):
            embedding = item.get("embedding", [])
            row_index = int(item.get("index", index))
        else:
            embedding = getattr(item, "embedding", [])
            row_index = int(getattr(item, "index", index))
        rows.append((row_index, [float(value) for value in embedding]))
    rows.sort(key=lambda item: item[0])
    if not rows:
        raise RuntimeError("embedding response did not contain vectors")
    return [vector for _index, vector in rows]


def _first_value(values: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        if values.get(key):
            return values[key]
    for key in keys:
        if key in os.environ and os.environ[key]:
            return os.environ[key]
    return default


def _int_env(name: str, default: int) -> int:
    return _int_value(os.environ.get(name, ""), default)


def _int_value(raw: str, default: int) -> int:
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _disable_dense_embeddings() -> bool:
    return os.environ.get("DISABLE_DENSE_EMBEDDINGS", "").lower() in {"1", "true", "yes", "on"}


def embed_query(provider: EmbeddingProvider, query: str) -> list[float]:
    """Embed a retrieval query with provider-specific query handling."""

    embed_query_fn = getattr(provider, "embed_query", None)
    if callable(embed_query_fn):
        return list(map(float, embed_query_fn(query)))
    return provider.embed(query)


def _load_existing_payload(path: Path, model_id: str) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if payload.get("model_id") != model_id:
        return {}
    return payload


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity for dense vectors."""

    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=False)) / (left_norm * right_norm)
