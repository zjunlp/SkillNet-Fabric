"""Embedding providers and query-vector loading."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from skillfabric.runtime.llm import DEFAULT_API_BASE, read_env_file

DEFAULT_EMBEDDING_MODEL_ID = "openai/text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSION = 1536
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_EMBEDDING_TEXT_CHARS = 4_000
DEFAULT_EMBEDDING_MAX_RETRIES = 2
DEFAULT_EMBEDDING_RETRY_BACKOFF_SECONDS = 1.0
_RECORD_KEYS = {
    "key",
    "skill_id",
    "kind",
    "field_name",
    "text_hash",
    "vector",
}


class EmbeddingProvider(Protocol):
    model_id: str
    dimension: int

    def embed(self, text: str) -> list[float]:
        """Embed one document or query."""


@dataclass(frozen=True, slots=True)
class LoadedEmbeddingStore:
    vectors: dict[str, list[float]]
    model_id: str
    dimension: int


@dataclass(slots=True)
class ApiEmbeddingProvider:
    model_id: str = DEFAULT_EMBEDDING_MODEL_ID
    dimension: int = DEFAULT_EMBEDDING_DIMENSION
    api_key: str = ""
    api_base: str = ""
    timeout: float = 120.0
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    max_text_chars: int = DEFAULT_EMBEDDING_TEXT_CHARS
    max_retries: int = DEFAULT_EMBEDDING_MAX_RETRIES
    retry_backoff_seconds: float = DEFAULT_EMBEDDING_RETRY_BACKOFF_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        _require_int(self.dimension, name="dimension", minimum=1)
        _require_float(self.timeout, name="timeout", minimum_exclusive=0.0)
        _require_int(self.batch_size, name="batch_size", minimum=1)
        _require_int(self.max_text_chars, name="max_text_chars", minimum=0)
        _require_int(self.max_retries, name="max_retries", minimum=0)
        _require_nonnegative_float(
            self.retry_backoff_seconds,
            name="retry_backoff_seconds",
        )
        for name, value in (("api_key", self.api_key), ("api_base", self.api_base)):
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")

    @classmethod
    def from_env(
        cls,
        *,
        env_path: str | Path | None = ".env",
        model_id: str | None = None,
        dimension: int | None = None,
    ) -> ApiEmbeddingProvider:
        values = read_env_file(env_path)
        resolved_model = (
            model_id
            if model_id is not None
            else _first_value(
                values,
                "EMBEDDING_MODEL",
                default=DEFAULT_EMBEDDING_MODEL_ID,
            )
        )
        return cls(
            model_id=resolved_model,
            dimension=(
                dimension
                if dimension is not None
                else _parse_int(
                    _first_value(
                        values,
                        "EMBEDDING_DIMENSION",
                        default=str(DEFAULT_EMBEDDING_DIMENSION),
                    ),
                    name="EMBEDDING_DIMENSION",
                )
            ),
            api_key=_first_value(values, "EMBEDDING_API_KEY")
            or _first_value(values, "API_KEY", "OPENAI_API_KEY"),
            api_base=_first_value(values, "EMBEDDING_BASE_URL")
            or _first_value(
                values,
                "BASE_URL",
                "OPENAI_BASE_URL",
                "OPENAI_API_BASE",
                default=DEFAULT_API_BASE,
            ),
            timeout=_parse_float(
                _first_value(values, "EMBEDDING_TIMEOUT", default="120"),
                name="EMBEDDING_TIMEOUT",
            ),
            batch_size=_parse_int(
                _first_value(
                    values,
                    "EMBEDDING_BATCH_SIZE",
                    default=str(DEFAULT_EMBEDDING_BATCH_SIZE),
                ),
                name="EMBEDDING_BATCH_SIZE",
            ),
            max_text_chars=_parse_int(
                _first_value(
                    values,
                    "EMBEDDING_TEXT_CHARS",
                    default=str(DEFAULT_EMBEDDING_TEXT_CHARS),
                ),
                name="EMBEDDING_TEXT_CHARS",
            ),
            max_retries=_parse_int(
                _first_value(
                    values,
                    "EMBEDDING_MAX_RETRIES",
                    default=str(DEFAULT_EMBEDDING_MAX_RETRIES),
                ),
                name="EMBEDDING_MAX_RETRIES",
            ),
            retry_backoff_seconds=_parse_float(
                _first_value(
                    values,
                    "EMBEDDING_RETRY_BACKOFF_SECONDS",
                    default=str(DEFAULT_EMBEDDING_RETRY_BACKOFF_SECONDS),
                ),
                name="EMBEDDING_RETRY_BACKOFF_SECONDS",
            ),
        )

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - dependency smoke covers this path.
            raise RuntimeError("litellm is required for API embeddings") from exc
        vectors: list[list[float]] = []
        prepared = [_truncate_embedding_text(text, self.max_text_chars) for text in texts]
        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
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
            vectors.extend(self._embed_batch(litellm, kwargs))
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"embedding API returned {len(vectors)} vectors for {len(texts)} texts"
            )
        dimensions = {len(vector) for vector in vectors}
        if 0 in dimensions or len(dimensions) != 1:
            raise RuntimeError("embedding API returned empty or inconsistent vectors")
        actual_dimension = dimensions.pop()
        if self.dimension > 0 and actual_dimension != self.dimension:
            raise RuntimeError(
                f"embedding API returned dimension {actual_dimension}; expected {self.dimension}"
            )
        if any(not _is_finite_nonzero_vector(vector) for vector in vectors):
            raise RuntimeError("embedding API vectors must be finite and non-zero")
        self.dimension = actual_dimension
        return vectors

    def _embed_batch(self, litellm: Any, kwargs: dict[str, Any]) -> list[list[float]]:
        for attempt in range(self.max_retries + 1):
            try:
                return _vectors_from_embedding_response(litellm.embedding(**kwargs))
            except Exception:
                if attempt == self.max_retries:
                    raise
                if self.retry_backoff_seconds:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
        raise AssertionError("embedding retry loop ended unexpectedly")


def default_embedding_provider(
    *,
    env_path: str | Path | None = ".env",
) -> EmbeddingProvider:
    return ApiEmbeddingProvider.from_env(env_path=env_path)


def embedding_provider_for_model(
    model_id: str,
    *,
    dimension: int,
    env_path: str | Path | None = ".env",
) -> EmbeddingProvider:
    if not model_id or dimension <= 0:
        raise ValueError("embedding store requires a model id and positive dimension")
    return ApiEmbeddingProvider.from_env(
        env_path=env_path,
        model_id=model_id,
        dimension=dimension,
    )


def load_skill_embedding_store(path: str | Path) -> LoadedEmbeddingStore:
    """Load contract-document vectors used by query routing."""

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"embedding store not found: {target}; rebuild the workspace")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid embedding store JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "model_id",
        "dimension",
        "records",
    }:
        raise ValueError("embedding store must use the canonical fields")
    model_id = payload["model_id"]
    dimension = payload["dimension"]
    rows = payload["records"]
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("embedding store model_id must be a non-empty string")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("embedding store dimension must be a positive integer")
    if not isinstance(rows, list):
        raise ValueError("embedding store metadata is invalid")
    vectors: dict[str, list[float]] = {}
    record_keys: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != _RECORD_KEYS:
            raise ValueError(f"embedding store record {index} has invalid fields")
        key = _required_store_string(row["key"], label=f"record {index} key")
        if key in record_keys:
            raise ValueError(f"embedding store contains duplicate record key: {key}")
        record_keys.add(key)
        skill_id = _required_store_string(
            row["skill_id"],
            label=f"record {index} skill_id",
        )
        if not isinstance(row["field_name"], str):
            raise ValueError(f"embedding store record {index} field_name must be a string")
        _required_store_string(row["text_hash"], label=f"record {index} text_hash")
        kind = row["kind"]
        if kind not in {"skill", "requires", "produces"}:
            raise ValueError(f"embedding store record {index} has invalid kind")
        raw_vector = row["vector"]
        if not isinstance(raw_vector, list):
            raise ValueError(f"embedding store record {index} vector must be a list")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_vector
        ):
            raise ValueError(f"embedding store record {index} vector must contain numbers")
        vector = [float(value) for value in raw_vector]
        if len(vector) != dimension or any(not math.isfinite(value) for value in vector):
            raise ValueError(f"embedding store record {index} has an invalid vector")
        if not _is_finite_nonzero_vector(vector):
            raise ValueError(f"embedding store record {index} vector must have non-zero norm")
        if kind != "skill":
            continue
        if skill_id in vectors:
            raise ValueError(f"duplicate skill embedding: {skill_id}")
        vectors[skill_id] = vector
    if not vectors:
        raise ValueError("embedding store contains no routable skill vectors")
    return LoadedEmbeddingStore(vectors=vectors, model_id=model_id, dimension=dimension)


def embed_query(provider: EmbeddingProvider, query: str) -> list[float]:
    embed_query_fn = getattr(provider, "embed_query", None)
    if callable(embed_query_fn):
        return list(map(float, embed_query_fn(query)))
    return list(map(float, provider.embed(query)))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _truncate_embedding_text(text: str, max_chars: int) -> str:
    if max_chars == 0 or len(text) <= max_chars:
        return text
    head = max(1, max_chars * 2 // 3)
    tail = max(1, max_chars - head)
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
    if [row_index for row_index, _vector in rows] != list(range(len(rows))):
        raise RuntimeError("embedding response indexes must be unique and contiguous")
    return [vector for _index, vector in rows]


def _is_finite_nonzero_vector(vector: list[float]) -> bool:
    return all(math.isfinite(value) for value in vector) and any(value != 0.0 for value in vector)


def _first_value(values: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        if values.get(key):
            return values[key]
    for key in keys:
        if os.environ.get(key):
            return os.environ[key]
    return default


def _parse_int(raw: str, *, name: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _required_store_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"embedding store {label} must be a non-empty string")
    return value.strip()


def _parse_float(raw: str, *, name: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


def _require_int(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _require_float(value: object, *, name: str, minimum_exclusive: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number greater than {minimum_exclusive}")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= minimum_exclusive:
        raise ValueError(f"{name} must be a finite number greater than {minimum_exclusive}")
    return resolved


def _require_nonnegative_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return resolved


__all__ = [
    "DEFAULT_EMBEDDING_MODEL_ID",
    "ApiEmbeddingProvider",
    "EmbeddingProvider",
    "LoadedEmbeddingStore",
    "cosine_similarity",
    "default_embedding_provider",
    "embed_query",
    "embedding_provider_for_model",
    "load_skill_embedding_store",
]
