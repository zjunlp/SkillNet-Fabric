"""Candidate grouping for interface term canonicalization."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from skillfabric.compiled_graph.canonicalization.models import (
    CanonicalizationCluster,
    RawContractObject,
)
from skillfabric.indexing.embeddings import (
    EmbeddingProvider,
    default_embedding_provider,
)


class CanonicalSemanticEmbedder(Protocol):
    """Embedding interface for semantic candidate generation."""

    model_id: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per text."""


@dataclass(slots=True)
class EmbeddingProviderCanonicalEmbedder:
    """Adapter from the shared embedding provider to canonicalization."""

    provider: EmbeddingProvider

    @property
    def model_id(self) -> str:
        return self.provider.model_id

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embed_many = getattr(self.provider, "embed_many", None)
        if callable(embed_many):
            return [list(map(float, vector)) for vector in embed_many(texts)]
        return [list(map(float, self.provider.embed(text))) for text in texts]


@dataclass(slots=True)
class SemanticCandidatePair:
    """A semantic neighbor pair used only to form small resolver inputs."""

    left_key: str
    right_key: str
    score: float


class HashingCanonicalEmbedder:
    """Deterministic local embedder used when no real embedding provider is available."""

    model_id = "hashing-canonical-embedder"
    dimension = 64

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimension
            for token in normalized_candidate_text(text).split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                for index, byte in enumerate(digest):
                    vector[index % self.dimension] += (byte / 255.0) - 0.5
            vectors.append(_normalize_vector(vector))
        return vectors


def normalized_candidate_text(value: str) -> str:
    """Normalize text mechanically without applying semantic aliases."""

    text = value.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    return " ".join(text.split())


def candidate_groups_from_terms(
    terms: list[RawContractObject],
    *,
    semantic_embedder: CanonicalSemanticEmbedder | None = None,
    embedding_cache_path: str | Path | None = None,
    semantic_threshold: float = 0.82,
    semantic_top_k: int = 3,
    max_group_size: int = 12,
    include_semantic: bool = True,
) -> list[CanonicalizationCluster]:
    """Build small candidate groups for deterministic or provider resolution."""

    if not terms:
        return []
    exact_groups = _normalized_exact_groups(terms)
    exact_group_keys = {term.key for group in exact_groups for term in group}
    semantic_groups: list[list[RawContractObject]] = []
    if include_semantic:
        semantic_pairs = generate_semantic_candidate_pairs(
            [term for term in terms if term.key not in exact_group_keys],
            embedder=semantic_embedder or _default_embedder(),
            threshold=semantic_threshold,
            top_k=semantic_top_k,
            cache_path=embedding_cache_path,
        )
        semantic_groups = _groups_from_pairs(terms, semantic_pairs)
    groups = [group for group in [*exact_groups, *semantic_groups] if _group_has_producer_and_consumer(group)]
    output: list[CanonicalizationCluster] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        for chunk in _chunk_terms(sorted(group, key=lambda item: item.key), max_group_size):
            key = tuple(term.key for term in chunk)
            if key in seen:
                continue
            seen.add(key)
            output.append(CanonicalizationCluster(cluster_id=_cluster_id(chunk), terms=list(chunk)))
    return sorted(output, key=lambda item: item.cluster_id)


def generate_semantic_candidate_pairs(
    terms: list[RawContractObject],
    *,
    embedder: CanonicalSemanticEmbedder,
    threshold: float = 0.82,
    top_k: int = 3,
    cache_path: str | Path | None = None,
) -> list[SemanticCandidatePair]:
    """Generate top-k semantic neighbor pairs."""

    if len(terms) < 2:
        return []
    cached = _load_embedding_cache(cache_path, embedder.model_id)
    texts_by_key = {term.key: _semantic_text(term) for term in terms}
    cache_key_by_term = {
        term.key: _embedding_text_cache_key(embedder.model_id, texts_by_key[term.key])
        for term in terms
    }
    vectors_by_key: dict[str, list[float]] = {}
    for term in terms:
        cached_vector = cached.get(cache_key_by_term[term.key])
        if cached_vector is not None:
            vectors_by_key[term.key] = cached_vector
    missing_terms = [term for term in terms if term.key not in vectors_by_key]
    if missing_terms:
        vectors = embedder.embed_texts([texts_by_key[term.key] for term in missing_terms])
        for term, vector in zip(missing_terms, vectors, strict=False):
            normalized = _normalize_vector(vector)
            vectors_by_key[term.key] = normalized
            cached[cache_key_by_term[term.key]] = normalized
        _write_embedding_cache(cache_path, embedder.model_id, cached)

    pairs: dict[tuple[str, str], SemanticCandidatePair] = {}
    for left in terms:
        scored: list[tuple[float, RawContractObject]] = []
        for right in terms:
            if left.key == right.key:
                continue
            score = _dot(vectors_by_key[left.key], vectors_by_key[right.key])
            if score >= threshold:
                scored.append((score, right))
        for score, right in sorted(scored, key=lambda item: (-item[0], item[1].key))[:top_k]:
            key = tuple(sorted((left.key, right.key)))
            existing = pairs.get(key)
            if existing is None or score > existing.score:
                pairs[key] = SemanticCandidatePair(key[0], key[1], score)
    return sorted(pairs.values(), key=lambda item: (item.left_key, item.right_key))


def contract_object_type(kind: str) -> str:
    """Map interface kinds to broad canonicalization object types."""

    normalized = kind.lower().strip()
    if normalized in {"world_state", "physical_state", "environment_state", "state", "condition"}:
        return "state"
    if normalized in {"belief_state", "belief", "memory_state", "knowledge_state", "observation_state"}:
        return "belief_state"
    if normalized in {"planning_state", "planning", "plan_state", "routing_state"}:
        return "planning_state"
    if normalized in {"credential", "environment", "text", "report", "data"}:
        return normalized
    return "artifact"


def _default_embedder() -> CanonicalSemanticEmbedder:
    try:
        return EmbeddingProviderCanonicalEmbedder(default_embedding_provider())
    except Exception:
        return HashingCanonicalEmbedder()


def _normalized_exact_groups(terms: list[RawContractObject]) -> list[list[RawContractObject]]:
    buckets: dict[str, list[RawContractObject]] = defaultdict(list)
    for term in terms:
        normalized = normalized_candidate_text(term.name)
        if normalized:
            buckets[normalized].append(term)
    return [group for group in buckets.values() if len(group) >= 2]


def _groups_from_pairs(
    terms: list[RawContractObject],
    pairs: list[SemanticCandidatePair],
) -> list[list[RawContractObject]]:
    by_key = {term.key: term for term in terms}
    parents = {term.key: term.key for term in terms}

    def find(key: str) -> str:
        parent = parents[key]
        if parent != key:
            parents[key] = find(parent)
        return parents[key]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for pair in pairs:
        if pair.left_key in parents and pair.right_key in parents:
            union(pair.left_key, pair.right_key)
    groups: dict[str, list[RawContractObject]] = defaultdict(list)
    for key in parents:
        root = find(key)
        if key in by_key:
            groups[root].append(by_key[key])
    return [group for group in groups.values() if len(group) >= 2]


def _group_has_producer_and_consumer(group: list[RawContractObject]) -> bool:
    roles = {term.role for term in group}
    return "produces" in roles and "requires" in roles


def _chunk_terms(terms: list[RawContractObject], size: int) -> list[list[RawContractObject]]:
    return [terms[index : index + size] for index in range(0, len(terms), size)]


def _cluster_id(terms: list[RawContractObject]) -> str:
    digest = hashlib.sha256(
        json.dumps([term.key for term in terms], sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return f"canonical-group:{digest}"


def _semantic_text(term: RawContractObject) -> str:
    return " | ".join(
        item
        for item in [
            normalized_candidate_text(term.name),
            contract_object_type(term.kind),
            term.role,
            normalized_candidate_text(term.description),
        ]
        if item
    )


def _embedding_text_cache_key(model_id: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model_id}:{digest}"


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return [0.0 for _value in vector]
    return [value / norm for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


def _load_embedding_cache(path: str | Path | None, model_id: str) -> dict[str, list[float]]:
    if path is None:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("model_id") != model_id:
        return {}
    vectors = payload.get("vectors", {})
    if not isinstance(vectors, dict):
        return {}
    return {
        str(key): [float(item) for item in value]
        for key, value in vectors.items()
        if isinstance(value, list)
    }


def _write_embedding_cache(path: str | Path | None, model_id: str, vectors: dict[str, list[float]]) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "model_id": model_id,
                "vectors": vectors,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
