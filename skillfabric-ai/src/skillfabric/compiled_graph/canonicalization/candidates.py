"""Candidate graph generation for canonical contract objects."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from skillfabric.compiled_graph.canonicalization.models import RawContractObject
from skillfabric.indexing.embeddings import (
    EmbeddingProvider,
    default_embedding_provider,
)


@dataclass(slots=True)
class CanonicalCandidateEdge:
    """Evidence that two raw contract objects may share one canonical object."""

    left_object_id: str
    right_object_id: str
    left_text: str
    right_text: str
    object_type: str
    method: str
    score: float
    features: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        left, right = sorted([self.left_object_id, self.right_object_id])
        return (left, right, self.method)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_object_id": self.left_object_id,
            "right_object_id": self.right_object_id,
            "left_text": self.left_text,
            "right_text": self.right_text,
            "object_type": self.object_type,
            "method": self.method,
            "score": round(float(self.score), 6),
            "features": dict(self.features),
        }


@dataclass(slots=True)
class CanonicalCandidateComponent:
    """Connected candidate component sent to the final canonicalization judge."""

    component_id: str
    object_type: str
    member_ids: list[str]
    candidate_edges: list[CanonicalCandidateEdge] = field(default_factory=list)
    max_score: float = 0.0
    mean_score: float = 0.0
    methods_present: list[str] = field(default_factory=list)
    ambiguous: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "object_type": self.object_type,
            "member_ids": list(self.member_ids),
            "candidate_edges": [edge.to_dict() for edge in self.candidate_edges],
            "max_score": round(float(self.max_score), 6),
            "mean_score": round(float(self.mean_score), 6),
            "methods_present": list(self.methods_present),
            "ambiguous": bool(self.ambiguous),
        }


@dataclass(slots=True)
class CandidateGenerationResult:
    """Candidate generation output used for artifacts and canonicalization clusters."""

    edges: list[CanonicalCandidateEdge]
    components: list[CanonicalCandidateComponent]
    warnings: list[str] = field(default_factory=list)


class CanonicalSemanticEmbedder(Protocol):
    """Embedding provider for canonical object candidate generation."""

    model_id: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return dense vectors for candidate texts."""


class HashingCanonicalEmbedder:
    """Small deterministic fallback embedder for offline tests and unavailable local models."""

    model_id = "hashing-canonical-embedder"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vector(text) for text in texts]


class EmbeddingProviderCanonicalEmbedder:
    """Adapter that reuses the build embedding provider for canonical candidates."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider
        self.model_id = provider.model_id

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embed_many = getattr(self.provider, "embed_many", None)
        if callable(embed_many):
            return [list(map(float, vector)) for vector in embed_many(texts)]
        return [self.provider.embed(text) for text in texts]


class ApiCanonicalEmbedder(EmbeddingProviderCanonicalEmbedder):
    """Default API-backed canonical object embedder."""

    def __init__(self) -> None:
        super().__init__(default_embedding_provider())


def build_candidate_graph(
    terms: list[RawContractObject],
    *,
    semantic_embedder: CanonicalSemanticEmbedder | None = None,
    embedding_cache_path: str | Path | None = None,
    cache_digest: str = "",
    lexical_threshold: float = 0.72,
    semantic_threshold: float = 0.76,
    semantic_top_k: int = 5,
) -> CandidateGenerationResult:
    """Build lexical and semantic candidate components for raw contract objects."""

    lexical_edges = generate_lexical_candidates(terms, threshold=lexical_threshold)
    warnings: list[str] = []
    try:
        semantic_edges = generate_semantic_candidates(
            terms,
            embedder=semantic_embedder or ApiCanonicalEmbedder(),
            top_k=semantic_top_k,
            threshold=semantic_threshold,
            cache_path=embedding_cache_path,
            cache_digest=cache_digest,
        )
    except Exception as exc:
        warnings.append(f"semantic candidate generation unavailable: {type(exc).__name__}: {exc}")
        semantic_edges = generate_semantic_candidates(
            terms,
            embedder=HashingCanonicalEmbedder(),
            top_k=semantic_top_k,
            threshold=semantic_threshold,
            cache_path=None,
            cache_digest=cache_digest,
        )
    edges = _dedupe_edges([*lexical_edges, *semantic_edges])
    components = build_candidate_components(terms, edges)
    return CandidateGenerationResult(edges=edges, components=components, warnings=warnings)


def generate_lexical_candidates(
    terms: list[RawContractObject],
    *,
    threshold: float = 0.72,
) -> list[CanonicalCandidateEdge]:
    """Generate RapidFuzz lexical candidate edges inside object-type buckets."""

    try:
        from rapidfuzz import fuzz
    except Exception as exc:  # pragma: no cover - dependency smoke covers this path.
        raise RuntimeError(f"rapidfuzz unavailable: {exc}") from exc
    edges: list[CanonicalCandidateEdge] = []
    by_type = _terms_by_object_type(terms)
    for object_type, bucket in by_type.items():
        for left_index, left in enumerate(bucket):
            for right in bucket[left_index + 1 :]:
                left_text = normalized_candidate_text(left.name)
                right_text = normalized_candidate_text(right.name)
                if not left_text or not right_text:
                    continue
                token_set = fuzz.token_set_ratio(left_text, right_text) / 100.0
                token_sort = fuzz.token_sort_ratio(left_text, right_text) / 100.0
                token_overlap = _token_overlap(left_text, right_text)
                exact = 1.0 if left_text == right_text else 0.0
                score = max(exact, token_set, token_sort, token_overlap)
                if score < threshold:
                    continue
                edges.append(
                    CanonicalCandidateEdge(
                        left_object_id=left.key,
                        right_object_id=right.key,
                        left_text=left.name,
                        right_text=right.name,
                        object_type=object_type,
                        method="lexical",
                        score=score,
                        features={
                            "token_set_ratio": round(token_set, 6),
                            "token_sort_ratio": round(token_sort, 6),
                            "token_overlap": round(token_overlap, 6),
                            "normalized_exact": bool(exact),
                        },
                    )
                )
    return sorted(edges, key=lambda edge: (-edge.score, edge.key))


def generate_semantic_candidates(
    terms: list[RawContractObject],
    *,
    embedder: CanonicalSemanticEmbedder,
    top_k: int = 5,
    threshold: float = 0.76,
    cache_path: str | Path | None = None,
    cache_digest: str = "",
) -> list[CanonicalCandidateEdge]:
    """Generate nearest-neighbor candidate edges inside object-type buckets."""

    cached = _load_embedding_cache(cache_path, embedder.model_id, cache_digest)
    vectors_by_key: dict[str, list[float]] = dict(cached.get("vectors", {}))
    texts_by_key = {term.key: _semantic_text(term) for term in terms}
    missing_terms = [term for term in terms if term.key not in vectors_by_key]
    if missing_terms:
        vectors = embedder.embed_texts([texts_by_key[term.key] for term in missing_terms])
        for term, vector in zip(missing_terms, vectors, strict=False):
            vectors_by_key[term.key] = _normalize_vector(vector)
        _write_embedding_cache(cache_path, embedder.model_id, cache_digest, vectors_by_key)

    edges: list[CanonicalCandidateEdge] = []
    for object_type, bucket in _terms_by_object_type(terms).items():
        if len(bucket) < 2:
            continue
        keys = [term.key for term in bucket]
        matrix = [vectors_by_key[key] for key in keys if key in vectors_by_key]
        if len(matrix) != len(keys) or not matrix:
            continue
        normalized = [_normalize_vector(vector) for vector in matrix]
        neighbors = min(max(top_k, 1), len(keys) - 1)
        for left_pos, left_vector in enumerate(normalized):
            scored: list[tuple[float, int]] = []
            for right_pos, right_vector in enumerate(normalized):
                if right_pos == left_pos:
                    continue
                scored.append((_dot(left_vector, right_vector), right_pos))
            scored.sort(key=lambda item: (-item[0], keys[item[1]]))
            for score_value, right_pos in scored[:neighbors]:
                left = bucket[left_pos]
                right = bucket[right_pos]
                if left.key > right.key:
                    continue
                if score_value < threshold:
                    continue
                edges.append(
                    CanonicalCandidateEdge(
                        left_object_id=left.key,
                        right_object_id=right.key,
                        left_text=left.name,
                        right_text=right.name,
                        object_type=object_type,
                        method="semantic",
                        score=score_value,
                        features={
                            "embedding_model_id": embedder.model_id,
                            "top_k": top_k,
                        },
                    )
                )
    return sorted(edges, key=lambda edge: (-edge.score, edge.key))


def build_candidate_components(
    terms: list[RawContractObject],
    edges: list[CanonicalCandidateEdge],
) -> list[CanonicalCandidateComponent]:
    """Build connected components over candidate edges, preserving isolated terms."""

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
        if left_root == right_root:
            return
        parents[max(left_root, right_root)] = min(left_root, right_root)

    for edge in edges:
        if edge.left_object_id in parents and edge.right_object_id in parents:
            union(edge.left_object_id, edge.right_object_id)

    component_members: dict[str, list[str]] = defaultdict(list)
    for key in parents:
        component_members[find(key)].append(key)
    edges_by_component: dict[str, list[CanonicalCandidateEdge]] = defaultdict(list)
    for edge in edges:
        root = find(edge.left_object_id)
        edges_by_component[root].append(edge)

    components: list[CanonicalCandidateComponent] = []
    for root, members in component_members.items():
        member_ids = sorted(members)
        component_edges = sorted(edges_by_component.get(root, []), key=lambda edge: (-edge.score, edge.key))
        object_type = _component_object_type([by_key[key] for key in member_ids])
        scores = [edge.score for edge in component_edges]
        methods = sorted({edge.method for edge in component_edges})
        component_id = _component_id(object_type, member_ids)
        components.append(
            CanonicalCandidateComponent(
                component_id=component_id,
                object_type=object_type,
                member_ids=member_ids,
                candidate_edges=component_edges,
                max_score=max(scores) if scores else 0.0,
                mean_score=(sum(scores) / len(scores)) if scores else 0.0,
                methods_present=methods,
                ambiguous=_is_ambiguous_component(member_ids, component_edges),
            )
        )
    return sorted(components, key=lambda item: item.component_id)


def normalized_candidate_text(value: str) -> str:
    """Normalize candidate text without applying domain-deciding ontology aliases."""

    lowered = value.lower().replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"[^a-z0-9.]+", " ", lowered)
    return " ".join(cleaned.split())


def contract_object_type(kind: str) -> str:
    """Map interface kinds to broad canonicalization object types."""

    normalized = kind.lower().strip().replace("-", "_").replace(" ", "_")
    if normalized in {"state", "condition", "world_state", "physical_state", "environment_state"}:
        return "state"
    if normalized in {"belief_state", "belief", "memory_state", "knowledge_state", "observation_state"}:
        return "belief_state"
    if normalized in {"planning_state", "planning", "plan_state", "routing_state"}:
        return "planning_state"
    if normalized in {"credential", "environment", "text", "report", "data"}:
        return normalized
    return "artifact"


def _terms_by_object_type(terms: list[RawContractObject]) -> dict[str, list[RawContractObject]]:
    by_type: dict[str, list[RawContractObject]] = defaultdict(list)
    for term in terms:
        by_type[contract_object_type(term.kind)].append(term)
    return {
        object_type: sorted(bucket, key=lambda term: term.key)
        for object_type, bucket in by_type.items()
    }


def _component_object_type(terms: list[RawContractObject]) -> str:
    types = [contract_object_type(term.kind) for term in terms]
    if any(item in {"belief_state", "planning_state"} for item in types):
        if len(set(types) & {"belief_state", "planning_state"}) == 1:
            return next(item for item in ("belief_state", "planning_state") if item in types)
    for preferred in ("state", "credential", "environment", "artifact", "data", "report", "text"):
        if preferred in types:
            return preferred
    return "artifact"


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _semantic_text(term: RawContractObject) -> str:
    return " ".join(
        item
        for item in [
            term.name,
            term.kind,
            term.role,
            term.description,
        ]
        if item
    )


def _dedupe_edges(edges: list[CanonicalCandidateEdge]) -> list[CanonicalCandidateEdge]:
    by_key: dict[tuple[str, str, str], CanonicalCandidateEdge] = {}
    for edge in edges:
        existing = by_key.get(edge.key)
        if existing is None or edge.score > existing.score:
            by_key[edge.key] = edge
    return sorted(by_key.values(), key=lambda edge: (-edge.score, edge.key))


def _component_id(object_type: str, member_ids: list[str]) -> str:
    digest = hashlib.sha256("\n".join(member_ids).encode("utf-8")).hexdigest()[:12]
    return f"{object_type}:component:{digest}"


def _is_ambiguous_component(member_ids: list[str], edges: list[CanonicalCandidateEdge]) -> bool:
    if len(member_ids) > 8:
        return True
    by_node: dict[str, list[float]] = defaultdict(list)
    methods = {edge.method for edge in edges}
    for edge in edges:
        by_node[edge.left_object_id].append(edge.score)
        by_node[edge.right_object_id].append(edge.score)
    for scores in by_node.values():
        ordered = sorted(scores, reverse=True)
        if len(ordered) >= 2 and ordered[0] - ordered[1] < 0.04:
            return True
    if methods == {"lexical", "semantic"}:
        lexical = [edge.score for edge in edges if edge.method == "lexical"]
        semantic = [edge.score for edge in edges if edge.method == "semantic"]
        if lexical and semantic and max(semantic) - max(lexical) > 0.25:
            return True
    return False


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return list(vector)
    return [float(value) / norm for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=False))


def _hash_vector(text: str, dimension: int = 64) -> list[float]:
    vector = [0.0] * dimension
    for token in normalized_candidate_text(text).split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for index, byte in enumerate(digest[:dimension]):
            vector[index] += (byte / 255.0) - 0.5
    return _normalize_vector(vector)


def _load_embedding_cache(
    path: str | Path | None,
    model_id: str,
    cache_digest: str,
) -> dict[str, Any]:
    if path is None:
        return {"vectors": {}}
    target = Path(path)
    if not target.exists():
        return {"vectors": {}}
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"vectors": {}}
    if payload.get("model_id") != model_id or payload.get("cache_digest") != cache_digest:
        return {"vectors": {}}
    vectors = payload.get("vectors", {})
    if not isinstance(vectors, dict):
        vectors = {}
    return {"vectors": {str(key): [float(item) for item in value] for key, value in vectors.items() if isinstance(value, list)}}


def _write_embedding_cache(
    path: str | Path | None,
    model_id: str,
    cache_digest: str,
    vectors: dict[str, list[float]],
) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "cache_digest": cache_digest,
                "vectors": vectors,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
