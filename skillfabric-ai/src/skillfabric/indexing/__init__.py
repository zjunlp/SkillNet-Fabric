"""Indexing package for BM25, embeddings, and manifests."""

from skillfabric.indexing.bm25 import build_bm25_index, search_bm25
from skillfabric.indexing.canonical import canonical_skill_text, hash_text
from skillfabric.indexing.embeddings import (
    DEFAULT_EMBEDDING_MODEL_ID,
    ApiEmbeddingProvider,
    DisabledEmbeddingProvider,
    build_embedding_store,
    cosine_similarity,
    default_embedding_provider,
    embed_query,
    embedding_provider_for_model,
    load_embedding_store,
    load_embedding_store_payload,
)
from skillfabric.indexing.neighbors import NeighborScore, build_neighbor_scores

__all__ = [
    "DEFAULT_EMBEDDING_MODEL_ID",
    "ApiEmbeddingProvider",
    "DisabledEmbeddingProvider",
    "NeighborScore",
    "build_bm25_index",
    "build_embedding_store",
    "build_neighbor_scores",
    "canonical_skill_text",
    "cosine_similarity",
    "default_embedding_provider",
    "embed_query",
    "embedding_provider_for_model",
    "hash_text",
    "load_embedding_store",
    "load_embedding_store_payload",
    "search_bm25",
]
