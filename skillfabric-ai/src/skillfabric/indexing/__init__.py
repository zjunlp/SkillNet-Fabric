"""Indexing package for BM25, embeddings, and manifests."""

from skillfabric.indexing.bm25 import build_bm25_index, search_bm25
from skillfabric.indexing.canonical import canonical_skill_text, hash_text
from skillfabric.indexing.embeddings import (
    DEFAULT_EMBEDDING_MODEL_ID,
    ApiEmbeddingProvider,
    cosine_similarity,
    default_embedding_provider,
    embed_query,
    embedding_provider_for_model,
    load_skill_embedding_store,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL_ID",
    "ApiEmbeddingProvider",
    "build_bm25_index",
    "canonical_skill_text",
    "cosine_similarity",
    "default_embedding_provider",
    "embed_query",
    "embedding_provider_for_model",
    "hash_text",
    "load_skill_embedding_store",
    "search_bm25",
]
