"""Retrieval seed scoring for router bundles."""

from __future__ import annotations

import json
import re
from pathlib import Path

from skillfabric.indexing.bm25 import search_bm25
from skillfabric.indexing.canonical import canonical_skill_text
from skillfabric.indexing.embeddings import (
    cosine_similarity,
    embed_query,
    embedding_provider_for_model,
    load_embedding_store_payload,
)
from skillfabric.registry.models import SkillNode
from skillfabric.router.models import RouterSkillCandidate
from skillfabric.storage import Workspace

_RRF_K = 60
_RRF_SCORE_SCALE = 20.0


def _seed_scores(
    workspace: Workspace,
    query: str,
    skills: dict[str, SkillNode],
    *,
    warnings: list[str],
    env_file: str | Path | None = None,
) -> dict[str, RouterSkillCandidate]:
    seeds: dict[str, RouterSkillCandidate] = {}
    bm25_hits = search_bm25(workspace.index_dir / "bm25.sqlite", query, limit=max(len(skills), 10))
    _add_rrf_channel(seeds, skills, bm25_hits, "bm25")
    _add_rrf_channel(seeds, skills, _rank_scores(_lexical_scores(query, skills)), "lexical")

    embeddings_path = workspace.index_dir / "embeddings.json"
    if embeddings_path.exists():
        try:
            store = load_embedding_store_payload(embeddings_path)
            embeddings = store.vectors
            if not _embedding_store_is_reconstructable(store.provider):
                warnings.append(
                    "embedding search skipped: stored embedding provider "
                    f"{store.provider!r} is not available at route time"
                )
                return seeds
            if env_file is None:
                provider = embedding_provider_for_model(store.model_id, dimension=store.dimension)
            else:
                provider = embedding_provider_for_model(
                    store.model_id,
                    dimension=store.dimension,
                    env_path=env_file,
                )
            query_vector = embed_query(provider, query)
            embedding_scores: dict[str, float] = {}
            for skill_id, vector in embeddings.items():
                score = max(cosine_similarity(query_vector, vector), 0.0)
                if score > 0:
                    embedding_scores[skill_id] = score
            _add_rrf_channel(seeds, skills, _rank_scores(embedding_scores), "embedding")
        except (json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
            warnings.append(f"embedding search skipped: {exc}")
    else:
        warnings.append(f"embedding store not found: {embeddings_path}")
    return seeds


def _embedding_store_is_reconstructable(provider: str) -> bool:
    if not provider:
        return True
    normalized = provider.strip().lower()
    return normalized in {
        "api",
        "litellm",
        "openai",
        "sentence-transformers",
        "sentencetransformerembeddingprovider",
        "disabled",
    }


def _lexical_scores(query: str, skills: dict[str, SkillNode]) -> dict[str, float]:
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return {}
    intent_tokens = _intent_tokens(query_tokens)
    scores: dict[str, float] = {}
    for skill in skills.values():
        name_tokens = set(_tokens(skill.name))
        title_tokens = set(_tokens(f"{skill.name} {skill.description}"))
        body_tokens = set(_tokens(canonical_skill_text(skill)))
        title_overlap = len(query_tokens & title_tokens) / len(query_tokens)
        body_overlap = len(query_tokens & body_tokens) / len(query_tokens)
        name_intent_match = bool(intent_tokens & name_tokens)
        if title_overlap or body_overlap or name_intent_match:
            intent_boost = 0.25 if name_intent_match else 0.0
            scores[skill.id] = min(1.0, (0.65 * title_overlap) + (0.20 * body_overlap) + intent_boost)
    return scores


def _intent_tokens(query_tokens: set[str]) -> set[str]:
    synonyms = {
        "get": {"get", "take", "pick", "retrieve", "acquire"},
        "take": {"get", "take", "pick", "retrieve", "acquire"},
        "pick": {"get", "take", "pick", "retrieve", "acquire"},
        "retrieve": {"get", "take", "pick", "retrieve", "acquire"},
        "put": {"put", "place", "store"},
        "place": {"put", "place", "store"},
        "store": {"put", "place", "store"},
        "find": {"find", "locate", "search"},
        "locate": {"find", "locate", "search"},
        "search": {"find", "locate", "search"},
        "heat": {"heat", "warm", "cook"},
        "warm": {"heat", "warm", "cook"},
        "cook": {"heat", "warm", "cook"},
        "cool": {"cool", "chill"},
        "chill": {"cool", "chill"},
        "clean": {"clean", "wash"},
        "wash": {"clean", "wash"},
        "open": {"open"},
        "close": {"close"},
        "verify": {"verify", "check", "inspect"},
        "check": {"verify", "check", "inspect"},
        "inspect": {"verify", "check", "inspect"},
    }
    output = set(query_tokens)
    for token in query_tokens:
        output.update(synonyms.get(token, set()))
    return output


def _tokens(text: str) -> list[str]:
    return [_stem_token(token) for token in re.findall(r"[a-z0-9][a-z0-9_.+-]*", text.lower())]


def _rank_scores(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _add_rrf_channel(
    seeds: dict[str, RouterSkillCandidate],
    skills: dict[str, SkillNode],
    ranked_items: list[tuple[str, float]],
    source: str,
) -> None:
    for rank, (skill_id, raw_score) in enumerate(ranked_items, start=1):
        if raw_score <= 0:
            continue
        score = _RRF_SCORE_SCALE / (_RRF_K + rank)
        _add_seed_score(seeds, skills, skill_id, score, source)


def _stem_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 5 and token.endswith("er"):
        root = token[:-2]
        if root.endswith(("plac", "dispos", "retriev")):
            return f"{root}e"
        return root
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _add_seed_score(
    seeds: dict[str, RouterSkillCandidate],
    skills: dict[str, SkillNode],
    skill_id: str,
    score: float,
    source: str,
) -> None:
    skill = skills.get(skill_id)
    if skill is None:
        return
    candidate = seeds.setdefault(
        skill_id,
        RouterSkillCandidate(
            skill_id=skill_id,
            name=skill.name,
            score=0.0,
            sources=[],
            graph_depth=0,
            reason="query seed",
        ),
    )
    candidate.score += score
    candidate.seed_score += score
    candidate.score_breakdown[source] = candidate.score_breakdown.get(source, 0.0) + score
    candidate.sources.append(source)


def _apply_hard_include_scores(
    seeds: dict[str, RouterSkillCandidate],
    skills: dict[str, SkillNode],
    hard_includes: dict[str, list[str]],
) -> None:
    for skill_id, requirement_ids in hard_includes.items():
        for requirement_id in requirement_ids:
            _add_seed_score(seeds, skills, skill_id, 2.0, f"coverage:{requirement_id}")
        if skill_id in seeds:
            seeds[skill_id].reason = "required for explicit task coverage"
