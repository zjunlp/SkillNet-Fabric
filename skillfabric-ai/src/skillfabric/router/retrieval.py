"""Retrieval seed scoring for router bundles."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.execution.models import ExecutionIndexRecord
from skillfabric.compiled_graph.interface.models import SkillInterface
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
_GRAPH_GROUNDED_SCORE_SCALE = 14.0


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
        "disabled",
    }


def _lexical_scores(query: str, skills: dict[str, SkillNode]) -> dict[str, float]:
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return {}
    scores: dict[str, float] = {}
    for skill in skills.values():
        title_tokens = set(_tokens(f"{skill.name} {skill.description}"))
        body_tokens = set(_tokens(canonical_skill_text(skill)))
        title_overlap = len(query_tokens & title_tokens) / len(query_tokens)
        body_overlap = len(query_tokens & body_tokens) / len(query_tokens)
        if title_overlap or body_overlap:
            scores[skill.id] = min(1.0, (0.65 * title_overlap) + (0.20 * body_overlap))
    return scores


def apply_graph_grounded_scores(
    workspace: Workspace,
    query: str,
    skills: dict[str, SkillNode],
    seeds: dict[str, RouterSkillCandidate],
    *,
    interfaces: dict[str, SkillInterface],
    execution_index: list[ExecutionIndexRecord],
) -> None:
    """Add soft seed scores from compiled interfaces, canonical objects, and execution objects."""

    query_terms = set(_tokens(query))
    if query_terms:
        _add_soft_ranked_scores(
            seeds,
            skills,
            _interface_scores(query_terms, skills, interfaces),
            "interface:field",
        )
        object_hits = _object_hits(workspace, query_terms)
        for source, scores in object_hits.items():
            _add_soft_ranked_scores(seeds, skills, scores, source)
        _add_soft_ranked_scores(
            seeds,
            skills,
            _execution_object_scores(query_terms, skills, execution_index),
            "execution:object",
        )


def _add_soft_ranked_scores(
    seeds: dict[str, RouterSkillCandidate],
    skills: dict[str, SkillNode],
    scores: dict[str, float],
    source: str,
) -> None:
    for rank, (skill_id, raw_score) in enumerate(_rank_scores(scores), start=1):
        if raw_score <= 0:
            continue
        score = _GRAPH_GROUNDED_SCORE_SCALE / (_RRF_K + rank)
        _add_seed_score(seeds, skills, skill_id, score, source)


def _interface_scores(
    query_terms: set[str],
    skills: dict[str, SkillNode],
    interfaces: dict[str, SkillInterface],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for skill_id, interface in interfaces.items():
        if skill_id not in skills:
            continue
        score = _term_score(query_terms, _interface_text(interface, skills[skill_id]))
        if score > 0:
            scores[skill_id] = score
    return scores


def _interface_text(interface: SkillInterface, skill: SkillNode) -> str:
    fields = [
        interface.capability_summary,
        interface.when_to_use,
        interface.granularity,
        interface.execution_role,
        _fields_text(interface.requires),
        _fields_text(interface.produces),
        _fields_text(interface.uses_tools),
        canonical_skill_text(skill)[:1200],
    ]
    return " ".join(item for item in fields if item)


def _fields_text(fields: list[Any]) -> str:
    parts: list[str] = []
    for field in fields:
        parts.append(str(getattr(field, "name", "")))
        parts.append(str(getattr(field, "description", "")))
        for evidence in getattr(field, "evidence", []) or []:
            parts.append(str(getattr(evidence, "text", "")))
    return " ".join(item for item in parts if item)


def _object_hits(workspace: Workspace, query_terms: set[str]) -> dict[str, dict[str, float]]:
    hits: dict[str, dict[str, float]] = {
        "object:produces": {},
        "object:requires": {},
    }
    for payload in _read_jsonl(workspace.execution_dir / "canonical_objects.jsonl"):
        text = _canonical_object_text(payload)
        score = _term_score(query_terms, text)
        if score <= 0:
            continue
        for skill_id in _string_list(payload.get("produced_by", [])):
            hits["object:produces"][skill_id] = max(hits["object:produces"].get(skill_id, 0.0), score)
        for skill_id in _string_list(payload.get("required_by", [])):
            hits["object:requires"][skill_id] = max(hits["object:requires"].get(skill_id, 0.0), score)
    return {source: scores for source, scores in hits.items() if scores}


def _canonical_object_text(payload: dict[str, Any]) -> str:
    aliases = payload.get("aliases", [])
    return " ".join(
        [
            str(payload.get("canonical_id", "")),
            str(payload.get("name", "")),
            str(payload.get("type", "")),
            str(payload.get("description", "")),
            str(payload.get("reason", "")),
            " ".join(_string_list(aliases)),
        ]
    )


def _execution_object_scores(
    query_terms: set[str],
    skills: dict[str, SkillNode],
    execution_index: list[ExecutionIndexRecord],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for record in execution_index:
        score = _term_score(
            query_terms,
            " ".join(
                [
                    record.relation_type,
                    record.canonical_object,
                    record.projected_edge_type,
                    record.reason,
                ]
            ),
        )
        if score <= 0:
            continue
        for skill_id in (record.source_skill, record.target_skill):
            if skill_id in skills:
                scores[skill_id] = max(scores.get(skill_id, 0.0), score)
    return scores


def _term_score(query_terms: set[str], text: str) -> float:
    text_terms = set(_tokens(text))
    if not text_terms:
        return 0.0
    matches = query_terms & text_terms
    if not matches:
        return 0.0
    return (len(matches) / max(len(query_terms), 1)) + (0.05 * len(matches & _high_value_terms(text_terms)))


def _high_value_terms(terms: set[str]) -> set[str]:
    return {term for term in terms if len(term) >= 6}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


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
