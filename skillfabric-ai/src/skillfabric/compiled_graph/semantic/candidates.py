"""Contract-aware FAISS retrieval for semantic relation candidates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.contracts.models import ContractField, SkillContract
from skillfabric.compiled_graph.models import EvidenceRef
from skillfabric.compiled_graph.semantic.models import (
    CandidateHit,
    CandidatePair,
    CandidateRetrievalResult,
    EmbeddingKind,
    EmbeddingRecord,
)
from skillfabric.indexing.bm25 import search_bm25
from skillfabric.indexing.canonical import compact_contract_text, contract_skill_text
from skillfabric.indexing.embeddings import EmbeddingProvider
from skillfabric.indexing.ranking import reciprocal_rank_fusion
from skillfabric.registry.models import SkillNode
from skillfabric.storage import atomic_write_text

DEFAULT_CANDIDATE_TOP_K = 8
_EMBEDDING_SCHEMA_VERSION = "2.0"
_EMBEDDING_STORE_KEYS = {"schema_version", "model_id", "dimension", "records"}
_EMBEDDING_RECORD_KEYS = {
    "key",
    "skill_id",
    "kind",
    "field_name",
    "text_hash",
    "vector",
}
_CANONICAL_SKILL_ID_RE = re.compile(
    r"(?<![a-z0-9_.-])(skill:[a-z0-9][a-z0-9_.-]*)(?![a-z0-9_.-])",
    flags=re.IGNORECASE,
)
_MARKDOWN_CODE_RE = re.compile(r"`+([^`\n]+?)`+")


class CandidateRetrievalError(RuntimeError):
    """Raised when semantic candidate retrieval cannot be completed exactly."""


@dataclass(frozen=True, slots=True)
class _EmbeddingSpec:
    key: str
    skill_id: str
    kind: EmbeddingKind
    field_name: str
    text: str
    evidence: tuple[EvidenceRef, ...]

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def retrieve_candidate_pairs(
    contracts: dict[str, SkillContract],
    skills: list[SkillNode],
    *,
    provider: EmbeddingProvider,
    bm25_path: str | Path,
    store_path: str | Path | None = None,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
) -> CandidateRetrievalResult:
    """Retrieve bounded review candidates without assigning graph semantics."""

    _validate_top_k(candidate_top_k, name="candidate_top_k")
    ordered_skills = sorted(skills, key=lambda skill: skill.id)
    skill_ids = [skill.id for skill in ordered_skills]
    if len(skill_ids) != len(set(skill_ids)):
        raise CandidateRetrievalError("skill ids must be unique")
    if set(contracts) != set(skill_ids):
        missing = sorted(set(skill_ids) - set(contracts))
        extra = sorted(set(contracts) - set(skill_ids))
        raise CandidateRetrievalError(
            f"contract ids must exactly match skill ids; missing={missing}, extra={extra}"
        )
    for skill in ordered_skills:
        contract = contracts[skill.id]
        if contract.skill_id != skill.id or contract.content_hash != skill.content_hash:
            raise CandidateRetrievalError(f"contract identity does not match {skill.id}")

    specs = _embedding_specs(contracts, ordered_skills)
    records, cache_hits, embedded_count = _build_embeddings(
        specs,
        provider=provider,
        store_path=store_path,
    )
    records_by_key = {record.key: record for record in records}
    hits: dict[tuple[str, str], list[CandidateHit]] = defaultdict(list)
    _add_handoff_hits(
        hits,
        specs,
        records_by_key,
        top_k=candidate_top_k,
    )
    _add_similarity_hits(
        hits,
        specs,
        records_by_key,
        top_k=candidate_top_k,
    )
    _add_lexical_hits(
        hits,
        contracts,
        ordered_skills,
        bm25_path=Path(bm25_path),
        top_k=candidate_top_k,
    )
    _add_explicit_reference_hits(hits, ordered_skills)

    selected_hits = _select_candidate_hits(
        hits,
        skill_ids=skill_ids,
        top_k=candidate_top_k,
    )
    pairs = tuple(_candidate_pairs(selected_hits))
    channel_counts = {
        channel: sum(1 for pair in pairs if channel in pair.channels)
        for channel in ("handoff", "explicit_reference", "similarity", "lexical")
    }
    dimension = len(records[0].vector) if records else 0
    metrics: dict[str, int | float | str] = {
        "model_id": str(provider.model_id),
        "dimension": dimension,
        "embedded_record_count": len(records),
        "new_embedding_count": embedded_count,
        "cache_hit_count": cache_hits,
        "candidate_pair_count": len(pairs),
        "handoff_pair_count": channel_counts["handoff"],
        "explicit_reference_pair_count": channel_counts["explicit_reference"],
        "similarity_pair_count": channel_counts["similarity"],
        "lexical_pair_count": channel_counts["lexical"],
    }
    return CandidateRetrievalResult(pairs=pairs, metrics=metrics)


def _embedding_specs(
    contracts: dict[str, SkillContract],
    skills: list[SkillNode],
) -> list[_EmbeddingSpec]:
    specs: list[_EmbeddingSpec] = []
    for skill in skills:
        contract = contracts[skill.id]
        specs.append(
            _EmbeddingSpec(
                key=f"skill:{skill.id}",
                skill_id=skill.id,
                kind="skill",
                field_name="",
                text=contract_skill_text(skill, contract),
                evidence=contract.evidence,
            )
        )
        for kind, fields in (("requires", contract.requires), ("produces", contract.produces)):
            for index, field in enumerate(fields):
                specs.append(
                    _EmbeddingSpec(
                        key=f"{kind}:{skill.id}:{index}",
                        skill_id=skill.id,
                        kind=kind,
                        field_name=field.name,
                        text=_field_text(field),
                        evidence=field.evidence,
                    )
                )
    return specs


def _field_text(field: ContractField) -> str:
    evidence = " ".join(item.text for item in field.evidence)
    return "\n".join(part for part in (field.name, field.description, evidence) if part).strip()


def _build_embeddings(
    specs: list[_EmbeddingSpec],
    *,
    provider: EmbeddingProvider,
    store_path: str | Path | None,
) -> tuple[list[EmbeddingRecord], int, int]:
    expected_dimension = _provider_dimension(provider)
    cached = _load_embedding_cache(
        store_path,
        model_id=str(provider.model_id),
        expected_dimension=expected_dimension,
    )
    records: list[EmbeddingRecord | None] = [None] * len(specs)
    pending: list[tuple[int, _EmbeddingSpec]] = []
    cache_hits = 0
    for index, spec in enumerate(specs):
        old = cached.get(spec.key)
        if old is not None and old.text_hash == spec.text_hash:
            records[index] = old
            cache_hits += 1
        else:
            pending.append((index, spec))

    vectors = _embed_many(provider, [spec.text for _, spec in pending])
    if len(vectors) != len(pending):
        raise CandidateRetrievalError(
            f"embedding provider returned {len(vectors)} vectors for {len(pending)} texts"
        )
    for (index, spec), vector in zip(pending, vectors, strict=True):
        records[index] = EmbeddingRecord(
            key=spec.key,
            skill_id=spec.skill_id,
            kind=spec.kind,
            field_name=spec.field_name,
            text_hash=spec.text_hash,
            vector=tuple(float(value) for value in vector),
        )
    complete = [record for record in records if record is not None]
    _validate_vectors(complete, expected_dimension=expected_dimension)
    expected_keys = {spec.key for spec in specs}
    if pending or set(cached) != expected_keys:
        _write_embedding_cache(store_path, model_id=str(provider.model_id), records=complete)
    return complete, cache_hits, len(pending)


def _embed_many(provider: EmbeddingProvider, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    embed_many = getattr(provider, "embed_many", None)
    if callable(embed_many):
        return [list(map(float, vector)) for vector in embed_many(texts)]
    return [list(map(float, provider.embed(text))) for text in texts]


def _validate_vectors(
    records: list[EmbeddingRecord],
    *,
    expected_dimension: int | None = None,
) -> None:
    if not records:
        return
    dimensions = {len(record.vector) for record in records}
    if 0 in dimensions:
        raise CandidateRetrievalError("embedding vectors must be non-empty")
    if len(dimensions) != 1:
        raise CandidateRetrievalError("embedding vectors must share one dimension")
    if expected_dimension is not None and dimensions != {expected_dimension}:
        raise CandidateRetrievalError(
            f"embedding vectors must match provider dimension {expected_dimension}"
        )
    if any(not math.isfinite(value) for record in records for value in record.vector):
        raise CandidateRetrievalError("embedding vectors must contain only finite values")
    if any(not any(value != 0.0 for value in record.vector) for record in records):
        raise CandidateRetrievalError("embedding vectors must have non-zero norm")


def _add_handoff_hits(
    hits: dict[tuple[str, str], list[CandidateHit]],
    specs: list[_EmbeddingSpec],
    records: dict[str, EmbeddingRecord],
    *,
    top_k: int,
) -> None:
    if top_k <= 0:
        return
    requirements = [spec for spec in specs if spec.kind == "requires"]
    products = [spec for spec in specs if spec.kind == "produces"]
    if not requirements or not products:
        return
    requirement_vectors = [records[spec.key].vector for spec in requirements]
    same_skill_counts: dict[str, int] = defaultdict(int)
    requirements_by_name: dict[str, list[_EmbeddingSpec]] = defaultdict(list)
    for spec in requirements:
        same_skill_counts[spec.skill_id] += 1
        requirements_by_name[_exact_field_key(spec.field_name)].append(spec)
    max_fields_per_skill = max(same_skill_counts.values())
    for product in products:
        exact_skill_ids: set[str] = set()
        for requirement in requirements_by_name[_exact_field_key(product.field_name)]:
            if requirement.skill_id == product.skill_id or requirement.skill_id in exact_skill_ids:
                continue
            exact_skill_ids.add(requirement.skill_id)
            _append_hit(
                hits,
                CandidateHit(
                    channel="handoff",
                    query_skill=product.skill_id,
                    matched_skill=requirement.skill_id,
                    rank=1,
                    query_field=f"produces:{product.field_name}",
                    matched_field=f"requires:{requirement.field_name}",
                    evidence=product.evidence + requirement.evidence,
                ),
            )
        search_k = min(
            len(requirements),
            same_skill_counts[product.skill_id]
            + ((top_k + len(exact_skill_ids)) * max_fields_per_skill),
        )
        neighbors = _faiss_search(
            requirement_vectors,
            [records[product.key].vector],
            search_k=search_k,
        )[0]
        matched_skills: set[str] = set(exact_skill_ids)
        approximate_rank = 0
        for index, _score in neighbors:
            requirement = requirements[index]
            if requirement.skill_id == product.skill_id or requirement.skill_id in matched_skills:
                continue
            matched_skills.add(requirement.skill_id)
            approximate_rank += 1
            _append_hit(
                hits,
                CandidateHit(
                    channel="handoff",
                    query_skill=product.skill_id,
                    matched_skill=requirement.skill_id,
                    rank=approximate_rank,
                    query_field=f"produces:{product.field_name}",
                    matched_field=f"requires:{requirement.field_name}",
                    evidence=product.evidence + requirement.evidence,
                ),
            )
            if approximate_rank >= top_k:
                break


def _exact_field_key(value: str) -> str:
    normalized = value.strip()
    if len(normalized) >= 2 and normalized.startswith("`") and normalized.endswith("`"):
        normalized = normalized.strip("`").strip()
    return " ".join(normalized.casefold().split())


def _add_similarity_hits(
    hits: dict[tuple[str, str], list[CandidateHit]],
    specs: list[_EmbeddingSpec],
    records: dict[str, EmbeddingRecord],
    *,
    top_k: int,
) -> None:
    if top_k <= 0:
        return
    documents = [spec for spec in specs if spec.kind == "skill"]
    if len(documents) < 2:
        return
    vectors = [records[spec.key].vector for spec in documents]
    rows = _faiss_search(vectors, vectors, search_k=min(len(documents), top_k + 1))
    for source_index, neighbors in enumerate(rows):
        source = documents[source_index]
        rank = 0
        for target_index, _score in neighbors:
            target = documents[target_index]
            if target.skill_id == source.skill_id:
                continue
            rank += 1
            _append_hit(
                hits,
                CandidateHit(
                    channel="similarity",
                    query_skill=source.skill_id,
                    matched_skill=target.skill_id,
                    rank=rank,
                    evidence=source.evidence + target.evidence,
                ),
            )
            if rank >= top_k:
                break


def _add_lexical_hits(
    hits: dict[tuple[str, str], list[CandidateHit]],
    contracts: dict[str, SkillContract],
    skills: list[SkillNode],
    *,
    bm25_path: Path,
    top_k: int,
) -> None:
    if top_k <= 0:
        return
    search_limit = min(len(skills), top_k + 1)
    for skill in skills:
        rank = 0
        query = compact_contract_text(skill, contracts[skill.id])
        for result in search_bm25(bm25_path, query, limit=search_limit):
            if result.skill_id == skill.id:
                continue
            rank += 1
            _append_hit(
                hits,
                CandidateHit(
                    channel="lexical",
                    query_skill=skill.id,
                    matched_skill=result.skill_id,
                    rank=rank,
                    evidence=contracts[skill.id].evidence
                    + contracts[result.skill_id].evidence,
                ),
            )
            if rank >= top_k:
                break


def _add_explicit_reference_hits(
    hits: dict[tuple[str, str], list[CandidateHit]],
    skills: list[SkillNode],
) -> None:
    aliases: dict[str, set[str]] = defaultdict(set)
    for skill in skills:
        aliases[skill.id.casefold()].add(skill.id)
        aliases[skill.name.casefold()].add(skill.id)
    seen: set[tuple[str, str, int]] = set()
    for skill in skills:
        for line_number, line in enumerate(skill.raw_text.splitlines(), start=1):
            for alias in sorted(_structured_references(line)):
                for target_id in sorted(aliases.get(alias, ())):
                    if target_id == skill.id:
                        continue
                    marker = (skill.id, target_id, line_number)
                    if marker in seen:
                        continue
                    seen.add(marker)
                    _append_hit(
                        hits,
                        CandidateHit(
                            channel="explicit_reference",
                            query_skill=skill.id,
                            matched_skill=target_id,
                            rank=1,
                            evidence=(EvidenceRef(skill=skill.id, line=line_number, text=line),),
                        ),
                    )


def _structured_references(line: str) -> set[str]:
    references = {match.group(1).casefold() for match in _CANONICAL_SKILL_ID_RE.finditer(line)}
    references.update(
        match.group(1).strip().casefold()
        for match in _MARKDOWN_CODE_RE.finditer(line)
        if match.group(1).strip()
    )
    return references


def _append_hit(
    hits: dict[tuple[str, str], list[CandidateHit]],
    hit: CandidateHit,
) -> None:
    if hit.query_skill == hit.matched_skill:
        return
    key = tuple(sorted((hit.query_skill, hit.matched_skill)))
    hits[key].append(hit)


def _select_candidate_hits(
    hits: dict[tuple[str, str], list[CandidateHit]],
    *,
    skill_ids: list[str],
    top_k: int,
) -> dict[tuple[str, str], list[CandidateHit]]:
    selected = {
        key
        for key, pair_hits in hits.items()
        if any(hit.channel == "explicit_reference" for hit in pair_hits)
    }
    hits_by_query: dict[str, dict[str, list[CandidateHit]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pair_hits in hits.values():
        for hit in pair_hits:
            if hit.channel != "explicit_reference":
                hits_by_query[hit.query_skill][hit.channel].append(hit)

    for skill_id in skill_ids:
        channels: dict[str, list[str]] = {}
        for channel, channel_hits in hits_by_query.get(skill_id, {}).items():
            ordered = sorted(
                channel_hits,
                key=lambda hit: (hit.rank, hit.matched_skill, hit.query_field, hit.matched_field),
            )
            channels[channel] = list(dict.fromkeys(hit.matched_skill for hit in ordered))
        for matched_skill in channels.get("handoff", ())[:top_k]:
            selected.add(tuple(sorted((skill_id, matched_skill))))
        for fused in reciprocal_rank_fusion(channels)[:top_k]:
            selected.add(tuple(sorted((skill_id, fused.skill_id))))
    return {key: hits[key] for key in sorted(selected)}


def _candidate_pairs(
    hits: dict[tuple[str, str], list[CandidateHit]],
) -> list[CandidatePair]:
    pairs: list[CandidatePair] = []
    channel_order = {"handoff": 0, "explicit_reference": 1, "similarity": 2, "lexical": 3}
    for (skill_a, skill_b), pair_hits in hits.items():
        unique: dict[tuple[Any, ...], CandidateHit] = {}
        for hit in pair_hits:
            key = (
                hit.channel,
                hit.query_skill,
                hit.matched_skill,
                hit.rank,
                hit.query_field,
                hit.matched_field,
            )
            unique.setdefault(key, hit)
        ordered_hits = tuple(
            sorted(
                unique.values(),
                key=lambda hit: (
                    channel_order[hit.channel],
                    hit.rank,
                    hit.query_skill,
                    hit.matched_skill,
                    hit.query_field,
                    hit.matched_field,
                ),
            )
        )
        pairs.append(CandidatePair(skill_a=skill_a, skill_b=skill_b, hits=ordered_hits))
    pairs.sort(
        key=lambda pair: (
            0 if "handoff" in pair.channels else 1,
            0 if "explicit_reference" in pair.channels else 1,
            pair.key,
        )
    )
    return pairs


def _faiss_search(
    index_vectors: list[tuple[float, ...]],
    query_vectors: list[tuple[float, ...]],
    *,
    search_k: int,
) -> list[list[tuple[int, float]]]:
    if not index_vectors or not query_vectors or search_k <= 0:
        return [[] for _ in query_vectors]
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise CandidateRetrievalError(
            "semantic candidate retrieval requires faiss-cpu and numpy"
        ) from exc
    index_matrix = np.ascontiguousarray(np.asarray(index_vectors, dtype="float32"))
    query_matrix = np.ascontiguousarray(np.asarray(query_vectors, dtype="float32"))
    if index_matrix.ndim != 2 or query_matrix.ndim != 2:
        raise CandidateRetrievalError("FAISS vectors must be two-dimensional matrices")
    if index_matrix.shape[1] != query_matrix.shape[1]:
        raise CandidateRetrievalError("FAISS index and query dimensions must match")
    faiss.normalize_L2(index_matrix)
    faiss.normalize_L2(query_matrix)
    index = faiss.IndexFlatIP(index_matrix.shape[1])
    index.add(index_matrix)
    scores, indices = index.search(query_matrix, min(search_k, len(index_vectors)))
    return [
        [
            (int(index_value), float(score))
            for index_value, score in zip(index_row, score_row, strict=True)
            if int(index_value) >= 0
        ]
        for score_row, index_row in zip(scores, indices, strict=True)
    ]


def _load_embedding_cache(
    path: str | Path | None,
    *,
    model_id: str,
    expected_dimension: int,
) -> dict[str, EmbeddingRecord]:
    if path is None or not Path(path).exists():
        return {}
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateRetrievalError(f"failed to read embedding store: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _EMBEDDING_STORE_KEYS:
        raise CandidateRetrievalError("embedding store must use the exact schema-v2 fields")
    if payload.get("schema_version") != _EMBEDDING_SCHEMA_VERSION:
        raise CandidateRetrievalError("embedding store schema is obsolete; rebuild the workspace")
    stored_model_id = payload.get("model_id")
    if not isinstance(stored_model_id, str) or not stored_model_id.strip():
        raise CandidateRetrievalError("embedding store model_id must be a non-empty string")
    stored_dimension = payload.get("dimension")
    if (
        isinstance(stored_dimension, bool)
        or not isinstance(stored_dimension, int)
        or stored_dimension <= 0
    ):
        raise CandidateRetrievalError("embedding store dimension must be a positive integer")
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise CandidateRetrievalError("embedding store records must be a list")
    records: dict[str, EmbeddingRecord] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise CandidateRetrievalError("embedding store records must be objects")
        if set(row) != _EMBEDDING_RECORD_KEYS:
            raise CandidateRetrievalError(f"embedding store record {index} has invalid fields")
        try:
            kind = row["kind"]
            if kind not in {"skill", "requires", "produces"}:
                raise ValueError(f"unsupported embedding kind: {kind}")
            vector = row["vector"]
            if not isinstance(vector, list):
                raise TypeError("vector must be a list")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector
            ):
                raise TypeError("vector values must be numbers")
            record = EmbeddingRecord(
                key=_cache_string(row["key"], label="key"),
                skill_id=_cache_string(row["skill_id"], label="skill_id"),
                kind=kind,
                field_name=_cache_string(row["field_name"], label="field_name", allow_empty=True),
                text_hash=_cache_string(row["text_hash"], label="text_hash"),
                vector=tuple(float(value) for value in vector),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CandidateRetrievalError(f"invalid embedding store record: {exc}") from exc
        if record.key in records:
            raise CandidateRetrievalError(f"embedding store contains duplicate key: {record.key}")
        records[record.key] = record
    _validate_vectors(list(records.values()), expected_dimension=stored_dimension)
    if stored_model_id != model_id or stored_dimension != expected_dimension:
        return {}
    return records


def _write_embedding_cache(
    path: str | Path | None,
    *,
    model_id: str,
    records: list[EmbeddingRecord],
) -> None:
    if path is None:
        return
    dimension = len(records[0].vector) if records else 0
    payload = {
        "schema_version": _EMBEDDING_SCHEMA_VERSION,
        "model_id": model_id,
        "dimension": dimension,
        "records": [record.to_dict() for record in records],
    }
    atomic_write_text(
        Path(path),
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def _cache_string(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        requirement = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{label} must be {requirement}")
    return value


def _provider_dimension(provider: EmbeddingProvider) -> int:
    dimension = getattr(provider, "dimension", None)
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise CandidateRetrievalError("embedding provider dimension must be a positive integer")
    return dimension


def _validate_top_k(value: Any, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CandidateRetrievalError(f"{name} must be a non-negative integer")
