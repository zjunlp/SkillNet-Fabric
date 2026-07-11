"""Compile SkillInterface fields into a canonical interface object registry."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.canonicalization.candidates import (
    CanonicalSemanticEmbedder,
    candidate_groups_from_terms,
    contract_object_type,
    normalized_candidate_text,
)
from skillfabric.compiled_graph.canonicalization.models import (
    CanonicalAssignment,
    CanonicalizationBuild,
    CanonicalizationCluster,
    CanonicalizationProvider,
    CanonicalObject,
    RawContractObject,
)
from skillfabric.compiled_graph.canonicalization.prompts import (
    CANONICALIZATION_PROMPT_ID,
    build_canonicalization_messages,
)
from skillfabric.compiled_graph.interface.models import InterfaceField, SkillInterface
from skillfabric.runtime.jobs import LLMJobOptions, run_llm_jobs
from skillfabric.runtime.llm import LLMConfig, litellm_completion, response_to_jsonable

CANONICALIZATION_CACHE_ID = CANONICALIZATION_PROMPT_ID


class LiteLLMCanonicalizationProvider:
    """LiteLLM-backed resolver for unresolved interface term groups."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @property
    def model_id(self) -> str:
        return self.config.model

    @classmethod
    def from_env(cls, *, env_path: str | Path | None = None) -> LiteLLMCanonicalizationProvider:
        return cls(config=LLMConfig.from_env(env_path=env_path))

    def canonicalize(self, cluster: CanonicalizationCluster) -> dict[str, Any]:
        response = litellm_completion(
            messages=build_canonicalization_messages(cluster),
            config=self.config,
            usage_operation="kg_build.canonicalization",
            usage_metadata={"cluster_id": cluster.cluster_id},
        )
        text = _extract_response_text(response)
        parsed = json.loads(_strip_fence(text))
        if not isinstance(parsed, dict):
            raise ValueError("canonicalization response must be a JSON object")
        _validate_provider_payload(parsed)
        return parsed


def canonicalize_contract_objects(
    interfaces: dict[str, SkillInterface],
    *,
    provider: CanonicalizationProvider | None = None,
    cache_path: str | Path | None = None,
    job_options: LLMJobOptions | None = None,
    semantic_embedder: CanonicalSemanticEmbedder | None = None,
) -> CanonicalizationBuild:
    """Canonicalize all requires/produces terms in a skill pool."""

    if provider is None:
        raise ValueError("canonicalization provider is required")
    raw_terms = _collect_raw_terms(interfaces)
    clusters = candidate_groups_from_terms(
        raw_terms,
        semantic_embedder=semantic_embedder,
        embedding_cache_path=_embedding_cache_path(cache_path),
    )
    cache = _load_cache(cache_path)
    results: dict[str, dict[str, Any]] = {}
    pending: list[CanonicalizationCluster] = []
    cache_hits = 0

    for cluster in clusters:
        cached = cache.get(_cache_key(cluster, provider.model_id))
        if isinstance(cached, dict):
            cache_hits += 1
            results[cluster.cluster_id] = cached
        else:
            pending.append(cluster)

    def canonicalize_one(cluster: CanonicalizationCluster) -> dict[str, Any]:
        raw = provider.canonicalize(cluster)
        if not isinstance(raw, dict):
            raise ValueError("canonicalization provider output must be a JSON object")
        _validate_provider_payload(raw)
        return raw

    def on_success(outcome) -> None:
        cluster = outcome.item
        raw = outcome.value
        if not isinstance(raw, dict):
            return
        cached_raw = dict(raw)
        cached_raw["_provenance"] = _provenance(provider.model_id)
        cache[_cache_key(cluster, provider.model_id)] = cached_raw
        results[cluster.cluster_id] = cached_raw
        _write_cache(cache_path, cache)

    outcomes = run_llm_jobs(
        pending,
        canonicalize_one,
        options=job_options,
        label="canonicalization",
        on_success=on_success,
    )
    for outcome in outcomes:
        if outcome.ok:
            continue
        cluster = outcome.item
        results[cluster.cluster_id] = {
            "_provenance": "provider_failed",
            "canonical_objects": [],
            "omitted_term_ids": [term.term_id for term in cluster.terms],
        }
    _write_cache(cache_path, cache)
    return _build_from_results(
        raw_terms,
        clusters,
        results,
        model_id=provider.model_id,
        llm_call_count=len(pending),
        cache_hit_count=cache_hits,
    )


def _collect_raw_terms(interfaces: dict[str, SkillInterface]) -> list[RawContractObject]:
    terms: list[RawContractObject] = []
    for skill_id, interface in sorted(interfaces.items()):
        terms.extend(_terms_from_fields(skill_id, "requires", interface.requires))
        terms.extend(_terms_from_fields(skill_id, "produces", interface.produces))
    return sorted(terms, key=lambda item: item.key)


def _terms_from_fields(skill_id: str, role: str, fields: list[InterfaceField]) -> list[RawContractObject]:
    output: list[RawContractObject] = []
    for field in fields:
        name = field.name.strip()
        if not name:
            continue
        output.append(
            RawContractObject(
                skill_id=skill_id,
                role=role,
                name=name,
                kind=field.kind,
                description=field.description,
                confidence=field.confidence,
                evidence=field.evidence,
            )
        )
    return output


def _build_from_results(
    raw_terms: list[RawContractObject],
    clusters: list[CanonicalizationCluster],
    results: dict[str, dict[str, Any]],
    *,
    model_id: str,
    llm_call_count: int = 0,
    cache_hit_count: int = 0,
) -> CanonicalizationBuild:
    terms_by_id = {term.term_id: term for term in raw_terms}
    cluster_terms = {cluster.cluster_id: {term.term_id for term in cluster.terms} for cluster in clusters}
    objects: dict[str, CanonicalObject] = {}
    assignments: dict[str, CanonicalAssignment] = {}
    assigned_term_ids: set[str] = set()

    for cluster_id, raw in sorted(results.items()):
        allowed_term_ids = cluster_terms.get(cluster_id, set())
        provenance = str(raw.get("_provenance") or _provenance(model_id))
        for item in raw.get("canonical_objects", []):
            if not isinstance(item, dict):
                continue
            name = _canonical_output_name(str(item.get("name", "")))
            object_type = contract_object_type(str(item.get("type", "artifact")))
            term_ids = [
                str(term_id)
                for term_id in item.get("term_ids", [])
                if str(term_id) in allowed_term_ids
                and str(term_id) in terms_by_id
                and str(term_id) not in assigned_term_ids
            ]
            term_ids = list(dict.fromkeys(term_ids))
            if not name or not term_ids:
                continue
            canonical_id = f"{object_type}:{name}"
            canonical = objects.setdefault(
                canonical_id,
                CanonicalObject(
                    canonical_id=canonical_id,
                    name=name,
                    type=object_type,
                    aliases=[],
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                    provenance=provenance,
                ),
            )
            canonical.confidence = max(canonical.confidence, float(item.get("confidence", 0.0) or 0.0))
            for term_id in term_ids:
                term = terms_by_id[term_id]
                canonical.aliases.append(term.name)
                if term.role == "requires":
                    canonical.required_by.append(term.skill_id)
                elif term.role == "produces":
                    canonical.produced_by.append(term.skill_id)
                assignments[term.key] = CanonicalAssignment(
                    raw_key=term.key,
                    skill_id=term.skill_id,
                    role=term.role,
                    raw_name=term.name,
                    raw_kind=term.kind,
                    canonical_id=canonical.canonical_id,
                )
                assigned_term_ids.add(term_id)

    for canonical in objects.values():
        canonical.aliases = sorted(set(canonical.aliases))
        canonical.required_by = sorted(set(canonical.required_by))
        canonical.produced_by = sorted(set(canonical.produced_by))

    return CanonicalizationBuild(
        objects=sorted(objects.values(), key=lambda item: item.canonical_id),
        assignments=sorted(assignments.values(), key=lambda item: item.raw_key),
        raw_terms=raw_terms,
        warnings=[],
        model_id=model_id,
        cluster_count=len(clusters),
        llm_call_count=llm_call_count,
        cache_hit_count=cache_hit_count,
        omitted_term_count=_omitted_term_count(raw_terms, assignments),
    )


def _validate_provider_payload(raw: dict[str, Any]) -> None:
    canonical_objects = raw.get("canonical_objects", [])
    omitted_term_ids = raw.get("omitted_term_ids", [])
    if not isinstance(canonical_objects, list):
        raise ValueError("canonical_objects must be a list")
    if not isinstance(omitted_term_ids, list):
        raise ValueError("omitted_term_ids must be a list")
    for item in canonical_objects:
        if not isinstance(item, dict):
            raise ValueError("canonical_objects items must be objects")
        if "name" not in item or "term_ids" not in item:
            raise ValueError("canonical object must include name and term_ids")
        if not isinstance(item.get("term_ids"), list):
            raise ValueError("canonical object term_ids must be a list")


def _canonical_output_name(value: str) -> str:
    text = normalized_candidate_text(value)
    return re.sub(r"_+", "_", "_".join(text.split())).strip("_")


def _provenance(model_id: str) -> str:
    del model_id
    return "llm_canonicalized"


def _cache_key(cluster: CanonicalizationCluster, model_id: str) -> str:
    raw = json.dumps(
        {
            "cache_id": CANONICALIZATION_CACHE_ID,
            "model_id": model_id,
            "cluster": cluster.to_dict(),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _omitted_term_count(
    raw_terms: list[RawContractObject],
    assignments: dict[str, CanonicalAssignment],
) -> int:
    assigned = {assignment.raw_key for assignment in assignments.values()}
    return sum(1 for term in raw_terms if term.key not in assigned)


def _embedding_cache_path(cache_path: str | Path | None) -> Path | None:
    if cache_path is None:
        return None
    return Path(cache_path).with_name("canonical_embeddings.json")


def _load_cache(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def _write_cache(path: str | Path | None, payload: dict[str, dict[str, Any]]) -> None:
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extract_response_text(response: Any) -> str:
    payload = response_to_jsonable(response)
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and message.get("content") is not None:
                    return str(message["content"])
                if first.get("text") is not None:
                    return str(first["text"])
        if payload.get("output_text") is not None:
            return str(payload["output_text"])
        output = payload.get("output")
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("text") is not None:
                            parts.append(str(part["text"]))
            if parts:
                return "\n".join(parts)
    return str(payload)


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped
