"""Compile SkillContract fields into a canonical object registry."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.canonicalization.candidates import (
    CanonicalCandidateComponent,
    CanonicalCandidateEdge,
    CanonicalSemanticEmbedder,
    build_candidate_graph,
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


class DeterministicCanonicalizationProvider:
    """Rule-based canonicalizer used as offline fallback."""

    model_id = "deterministic-canonicalization"

    def canonicalize(self, cluster: CanonicalizationCluster) -> dict[str, Any]:
        canonical_name = _deterministic_canonical_name(cluster.terms, cluster.object_type)
        if not canonical_name or _is_generic(canonical_name):
            return {
                "canonical_objects": [],
                "assignments": [],
            }
        return {
            "canonical_objects": [
                {
                    "canonical_name": canonical_name,
                    "type": cluster.object_type,
                    "description": f"Canonical {cluster.object_type} object for {canonical_name}.",
                    "aliases": sorted({term.name for term in cluster.terms}),
                    "promoted": True,
                    "confidence": 0.76 if cluster.candidate_edges else 0.7,
                    "reason": "Deterministic candidate-graph canonicalization.",
                }
            ],
            "assignments": [
                {
                    "raw_name": term.name,
                    "canonical_name": canonical_name,
                    "confidence": 0.76 if cluster.candidate_edges else 0.7,
                    "reason": "Deterministic candidate-graph canonicalization.",
                }
                for term in cluster.terms
            ],
        }


class LiteLLMCanonicalizationProvider:
    """LiteLLM-backed pool-level canonicalization provider."""

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
            usage_metadata={"cluster_id": cluster.cluster_id, "object_type": cluster.object_type},
        )
        text = _extract_response_text(response)
        parsed = json.loads(_strip_fence(text))
        if not isinstance(parsed, dict):
            raise ValueError("canonicalization response must be a JSON object")
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

    provider = provider or DeterministicCanonicalizationProvider()
    raw_terms = _collect_raw_terms(interfaces)
    candidate_result = build_candidate_graph(
        raw_terms,
        semantic_embedder=semantic_embedder,
        embedding_cache_path=_embedding_cache_path(cache_path),
        cache_digest=_raw_terms_digest(raw_terms),
    )
    clusters = _clusters_from_components(raw_terms, candidate_result.components)
    cache = _load_cache(cache_path)
    results: dict[str, dict[str, Any]] = {}
    pending: list[CanonicalizationCluster] = []

    for cluster in clusters:
        deterministic = _deterministic_cluster_result(cluster)
        if deterministic is not None:
            results[cluster.cluster_id] = deterministic
            continue
        key = _cache_key(cluster, provider.model_id)
        cached = cache.get(key)
        if isinstance(cached, dict):
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
    fallback = DeterministicCanonicalizationProvider()
    for outcome in outcomes:
        if outcome.ok:
            continue
        cluster = outcome.item
        raw = fallback.canonicalize(cluster)
        raw["_provenance"] = "deterministic_fallback"
        results[cluster.cluster_id] = raw
    _write_cache(cache_path, cache)
    return _build_from_results(
        raw_terms,
        clusters,
        results,
        model_id=provider.model_id,
        candidate_edges=candidate_result.edges,
        candidate_components=candidate_result.components,
        warnings=candidate_result.warnings,
    )


def _collect_raw_terms(interfaces: dict[str, SkillInterface]) -> list[RawContractObject]:
    terms: list[RawContractObject] = []
    for skill_id, interface in sorted(interfaces.items()):
        terms.extend(_terms_from_fields(skill_id, "requires", interface.requires))
        terms.extend(_terms_from_fields(skill_id, "produces", interface.produces))
    return terms


def _terms_from_fields(skill_id: str, role: str, fields: list[InterfaceField]) -> list[RawContractObject]:
    output: list[RawContractObject] = []
    for field in fields:
        name = field.name.strip()
        if not name or _is_generic(name):
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


def _clusters_from_components(
    raw_terms: list[RawContractObject],
    components: list[CanonicalCandidateComponent],
) -> list[CanonicalizationCluster]:
    terms_by_key = {term.key: term for term in raw_terms}
    clusters: list[CanonicalizationCluster] = []
    for component in components:
        terms = [terms_by_key[key] for key in component.member_ids if key in terms_by_key]
        if not terms:
            continue
        clusters.append(
            CanonicalizationCluster(
                cluster_id=component.component_id,
                object_type=component.object_type,
                terms=sorted(terms, key=lambda item: item.key),
                candidate_edges=[edge.to_dict() for edge in component.candidate_edges],
                ambiguous=component.ambiguous,
                methods_present=list(component.methods_present),
            )
        )
    return sorted(clusters, key=lambda item: item.cluster_id)


def _deterministic_cluster_result(cluster: CanonicalizationCluster) -> dict[str, Any] | None:
    if not _deterministic_cluster_eligible(cluster):
        return None
    canonical_name = _deterministic_canonical_name(cluster.terms, cluster.object_type)
    if not canonical_name or _is_generic(canonical_name):
        return {
            "_provenance": "deterministic_exact",
            "canonical_objects": [],
            "assignments": [],
        }
    reason = "Deterministic singleton or normalized-exact contract term."
    return {
        "_provenance": "deterministic_exact",
        "canonical_objects": [
            {
                "canonical_name": canonical_name,
                "type": cluster.object_type,
                "description": f"Canonical {cluster.object_type} object for {canonical_name}.",
                "aliases": sorted({term.name for term in cluster.terms}),
                "promoted": True,
                "confidence": 0.95,
                "reason": reason,
            }
        ],
        "assignments": [
            {
                "raw_name": term.name,
                "canonical_name": canonical_name,
                "confidence": 0.95,
                "reason": reason,
            }
            for term in _unique_terms_by_name(cluster.terms)
        ],
    }


def _deterministic_cluster_eligible(cluster: CanonicalizationCluster) -> bool:
    if cluster.ambiguous:
        return False
    if len(cluster.terms) == 1:
        return True
    normalized_names = {_normalize_name(term.name) for term in cluster.terms}
    normalized_names.discard("")
    return len(normalized_names) == 1


def _unique_terms_by_name(terms: list[RawContractObject]) -> list[RawContractObject]:
    by_name: dict[str, RawContractObject] = {}
    for term in terms:
        by_name.setdefault(term.name.lower(), term)
    return sorted(by_name.values(), key=lambda item: item.key)


def _cluster_object_type(terms: list[RawContractObject]) -> str:
    types = [_object_type(term.kind) for term in terms]
    if any(item in {"belief_state", "planning_state"} for item in types):
        if len(set(types) & {"belief_state", "planning_state"}) == 1:
            return next(item for item in ("belief_state", "planning_state") if item in types)
    for preferred in ("state", "credential", "environment", "artifact", "data", "report", "text"):
        if preferred in types:
            return preferred
    return "artifact"


def _build_from_results(
    raw_terms: list[RawContractObject],
    clusters: list[CanonicalizationCluster],
    results: dict[str, dict[str, Any]],
    *,
    model_id: str,
    candidate_edges: list[CanonicalCandidateEdge],
    candidate_components: list[CanonicalCandidateComponent],
    warnings: list[str],
) -> CanonicalizationBuild:
    raw_by_cluster = {cluster.cluster_id: cluster for cluster in clusters}
    objects: dict[str, CanonicalObject] = {}
    assignments: list[CanonicalAssignment] = []
    term_lookup: dict[tuple[str, str], list[RawContractObject]] = defaultdict(list)
    for cluster in clusters:
        for term in cluster.terms:
            term_lookup[(term.name.lower(), cluster.cluster_id)].append(term)
    seen_assignments: set[tuple[str, str]] = set()

    for cluster_id, raw in sorted(results.items()):
        cluster = raw_by_cluster[cluster_id]
        object_by_name: dict[str, CanonicalObject] = {}
        for item in raw.get("canonical_objects", []):
            if not isinstance(item, dict):
                continue
            name = _canonical_output_name(str(item.get("canonical_name", "")))
            object_type = _object_type(str(item.get("type", cluster.object_type)))
            if not _cluster_object_type_compatible(cluster.object_type, object_type):
                object_type = cluster.object_type
            if not name:
                continue
            canonical_id = f"{object_type}:{name}"
            provenance = str(raw.get("_provenance") or _provenance(model_id))
            canonical = objects.setdefault(
                canonical_id,
                CanonicalObject(
                    canonical_id=canonical_id,
                    name=name,
                    type=object_type,
                    description=str(item.get("description", "")),
                    aliases=[str(alias) for alias in item.get("aliases", [])],
                    promoted=bool(item.get("promoted", True)),
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                    provenance=provenance,
                    reason=str(item.get("reason", "")),
                ),
            )
            canonical.aliases.extend(str(alias) for alias in item.get("aliases", []))
            canonical.confidence = max(canonical.confidence, float(item.get("confidence", 0.0) or 0.0))
            canonical.promoted = canonical.promoted and bool(item.get("promoted", True))
            object_by_name[name] = canonical

        for item in raw.get("assignments", []):
            if not isinstance(item, dict):
                continue
            raw_name = str(item.get("raw_name", ""))
            terms = term_lookup.get((raw_name.lower(), cluster_id), [])
            if not terms:
                continue
            canonical_name = _canonical_output_name(str(item.get("canonical_name", "")))
            canonical = object_by_name.get(canonical_name)
            if canonical is None:
                continue
            for term in terms:
                if not _assignment_type_compatible(term, canonical.type):
                    continue
                dedupe_key = (term.key, canonical.canonical_id)
                if dedupe_key in seen_assignments:
                    continue
                seen_assignments.add(dedupe_key)
                assignment = CanonicalAssignment(
                    raw_key=term.key,
                    skill_id=term.skill_id,
                    role=term.role,
                    raw_name=term.name,
                    raw_kind=term.kind,
                    canonical_id=canonical.canonical_id,
                    confidence=float(item.get("confidence", canonical.confidence) or 0.0),
                    reason=str(item.get("reason", canonical.reason)),
                    provenance=canonical.provenance,
                )
                assignments.append(assignment)
                if term.role == "requires":
                    canonical.required_by.append(term.skill_id)
                elif term.role == "produces":
                    canonical.produced_by.append(term.skill_id)
                canonical.aliases.append(term.name)

    for canonical in objects.values():
        canonical.required_by = sorted(set(canonical.required_by))
        canonical.produced_by = sorted(set(canonical.produced_by))
        canonical.aliases = sorted(set(canonical.aliases))
        canonical.reuse_count = len(set(canonical.required_by) | set(canonical.produced_by))
        canonical.promoted = _should_promote(canonical)

    return CanonicalizationBuild(
        objects=sorted(objects.values(), key=lambda item: item.canonical_id),
        assignments=sorted(assignments, key=lambda item: item.raw_key),
        raw_terms=sorted(raw_terms, key=lambda item: item.key),
        candidate_edges=sorted(candidate_edges, key=lambda item: (item.left_object_id, item.right_object_id, item.method)),
        candidate_components=sorted(candidate_components, key=lambda item: item.component_id),
        warnings=list(warnings),
        model_id=model_id,
    )


def _validate_provider_payload(raw: dict[str, Any]) -> None:
    for key in ("canonical_objects", "assignments"):
        if key in raw and not isinstance(raw[key], list):
            raise ValueError(f"{key} must be a list")


def _should_promote(canonical: CanonicalObject) -> bool:
    if canonical.confidence < 0.7:
        return False
    if _is_generic(canonical.name):
        return False
    if _is_broad_handoff_name(canonical.name):
        return False
    if canonical.type in {"belief_state", "planning_state"}:
        return False
    if canonical.type in {"state", "credential", "environment"}:
        return canonical.reuse_count >= 2 or bool(canonical.required_by and canonical.produced_by)
    return bool(canonical.required_by and canonical.produced_by)


def _deterministic_canonical_name(terms: list[RawContractObject], object_type: str) -> str:
    candidates = [
        normalized_candidate_text(term.name)
        for term in terms
        if normalized_candidate_text(term.name) and not _is_generic(normalized_candidate_text(term.name))
    ]
    if not candidates:
        return ""
    ranked = sorted(candidates, key=lambda item: (len(item.split()), len(item), item))
    return _canonical_output_name(ranked[0])


def _object_type(kind: str) -> str:
    return contract_object_type(kind)


def _normalize_name(value: str) -> str:
    return normalized_candidate_text(value)


def _canonical_output_name(value: str) -> str:
    return "_".join(_normalize_name(value).split())


def _is_generic(value: str) -> bool:
    normalized = _canonical_output_name(value)
    return normalized in {
        "artifact",
        "content",
        "data",
        "file",
        "input",
        "object",
        "output",
        "result",
        "state",
        "text",
    }


def _is_broad_handoff_name(value: str) -> bool:
    tokens = set(_normalize_name(value).split())
    if not tokens:
        return True
    if _looks_like_context_or_input_placeholder(tokens):
        return True
    if _looks_like_path_placeholder(tokens):
        return True
    return False


def _looks_like_context_or_input_placeholder(tokens: set[str]) -> bool:
    broad_context_tokens = {
        "context",
        "input",
        "material",
        "materials",
        "reference",
        "references",
        "snippet",
        "snippets",
        "source",
    }
    if not (tokens & broad_context_tokens):
        return False
    informative_tokens = tokens - broad_context_tokens - {
        "artifact",
        "artifacts",
        "data",
        "document",
        "documents",
        "file",
        "files",
        "project",
        "task",
        "text",
    }
    return not informative_tokens


def _looks_like_path_placeholder(tokens: set[str]) -> bool:
    if not ({"path", "directory"} & tokens):
        return False
    return tokens <= {"directory", "file", "input", "output", "path", "project", "target"}


def _cluster_object_type_compatible(cluster_type: str, object_type: str) -> bool:
    if cluster_type == "state":
        return object_type == "state"
    if cluster_type in {"belief_state", "planning_state"}:
        return object_type == cluster_type
    if object_type in {"belief_state", "planning_state"}:
        return cluster_type == object_type
    return True


def _assignment_type_compatible(term: RawContractObject, canonical_type: str) -> bool:
    raw_type = _object_type(term.kind)
    if raw_type in {"belief_state", "planning_state"}:
        return canonical_type == raw_type
    if canonical_type in {"belief_state", "planning_state"}:
        return raw_type == canonical_type
    if canonical_type == "state":
        return raw_type == "state" or _looks_like_world_state(term.name)
    return raw_type != "state" or canonical_type == "state"


def _looks_like_world_state(value: str) -> bool:
    tokens = set(_normalize_name(value).split())
    phrase = " ".join(tokens)
    if not tokens:
        return False
    if {"inventory", "held", "holding", "open", "closed", "clean", "cleaned", "heated", "cooled"} & tokens:
        return True
    if "located" in tokens or "accessible" in tokens or "authenticated" in tokens:
        return True
    if "in_hand" in value.lower() or "in_inventory" in value.lower():
        return True
    return any(item in phrase for item in ("object inventory", "object hand", "receptacle open", "receptacle closed"))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _provenance(model_id: str) -> str:
    if model_id == DeterministicCanonicalizationProvider.model_id:
        return "deterministic_fallback"
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


def _raw_terms_digest(raw_terms: list[RawContractObject]) -> str:
    payload = [term.to_dict() for term in sorted(raw_terms, key=lambda item: item.key)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


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
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
