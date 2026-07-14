"""Fail-closed validation of explorer output against a query wiki."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillfabric.compiled_graph.models import GRAPH_SCHEMA_VERSION, Edge
from skillfabric.router.models import (
    RouteNearMiss,
    RouterBundle,
    RouteRelationEvidence,
    RouteResult,
    RouteSelectedSkill,
)
from skillfabric.wiki.explorer.skill_package import SkillPackage


@dataclass(frozen=True, slots=True)
class SkillPackageValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
        }


def validate_skill_package(
    package: SkillPackage,
    query_wiki_root: Path,
    *,
    max_selected_skills: int = 8,
) -> SkillPackageValidationResult:
    """Validate selection ids and evidence without treating graph edges as authority."""

    root = query_wiki_root.resolve()
    manifest = _load_manifest(root)
    manifest_skills = {str(item["skill_id"]): item for item in manifest["skills"]}
    errors: list[str] = []
    selected_ids = [item.skill_id for item in package.selected_skills]
    selected_set = set(selected_ids)

    if len(selected_ids) > max(0, max_selected_skills):
        errors.append(
            f"selected skill count {len(selected_ids)} exceeds max_selected_skills={max_selected_skills}"
        )
    if len(selected_set) != len(selected_ids):
        errors.append("selected_skills contains duplicate skill ids")

    pages_read = list(package.wiki_pages_read)
    if len(set(pages_read)) != len(pages_read):
        errors.append("wiki_pages_read contains duplicate paths")
    for path in pages_read:
        error = _path_error(root, path, label="wiki page")
        if error:
            errors.append(error)
    pages_read_set = set(pages_read)

    for selected in package.selected_skills:
        row = manifest_skills.get(selected.skill_id)
        if row is None:
            errors.append(f"selected skill not in query_wiki manifest: {selected.skill_id}")
        elif not row.get("selectable", False):
            errors.append(f"selected skill is not selectable: {selected.skill_id}")
        if not selected.evidence:
            errors.append(f"selected skill has no evidence: {selected.skill_id}")
        own_evidence_paths = (
            {str(row["card_path"]), str(row["source_path"])} if row is not None else set()
        )
        cited_paths = {evidence.path for evidence in selected.evidence}
        for evidence in selected.evidence:
            error = _path_error(root, evidence.path, label="evidence path")
            if error:
                errors.append(error)
            if evidence.path not in pages_read_set:
                errors.append(
                    f"selected skill evidence was not declared in wiki_pages_read: {evidence.path}"
                )
        if row is not None and own_evidence_paths.isdisjoint(cited_paths):
            errors.append(
                f"selected skill evidence must cite its own card or source: {selected.skill_id}"
            )

    near_miss_ids: set[str] = set()
    for near_miss in package.near_misses:
        if near_miss.skill_id in near_miss_ids:
            errors.append(f"near_misses contains duplicate skill id: {near_miss.skill_id}")
        near_miss_ids.add(near_miss.skill_id)
        if near_miss.skill_id not in manifest_skills:
            errors.append(f"near miss not in query_wiki manifest: {near_miss.skill_id}")
        if near_miss.skill_id in selected_set:
            errors.append(f"near miss is also selected: {near_miss.skill_id}")

    if not selected_ids and not package.coverage_gaps:
        errors.append("empty selected_skills requires at least one explicit coverage gap")
    return SkillPackageValidationResult(valid=not errors, errors=tuple(errors))


def route_from_skill_package(
    package: SkillPackage,
    bundle: RouterBundle,
) -> RouteResult:
    """Convert selection to a route and attach graph relations as non-authoritative evidence."""

    candidates = {item.skill_id: item.name for item in bundle.selected_skills}
    candidates.update({item.skill_id: item.name for item in bundle.alternatives})
    selected_ids = {item.skill_id for item in package.selected_skills}
    selected = tuple(
        RouteSelectedSkill(
            skill_id=item.skill_id,
            name=candidates.get(item.skill_id, item.skill_id.removeprefix("skill:")),
            reason=item.role,
            evidence=tuple(evidence.path for evidence in item.evidence),
        )
        for item in package.selected_skills
    )
    relation_evidence = tuple(
        _route_relation(edge)
        for edge in sorted(
            bundle.graph_edges,
            key=lambda item: (item.type, item.source, item.target),
        )
        if edge.type in {"depend_on", "compose_with"}
        and edge.source in selected_ids
        and edge.target in selected_ids
    )
    return RouteResult(
        selected_skills=selected,
        relation_evidence=relation_evidence,
        near_misses=tuple(
            RouteNearMiss(skill_id=item.skill_id, reason=item.reason)
            for item in package.near_misses
        ),
        coverage_gaps=package.coverage_gaps,
        wiki_pages_read=package.wiki_pages_read,
        rationale=package.rationale,
    )


def _route_relation(edge: Edge) -> RouteRelationEvidence:
    return RouteRelationEvidence(
        relation_type=edge.type,
        source_skill=edge.source,
        target_skill=edge.target,
        confidence=edge.confidence,
        reason=edge.reason,
        evidence=tuple(f"{item.skill}:{item.line}" for item in edge.evidence),
    )


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "query",
        "skills",
        "semantic_edges_path",
        "alternatives",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("query_wiki manifest must use the canonical fields")
    if payload["schema_version"] != GRAPH_SCHEMA_VERSION:
        raise ValueError("query_wiki manifest schema is obsolete; rebuild the route")
    if not isinstance(payload["skills"], list):
        raise ValueError("query_wiki manifest skills must be a list")
    expected_skill_keys = {
        "skill_id",
        "name",
        "description",
        "selectable",
        "origin",
        "card_path",
        "source_path",
        "route",
        "alternative",
    }
    seen_skill_ids: set[str] = set()
    for index, item in enumerate(payload["skills"]):
        if not isinstance(item, dict) or set(item) != expected_skill_keys:
            raise ValueError(f"query_wiki manifest skills[{index}] has invalid fields")
        skill_id = item["skill_id"]
        if not isinstance(skill_id, str) or not skill_id.strip():
            raise ValueError(f"query_wiki manifest skills[{index}] has invalid skill_id")
        if skill_id in seen_skill_ids:
            raise ValueError(f"query_wiki manifest contains duplicate skill id: {skill_id}")
        seen_skill_ids.add(skill_id)
        if not isinstance(item["selectable"], bool):
            raise ValueError(f"query_wiki manifest skills[{index}] selectable must be boolean")
        for field_name in ("card_path", "source_path"):
            value = item[field_name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"query_wiki manifest skills[{index}] {field_name} must be non-empty"
                )
    return payload


def _path_error(root: Path, relative_path: str, *, label: str) -> str | None:
    path = Path(relative_path)
    if not relative_path or path.is_absolute():
        return f"{label} must be a non-empty relative query_wiki path: {relative_path}"
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return f"{label} escapes query_wiki: {relative_path}"
    if not candidate.is_file():
        return f"{label} is missing: {relative_path}"
    return None


__all__ = [
    "SkillPackageValidationResult",
    "route_from_skill_package",
    "validate_skill_package",
]
