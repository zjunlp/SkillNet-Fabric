"""Deterministic SkillPackage validation and RouteResult conversion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillfabric.router.models import RouteEdge, RouterBundle, RouteResult, RouteSelectedSkill
from skillfabric.router.route_edges import (
    _edges_from_ordered_skill_ids,
    _edges_from_workflow_hints,
    _merge_edges,
    _reconcile_route_edges,
)
from skillfabric.task_understanding import coverage_diagnostics
from skillfabric.wiki.explorer.skill_package import (
    SkillPackage,
    SkillPackageNearMiss,
    SkillPackageRequiredEdge,
    SkillPackageSelectedSkill,
)


@dataclass(slots=True)
class SkillPackageValidationResult:
    """Validation result for route-time explorer output."""

    valid: bool
    valid_package: SkillPackage
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "valid_package": self.valid_package.to_dict(),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def validate_skill_package(package: SkillPackage, query_wiki_root: Path) -> SkillPackageValidationResult:
    """Validate explorer output against query_wiki manifest and file boundaries."""

    manifest = json.loads((query_wiki_root / "manifest.json").read_text(encoding="utf-8"))
    manifest_skills = {item["skill_id"]: item for item in manifest.get("skills", []) if isinstance(item, dict)}
    selected: list[SkillPackageSelectedSkill] = []
    selected_ids: set[str] = set()
    errors: list[str] = []
    warnings: list[str] = []
    for skill in package.selected_skills:
        valid_evidence = []
        for evidence in skill.evidence:
            if not _path_is_inside(query_wiki_root, evidence.path):
                errors.append(f"evidence path escapes query_wiki: {evidence.path}")
                continue
            if not (query_wiki_root / evidence.path).exists():
                errors.append(f"evidence path missing: {evidence.path}")
                continue
            valid_evidence.append(evidence)
        row = manifest_skills.get(skill.skill_id)
        if row is None:
            errors.append(f"selected skill not in query_wiki manifest: {skill.skill_id}")
            continue
        if not row.get("selectable", False):
            errors.append(f"selected skill is not selectable: {skill.skill_id}")
            continue
        if skill.scope != row.get("scope"):
            errors.append(f"selected skill scope mismatch: {skill.skill_id}")
            continue
        if not valid_evidence:
            errors.append(f"selected skill has no valid evidence: {skill.skill_id}")
            continue
        selected.append(
            SkillPackageSelectedSkill(
                skill_id=skill.skill_id,
                scope=skill.scope,
                role=skill.role,
                evidence=valid_evidence,
            )
        )
        selected_ids.add(skill.skill_id)

    required_edges: list[SkillPackageRequiredEdge] = []
    for edge in package.required_edges:
        if edge.evidence_path:
            if not _path_is_inside(query_wiki_root, edge.evidence_path):
                errors.append(f"edge evidence path escapes query_wiki: {edge.evidence_path}")
                continue
            if not _valid_edge_evidence_path(edge.evidence_path):
                errors.append(f"edge evidence path must be edges/*.jsonl or workflows/*.md: {edge.evidence_path}")
                continue
            if not (query_wiki_root / edge.evidence_path).exists():
                errors.append(f"edge evidence path missing: {edge.evidence_path}")
                continue
        if edge.before not in selected_ids or edge.after not in selected_ids:
            warnings.append(f"dropped required edge whose endpoints are not selected: {edge.before} -> {edge.after}")
            continue
        required_edges.append(edge)

    near_misses: list[SkillPackageNearMiss] = []
    for near_miss in package.near_misses:
        if near_miss.skill_id not in manifest_skills:
            warnings.append(f"dropped near miss outside manifest: {near_miss.skill_id}")
            continue
        if near_miss.skill_id in selected_ids:
            warnings.append(f"dropped near miss already selected: {near_miss.skill_id}")
            continue
        near_misses.append(near_miss)

    valid_package = SkillPackage(
        selected_skills=selected,
        required_edges=required_edges,
        ordered_hints=[
            hint for hint in package.ordered_hints if hint.skill_id in selected_ids
        ],
        near_misses=near_misses,
        coverage_notes=list(package.coverage_notes),
        rationale=package.rationale,
    )
    return SkillPackageValidationResult(
        valid=bool(selected),
        valid_package=valid_package,
        errors=errors,
        warnings=warnings,
    )


def route_from_skill_package(
    package: SkillPackage,
    bundle: RouterBundle,
    *,
    query: str,
    trace_id: str,
    trace_dir: Path,
    warnings: list[str],
    max_selected_skills: int = 8,
) -> RouteResult:
    """Convert a validated SkillPackage to the stable RouteResult contract."""

    candidates = {item.skill_id: item for item in bundle.selected_skills}
    selected: list[RouteSelectedSkill] = []
    for item in package.selected_skills[:max_selected_skills]:
        candidate = candidates.get(item.skill_id)
        score = candidate.score if candidate is not None else 0.0
        name = candidate.name if candidate is not None else item.skill_id.removeprefix("skill:")
        selected.append(
            RouteSelectedSkill(
                skill_id=item.skill_id,
                name=name,
                rank=len(selected) + 1,
                score=score,
                reason=item.role,
                evidence=[evidence.path for evidence in item.evidence],
            )
        )
    selected_ids = {item.skill_id for item in selected}
    package_edges = [
        RouteEdge(
            before_skill=edge.before,
            after_skill=edge.after,
            edge_type=edge.relation_type,
            confidence=0.0,
            reason=edge.reason,
            source="wiki_agent",
        )
        for edge in package.required_edges
    ]
    hint_edges = _edges_from_ordered_skill_ids(
        [hint.skill_id for hint in package.ordered_hints],
        selected_ids,
        source="wiki_agent",
        warnings=warnings,
    )
    required_edges = _reconcile_route_edges(
        _merge_edges([*package_edges, *_edges_from_workflow_hints(bundle, selected_ids)]),
        hint_edges,
        warnings=warnings,
    )
    ordered_hints = _merge_edges([*hint_edges, *required_edges])
    return RouteResult(
        query=query,
        trace_id=trace_id,
        trace_dir=trace_dir,
        selected_skills=selected,
        required_edges=required_edges,
        ordered_hints=ordered_hints,
        near_misses=[item.to_dict() for item in package.near_misses],
        wiki_pages_read=[evidence for skill in selected for evidence in skill.evidence],
        task_understanding=bundle.task_understanding,
        coverage_diagnostics=coverage_diagnostics(bundle.task_understanding, selected_ids),
        rationale=package.rationale,
        provenance="claude_code",
        warnings=warnings,
    )


def _path_is_inside(root: Path, rel_path: str) -> bool:
    try:
        (root / rel_path).resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _valid_edge_evidence_path(path: str) -> bool:
    return (path.startswith("edges/") and path.endswith(".jsonl")) or (
        path.startswith("workflows/") and path.endswith(".md")
    ) or (
        path.startswith("skills/") and path.endswith(".md")
    )
