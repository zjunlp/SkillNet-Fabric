"""Load compiled graph artifacts into a wiki-oriented view."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from skillfabric.compiled_graph.execution.models import (
    ArtifactNode,
    ExecutionEdge,
    ExecutionIndexRecord,
    ScenarioNode,
)
from skillfabric.compiled_graph.interface.models import SkillInterface
from skillfabric.compiled_graph.models import Edge, GraphDocument
from skillfabric.registry.models import SkillNode
from skillfabric.storage import Workspace


@dataclass(slots=True)
class WikiSource:
    """Compiled graph data reshaped for wiki generation."""

    build_id: str
    skills: dict[str, SkillNode]
    interfaces: dict[str, SkillInterface]
    raw_artifacts: dict[str, ArtifactNode]
    raw_scenarios: dict[str, ScenarioNode]
    core_edges: list[Edge]
    raw_skill_artifact_edges: list[ExecutionEdge]
    raw_skill_scenario_edges: list[ExecutionEdge]
    execution_index: list[ExecutionIndexRecord]
    evidence_lookup: dict[tuple[str, str, str], list[dict[str, Any]]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    def skill_core_links(self, skill_id: str) -> list[Edge]:
        return [
            edge
            for edge in self.core_edges
            if edge.source == skill_id or edge.target == skill_id
        ]

    def skill_execution_links(self, skill_id: str) -> dict[str, list[ExecutionEdge]]:
        return {
            "workflow_hints": [
                record
                for record in self.execution_index
                if record.source_skill == skill_id or record.target_skill == skill_id
            ],
        }


def load_wiki_source(workspace: Workspace) -> WikiSource:
    """Read compiled graph artifacts and return a wiki-oriented view."""

    compiled_path = workspace.graph_dir / "compiled.json"
    if not compiled_path.exists():
        raise FileNotFoundError(f"compiled skill graph not found: {compiled_path}; run `skillfabric build` first")
    payload = json.loads(compiled_path.read_text(encoding="utf-8"))
    core_graph = GraphDocument.from_dict(payload.get("core_graph", {}))
    skills = {
        node.id: node
        for node in core_graph.nodes
        if isinstance(node, SkillNode)
    }
    _merge_raw_skills(workspace, skills)
    interfaces = {
        interface.skill_id: interface
        for interface in (
            SkillInterface.from_dict(item)
            for item in payload.get("interfaces", [])
            if isinstance(item, dict)
        )
    }
    execution = payload.get("execution_graph", {})
    debug = execution.get("debug_extraction", {}) if isinstance(execution.get("debug_extraction", {}), dict) else {}
    raw_artifacts = {
        node.id: node
        for node in (
            ArtifactNode.from_dict(item)
            for item in debug.get("raw_artifact_nodes", [])
            if isinstance(item, dict)
        )
    }
    raw_scenarios = {
        node.id: node
        for node in (
            ScenarioNode.from_dict(item)
            for item in debug.get("raw_scenario_nodes", [])
            if isinstance(item, dict)
        )
    }
    raw_skill_artifact_edges = _execution_edges(debug.get("raw_skill_artifact_edges", []))
    raw_skill_scenario_edges = _execution_edges(debug.get("raw_skill_scenario_edges", []))
    execution_index = [
        ExecutionIndexRecord.from_dict(item)
        for item in execution.get("execution_index", [])
        if isinstance(item, dict)
    ]
    return WikiSource(
        build_id=core_graph.build_id,
        skills=skills,
        interfaces=interfaces,
        raw_artifacts=raw_artifacts,
        raw_scenarios=raw_scenarios,
        core_edges=core_graph.edges,
        raw_skill_artifact_edges=raw_skill_artifact_edges,
        raw_skill_scenario_edges=raw_skill_scenario_edges,
        execution_index=execution_index,
        evidence_lookup=_load_evidence_lookup(workspace),
        stats=dict(payload.get("stats", {})),
    )


def _merge_raw_skills(workspace: Workspace, skills: dict[str, SkillNode]) -> None:
    registry_path = workspace.graph_dir / "registry.jsonl"
    if not registry_path.exists():
        return
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        skill = SkillNode.from_dict(json.loads(line))
        if skill.id in skills:
            skills[skill.id] = skill


def _execution_edges(payload: Any) -> list[ExecutionEdge]:
    if not isinstance(payload, list):
        return []
    return [ExecutionEdge.from_dict(item) for item in payload if isinstance(item, dict)]


def _load_evidence_lookup(workspace: Workspace) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for path in (
        workspace.graph_dir / "edge_evidence.jsonl",
        workspace.graph_dir / "interface_evidence.jsonl",
        workspace.graph_dir / "execution_evidence.jsonl",
    ):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = _evidence_key(row)
            lookup.setdefault(key, []).append(row)
    return lookup


def _evidence_key(row: dict[str, Any]) -> tuple[str, str, str]:
    candidate = row.get("candidate")
    if isinstance(candidate, dict):
        return (
            str(candidate.get("source_skill", candidate.get("source", row.get("skill_id", "")))),
            str(candidate.get("target_skill", candidate.get("target", ""))),
            str(candidate.get("flow_type", candidate.get("edge_type", "execution"))),
        )
    edge = row.get("edge")
    if isinstance(edge, dict):
        return (str(edge.get("source", "")), str(edge.get("target", "")), str(edge.get("type", "")))
    return (str(row.get("skill_id", "")), "", "interface")
