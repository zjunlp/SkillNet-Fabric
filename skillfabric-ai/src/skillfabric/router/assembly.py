"""Artifact loading and context assembly for router bundles."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from skillfabric.compiled_graph.execution.models import ExecutionIndexRecord
from skillfabric.compiled_graph.models import GraphDocument
from skillfabric.registry.models import SkillNode
from skillfabric.router.models import RouterWorkflowHint
from skillfabric.storage import Workspace
from skillfabric.wiki.pages import slug


def _load_graph(workspace: Workspace) -> GraphDocument:
    path = workspace.graph_dir / "graph.json"
    if not path.exists():
        return GraphDocument(schema_version="1.0", build_id="", nodes=[], edges=[], stats={}, config_digest="")
    return _load_graph_cached(*_file_cache_key(path))


def _load_registry_skills(workspace: Workspace) -> dict[str, SkillNode]:
    path = _first_existing(workspace.graph_dir / "registry.jsonl", workspace.graph_dir / "skills.jsonl")
    if not path.exists():
        return {}
    return _load_registry_skills_cached(*_file_cache_key(path))


@lru_cache(maxsize=16)
def _load_graph_cached(path: str, _mtime_ns: int, _size: int) -> GraphDocument:
    return GraphDocument.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@lru_cache(maxsize=16)
def _load_registry_skills_cached(path: str, _mtime_ns: int, _size: int) -> dict[str, SkillNode]:
    skills: dict[str, SkillNode] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            skill = SkillNode.from_dict(json.loads(line))
            skills[skill.id] = skill
    return skills


def _file_cache_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _workflow_hints(
    workspace: Workspace,
    selected_ids: set[str],
    *,
    confidence_threshold: float,
    limit: int,
) -> list[RouterWorkflowHint]:
    path = workspace.graph_dir / "execution_index.jsonl"
    if not path.exists() or limit == 0:
        return []
    hints: list[RouterWorkflowHint] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = ExecutionIndexRecord.from_dict(json.loads(line))
        if record.source_skill not in selected_ids or record.target_skill not in selected_ids:
            continue
        if record.confidence < confidence_threshold:
            continue
        hints.append(
            RouterWorkflowHint(
                source_skill=record.source_skill,
                target_skill=record.target_skill,
                relation_type=record.relation_type,
                canonical_object=record.canonical_object,
                confidence=record.confidence,
                projected_edge_type=record.projected_edge_type,
                reason=record.reason,
            )
        )
    hints.sort(key=lambda item: (-item.confidence, item.source_skill, item.target_skill, item.canonical_object))
    return hints[:limit]


def _wiki_pages(
    workspace: Workspace,
    selected_ids: set[str],
) -> list[str]:
    pages: list[Path] = []
    for skill_id in sorted(selected_ids):
        path = workspace.wiki_skill_cards_dir / f"{slug(skill_id)}.md"
        if path.exists():
            pages.append(path)
    return [str(path) for path in pages if "/debug/" not in str(path)]
