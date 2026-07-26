"""Load canonical graph artifacts for wiki generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from skillfabric.compiled_graph.contracts.models import SkillContract
from skillfabric.compiled_graph.models import Edge, GraphDocument
from skillfabric.registry.models import SkillNode
from skillfabric.storage import Workspace

_SOURCE_ARTIFACTS = (
    "status.json",
    "graph/graph.json",
    "graph/registry.jsonl",
    "graph/contracts.jsonl",
)
_SourceArtifactIdentity = tuple[str, int, int, int, int]
_SourceCacheKey = tuple[str, tuple[_SourceArtifactIdentity, ...]]


@dataclass(slots=True)
class WikiSource:
    """Canonical graph data consumed by wiki renderers."""

    build_id: str
    skills: dict[str, SkillNode]
    contracts: dict[str, SkillContract]
    core_edges: list[Edge]

    @property
    def operational_edges(self) -> list[Edge]:
        return [edge for edge in self.core_edges if edge.type in {"depend_on", "compose_with"}]

    def skill_core_links(self, skill_id: str) -> list[Edge]:
        return [
            edge for edge in self.core_edges if edge.source == skill_id or edge.target == skill_id
        ]


_SOURCE_CACHE: dict[_SourceCacheKey, WikiSource] = {}
_SOURCE_CACHE_LOCK = Lock()


def load_wiki_source(workspace: Workspace) -> WikiSource:
    """Load one validated ready-workspace snapshot per process."""

    with _SOURCE_CACHE_LOCK:
        key = _workspace_source_cache_key(workspace)
        cached = _SOURCE_CACHE.get(key)
        if cached is not None:
            return cached
        source = _load_wiki_source_uncached(workspace)
        if _workspace_source_cache_key(workspace) != key:
            raise RuntimeError("workspace artifacts changed during WikiSource loading")
        _SOURCE_CACHE.clear()
        _SOURCE_CACHE[key] = source
        return source


def clear_wiki_source_cache() -> None:
    """Release the process-local ready-workspace snapshot."""

    with _SOURCE_CACHE_LOCK:
        _SOURCE_CACHE.clear()


def _load_wiki_source_uncached(workspace: Workspace) -> WikiSource:
    """Read and validate canonical graph, registry, and contract artifacts."""

    status_path = _required_path(workspace.status_path)
    graph_path = _required_path(workspace.graph_dir / "graph.json")
    registry_path = _required_path(workspace.graph_dir / "registry.jsonl")
    contracts_path = _required_path(workspace.graph_dir / "contracts.jsonl")
    status = _read_json_object(status_path)
    if (
        set(status) != {"state", "stage", "build_id"}
        or status.get("state") != "ready"
        or status.get("stage") != "complete"
    ):
        raise ValueError("SkillFabric workspace is not ready; complete a successful rebuild")
    graph = GraphDocument.from_dict(json.loads(graph_path.read_text(encoding="utf-8")))
    if status.get("build_id") != graph.build_id:
        raise ValueError("workspace status and graph build ids differ; rebuild the workspace")
    skills: dict[str, SkillNode] = {}
    for row in _read_jsonl(registry_path):
        skill = SkillNode.from_dict(row)
        if skill.id in skills:
            raise ValueError(f"registry contains duplicate skill id: {skill.id}")
        skills[skill.id] = skill
    contracts: dict[str, SkillContract] = {}
    for row in _read_jsonl(contracts_path):
        contract = SkillContract.from_dict(row)
        if contract.skill_id in contracts:
            raise ValueError(f"contracts contain duplicate skill id: {contract.skill_id}")
        contracts[contract.skill_id] = contract
    graph_skill_ids = {skill.id for skill in graph.nodes}
    if graph_skill_ids != set(skills) or graph_skill_ids != set(contracts):
        raise ValueError("graph, registry, and contract ids differ; rebuild the workspace")
    for skill_id, contract in contracts.items():
        if contract.content_hash != skills[skill_id].content_hash:
            raise ValueError(f"contract content hash differs for {skill_id}; rebuild the workspace")
    return WikiSource(
        build_id=graph.build_id,
        skills=skills,
        contracts=contracts,
        core_edges=graph.edges,
    )


def _workspace_source_cache_key(
    workspace: Workspace,
) -> _SourceCacheKey:
    root = workspace.root.resolve()
    identities: list[_SourceArtifactIdentity] = []
    for relative in _SOURCE_ARTIFACTS:
        path = root / relative
        try:
            identity = path.stat()
        except FileNotFoundError:
            return str(root), ()
        identities.append(
            (
                relative,
                identity.st_dev,
                identity.st_ino,
                identity.st_size,
                identity.st_mtime_ns,
            )
        )
    return str(root), tuple(identities)


def _required_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"required semantic artifact not found: {path}; run `skillfabric build`"
        )
    return path


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path} line {line_number} must be a JSON object")
            rows.append(payload)
    return rows
