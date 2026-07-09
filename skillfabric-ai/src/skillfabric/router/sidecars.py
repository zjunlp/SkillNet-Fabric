"""Load route-time sidecar artifacts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from skillfabric.compiled_graph.execution.models import ExecutionIndexRecord
from skillfabric.compiled_graph.interface.models import SkillInterface
from skillfabric.storage import Workspace


def load_interfaces(workspace: Workspace) -> dict[str, SkillInterface]:
    """Load skill interface sidecars if present."""

    path = _first_existing(workspace.graph_dir / "contracts.jsonl", workspace.graph_dir / "skill_interfaces.jsonl")
    if not path.exists():
        return {}
    return _load_interfaces_cached(*_file_cache_key(path))


@lru_cache(maxsize=16)
def _load_interfaces_cached(path: str, _mtime_ns: int, _size: int) -> dict[str, SkillInterface]:
    interfaces: dict[str, SkillInterface] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            interface = SkillInterface.from_dict(json.loads(line))
            interfaces[interface.skill_id] = interface
    return interfaces


def load_execution_index(workspace: Workspace) -> list[ExecutionIndexRecord]:
    """Load execution-index sidecars if present."""

    path = workspace.graph_dir / "execution_index.jsonl"
    if not path.exists():
        return []
    return _load_execution_index_cached(*_file_cache_key(path))


@lru_cache(maxsize=16)
def _load_execution_index_cached(path: str, _mtime_ns: int, _size: int) -> list[ExecutionIndexRecord]:
    return [
        ExecutionIndexRecord.from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _file_cache_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]
