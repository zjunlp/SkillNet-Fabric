"""Deterministic provenance for scanned Skill pools."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from skillfabric.registry.models import SkillNode
from skillfabric.registry.scanner import scan_skill_root


def skill_pool_provenance(
    skill_root: str | Path,
    *,
    skills: Iterable[SkillNode],
) -> dict[str, str]:
    """Hash graph inputs and complete top-level Skill packages."""

    nodes = sorted(
        ({"id": skill.id, "content_hash": skill.content_hash} for skill in skills),
        key=lambda item: item["id"],
    )
    ids = [node["id"] for node in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("cannot fingerprint a Skill pool with duplicate IDs")
    return {
        "graph_input_sha256": _json_sha256(nodes),
        "package_sha256": _package_sha256(Path(skill_root)),
    }


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_sha256(skill_root: Path) -> str:
    if skill_root.is_symlink():
        raise ValueError(f"Skill packages may not contain symlinks: {skill_root}")
    root = skill_root.resolve()
    if root.is_file():
        files = [root]
        base = root.parent
    else:
        package_roots = [path.parent for path in scan_skill_root(root)]
        files = []
        for package in package_roots:
            if package.is_symlink():
                raise ValueError(f"Skill packages may not contain symlinks: {package}")
            for path in package.rglob("*"):
                if any(part.startswith("._") for part in path.parts):
                    continue
                if path.is_symlink():
                    raise ValueError(f"Skill packages may not contain symlinks: {path}")
                if path.is_file():
                    files.append(path)
        files.sort()
        base = root

    records: list[dict[str, Any]] = [
        {
            "path": path.relative_to(base).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in files
    ]
    return _json_sha256(records)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
