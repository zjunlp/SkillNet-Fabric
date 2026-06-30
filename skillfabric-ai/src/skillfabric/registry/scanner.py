"""Scan SKILL.md files."""

from __future__ import annotations

from pathlib import Path


def scan_skill_root(skill_root: str | Path) -> list[Path]:
    """Scan all SKILL.md files under a skill root."""

    root = Path(skill_root)
    if not root.exists():
        raise FileNotFoundError(f"skill root not found: {root}")
    if root.is_file():
        if root.name != "SKILL.md":
            raise ValueError(f"skill file must be named SKILL.md: {root}")
        return [root]
    return sorted(path for path in root.rglob("SKILL.md") if path.is_file())


def scan_and_parse(skill_root: str | Path, *, workspace: str | Path = ".skillfabric"):
    """Scan and parse a skill root."""

    from skillfabric.registry.parser import parse_skill_file

    return [parse_skill_file(path, workspace=workspace) for path in scan_skill_root(skill_root)]
