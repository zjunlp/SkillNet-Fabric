"""Prepare selected skills as native Claude Code skill directories."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from skillfabric.router.models import RouteResult
from skillfabric.wiki.pages import slug


class NativeSkillRuntimeError(RuntimeError):
    """Raised when selected skills cannot be installed for native runtime use."""


@dataclass(slots=True)
class NativeSkillRuntimeResult:
    """Native Claude Code skill runtime installation result."""

    enabled_skill_names: list[str]
    native_skill_paths: list[str]
    copied_native_skill_files: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "enabled_skill_names": list(self.enabled_skill_names),
            "native_skill_paths": list(self.native_skill_paths),
            "copied_native_skill_files": list(self.copied_native_skill_files),
        }


def prepare_native_skill_runtime(
    *,
    skill_root: str | Path,
    route: RouteResult,
    execution_workspace: str | Path,
) -> NativeSkillRuntimeResult:
    """Install only route-selected original skill directories into `.claude/skills`."""

    source_root = Path(skill_root)
    workspace = Path(execution_workspace)
    target_root = workspace / ".claude" / "skills"
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    enabled_skill_names: list[str] = []
    native_skill_paths: list[str] = []
    copied_files: list[str] = []
    seen: set[str] = set()

    for selected in route.selected_skills:
        native_name = slug(selected.skill_id)
        if native_name in seen:
            continue
        seen.add(native_name)
        source = source_root / native_name
        if not source.is_dir():
            raise NativeSkillRuntimeError(
                f"missing selected native skill directory for {selected.skill_id}: {source}"
            )
        if not (source / "SKILL.md").is_file():
            raise NativeSkillRuntimeError(
                f"selected native skill {selected.skill_id} is missing SKILL.md: {source / 'SKILL.md'}"
            )
        target = target_root / native_name
        shutil.copytree(source, target)
        enabled_skill_names.append(native_name)
        native_skill_paths.append(target.relative_to(workspace).as_posix())
        copied_files.extend(
            path.relative_to(workspace).as_posix()
            for path in sorted(target.rglob("*"))
            if path.is_file()
        )

    return NativeSkillRuntimeResult(
        enabled_skill_names=enabled_skill_names,
        native_skill_paths=native_skill_paths,
        copied_native_skill_files=copied_files,
    )
