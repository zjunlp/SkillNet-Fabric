from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.orchestrator.native_skills import (
    NativeSkillRuntimeError,
    prepare_native_skill_runtime,
)
from skillfabric.router.models import RouteResult, RouteSelectedSkill


def _route(workspace: Path, skill_ids: list[str]) -> RouteResult:
    return RouteResult(
        query="Create the requested artifacts.",
        trace_id="native-skill-test",
        trace_dir=workspace / "runs" / "native-skill-test",
        selected_skills=[
            RouteSelectedSkill(
                skill_id=skill_id,
                name=skill_id.split(":", 1)[-1],
                rank=index,
                reason="Selected by SkillFabric.",
            )
            for index, skill_id in enumerate(skill_ids, start=1)
        ],
        provenance="test",
    )


class NativeSkillRuntimeTests(unittest.TestCase):
    def test_prepare_native_skill_runtime_copies_only_selected_skill_directories(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / "skills"
            docx = skill_root / "docx"
            docx.mkdir(parents=True)
            (docx / "SKILL.md").write_text("# docx\n", encoding="utf-8")
            (docx / "ooxml.md").write_text("OOXML reference\n", encoding="utf-8")
            outside = skill_root / "outside"
            outside.mkdir()
            (outside / "SKILL.md").write_text("# outside\n", encoding="utf-8")
            execution_workspace = root / "workspace"

            result = prepare_native_skill_runtime(
                skill_root=skill_root,
                route=_route(root / ".skillfabric", ["skill:docx"]),
                execution_workspace=execution_workspace,
            )

            self.assertEqual(result.enabled_skill_names, ["docx"])
            self.assertEqual(result.native_skill_paths, [".claude/skills/docx"])
            self.assertIn(".claude/skills/docx/SKILL.md", result.copied_native_skill_files)
            self.assertIn(".claude/skills/docx/ooxml.md", result.copied_native_skill_files)
            self.assertTrue((execution_workspace / ".claude" / "skills" / "docx" / "SKILL.md").exists())
            self.assertTrue((execution_workspace / ".claude" / "skills" / "docx" / "ooxml.md").exists())
            self.assertFalse((execution_workspace / ".claude" / "skills" / "outside").exists())

    def test_prepare_native_skill_runtime_replaces_stale_installed_skills(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / "skills"
            docx = skill_root / "docx"
            docx.mkdir(parents=True)
            (docx / "SKILL.md").write_text("# docx\n", encoding="utf-8")
            execution_workspace = root / "workspace"
            stale = execution_workspace / ".claude" / "skills" / "old-skill"
            stale.mkdir(parents=True)
            (stale / "SKILL.md").write_text("# stale\n", encoding="utf-8")

            prepare_native_skill_runtime(
                skill_root=skill_root,
                route=_route(root / ".skillfabric", ["skill:docx"]),
                execution_workspace=execution_workspace,
            )

            self.assertTrue((execution_workspace / ".claude" / "skills" / "docx").is_dir())
            self.assertFalse(stale.exists())

    def test_prepare_native_skill_runtime_rejects_missing_selected_skill_directory(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / "skills"
            skill_root.mkdir()

            with self.assertRaisesRegex(NativeSkillRuntimeError, "missing selected native skill"):
                prepare_native_skill_runtime(
                    skill_root=skill_root,
                    route=_route(root / ".skillfabric", ["skill:docx"]),
                    execution_workspace=root / "workspace",
                )

    def test_prepare_native_skill_runtime_rejects_directory_without_skill_md(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / "skills"
            (skill_root / "docx").mkdir(parents=True)

            with self.assertRaisesRegex(NativeSkillRuntimeError, "SKILL.md"):
                prepare_native_skill_runtime(
                    skill_root=skill_root,
                    route=_route(root / ".skillfabric", ["skill:docx"]),
                    execution_workspace=root / "workspace",
                )


if __name__ == "__main__":
    unittest.main()
