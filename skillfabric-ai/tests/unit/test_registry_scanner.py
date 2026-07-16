from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.registry.scanner import scan_skill_root


def _write_skill(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("---\nname: test\ndescription: Test skill.\n---\n", encoding="utf-8")
    return path


class RegistryScannerTests(unittest.TestCase):
    def test_directory_scan_returns_only_top_level_skill_packages(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            top_level = _write_skill(root / "top-level" / "SKILL.md")
            _write_skill(root / "top-level" / "references" / "SKILL.md")
            _write_skill(root / "._appledouble" / "SKILL.md")
            (root / "top-level" / "._SKILL.md").write_text("ignored", encoding="utf-8")

            self.assertEqual(scan_skill_root(root), [top_level])

    def test_direct_skill_file_input_is_preserved(self) -> None:
        with TemporaryDirectory() as directory:
            skill_file = _write_skill(Path(directory) / "standalone" / "SKILL.md")

            self.assertEqual(scan_skill_root(skill_file), [skill_file])
