from __future__ import annotations

from pathlib import Path

import pytest

from skillfabric.registry.provenance import skill_pool_provenance
from skillfabric.registry.scanner import scan_and_parse


def _write_skill(root: Path, skill_id: str) -> Path:
    package = root / skill_id
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text(
        f"---\nname: {skill_id}\ndescription: Test Skill\n---\n\n# {skill_id}\n",
        encoding="utf-8",
    )
    return package


def test_skill_pool_provenance_separates_graph_inputs_from_package_files(
    tmp_path: Path,
) -> None:
    package = _write_skill(tmp_path, "example")
    initial = skill_pool_provenance(tmp_path, skills=scan_and_parse(tmp_path))

    (package / "resource.txt").write_text("resource\n", encoding="utf-8")
    with_resource = skill_pool_provenance(tmp_path, skills=scan_and_parse(tmp_path))

    assert with_resource["graph_input_sha256"] == initial["graph_input_sha256"]
    assert with_resource["package_sha256"] != initial["package_sha256"]


def test_skill_pool_provenance_ignores_appledouble_files(tmp_path: Path) -> None:
    package = _write_skill(tmp_path, "example")
    initial = skill_pool_provenance(tmp_path, skills=scan_and_parse(tmp_path))

    (package / "._SKILL.md").write_text("metadata\n", encoding="utf-8")

    assert skill_pool_provenance(
        tmp_path,
        skills=scan_and_parse(tmp_path),
    ) == initial


def test_skill_pool_provenance_rejects_symlinked_directories(tmp_path: Path) -> None:
    package = _write_skill(tmp_path / "pool", "example")
    external = tmp_path / "external"
    external.mkdir()
    (package / "linked").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        skill_pool_provenance(
            tmp_path / "pool",
            skills=scan_and_parse(tmp_path / "pool"),
        )
