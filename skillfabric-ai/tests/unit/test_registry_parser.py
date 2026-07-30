from __future__ import annotations

from pathlib import Path

from skillfabric.registry.parser import parse_skill_file


def test_parser_preserves_normalized_name_longer_than_64_characters(tmp_path: Path) -> None:
    name = "professional-senior-chrome-extension-architect-and-developer-with-browser-automation"
    assert len(name) > 64
    skill_path = tmp_path / "long-name" / "SKILL.md"
    skill_path.parent.mkdir()
    skill_path.write_text(
        f"---\nname: {name}\ndescription: Long but valid Skill name.\n---\n\n# Body\n",
        encoding="utf-8",
    )

    skill = parse_skill_file(skill_path)

    assert skill.name == name
    assert skill.id == f"skill:{name}"


def test_parser_preserves_unicode_letters_when_normalizing_name(tmp_path: Path) -> None:
    skill_path = tmp_path / "unicode-name" / "SKILL.md"
    skill_path.parent.mkdir()
    skill_path.write_text(
        "---\nname: 设计 原则\ndescription: Unicode Skill name.\n---\n\n# Body\n",
        encoding="utf-8",
    )

    skill = parse_skill_file(skill_path)

    assert skill.name == "设计-原则"
    assert skill.id == "skill:设计-原则"
