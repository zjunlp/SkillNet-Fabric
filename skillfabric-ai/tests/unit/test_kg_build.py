from __future__ import annotations

import pytest

from skillfabric.compiled_graph.models import Edge, GraphDocument
from skillfabric.indexing.canonical import canonical_skill_text
from skillfabric.registry.parser import parse_skill_file
from skillfabric.registry.scanner import scan_skill_root

FIXTURE_SKILLS = pytest.importorskip("tests.unit.wiki_helpers").FIXTURE_SKILLS


def test_parser_uses_frontmatter_and_keeps_source_out_of_canonical_node() -> None:
    skill = parse_skill_file(FIXTURE_SKILLS / "pdf-table-parser" / "SKILL.md")

    assert skill.id == "skill:pdf-table-parser"
    assert skill.name == "pdf-table-parser"
    assert "Extract tables" in skill.description
    assert "pdf-table-parser" in canonical_skill_text(skill)
    payload = skill.to_dict(include_raw_text=False)
    assert set(payload) == {"id", "type", "name", "description", "content_hash"}
    assert "raw_text" not in payload


def test_parser_rejects_a_skill_without_required_frontmatter(tmp_path) -> None:
    skill_file = tmp_path / "missing-frontmatter" / "SKILL.md"
    skill_file.parent.mkdir()
    skill_file.write_text("# Missing frontmatter\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML frontmatter"):
        parse_skill_file(skill_file)


def test_parser_rejects_invalid_yaml_instead_of_using_a_partial_parser(tmp_path) -> None:
    skill_file = tmp_path / "invalid-yaml" / "SKILL.md"
    skill_file.parent.mkdir()
    skill_file.write_text(
        "---\nname: [unterminated\n---\n\n# Invalid\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid YAML frontmatter"):
        parse_skill_file(skill_file)


def test_parser_normalizes_frontmatter_name_to_kebab_case(tmp_path) -> None:
    skill_file = tmp_path / "api-integration-specialist" / "SKILL.md"
    skill_file.parent.mkdir()
    skill_file.write_text(
        "---\n"
        "name: API Integration Specialist\n"
        "description: Integrate external APIs.\n"
        "---\n\n"
        "# API Integration Specialist\n",
        encoding="utf-8",
    )

    skill = parse_skill_file(skill_file)

    assert skill.id == "skill:api-integration-specialist"
    assert skill.name == "api-integration-specialist"


def test_skill_node_loader_rejects_unknown_serialized_fields() -> None:
    payload = parse_skill_file(FIXTURE_SKILLS / "pdf-table-parser" / "SKILL.md").to_dict(
        include_raw_text=True
    )
    payload["unused"] = True

    with pytest.raises(ValueError, match="skill node fields"):
        type(parse_skill_file(FIXTURE_SKILLS / "pdf-table-parser" / "SKILL.md")).from_dict(payload)


def test_scan_returns_all_skill_documents_in_stable_order() -> None:
    paths = scan_skill_root(FIXTURE_SKILLS)

    assert len(paths) == 7
    assert paths == sorted(paths)


def test_graph_document_uses_only_current_canonical_fields() -> None:
    valid = {
        "build_id": "build",
        "nodes": [],
        "edges": [],
    }

    assert GraphDocument.from_dict(valid).build_id == "build"
    with pytest.raises(ValueError, match="canonical fields"):
        GraphDocument.from_dict({**valid, "communities": []})


def test_only_alternative_edges_require_canonical_endpoint_order() -> None:
    directional = {
        "source": "skill:z-upstream",
        "target": "skill:a-downstream",
        "confidence": 0.9,
        "evidence": [],
        "reason": "Validated relation.",
    }

    assert Edge.from_dict({**directional, "type": "compose_with"}).source == "skill:z-upstream"
    with pytest.raises(ValueError, match="alternative edge endpoints"):
        Edge.from_dict({**directional, "type": "similar_to"})
