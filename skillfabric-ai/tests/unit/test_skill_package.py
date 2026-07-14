from __future__ import annotations

import json
from typing import Any

import pytest

from skillfabric.router.bundle import RouterBundleConfig, build_router_bundle
from skillfabric.wiki.explorer.skill_package import SkillPackage
from skillfabric.wiki.explorer.validation import route_from_skill_package, validate_skill_package
from skillfabric.wiki.query_wiki import materialize_query_wiki
from tests.unit.fake_embeddings import FakeEmbeddingProvider
from tests.unit.wiki_helpers import build_fixture_workspace


def _package(**overrides: Any) -> SkillPackage:
    payload: dict[str, Any] = {
        "selected_skills": [
            {
                "skill_id": "skill:pdf-table-parser",
                "role": "Parse PDF tables into normalized data.",
                "evidence": [{"path": "skills/cards/pdf-table-parser.md"}],
            },
            {
                "skill_id": "skill:financial-kpi-extractor",
                "role": "Extract financial KPIs from normalized tables.",
                "evidence": [{"path": "skills/cards/financial-kpi-extractor.md"}],
            },
        ],
        "near_misses": [],
        "coverage_gaps": [],
        "wiki_pages_read": [
            "skills/cards/pdf-table-parser.md",
            "skills/cards/financial-kpi-extractor.md",
            "edges/semantic_edges.jsonl",
        ],
        "rationale": "Parse tables before extracting KPIs.",
    }
    payload.update(overrides)
    return SkillPackage.from_dict(payload)


def _query_context(tmp_path):
    workspace_path = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace_path)
    bundle = build_router_bundle(
        RouterBundleConfig(
            workspace=workspace_path,
            query="extract financial KPIs from a PDF report",
            seed_limit=2,
            expanded_limit=8,
        ),
        embedding_provider=FakeEmbeddingProvider(),
    )
    query_wiki = materialize_query_wiki(
        workspace_path,
        bundle,
        trace_dir=workspace_path / "runs" / "package-test",
    )
    return bundle, query_wiki


def test_valid_package_projects_graph_relations_as_route_evidence(tmp_path) -> None:
    bundle, query_wiki = _query_context(tmp_path)
    package = _package()

    validation = validate_skill_package(package, query_wiki.root, max_selected_skills=8)
    route = route_from_skill_package(package, bundle)

    assert validation.valid, validation.errors
    assert route.selected_skill_ids == [
        "skill:pdf-table-parser",
        "skill:financial-kpi-extractor",
    ]
    assert len(route.relation_evidence) == 1
    relation = route.relation_evidence[0]
    assert relation.relation_type == "depend_on"
    assert relation.source_skill == "skill:pdf-table-parser"
    assert relation.target_skill == "skill:financial-kpi-extractor"
    assert relation.confidence == pytest.approx(0.94)


def test_external_skill_and_path_traversal_are_errors(tmp_path) -> None:
    _, query_wiki = _query_context(tmp_path)
    package = _package(
        selected_skills=[
            {
                "skill_id": "skill:not-in-manifest",
                "role": "Invalid selection.",
                "evidence": [{"path": "../outside.md"}],
            }
        ],
        wiki_pages_read=["../outside.md"],
    )

    validation = validate_skill_package(package, query_wiki.root, max_selected_skills=8)

    assert not validation.valid
    assert any("not in query_wiki manifest" in error for error in validation.errors)
    assert any("escapes query_wiki" in error for error in validation.errors)


def test_selected_skill_must_cite_its_own_card_or_source(tmp_path) -> None:
    _, query_wiki = _query_context(tmp_path)
    selected = _package().to_dict()["selected_skills"]
    selected[0]["evidence"] = [{"path": "skills/cards/financial-kpi-extractor.md"}]

    validation = validate_skill_package(
        _package(
            selected_skills=selected,
            wiki_pages_read=[
                "index.md",
                "skills/cards/pdf-table-parser.md",
                "skills/cards/financial-kpi-extractor.md",
                "edges/semantic_edges.jsonl",
            ],
        ),
        query_wiki.root,
        max_selected_skills=8,
    )

    assert not validation.valid
    assert any("own card or source" in error for error in validation.errors)


def test_selected_skill_may_also_cite_shared_comparison_context(tmp_path) -> None:
    _, query_wiki = _query_context(tmp_path)
    selected = _package().to_dict()["selected_skills"]
    selected[0]["evidence"] = [
        {"path": "index.md"},
        {"path": "skills/cards/pdf-table-parser.md"},
    ]

    validation = validate_skill_package(
        _package(
            selected_skills=selected,
            wiki_pages_read=[
                "index.md",
                "skills/cards/pdf-table-parser.md",
                "skills/cards/financial-kpi-extractor.md",
                "edges/semantic_edges.jsonl",
            ],
        ),
        query_wiki.root,
        max_selected_skills=8,
    )

    assert validation.valid, validation.errors


def test_manifest_rejects_duplicate_skill_rows(tmp_path) -> None:
    _, query_wiki = _query_context(tmp_path)
    manifest_path = query_wiki.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["skills"].append(dict(manifest["skills"][0]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate skill id"):
        validate_skill_package(_package(), query_wiki.root, max_selected_skills=8)


def test_selected_dependent_does_not_force_compiled_prerequisite(tmp_path) -> None:
    bundle, query_wiki = _query_context(tmp_path)
    package = _package(
        selected_skills=[_package().to_dict()["selected_skills"][1]],
        wiki_pages_read=["skills/cards/financial-kpi-extractor.md"],
    )

    validation = validate_skill_package(package, query_wiki.root, max_selected_skills=8)
    route = route_from_skill_package(package, bundle)

    assert validation.valid, validation.errors
    assert route.selected_skill_ids == ["skill:financial-kpi-extractor"]
    assert route.relation_evidence == ()


def test_empty_selection_requires_an_explicit_coverage_gap(tmp_path) -> None:
    _, query_wiki = _query_context(tmp_path)
    package = _package(
        selected_skills=[],
        wiki_pages_read=[],
        coverage_gaps=[],
    )

    validation = validate_skill_package(package, query_wiki.root, max_selected_skills=8)

    assert not validation.valid
    assert any("coverage gap" in error for error in validation.errors)


@pytest.mark.parametrize(
    "field",
    ["selected_skills", "near_misses"],
)
def test_skill_package_rejects_non_object_array_items(field: str) -> None:
    payload = _package().to_dict()
    payload[field] = ["not-an-object"]

    with pytest.raises(ValueError, match=rf"{field}\[0\] must be an object"):
        SkillPackage.from_dict(payload)


@pytest.mark.parametrize("evidence", [[], ["not-an-object"]])
def test_selected_skill_requires_object_evidence(evidence: list[object]) -> None:
    payload = _package().to_dict()
    payload["selected_skills"][0]["evidence"] = evidence

    with pytest.raises(ValueError, match="selected skill evidence"):
        SkillPackage.from_dict(payload)
