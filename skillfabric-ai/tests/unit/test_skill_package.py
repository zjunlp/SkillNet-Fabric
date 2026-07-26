from __future__ import annotations

import json
from typing import Any

import pytest

from skillfabric.router.bundle import RouterBundleConfig, build_router_bundle
from skillfabric.router.models import RouterAlternative, RouterBundle
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


def _minimal_query_wiki(tmp_path, *, skill_count: int = 3):
    root = tmp_path / "query-wiki"
    rows = []
    for index in range(skill_count):
        skill_id = f"skill:skill-{index}"
        card_path = f"skills/cards/skill-{index}.md"
        source_path = f"skills/sources/skill-{index}/SKILL.md"
        for relative_path in (card_path, source_path):
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {skill_id}\n", encoding="utf-8")
        rows.append(
            {
                "skill_id": skill_id,
                "name": f"Skill {index}",
                "description": "Fixture skill.",
                "selectable": True,
                "origin": "seed",
                "card_path": card_path,
                "source_path": source_path,
                "route": {},
                "alternative": False,
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "query": "fixture",
                "skills": rows,
                "semantic_edges_path": "edges/semantic_edges.jsonl",
                "alternatives": [],
            }
        ),
        encoding="utf-8",
    )
    return root, rows


def _counted_package(rows: list[dict[str, Any]], count: int) -> SkillPackage:
    selected = [
        {
            "skill_id": row["skill_id"],
            "role": f"Use {row['skill_id']}.",
            "evidence": [{"path": row["card_path"]}],
        }
        for row in rows[:count]
    ]
    return SkillPackage.from_dict(
        {
            "selected_skills": selected,
            "near_misses": [],
            "coverage_gaps": [],
            "wiki_pages_read": [row["card_path"] for row in rows[:count]],
            "rationale": "Fixture selection.",
        }
    )


@pytest.mark.parametrize(
    ("selected_count", "expected_valid"),
    [(1, False), (2, True), (3, False)],
)
def test_exact_selected_skill_count_is_enforced(
    tmp_path,
    selected_count: int,
    expected_valid: bool,
) -> None:
    root, rows = _minimal_query_wiki(tmp_path)

    validation = validate_skill_package(
        _counted_package(rows, selected_count),
        root,
        max_selected_skills=3,
        required_selected_skills=2,
    )

    assert validation.valid is expected_valid
    if not expected_valid:
        assert validation.errors == (
            f"selected skill count {selected_count} differs from required_selected_skills=2",
        )


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


@pytest.mark.parametrize("prefix", ["query_wiki/", "./", "./query_wiki/"])
def test_skill_package_normalizes_query_wiki_relative_paths(prefix: str) -> None:
    payload = _package().to_dict()
    canonical_path = "skills/cards/pdf-table-parser.md"
    payload["selected_skills"][0]["evidence"] = [{"path": f"{prefix}{canonical_path}"}]
    payload["wiki_pages_read"][0] = f"{prefix}{canonical_path}"

    package = SkillPackage.from_dict(payload)

    assert package.selected_skills[0].evidence[0].path == canonical_path
    assert package.wiki_pages_read[0] == canonical_path


def test_external_skill_and_path_traversal_are_errors(tmp_path) -> None:
    bundle, query_wiki = _query_context(tmp_path)
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
    with pytest.raises(ValueError, match="not a selectable bundle candidate"):
        route_from_skill_package(package, bundle)


def test_alternative_only_skill_cannot_bypass_bounded_candidates() -> None:
    bundle = RouterBundle(
        query="Use a near substitute.",
        selected_skills=(),
        graph_edges=(),
        alternatives=(
            RouterAlternative(
                skill_id="skill:similar-only",
                name="Similar Only",
                alternative_to="skill:preferred",
                confidence=0.9,
                reason="Validated near substitute.",
            ),
        ),
    )
    package = _package(
        selected_skills=[
            {
                "skill_id": "skill:similar-only",
                "role": "Use the alternative.",
                "evidence": [{"path": "index.md"}],
            }
        ],
        wiki_pages_read=["index.md"],
    )

    with pytest.raises(ValueError, match="not a selectable bundle candidate"):
        route_from_skill_package(package, bundle)


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


def test_manifest_rejects_alternative_only_candidate_rows(tmp_path) -> None:
    _, query_wiki = _query_context(tmp_path)
    manifest_path = query_wiki.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["skills"][0]["origin"] = "similar_alternative"
    manifest["skills"][0]["route"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid origin"):
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
