from __future__ import annotations

import json
from typing import Any

import pytest

from skillfabric.router.bundle import RouterBundleConfig, build_router_bundle
from skillfabric.router.models import RouterAlternative, RouterBundle
from skillfabric.wiki.explorer.skill_package import SkillPackage
from skillfabric.wiki.explorer.validation import route_from_skill_package, validate_skill_package
from skillfabric.wiki.task_wiki import materialize_task_wiki
from tests.support import FakeEmbeddingProvider, build_fixture_workspace


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
    task_wiki = materialize_task_wiki(
        workspace_path,
        bundle,
        trace_dir=workspace_path / "runs" / "package-test",
    )
    return bundle, task_wiki


def _minimal_task_wiki(tmp_path, *, skill_count: int = 3):
    root = tmp_path / "task-wiki"
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


def test_task_wiki_reuses_full_wiki_source_pages(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    bundle = build_router_bundle(
        RouterBundleConfig(workspace=workspace, query="extract KPIs"),
        embedding_provider=FakeEmbeddingProvider(),
    )
    task_wiki = materialize_task_wiki(
        workspace,
        bundle,
        trace_dir=workspace / "runs" / "projection-test",
    )

    full_source = (workspace / "wiki" / "skills" / "sources" / "pdf-table-parser.md").read_text(
        encoding="utf-8"
    )
    task_source = (task_wiki.root / "skills" / "sources" / "pdf-table-parser.md").read_text(
        encoding="utf-8"
    )
    metadata = json.loads((task_wiki.root / "source.json").read_text(encoding="utf-8"))

    assert task_source == full_source
    assert metadata["build_id"] == "test-build"
    assert set(metadata["skills"]) == {candidate.skill_id for candidate in bundle.selected_skills}


def test_task_wiki_rejects_stale_full_wiki(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    manifest_path = workspace / "wiki" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["build_id"] = "stale-build"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bundle = build_router_bundle(
        RouterBundleConfig(workspace=workspace, query="extract KPIs"),
        embedding_provider=FakeEmbeddingProvider(),
    )

    with pytest.raises(ValueError, match="different builds"):
        materialize_task_wiki(
            workspace,
            bundle,
            trace_dir=workspace / "runs" / "stale-wiki",
        )


def test_task_wiki_supports_default_relative_workspace(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    relative_workspace = ".skillfabric"
    bundle = build_router_bundle(
        RouterBundleConfig(workspace=relative_workspace, query="extract KPIs"),
        embedding_provider=FakeEmbeddingProvider(),
    )

    result = materialize_task_wiki(
        relative_workspace,
        bundle,
        trace_dir=workspace / "runs" / "relative-workspace",
    )

    assert result.root.is_dir()
    assert (result.root / "skills" / "sources" / "pdf-table-parser.md").is_file()


def test_valid_package_projects_graph_relations_as_route_evidence(tmp_path) -> None:
    bundle, task_wiki = _query_context(tmp_path)
    package = _package()

    validation = validate_skill_package(package, task_wiki.root, max_selected_skills=8)
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


@pytest.mark.parametrize("prefix", ["task_wiki/", "./", "./task_wiki/"])
def test_skill_package_normalizes_task_wiki_relative_paths(prefix: str) -> None:
    payload = _package().to_dict()
    canonical_path = "skills/cards/pdf-table-parser.md"
    payload["selected_skills"][0]["evidence"] = [{"path": f"{prefix}{canonical_path}"}]
    payload["wiki_pages_read"][0] = f"{prefix}{canonical_path}"

    package = SkillPackage.from_dict(payload)

    assert package.selected_skills[0].evidence[0].path == canonical_path
    assert package.wiki_pages_read[0] == canonical_path


def test_external_skill_and_path_traversal_are_errors(tmp_path) -> None:
    bundle, task_wiki = _query_context(tmp_path)
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

    validation = validate_skill_package(package, task_wiki.root, max_selected_skills=8)

    assert not validation.valid
    assert any("not in task_wiki manifest" in error for error in validation.errors)
    assert any("escapes task_wiki" in error for error in validation.errors)
    with pytest.raises(ValueError, match="not a selectable bundle candidate"):
        route_from_skill_package(package, bundle)


def test_alternative_only_skill_cannot_bypass_candidate_budget() -> None:
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
    _, task_wiki = _query_context(tmp_path)
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
        task_wiki.root,
        max_selected_skills=8,
    )

    assert not validation.valid
    assert any("own card or source" in error for error in validation.errors)


def test_selected_skill_may_also_cite_shared_comparison_context(tmp_path) -> None:
    _, task_wiki = _query_context(tmp_path)
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
        task_wiki.root,
        max_selected_skills=8,
    )

    assert validation.valid, validation.errors


def test_manifest_rejects_duplicate_skill_rows(tmp_path) -> None:
    _, task_wiki = _query_context(tmp_path)
    manifest_path = task_wiki.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["skills"].append(dict(manifest["skills"][0]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate skill id"):
        validate_skill_package(_package(), task_wiki.root, max_selected_skills=8)


def test_manifest_rejects_alternative_only_candidate_rows(tmp_path) -> None:
    _, task_wiki = _query_context(tmp_path)
    manifest_path = task_wiki.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["skills"][0]["origin"] = "similar_alternative"
    manifest["skills"][0]["route"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid origin"):
        validate_skill_package(_package(), task_wiki.root, max_selected_skills=8)


def test_selected_dependent_does_not_force_compiled_prerequisite(tmp_path) -> None:
    bundle, task_wiki = _query_context(tmp_path)
    package = _package(
        selected_skills=[_package().to_dict()["selected_skills"][1]],
        wiki_pages_read=["skills/cards/financial-kpi-extractor.md"],
    )

    validation = validate_skill_package(package, task_wiki.root, max_selected_skills=8)
    route = route_from_skill_package(package, bundle)

    assert validation.valid, validation.errors
    assert route.selected_skill_ids == ["skill:financial-kpi-extractor"]
    assert route.relation_evidence == ()


def test_empty_selection_requires_an_explicit_coverage_gap(tmp_path) -> None:
    _, task_wiki = _query_context(tmp_path)
    package = _package(
        selected_skills=[],
        wiki_pages_read=[],
        coverage_gaps=[],
    )

    validation = validate_skill_package(package, task_wiki.root, max_selected_skills=8)

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
