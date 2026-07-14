from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillfabric.compiled_graph.builder import (
    BuildConfig,
    _BuildDependencies,
    build_graph,
)
from skillfabric.compiled_graph.semantic.candidates import CandidateRetrievalError
from skillfabric.runtime.usage import LLMUsageTracker
from tests.unit.fake_embeddings import FakeEmbeddingProvider
from tests.unit.semantic_fixtures import (
    FixtureContractExtractor,
    FixtureRelationJudge,
    StaticContractExtractor,
    StaticRelationJudge,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SKILLS = ROOT / "fixtures" / "skills"


def _build(workspace: Path, *, contracts=None, judge=None):
    return build_graph(
        BuildConfig(skill_root=FIXTURE_SKILLS, workspace=workspace),
        dependencies=_BuildDependencies(
            contract_extractor=contracts or FixtureContractExtractor(),
            relation_judge=judge or FixtureRelationJudge(),
            embedding_provider=FakeEmbeddingProvider(),
            build_id="semantic-builder-test",
        ),
    )


def test_builder_writes_current_semantic_artifacts(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"

    result = _build(workspace)

    assert result.graph.schema_version == "3.0"
    assert {path.name for path in (workspace / "graph").iterdir()} == {
        "registry.jsonl",
        "contracts.jsonl",
        "relation_decisions.jsonl",
        "graph.json",
        "bm25.sqlite",
        "embeddings.json",
    }
    assert {path.name for path in (workspace / "cache").iterdir()} == {
        "contracts.json",
        "relation_decisions.json",
    }
    assert {path.name for path in (workspace / "reports").iterdir()} == {
        "build_summary.json",
        "llm_usage.jsonl",
    }


def test_fixture_build_recovers_operational_chains_without_similarity_noise(tmp_path) -> None:
    result = _build(tmp_path / ".skillfabric")
    edges = {(edge.source, edge.target, edge.type) for edge in result.graph.edges}

    assert edges == {
        ("skill:pdf-table-parser", "skill:financial-kpi-extractor", "depend_on"),
        ("skill:financial-kpi-extractor", "skill:report-writer", "depend_on"),
        ("skill:webshop-product-search", "skill:webshop-product-evaluator", "depend_on"),
        ("skill:analyze-ci", "skill:testing-python", "compose_with"),
    }
    assert result.stats["edge_counts"] == {
        "depend_on": 3,
        "compose_with": 1,
        "similar_to": 0,
    }


def test_duplicate_skill_ids_fail_before_contract_extraction(tmp_path) -> None:
    skill_root = tmp_path / "skills"
    for directory in ("first", "second"):
        path = skill_root / directory / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text(
            "---\nname: duplicate\ndescription: Duplicate fixture.\n---\n\n# Duplicate\n",
            encoding="utf-8",
        )

    class CountingExtractor:
        model_id = "counting-model"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def extract(self, skill):
            self.calls.append(skill.id)
            return {
                "capability": "Duplicate capability.",
                "when_to_use": "Use for duplicate tests.",
                "requires": [],
                "produces": [],
                "tools": [],
                "evidence": [{"line": 1}],
            }

    extractor = CountingExtractor()
    workspace = tmp_path / ".skillfabric"

    with pytest.raises(ValueError, match="duplicate skill id"):
        build_graph(
            BuildConfig(skill_root=skill_root, workspace=workspace),
            dependencies=_BuildDependencies(
                contract_extractor=extractor,
                relation_judge=StaticRelationJudge(model_id="unused", responses={}),
                embedding_provider=FakeEmbeddingProvider(),
                build_id="duplicate-skill-test",
            ),
        )

    assert extractor.calls == []
    status = json.loads((workspace / "status.json").read_text(encoding="utf-8"))
    assert status["failed_stage"] == "scan"


def test_graph_edges_have_one_canonical_semantic_schema(tmp_path) -> None:
    result = _build(tmp_path / ".skillfabric")

    for edge in result.graph.edges:
        assert set(edge.to_dict()) == {
            "source",
            "target",
            "type",
            "confidence",
            "evidence",
            "reason",
        }
        assert edge.evidence
    relation_rows = [
        json.loads(line)
        for line in (result.workspace.graph_dir / "relation_decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(
        set(row)
        == {
            "candidate",
            "relation",
            "source_skill",
            "target_skill",
            "confidence",
            "reason",
            "evidence",
        }
        for row in relation_rows
    )


def test_second_build_reuses_contract_and_relation_decision_caches(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    contracts = FixtureContractExtractor()
    judge = FixtureRelationJudge()

    first = _build(workspace, contracts=contracts, judge=judge)
    first_contract_calls = len(contracts.calls)
    first_judge_calls = len(judge.calls)
    second = _build(workspace, contracts=contracts, judge=judge)

    assert first_contract_calls == len(first.graph.nodes)
    assert first_judge_calls == first.stats["candidate_pair_count"]
    assert len(contracts.calls) == first_contract_calls
    assert len(judge.calls) == first_judge_calls
    assert second.stats["contract_cache_hits"] == len(second.graph.nodes)
    assert second.stats["relation_cache_hits"] == second.stats["candidate_pair_count"]


def test_rebuild_rejects_corrupted_embedding_store_instead_of_deleting_it(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    _build(workspace)
    store = workspace / "graph" / "embeddings.json"
    store.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(CandidateRetrievalError, match="embedding store"):
        _build(workspace)

    assert store.read_text(encoding="utf-8") == "not-json\n"


def test_build_summary_excludes_usage_from_previous_builds(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    usage_path = workspace / "reports" / "llm_usage.jsonl"
    tracker = LLMUsageTracker(usage_path)
    tracker.record_completion(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "old build"}],
        response={
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
        operation="graph.contract_extraction",
        duration_ms=1,
        status="completed",
        metadata={"build_id": "old-build"},
    )

    _build(workspace)

    summary = json.loads((workspace / "reports" / "build_summary.json").read_text(encoding="utf-8"))
    assert summary["llm_usage"]["total_calls"] == 0
    assert summary["llm_usage"]["total_tokens"] == 0


def test_build_rejects_an_incompatible_workspace_without_mutating_it(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    workspace.mkdir()
    status = workspace / "status.json"
    status.write_text('{"schema_version":"1.0","state":"ready"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"incompatible.*new workspace"):
        _build(workspace)

    assert json.loads(status.read_text(encoding="utf-8"))["schema_version"] == "1.0"


def test_failed_contract_stage_writes_sanitized_status(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    invalid = StaticContractExtractor(
        model_id="invalid-contract-model",
        responses={skill_id: {"error": "sk-sensitive-value"} for skill_id in _fixture_skill_ids()},
    )

    with pytest.raises(Exception, match="contract extraction failed"):
        _build(
            workspace,
            contracts=invalid,
            judge=StaticRelationJudge(model_id="unused", responses={}),
        )

    status = json.loads((workspace / "status.json").read_text(encoding="utf-8"))
    assert status["schema_version"] == "3.0"
    assert status["state"] == "failed"
    assert status["failed_stage"] == "contracts"
    assert "sk-sensitive-value" not in json.dumps(status)


def _fixture_skill_ids() -> list[str]:
    return [f"skill:{path.parent.name}" for path in sorted(FIXTURE_SKILLS.glob("*/SKILL.md"))]
