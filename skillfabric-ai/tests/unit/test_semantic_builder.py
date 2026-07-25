from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from skillfabric.compiled_graph.builder import (
    BuildConfig,
    _BuildDependencies,
    build_graph,
)
from skillfabric.compiled_graph.semantic.candidates import CandidateRetrievalError
from skillfabric.runtime.jobs import LLMJobOptions
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

    assert set(result.graph.to_dict()) == {"build_id", "nodes", "edges"}
    assert {path.name for path in (workspace / "graph").iterdir()} == {
        "registry.jsonl",
        "contracts.jsonl",
        "relation_decisions.jsonl",
        "graph.json",
        "bm25.sqlite",
        "embeddings.json",
        "embeddings.npy",
    }
    assert {path.name for path in (workspace / "cache").iterdir()} == {
        "contracts.json",
        "relation_decisions.json",
        "embeddings.json",
    }
    assert {path.name for path in (workspace / "reports").iterdir()} == {
        "build_summary.json",
        "llm_usage.jsonl",
    }


class FormalFakeEmbeddingProvider(FakeEmbeddingProvider):
    model_id = "openai/bge-m3"
    batch_size = 16
    concurrency = 4
    timeout = 30.0
    max_retries = 8

    def __init__(self) -> None:
        super().__init__(dimension=1024)


def test_build_summary_publishes_builder_and_embedding_protocol(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    provider_model = "openai/responses/gpt-5.6-luna"

    result = build_graph(
        BuildConfig(
            skill_root=FIXTURE_SKILLS,
            workspace=workspace,
            llm_model=provider_model,
            llm_reasoning_effort="medium",
            llm_options=LLMJobOptions(
                checkpoint_interval=25,
                circuit_breaker_threshold=7,
            ),
        ),
        dependencies=_BuildDependencies(
            contract_extractor=FixtureContractExtractor(model_id=provider_model),
            relation_judge=FixtureRelationJudge(model_id=provider_model),
            embedding_provider=FormalFakeEmbeddingProvider(),
            build_id="luna-test-build",
        ),
    )

    summary = json.loads(
        (result.workspace.reports_dir / "build_summary.json").read_text(encoding="utf-8")
    )
    assert summary["builder"] == {
        "model": "gpt-5.6-luna",
        "provider_model": provider_model,
        "reasoning_effort": "medium",
    }
    assert summary["llm_reliability"] == {
        "checkpoint_interval": 25,
        "circuit_breaker_threshold": 7,
    }
    assert {
        key: summary["embedding"][key]
        for key in (
            "model_id",
            "dimension",
            "batch_size",
            "concurrency",
            "timeout_seconds",
            "max_retries",
        )
    } == {
        "model_id": "openai/bge-m3",
        "dimension": 1024,
        "batch_size": 16,
        "concurrency": 4,
        "timeout_seconds": 30.0,
        "max_retries": 8,
    }
    assert set(summary["skill_pool"]) == {
        "graph_input_sha256",
        "package_sha256",
    }
    assert all(len(value) == 64 for value in summary["skill_pool"].values())


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
    store = workspace / "cache" / "embeddings.json"
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


def test_rebuild_replaces_existing_status_without_a_version_gate(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    workspace.mkdir()
    status = workspace / "status.json"
    status.write_text('{"state":"ready","build_id":"previous"}\n', encoding="utf-8")

    result = _build(workspace)

    assert result.graph.build_id == "semantic-builder-test"
    assert json.loads(status.read_text(encoding="utf-8")) == {
        "state": "ready",
        "stage": "complete",
        "build_id": "semantic-builder-test",
    }


def test_build_lock_contention_does_not_overwrite_active_status(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    workspace.mkdir()
    status = workspace / "status.json"
    active_status = {
        "state": "building",
        "stage": "contracts",
        "build_id": "active-build",
    }
    status.write_text(json.dumps(active_status) + "\n", encoding="utf-8")
    (workspace / "build.lock").write_text(str(os.getpid()), encoding="utf-8")

    with pytest.raises(RuntimeError, match="build lock already exists"):
        _build(workspace)

    assert json.loads(status.read_text(encoding="utf-8")) == active_status


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
    assert status["state"] == "failed"
    assert status["failed_stage"] == "contracts"
    assert "sk-sensitive-value" not in json.dumps(status)


def _fixture_skill_ids() -> list[str]:
    return [f"skill:{path.parent.name}" for path in sorted(FIXTURE_SKILLS.glob("*/SKILL.md"))]
