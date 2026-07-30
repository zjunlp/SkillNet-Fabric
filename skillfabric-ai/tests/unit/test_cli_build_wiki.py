from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

from skillfabric.cli import main as cli_main
from skillfabric.compiled_graph.builder import BuildResult
from skillfabric.compiled_graph.models import GraphDocument
from skillfabric.registry.models import SkillNode
from skillfabric.storage import Workspace
from skillfabric.wiki.models import WikiBuildResult, WikiHealthReport

FIXTURE_SKILLS = Path(__file__).resolve().parents[1] / "fixtures" / "skills"


def _env_file(path: Path) -> None:
    path.write_text(
        "API_KEY=test-key\n"
        "BASE_URL=https://api.example.test/v1\n"
        "MODEL=openai/test-model\n"
        "EMBEDDING_MODEL=openai/test-embedding\n",
        encoding="utf-8",
    )


def _build_result(root: Path) -> BuildResult:
    workspace = Workspace(root)
    workspace.ensure()
    skill = SkillNode(
        id="skill:test",
        type="skill",
        name="test",
        description="Test skill.",
        content_hash="hash-test",
    )
    graph = GraphDocument(
        build_id="test-build",
        nodes=[skill],
        edges=[],
    )
    for path in (
        workspace.graph_dir / "registry.jsonl",
        workspace.graph_dir / "contracts.jsonl",
        workspace.graph_dir / "relation_decisions.jsonl",
        workspace.graph_dir / "graph.json",
        workspace.graph_dir / "embeddings.json",
    ):
        path.write_text("{}\n", encoding="utf-8")
    return BuildResult(
        graph=graph,
        workspace=workspace,
        stats={
            "edge_counts": {
                "depend_on": 2,
                "compose_with": 1,
                "similar_to": 0,
            }
        },
    )


def _wiki_result(root: Path) -> WikiBuildResult:
    return WikiBuildResult(
        pages_written=3,
        health=WikiHealthReport(),
        workspace=root,
    )


def test_build_cli_reports_canonical_artifacts(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    env_file = tmp_path / ".env.test"
    _env_file(env_file)
    output = io.StringIO()

    with (
        patch("skillfabric.cli.build_graph", return_value=_build_result(workspace)),
        patch(
            "skillfabric.cli.build_wiki",
            return_value=_wiki_result(workspace),
        ) as build_wiki_mock,
        contextlib.redirect_stdout(output),
    ):
        cli_main(
            [
                "build",
                "--skill-root",
                str(FIXTURE_SKILLS),
                "--workspace",
                str(workspace),
                "--env-file",
                str(env_file),
            ]
        )

    payload = json.loads(output.getvalue())
    assert build_wiki_mock.call_count == 1
    wiki_config = build_wiki_mock.call_args.args[0]
    assert wiki_config.workspace == workspace
    assert not hasattr(wiki_config, "env_file")
    assert not hasattr(wiki_config, "use_llm_summaries")
    assert not hasattr(wiki_config, "llm_options")
    assert set(payload["graph"]) == {"node_count", "edge_count", "edge_counts"}
    assert payload["graph"]["edge_counts"] == {
        "depend_on": 2,
        "compose_with": 1,
        "similar_to": 0,
    }
    assert set(payload["artifacts"]) == {
        "registry",
        "contracts",
        "relation_decisions",
        "graph",
        "bm25",
        "embeddings",
        "build_summary",
        "llm_usage",
        "status",
        "wiki",
    }
    assert payload["artifacts"]["graph"].endswith("graph/graph.json")
    metrics = json.loads((workspace / "reports" / "build_summary.json").read_text(encoding="utf-8"))
    assert metrics["wiki"] == {"pages_written": 3}
    assert "wiki_summary" not in metrics


def test_build_cli_skip_wiki_does_not_invoke_wiki_materialization(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    env_file = tmp_path / ".env.test"
    _env_file(env_file)
    output = io.StringIO()

    with (
        patch("skillfabric.cli.build_graph", return_value=_build_result(workspace)),
        patch("skillfabric.cli.build_wiki") as build_wiki_mock,
        contextlib.redirect_stdout(output),
    ):
        cli_main(
            [
                "build",
                "--skill-root",
                str(FIXTURE_SKILLS),
                "--workspace",
                str(workspace),
                "--env-file",
                str(env_file),
                "--skip-wiki",
            ]
        )

    payload = json.loads(output.getvalue())
    assert build_wiki_mock.call_count == 0
    assert "wiki" not in payload["artifacts"]


def test_build_cli_forwards_build_only_llm_overrides(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    env_file = tmp_path / ".env.test"
    _env_file(env_file)

    with (
        patch("skillfabric.cli.build_graph", return_value=_build_result(workspace)) as build_mock,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        cli_main(
            [
                "build",
                "--skill-root",
                str(FIXTURE_SKILLS),
                "--workspace",
                str(workspace),
                "--env-file",
                str(env_file),
                "--llm-model",
                "openai/responses/gpt-5.6-luna",
                "--llm-reasoning-effort",
                "medium",
                "--llm-checkpoint-interval",
                "25",
                "--llm-circuit-breaker-threshold",
                "7",
                "--skip-wiki",
            ]
        )

    config = build_mock.call_args.args[0]
    assert config.llm_model == "openai/responses/gpt-5.6-luna"
    assert config.llm_reasoning_effort == "medium"
    assert config.llm_options.checkpoint_interval == 25
    assert config.llm_options.circuit_breaker_threshold == 7
