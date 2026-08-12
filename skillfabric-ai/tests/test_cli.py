from __future__ import annotations

import contextlib
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from skillfabric.cli import PUBLIC_COMMANDS
from skillfabric.cli import main as cli_main
from skillfabric.router.models import RouteResult, RouteSelectedSkill
from tests.support import build_fixture_workspace


def _route() -> RouteResult:
    return RouteResult(
        selected_skills=(
            RouteSelectedSkill(
                skill_id="skill:test",
                name="test",
                reason="Handle the test task.",
                evidence=("skills/cards/test.md",),
            ),
        ),
        relation_evidence=(),
        near_misses=(),
        coverage_gaps=(),
        wiki_pages_read=("skills/cards/test.md",),
        rationale="One skill covers the task.",
    )


def test_public_cli_surface_exposes_core_commands() -> None:
    output = io.StringIO()

    with pytest.raises(SystemExit) as raised, contextlib.redirect_stdout(output):
        cli_main(["--help"])

    assert raised.value.code == 0
    text = output.getvalue()
    assert set(PUBLIC_COMMANDS) == {
        "init",
        "help",
        "build",
        "route",
        "plan",
        "query-wiki",
        "doctor-state",
        "run-state",
    }
    assert "Generate one execution prompt" in text


def test_public_cli_help_hides_internal_runtime_tuning() -> None:
    output = io.StringIO()

    with pytest.raises(SystemExit) as raised, contextlib.redirect_stdout(output):
        cli_main(["build", "--help"])

    assert raised.value.code == 0
    text = output.getvalue()
    assert "--llm-progress-every" in text
    for option in (
        "--llm-concurrency",
        "--llm-max-retries",
        "--llm-batch-size",
        "--llm-checkpoint-interval",
        "--llm-circuit-breaker-threshold",
    ):
        assert option not in text


def test_public_cli_rejects_removed_progress_json_option() -> None:
    with pytest.raises(SystemExit):
        cli_main(["build", "--skill-root", "skills", "--progress-json"])


def test_build_cli_defaults_to_a_human_summary(tmp_path) -> None:
    workspace = SimpleNamespace(
        root=tmp_path / ".skillfabric",
        graph_dir=tmp_path / ".skillfabric" / "graph",
        wiki_dir=tmp_path / ".skillfabric" / "wiki",
        status_path=tmp_path / ".skillfabric" / "status.json",
    )
    result = SimpleNamespace(
        workspace=workspace,
        graph=SimpleNamespace(build_id="build-test", nodes=[None] * 7, edges=[None] * 12),
        stats={"edge_counts": {"depend_on": 4, "similar_to": 8}},
    )
    output = io.StringIO()

    with (
        patch("skillfabric.cli._require_api_configuration"),
        patch("skillfabric.cli.ApiEmbeddingProvider.from_env", return_value=object()),
        patch("skillfabric.cli.build_graph", return_value=result),
        patch("skillfabric.cli.build_wiki", return_value=SimpleNamespace(pages_written=19)),
        contextlib.redirect_stdout(output),
    ):
        cli_main(["build", "--skill-root", "skills"])

    text = output.getvalue()
    assert "Build complete" in text
    assert "7 skills" in text
    assert "12 edges" in text
    assert str(workspace.root) in text


def test_route_cli_constructs_only_current_router_config_fields() -> None:
    output = io.StringIO()

    with (
        patch("skillfabric.cli.route_task", return_value=_route()) as route_mock,
        contextlib.redirect_stdout(output),
    ):
        cli_main(
            [
                "route",
                "test task",
                "--workspace",
                ".skillfabric-test",
                "--max-selected-skills",
                "3",
                "--trace-id",
                "test-trace",
                "--json",
            ]
        )

    config = route_mock.call_args.args[0]
    assert config.query == "test task"
    assert config.max_selected_skills == 3
    assert config.seed_limit == 24
    assert config.expanded_limit == 100
    assert config.max_depth == 2
    assert config.trace_id == "test-trace"
    assert json.loads(output.getvalue())["selected_skills"][0]["skill_id"] == "skill:test"


def test_route_cli_defaults_to_a_human_summary() -> None:
    output = io.StringIO()

    with (
        patch("skillfabric.cli.route_task", return_value=_route()),
        contextlib.redirect_stdout(output),
    ):
        cli_main(["route", "test task"])

    text = output.getvalue()
    assert "Route complete" in text
    assert "Selected skills" in text
    assert "test" in text
    assert "Handle the test task." in text
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


def test_route_cli_rejects_internal_tuning_options() -> None:
    output = io.StringIO()

    with contextlib.redirect_stdout(output), pytest.raises(SystemExit):
        cli_main(["route", "unsupported task", "--seed-limit", "0"])


def test_plan_cli_calls_prompt_planner_once(tmp_path) -> None:
    package_root = tmp_path / ".skillfabric" / "runs" / "plan" / "execution_package"
    result = SimpleNamespace(
        to_dict=lambda: {
            "root": str(package_root),
            "prompt_path": str(package_root / "execution_prompt.md"),
            "estimated_prompt_tokens": 321,
        }
    )
    output = io.StringIO()

    with (
        patch(
            "skillfabric.cli._plan_route_context",
            return_value=(_route(), "test task", package_root),
        ),
        patch("skillfabric.cli.plan_execution_package", return_value=result) as planner,
        contextlib.redirect_stdout(output),
    ):
        cli_main(
            [
                "plan",
                "test task",
                "--workspace",
                str(tmp_path / ".skillfabric"),
                "--json",
            ]
        )

    planner.assert_called_once()
    assert "planner_context_max_tokens" not in planner.call_args.kwargs
    assert json.loads(output.getvalue())["estimated_prompt_tokens"] == 321


def test_build_cli_forwards_only_explicit_progress_setting(tmp_path) -> None:
    workspace = SimpleNamespace(
        root=tmp_path / ".skillfabric",
        graph_dir=tmp_path / ".skillfabric" / "graph",
        wiki_dir=tmp_path / ".skillfabric" / "wiki",
        status_path=tmp_path / ".skillfabric" / "status.json",
    )
    result = SimpleNamespace(
        workspace=workspace,
        graph=SimpleNamespace(build_id="build-test", nodes=[], edges=[]),
        stats={},
    )
    output = io.StringIO()

    with (
        patch("skillfabric.cli._require_api_configuration"),
        patch("skillfabric.cli.ApiEmbeddingProvider.from_env", return_value=object()),
        patch("skillfabric.cli.LLMJobOptions.from_env", return_value=object()) as options,
        patch("skillfabric.cli.build_graph", return_value=result),
        patch("skillfabric.cli.build_wiki", return_value=SimpleNamespace(pages_written=0)),
        contextlib.redirect_stdout(output),
    ):
        cli_main(["build", "--skill-root", "skills", "--llm-progress-every", "25", "--json"])

    assert options.call_args.kwargs == {"env_path": ".env", "progress_every": 25}


def test_plan_cli_defaults_to_a_human_summary(tmp_path) -> None:
    package_root = tmp_path / ".skillfabric" / "runs" / "plan" / "execution_package"
    result = SimpleNamespace(
        to_dict=lambda: {
            "root": str(package_root),
            "prompt_path": str(package_root / "execution_prompt.md"),
            "planner_output_path": str(package_root / "planner_output.json"),
            "planner_validation_path": str(package_root / "planner_validation.json"),
            "estimated_prompt_tokens": 321,
        }
    )
    output = io.StringIO()

    with (
        patch(
            "skillfabric.cli._plan_route_context",
            return_value=(_route(), "test task", package_root),
        ),
        patch("skillfabric.cli.plan_execution_package", return_value=result),
        contextlib.redirect_stdout(output),
    ):
        cli_main(["plan", "test task", "--workspace", str(tmp_path / ".skillfabric")])

    text = output.getvalue()
    assert "Execution plan ready" in text
    assert "321 tokens" in text
    assert str(package_root / "execution_prompt.md") in text


def test_plan_route_file_rejects_non_string_query_artifact(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    trace = workspace / "runs" / "bad-query"
    trace.mkdir(parents=True)
    (trace / "route.json").write_text(json.dumps(_route().to_dict()), encoding="utf-8")
    (trace / "query.json").write_text(json.dumps({"query": 123}), encoding="utf-8")

    with (
        patch("skillfabric.cli.plan_execution_package") as planner,
        pytest.raises(SystemExit, match="non-empty string query"),
    ):
        cli_main(
            [
                "plan",
                "--workspace",
                str(workspace),
                "--route-file",
                str(trace / "route.json"),
            ]
        )

    planner.assert_not_called()


def test_doctor_state_reports_readiness_without_configuration_values(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    env_file = tmp_path / ".env.test"
    env_file.write_text(
        "API_KEY=private-value\n"
        "BASE_URL=https://example.test/v1\n"
        "MODEL=openai/test-model\n"
        "EMBEDDING_MODEL=openai/test-embedding\n",
        encoding="utf-8",
    )
    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        cli_main(
            [
                "doctor-state",
                "--json",
                "--workspace",
                str(workspace),
                "--env-file",
                str(env_file),
            ]
        )

    text = output.getvalue()
    payload = json.loads(text)
    assert payload["workspace_ready"] is True
    assert payload["skill_count"] == 7
    assert payload["next_action"] == "ready"
    assert "private-value" not in text


def test_doctor_state_rejects_graph_from_a_different_build(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    status_path = workspace / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["build_id"] = "newer-build"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    env_file = tmp_path / ".env.test"
    env_file.write_text(
        "API_KEY=test-value\n"
        "BASE_URL=https://example.test/v1\n"
        "MODEL=openai/test-model\n"
        "EMBEDDING_MODEL=openai/test-embedding\n",
        encoding="utf-8",
    )
    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        cli_main(
            [
                "doctor-state",
                "--json",
                "--workspace",
                str(workspace),
                "--env-file",
                str(env_file),
            ]
        )

    payload = json.loads(output.getvalue())
    assert payload["workspace_ready"] is False
    assert payload["build_id"] == "newer-build"
    assert payload["skill_count"] == 0
    assert payload["next_action"] == "build"


def test_doctor_state_rejects_noncanonical_ready_status(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    build_fixture_workspace(workspace)
    status_path = workspace / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["unused"] = True
    status_path.write_text(json.dumps(status), encoding="utf-8")
    env_file = tmp_path / ".env.test"
    env_file.write_text(
        "API_KEY=test-value\n"
        "BASE_URL=https://example.test/v1\n"
        "MODEL=openai/test-model\n"
        "EMBEDDING_MODEL=openai/test-embedding\n",
        encoding="utf-8",
    )
    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        cli_main(
            [
                "doctor-state",
                "--json",
                "--workspace",
                str(workspace),
                "--env-file",
                str(env_file),
            ]
        )

    payload = json.loads(output.getvalue())
    assert payload["workspace_ready"] is False


def test_run_state_reuses_only_matching_prompt_package(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    root = workspace / "runs" / "run-test" / "execution_package"
    root.mkdir(parents=True)
    (root / "execution_prompt.md").write_text("Execute the task.\n", encoding="utf-8")
    (root / "planner_output.json").write_text(
        json.dumps({"execution_prompt": "Use the selected skills, then verify the result."}),
        encoding="utf-8",
    )
    (root / "planner_validation.json").write_text(
        '{"valid": true, "errors": []}\n',
        encoding="utf-8",
    )
    (root / "planner_request.json").write_text(
        json.dumps({"task": "original task"}),
        encoding="utf-8",
    )
    (root / "route.json").write_text(json.dumps(_route().to_dict()), encoding="utf-8")

    reuse_output = io.StringIO()
    with contextlib.redirect_stdout(reuse_output):
        cli_main(["run-state", "--json", "original task", "--workspace", str(workspace)])
    different_output = io.StringIO()
    with contextlib.redirect_stdout(different_output):
        cli_main(["run-state", "--json", "different task", "--workspace", str(workspace)])

    reuse = json.loads(reuse_output.getvalue())
    different = json.loads(different_output.getvalue())
    assert reuse["action"] == "reuse_prompt"
    assert reuse["prompt_path"].endswith("execution_prompt.md")
    assert different["action"] == "prepare_required"


def test_run_state_ignores_incomplete_execution_package(tmp_path) -> None:
    workspace = tmp_path / ".skillfabric"
    root = workspace / "runs" / "incomplete" / "execution_package"
    root.mkdir(parents=True)
    (root / "execution_prompt.md").write_text("Execute.\n", encoding="utf-8")
    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        cli_main(["run-state", "--json", "task", "--workspace", str(workspace)])

    assert json.loads(output.getvalue())["action"] == "prepare_required"
