from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from skillfabric.cli import main as cli_main
from skillfabric.router.models import RouteResult
from skillfabric.storage import atomic_write_text
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig
from tests.unit.wiki_helpers import build_fixture_workspace


class RoutePlanCliTests(unittest.TestCase):
    def test_public_cli_surface_only_exposes_core_commands(self) -> None:
        help_output = io.StringIO()
        with contextlib.redirect_stdout(help_output):
            cli_main(["--help"])
        help_text = help_output.getvalue()

        for command in ("init", "help", "build", "route", "plan", "doctor-state", "run-state"):
            self.assertIn(command, help_text)
        for command in (
            "package",
            "scan",
            "build-wiki",
            "wiki-status",
            "get-status",
            "get-skill-neighbors",
            "query-bundle",
            "build-execution-package",
            "eval",
            "llm-smoke",
            "export-neo4j",
        ):
            self.assertNotIn(command, help_text)
            with self.assertRaises(SystemExit) as raised:
                with contextlib.redirect_stderr(io.StringIO()):
                    cli_main([command, "--help"])
            self.assertNotEqual(raised.exception.code, 0)

    def test_help_config_prints_api_configuration_policy(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            cli_main(["help", "config"])

        text = output.getvalue()
        self.assertIn("API_KEY", text)
        self.assertIn("BASE_URL", text)
        self.assertIn("MODEL", text)
        self.assertIn("EMBEDDING_MODEL", text)
        self.assertIn("EMBEDDING_API_KEY", text)
        self.assertIn("EMBEDDING_BASE_URL", text)
        self.assertIn("OPENAI_API_KEY", text)
        self.assertIn("Claude Code SDK", text)
        self.assertNotIn("SKILLFABRIC_API_KEY", text)
        self.assertNotIn("SKILLFABRIC_BASE_URL", text)

    def test_route_cli_defaults_to_llm_router(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            captured = {}

            def fake_route(config):
                captured["config"] = config
                return RouteResult(
                    query=config.query,
                    trace_id=config.trace_id or "trace",
                    trace_dir=workspace / "runs" / "trace",
                    selected_skills=[],
                    provenance="claude_code",
                )

            output = io.StringIO()
            with patch("skillfabric.cli.route_task", side_effect=fake_route):
                with contextlib.redirect_stdout(output):
                    cli_main(
                        [
                            "route",
                            "extract financial KPIs from a PDF report",
                            "--workspace",
                            str(workspace),
                        ]
                    )

            payload = json.loads(output.getvalue())
            self.assertEqual(payload["provenance"], "claude_code")
            self.assertTrue(captured["config"].use_llm_router)
            self.assertEqual(captured["config"].explorer_backend, "claude-code")

    def test_route_and_plan_cli_offline_requires_agent_mode(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            query = "extract financial KPIs from a PDF report"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            route_output = io.StringIO()
            with contextlib.redirect_stdout(route_output):
                cli_main(
                    [
                        "route",
                        query,
                        "--workspace",
                        str(workspace),
                        "--skip-llm-router",
                        "--explorer-backend",
                        "fallback",
                    ]
                )
            route_payload = json.loads(route_output.getvalue())
            self.assertGreaterEqual(len(route_payload["selected_skills"]), 1)
            route_path = Path(route_payload["trace_dir"]) / "route.json"
            self.assertTrue(route_path.exists())

            with self.assertRaises(SystemExit) as raised:
                cli_main(
                    [
                        "plan",
                        "--route-file",
                        str(route_path),
                        "--workspace",
                        str(workspace),
                        "--renderer",
                        "codex",
                    ]
                )
            self.assertIn("requires --agent-mode prepare", str(raised.exception))

            prepare_output = io.StringIO()
            with contextlib.redirect_stdout(prepare_output):
                cli_main(
                    [
                        "plan",
                        "--route-file",
                        str(route_path),
                        "--workspace",
                        str(workspace),
                        "--renderer",
                        "codex",
                        "--agent-mode",
                        "prepare",
                    ]
                )
            prepared = json.loads(prepare_output.getvalue())
            package_root = Path(prepared["root"])
            self.assertTrue((package_root / "planner_request.json").exists())
            self.assertFalse((package_root / "execution_prompt.md").exists())
            self.assertFalse((package_root / "agent_run_spec_draft.json").exists())

            planner_output = {"execution_prompt": "# Prompt\n\nExecute the task."}
            finalize_output = io.StringIO()
            with patch("sys.stdin", io.StringIO(json.dumps(planner_output))):
                with contextlib.redirect_stdout(finalize_output):
                    cli_main(
                        [
                            "plan",
                            "--workspace",
                            str(workspace),
                            "--renderer",
                            "codex",
                            "--agent-mode",
                            "finalize",
                            "--package-root",
                            str(package_root),
                            "--planner-output-file",
                            "-",
                        ]
                    )
            finalized = json.loads(finalize_output.getvalue())
            self.assertEqual(Path(finalized["prompt_path"]), package_root / "execution_prompt.md")
            self.assertEqual(finalized["renderer"], "codex")
            self.assertTrue((package_root / "execution_prompt.md").exists())
            self.assertFalse((package_root / "workflow_plan.json").exists())
            self.assertFalse((package_root / "agent_run_spec.json").exists())

    def test_plan_cli_can_prepare_from_query(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            query = "extract financial KPIs from a PDF report"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                cli_main(
                    [
                        "plan",
                        query,
                        "--workspace",
                        str(workspace),
                        "--skip-llm-router",
                        "--explorer-backend",
                        "fallback",
                        "--renderer",
                        "claude-code",
                        "--agent-mode",
                        "prepare",
                    ]
                )

            payload = json.loads(output.getvalue())
            package_root = Path(payload["root"])
            self.assertTrue((package_root / "planner_request.json").exists())
            self.assertFalse((package_root / "execution_prompt.md").exists())
            self.assertEqual(payload["renderer"], "claude-code")

    def test_plan_cli_without_agent_mode_rejects_query(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            query = "extract financial KPIs from a PDF report"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            with self.assertRaises(SystemExit) as raised:
                cli_main(
                    [
                        "plan",
                        query,
                        "--workspace",
                        str(workspace),
                        "--skip-llm-router",
                        "--explorer-backend",
                        "fallback",
                        "--renderer",
                        "claude-code",
                    ]
                )

            self.assertIn("direct deterministic planning was removed", str(raised.exception))

    def test_doctor_state_reports_non_secret_readiness(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = root / ".env"
            workspace = root / ".skillfabric"
            env_file.write_text(
                "API_KEY=secret-value\n"
                "BASE_URL=https://example.test/v1\n"
                "MODEL=test-model\n"
                "EMBEDDING_MODEL=test-embed\n",
                encoding="utf-8",
            )
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                cli_main(["doctor-state", "--env-file", str(env_file), "--workspace", str(workspace)])

            raw = output.getvalue()
            state = json.loads(raw)
            self.assertTrue(state["cli_available"])
            self.assertTrue(state["api_configured"])
            self.assertEqual(state["missing"], [])
            self.assertEqual(state["workspace_status"]["stage"], "not_built")
            self.assertFalse(state["workspace_ready"])
            self.assertNotIn("secret-value", raw)

    def test_doctor_state_summarizes_existing_workspace_status(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = root / ".env"
            workspace = root / ".skillfabric"
            env_file.write_text(
                "API_KEY=secret-value\n"
                "BASE_URL=https://example.test/v1\n"
                "MODEL=test-model\n"
                "EMBEDDING_MODEL=test-embed\n",
                encoding="utf-8",
            )
            workspace.mkdir()
            atomic_write_text(
                workspace / "status.json",
                json.dumps(
                    {
                        "stage": "complete",
                        "build_id": 123,
                        "skill_count": 7,
                        "warnings": ["warning"],
                    }
                ),
            )
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                cli_main(["doctor-state", "--env-file", str(env_file), "--workspace", str(workspace)])

            raw = output.getvalue()
            state = json.loads(raw)
            self.assertTrue(state["workspace_ready"])
            self.assertEqual(state["workspace_status"]["stage"], "complete")
            self.assertEqual(state["workspace_status"]["build_id"], 123)
            self.assertEqual(state["workspace_status"]["skill_count"], 7)
            self.assertEqual(state["workspace_status"]["warnings_count"], 1)
            self.assertNotIn("secret-value", raw)

    def test_doctor_state_treats_build_summary_status_as_complete(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_file = root / ".env"
            workspace = root / ".skillfabric"
            env_file.write_text(
                "API_KEY=secret-value\n"
                "BASE_URL=https://example.test/v1\n"
                "MODEL=test-model\n"
                "EMBEDDING_MODEL=test-embed\n",
                encoding="utf-8",
            )
            workspace.mkdir()
            atomic_write_text(
                workspace / "status.json",
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "build_id": "build-123",
                        "skill_count": 7,
                        "artifacts": {"graph": "graph/graph.json"},
                    }
                ),
            )
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                cli_main(["doctor-state", "--env-file", str(env_file), "--workspace", str(workspace)])

            state = json.loads(output.getvalue())
            self.assertTrue(state["workspace_ready"])
            self.assertEqual(state["workspace_status"]["stage"], "complete")

    def test_plan_cli_latest_returns_most_recent_finalized_execution_prompt(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            route_output = io.StringIO()
            with contextlib.redirect_stdout(route_output):
                cli_main(
                    [
                        "route",
                        "extract financial KPIs from a PDF report",
                        "--workspace",
                        str(workspace),
                        "--skip-llm-router",
                        "--explorer-backend",
                        "fallback",
                    ]
                )
            route_payload = json.loads(route_output.getvalue())

            prepare_output = io.StringIO()
            with contextlib.redirect_stdout(prepare_output):
                cli_main(
                    [
                        "plan",
                        "--route-file",
                        str(Path(route_payload["trace_dir"]) / "route.json"),
                        "--workspace",
                        str(workspace),
                        "--renderer",
                        "claude-code",
                        "--agent-mode",
                        "prepare",
                    ]
                )
            prepared = json.loads(prepare_output.getvalue())
            package_root = Path(prepared["root"])

            with patch("sys.stdin", io.StringIO('{"execution_prompt": "# Prompt\\n\\nExecute it."}')):
                with contextlib.redirect_stdout(io.StringIO()):
                    cli_main(
                        [
                            "plan",
                            "--workspace",
                            str(workspace),
                            "--renderer",
                            "claude-code",
                            "--agent-mode",
                            "finalize",
                            "--package-root",
                            str(package_root),
                            "--planner-output-file",
                            "-",
                        ]
                    )

            latest_output = io.StringIO()
            with contextlib.redirect_stdout(latest_output):
                cli_main(
                    [
                        "plan",
                        "--workspace",
                        str(workspace),
                        "--agent-mode",
                        "latest",
                    ]
                )

            latest = json.loads(latest_output.getvalue())
            self.assertTrue(latest["found"])
            self.assertEqual(Path(latest["package_root"]), package_root)
            self.assertEqual(Path(latest["prompt_path"]), package_root / "execution_prompt.md")
            self.assertEqual(latest["trace_id"], route_payload["trace_id"])
            self.assertEqual(latest["task"], route_payload["query"])
            self.assertGreaterEqual(len(latest["selected_skills"]), 1)

    def test_run_state_reuses_latest_finalized_execution_prompt(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            route_output = io.StringIO()
            with contextlib.redirect_stdout(route_output):
                cli_main(
                    [
                        "route",
                        "extract financial KPIs from a PDF report",
                        "--workspace",
                        str(workspace),
                        "--skip-llm-router",
                        "--explorer-backend",
                        "fallback",
                    ]
                )
            route_payload = json.loads(route_output.getvalue())

            prepare_output = io.StringIO()
            with contextlib.redirect_stdout(prepare_output):
                cli_main(
                    [
                        "plan",
                        "--route-file",
                        str(Path(route_payload["trace_dir"]) / "route.json"),
                        "--workspace",
                        str(workspace),
                        "--renderer",
                        "claude-code",
                        "--agent-mode",
                        "prepare",
                    ]
                )
            prepared = json.loads(prepare_output.getvalue())
            package_root = Path(prepared["root"])

            with patch("sys.stdin", io.StringIO('{"execution_prompt": "# Prompt\\n\\nExecute it."}')):
                with contextlib.redirect_stdout(io.StringIO()):
                    cli_main(
                        [
                            "plan",
                            "--workspace",
                            str(workspace),
                            "--renderer",
                            "claude-code",
                            "--agent-mode",
                            "finalize",
                            "--package-root",
                            str(package_root),
                            "--planner-output-file",
                            "-",
                        ]
                    )

            state_output = io.StringIO()
            with contextlib.redirect_stdout(state_output):
                cli_main(["run-state", "--workspace", str(workspace)])

            state = json.loads(state_output.getvalue())
            self.assertEqual(state["action"], "reuse_prompt")
            self.assertEqual(Path(state["prompt_path"]), package_root / "execution_prompt.md")
            self.assertEqual(Path(state["package_root"]), package_root)
            self.assertEqual(state["trace_id"], route_payload["trace_id"])
            self.assertEqual(state["task"], route_payload["query"])
            self.assertGreaterEqual(len(state["selected_skills"]), 1)

    def test_run_state_prepares_new_task_instead_of_reusing_unrelated_prompt(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            route_output = io.StringIO()
            with contextlib.redirect_stdout(route_output):
                cli_main(
                    [
                        "route",
                        "extract financial KPIs from a PDF report",
                        "--workspace",
                        str(workspace),
                        "--skip-llm-router",
                        "--explorer-backend",
                        "fallback",
                    ]
                )
            route_payload = json.loads(route_output.getvalue())

            prepare_output = io.StringIO()
            with contextlib.redirect_stdout(prepare_output):
                cli_main(
                    [
                        "plan",
                        "--route-file",
                        str(Path(route_payload["trace_dir"]) / "route.json"),
                        "--workspace",
                        str(workspace),
                        "--renderer",
                        "claude-code",
                        "--agent-mode",
                        "prepare",
                    ]
                )
            prepared = json.loads(prepare_output.getvalue())

            with patch("sys.stdin", io.StringIO('{"execution_prompt": "# Prompt\\n\\nExecute it."}')):
                with contextlib.redirect_stdout(io.StringIO()):
                    cli_main(
                        [
                            "plan",
                            "--workspace",
                            str(workspace),
                            "--renderer",
                            "claude-code",
                            "--agent-mode",
                            "finalize",
                            "--package-root",
                            str(prepared["root"]),
                            "--planner-output-file",
                            "-",
                        ]
                    )

            state_output = io.StringIO()
            with contextlib.redirect_stdout(state_output):
                cli_main(
                    [
                        "run-state",
                        "review",
                        "graph-based",
                        "skill",
                        "routing",
                        "--workspace",
                        str(workspace),
                    ]
                )

            state = json.loads(state_output.getvalue())
            self.assertEqual(state["action"], "prepare_required")
            self.assertTrue(state["prepared_prompt_found"])
            self.assertTrue(state["existing_prompt_ignored"])
            self.assertEqual(state["task"], "review graph-based skill routing")
            self.assertEqual(state["existing_task"], route_payload["query"])

    def test_run_state_requires_task_when_no_prepared_prompt_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                cli_main(["run-state", "--workspace", str(workspace)])

            state = json.loads(output.getvalue())
            self.assertEqual(state["action"], "missing_task")
            self.assertFalse(state["prepared_prompt_found"])
            self.assertEqual(state["workspace"], str(workspace))


if __name__ == "__main__":
    unittest.main()
