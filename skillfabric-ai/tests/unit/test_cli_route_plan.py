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
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig
from tests.unit.wiki_helpers import build_fixture_workspace


def _write_task_atoms(path: Path, query: str) -> Path:
    payload = {
        "schema_version": "1.0",
        "atoms": [
            {
                "id": "a1",
                "kind": "action",
                "text": "extract financial KPIs",
                "evidence": "extract financial KPIs",
                "required": True,
                "depends_on": [],
            },
            {
                "id": "a2",
                "kind": "artifact",
                "text": "PDF report",
                "evidence": "PDF report",
                "required": True,
                "depends_on": [],
            },
        ],
    }
    assert payload["atoms"][0]["evidence"].lower() in query.lower()
    assert payload["atoms"][1]["evidence"].lower() in query.lower()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class RoutePlanCliTests(unittest.TestCase):
    def test_public_cli_surface_only_exposes_core_commands(self) -> None:
        help_output = io.StringIO()
        with contextlib.redirect_stdout(help_output):
            cli_main(["--help"])
        help_text = help_output.getvalue()

        for command in ("init", "help", "build", "route", "plan"):
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

    def test_route_and_plan_cli_offline(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            query = "extract financial KPIs from a PDF report"
            task_atoms_path = _write_task_atoms(Path(tmp) / "task_atoms.json", query)
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
                        "--task-atoms-file",
                        str(task_atoms_path),
                    ]
                )
            route_payload = json.loads(route_output.getvalue())
            self.assertGreaterEqual(len(route_payload["selected_skills"]), 1)
            route_path = Path(route_payload["trace_dir"]) / "route.json"
            self.assertTrue(route_path.exists())

            plan_output = io.StringIO()
            with contextlib.redirect_stdout(plan_output):
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
            plan_payload = json.loads(plan_output.getvalue())
            package_root = Path(plan_payload["root"])
            self.assertTrue((package_root / "execution_prompt.md").exists())
            self.assertTrue((package_root / "agent_run_spec.json").exists())
            self.assertEqual(Path(plan_payload["prompt_path"]), package_root / "execution_prompt.md")
            self.assertIn("Codex", plan_payload["entry_prompt"]["label"])
            self.assertIn("execution_prompt.md", plan_payload["entry_prompt"]["prompt"])
            self.assertEqual(
                [item["skill_id"] for item in plan_payload["agent_run_spec"]["selected_skills"]],
                [item["skill_id"] for item in route_payload["selected_skills"]],
            )

    def test_plan_cli_can_route_from_query(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            query = "extract financial KPIs from a PDF report"
            task_atoms_path = _write_task_atoms(Path(tmp) / "task_atoms.json", query)
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
                        "--task-atoms-file",
                        str(task_atoms_path),
                    ]
                )

            payload = json.loads(output.getvalue())
            package_root = Path(payload["root"])
            self.assertTrue((package_root / "execution_prompt.md").exists())
            self.assertEqual(payload["renderer"], "claude-code")


if __name__ == "__main__":
    unittest.main()
