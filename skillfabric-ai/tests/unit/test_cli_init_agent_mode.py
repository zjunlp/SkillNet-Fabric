from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from skillfabric.cli import main as cli_main
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig
from tests.unit.wiki_helpers import build_fixture_workspace


class InitAndAgentModeCliTests(unittest.TestCase):
    def test_init_check_json_reports_missing_config_without_secrets(self) -> None:
        with TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"

            output = io.StringIO()
            with patch.dict(os.environ, {}, clear=True):
                with contextlib.redirect_stdout(output):
                    cli_main(["init", "--env-file", str(env_file), "--check", "--json"])

            payload = json.loads(output.getvalue())
            self.assertFalse(payload["configured"])
            self.assertEqual(
                payload["missing"],
                ["API_KEY", "BASE_URL", "MODEL", "EMBEDDING_MODEL"],
            )
            self.assertNotIn("sk-", output.getvalue())

            env_file.write_text(
                "\n".join(
                    [
                        "API_KEY=sk-secret-value",
                        "BASE_URL=https://api.example.test/v1",
                        "MODEL=openai/test-chat",
                        "EMBEDDING_MODEL=openai/test-embedding",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            configured_output = io.StringIO()
            with patch.dict(os.environ, {}, clear=True):
                with contextlib.redirect_stdout(configured_output):
                    cli_main(["init", "--env-file", str(env_file), "--check", "--json"])

            configured = json.loads(configured_output.getvalue())
            self.assertTrue(configured["configured"])
            self.assertEqual(configured["missing"], [])
            self.assertNotIn("sk-secret-value", configured_output.getvalue())

    def test_init_check_accepts_claude_code_endpoint_env_without_secrets(self) -> None:
        with TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            output = io.StringIO()

            with patch.dict(
                os.environ,
                {
                    "ANTHROPIC_AUTH_TOKEN": "sk-cc-token",
                    "ANTHROPIC_BASE_URL": "http://gateway.example",
                    "ANTHROPIC_MODEL": "gpt-5.4-mini",
                    "EMBEDDING_MODEL": "BAAI/bge-large-en-v1.5",
                },
                clear=True,
            ):
                with contextlib.redirect_stdout(output):
                    cli_main(["init", "--env-file", str(env_file), "--check", "--json"])

            payload = json.loads(output.getvalue())
            self.assertTrue(payload["configured"])
            self.assertEqual(payload["missing"], [])
            self.assertEqual(payload["sources"]["API_KEY"], "shell")
            self.assertNotIn("sk-cc-token", output.getvalue())

    def test_init_writes_env_file_with_private_permissions_and_preserves_key(self) -> None:
        with TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("API_KEY=sk-existing\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                with patch("builtins.input", side_effect=["https://api.example.test/v1", "openai/chat", "openai/embed"]):
                    with patch("getpass.getpass", side_effect=AssertionError("API key should not be overwritten")):
                        cli_main(["init", "--env-file", str(env_file)])

            text = env_file.read_text(encoding="utf-8")
            self.assertIn("API_KEY=sk-existing", text)
            self.assertIn("BASE_URL=https://api.example.test/v1", text)
            self.assertIn("MODEL=openai/chat", text)
            self.assertIn("EMBEDDING_MODEL=openai/embed", text)
            self.assertEqual(stat.S_IMODE(os.stat(env_file).st_mode), 0o600)

    def test_agent_mode_prepare_and_finalize_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            prepare_output = io.StringIO()
            with contextlib.redirect_stdout(prepare_output):
                cli_main(
                    [
                        "route",
                        "extract financial KPIs from a PDF report",
                        "--workspace",
                        str(workspace),
                        "--trace-id",
                        "agent-route",
                        "--agent-mode",
                        "prepare",
                    ]
                )

            prepared = json.loads(prepare_output.getvalue())
            trace_dir = Path(prepared["trace_dir"])
            query_wiki_root = Path(prepared["query_wiki_root"])
            self.assertEqual(prepared["trace_id"], "agent-route")
            self.assertTrue(query_wiki_root.exists())
            self.assertTrue((trace_dir / "router_bundle.json").exists())
            self.assertTrue((trace_dir / "agent_route_request.json").exists())
            self.assertIn("selected_skills", prepared["expected_schema"]["properties"])

            manifest = json.loads((query_wiki_root / "manifest.json").read_text(encoding="utf-8"))
            selected = next(item for item in manifest["skills"] if item["selectable"])
            skill_package_file = trace_dir / "agent_skill_package.json"
            skill_package_file.write_text(
                json.dumps(
                    {
                        "selected_skills": [
                            {
                                "skill_id": selected["skill_id"],
                                "scope": selected["scope"],
                                "role": "Use this skill for the main task capability.",
                                "evidence": [{"path": selected["page_path"], "reason": "Selected from query wiki."}],
                            }
                        ],
                        "required_edges": [],
                        "ordered_hints": [],
                        "near_misses": [],
                        "coverage_notes": [],
                        "rationale": "Single evidence-backed skill selection.",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            finalize_output = io.StringIO()
            with contextlib.redirect_stdout(finalize_output):
                cli_main(
                    [
                        "route",
                        "extract financial KPIs from a PDF report",
                        "--workspace",
                        str(workspace),
                        "--trace-id",
                        "agent-route",
                        "--agent-mode",
                        "finalize",
                        "--skill-package-file",
                        str(skill_package_file),
                    ]
                )

            route = json.loads(finalize_output.getvalue())
            self.assertEqual(route["trace_id"], "agent-route")
            self.assertEqual(route["selected_skills"][0]["skill_id"], selected["skill_id"])
            self.assertTrue((trace_dir / "route.json").exists())
            self.assertTrue((trace_dir / "agent_route_validation.json").exists())

    def test_query_wiki_card_cli_outputs_header_without_raw_source(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))
            prepare_output = io.StringIO()
            with contextlib.redirect_stdout(prepare_output):
                cli_main(
                    [
                        "route",
                        "extract financial KPIs from a PDF report",
                        "--workspace",
                        str(workspace),
                        "--trace-id",
                        "card-helper",
                        "--agent-mode",
                        "prepare",
                    ]
                )
            prepared = json.loads(prepare_output.getvalue())
            query_wiki_root = prepared["query_wiki_root"]

            card_output = io.StringIO()
            with contextlib.redirect_stdout(card_output):
                cli_main(["query-wiki", "card", query_wiki_root, "skill:pdf-table-parser"])

            text = card_output.getvalue()
            self.assertIn("# skill:pdf-table-parser", text)
            self.assertIn("## Card", text)
            self.assertNotIn("## Source", text)

    def test_agent_mode_finalize_accepts_skill_package_from_stdin(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))
            prepare_output = io.StringIO()
            with contextlib.redirect_stdout(prepare_output):
                cli_main(
                    [
                        "route",
                        "extract financial KPIs from a PDF report",
                        "--workspace",
                        str(workspace),
                        "--trace-id",
                        "agent-stdin",
                        "--agent-mode",
                        "prepare",
                    ]
                )

            prepared = json.loads(prepare_output.getvalue())
            trace_dir = Path(prepared["trace_dir"])
            query_wiki_root = Path(prepared["query_wiki_root"])
            manifest = json.loads((query_wiki_root / "manifest.json").read_text(encoding="utf-8"))
            selected = next(item for item in manifest["skills"] if item["selectable"])
            skill_package_json = json.dumps(
                {
                    "selected_skills": [
                        {
                            "skill_id": selected["skill_id"],
                            "scope": selected["scope"],
                            "role": "Use this skill for the main task capability.",
                            "evidence": [{"path": selected["page_path"], "reason": "Selected from query wiki."}],
                        }
                    ],
                    "required_edges": [],
                    "ordered_hints": [],
                    "near_misses": [],
                    "coverage_notes": [],
                    "rationale": "Single evidence-backed skill selection.",
                },
                ensure_ascii=False,
            )

            finalize_output = io.StringIO()
            with patch("sys.stdin", io.StringIO(skill_package_json)):
                with contextlib.redirect_stdout(finalize_output):
                    cli_main(
                        [
                            "route",
                            "extract financial KPIs from a PDF report",
                            "--workspace",
                            str(workspace),
                            "--trace-id",
                            "agent-stdin",
                            "--agent-mode",
                            "finalize",
                            "--skill-package-file",
                            "-",
                        ]
                    )

            route = json.loads(finalize_output.getvalue())
            self.assertEqual(route["trace_id"], "agent-stdin")
            self.assertEqual(route["selected_skills"][0]["skill_id"], selected["skill_id"])
            self.assertEqual(json.loads((trace_dir / "agent_skill_package.json").read_text(encoding="utf-8")), json.loads(skill_package_json))
            self.assertTrue((trace_dir / "route.json").exists())

    def test_agent_mode_finalize_rejects_skill_package_file_outside_trace_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))
            prepare_output = io.StringIO()
            with contextlib.redirect_stdout(prepare_output):
                cli_main(
                    [
                        "route",
                        "extract financial KPIs from a PDF report",
                        "--workspace",
                        str(workspace),
                        "--trace-id",
                        "agent-outside",
                        "--agent-mode",
                        "prepare",
                    ]
                )
            prepared = json.loads(prepare_output.getvalue())
            query_wiki_root = Path(prepared["query_wiki_root"])
            manifest = json.loads((query_wiki_root / "manifest.json").read_text(encoding="utf-8"))
            selected = next(item for item in manifest["skills"] if item["selectable"])
            outside_file = Path(tmp) / "agent_skill_package.json"
            outside_file.write_text(
                json.dumps(
                    {
                        "selected_skills": [
                            {
                                "skill_id": selected["skill_id"],
                                "scope": selected["scope"],
                                "role": "Use this skill for the main task capability.",
                                "evidence": [{"path": selected["page_path"], "reason": "Selected from query wiki."}],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit) as raised:
                cli_main(
                    [
                        "route",
                        "extract financial KPIs from a PDF report",
                        "--workspace",
                        str(workspace),
                        "--trace-id",
                        "agent-outside",
                        "--agent-mode",
                        "finalize",
                        "--skill-package-file",
                        str(outside_file),
                    ]
                )

            self.assertIn("must be inside trace directory", str(raised.exception))

    def test_agent_mode_finalize_rejects_invalid_skill_package(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))
            cli_main(
                [
                    "route",
                    "extract financial KPIs from a PDF report",
                    "--workspace",
                    str(workspace),
                    "--trace-id",
                    "agent-invalid",
                    "--agent-mode",
                    "prepare",
                ]
            )
            package_file = workspace / "runs" / "agent-invalid" / "bad_skill_package.json"
            package_file.write_text(
                json.dumps(
                    {
                        "selected_skills": [
                            {
                                "skill_id": "skill:not-in-manifest",
                                "scope": "core",
                                "role": "Invalid.",
                                "evidence": [{"path": "../outside.md", "reason": "bad"}],
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit) as raised:
                with contextlib.redirect_stderr(io.StringIO()):
                    cli_main(
                        [
                            "route",
                            "extract financial KPIs from a PDF report",
                            "--workspace",
                            str(workspace),
                            "--trace-id",
                            "agent-invalid",
                            "--agent-mode",
                            "finalize",
                            "--skill-package-file",
                            str(package_file),
                        ]
                    )

            self.assertNotEqual(raised.exception.code, 0)

    def test_plan_agent_mode_prepare_and_finalize_round_trip(self) -> None:
        from skillfabric import SkillFabric

        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_fixture_workspace(workspace)
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))
            client = SkillFabric(workspace=workspace)
            route = client.route(
                "extract financial KPIs from a PDF report",
                use_llm_router=False,
                explorer_backend="fallback",
            )
            route_path = route.trace_dir / "route.json"

            prepare_output = io.StringIO()
            with contextlib.redirect_stdout(prepare_output):
                cli_main(
                    [
                        "plan",
                        "--route-file",
                        str(route_path),
                        "--workspace",
                        str(workspace),
                        "--agent-mode",
                        "prepare",
                    ]
                )

            prepared = json.loads(prepare_output.getvalue())
            package_root = Path(prepared["root"])
            self.assertTrue((package_root / "planner_request.json").exists())
            self.assertTrue((package_root / "PLANNER.md").exists())
            self.assertFalse((package_root / "execution_prompt.md").exists())
            self.assertNotIn("draft_agent_run_spec", prepared)
            planner_output = {
                "execution_prompt": "# Fixture Planner Prompt\n\nExecute the task with selected skills.",
            }

            finalize_output = io.StringIO()
            with patch("sys.stdin", io.StringIO("```json\n" + json.dumps(planner_output) + "\n```")):
                with contextlib.redirect_stdout(finalize_output):
                    cli_main(
                        [
                            "plan",
                            "--workspace",
                            str(workspace),
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
            self.assertEqual(
                (package_root / "execution_prompt.md").read_text(encoding="utf-8"),
                "# Fixture Planner Prompt\n\nExecute the task with selected skills.\n",
            )
            self.assertFalse((package_root / "workflow_plan.json").exists())
            self.assertFalse((package_root / "agent_run_spec.json").exists())
            self.assertFalse((package_root / "agent_run_spec_draft.json").exists())
            self.assertFalse((package_root / "handoff_prompt.md").exists())


if __name__ == "__main__":
    unittest.main()
